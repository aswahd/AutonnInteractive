"""Differentiable single-patch refinement through fixed original nnInteractive weights."""

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint

from .distillation import FrozenTeacher, TeacherConfig
from .losses import binary_logits, segmentation_loss
from .spatial import resize_volume, teacher_inputs


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = False
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    max_weight: float = 0.2
    ramp_updates: int = 2000

    def __post_init__(self):
        TeacherConfig(**self.__dict__)

    def weight(self, completed_updates):
        return TeacherConfig.weight(self, completed_updates)


REFINEMENT_STAT_FIELDS = ("episodes", "positive", "initial_dice", "refined_dice", "improved", "worsened")


class FrozenRefiner(FrozenTeacher):
    """Reuse strict frozen loading only; this forward intentionally retains input gradients."""

    def predict_logits(self, query, soft_prior):
        prompts = query.new_zeros(query.shape[0], self.source["prompt_channels"], *query.shape[2:])
        channel = self.source["channel_mapping"]["prev_seg"]
        prompts[:, channel : channel + 1] = soft_prior
        self.network.eval()
        with torch.autocast(query.device.type, dtype=torch.bfloat16, enabled=query.is_cuda):
            # Recompute frozen activations during backward to fit alongside the 192^3 student.
            return checkpoint(self.network, torch.cat((query, prompts), 1), use_reentrant=False)

    def loss(self, student_logits, batch, seed=None):
        query, teacher_logits, _ = teacher_inputs(student_logits, batch, getattr(self, "input_size", None))
        prior = binary_logits(teacher_logits.float()).sigmoid()
        refined = self.predict_logits(query, prior)
        loss, _ = segmentation_loss(binary_logits(refined), batch.query_targets, batch.query_valid)
        with torch.no_grad():
            truth = (batch.query_targets > 0.5) & batch.query_valid
            positive = truth.flatten(1).any(1)
            scores = []
            for logits in (student_logits, refined):
                probability = resize_volume(binary_logits(logits.float()), batch.query_targets.shape[-3:]).sigmoid()
                prediction = (probability > 0.5) & batch.query_valid
                denominator = prediction.flatten(1).sum(1) + truth.flatten(1).sum(1)
                scores.append(2 * (prediction & truth).flatten(1).sum(1) / denominator.clamp_min(1))
            initial, final = scores
            stats = torch.stack(
                (
                    positive.new_tensor(len(positive), dtype=torch.float64),
                    positive.sum(),
                    initial[positive].sum(),
                    final[positive].sum(),
                    ((final > initial) & positive).sum(),
                    ((final < initial) & positive).sum(),
                )
            )
        return loss, stats


def summarize_refinement(totals):
    episodes, positive, initial, refined, improved, worsened = totals.detach().cpu().tolist()
    return {
        "train/refinement_stat_episodes": int(episodes),
        "train/refinement_positive_episodes": int(positive),
        "train/student_initial_dice": initial / positive if positive else None,
        "train/refined_dice": refined / positive if positive else None,
        "train/refinement_dice_improvement": (refined - initial) / positive if positive else None,
        "train/refinement_improvement_rate": improved / positive if positive else None,
        "train/refinement_worsening_rate": worsened / positive if positive else None,
    }


def evaluation_refiner(model, source):
    """Identical detached resized-context inference for all three large-FOV training arms."""
    if not model.config.resized_context:
        return None
    config = TeacherConfig(
        enabled=True,
        checkpoint=source["checkpoint_file"],
        checkpoint_sha256=source["checkpoint_sha256"],
        soft_prior=True,
    )
    return FrozenTeacher(config, source, next(model.parameters()).device, model.config)
