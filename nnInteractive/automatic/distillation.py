"""Detached single-patch nnInteractive distillation."""

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .backbone import load_original, sha256
from .losses import binary_logits
from .optimization import episode_rng
from .spatial import resize_volume, teacher_inputs


def state_digest(network):
    digest = hashlib.sha256()
    for name, tensor in network.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TeacherConfig:
    enabled: bool = False
    checkpoint: str = ""
    checkpoint_sha256: str = ""
    call_probability: float = 0.25
    initial_threshold: float = 0.5
    skip_empty_initial_mask: bool = True
    soft_prior: bool = False
    max_weight: float = 0.2
    ramp_updates: int = 2000

    def __post_init__(self):
        if not 0 <= self.call_probability <= 1 or not 0 < self.initial_threshold < 1:
            raise ValueError("invalid teacher sampling probability or prior threshold")
        if (
            not math.isfinite(self.max_weight)
            or self.max_weight < 0
            or type(self.ramp_updates) is not int
            or self.ramp_updates < 0
        ):
            raise ValueError(
                "distillation weight and integer ramp must be finite and nonnegative"
            )
        if self.enabled and (not self.checkpoint or len(self.checkpoint_sha256) != 64):
            raise ValueError(
                "enabled teacher requires an explicit checkpoint and SHA-256"
            )

    def weight(self, completed_updates):
        return (
            self.max_weight * min(1.0, completed_updates / self.ramp_updates)
            if self.ramp_updates
            else self.max_weight
        )


def selected_episodes(indices, seed, probability, device):
    """A separate per-episode RNG stream leaves data and support dropout unchanged."""
    return torch.tensor(
        [
            episode_rng(seed + 2000000, index).random() < probability
            for index in indices
        ],
        device=device,
        dtype=torch.bool,
    )


def distillation_loss(student_logits, teacher_probability, valid):
    """Soft-target BCE averaged over valid voxels per invoked episode; no quality gate."""
    z = binary_logits(student_logits.float())
    if z.shape != teacher_probability.shape or z.shape != valid.shape:
        raise ValueError("KD requires aligned B1DHW probabilities and validity")
    if valid.dtype != torch.bool or not valid.flatten(1).any(1).all():
        raise ValueError(
            "KD requires boolean validity with supervised voxels in every episode"
        )
    t = teacher_probability.detach().float()
    terms = F.binary_cross_entropy_with_logits(
        torch.where(valid, z, 0.0), torch.where(valid, t, 0.0), reduction="none"
    )
    return (terms * valid).flatten(1).sum(1) / valid.flatten(1).sum(1)


STAT_FIELDS = (
    "episodes",
    "selected",
    "empty",
    "invoked",
    "positive",
    "student_dice",
    "teacher_dice",
    "improved_positive",
    "valid_voxels",
    "confident_voxels",
    "negative",
    "student_negative_probability",
    "teacher_negative_probability",
)


class FrozenTeacher:
    """Separate frozen network, excluded from the student and its checkpoint/optimizer."""

    def __init__(self, config, student_source, device, model_config=None):
        self.config = config
        self.input_size = (
            None if model_config is None else tuple(model_config.teacher_input_size)
        )
        path = Path(config.checkpoint)
        if (
            config.checkpoint_sha256 != student_source["checkpoint_sha256"]
            or sha256(path) != config.checkpoint_sha256
        ):
            raise ValueError(
                "teacher must match the student's original pretrained checkpoint hash"
            )
        if not path.parent.name.startswith("fold_"):
            raise ValueError(
                "teacher checkpoint must be inside its original fold directory"
            )
        threads = torch.get_num_threads()
        try:
            # Loading constructs CPU modules. Preserve the student's/dropout's RNG stream.
            with torch.random.fork_rng(devices=[]):
                network, self.source = load_original(
                    path.parent.parent, path.parent.name[5:], path.name
                )
        finally:
            torch.set_num_threads(threads)
        for key in ("channel_mapping", "prompt_channels", "native_patch_size"):
            if self.source[key] != student_source[key]:
                raise ValueError(
                    f"teacher/student native input contract differs: {key}"
                )
        self.network = network.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def predict_logits(self, query, initial_mask):
        """Only CT and a detached prior enter the original network; no AutoZoom engine."""
        prompts = query.new_zeros(
            query.shape[0], self.source["prompt_channels"], *query.shape[2:]
        )
        prompts[
            :,
            self.source["channel_mapping"]["prev_seg"] : self.source["channel_mapping"][
                "prev_seg"
            ]
            + 1,
        ] = initial_mask.detach()
        self.network.eval()
        with torch.autocast(
            query.device.type,
            dtype=torch.bfloat16,
            enabled=query.is_cuda and torch.cuda.is_bf16_supported(),
        ):
            logits = self.network(torch.cat((query.detach(), prompts), dim=1))
        if (
            logits.shape != (query.shape[0], 2, *query.shape[2:])
            or not torch.isfinite(logits).all()
        ):
            raise FloatingPointError("invalid teacher output")
        return logits

    @torch.no_grad()
    def predict(self, query, initial_mask):
        return binary_logits(self.predict_logits(query, initial_mask).float()).sigmoid()

    def loss(self, student_logits, batch, seed):
        query, student_logits, teacher_valid = teacher_inputs(
            student_logits, batch, getattr(self, "input_size", None)
        )
        batch_size = student_logits.shape[0]
        selected = selected_episodes(
            [m["global_episode_index"] for m in batch.metadata],
            seed,
            self.config.call_probability,
            student_logits.device,
        )
        student_probability = binary_logits(student_logits.detach().float()).sigmoid()
        initial = (student_probability > self.config.initial_threshold).float()
        empty = ~initial.flatten(1).any(1)
        invoked = selected & ~empty if self.config.skip_empty_initial_mask else selected
        stats = torch.zeros(
            len(STAT_FIELDS), dtype=torch.float64, device=student_logits.device
        )
        stats[0], stats[1], stats[2], stats[3] = (
            batch_size,
            selected.sum(),
            (selected & empty).sum(),
            invoked.sum(),
        )
        loss = student_logits[:, :, 0, 0, 0].float().sum() * 0
        if invoked.any():
            prior = student_probability if self.config.soft_prior else initial
            probability = self.predict(query[invoked], prior[invoked])
            valid = teacher_valid[invoked]
            per_episode = distillation_loss(student_logits[invoked], probability, valid)
            loss = per_episode.sum() / batch_size
            # GT is used only below for diagnostics; it cannot gate KD or enter predict().
            truth = (
                resize_volume(
                    batch.query_targets[invoked], valid.shape[-3:], "nearest-exact"
                )
                > 0.5
            ) & valid
            positive = truth.flatten(1).any(1)
            dice, foreground = [], []
            for p in (student_probability[invoked], probability):
                prediction = (p > 0.5) & valid
                denominator = prediction.flatten(1).sum(1) + truth.flatten(1).sum(1)
                dice.append(
                    2
                    * (prediction & truth).flatten(1).sum(1)
                    / denominator.clamp_min(1)
                )
                foreground.append(
                    (p * valid).flatten(1).sum(1) / valid.flatten(1).sum(1)
                )
            sd, td = dice
            stats[4:] = torch.stack(
                (
                    positive.sum(),
                    sd[positive].sum(),
                    td[positive].sum(),
                    ((td > sd) & positive).sum(),
                    valid.sum(),
                    (valid & ((probability > 0.9) | (probability < 0.1))).sum(),
                    (~positive).sum(),
                    foreground[0][~positive].sum(),
                    foreground[1][~positive].sum(),
                )
            ).detach()
        return loss, stats


def summarize_teacher(totals):
    """Every invoked prediction is used. GT quality metrics never change supervision."""
    s = dict(zip(STAT_FIELDS, totals.detach().cpu().tolist()))

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    return {
        "train/teacher_stat_episodes": int(s["episodes"]),
        "train/teacher_selected_episodes": int(s["selected"]),
        "train/teacher_invoked_episodes": int(s["invoked"]),
        "train/teacher_accepted_episodes": int(s["invoked"]),
        "train/teacher_selected_fraction": ratio(s["selected"], s["episodes"]),
        "train/teacher_call_fraction": ratio(s["invoked"], s["episodes"]),
        "train/teacher_acceptance_fraction": 1.0 if s["invoked"] else None,
        "train/teacher_accepted_batch_fraction": ratio(s["invoked"], s["episodes"]),
        "train/teacher_empty_prior_fraction": ratio(s["empty"], s["selected"]),
        "train/teacher_empty_prior_batch_fraction": ratio(s["empty"], s["episodes"]),
        "train/student_initial_dice": ratio(s["student_dice"], s["positive"]),
        "train/teacher_refined_dice": ratio(s["teacher_dice"], s["positive"]),
        "train/teacher_minus_student_dice": ratio(
            s["teacher_dice"] - s["student_dice"], s["positive"]
        ),
        "train/teacher_improved_positive_fraction": ratio(
            s["improved_positive"], s["positive"]
        ),
        "train/teacher_positive_invoked_episodes": int(s["positive"]),
        "train/teacher_confident_voxel_fraction": ratio(
            s["confident_voxels"], s["valid_voxels"]
        ),
        "train/student_negative_foreground_probability": ratio(
            s["student_negative_probability"], s["negative"]
        ),
        "train/teacher_negative_foreground_probability": ratio(
            s["teacher_negative_probability"], s["negative"]
        ),
    }
