"""Training Dice ignores empty targets, invalid voxels, and exact-mask controls."""

from types import SimpleNamespace

import torch

from nnInteractive.automatic.losses import training_dice_totals
from nnInteractive.automatic.train import format_progress


def test_training_dice_totals_and_empty_groups():
    targets = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]]).reshape(2, 1, 1, 1, 4).float()
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[..., -1] = False
    support_targets = targets[:1, None].expand(2, 2, 1, 1, 1, 4).clone()
    query = torch.cat((torch.zeros_like(targets), targets.new_tensor([1, -1, 1, 1]).expand_as(targets)), 1)
    supports = torch.cat((torch.zeros_like(support_targets), support_targets * 2 - 1), 2)
    supports[0, 0, 1] = -1
    batch = SimpleNamespace(
        query_targets=targets,
        query_valid=valid,
        support_targets=support_targets,
        support_valid=valid[:, None].expand_as(support_targets),
        support_eligible=torch.tensor([[True, False], [False, True]]),
    )
    outputs = {"query_logits": query.requires_grad_(), "support_logits": supports.requires_grad_()}
    totals = training_dice_totals(outputs, batch)
    torch.testing.assert_close(totals, torch.tensor([0.5, 1, 1, 2]))
    assert not totals.requires_grad
    targets.zero_()
    batch.support_eligible.zero_()
    assert not training_dice_totals(outputs, batch).any()


def test_progress_is_concise_and_handles_missing_dice():
    row = dict(update=640, loss=0.43727, query_dice=0.81, support_dice=0.75, rank0_step_seconds=6.1705)
    assert format_progress(row, 50000) == "step 640/50000 | loss 0.4373 | query Dice 0.810 | support Dice 0.750 | 6.17s"
    row.update(query_dice=None, support_dice=None)
    assert "query Dice n/a | support Dice n/a" in format_progress(row, 50000)
