"""Native refinement executes, preserves the input grid, and exposes both results."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_infer import IdentityLogitModel

from nnInteractive.automatic.autozoom import NativeAutoZoomRefiner
from nnInteractive.automatic.infer import predict_volume, refine_volume
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def test_selectable_modes_preserve_student_probabilities_and_training_state():
    class BackgroundRefiner:
        def refine(self, image, initial):
            assert initial.any() and image.shape == (1, 12, 8, 8)
            return np.zeros_like(initial), 3

    query = np.ones((1, 12, 8, 8), dtype=np.float32)
    supports, prompts = torch.ones(1, 1, 8, 8, 8), torch.ones(1, 7, 8, 8, 8)
    model = IdentityLogitModel()
    output = predict_volume(
        model, query, supports, prompts, stride=(4,) * 3, mode="autozoom", refiner=BackgroundRefiner()
    )
    assert output.mode == "autozoom" and output.probabilities is None and not output.mask.any()
    assert output.student_prediction.mask.all() and output.student_prediction.probabilities is not None
    assert output.native_forward_passes == 3 and output.coverage_min > 0 and model.training
    assert output.windows == model.calls  # No second student forward for the paired result.
    with pytest.raises(ValueError, match="original sliding-window"):
        refine_volume(output, query, BackgroundRefiner())
    with pytest.raises(ValueError, match="requires"):
        predict_volume(model, query, supports, prompts, mode="autozoom")
    with pytest.raises(ValueError, match="mode"):
        predict_volume(model, query, supports, prompts, mode="oracle")


def native_test_refiner():
    class BackgroundNetwork(torch.nn.Module):
        def forward(self, inputs):
            assert inputs.shape == (1, 8, 64, 64, 64)
            assert not inputs[:, 2:].any()  # Only previous segmentation, never synthetic correction prompts.
            return torch.cat((torch.ones_like(inputs[:, :1]), torch.zeros_like(inputs[:, :1])), 1)

    session = nnInteractiveInferenceSession(
        device=torch.device("cpu"),
        torch_n_threads=2,
        do_autozoom=True,
        interactions_storage="tensor",
        enable_undo=False,
    )
    session._apply_capability(session._legacy_default_capability())
    session.pad_mode_data = "constant"
    session.configuration_manager = SimpleNamespace(patch_size=(64, 64, 64))
    session.network = BackgroundNetwork().eval()
    refiner = object.__new__(NativeAutoZoomRefiner)
    refiner.session = session
    return refiner


def test_actual_native_autozoom_no_renormalization_no_input_mutation_or_state_leak():
    refiner = native_test_refiner()
    query = np.full((1, 80, 76, 70), 123.5, dtype=np.float32)
    prior = np.zeros(query.shape[1:], dtype=np.uint8)
    prior[10:70, 8:68, 5:65] = 1  # Extent + native context margin exceeds a 64^3 patch.
    before = prior.copy()
    factors = []
    build = refiner.session._build_network_input

    def inspect(center, factor):
        factors.append(factor)
        np.testing.assert_array_equal(refiner.session.preprocessed_image.numpy(), query)
        return build(center, factor)

    try:
        with (
            patch.object(refiner.session, "_build_network_input", side_effect=inspect),
            patch.object(refiner.session, "set_image", side_effect=AssertionError("must not normalize again")),
        ):
            result, calls = refiner.refine(query, prior)
        assert result.shape == prior.shape and result.dtype == np.uint8 and not result.any()
        assert calls > 1 and max(factors) > 1  # Actual coarse + local refinement network passes.
        np.testing.assert_array_equal(prior, before)
        assert np.all(query == 123.5)
        assert refiner.session.preprocessed_image is None and refiner.session.target_buffer is None
        assert not refiner.session.new_interaction_centers
        empty, calls = refiner.refine(query, np.zeros_like(prior))
        assert not empty.any() and calls == 0
        again, _ = refiner.refine(query, prior)
        np.testing.assert_array_equal(result, again)
    finally:
        refiner.close()


def test_native_refiner_rejects_malformed_masks_and_changed_checkpoint():
    with patch("nnInteractive.automatic.autozoom.sha256", return_value="changed"):
        with pytest.raises(ValueError, match="hash changed"):
            NativeAutoZoomRefiner({"checkpoint_file": "unused", "checkpoint_sha256": "expected"}, "cpu")
    refiner = native_test_refiner()
    try:
        query = np.zeros((1, 64, 64, 64), dtype=np.float32)
        for prior in (np.ones((4, 4, 4)), np.full((64, 64, 64), 0.5)):
            with pytest.raises(ValueError, match="binary and match"):
                refiner.refine(query, prior)
    finally:
        refiner.close()
