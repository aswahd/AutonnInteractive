"""Public support-input validation, checkpoint dispatch and annotation-free inference."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nnInteractive.automatic import AutomaticSegmentationSession, load_model
from nnInteractive.automatic.prototype.checkpoint import build_model


@pytest.mark.parametrize("command", ["train", "validate", "compare"])
def test_cli_help_without_scheduler(command, monkeypatch):
    from nnInteractive.automatic import __main__

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["nninteractive-automatic", command, "--help"])
    with pytest.raises(SystemExit) as result:
        __main__.main()
    assert result.value.code == 0


@pytest.mark.parametrize("options", [["--max-training-seconds", "60"], ["--stop-after", "1000"]])
def test_training_rejects_cli_budget_overrides(options, monkeypatch):
    from nnInteractive.automatic import __main__

    monkeypatch.setattr("sys.argv", ["automatic", "train", "--config", "unused", "--output", "unused", *options])
    with pytest.raises(SystemExit) as result:
        __main__.main()
    assert result.value.code == 2


def test_cycle_command_is_removed(monkeypatch):
    from nnInteractive.automatic import __main__

    monkeypatch.setattr("sys.argv", ["automatic", "cycle"])
    with pytest.raises(SystemExit) as result:
        __main__.main()
    assert result.value.code == 2


def test_cli_passes_subcommand_help_and_options_to_the_training_parser(monkeypatch):
    from nnInteractive.automatic import __main__, train

    captured = []
    monkeypatch.setattr("sys.argv", ["nninteractive-automatic", "train", "--config", "example.json", "--help"])
    monkeypatch.setattr(train, "main", lambda: captured.append(__main__.sys.argv[1:]))
    __main__.main()
    assert captured == [["--config", "example.json", "--help"]]


def test_unified_loader_preserves_prototype_format(tmp_path):
    config = dict(backbone=None, width=2, mode="prototype", memory_grid=2)
    model = build_model(config).eval()
    path = tmp_path / "prototype.pt"
    torch.save(dict(format_version=1, model_config=config, model=model.state_dict()), path)
    restored, state = load_model(path)
    assert state["format_version"] == 1
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, restored.state_dict()[name], atol=0, rtol=0)


def test_public_session_validates_foreground_and_does_not_require_query_annotations(monkeypatch):
    class Student(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(patch_size=(4,) * 3, prompt_channels=7, context_size=(8,) * 3)

        def predict_query(self, query, supports, prompts):
            return torch.cat((query * 0, query * 0 + 1), 1)

    mapping = dict(prev_seg=0, bbox2d=[1, 2], lasso=[1, 2], points=[3, 4], scribble=[5, 6])
    monkeypatch.setattr(
        "nnInteractive.automatic.session.load_model",
        lambda *a: (Student(), dict(format="prompt_encoded_icl3d_v2", source=dict(channel_mapping=mapping))),
    )
    session = AutomaticSegmentationSession("unused", "cpu")
    query = np.zeros((1, 9, 7, 10), np.float32)
    with pytest.raises(RuntimeError, match="set_context"):
        session.predict(query)
    images, prompts = torch.zeros(2, 1, 4, 4, 4), torch.zeros(2, 7, 4, 4, 4)
    with pytest.raises(ValueError, match="foreground"):
        session.set_context(images, prompts)
    prompts[0, 3, 1, 1, 1] = 1
    session.set_context(images, prompts)
    prompts.zero_()  # Subsequent caller mutation must not erase the stored context.
    assert session.context[1].any()
    result = session.predict(query)
    assert result.mask.shape == query.shape[1:] and result.coverage_min > 0
    np.testing.assert_allclose(result.probabilities, torch.sigmoid(torch.tensor(1.0)).item(), rtol=1e-6)
    with pytest.raises(ValueError, match="refinement must"):
        session.predict(query, refinement="unknown")
