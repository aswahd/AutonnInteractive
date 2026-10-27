"""Leakage and final-selection boundaries, using metadata fixtures only."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from nnInteractive.automatic.backbone import sha256
from nnInteractive.automatic.config import ModelConfig
from nnInteractive.automatic.transfer import (
    bind_transfer_protocol,
    final_evaluation_policy,
)


@pytest.fixture
def protocol(tmp_path):
    targets = [f"target_{i}" for i in range(43)]
    cohort = {
        "training_targets": targets[:20],
        "held_out_targets": targets[20:],
        "target_regions": {target: [i + 1] for i, target in enumerate(targets)},
        "training_volumes": ["train"],
        "seen_target_development_query_volumes": ["dev"],
        "inference_support_volumes": ["support"],
        "final_query_volumes": ["test"],
    }
    training = {
        "regions": cohort["target_regions"],
        "splits": {"train": ["train"], "val": ["dev"]},
        "source_sha256": {"source": "same"},
    }
    final = {**training, "splits": {"train": ["support"], "val": ["test"]}, "evaluation_only": True}
    spec = {"targets": targets[:20], "full_volume_query_ids": ["dev"], "full_volume_prompt_type": "dense_mask"}
    for key, data in (("cohort_policy_file", cohort), ("final_evaluation_manifest", final)):
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(data))
        spec[key] = str(path)
    return spec, training, cohort


@pytest.mark.parametrize(
    "failure", ["pool_overlap", "label_overlap", "heldout_selection", "wrong_manifest", "inference_training"]
)
def test_training_rejects_leakage(protocol, failure):
    spec, training, cohort = protocol
    if failure == "pool_overlap":
        cohort["final_query_volumes"] = ["train"]
    elif failure == "label_overlap":
        cohort["target_regions"]["target_20"] = [1]
    elif failure == "heldout_selection":
        spec["targets"][0] = "target_20"
    elif failure == "wrong_manifest":
        training["splits"]["train"] = ["support"]
    else:
        training["evaluation_only"] = True
    from pathlib import Path

    Path(spec["cohort_policy_file"]).write_text(json.dumps(cohort))
    with pytest.raises(ValueError):
        bind_transfer_protocol(spec, training)


def selected_state(protocol, tmp_path):
    spec, training, _ = protocol
    selected = {
        "manifests": {
            **spec,
            **bind_transfer_protocol(spec, training),
            "validation_seed": 102026,
            "array_directory": str(tmp_path),
            "prompt_swap_pairs": [],
        },
        "train_config": {"updates": 50000},
        "source": {},
        "best_full_volume_macro_dice": 0.6,
        "completed_updates": 45000,
        "pending_validation": None,
    }
    torch.save({**selected, "completed_updates": 50000}, tmp_path / "last.pt")
    torch.save(selected, tmp_path / "best.pt")
    return selected


@pytest.mark.parametrize("failure", ["unfinished", "pending", "different_run", "changed_policy"])
def test_final_test_refuses_early_or_changed_protocol(protocol, tmp_path, failure):
    selected = selected_state(protocol, tmp_path)
    latest = copy.deepcopy(selected)
    latest["completed_updates"] = 49999 if failure == "unfinished" else 50000
    if failure == "pending":
        latest["pending_validation"] = {"update": 50000}
    if failure == "different_run":
        latest["source"] = {"different": True}
    if failure == "changed_policy":
        from pathlib import Path

        with Path(protocol[0]["cohort_policy_file"]).open("a") as handle:
            handle.write("\n")
    torch.save(latest, tmp_path / "last.pt")
    with pytest.raises(ValueError):
        final_evaluation_policy(tmp_path, selected)


@pytest.mark.parametrize("job_id", [None, "test"])
def test_final_cli_reports_groups_without_changing_checkpoints(protocol, tmp_path, monkeypatch, job_id):
    from nnInteractive.automatic import validate_cli

    selected = selected_state(protocol, tmp_path)
    before = {name: sha256(tmp_path / name) for name in ("last.pt", "best.pt")}
    if job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.argv", ["validate_cli", "--output", str(tmp_path), "--final-test"])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    cpu = torch.device("cpu")
    monkeypatch.setattr(torch, "device", lambda *_: cpu)
    model = SimpleNamespace(config=ModelConfig(patch_size=(8, 8, 8), context_patch_size=(8, 8, 8)), eval=lambda: None)
    monkeypatch.setattr(validate_cli, "load_checkpoint", lambda *_: (model, selected))
    monkeypatch.setattr(validate_cli, "EpisodeSampler", lambda *a, **kw: SimpleNamespace(targets=kw["targets"]))

    evaluated = []

    def evaluate(model, sampler, queries, seed, rank, world, progress, prompt_type, refiner=None):
        assert queries == ["test"] and prompt_type == "dense_mask"
        for index in progress.indices(len(sampler.targets)):
            evaluated.append(index)
            progress.record(
                index,
                [
                    {
                        "index": index,
                        "target": sampler.targets[index],
                        "prompt_type": prompt_type,
                        "dice": 0.8 if index < 20 else 0.2,
                    }
                ],
            )

    monkeypatch.setattr(validate_cli, "evaluate_volumes", evaluate)
    validate_cli.main()
    summary = json.loads((tmp_path / "final_test/update_45000/full_volume/summary.json").read_text())
    assert summary["seen"]["positive_macro_dice"] == pytest.approx(0.8)
    assert summary["held_out"]["positive_macro_dice"] == pytest.approx(0.2)
    assert before == {name: sha256(tmp_path / name) for name in before}
    # Repeat with the same journal: committed tasks are not re-evaluated.
    validate_cli.main()
    assert len(evaluated) == 43
    assert before == {name: sha256(tmp_path / name) for name in before}
    directory = tmp_path / "final_test/update_45000"
    reports = list(directory.glob("allocation_*.json"))
    assert len(reports) == (2 if job_id is None else 1)
    assert all(json.loads(path.read_text())["job_id"] == job_id for path in reports)
