import json

import pytest
import torch

from nnInteractive.automatic.prototype.checkpoint import build_model, load_model
from nnInteractive.automatic.prototype.data import Episodes, region_mask
from nnInteractive.automatic.prototype.engine import loss_function, score
from nnInteractive.automatic.prototype.model import InContextUNet3D
from nnInteractive.automatic.prototype.session import InContextInferenceSession


def test_hierarchical_targets_and_unknown_voxels():
    segmentation = torch.tensor([-1, 0, 17, 29, 42, 20])
    assert region_mask(segmentation, [17, 29, 42]).tolist() == [0, 0, 1, 1, 1, 0]
    logits = torch.tensor([[[[[10.0, -10.0]]]]])
    target = torch.zeros_like(logits)
    valid = torch.tensor([[[[[False, True]]]]])
    assert score(logits, target, valid)["dice"] == 1
    assert torch.isfinite(loss_function(logits, target, valid))


@pytest.mark.parametrize("mode", ["prototype", "memory"])
def test_context_invariance_sensitivity_and_gradients(mode):
    torch.manual_seed(7)
    model = InContextUNet3D(width=4, mode=mode, memory_grid=2)
    query = torch.randn(1, 1, 16, 16, 16)
    images = torch.randn(1, 2, 1, 16, 16, 16)
    masks = (images > 0).float()
    model.eval()
    with torch.no_grad():
        prediction = model(query, images, masks)
        permuted = model(query, images.flip(1), masks.flip(1))
        changed = model(query, images, 1 - masks)
    assert prediction.shape == query.shape
    torch.testing.assert_close(prediction, permuted, atol=1e-5, rtol=1e-5)
    assert (prediction - changed).abs().max() > 1e-4
    model.train()
    images.requires_grad_()
    output = model(query, images, masks)
    loss = loss_function(output, (query > 0).float(), torch.ones_like(query, dtype=torch.bool))
    loss.backward()
    assert images.grad.abs().sum() > 0
    assert model.fusions[0].weight.grad.abs().sum() > 0
    if mode == "memory":
        assert model.mask_embedding.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_combined_attention_contains_prototypes_and_mask_memory():
    model = InContextUNet3D(width=4, mode="memory", memory_grid=2, prototype_tokens=True).eval()
    query = torch.randn(1, 1, 16, 16, 16)
    images = torch.randn(1, 2, 1, 16, 16, 16, requires_grad=True)
    masks = (images.detach() > 0).float()
    context = model.encode_context(images, masks)
    assert context["tokens"].shape == (1, 2 + 2 * 2**3, 32)
    context["tokens"].retain_grad()
    output = model.predict_with_context(query, None, context)
    permuted = model(query, images.flip(1), masks.flip(1))
    torch.testing.assert_close(output, permuted, atol=1e-5, rtol=1e-5)
    output.square().mean().backward()
    assert context["tokens"].grad[:, :2].abs().sum() > 0
    assert context["tokens"].grad[:, 2:].abs().sum() > 0
    assert torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0


def make_cache(tmp_path):
    cases = ["train_a", "train_b", "train_c", "val_a"]
    image = torch.randn(1, 16, 16, 16)
    segmentation = torch.ones_like(image, dtype=torch.int16)
    segmentation[:, 8:] = 2
    torch.save({case: {"image": image, "seg": segmentation} for case in cases}, tmp_path / "shard.pt")
    manifest = {
        "regions": {"one": [1], "union": [1, 2]},
        "shape": [16, 16, 16],
        "splits": {"train": cases[:3], "val": cases[3:]},
        "cases": {case: {"shard": "shard.pt", "patient": case, "present": ["one", "union"]} for case in cases},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_disjoint_fixed_episodes_and_union(tmp_path):
    make_cache(tmp_path)
    validation = Episodes(tmp_path, "val", ["one", "union"], 2)
    assert len(validation) == 2
    first, second = validation[0], validation[0]
    assert first["query_id"] not in first["support_ids"]
    assert set(first["support_ids"]) <= {"train_a", "train_b", "train_c"}
    assert len(set(first["support_ids"])) == 2
    assert first["support_ids"] == second["support_ids"]
    assert validation[1]["target"].all()
    training = Episodes(tmp_path, "train", ["one"], 2, episodes=8)
    for index in range(len(training)):
        item = training[index]
        assert item["query_id"] not in item["support_ids"]


def test_accumulated_updates_match_full_batch_and_resume(tmp_path, monkeypatch):
    from nnInteractive.automatic.prototype.__main__ import main

    cache = tmp_path / "cache"
    cache.mkdir()
    make_cache(cache)

    def run(name, updates, micro_batch, resume=False):
        output = tmp_path / name
        argv = [
            "coarse",
            "train",
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--regions",
            "one",
            "--mode",
            "prototype",
            "--width",
            "4",
            "--shots",
            "1",
            "--device",
            "cpu",
            "--threads",
            "1",
            "--batch-size",
            "2",
            "--seed",
            "2026",
            "--steps",
            str(updates),
            "--validate-every",
            "3",
            "--micro-batch-size",
            str(micro_batch),
        ]
        if resume:
            argv.extend(["--resume", str(output / "last.pt")])
        monkeypatch.setattr("sys.argv", argv)
        main()
        return torch.load(output / "last.pt", weights_only=False)

    # Compare gradients before Adam amplifies roundoff near zero; resume itself
    # must reproduce a continuous run with the same microbatch size exactly.
    physical = run("physical", 1, 2)
    accumulated = run("one_update", 1, 1)
    for key, state in physical["optimizer"]["state"].items():
        torch.testing.assert_close(
            accumulated["optimizer"]["state"][key]["exp_avg"], state["exp_avg"], atol=1e-6, rtol=1e-4
        )
    full = run("full", 4, 1)
    run("accumulated", 3, 1)
    resumed = run("accumulated", 4, 1, resume=True)
    assert resumed["step"] == full["step"] == 4
    for name, value in full["model"].items():
        torch.testing.assert_close(resumed["model"][name], value, atol=0, rtol=0)
    assert {int(state["step"]) for state in resumed["optimizer"]["state"].values()} <= {1, 2, 3, 4}


def test_checkpoint_and_label_free_session(tmp_path):
    config = {"width": 4, "mode": "memory", "memory_grid": 2, "backbone": None}
    model = build_model(config).eval()
    path = tmp_path / "model.pt"
    torch.save(
        {"format_version": 1, "model_config": config, "model": model.state_dict(), "input_shape": [16, 16, 16]}, path
    )
    restored, _ = load_model(path)
    for original, recovered in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(original, recovered)
    session = InContextInferenceSession(str(path), "cpu")
    image = torch.randn(1, 16, 16, 16)
    with pytest.raises(RuntimeError):
        session.predict(image)
    with pytest.raises(ValueError):
        session.set_context([image], [torch.zeros_like(image)])
    session.set_context([image], [(image > 0).float()])
    probability = session.predict(image)
    assert probability.shape == image.shape[1:]
    assert ((probability >= 0) & (probability <= 1)).all()


def test_sparse_context_does_not_read_hidden_support_labels():
    model = InContextUNet3D(width=4, memory_grid=2).eval()
    images = torch.randn(1, 2, 1, 16, 16, 16)
    prompts = torch.zeros(1, 2, 7, 16, 16, 16)
    prompts[:, :, 3, 8, 8, 8] = 1
    hidden = torch.randint(0, 2, (1, 2, 1, 16, 16, 16)).float()
    sparse = torch.zeros(1, 2, dtype=torch.bool)
    with torch.no_grad():
        first, _ = model.unify_context(images, prompts, hidden, sparse)
        second, _ = model.unify_context(images, prompts, 1 - hidden, sparse)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["point", "scribble", "lasso", "bbox", "initial_mask"])
def test_prompt_channels_and_local_training(kind):
    from nnInteractive.automatic.prototype.prompts import (
        add_error_correction,
        simulate_prompt,
    )

    target = torch.zeros(1, 16, 16, 16)
    target[:, 4:12, 4:12, 4:12] = 1
    prompts = simulate_prompt(target, kind, seed=23)
    assert prompts.shape == (7, 16, 16, 16)
    assert prompts.any()
    model = InContextUNet3D(width=4, memory_grid=2)
    image = torch.randn(1, 1, 16, 16, 16)
    logits = model(image, query_prompts=prompts[None])
    corrected = add_error_correction(prompts[None], logits.sigmoid().bfloat16(), target[None])
    assert corrected[:, 3:5].any()
    loss = loss_function(logits, target[None], torch.ones_like(target[None], dtype=torch.bool))
    loss.backward()
    assert model.backbone.encoder.stem.convs[0].conv.weight.grad[:, 1:].abs().sum() > 0


def test_cached_memory_and_original_no_context_network():
    from nnInteractive.automatic.prototype.interactive_session import (
        ContextNetworkAdapter,
    )

    model = InContextUNet3D(width=4, memory_grid=2).eval()
    image = torch.randn(1, 1, 16, 16, 16)
    prompts = torch.zeros(1, 7, 16, 16, 16)
    inputs = torch.cat([image, prompts], 1)
    adapter = ContextNetworkAdapter(model).eval()
    with torch.no_grad():
        torch.testing.assert_close(adapter(inputs), model.backbone(inputs), rtol=0, atol=0)
        masks = (image[:, None] > 0).float()
        adapter.context = model.encode_context(image[:, None], masks)
        output = adapter(inputs)
        direct = model(image, image[:, None], masks)
        torch.testing.assert_close(output[:, 1:2] - output[:, :1], direct)


@pytest.mark.parametrize("storage", ["tensor", "blosc2"])
def test_original_interaction_interfaces_and_undo(tmp_path, storage):
    import numpy as np

    from nnInteractive.automatic.prototype.interactive_session import (
        TemporalInferenceSession,
    )

    config = {"width": 4, "mode": "memory", "memory_grid": 2, "backbone": None}
    model = build_model(config).eval()
    path = tmp_path / "interactive.pt"
    torch.save(
        {"format_version": 1, "model_config": config, "model": model.state_dict(), "input_shape": [64, 64, 64]}, path
    )
    session = TemporalInferenceSession(
        device=torch.device("cpu"), torch_n_threads=2, interactions_storage=storage, do_autozoom=False
    )
    session.initialize_from_icl_checkpoint(str(path))
    image = np.random.default_rng(3).normal(size=(1, 16, 16, 16)).astype(np.float32)
    target = np.zeros((16, 16, 16), dtype=np.uint8)
    session.set_image(image)
    session.set_target_buffer(target)
    lasso = np.zeros_like(target)
    lasso[8, 4:12, 4:12] = 1
    session.add_lasso_interaction(lasso, True, run_prediction=False)
    assert session.undo()
    session.add_scribble_interaction(lasso, True, run_prediction=False)
    session.add_bbox_interaction([[8, 9], [4, 12], [4, 12]], True, run_prediction=False)
    session.add_point_interaction((8, 8, 8), True, run_prediction=True)
    assert np.isin(target, [0, 1]).all()
    dense = np.zeros_like(target)
    dense[4:12, 4:12, 4:12] = 1
    session.add_initial_seg_interaction(dense)
    session.add_current_as_context()
    session.set_image(image)
    session.set_target_buffer(np.zeros_like(target))
    session.predict_from_context(refine=False)
    assert session.context_adapter.context is not None
    session.add_lasso_interaction(lasso, True, run_prediction=True)
    assert session.undo()
    session.clear_context()
    assert session.context_adapter.context is None
    session.executor.shutdown()


def test_balanced_loss_does_not_reward_all_background_for_tiny_targets():
    target = torch.zeros(1, 1, 16, 16, 16)
    target[..., 8, 8, 8] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    background = torch.full_like(target, -6)
    uncertain = torch.zeros_like(target)
    assert loss_function(background, target, valid, "class") > loss_function(uncertain, target, valid, "class")
    assert loss_function(background, target, valid, "voxel") < loss_function(uncertain, target, valid, "voxel")
    assert torch.isfinite(loss_function(background, torch.zeros_like(target), valid, "class"))


def test_pretrained_loader_preserves_weights_and_runtime_metadata(tmp_path):
    from nnInteractive.automatic.prototype.checkpoint import (
        compact_configuration,
        initialize_backbone,
    )
    from nnInteractive.automatic.prototype.interactive_session import (
        TemporalInferenceSession,
    )

    source = tmp_path / "source"
    (source / "fold_0").mkdir(parents=True)
    configuration = compact_configuration(4, [64, 64, 64])
    (source / "plans.json").write_text(
        json.dumps({"label_manager": "LabelManager", "configurations": {"test": configuration}})
    )
    (source / "dataset.json").write_text(
        json.dumps({"channel_names": {"0": "CT"}, "labels": {"background": 0, "target": 1}})
    )
    (source / "inference_info.json").write_text(
        json.dumps(
            {"interaction_channels": 6, "point_radius": 3, "interaction_decay": 0.91, "preferred_scribble_thickness": 3}
        )
    )
    (source / "LICENSE").write_text("CC0-1.0\n")
    original = InContextUNet3D(width=4, memory_grid=2)
    torch.save(
        {
            "network_weights": original.backbone.state_dict(),
            "trainer_name": "nnInteractiveTrainer_stub",
            "init_args": {"configuration": "test"},
        },
        source / "fold_0" / "checkpoint_final.pth",
    )
    config = {"width": 4, "mode": "memory", "memory_grid": 2, "backbone": None}
    initialized = initialize_backbone(config, str(source), "0", "checkpoint_final.pth")
    for first, second in zip(original.backbone.parameters(), initialized.backbone.parameters()):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    checkpoint = tmp_path / "icl.pt"
    torch.save(
        {"format_version": 1, "model_config": config, "model": initialized.state_dict(), "input_shape": [64, 64, 64]},
        checkpoint,
    )
    session = TemporalInferenceSession(device=torch.device("cpu"), torch_n_threads=2)
    session.initialize_from_icl_checkpoint(str(checkpoint))
    assert session.point_interaction.point_radius == 3
    assert session.point_interaction.use_distance_transform
    assert session.interaction_decay == 0.91
    assert session.preferred_scribble_thickness == [3, 3, 3]
    assert session.num_interaction_channels == 7
    assert session.configuration_manager.patch_size == [64, 64, 64]
    session.executor.shutdown()
    from nnInteractive.automatic.prototype.checkpoint import export_inference

    exported = tmp_path / "frozen.pt"
    export_inference(checkpoint, exported, str(source))
    frozen = TemporalInferenceSession(device=torch.device("cpu"), torch_n_threads=2)
    frozen.initialize_from_icl_checkpoint(str(exported))
    for expected, actual in zip(original.backbone.parameters(), frozen.context_adapter.native_refiner.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        assert not actual.requires_grad
    frozen.executor.shutdown()


def test_retrieval_excludes_query_and_repeated_patient(tmp_path):
    from nnInteractive.automatic.prototype.retrieval import SupportRetriever, file_sha256

    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    cache = tmp_path / "features.pt"
    torch.save(
        {
            "manifest_sha256": file_sha256(manifest),
            "checkpoint_sha256": "synthetic",
            "case_ids": ["q", "same_patient", "a", "b", "c"],
            "features": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.99, 0.1], [0.95, 0.2], [0.0, 1.0]]),
        },
        cache,
    )
    retriever = SupportRetriever(cache, manifest)
    patients = {"q": "p0", "same_patient": "p0", "a": "p1", "b": "p1", "c": "p2"}
    selected = retriever.select("q", list(patients), patients, 2)
    assert selected == ["a", "c"]
    manifest.write_text('{"changed": true}')
    with pytest.raises(ValueError):
        SupportRetriever(cache, manifest)


def test_frozen_native_refinement_retains_original_logits_with_context(tmp_path):
    from nnInteractive.automatic.prototype.checkpoint import export_inference
    from nnInteractive.automatic.prototype.interactive_session import (
        ContextNetworkAdapter,
    )

    config = {"width": 4, "mode": "memory", "memory_grid": 2, "backbone": None}
    trained = build_model(config).eval()
    reference = build_model(config).backbone.eval().requires_grad_(False)
    adapter = ContextNetworkAdapter(trained, reference).eval()
    image = torch.randn(1, 1, 16, 16, 16)
    inputs = torch.cat([image, torch.randn(1, 7, 16, 16, 16)], 1)
    with torch.no_grad():
        expected = reference(inputs)
        torch.testing.assert_close(adapter(inputs), expected, atol=0, rtol=0)
        adapter.context = trained.encode_context(image[:, None], (image[:, None] > 0).float())
        torch.testing.assert_close(adapter(inputs), expected, atol=0, rtol=0)
        # The independently learned coarse context remains available for mask initialization.
        assert trained.predict_with_context(image, None, adapter.context).shape == image.shape
    source, exported = tmp_path / "training.pt", tmp_path / "inference.pt"
    torch.save(
        {
            "format_version": 1,
            "model_config": config,
            "model": trained.state_dict(),
            "optimizer": {"large_state": torch.ones(100)},
            "input_shape": [64, 64, 64],
        },
        source,
    )
    export_inference(source, exported)
    _, restored = load_model(exported)
    assert "optimizer" not in restored
    assert len(restored["inference_export"]["training_checkpoint_sha256"]) == 64
    with pytest.raises(FileExistsError):
        export_inference(source, exported)


def test_prompt_simulation_and_correction_ignore_unknown_voxels():
    from nnInteractive.automatic.prototype.prompts import (
        add_error_correction,
        simulate_prompt,
    )

    target = torch.zeros(1, 16, 16, 16)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, :8] = False
    prompt = simulate_prompt(target, "point", 2, valid)
    assert not prompt[:, :8].any()
    probability = (~valid).float()[None]
    revised = add_error_correction(torch.zeros_like(prompt)[None], probability, target[None], valid[None])
    assert not revised[:, 1:].any()  # Errors only in unknown voxels do not generate corrective prompts.


def test_training_checkpoint_and_resume_continue_episode_sequence(tmp_path, monkeypatch):
    from argparse import Namespace

    from nnInteractive.automatic.prototype.engine import train

    def unavailable_git(*args, **kwargs):
        raise FileNotFoundError("Git is not installed")

    monkeypatch.setattr("nnInteractive.automatic.prototype.engine.subprocess.check_output", unavailable_git)
    cache = tmp_path / "cache"
    cache.mkdir()
    make_cache(cache)
    args = Namespace(
        cache=str(cache),
        output=str(tmp_path / "first"),
        regions=["one", "union"],
        mode="memory",
        width=4,
        memory_grid=2,
        interactive_probability=0.5,
        bce_balance="class",
        selection="joint",
        shots=1,
        steps=2,
        batch_size=1,
        learning_rate=0.001,
        backbone_learning_rate=0.001,
        workers=0,
        validate_every=1,
        log_every=1,
        seed=2026,
        device="cpu",
        threads=2,
        resume=None,
        backbone_model_dir=None,
        backbone_fold="0",
        backbone_checkpoint="checkpoint_final.pth",
    )
    train(args)
    first = torch.load(tmp_path / "first/last.pt", weights_only=True)
    assert first["step"] == 2
    assert first["provenance"]["manifest_sha256"]
    assert first["provenance"]["git_commit"] is None
    assert first["provenance"]["source_sha256"]
    args.output, args.resume, args.steps = str(tmp_path / "continued"), str(tmp_path / "first/last.pt"), 3
    train(args)
    continued = torch.load(tmp_path / "continued/last.pt", weights_only=True)
    assert continued["step"] == 3
    assert (tmp_path / "continued/best.pt").is_file()
    assert continued["optimizer"]["state"][0]["step"].item() == 3
