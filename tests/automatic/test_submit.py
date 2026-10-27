"""The H200 launcher snapshots source and preserves explicit and automatic resume."""

import json
import os
import runpy
import signal
import subprocess
import tarfile
import time
from pathlib import Path

import pytest
import torch


@pytest.fixture
def submission(tmp_path):
    script = Path(__file__).resolve().parents[2] / "submit_train.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    for name in ("nnInteractive/source.py", "client/source.py", "scripts/automatic/launch.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source snapshot\n")
    python = tmp_path / "python"
    python.write_text(
        '#!/bin/bash\nprintf "%s\\0" "$@" > "$NNI_TEST_CAPTURE"\n'
        'if [[ "${NNI_TEST_WAIT:-0}" == 1 ]]; then\n'
        "trap 'echo checkpoint > \"$NNI_TEST_CAPTURE.signal\"; exit 0' USR1\n"
        "while true; do sleep 0.1; done\nfi\n"
    )
    python.chmod(0o755)
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(python, arcname="bin/python")
    config, output = tmp_path / "config with spaces.json", tmp_path / "output with spaces"
    config.write_text("{}")
    output.mkdir()
    (output / "last.pt").touch()
    capture = tmp_path / "arguments"
    env = dict(
        os.environ,
        BASH_ENV="/dev/null",
        NNI_RUNTIME_ARCHIVE=str(archive),
        NNI_SOURCE_DIR=str(tmp_path),
        NNI_RUN_DIR=str(tmp_path / "named runs"),
        NNI_TRAIN_MODULE="nnInteractive.automatic.prototype.run_config",
        NNI_TEST_CAPTURE=str(capture),
        SLURM_TMPDIR=str(tmp_path / "node-local"),
        SLURM_JOB_ID="123",
    )
    return script, env, config, output, capture


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("extra", [[], ["--resume", "saved run/last.pt", "--threads", "2"]])
def test_submit_forwards_training_options(submission, named, extra):
    script, env, config, output, capture = submission
    if named:
        root = Path(env["NNI_RUN_DIR"])
        root.mkdir()
        config.replace(root / "coarse.json")
        output.rename(root / "coarse")
        config, output = root / "coarse.json", root / "coarse"
    args = ["coarse"] if named else [str(config), str(output)]
    subprocess.run(["bash", str(script), *args, *extra], env=env, check=True)
    assert capture.read_bytes().decode().split("\0")[:-1] == [
        "scripts/automatic/launch.py",
        "nnInteractive.automatic.train",
        "--config",
        str(config),
        "--output",
        str(output),
        "--resume",
        str(output / "last.pt"),
        *extra,
    ]
    local = Path(env["SLURM_TMPDIR"]) / "nni-123"
    assert (local / "cache/huggingface").is_dir()
    assert (local / "source/nnInteractive/source.py").is_file()
    assert (local / "source/client/source.py").is_file()


def test_submit_forwards_checkpoint_signal(submission):
    script, env, config, output, capture = submission
    process = subprocess.Popen(["bash", str(script), str(config), str(output)], env={**env, "NNI_TEST_WAIT": "1"})
    try:
        deadline = time.monotonic() + 10
        while not capture.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert capture.exists()
        duplicate = subprocess.run(["bash", str(script), str(config), str(output)], env=env, timeout=10, check=False)
        assert duplicate.returncode != 0
        process.send_signal(signal.SIGUSR1)
        assert process.wait(timeout=10) == 0
        assert Path(str(capture) + ".signal").read_text().strip() == "checkpoint"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize("extra", [[], ["--threads", "2"]])
def test_runtime_settings(tmp_path, monkeypatch, extra):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"performance": {"cpu_threads": 4, "cudnn_benchmark": True}}))
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/automatic/launch.py"))
    monkeypatch.setattr("sys.argv", ["launch.py", "nnInteractive.automatic.train", "--config", str(config), *extra])
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    called = []
    monkeypatch.setattr(runpy, "run_module", lambda module, run_name: called.append((module, run_name)))
    namespace["main"]()
    import sys

    assert sys.argv[-2:] == (extra or ["--threads", "4"])
    assert torch.backends.cudnn.benchmark
    assert called == [("nnInteractive.automatic.train", "__main__")]
