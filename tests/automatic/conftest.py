"""Synthetic fixtures must not inherit an allocated job's real-data override."""

import pytest


@pytest.fixture(autouse=True)
def isolate_fixture_arrays(monkeypatch):
    monkeypatch.delenv("PROMPT_ICL_ARRAY_DIR", raising=False)
    monkeypatch.delenv("NNI_COARSE_CACHE", raising=False)
