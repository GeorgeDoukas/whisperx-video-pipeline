"""Tests for src.checkpoint."""

import json
import tempfile
from pathlib import Path

import pytest

from src.checkpoint import Checkpoint


@pytest.fixture
def tmp_checkpoint(tmp_path):
    return Checkpoint(tmp_path / "test.checkpoint.json")


def test_initial_state_empty(tmp_checkpoint):
    assert tmp_checkpoint.get("anything") is None
    assert not tmp_checkpoint.is_done("audio")


def test_set_and_get(tmp_checkpoint):
    tmp_checkpoint.set("foo", "bar")
    assert tmp_checkpoint.get("foo") == "bar"


def test_persists_to_disk(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint(path)
    cp.set("key", 42)

    # reload from disk
    cp2 = Checkpoint(path)
    assert cp2.get("key") == 42


def test_mark_done_and_is_done(tmp_checkpoint):
    assert not tmp_checkpoint.is_done("transcribe")
    tmp_checkpoint.mark_done("transcribe", data={"segments": []})
    assert tmp_checkpoint.is_done("transcribe")


def test_get_stage_data(tmp_checkpoint):
    payload = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    tmp_checkpoint.mark_done("align", data=payload)
    assert tmp_checkpoint.get_stage_data("align") == payload


def test_get_stage_data_none_when_not_set(tmp_checkpoint):
    assert tmp_checkpoint.get_stage_data("missing") is None


def test_reload_preserves_done_flags(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint(path)
    cp.mark_done("audio", data="/tmp/audio.wav")

    cp2 = Checkpoint(path)
    assert cp2.is_done("audio")
    assert cp2.get_stage_data("audio") == "/tmp/audio.wav"


def test_creates_parent_directories(tmp_path):
    nested = tmp_path / "a" / "b" / "cp.json"
    cp = Checkpoint(nested)
    cp.set("x", 1)
    assert nested.exists()


def test_unicode_data_roundtrip(tmp_path):
    path = tmp_path / "cp.json"
    cp = Checkpoint(path)
    greek = "Γεια σου κόσμε"
    cp.set("text", greek)

    cp2 = Checkpoint(path)
    assert cp2.get("text") == greek
