"""Tests for chunked transcription helpers in src.transcribe."""

import math
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
import numpy as np

from src.transcribe import WHISPER_SAMPLE_RATE, transcribe_chunked, transcribe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_audio(duration_seconds: float):
    """Return a numpy array of zeros representing silence."""
    return np.zeros(int(duration_seconds * WHISPER_SAMPLE_RATE), dtype="float32")


def _make_fake_model(segments_per_call=None):
    """Return a mock WhisperX model whose transcribe returns predictable output."""
    model = MagicMock()
    call_count = [0]

    def fake_transcribe(audio, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        segs = segments_per_call[idx] if segments_per_call else []
        return {"segments": segs, "language": "el"}

    model.transcribe.side_effect = fake_transcribe
    return model


@contextmanager
def _mock_whisperx(audio):
    """Inject a mock whisperx module with load_audio returning *audio*."""
    mock_wx = MagicMock()
    mock_wx.load_audio.return_value = audio
    with patch.dict(sys.modules, {"whisperx": mock_wx}):
        yield mock_wx


# ---------------------------------------------------------------------------
# transcribe (single-shot)
# ---------------------------------------------------------------------------


def test_transcribe_passes_initial_prompt(tmp_path):
    fake_audio = _make_fake_audio(5)
    model = MagicMock()
    model.transcribe.return_value = {"segments": [], "language": "el"}

    with _mock_whisperx(fake_audio):
        transcribe(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            initial_prompt="Keep English terms in English.",
        )

    call_kwargs = model.transcribe.call_args[1]
    assert call_kwargs["initial_prompt"] == "Keep English terms in English."


def test_transcribe_no_initial_prompt(tmp_path):
    fake_audio = _make_fake_audio(5)
    model = MagicMock()
    model.transcribe.return_value = {"segments": [], "language": "el"}

    with _mock_whisperx(fake_audio):
        transcribe(model, tmp_path / "audio.wav", language="el", batch_size=8, chunk_size=30)

    call_kwargs = model.transcribe.call_args[1]
    assert "initial_prompt" not in call_kwargs


# ---------------------------------------------------------------------------
# transcribe_chunked – basic behaviour
# ---------------------------------------------------------------------------


def test_transcribe_chunked_single_segment(tmp_path):
    """Audio shorter than one segment → one model.transcribe call."""
    from src.checkpoint import Checkpoint

    checkpoint = Checkpoint(tmp_path / "cp.json")
    fake_audio = _make_fake_audio(60)  # 1 minute = 60 seconds
    segs = [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}]
    model = _make_fake_model(segments_per_call=[segs])

    with _mock_whisperx(fake_audio):
        result = transcribe_chunked(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            checkpoint=checkpoint,
            segment_minutes=10,
        )

    assert model.transcribe.call_count == 1
    assert len(result["segments"]) == 1


def test_transcribe_chunked_multiple_segments(tmp_path):
    """Audio spanning multiple segments → multiple model.transcribe calls."""
    from src.checkpoint import Checkpoint

    checkpoint = Checkpoint(tmp_path / "cp.json")
    # 25 minutes of audio → 3 segments of 10 min each
    fake_audio = _make_fake_audio(25 * 60)
    segs_per_call = [
        [{"start": 0.0, "end": 1.0, "text": "seg0", "words": []}],
        [{"start": 0.0, "end": 1.0, "text": "seg1", "words": []}],
        [{"start": 0.0, "end": 1.0, "text": "seg2", "words": []}],
    ]
    model = _make_fake_model(segments_per_call=segs_per_call)

    with _mock_whisperx(fake_audio):
        result = transcribe_chunked(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            checkpoint=checkpoint,
            segment_minutes=10,
        )

    assert model.transcribe.call_count == 3
    assert len(result["segments"]) == 3


def test_transcribe_chunked_timestamps_offset(tmp_path):
    """Timestamps in later segments are shifted by the segment offset."""
    from src.checkpoint import Checkpoint

    checkpoint = Checkpoint(tmp_path / "cp.json")
    # 15 minutes → 2 segments of 10 min
    fake_audio = _make_fake_audio(15 * 60)
    segs_per_call = [
        [{"start": 0.0, "end": 5.0, "text": "first", "words": []}],
        [{"start": 0.0, "end": 3.0, "text": "second", "words": []}],
    ]
    model = _make_fake_model(segments_per_call=segs_per_call)

    with _mock_whisperx(fake_audio):
        result = transcribe_chunked(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            checkpoint=checkpoint,
            segment_minutes=10,
        )

    segs = result["segments"]
    assert segs[0]["start"] == pytest.approx(0.0)
    # Second segment starts at raw 0.0 + offset of 10*60 = 600 s
    assert segs[1]["start"] == pytest.approx(600.0)
    assert segs[1]["end"] == pytest.approx(603.0)


def test_transcribe_chunked_word_timestamps_offset(tmp_path):
    """Word-level timestamps are also shifted by the segment offset."""
    from src.checkpoint import Checkpoint

    checkpoint = Checkpoint(tmp_path / "cp.json")
    fake_audio = _make_fake_audio(15 * 60)
    segs_per_call = [
        [{"start": 0.0, "end": 1.0, "text": "a", "words": [{"start": 0.0, "end": 0.5, "word": "a"}]}],
        [{"start": 1.0, "end": 2.0, "text": "b", "words": [{"start": 1.0, "end": 1.5, "word": "b"}]}],
    ]
    model = _make_fake_model(segments_per_call=segs_per_call)

    with _mock_whisperx(fake_audio):
        result = transcribe_chunked(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            checkpoint=checkpoint,
            segment_minutes=10,
        )

    segs = result["segments"]
    # second segment word timestamps should be shifted by 600 s
    assert segs[1]["words"][0]["start"] == pytest.approx(601.0)
    assert segs[1]["words"][0]["end"] == pytest.approx(601.5)


# ---------------------------------------------------------------------------
# Resume behaviour
# ---------------------------------------------------------------------------


def test_transcribe_chunked_resumes_from_checkpoint(tmp_path):
    """Segments already in the checkpoint are not re-transcribed."""
    from src.checkpoint import Checkpoint

    checkpoint = Checkpoint(tmp_path / "cp.json")
    # Pre-populate chunk 0 in the checkpoint
    done_segs = [{"start": 5.0, "end": 6.0, "text": "from checkpoint", "words": []}]
    checkpoint.mark_done("transcribe_chunk_0", done_segs)

    fake_audio = _make_fake_audio(15 * 60)
    # Only chunk 1 needs to be transcribed
    segs_per_call = [
        [{"start": 0.0, "end": 1.0, "text": "live", "words": []}],
    ]
    model = _make_fake_model(segments_per_call=segs_per_call)

    with _mock_whisperx(fake_audio):
        result = transcribe_chunked(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            checkpoint=checkpoint,
            segment_minutes=10,
        )

    # model.transcribe called only once (for chunk 1, not chunk 0)
    assert model.transcribe.call_count == 1
    # Checkpoint segment comes first
    assert result["segments"][0]["text"] == "from checkpoint"
    assert result["segments"][1]["text"] == "live"


def test_transcribe_chunked_saves_each_chunk_to_checkpoint(tmp_path):
    """Each transcribed chunk is persisted in the checkpoint immediately."""
    from src.checkpoint import Checkpoint

    checkpoint = Checkpoint(tmp_path / "cp.json")
    fake_audio = _make_fake_audio(25 * 60)
    segs_per_call = [
        [{"start": 0.0, "end": 1.0, "text": "c0", "words": []}],
        [{"start": 0.0, "end": 1.0, "text": "c1", "words": []}],
        [{"start": 0.0, "end": 1.0, "text": "c2", "words": []}],
    ]
    model = _make_fake_model(segments_per_call=segs_per_call)

    with _mock_whisperx(fake_audio):
        transcribe_chunked(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            checkpoint=checkpoint,
            segment_minutes=10,
        )

    for i in range(3):
        assert checkpoint.is_done(f"transcribe_chunk_{i}")


def test_transcribe_chunked_initial_prompt_passed(tmp_path):
    """initial_prompt is forwarded to every model.transcribe call."""
    from src.checkpoint import Checkpoint

    checkpoint = Checkpoint(tmp_path / "cp.json")
    fake_audio = _make_fake_audio(60)
    model = MagicMock()
    model.transcribe.return_value = {"segments": [], "language": "el"}

    with _mock_whisperx(fake_audio):
        transcribe_chunked(
            model,
            tmp_path / "audio.wav",
            language="el",
            batch_size=8,
            chunk_size=30,
            checkpoint=checkpoint,
            segment_minutes=10,
            initial_prompt="Keep English.",
        )

    call_kwargs = model.transcribe.call_args[1]
    assert call_kwargs["initial_prompt"] == "Keep English."

