"""Tests for src.export (SRT and Markdown rendering)."""

from src.export import (
    _seconds_to_hms,
    _seconds_to_srt_time,
    to_markdown,
    to_srt,
    write_markdown,
    write_srt,
)


# ---------------------------------------------------------------------------
# Time formatting helpers
# ---------------------------------------------------------------------------


def test_seconds_to_srt_time_basic():
    assert _seconds_to_srt_time(0.0) == "00:00:00,000"


def test_seconds_to_srt_time_with_ms():
    assert _seconds_to_srt_time(3661.5) == "01:01:01,500"


def test_seconds_to_srt_time_rounding():
    # 1.9995 rounds to 000ms for the fractional part (floor * 1000)
    result = _seconds_to_srt_time(1.001)
    assert result == "00:00:01,001"


def test_seconds_to_hms():
    assert _seconds_to_hms(3661.9) == "01:01:01"
    assert _seconds_to_hms(0.0) == "00:00:00"


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": "  Hello world  "},
    {"start": 3.0, "end": 5.0, "text": "How are you?"},
]


def test_to_srt_index_starts_at_one():
    srt = to_srt(SEGMENTS)
    assert srt.startswith("1\n")


def test_to_srt_contains_both_segments():
    srt = to_srt(SEGMENTS)
    assert "Hello world" in srt
    assert "How are you?" in srt
    assert "2\n" in srt


def test_to_srt_timestamp_format():
    srt = to_srt(SEGMENTS)
    assert "00:00:00,000 --> 00:00:02,500" in srt


def test_to_srt_strips_whitespace():
    srt = to_srt(SEGMENTS)
    # Text should be stripped
    assert "  Hello world  " not in srt
    assert "Hello world" in srt


def test_to_srt_empty_segments():
    assert to_srt([]) == "\n"


def test_write_srt(tmp_path):
    path = tmp_path / "out.srt"
    write_srt(SEGMENTS, path)
    content = path.read_text(encoding="utf-8")
    assert "Hello world" in content
    assert "How are you?" in content


# ---------------------------------------------------------------------------
# Markdown dialog
# ---------------------------------------------------------------------------

DIARIZED_SEGMENTS = [
    {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "SPEAKER_00"},
    {"start": 2.5, "end": 4.0, "text": "Hi there", "speaker": "SPEAKER_01"},
    {"start": 4.5, "end": 6.0, "text": "How are you?", "speaker": "SPEAKER_00"},
]


def test_to_markdown_contains_title():
    md = to_markdown(SEGMENTS, title="My Video")
    assert "# My Video" in md


def test_to_markdown_default_speaker():
    md = to_markdown(SEGMENTS)
    assert "Speaker" in md


def test_to_markdown_named_speakers():
    md = to_markdown(DIARIZED_SEGMENTS)
    assert "SPEAKER_00" in md
    assert "SPEAKER_01" in md


def test_to_markdown_speaker_header_format():
    md = to_markdown(DIARIZED_SEGMENTS)
    assert "## [00:00:00] SPEAKER_00" in md


def test_to_markdown_groups_consecutive_same_speaker():
    # Two consecutive SPEAKER_00 segments should only produce one heading
    segs = [
        {"start": 0.0, "end": 1.0, "text": "One", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 2.0, "text": "Two", "speaker": "SPEAKER_00"},
    ]
    md = to_markdown(segs)
    assert md.count("## [") == 1


def test_to_markdown_skips_empty_text():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "", "speaker": "A"},
        {"start": 1.0, "end": 2.0, "text": "Hello", "speaker": "A"},
    ]
    md = to_markdown(segs)
    assert "Hello" in md


def test_write_markdown(tmp_path):
    path = tmp_path / "out.md"
    write_markdown(SEGMENTS, path, title="Test")
    content = path.read_text(encoding="utf-8")
    assert "# Test" in content
    assert "Hello world" in content


def test_to_markdown_greek_text():
    segs = [{"start": 0.0, "end": 2.0, "text": "Γεια σου κόσμε"}]
    md = to_markdown(segs)
    assert "Γεια σου κόσμε" in md
