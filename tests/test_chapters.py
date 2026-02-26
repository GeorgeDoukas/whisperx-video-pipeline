"""Tests for src.chapters (chapter generation and FFMETADATA serialisation)."""

from src.chapters import generate_chapters, to_ffmetadata


SEGMENTS = [
    {"start": 0.0, "end": 30.0, "text": "Introduction to the topic"},
    {"start": 30.0, "end": 120.0, "text": "First main section"},
    {"start": 120.0, "end": 360.0, "text": "Second main section"},
    {"start": 360.0, "end": 600.0, "text": "Third main section"},
    {"start": 600.0, "end": 720.0, "text": "Conclusion"},
]


# ---------------------------------------------------------------------------
# generate_chapters
# ---------------------------------------------------------------------------


def test_empty_segments_returns_empty():
    assert generate_chapters([]) == []


def test_single_short_segment_one_chapter():
    segs = [{"start": 0.0, "end": 10.0, "text": "Hello"}]
    chapters = generate_chapters(segs, interval=300)
    assert len(chapters) == 1
    assert chapters[0]["start"] == 0.0
    assert chapters[0]["end"] == 10.0


def test_chapter_count_matches_interval():
    # 720 seconds / 300 interval → 3 chapters  (0-300, 300-600, 600-720)
    chapters = generate_chapters(SEGMENTS, interval=300)
    assert len(chapters) == 3


def test_chapter_boundaries():
    chapters = generate_chapters(SEGMENTS, interval=300)
    assert chapters[0]["start"] == 0.0
    assert chapters[0]["end"] == 300.0
    assert chapters[1]["start"] == 300.0
    assert chapters[1]["end"] == 600.0
    assert chapters[2]["start"] == 600.0
    assert chapters[2]["end"] == 720.0


def test_chapter_titles_include_number():
    chapters = generate_chapters(SEGMENTS, interval=300)
    for i, ch in enumerate(chapters, start=1):
        assert f"Chapter {i}" in ch["title"]


def test_chapter_title_includes_excerpt():
    segs = [{"start": 0.0, "end": 10.0, "text": "My opening line"}]
    chapters = generate_chapters(segs, interval=300)
    assert "My opening line" in chapters[0]["title"]


def test_chapter_title_excerpt_capped_at_60_chars():
    long_text = "A" * 100
    segs = [{"start": 0.0, "end": 10.0, "text": long_text}]
    chapters = generate_chapters(segs, interval=300)
    # title is "Chapter 1: " + excerpt; excerpt <= 60 chars
    excerpt = chapters[0]["title"].split(": ", 1)[1]
    assert len(excerpt) <= 60


def test_last_chapter_end_matches_last_segment_end():
    chapters = generate_chapters(SEGMENTS, interval=300)
    assert chapters[-1]["end"] == SEGMENTS[-1]["end"]


# ---------------------------------------------------------------------------
# to_ffmetadata
# ---------------------------------------------------------------------------


def test_ffmetadata_header():
    meta = to_ffmetadata([])
    assert meta.startswith(";FFMETADATA1")


def test_ffmetadata_chapter_block():
    chapters = [{"start": 0.0, "end": 60.0, "title": "Intro"}]
    meta = to_ffmetadata(chapters)
    assert "[CHAPTER]" in meta
    assert "TIMEBASE=1/1000" in meta
    assert "START=0" in meta
    assert "END=60000" in meta
    assert "title=Intro" in meta


def test_ffmetadata_multiple_chapters():
    chapters = generate_chapters(SEGMENTS, interval=300)
    meta = to_ffmetadata(chapters)
    assert meta.count("[CHAPTER]") == len(chapters)


def test_ffmetadata_millisecond_conversion():
    chapters = [{"start": 1.5, "end": 3.75, "title": "Test"}]
    meta = to_ffmetadata(chapters)
    assert "START=1500" in meta
    assert "END=3750" in meta
