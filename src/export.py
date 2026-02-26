"""Export pipeline results to SRT subtitles and Markdown dialog."""

from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Time-formatting helpers
# ---------------------------------------------------------------------------


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert a float number of seconds to an SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seconds_to_hms(seconds: float) -> str:
    """Convert a float number of seconds to HH:MM:SS (no milliseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# SRT
# ---------------------------------------------------------------------------


def to_srt(segments: List[Dict[str, Any]]) -> str:
    """Render *segments* as an SRT subtitle string."""
    blocks: List[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _seconds_to_srt_time(seg["start"])
        end = _seconds_to_srt_time(seg["end"])
        text = seg.get("text", "").strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}")
    return "\n\n".join(blocks) + "\n"


def write_srt(segments: List[Dict[str, Any]], path: Path) -> None:
    """Write an SRT file to *path*."""
    path.write_text(to_srt(segments), encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown dialog
# ---------------------------------------------------------------------------


def to_markdown(segments: List[Dict[str, Any]], title: str = "Transcript") -> str:
    """Render *segments* as a Markdown dialog, grouping consecutive turns
    by the same speaker.

    The output is suitable for later LLM summarisation.
    """
    lines: List[str] = [f"# {title}\n"]
    current_speaker: str = ""

    for seg in segments:
        speaker: str = seg.get("speaker") or "Speaker"
        ts = _seconds_to_hms(seg["start"])
        text = seg.get("text", "").strip()
        if not text:
            continue
        if speaker != current_speaker:
            lines.append(f"\n## [{ts}] {speaker}\n")
            current_speaker = speaker
        lines.append(text)

    return "\n".join(lines) + "\n"


def write_markdown(
    segments: List[Dict[str, Any]], path: Path, title: str = "Transcript"
) -> None:
    """Write a Markdown dialog file to *path*."""
    path.write_text(to_markdown(segments, title=title), encoding="utf-8")
