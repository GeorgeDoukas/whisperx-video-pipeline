"""Chapter generation and ffmpeg chapter-embedding helpers."""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

DEFAULT_CHAPTER_INTERVAL = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Chapter generation
# ---------------------------------------------------------------------------


def generate_chapters(
    segments: List[Dict[str, Any]],
    interval: int = DEFAULT_CHAPTER_INTERVAL,
) -> List[Dict[str, Any]]:
    """Divide transcript *segments* into chapters of roughly *interval* seconds.

    Each chapter title is seeded from the first few words of the first
    segment in that window.
    """
    if not segments:
        return []

    total_end = segments[-1]["end"]
    chapters: List[Dict[str, Any]] = []
    current_time = 0.0
    chapter_num = 1

    while current_time < total_end:
        end_time = min(current_time + interval, total_end)

        # Pull a short excerpt from the first segment inside the window
        excerpt = ""
        for seg in segments:
            if seg["start"] >= current_time:
                excerpt = seg.get("text", "").strip()[:60]
                break

        title = f"Chapter {chapter_num}: {excerpt}" if excerpt else f"Chapter {chapter_num}"
        chapters.append({"start": current_time, "end": end_time, "title": title})

        current_time = end_time
        chapter_num += 1

    return chapters


# ---------------------------------------------------------------------------
# FFMETADATA serialisation
# ---------------------------------------------------------------------------


def to_ffmetadata(chapters: List[Dict[str, Any]]) -> str:
    """Serialise *chapters* to an FFMETADATA1 string accepted by ffmpeg."""
    lines = [";FFMETADATA1", ""]
    for ch in chapters:
        start_ms = int(ch["start"] * 1000)
        end_ms = int(ch["end"] * 1000)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={ch['title']}",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ffmpeg chapter embedding
# ---------------------------------------------------------------------------


def embed_chapters(
    video_path: Path,
    chapters: List[Dict[str, Any]],
    output_path: Path,
    show_progress: bool = True,
) -> None:
    """Write *chapters* into *video_path* and save result to *output_path*.

    Uses ffmpeg stream-copy so no re-encoding is required.
    """
    metadata_content = to_ffmetadata(chapters)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(metadata_content)
        meta_path = tmp.name

    try:
        cmd = ["ffmpeg", "-y"]
        if not show_progress:
            cmd += ["-loglevel", "quiet"]
        cmd += [
            "-i", str(video_path),
            "-i", meta_path,
            "-map_metadata", "1",
            "-codec", "copy",
            str(output_path),
        ]

        logger.info("Embedding chapters into video: %s", output_path)
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(meta_path)
