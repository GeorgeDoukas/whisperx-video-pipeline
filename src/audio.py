"""Audio extraction helpers using ffmpeg.

ffmpeg must be available on PATH (both Windows and Linux are supported).
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_path: Path, show_progress: bool = True) -> Path:
    """Extract audio track from *video_path* as a 16 kHz mono WAV file.

    If *output_path* already exists the function returns early, enabling
    resume across interrupted runs.
    """
    if output_path.exists():
        logger.info("Audio already extracted, skipping: %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y"]
    if not show_progress:
        cmd += ["-loglevel", "quiet"]
    cmd += [
        "-i", str(video_path),
        "-ac", "1",        # mono
        "-ar", "16000",    # 16 kHz – expected by Whisper
        "-vn",             # no video stream
        str(output_path),
    ]

    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_path


def get_duration(media_path: Path) -> float:
    """Return duration of a media file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])
