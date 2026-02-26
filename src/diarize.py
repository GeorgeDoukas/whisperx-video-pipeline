"""Optional speaker diarization via pyannote.audio through WhisperX."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def diarize(
    audio_path: Path,
    hf_token: str,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> Any:
    """Run pyannote diarization and return raw diarization segments."""
    import whisperx  # type: ignore

    logger.info("Running speaker diarization…")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token)

    kwargs: Dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    return diarize_model(str(audio_path), **kwargs)


def assign_speakers(
    aligned_result: Dict[str, Any],
    diarize_segments: Any,
) -> Dict[str, Any]:
    """Merge diarization labels into the aligned transcript segments."""
    import whisperx  # type: ignore

    logger.info("Assigning speaker labels to transcript segments…")
    return whisperx.assign_word_speakers(diarize_segments, aligned_result)
