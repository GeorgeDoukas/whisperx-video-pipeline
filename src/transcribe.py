"""WhisperX transcription and forced-alignment helpers."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_model(model_name: str, device: str, compute_type: str):
    """Load a WhisperX ASR model.

    Imported here so the rest of the codebase doesn't hard-depend on
    whisperx at import time.
    """
    import whisperx  # type: ignore

    logger.info("Loading WhisperX model '%s' on %s (%s)", model_name, device, compute_type)
    return whisperx.load_model(model_name, device, compute_type=compute_type)


def transcribe(
    model: Any,
    audio_path: Path,
    language: str,
    batch_size: int,
    chunk_size: int,
) -> Dict[str, Any]:
    """Run ASR on *audio_path* and return the raw WhisperX result dict."""
    import whisperx  # type: ignore

    logger.info("Loading audio: %s", audio_path)
    audio = whisperx.load_audio(str(audio_path))

    logger.info(
        "Transcribing (language=%s, batch_size=%d, chunk_size=%d)…",
        language,
        batch_size,
        chunk_size,
    )
    result: Dict[str, Any] = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
        chunk_size=chunk_size,
    )
    return result


def align(
    result: Dict[str, Any],
    audio_path: Path,
    language: str,
    device: str,
) -> Dict[str, Any]:
    """Run forced alignment on *result* to obtain word-level timestamps.

    Returns the aligned result dict.  If no alignment model is available
    for *language*, the original *result* is returned unchanged so the
    pipeline can still produce output.
    """
    import whisperx  # type: ignore

    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
    except Exception as exc:  # alignment model not available for language
        logger.warning(
            "Forced alignment not available for language '%s': %s – skipping alignment.",
            language,
            exc,
        )
        return result

    audio = whisperx.load_audio(str(audio_path))

    logger.info("Running forced alignment…")
    aligned: Dict[str, Any] = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    return aligned
