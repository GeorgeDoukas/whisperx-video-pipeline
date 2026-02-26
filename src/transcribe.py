"""WhisperX transcription and forced-alignment helpers."""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# WhisperX loads audio at 16 kHz (matches Whisper's expected input sample rate).
# See whisperx.load_audio / faster-whisper documentation for the current version.
WHISPER_SAMPLE_RATE = 16_000


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
    initial_prompt: Optional[str] = None,
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
    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "language": language,
        "chunk_size": chunk_size,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt
    result: Dict[str, Any] = model.transcribe(audio, **kwargs)
    return result


def transcribe_chunked(
    model: Any,
    audio_path: Path,
    language: str,
    batch_size: int,
    chunk_size: int,
    checkpoint: Any,
    segment_minutes: int = 10,
    initial_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe *audio_path* in fixed-length segments with per-segment checkpointing.

    This enables the pipeline to *resume* after an interruption: completed
    segments are loaded from *checkpoint* and skipped, so only the remaining
    segments are processed.

    Parameters
    ----------
    model:
        A loaded WhisperX ASR model.
    audio_path:
        Path to the WAV audio file.
    language:
        ISO-639-1 language code.
    batch_size:
        WhisperX transcription batch size.
    chunk_size:
        WhisperX internal VAD chunk size (seconds).
    checkpoint:
        A :class:`src.checkpoint.Checkpoint` instance used to persist progress.
    segment_minutes:
        Duration of each audio segment in minutes.  Smaller values mean finer
        resume granularity at the cost of more checkpoint writes.
    initial_prompt:
        Optional text passed to Whisper as an initial prompt (e.g. to hint
        that technical English terms should be kept in English).
    """
    import whisperx  # type: ignore

    logger.info("Loading audio for chunked transcription: %s", audio_path)
    audio = whisperx.load_audio(str(audio_path))

    segment_samples = segment_minutes * 60 * WHISPER_SAMPLE_RATE
    total_samples = len(audio)
    n_segments = math.ceil(total_samples / segment_samples)

    logger.info(
        "Audio length %.1f min → %d segment(s) of %d min each",
        total_samples / WHISPER_SAMPLE_RATE / 60,
        n_segments,
        segment_minutes,
    )

    all_segments: List[Dict[str, Any]] = []
    detected_language: Optional[str] = None

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "language": language,
        "chunk_size": chunk_size,
    }
    if initial_prompt:
        kwargs["initial_prompt"] = initial_prompt

    for i in range(n_segments):
        stage_key = f"transcribe_chunk_{i}"
        if checkpoint.is_done(stage_key):
            seg_data = checkpoint.get_stage_data(stage_key)
            logger.info("Segment %d/%d loaded from checkpoint (%d segs)", i + 1, n_segments, len(seg_data))
            all_segments.extend(seg_data)
            continue

        start_sample = i * segment_samples
        end_sample = min((i + 1) * segment_samples, total_samples)
        chunk_audio = audio[start_sample:end_sample]
        offset = start_sample / WHISPER_SAMPLE_RATE

        logger.info(
            "Transcribing segment %d/%d (%.1f–%.1f min)…",
            i + 1,
            n_segments,
            offset / 60,
            end_sample / WHISPER_SAMPLE_RATE / 60,
        )
        result = model.transcribe(chunk_audio, **kwargs)

        if detected_language is None:
            detected_language = result.get("language")

        segs: List[Dict[str, Any]] = result.get("segments", [])
        # Shift timestamps by the segment offset
        for seg in segs:
            seg["start"] += offset
            seg["end"] += offset
            for word in seg.get("words", []):
                if "start" in word:
                    word["start"] += offset
                if "end" in word:
                    word["end"] += offset

        checkpoint.mark_done(stage_key, segs)
        logger.info("Segment %d/%d complete: %d segments", i + 1, n_segments, len(segs))
        all_segments.extend(segs)

    return {"segments": all_segments, "language": detected_language or language}


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
