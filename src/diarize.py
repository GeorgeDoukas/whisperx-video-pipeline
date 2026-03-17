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
    import torch  # type: ignore
    import whisperx  # type: ignore
    from pyannote.audio import Pipeline  # type: ignore

    logger.info("Running speaker diarization…")
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )

    # Load audio as preloaded in-memory format that pyannote expects
    waveform = whisperx.load_audio(str(audio_path))
    audio = {
        "waveform": torch.from_numpy(waveform).unsqueeze(0),  # Add channel dimension
        "sample_rate": 16000,  # WhisperX uses 16kHz
    }

    kwargs: Dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    diarize_output = diarize_model(audio, **kwargs)
    
    # Convert DiarizeOutput to a format compatible with our speaker assignment
    # DiarizeOutput is iterable and yields (Segment, speaker_label) tuples
    segments = []
    for segment, speaker_label in diarize_output.iteritems():
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker_label,
        })
    
    return segments


def assign_speakers(
    aligned_result: Dict[str, Any],
    diarize_segments: Any,
) -> Dict[str, Any]:
    """Merge diarization labels into the aligned transcript segments."""
    logger.info("Assigning speaker labels to transcript segments…")
    
    # Assign speakers to words based on temporal overlap
    for segment in aligned_result.get("segments", []):
        for word in segment.get("words", []):
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            
            # Find the diarization segment that overlaps with this word
            for dia_seg in diarize_segments:
                dia_start = dia_seg.get("start", 0)
                dia_end = dia_seg.get("end", 0)
                
                # Check if word overlaps with diarization segment
                if word_start < dia_end and word_end > dia_start:
                    word["speaker"] = dia_seg.get("speaker", "UNKNOWN")
                    break
        
        # Also assign speaker to segment level based on dominant speaker in words
        speakers = []
        for word in segment.get("words", []):
            if "speaker" in word:
                speakers.append(word["speaker"])
        
        if speakers:
            # Use the most common speaker in the segment
            from collections import Counter
            segment["speaker"] = Counter(speakers).most_common(1)[0][0]
    
    return aligned_result
