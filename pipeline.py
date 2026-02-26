#!/usr/bin/env python3
"""Offline WhisperX pipeline for long video transcription.

Supports:
  - CPU-first, GPU optional
  - Windows & Linux
  - Resume-safe checkpointing
  - Progress-friendly output via tqdm
  - Greek language (and any other Whisper-supported language)
  - Optional speaker diarization
  - Markdown dialog output (suitable for LLM summarisation)
  - SRT subtitle output
  - Chapter embedding into the output video

Usage example
-------------
  python pipeline.py myvideo.mp4 --language el --embed-chapters

Run ``python pipeline.py --help`` for the full argument list.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm  # type: ignore

from src.audio import extract_audio
from src.chapters import DEFAULT_CHAPTER_INTERVAL, embed_chapters, generate_chapters
from src.checkpoint import Checkpoint
from src.diarize import assign_speakers, diarize
from src.export import write_markdown, write_srt
from src.transcribe import align, load_model, transcribe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _stage_audio(args: argparse.Namespace, checkpoint: Checkpoint) -> Path:
    """Stage 1 – extract audio from the input file."""
    stem = args.input.resolve().stem
    audio_path = args.output_dir.resolve() / f"{stem}.wav"
    if checkpoint.is_done("audio"):
        logger.info("Audio already extracted, resuming from checkpoint.")
        return Path(checkpoint.get_stage_data("audio"))

    with tqdm(total=1, desc="Extracting audio", unit="file") as pbar:
        extract_audio(args.input.resolve(), audio_path, show_progress=False)
        pbar.update(1)
    checkpoint.mark_done("audio", str(audio_path))
    logger.info("Audio extracted: %s", audio_path)
    return audio_path


def _stage_transcribe(
    args: argparse.Namespace, audio_path: Path, checkpoint: Checkpoint
) -> dict:
    """Stage 2 – run WhisperX ASR."""
    if checkpoint.is_done("transcribe"):
        logger.info("Transcription loaded from checkpoint.")
        return checkpoint.get_stage_data("transcribe")

    model = load_model(args.model, args.device, args.compute_type)
    with tqdm(total=1, desc="Transcribing", unit="file") as pbar:
        result = transcribe(
            model,
            audio_path,
            language=args.language,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        pbar.update(1)
    checkpoint.mark_done("transcribe", result)
    logger.info("Transcription complete: %d segments", len(result.get("segments", [])))
    del model
    return result


def _stage_align(
    args: argparse.Namespace, result: dict, audio_path: Path, checkpoint: Checkpoint
) -> dict:
    """Stage 3 – forced alignment."""
    if args.no_align:
        return result
    if checkpoint.is_done("align"):
        logger.info("Alignment loaded from checkpoint.")
        return checkpoint.get_stage_data("align")

    with tqdm(total=1, desc="Aligning", unit="file") as pbar:
        aligned_result = align(result, audio_path, language=args.language, device=args.device)
        pbar.update(1)
    checkpoint.mark_done("align", aligned_result)
    logger.info("Alignment complete.")
    return aligned_result


def _stage_diarize(
    args: argparse.Namespace, aligned_result: dict, audio_path: Path, checkpoint: Checkpoint
) -> dict:
    """Stage 4 – optional speaker diarization."""
    if not args.diarize:
        return aligned_result
    if checkpoint.is_done("diarize"):
        logger.info("Diarization loaded from checkpoint.")
        return checkpoint.get_stage_data("diarize")

    if not args.hf_token:
        logger.error(
            "Speaker diarization requires --hf-token (HuggingFace access token)."
        )
        sys.exit(1)

    with tqdm(total=1, desc="Diarizing", unit="file") as pbar:
        diarize_segments = diarize(
            audio_path,
            hf_token=args.hf_token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )
        pbar.update(1)
    result = assign_speakers(aligned_result, diarize_segments)
    checkpoint.mark_done("diarize", result)
    logger.info("Diarization complete.")
    return result


def _stage_export(
    args: argparse.Namespace, segments: list, checkpoint: Checkpoint
) -> None:
    """Stage 5 – write SRT and Markdown output files."""
    stem = args.input.resolve().stem
    output_dir = args.output_dir.resolve()
    srt_path = output_dir / f"{stem}.srt"
    md_path = output_dir / f"{stem}.md"

    with tqdm(total=2, desc="Exporting", unit="file") as pbar:
        write_srt(segments, srt_path)
        pbar.update(1)
        write_markdown(segments, md_path, title=stem)
        pbar.update(1)

    logger.info("SRT written:      %s", srt_path)
    logger.info("Markdown written: %s", md_path)


def _stage_embed_chapters(
    args: argparse.Namespace, segments: list
) -> None:
    """Stage 6 – generate chapters and embed them into the video."""
    if not args.embed_chapters:
        return

    chapters = generate_chapters(segments, interval=args.chapter_interval)
    if not chapters:
        logger.warning("No chapters generated (transcript may be empty).")
        return

    input_path = args.input.resolve()
    chaptered_path = args.output_dir.resolve() / f"{input_path.stem}.chaptered{input_path.suffix}"
    with tqdm(total=1, desc="Embedding chapters", unit="file") as pbar:
        embed_chapters(input_path, chapters, chaptered_path, show_progress=False)
        pbar.update(1)
    logger.info("Chaptered video written: %s", chaptered_path)


def run(args: argparse.Namespace) -> None:
    """Execute the full pipeline according to *args*."""

    input_path: Path = args.input.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    checkpoint = Checkpoint(output_dir / f"{stem}.checkpoint.json")

    audio_path = _stage_audio(args, checkpoint)
    result = _stage_transcribe(args, audio_path, checkpoint)
    aligned_result = _stage_align(args, result, audio_path, checkpoint)
    final_result = _stage_diarize(args, aligned_result, audio_path, checkpoint)

    segments: List = final_result.get("segments", [])
    _stage_export(args, segments, checkpoint)
    _stage_embed_chapters(args, segments)

    checkpoint.set("pipeline_complete", True)
    logger.info("Pipeline complete. Output directory: %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Offline WhisperX pipeline for long video transcription",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("input", type=Path, help="Input video or audio file")

    # Model / transcription
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model size (tiny/base/small/medium/large-v2/large-v3)",
    )
    parser.add_argument(
        "--language",
        default="el",
        help="ISO-639-1 language code (default: el for Greek)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        dest="compute_type",
        help="Quantisation type (default: int8 for CPU, float16 for CUDA)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        dest="batch_size",
        help="Transcription batch size",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        dest="chunk_size",
        help="Audio chunk size in seconds",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        dest="no_align",
        help="Skip forced-alignment stage",
    )

    # Diarization
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires --hf-token)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        dest="hf_token",
        help="HuggingFace token for pyannote diarization models",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        dest="min_speakers",
        help="Minimum number of speakers hint for diarization",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        dest="max_speakers",
        help="Maximum number of speakers hint for diarization",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        dest="output_dir",
        help="Directory for all output files (default: same directory as input)",
    )
    parser.add_argument(
        "--embed-chapters",
        action="store_true",
        dest="embed_chapters",
        help="Embed generated chapters into a copy of the input video",
    )
    parser.add_argument(
        "--chapter-interval",
        type=int,
        default=DEFAULT_CHAPTER_INTERVAL,
        dest="chapter_interval",
        help="Chapter interval in seconds",
    )

    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args()

    # Resolve defaults that depend on other arguments
    if args.output_dir is None:
        args.output_dir = args.input.parent
    if args.compute_type is None:
        args.compute_type = "int8" if args.device == "cpu" else "float16"

    run(args)


if __name__ == "__main__":
    main()
