# whisperx-video-pipeline

Offline WhisperX pipeline for long video transcription.

## Features

| Feature | Detail |
|---|---|
| **CPU-first, GPU optional** | Defaults to CPU (`int8`); pass `--device cuda` for GPU (`float16`) |
| **Windows & Linux** | Uses `pathlib` and `subprocess`; no platform-specific code |
| **Resume-safe** | JSON checkpoint saved alongside outputs; re-running skips completed stages |
| **Progress-friendly** | Per-stage `tqdm` progress bars |
| **Greek language optimised** | Default language is `el`; any Whisper-supported language works |
| **Optional diarization** | pyannote.audio via WhisperX; requires a HuggingFace token |
| **Markdown dialog** | LLM-ready transcript with speaker headings and timestamps |
| **SRT subtitles** | Standard `.srt` subtitle file |
| **Chapter embedding** | Chapters embedded into a copy of the video using `ffmpeg` stream-copy |

## Requirements

* Python ≥ 3.9
* [ffmpeg](https://ffmpeg.org/) on `PATH`
* Python packages (install once):

```bash
pip install -r requirements.txt
```

> **GPU acceleration** also requires a CUDA-capable GPU and the matching
> `torch` CUDA build.

## Quick start

```bash
# Transcribe a Greek video (CPU, medium model)
python pipeline.py lecture.mp4

# Same but with speaker diarization and chapters embedded
python pipeline.py lecture.mp4 --diarize --hf-token hf_XXX --embed-chapters

# English video, GPU, large model
python pipeline.py talk.mp4 --language en --device cuda --model large-v3

# Specify output directory
python pipeline.py lecture.mp4 --output-dir ./output
```

## CLI reference

```
python pipeline.py <input> [options]

positional arguments:
  input                 Input video or audio file

transcription options:
  --model MODEL         Whisper model size (tiny/base/small/medium/large-v2/large-v3) [default: medium]
  --language LANGUAGE   ISO-639-1 language code [default: el]
  --device {cpu,cuda}   Compute device [default: cpu]
  --compute-type TYPE   Quantisation type (int8 / float16 / float32) [auto]
  --batch-size N        Transcription batch size [default: 8]
  --chunk-size N        Audio chunk size in seconds [default: 30]
  --no-align            Skip forced-alignment stage

diarization options:
  --diarize             Enable speaker diarization
  --hf-token TOKEN      HuggingFace access token
  --min-speakers N      Minimum number of speakers
  --max-speakers N      Maximum number of speakers

output options:
  --output-dir DIR      Output directory [default: same as input]
  --embed-chapters      Embed chapters into a copy of the input video
  --chapter-interval N  Chapter interval in seconds [default: 300]
```

## Output files

Given `input.mp4` and `--output-dir ./out`:

| File | Description |
|---|---|
| `out/input.wav` | Extracted 16 kHz mono audio |
| `out/input.srt` | SRT subtitles |
| `out/input.md` | Markdown dialog (speaker-labelled if diarized) |
| `out/input.chaptered.mp4` | Video with embedded chapters (`--embed-chapters`) |
| `out/input.checkpoint.json` | Pipeline checkpoint (enables resume) |

## Resume behaviour

If the pipeline is interrupted, simply re-run the exact same command.
Completed stages are skipped automatically based on the checkpoint file.
Delete `*.checkpoint.json` to force a full re-run.

## Running tests

```bash
pip install pytest
pytest
```
