"""LLM-based transcript summarisation via LM Studio.

LM Studio exposes an OpenAI-compatible REST API (default: http://localhost:1234/v1).
This module connects to that endpoint through ``langchain-openai`` and generates
a structured summary of the transcript produced by the WhisperX pipeline.

Usage example
-------------
    from src.summarize import summarize_transcript

    summary = summarize_transcript(
        transcript="… full markdown transcript …",
        base_url="http://localhost:1234/v1",
        model="local-model",
    )
    print(summary)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = """\
You are an expert technical meeting assistant.
Below is a transcript of a recorded session. The transcript may contain a mix
of Greek and English text; technical terms in English should be kept in English
in your output.

Your task is to produce a concise, well-structured summary in the **same
primary language as the transcript** (Greek if the discussion is in Greek).
The summary must include:
1. A short overview (2-4 sentences).
2. Key topics discussed, as a bullet list.
3. Any decisions made or action items identified.

Transcript:
{transcript}
"""


def summarize_transcript(
    transcript: str,
    base_url: str = "http://localhost:1234/v1",
    model: str = "local-model",
    max_tokens: int = 1024,
    api_key: str = "lm-studio",  # LM Studio accepts any non-empty string for local inference
    temperature: float = 0.3,
) -> str:
    """Generate a summary of *transcript* using a local LM Studio model.

    Parameters
    ----------
    transcript:
        Full transcript text (plain text or Markdown dialog).
    base_url:
        Base URL of the LM Studio OpenAI-compatible server
        (e.g. ``"http://localhost:1234/v1"``).
    model:
        Model identifier as configured in LM Studio.
    max_tokens:
        Maximum number of tokens in the generated summary.
    api_key:
        Placeholder API key accepted by LM Studio (any non-empty string works).
    temperature:
        Sampling temperature for generation.

    Returns
    -------
    str
        Generated summary text.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for summarisation. "
            "Install it with: pip install langchain-openai"
        ) from exc

    from langchain_core.messages import HumanMessage  # type: ignore

    logger.info("Connecting to LM Studio at %s (model=%s)…", base_url, model)
    llm = ChatOpenAI(
        base_url=base_url,
        model=model,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    prompt = _SUMMARY_PROMPT.format(transcript=transcript)
    logger.info("Generating transcript summary…")
    response = llm.invoke([HumanMessage(content=prompt)])
    summary: str = response.content
    logger.info("Summary generated (%d chars).", len(summary))
    return summary


def write_summary(summary: str, path) -> None:
    """Write *summary* as a Markdown file to *path*."""
    from pathlib import Path

    path = Path(path)
    path.write_text(f"# Summary\n\n{summary}\n", encoding="utf-8")
    logger.info("Summary written: %s", path)
