"""Tests for src.summarize (LM Studio summarisation helpers)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_langchain_mocks():
    """Return (mock_langchain_openai_module, mock_langchain_core_module, mock_llm)."""
    fake_response = MagicMock()
    fake_response.content = "This is a summary."

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = fake_response

    mock_chat_openai_cls = MagicMock(return_value=mock_llm)

    mock_lc_openai = MagicMock()
    mock_lc_openai.ChatOpenAI = mock_chat_openai_cls

    mock_human_msg = MagicMock(side_effect=lambda content: MagicMock(content=content))
    mock_lc_core_messages = MagicMock()
    mock_lc_core_messages.HumanMessage = mock_human_msg

    return mock_lc_openai, mock_lc_core_messages, mock_llm, mock_chat_openai_cls


# ---------------------------------------------------------------------------
# summarize_transcript
# ---------------------------------------------------------------------------


def test_summarize_transcript_returns_string():
    """summarize_transcript returns the LLM response as a string."""
    from src.summarize import summarize_transcript

    mock_lc_openai, mock_lc_core_messages, mock_llm, _ = _make_langchain_mocks()
    mock_llm.invoke.return_value.content = "This is a summary."

    with patch.dict(sys.modules, {
        "langchain_openai": mock_lc_openai,
        "langchain_core": MagicMock(),
        "langchain_core.messages": mock_lc_core_messages,
    }):
        result = summarize_transcript("Some transcript text.")

    assert result == "This is a summary."


def test_summarize_transcript_uses_base_url():
    """The provided base_url is forwarded to ChatOpenAI."""
    from src.summarize import summarize_transcript

    mock_lc_openai, mock_lc_core_messages, mock_llm, mock_cls = _make_langchain_mocks()

    with patch.dict(sys.modules, {
        "langchain_openai": mock_lc_openai,
        "langchain_core": MagicMock(),
        "langchain_core.messages": mock_lc_core_messages,
    }):
        summarize_transcript("text", base_url="http://myserver:5678/v1")

    _, kwargs = mock_cls.call_args
    assert kwargs["base_url"] == "http://myserver:5678/v1"


def test_summarize_transcript_uses_model():
    """The provided model name is forwarded to ChatOpenAI."""
    from src.summarize import summarize_transcript

    mock_lc_openai, mock_lc_core_messages, mock_llm, mock_cls = _make_langchain_mocks()

    with patch.dict(sys.modules, {
        "langchain_openai": mock_lc_openai,
        "langchain_core": MagicMock(),
        "langchain_core.messages": mock_lc_core_messages,
    }):
        summarize_transcript("text", model="my-local-llama")

    _, kwargs = mock_cls.call_args
    assert kwargs["model"] == "my-local-llama"


def test_summarize_transcript_prompt_contains_transcript():
    """The transcript text is included in the prompt sent to the LLM."""
    from src.summarize import summarize_transcript

    mock_lc_openai, mock_lc_core_messages, mock_llm, _ = _make_langchain_mocks()

    captured: dict = {}

    def capture_msg(content):
        captured["content"] = content
        return MagicMock()

    mock_lc_core_messages.HumanMessage = capture_msg

    with patch.dict(sys.modules, {
        "langchain_openai": mock_lc_openai,
        "langchain_core": MagicMock(),
        "langchain_core.messages": mock_lc_core_messages,
    }):
        summarize_transcript("Hello world transcript")

    assert "Hello world transcript" in captured["content"]


def test_summarize_transcript_missing_langchain_raises():
    """ImportError is raised with a helpful message when langchain-openai is absent."""
    from src.summarize import summarize_transcript

    # Temporarily hide langchain_openai from sys.modules
    original = sys.modules.pop("langchain_openai", None)
    try:
        with patch.dict(sys.modules, {"langchain_openai": None}):
            with pytest.raises(ImportError, match="langchain-openai"):
                summarize_transcript("text")
    finally:
        if original is not None:
            sys.modules["langchain_openai"] = original


# ---------------------------------------------------------------------------
# write_summary
# ---------------------------------------------------------------------------


def test_write_summary_creates_file(tmp_path):
    """write_summary writes a Markdown file with a heading."""
    from src.summarize import write_summary

    path = tmp_path / "out.summary.md"
    write_summary("Great meeting summary.", path)

    content = path.read_text(encoding="utf-8")
    assert "# Summary" in content
    assert "Great meeting summary." in content


def test_write_summary_utf8(tmp_path):
    """write_summary handles non-ASCII (Greek) text correctly."""
    from src.summarize import write_summary

    path = tmp_path / "out.summary.md"
    write_summary("Σύνοψη: Τεχνική συνάντηση.", path)

    content = path.read_text(encoding="utf-8")
    assert "Σύνοψη" in content

