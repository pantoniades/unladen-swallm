"""Tests for OpenAICompatibleClient."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from unladen_swallm.client import OpenAICompatibleClient, _normalize_base_url
from unladen_swallm.models import Model


# ---------------------------------------------------------------------------
# _normalize_base_url
# ---------------------------------------------------------------------------

def test_normalize_base_url_appends_v1():
    assert _normalize_base_url("http://localhost:11434") == "http://localhost:11434/v1"


def test_normalize_base_url_no_double_v1():
    assert _normalize_base_url("http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_normalize_base_url_strips_trailing_slash():
    assert _normalize_base_url("http://localhost:11434/") == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# Helpers to build fake OpenAI model objects
# ---------------------------------------------------------------------------

def _fake_openai_model(model_id: str, owned_by: str = "library") -> MagicMock:
    m = MagicMock()
    m.id = model_id
    m.created = 1700000000
    m.owned_by = owned_by
    return m


def _fake_models_list_response(*model_ids: str) -> MagicMock:
    resp = MagicMock()
    resp.data = [_fake_openai_model(mid) for mid in model_ids]
    return resp


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

def test_list_models_returns_model_instances():
    """list_models() should return Model objects with correct names."""
    fake_response = _fake_models_list_response("gemma3", "llama3.2:3b")

    with patch("unladen_swallm.client.AsyncOpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.models = MagicMock()
        instance.models.list = AsyncMock(return_value=fake_response)
        instance.close = AsyncMock()

        client = OpenAICompatibleClient(host="http://localhost:11434")
        models = asyncio.run(client.list_models())

    assert [m.name for m in models] == ["gemma3", "llama3.2:3b"]
    assert all(isinstance(m, Model) for m in models)


# ---------------------------------------------------------------------------
# generate — success with token counts
# ---------------------------------------------------------------------------

def _make_chunk(content=None, role=None, prompt_tokens=None, completion_tokens=None):
    """Build a fake streaming chunk."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    delta.role = role
    choice = MagicMock()
    choice.delta = delta
    chunk.choices = [choice]
    if prompt_tokens is not None or completion_tokens is not None:
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        chunk.usage = usage
    else:
        chunk.usage = None
    return chunk


class _FakeStream:
    """Async context manager that yields fake chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for c in self._chunks:
            yield c


def test_generate_success_with_token_counts():
    """generate() should return timing metrics and token counts when available."""
    chunks = [
        _make_chunk(role="assistant"),           # role-only chunk, no content
        _make_chunk(content="Hello "),           # first content chunk → TTFT
        _make_chunk(content="world"),            # second content chunk
        _make_chunk(prompt_tokens=10, completion_tokens=2),  # usage chunk
    ]

    with patch("unladen_swallm.client.AsyncOpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat = MagicMock()
        instance.chat.completions = MagicMock()
        instance.chat.completions.create = AsyncMock(return_value=_FakeStream(chunks))
        instance.close = AsyncMock()

        client = OpenAICompatibleClient(host="http://localhost:11434")
        result = asyncio.run(client.generate("gemma3", "say hello"))

    assert result["content"] == "Hello world"
    assert result["token_counts_available"] is True
    assert result["prompt_tokens"] == 10
    assert result["completion_tokens"] == 2
    assert result["time_to_first_token"] is not None
    assert result["time_to_first_token"] >= 0
    assert result["tokens_per_sec"] is not None
    assert result["elapsed"] >= 0


# ---------------------------------------------------------------------------
# generate — stream_options fallback on 400/422
# ---------------------------------------------------------------------------

def test_generate_stream_options_fallback():
    """On 400/422, client disables stream_options and retries without usage data."""
    from openai import APIStatusError

    chunks_no_usage = [
        _make_chunk(content="Hi"),
    ]

    # First call raises 400; second call returns chunks without usage
    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate the server rejecting stream_options
            request = MagicMock()
            raise APIStatusError(
                message="stream_options not supported",
                response=MagicMock(status_code=400),
                body=None,
            )
        return _FakeStream(chunks_no_usage)

    with patch("unladen_swallm.client.AsyncOpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat = MagicMock()
        instance.chat.completions = MagicMock()
        instance.chat.completions.create = fake_create
        instance.close = AsyncMock()

        client = OpenAICompatibleClient(host="http://localhost:11434")
        result = asyncio.run(client.generate("gemma3", "hi"))

    assert result["content"] == "Hi"
    assert result["token_counts_available"] is False
    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None
    # Subsequent calls should skip stream_options (flag is now False)
    assert client._use_stream_options is False
