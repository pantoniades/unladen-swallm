"""Async client wrapping openai.AsyncOpenAI for OpenAI-compatible endpoints.

Works with Ollama (/v1), vLLM, llama-swap, OpenAI, and any other provider
that implements the OpenAI Chat Completions API.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, APIStatusError
from .models import Model

logger = logging.getLogger(__name__)


def _normalize_base_url(host: str) -> str:
    """Append /v1 if not already present."""
    url = host.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


class OpenAICompatibleClient:
    """Async client for any OpenAI-compatible LLM API.

    Defaults to http://localhost:11434 (Ollama's OpenAI-compat endpoint).
    Pass api_key="none" for local providers that don't require authentication.
    """

    def __init__(self, host: str = "http://localhost:11434", api_key: str = "none"):
        self._base_url = _normalize_base_url(host)
        self._client = AsyncOpenAI(base_url=self._base_url, api_key=api_key)
        self._use_stream_options = True  # auto-disabled on first 400/422

    async def close(self) -> None:
        await self._client.close()

    async def list_models(self) -> List[Model]:
        """Return a list of Model objects available on the server."""
        logger.debug("Fetching models from %s", self._base_url)
        response = await self._client.models.list()
        return [Model.from_openai(m) for m in response.data]

    async def generate(self, model: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Stream a chat completion and return timing + token metrics.

        Returns a dict with:
          content, elapsed, time_to_first_token, generation_time,
          prompt_tokens, completion_tokens, tokens_per_sec,
          elapsed_tokens_per_sec, token_counts_available
        """
        messages = [{"role": "user", "content": prompt}]

        async def _stream(include_usage: bool) -> Dict[str, Any]:
            stream_kwargs: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            if include_usage:
                stream_kwargs["stream_options"] = {"include_usage": True}

            content_parts: List[str] = []
            prompt_tokens: Optional[int] = None
            completion_tokens: Optional[int] = None
            time_to_first_token: Optional[float] = None

            start = asyncio.get_event_loop().time()
            async with await self._client.chat.completions.create(**stream_kwargs) as stream:
                async for chunk in stream:
                    # Capture usage from the final chunk (stream_options)
                    if chunk.usage is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens

                    if not chunk.choices:
                        continue

                    delta_content = chunk.choices[0].delta.content
                    if delta_content is not None and delta_content != "":
                        if time_to_first_token is None:
                            time_to_first_token = asyncio.get_event_loop().time() - start
                        content_parts.append(delta_content)

            elapsed = asyncio.get_event_loop().time() - start
            content = "".join(content_parts)

            generation_time: Optional[float] = None
            if time_to_first_token is not None:
                generation_time = elapsed - time_to_first_token

            token_counts_available = prompt_tokens is not None and completion_tokens is not None

            tokens_per_sec: Optional[float] = None
            elapsed_tokens_per_sec: Optional[float] = None
            if token_counts_available and completion_tokens is not None:
                if generation_time and generation_time > 0:
                    tokens_per_sec = completion_tokens / generation_time
                if elapsed > 0:
                    elapsed_tokens_per_sec = completion_tokens / elapsed

            return {
                "content": content,
                "elapsed": elapsed,
                "time_to_first_token": time_to_first_token,
                "generation_time": generation_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens_per_sec": tokens_per_sec,
                "elapsed_tokens_per_sec": elapsed_tokens_per_sec,
                "token_counts_available": token_counts_available,
            }

        if self._use_stream_options:
            try:
                return await _stream(include_usage=True)
            except APIStatusError as exc:
                if exc.status_code in (400, 422):
                    logger.debug(
                        "stream_options not supported by server (%s %s); disabling",
                        exc.status_code, exc.message,
                    )
                    self._use_stream_options = False
                    result = await _stream(include_usage=False)
                    result["token_counts_available"] = False
                    return result
                raise
        else:
            result = await _stream(include_usage=False)
            result["token_counts_available"] = False
            return result
