# unladen-swallm

CLI tool for benchmarking OpenAI-compatible LLM APIs. Compare inference engines head-to-head: same model, same prompts, same GPU — which service is faster?

## Project structure

- `unladen_swallm/cli.py` — Click CLI: `list-models`, `generate`, `benchmark`
- `unladen_swallm/client.py` — `OpenAICompatibleClient` wrapping `openai.AsyncOpenAI`
- `unladen_swallm/models.py` — `Model` and `Endpoint` dataclasses
- `unladen_swallm/stats.py` — Summary statistics (avg TTFT, tok/s, P50/P99)
- Entry point: `swallm` (defined in pyproject.toml)

## Commands

```bash
pytest                    # run tests
swallm benchmark -h       # see all benchmark options
swallm list-models        # list models on default endpoint
```

## Roadmap

### Next up

- **Concurrency scaling profiles**: Auto-run 1/2/4/8 concurrent requests per endpoint, report throughput scaling and efficiency loss. The data already shows Ollama serializes while llama-swap parallelizes — surface this automatically.

- **Thinking model support**: Handle `reasoning_content` alongside `content` in streaming responses. Separate thinking tokens from answer tokens in metrics. Currently Qwen3.5 shows empty content via Ollama's OpenAI compat layer.

- **Server-side timings**: Parse llama.cpp `timings` fields (prompt_ms, predicted_ms, predicted_per_second) when present in responses — far more precise than wall-clock measurement.

- **Cold vs warm start**: Distinguish model load time from inference time. First request to an unloaded model includes load time; subsequent requests don't. Report both. Critical for swap-based serving.

- **VRAM monitoring**: Capture GPU memory via `nvidia-smi` or `pynvml` before/after model load. Report delta per model. Flag models that don't fit.

### Done

- OpenAI-compatible client (any /v1/chat/completions endpoint)
- Multi-endpoint comparison (`--endpoint/-E`, sequential execution)
- GPU memory management (auto-unload between endpoints: Ollama keep_alive=0, llama-swap /unload, configurable delay)
- Summary statistics table (avg TTFT, avg tok/s, P50/P99, error count)
- Per-endpoint API keys (`--endpoint-key`)
- Concurrency modes (global, per-model)
- Prompts file with multi-line support

## Conventions

- Tests use `monkeypatch` with fake client classes (no real network calls)
- `httpx` for non-OpenAI HTTP calls (unload endpoints)
- `rich` for colored terminal output and summary tables
- Default timeout 120s (cold-start large models need it)
