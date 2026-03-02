# unladen-swallm

Benchmark OpenAI-compatible LLM APIs. Compare inference engines head-to-head: same model, same prompts, same GPU — which service is faster?

Works with [Ollama](https://ollama.ai/), [llama-swap](https://github.com/mostlygeek/llama-swap), [vLLM](https://docs.vllm.ai/), [LM Studio](https://lmstudio.ai/), and any other service that implements `/v1/chat/completions`.

## Features

- **Multi-endpoint comparison** — benchmark across services with a single command
- **Summary statistics** — avg TTFT, avg tok/s, P50/P99 latency per endpoint+model
- **Warmup** — automatically loads models before timed runs (skip cold-start noise)
- **GPU memory management** — auto-unloads models between endpoints (Ollama, llama-swap)
- **JSON output** — pipe results to Claude/GPT for quality evaluation
- **Multiple prompts** — test models across diverse inputs
- **Concurrent requests** — async execution for load testing

## Installation

```bash
git clone https://github.com/pantoniades/unladen-swallm.git
cd unladen-swallm
pip install -e .
```

## Quick Start

```bash
# Benchmark a model on the default endpoint (localhost:11434)
swallm benchmark -m qwen3.5:27b -f text

# Compare two inference engines
swallm benchmark \
  --endpoint ollama=http://localhost:11434 \
  --endpoint llama-swap=http://localhost:11500 \
  -m qwen3.5:27b -m qwen3.5-27b \
  -P prompts.txt -f text

# Save JSON results for analysis
swallm benchmark -m gemma3:12b -P prompts.txt -r -o results.json

# Evaluate quality with Claude
claude "Rate these LLM responses: $(cat results.json)"
```

## Commands

### benchmark

```bash
# Basic usage
swallm benchmark                        # All models, default prompt, JSON
swallm benchmark -m llama3.1:8b         # Specific model
swallm benchmark --prompt "..."         # Custom prompt
swallm benchmark -P prompts.txt         # Multiple prompts from file

# Multi-endpoint comparison
swallm benchmark \
  -E ollama=http://localhost:11434 \
  -E vllm=http://localhost:8000 \
  -m qwen3.5:27b

# Per-endpoint API keys (for cloud services)
swallm benchmark \
  -E local=http://localhost:11434 \
  -E cloud=https://api.openai.com \
  --endpoint-key cloud=sk-abc123 \
  -m gpt-4o-mini

# Output control
swallm benchmark -o results.json       # Save to file
swallm benchmark -f text               # Text format with summary table
swallm benchmark -r                    # Include full response text

# Performance
swallm benchmark -c 3                  # 3 concurrent requests
swallm benchmark -t 180                # 180 second timeout
swallm benchmark --no-warmup           # Skip warmup requests
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt TEXT` | Single prompt text | built-in default |
| `-P, --prompts-file PATH` | File with prompts (single-line or `"""` multi-line) | |
| `-m, --model TEXT` | Model(s) to test (repeatable) | all available |
| `-E, --endpoint TEXT` | Endpoint as `label=url` (repeatable) | `--host` value |
| `--endpoint-key TEXT` | Per-endpoint API key as `label=key` (repeatable) | `--api-key` value |
| `-H, --host TEXT` | LLM API host URL (env: `LLM_HOST`) | `http://localhost:11434` |
| `--api-key TEXT` | API key (env: `OPENAI_API_KEY`) | `none` |
| `-c, --concurrent INT` | Concurrent requests | `1` |
| `--concurrency-mode` | `global` or `per-model` | `per-model` |
| `-t, --timeout FLOAT` | Per-request timeout in seconds | `120` |
| `--endpoint-delay FLOAT` | Seconds between endpoints for GPU cooldown | `5` |
| `--warmup / --no-warmup` | Warmup request per model before timed runs | `--warmup` |
| `-f, --format` | `json` or `text` | `json` |
| `-o, --output PATH` | Write to file instead of stdout | |
| `-r, --response` | Include full response text | off |
| `--exclude-errors` | Omit failed results from output | |
| `--errors-only` | Only show failed results | |

### list-models

```bash
swallm list-models                     # Pretty format
swallm list-models -f compact          # One line per model
swallm list-models -n                  # No color
swallm list-models -H http://host:8000 # Different server
```

### generate

```bash
swallm generate llama3.1:8b "Explain quantum computing"
```

## JSON Output Format

```json
{
  "config": {
    "endpoints": [{"label": "ollama", "url": "http://localhost:11434"}],
    "prompts": ["What is 2+2?"],
    "models": ["llama3.1:8b"],
    "concurrency": 1,
    "concurrency_mode": "per-model",
    "timeout": 120.0
  },
  "results": [
    {
      "endpoint": "ollama",
      "model": "llama3.1:8b",
      "prompt": "What is 2+2?",
      "status": "ok",
      "elapsed": 1.234,
      "metrics": {
        "elapsed": 1.234,
        "time_to_first_token": 0.089,
        "generation_time": 1.145,
        "prompt_tokens": 10,
        "completion_tokens": 25,
        "tokens_per_sec": 21.83,
        "elapsed_tokens_per_sec": 20.26,
        "token_counts_available": true
      }
    }
  ],
  "summary": [
    {
      "endpoint": "ollama",
      "model": "llama3.1:8b",
      "avg_ttft": 0.089,
      "avg_tps": 21.83,
      "p50_elapsed": 1.234,
      "p99_elapsed": 1.234,
      "requests": 1,
      "errors": 0
    }
  ]
}
```

## Text Output

With `-f text`, results include a summary table:

```
Summary
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Endpoint   ┃ Model       ┃ Avg TTFT ┃ Avg tok/s ┃ P50 Latency ┃ P99 Latency ┃ Requests ┃ Errors ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ ollama     │ qwen3.5:27b │   0.085s │      16.1 │      3.210s │      3.210s │        5 │      0 │
│ llama-swap │ qwen3.5-27b │   0.042s │      38.8 │      1.340s │      1.340s │        5 │      0 │
└────────────┴─────────────┴──────────┴───────────┴─────────────┴─────────────┴──────────┴────────┘
```

The Endpoint column is hidden when using a single endpoint.

## Configuration

### Environment Variables

- `LLM_HOST` — Default server URL (default: `http://localhost:11434`)
- `OPENAI_API_KEY` — Default API key (default: `none`)
- `NO_COLOR` — Disable colored output

### Global Options

`-v, --verbose` must come **before** the subcommand:

```bash
swallm -v benchmark -m llama3.1:8b    # correct
```

## Development

```bash
pip install -e ".[dev]"
pytest -v
```

## Requirements

- Python 3.10+
- An OpenAI-compatible LLM server (Ollama, llama-swap, vLLM, etc.)
- Dependencies: `openai`, `click`, `rich`, `httpx` (auto-installed)

## License

MIT
