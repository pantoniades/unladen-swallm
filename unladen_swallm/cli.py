"""Simple Click CLI to benchmark OpenAI-compatible LLM APIs.

Provides a global --verbose / -v flag to enable DEBUG logging.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from io import StringIO
from random import shuffle
from typing import Optional, Sequence, List

import httpx

from .models import Endpoint, Model
from .stats import compute_summary

import click

from .client import OpenAICompatibleClient
from rich.console import Console
from rich.table import Table
from asyncio import TimeoutError

logger = logging.getLogger("unladen_swallm")

_CONSOLE = Console()


# Default prompt used by the benchmark command when none supplied
DEFAULT_PROMPT = (
    "Briefly explain plate tectonics in one paragraph suitable for a general audience, "
    "highlighting causes and why it matters for Earth's geography."
)


def _format_size(size) -> str:
    """Format size in bytes to human-readable string (MB/GB)."""
    if size is None:
        return "-"

    # If already a string, return as-is (might already be formatted)
    if isinstance(size, str):
        return size

    # Convert bytes to appropriate unit
    try:
        size_bytes = float(size)
        if size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.1f} MB"
        else:
            return f"{size_bytes / (1024**3):.2f} GB"
    except (ValueError, TypeError):
        return str(size)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """swallm - Benchmark OpenAI-compatible LLM APIs"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Suppress HTTP request logs from httpx/httpcore
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info(f"Starting swallm at {now}, verbose={verbose}")


async def _unload_models(endpoint_url: str, model_names: List[str]) -> None:
    """Best-effort unload of models from an endpoint to free GPU memory.

    Tries known provider-specific unload APIs and logs results.
    Failures are silently ignored — worst case we fall back to the delay.
    """
    base = endpoint_url.rstrip("/")
    async with httpx.AsyncClient(timeout=10.0) as http:
        # Ollama: POST /api/generate with keep_alive=0 per model
        for name in model_names:
            try:
                r = await http.post(
                    f"{base}/api/generate",
                    json={"model": name, "keep_alive": 0},
                )
                logger.info(f"Ollama unload {name}: {r.status_code}")
            except Exception as exc:
                logger.debug(f"Ollama unload {name} failed (not Ollama?): {exc}")

        # llama-swap: POST /unload (unloads all models)
        try:
            r = await http.post(f"{base}/unload")
            logger.info(f"llama-swap unload: {r.status_code}")
        except Exception as exc:
            logger.debug(f"llama-swap /unload failed (not llama-swap?): {exc}")


def _print_models(models: Sequence[Model], fmt: str = "pretty", color: bool = True) -> None:
    """Print models in either 'pretty' (multi-line) or 'compact' (single-line) format.

    color: whether to use color/ANSI Rich formatting (respects NO_COLOR env var).
    """
    console = Console(no_color=not color, force_terminal=color)

    if fmt == "pretty":
        for m in models:
            console.print(f"[bold green]{m.name}[/bold green]")
            console.print(f"  [magenta]size:[/magenta] {_format_size(m.size)}")
            console.print(f"  [yellow]params:[/yellow] {m.parameter_size or '-'}  [blue]quant:[/blue] {m.quantization_level or '-'}")
            console.print(f"  [white]family:[/white] {m.family or '-'}  [cyan]ctx:[/cyan] {m.context_length or '-'}")
            console.print("")

    else:  # compact
        for i, m in enumerate(models, start=1):
            console.print(f"[cyan]{i}[/cyan] [green]{m.name}[/green]  [magenta]{_format_size(m.size)}[/magenta]  [yellow]{m.parameter_size or '-'}[/yellow]  [blue]{m.quantization_level or '-'}[/blue]  [white]{m.family or '-'}[/white] [cyan]{m.context_length or '-'}[/cyan]")


def _resolve_endpoints(
    endpoint: tuple[str, ...],
    endpoint_key: tuple[str, ...],
    host: Optional[str],
    api_key: str,
) -> List[Endpoint]:
    """Build a list of :class:`Endpoint` from CLI options.

    If ``--endpoint`` is provided, ``--host`` is ignored.
    Otherwise, ``--host`` (or the default) is used as a single endpoint
    labelled ``"default"``.
    """
    if endpoint:
        key_map: dict[str, str] = {}
        for entry in endpoint_key:
            if "=" not in entry:
                raise click.BadParameter(
                    f"Invalid --endpoint-key format: {entry!r} (expected label=key)",
                    param_hint="--endpoint-key",
                )
            lbl, _, key = entry.partition("=")
            key_map[lbl] = key

        endpoints: List[Endpoint] = []
        seen_labels: set[str] = set()
        for entry in endpoint:
            if "=" not in entry:
                raise click.BadParameter(
                    f"Invalid --endpoint format: {entry!r} (expected label=url)",
                    param_hint="--endpoint",
                )
            label, _, url = entry.partition("=")
            if label in seen_labels:
                raise click.BadParameter(
                    f"Duplicate endpoint label: {label!r}",
                    param_hint="--endpoint",
                )
            seen_labels.add(label)
            endpoints.append(Endpoint(
                label=label,
                url=url,
                api_key=key_map.get(label, api_key),
            ))
        return endpoints

    resolved_host = host or "http://localhost:11434"
    return [Endpoint(label="default", url=resolved_host, api_key=api_key)]


@cli.command("list-models")
@click.option("-H", "--host", envvar="LLM_HOST", help="LLM API host URL", default=None)
@click.option("--api-key", envvar="OPENAI_API_KEY", default="none", help="API key (use OPENAI_API_KEY env var or 'none' for local providers)")
@click.option("-f", "--print-format", type=click.Choice(["pretty", "compact"]), default="pretty", help="Output format")
@click.option("-n", "--no-color", is_flag=True, help="Disable colored output")
def list_models(host: Optional[str], api_key: str, print_format: str, no_color: bool) -> None:
    """List models from the LLM API server."""

    async def _main():
        client = OpenAICompatibleClient(host=host or "http://localhost:11434", api_key=api_key)
        try:
            models = await client.list_models()
            # Determine color availability: user flag overrides Rich
            color_enabled = (not no_color) and ("NO_COLOR" not in __import__("os").environ)
            _print_models(models, fmt=print_format, color=color_enabled)
        except Exception as exc:  # noqa: BLE001 - surface to user and log
            logger.exception("list-models failed")
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        finally:
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing client", exc_info=True)

    asyncio.run(_main())


@cli.command("generate")
@click.argument("model")
@click.argument("prompt")
@click.option("-H", "--host", envvar="LLM_HOST", help="LLM API host URL", default=None)
@click.option("--api-key", envvar="OPENAI_API_KEY", default="none", help="API key (use OPENAI_API_KEY env var or 'none' for local providers)")
def generate(model: str, prompt: str, host: Optional[str], api_key: str) -> None:
    """Generate from MODEL using PROMPT and print response."""

    async def _main():
        client = OpenAICompatibleClient(host=host or "http://localhost:11434", api_key=api_key)
        try:
            resp = await client.generate(model, prompt)
            click.echo(json.dumps(resp, default=str, indent=2))
        except Exception as exc:
            logger.exception("generate failed")
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        finally:
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing client", exc_info=True)

    asyncio.run(_main())


@cli.command("benchmark")
@click.option("--prompt", "prompt_text", help="Single prompt text to send to each model")
@click.option("--prompts-file", "-P", "prompts_file", type=click.Path(exists=True), help="File with multiple prompts (one per line)")
@click.option("-c", "--concurrent", "concurrency", type=int, default=1, help="Number of concurrent requests")
@click.option("--concurrency-mode", type=click.Choice(["global", "per-model"]), default="per-model", help="Concurrency mode: global (all tasks) or per-model (per model)")
@click.option("-t", "--timeout", "timeout", type=float, default=120.0, help="Per-request timeout in seconds")
@click.option("--endpoint-delay", type=float, default=5.0, help="Seconds to wait between endpoints for GPU cooldown (0 to disable)")
@click.option('-m', '--model', multiple=True, help='Model(s) to benchmark')
@click.option("-H", "--host", envvar="LLM_HOST", help="LLM API host URL", default=None)
@click.option("-E", "--endpoint", "endpoint", multiple=True, help="Endpoint(s) as label=url (repeatable, e.g. --endpoint ollama=http://localhost:11434)")
@click.option("--endpoint-key", "endpoint_key", multiple=True, help="Per-endpoint API key as label=key (repeatable)")
@click.option("--api-key", envvar="OPENAI_API_KEY", default="none", help="API key (use OPENAI_API_KEY env var or 'none' for local providers)")
@click.option("-r", "--response", "response", help="Whether to include full response in output", default=False, is_flag=True)
@click.option("-f", "--format", "output_format", type=click.Choice(["json", "text"]), default="json", help="Output format (default: json)")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Write output to file instead of stdout")
@click.option("--exclude-errors", is_flag=True, help="Exclude failed results from output")
@click.option("--errors-only", is_flag=True, help="Only include failed results in output")
def benchmark(prompt_text: Optional[str], prompts_file: Optional[str], concurrency: int, concurrency_mode: str, timeout: float, endpoint_delay: float, model: List[str], host: Optional[str], endpoint: tuple[str, ...], endpoint_key: tuple[str, ...], api_key: str, response: bool, output_format: str, output_file: Optional[str], exclude_errors: bool, errors_only: bool) -> None:
    """Send prompts to models and report API stats plus timing.

    Supports single or multiple prompts and endpoints. Output in JSON (default) or text format.
    """

    # Build list of prompts
    prompts = []
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as fh:
            content = fh.read()
            # Parse prompts: support triple-quote multi-line and single-line formats
            parts = content.split('"""')
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    # Inside triple quotes - treat as single multi-line prompt
                    stripped = part.strip()
                    if stripped:
                        prompts.append(stripped)
                else:
                    # Outside triple quotes - each non-empty line is a prompt
                    for line in part.split('\n'):
                        stripped = line.strip()
                        if stripped:
                            prompts.append(stripped)
    elif prompt_text:
        prompts = [prompt_text]
    else:
        # Default prompt if nothing specified
        prompts = [DEFAULT_PROMPT]

    endpoints = _resolve_endpoints(endpoint, endpoint_key, host, api_key)
    multi_endpoint = len(endpoints) > 1

    async def _main():
        all_results: List[dict] = []

        for ep in endpoints:
            client = OpenAICompatibleClient(host=ep.url, api_key=ep.api_key)
            try:
                all_models = await client.list_models()
                model_names = [m.name for m in all_models]
                if model:
                    selected_models = [m for m in all_models if m.name in model]
                    missing = [item for item in model if item not in model_names]
                    if missing:
                        if selected_models:
                            logger.warning(f"Endpoint {ep.label}: models not found: {missing} (available: {model_names})")
                        else:
                            logger.warning(f"Endpoint {ep.label}: no matching models for {list(model)} (available: {model_names}), skipping")
                            continue
                else:
                    selected_models = all_models

                # Shuffle order we test the models to avoid any systematic bias
                shuffle(selected_models)

                # Simple worker that calls generate and times it
                async def run_for_model_and_prompt(m: Model, prompt: str) -> dict:
                    error_info = None
                    try:
                        # Respect timeout per request
                        coro = client.generate(m.name, prompt)
                        resp = await asyncio.wait_for(coro, timeout=timeout)
                        status = "ok"
                    except TimeoutError:
                        resp = {"error": "Generate timed out", "elapsed": timeout}
                        status = "timeout"
                        error_info = {
                            "type": "TimeoutError",
                            "message": f"Request timed out after {timeout}s",
                            "model": m.name,
                            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
                        }
                        logger.error(f"Timeout generating for model {m.name} after {timeout}s")
                    except Exception as exc:
                        resp = {"error": str(exc), "elapsed": 0.0}
                        status = "error"
                        error_info = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "model": m.name,
                            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt
                        }
                        logger.error(f"Error generating for model {m.name}: {type(exc).__name__}: {exc}")
                    return {"endpoint": ep.label, "model": m, "prompt": prompt, "status": status, "elapsed": resp.get("elapsed", 0.0), "response": resp, "error_info": error_info}

                # Execute with appropriate concurrency mode
                results: List[dict] = []
                total_tasks = len(selected_models) * len(prompts)

                if concurrency_mode == "global":
                    # Global concurrency: limit total concurrent tasks
                    tasks_to_run = [(m, p) for m in selected_models for p in prompts]
                    logger.info(f"[{ep.label}] Starting benchmark: {len(selected_models)} models x {len(prompts)} prompts = {total_tasks} total tasks (global concurrency={concurrency})")

                    if concurrency <= 1:
                        for idx, (m, p) in enumerate(tasks_to_run, 1):
                            r = await run_for_model_and_prompt(m, p)
                            results.append(r)
                            if idx % 5 == 0:
                                logger.info(f"Progress: {idx}/{total_tasks} tasks completed")
                    else:
                        sem = asyncio.Semaphore(concurrency)

                        async def sem_task(m: Model, p: str):
                            async with sem:
                                return await run_for_model_and_prompt(m, p)

                        tasks = [asyncio.create_task(sem_task(m, p)) for m, p in tasks_to_run]
                        completed = 0
                        for t in asyncio.as_completed(tasks):
                            results.append(await t)
                            completed += 1
                            if completed % 5 == 0:
                                logger.info(f"Progress: {completed}/{total_tasks} tasks completed")
                else:
                    # Per-model concurrency: run prompts concurrently within each model
                    logger.info(f"[{ep.label}] Starting benchmark: {len(selected_models)} models x {len(prompts)} prompts = {total_tasks} total tasks (per-model concurrency={concurrency})")
                    for model_idx, m in enumerate(selected_models, 1):
                        logger.info(f"Model {model_idx}/{len(selected_models)}: {m.name} ({len(prompts)} prompts)")
                        if concurrency <= 1:
                            # Sequential for this model
                            for prompt_idx, p in enumerate(prompts, 1):
                                r = await run_for_model_and_prompt(m, p)
                                results.append(r)
                                if prompt_idx % 5 == 0:
                                    logger.info(f"  Progress: {prompt_idx}/{len(prompts)} prompts completed for {m.name}")
                        else:
                            # Concurrent prompts for this model
                            sem = asyncio.Semaphore(concurrency)

                            async def sem_task(model_arg: Model, prompt_arg: str):
                                async with sem:
                                    return await run_for_model_and_prompt(model_arg, prompt_arg)

                            tasks = [asyncio.create_task(sem_task(m, p)) for p in prompts]
                            completed = 0
                            for t in asyncio.as_completed(tasks):
                                results.append(await t)
                                completed += 1
                                if completed % 5 == 0:
                                    logger.info(f"  Progress: {completed}/{len(prompts)} prompts completed for {m.name}")

                all_results.extend(results)
                models_used = [m.name for m in selected_models]
            except Exception as exc:
                logger.exception(f"benchmark failed for endpoint {ep.label}")
                click.echo(f"Error on endpoint {ep.label}: {exc}", err=True)
                models_used = []
                if not multi_endpoint:
                    sys.exit(1)
            finally:
                try:
                    await client.close()
                except Exception:
                    logger.debug("Error closing client", exc_info=True)

            # Unload models and wait between endpoints to free GPU memory
            if multi_endpoint and ep is not endpoints[-1]:
                if models_used:
                    await _unload_models(ep.url, models_used)
                if endpoint_delay > 0:
                    logger.info(f"Waiting {endpoint_delay}s between endpoints for GPU cooldown...")
                    await asyncio.sleep(endpoint_delay)

        if not all_results:
            click.echo("No results collected.", err=True)
            sys.exit(1)

        # Filter results based on error flags
        filtered_results = all_results
        if exclude_errors:
            filtered_results = [r for r in all_results if r["status"] == "ok"]
        elif errors_only:
            filtered_results = [r for r in all_results if r["status"] != "ok"]

        # Compute summary statistics
        summary = compute_summary(all_results)

        # Format output
        if output_format == "json":
            output_data = {
                "config": {
                    "endpoints": [ep.to_dict() for ep in endpoints],
                    "prompts": prompts,
                    "models": list(model) if model else None,
                    "concurrency": concurrency,
                    "concurrency_mode": concurrency_mode,
                    "timeout": timeout,
                },
                "results": [],
                "summary": summary,
            }

            for r in filtered_results:
                current_model = r["model"]
                result_entry = {
                    "endpoint": r["endpoint"],
                    "model": current_model.name,
                    "prompt": r["prompt"],
                    "status": r["status"],
                    "elapsed": r["elapsed"],
                }

                if r["status"] == "ok":
                    resp = r["response"]
                    result_entry["metrics"] = {
                        "elapsed": resp.get("elapsed"),
                        "time_to_first_token": resp.get("time_to_first_token"),
                        "generation_time": resp.get("generation_time"),
                        "prompt_tokens": resp.get("prompt_tokens"),
                        "completion_tokens": resp.get("completion_tokens"),
                        "tokens_per_sec": resp.get("tokens_per_sec"),
                        "elapsed_tokens_per_sec": resp.get("elapsed_tokens_per_sec"),
                        "token_counts_available": resp.get("token_counts_available", False),
                    }

                    if response:
                        result_entry["response"] = resp.get("content", "")
                else:
                    # Include detailed error info
                    if r["error_info"]:
                        result_entry["error"] = r["error_info"]
                    else:
                        result_entry["error"] = r["response"].get("error", "Unknown error")

                output_data["results"].append(result_entry)

            output_str = json.dumps(output_data, default=str, indent=2)
        else:  # text format
            output_lines = []

            for r in filtered_results:
                current_model = r["model"]
                prefix = f"[{r['endpoint']}] " if multi_endpoint else ""
                if r["status"] != "ok":
                    # Color-coded error output
                    if r["status"] == "timeout":
                        error_line = f"[yellow]{prefix}⏱ {current_model.name}[/yellow] - timeout after {r['elapsed']:.3f}s"
                    else:
                        error_msg = r["error_info"]["message"] if r["error_info"] else r['response'].get('error', 'Unknown error')
                        error_line = f"[red]{prefix}✗ {current_model.name}[/red] - {error_msg} (after {r['elapsed']:.3f}s)"
                    output_lines.append(error_line)
                    output_lines.append("")
                    continue

                # Success output
                resp = r["response"]
                token_counts_available = resp.get("token_counts_available", False)
                output_lines.append(f"[bold green]{prefix}✓ {current_model.name}[/bold green]")
                output_lines.append(f"  prompt: {r['prompt'][:80]}{'...' if len(r['prompt']) > 80 else ''}")
                if current_model.size:
                    output_lines.append(f"  size: {_format_size(current_model.size)}")
                output_lines.append(f"  elapsed: {r['elapsed']:.3f}s")
                ttft = resp.get("time_to_first_token")
                output_lines.append(f"  time_to_first_token: {f'{ttft:.3f}s' if ttft is not None else '-'}")
                gen_time = resp.get("generation_time")
                output_lines.append(f"  generation_time: {f'{gen_time:.3f}s' if gen_time is not None else '-'}")
                if token_counts_available:
                    output_lines.append(f"  completion_tokens: {resp.get('completion_tokens', '-')}")
                    tps = resp.get("tokens_per_sec")
                    output_lines.append(f"  tokens_per_sec: {f'{tps:.2f}' if tps is not None else '-'}")
                else:
                    output_lines.append(f"  completion_tokens: n/a")
                    output_lines.append(f"  tokens_per_sec: n/a")
                if response:
                    output_lines.append(f"  response: {resp.get('content', '')}")
                output_lines.append("")

            # Render per-result lines with Rich for color support
            string_io = StringIO()
            temp_console = Console(file=string_io, force_terminal=True)
            for line in output_lines:
                temp_console.print(line)

            # Append summary table
            table = Table(title="Summary")
            if multi_endpoint:
                table.add_column("Endpoint")
            table.add_column("Model")
            table.add_column("Avg TTFT", justify="right")
            table.add_column("Avg tok/s", justify="right")
            table.add_column("P50 Latency", justify="right")
            table.add_column("P99 Latency", justify="right")
            table.add_column("Requests", justify="right")
            table.add_column("Errors", justify="right")

            for s in summary:
                row: List[str] = []
                if multi_endpoint:
                    row.append(s["endpoint"])
                row.append(s["model"])
                row.append(f"{s['avg_ttft']:.3f}s" if s["avg_ttft"] is not None else "-")
                row.append(f"{s['avg_tps']:.1f}" if s["avg_tps"] is not None else "-")
                row.append(f"{s['p50_elapsed']:.3f}s" if s["p50_elapsed"] is not None else "-")
                row.append(f"{s['p99_elapsed']:.3f}s" if s["p99_elapsed"] is not None else "-")
                row.append(str(s["requests"]))
                row.append(str(s["errors"]))
                table.add_row(*row)

            temp_console.print(table)
            output_str = string_io.getvalue()

        # Write to file or stdout
        if output_file:
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write(output_str)
            click.echo(f"Results written to {output_file}")
        else:
            click.echo(output_str)

    asyncio.run(_main())


if __name__ == "__main__":
    cli()
