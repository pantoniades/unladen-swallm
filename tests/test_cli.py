import json

from click.testing import CliRunner
import pytest


def test_cli_list_models(monkeypatch):
    """The list-models command should run and print model names."""
    class FakeClient:
        def __init__(self, host=None, api_key="none"):
            pass

        async def list_models(self):
            from unladen_swallm.models import Model

            return [Model(name="one"), Model(name="two")]

        async def close(self):
            return None

    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient)

    from unladen_swallm.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["list-models"])
    assert result.exit_code == 0
    # Pretty format: multi-line blocks with names present
    assert "one" in result.output
    assert "size:" in result.output


def test_cli_compact_and_no_color(monkeypatch):
    class FakeClient2:
        def __init__(self, host=None, api_key="none"):
            pass

        async def list_models(self):
            from unladen_swallm.models import Model

            return [Model(name="alpha", size="1.4 GB", parameter_size="7B", quantization_level="q4_0", family="gemma", context_length=8192)]

        async def close(self):
            return None

    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient2)
    from unladen_swallm.cli import cli

    runner = CliRunner()
    # compact format
    result = runner.invoke(cli, ["list-models", "--print-format", "compact"])
    assert result.exit_code == 0
    assert "alpha" in result.output
    assert "size=1.4 GB" in result.output or "1.4 GB" in result.output

    # no-color should still produce readable text
    result2 = runner.invoke(cli, ["list-models", "--print-format", "compact", "--no-color"])
    assert result2.exit_code == 0
    assert "alpha" in result2.output


def test_cli_generate(monkeypatch):
    """The generate command should run and print the response dict."""
    class FakeClient:
        def __init__(self, host=None, api_key="none"):
            pass

        async def generate(self, model, prompt):
            return {"content": "ok", "elapsed": 0.1}

        async def close(self):
            return None

    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient)

    from unladen_swallm.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "mymodel", "hello"])
    assert result.exit_code == 0
    assert "content" in result.output
    assert "ok" in result.output


# ---------------------------------------------------------------------------
# _resolve_endpoints
# ---------------------------------------------------------------------------

def test_resolve_endpoints_from_host():
    from unladen_swallm.cli import _resolve_endpoints

    eps = _resolve_endpoints((), (), "http://myhost:1234", "mykey")
    assert len(eps) == 1
    assert eps[0].label == "default"
    assert eps[0].url == "http://myhost:1234"
    assert eps[0].api_key == "mykey"


def test_resolve_endpoints_default_host():
    from unladen_swallm.cli import _resolve_endpoints

    eps = _resolve_endpoints((), (), None, "none")
    assert eps[0].url == "http://localhost:11434"


def test_resolve_endpoints_from_endpoint_flags():
    from unladen_swallm.cli import _resolve_endpoints

    eps = _resolve_endpoints(
        ("ollama=http://localhost:11434", "vllm=http://localhost:8000"),
        ("vllm=sk-123",),
        "http://ignored",
        "fallback-key",
    )
    assert len(eps) == 2
    assert eps[0].label == "ollama"
    assert eps[0].api_key == "fallback-key"
    assert eps[1].label == "vllm"
    assert eps[1].api_key == "sk-123"


def test_resolve_endpoints_duplicate_label():
    from unladen_swallm.cli import _resolve_endpoints

    with pytest.raises(Exception, match="Duplicate"):
        _resolve_endpoints(("a=http://x", "a=http://y"), (), None, "none")


def test_resolve_endpoints_bad_format():
    from unladen_swallm.cli import _resolve_endpoints

    with pytest.raises(Exception):
        _resolve_endpoints(("no-equals-sign",), (), None, "none")


# ---------------------------------------------------------------------------
# Benchmark: backward-compatible --host without --endpoint
# ---------------------------------------------------------------------------

def _make_fake_client_class(models, generate_resp=None):
    """Return a FakeClient class that records which hosts it was created for."""
    hosts_seen = []
    if generate_resp is None:
        generate_resp = {
            "content": "hello",
            "elapsed": 1.0,
            "time_to_first_token": 0.1,
            "generation_time": 0.9,
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "tokens_per_sec": 11.1,
            "elapsed_tokens_per_sec": 10.0,
            "token_counts_available": True,
        }

    class FakeClient:
        def __init__(self, host=None, api_key="none"):
            hosts_seen.append(host)
            self._base_url = host + "/v1" if host else "http://localhost:11434/v1"

        async def list_models(self):
            return list(models)

        async def generate(self, model, prompt):
            return dict(generate_resp)

        async def close(self):
            pass

    FakeClient._hosts_seen = hosts_seen
    return FakeClient


def test_benchmark_backward_compat_host(monkeypatch):
    """--host without --endpoint should produce 'default' endpoint results."""
    from unladen_swallm.models import Model
    from unladen_swallm.cli import cli

    FakeClient = _make_fake_client_class([Model(name="m1")])
    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "-H", "http://myhost:1234", "-m", "m1", "--prompt", "hi", "-f", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["results"][0]["endpoint"] == "default"
    assert "summary" in data


def test_benchmark_multi_endpoint_json(monkeypatch):
    """Multiple --endpoint flags should produce sequential results tagged by label."""
    from unladen_swallm.models import Model
    from unladen_swallm.cli import cli

    FakeClient = _make_fake_client_class([Model(name="m1")])
    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "benchmark",
        "-E", "ollama=http://host1:11434",
        "-E", "vllm=http://host2:8000",
        "-m", "m1",
        "--prompt", "hi",
        "-f", "json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)

    # Should have results from both endpoints
    endpoints_in_results = {r["endpoint"] for r in data["results"]}
    assert endpoints_in_results == {"ollama", "vllm"}

    # Config should list endpoints
    assert len(data["config"]["endpoints"]) == 2

    # Summary should have entries for both
    summary_endpoints = {s["endpoint"] for s in data["summary"]}
    assert summary_endpoints == {"ollama", "vllm"}

    # Client should have been created for both hosts sequentially
    assert FakeClient._hosts_seen == ["http://host1:11434", "http://host2:8000"]


def test_benchmark_multi_endpoint_missing_model(monkeypatch):
    """When a model is missing on one endpoint, warn and skip that endpoint."""
    from unladen_swallm.models import Model
    from unladen_swallm.cli import cli

    call_count = 0

    class FakeClient:
        def __init__(self, host=None, api_key="none"):
            nonlocal call_count
            call_count += 1
            self._call = call_count
            self._base_url = host + "/v1" if host else "http://localhost:11434/v1"

        async def list_models(self):
            if self._call == 1:
                return [Model(name="m1")]
            # Second endpoint has no matching models
            return [Model(name="other")]

        async def generate(self, model, prompt):
            return {
                "content": "ok", "elapsed": 1.0, "time_to_first_token": 0.1,
                "generation_time": 0.9, "prompt_tokens": 5, "completion_tokens": 10,
                "tokens_per_sec": 11.0, "elapsed_tokens_per_sec": 10.0,
                "token_counts_available": True,
            }

        async def close(self):
            pass

    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "benchmark",
        "-E", "a=http://h1:1", "-E", "b=http://h2:2",
        "-m", "m1", "--prompt", "hi", "-f", "json",
    ])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    # Only endpoint 'a' should have results
    assert all(r["endpoint"] == "a" for r in data["results"])


def test_benchmark_text_summary_table(monkeypatch):
    """Text output should include a summary table."""
    from unladen_swallm.models import Model
    from unladen_swallm.cli import cli

    FakeClient = _make_fake_client_class([Model(name="m1")])
    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark", "-m", "m1", "--prompt", "hi", "-f", "text"])
    assert result.exit_code == 0, result.output
    assert "Summary" in result.output
    assert "Avg TTFT" in result.output
    assert "m1" in result.output


def test_benchmark_endpoint_key(monkeypatch):
    """--endpoint-key should override the global --api-key for that endpoint."""
    from unladen_swallm.models import Model
    from unladen_swallm.cli import cli

    keys_seen = []

    class FakeClient:
        def __init__(self, host=None, api_key="none"):
            keys_seen.append(api_key)
            self._base_url = host + "/v1" if host else "http://localhost:11434/v1"

        async def list_models(self):
            return [Model(name="m1")]

        async def generate(self, model, prompt):
            return {
                "content": "ok", "elapsed": 1.0, "time_to_first_token": 0.1,
                "generation_time": 0.9, "prompt_tokens": 5, "completion_tokens": 10,
                "tokens_per_sec": 11.0, "elapsed_tokens_per_sec": 10.0,
                "token_counts_available": True,
            }

        async def close(self):
            pass

    monkeypatch.setattr("unladen_swallm.cli.OpenAICompatibleClient", FakeClient)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "benchmark",
        "-E", "local=http://h1:1",
        "-E", "cloud=http://h2:2",
        "--endpoint-key", "cloud=sk-secret",
        "--api-key", "fallback",
        "-m", "m1", "--prompt", "hi", "-f", "json",
    ])
    assert result.exit_code == 0, result.output
    assert keys_seen == ["fallback", "sk-secret"]
