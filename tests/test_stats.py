"""Tests for unladen_swallm.stats."""
from unladen_swallm.stats import _percentile, compute_summary


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------

def test_percentile_empty():
    assert _percentile([], 50) is None


def test_percentile_single():
    assert _percentile([7.0], 50) == 7.0
    assert _percentile([7.0], 99) == 7.0


def test_percentile_two_values():
    assert _percentile([1.0, 3.0], 50) == 2.0


def test_percentile_p0_and_p100():
    data = [10.0, 20.0, 30.0]
    assert _percentile(data, 0) == 10.0
    assert _percentile(data, 100) == 30.0


def test_percentile_unsorted_input():
    assert _percentile([5.0, 1.0, 3.0], 50) == 3.0


# ---------------------------------------------------------------------------
# compute_summary
# ---------------------------------------------------------------------------

def _ok_result(endpoint, model_name, elapsed, ttft=None, tps=None):
    return {
        "endpoint": endpoint,
        "model": model_name,
        "status": "ok",
        "elapsed": elapsed,
        "response": {
            "time_to_first_token": ttft,
            "tokens_per_sec": tps,
        },
    }


def _error_result(endpoint, model_name):
    return {
        "endpoint": endpoint,
        "model": model_name,
        "status": "error",
        "elapsed": 0.0,
        "response": {},
    }


def test_compute_summary_single_group():
    results = [
        _ok_result("default", "m1", 1.0, ttft=0.1, tps=50.0),
        _ok_result("default", "m1", 3.0, ttft=0.3, tps=30.0),
    ]
    summary = compute_summary(results)
    assert len(summary) == 1
    s = summary[0]
    assert s["endpoint"] == "default"
    assert s["model"] == "m1"
    assert s["requests"] == 2
    assert s["errors"] == 0
    assert s["avg_ttft"] == pytest.approx(0.2)
    assert s["avg_tps"] == pytest.approx(40.0)
    assert s["p50_elapsed"] == pytest.approx(2.0)


def test_compute_summary_multi_endpoint():
    results = [
        _ok_result("ollama", "m1", 2.0, ttft=0.2, tps=40.0),
        _ok_result("vllm", "m1", 1.0, ttft=0.1, tps=80.0),
    ]
    summary = compute_summary(results)
    assert len(summary) == 2
    labels = [s["endpoint"] for s in summary]
    assert labels == ["ollama", "vllm"]


def test_compute_summary_with_errors():
    results = [
        _ok_result("default", "m1", 1.0, ttft=0.1, tps=50.0),
        _error_result("default", "m1"),
    ]
    summary = compute_summary(results)
    assert len(summary) == 1
    s = summary[0]
    assert s["requests"] == 2
    assert s["errors"] == 1
    assert s["avg_ttft"] == pytest.approx(0.1)


def test_compute_summary_missing_metrics():
    results = [
        _ok_result("default", "m1", 2.0, ttft=None, tps=None),
    ]
    summary = compute_summary(results)
    s = summary[0]
    assert s["avg_ttft"] is None
    assert s["avg_tps"] is None
    assert s["p50_elapsed"] == pytest.approx(2.0)


def test_compute_summary_model_with_name_attr():
    """Results with Model objects (having .name attr) should work."""
    from unladen_swallm.models import Model

    results = [
        {
            "endpoint": "ep",
            "model": Model(name="x"),
            "status": "ok",
            "elapsed": 1.0,
            "response": {"time_to_first_token": 0.05, "tokens_per_sec": 100.0},
        },
    ]
    summary = compute_summary(results)
    assert summary[0]["model"] == "x"


# Need pytest for approx
import pytest
