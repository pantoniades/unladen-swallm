"""Summary statistics for benchmark results."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


def _percentile(data: List[float], p: float) -> Optional[float]:
    """Compute the *p*-th percentile (0–100) using linear interpolation.

    Returns ``None`` for empty *data*.
    """
    if not data:
        return None
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    # Map percentile to a 0-based index
    k = (p / 100.0) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_data[lo] + frac * (sorted_data[hi] - sorted_data[lo])


def compute_summary(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group *results* by ``(endpoint, model)`` and compute summary stats.

    Each result dict is expected to have at minimum:
      - ``endpoint`` (str) — endpoint label
      - ``model`` — object with a ``.name`` attribute, or a str
      - ``status`` (str) — ``"ok"`` for success
      - ``elapsed`` (float)
      - ``response`` (dict) — with optional ``time_to_first_token`` and
        ``tokens_per_sec`` keys (only for ``status == "ok"``)

    Returns a list of summary dicts sorted by ``(endpoint, model)``.
    """
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        endpoint = r.get("endpoint", "default")
        model_obj = r["model"]
        model_name = model_obj.name if hasattr(model_obj, "name") else str(model_obj)
        groups[(endpoint, model_name)].append(r)

    summaries: List[Dict[str, Any]] = []
    for (endpoint, model_name), group in sorted(groups.items()):
        elapsed_values: List[float] = []
        ttft_values: List[float] = []
        tps_values: List[float] = []
        errors = 0

        for r in group:
            if r["status"] != "ok":
                errors += 1
                continue
            elapsed_values.append(r["elapsed"])
            resp = r.get("response") or {}
            ttft = resp.get("time_to_first_token")
            if ttft is not None:
                ttft_values.append(ttft)
            tps = resp.get("tokens_per_sec")
            if tps is not None:
                tps_values.append(tps)

        summaries.append({
            "endpoint": endpoint,
            "model": model_name,
            "avg_ttft": sum(ttft_values) / len(ttft_values) if ttft_values else None,
            "avg_tps": sum(tps_values) / len(tps_values) if tps_values else None,
            "p50_elapsed": _percentile(elapsed_values, 50),
            "p99_elapsed": _percentile(elapsed_values, 99),
            "requests": len(group),
            "errors": errors,
        })

    return summaries
