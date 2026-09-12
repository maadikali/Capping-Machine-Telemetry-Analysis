"""
Correlation checks across heads / signals, e.g. "performance of head 1 vs
head 5" and "does higher torque correlate with higher success rate?".
"""
from __future__ import annotations

import pandas as pd

from capping_telemetry.config import Settings
from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C


def torque_correlation_between_heads(events: pd.DataFrame, head_a: str, head_b: str, settings: Settings) -> dict:
    """
    Correlates torque time series of two heads by aligning on closure
    index (i-th closure of head_a vs i-th closure of head_b), since heads
    don't share identical timestamps. No Load closures are excluded first
    (their torque is ~0 by definition and would swamp the correlation).
    """
    min_events = settings.analytics.correlation_min_events
    attempted = events[~events[C.is_no_load]]
    a = attempted[attempted[C.head_id] == head_a].sort_values(C.timestamp)[C.torque].reset_index(drop=True)
    b = attempted[attempted[C.head_id] == head_b].sort_values(C.timestamp)[C.torque].reset_index(drop=True)

    n = min(len(a), len(b))
    if n < min_events:
        return {
            "head_a": head_a, "head_b": head_b, "n_pairs": n,
            "correlation": None,
            "note": f"Fewer than {min_events} paired events - correlation not reliable",
        }

    corr = float(a.iloc[:n].corr(b.iloc[:n]))
    return {"head_a": head_a, "head_b": head_b, "n_pairs": n, "correlation": round(corr, 3)}


def torque_vs_success_correlation(events: pd.DataFrame) -> dict:
    """Point-biserial-style correlation between torque and binary success flag,
    restricted to attempted (non-No-Load) closures."""
    attempted = events[~events[C.is_no_load]]
    if attempted.empty or attempted[C.torque].std() == 0:
        return {"correlation": None, "n_events": len(attempted)}
    corr = float(attempted[C.torque].corr(attempted[C.success].astype(float)))
    return {"correlation": round(corr, 3), "n_events": len(attempted)}


def rank_heads_by_deviation(events: pd.DataFrame) -> pd.DataFrame:
    """Ranks heads by how far their mean torque and success rate sit
    from the fleet-wide average. Computed over attempted (non-No-Load)
    closures only."""
    attempted = events[~events[C.is_no_load]]
    if attempted.empty:
        return pd.DataFrame(columns=["head_id", "mean_torque_nm", "success_rate_pct", "deviation_score"])

    per_head = attempted.groupby(C.head_id).agg(
        mean_torque_nm=(C.torque, "mean"),
        success_rate_pct=(C.success, "mean"),
    )
    per_head["success_rate_pct"] *= 100
    fleet_torque = per_head["mean_torque_nm"].mean()
    fleet_success = per_head["success_rate_pct"].mean()

    per_head["deviation_score"] = (
        (per_head["mean_torque_nm"] - fleet_torque).abs() / (per_head["mean_torque_nm"].std() or 1)
        + (per_head["success_rate_pct"] - fleet_success).abs() / (per_head["success_rate_pct"].std() or 1)
    )
    return per_head.round(2).reset_index().sort_values("deviation_score", ascending=False)
