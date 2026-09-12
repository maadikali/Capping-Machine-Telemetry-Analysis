"""
Anomaly detection over closure events: out-of-range torque and
statistically unusual heads/periods.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from capping_telemetry.config import Settings
from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C


def out_of_range_torque(events: pd.DataFrame, settings: Settings, exclude_zero_torque: bool = True) -> pd.DataFrame:
    """
    Closures with torque outside the expected operating range.

    exclude_zero_torque=True (default) drops exact-zero readings, since
    those mostly come from "No Load" cycles rather than genuine faults.
    Use zero_torque_summary() to quantify zero readings separately.
    """
    lo, hi = settings.analytics.torque_expected_range_nm
    mask = (events[C.torque] < lo) | (events[C.torque] > hi)
    if exclude_zero_torque:
        mask &= events[C.torque] != 0
    return events[mask][[C.timestamp, C.head_id, C.torque, C.status, C.success]].sort_values(C.timestamp)


def zero_torque_summary(events: pd.DataFrame) -> dict:
    """
    Aggregated view of zero-torque events: how many, on which heads,
    and what share of that head's closures they represent. Status code
    2 ("No Load") is the authoritative signal for this condition - this
    is mainly an independent cross-check against it (see
    torque_status_consistency_check).
    """
    if events.empty:
        return {"total_events": 0, "zero_torque_events": 0, "zero_torque_pct": None, "per_head": []}

    zero_mask = events[C.torque] == 0
    total = len(events)
    n_zero = int(zero_mask.sum())

    per_head = (
        events.assign(_is_zero=zero_mask)
        .groupby(C.head_id)
        .agg(total_closures=(C.torque, "size"), zero_torque_count=("_is_zero", "sum"))
    )
    per_head["zero_torque_pct"] = (100 * per_head["zero_torque_count"] / per_head["total_closures"]).round(1)
    per_head = per_head.reset_index().sort_values("zero_torque_pct", ascending=False)

    return {
        "total_events": total,
        "zero_torque_events": n_zero,
        "zero_torque_pct": round(100 * n_zero / total, 1),
        "per_head": per_head.to_dict(orient="records"),
        "note": (
            "Cross-check only - the authoritative 'No Load' signal is "
            "status code 2 (see overall_success_rate's 'no_load' field). "
            "Use torque_status_consistency_check to see where the two "
            "signals disagree."
        ),
    }


def statistical_outliers(events: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """Per-head z-score outliers on torque, independent of the fixed range check above."""
    subset = events[events[C.success]].copy()
    if subset.empty:
        return subset

    grouped = subset.groupby(C.head_id)[C.torque]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, pd.NA)
    subset["torque_zscore"] = ((subset[C.torque] - mean) / std).fillna(0.0)
    scored = subset
    return scored[scored["torque_zscore"].abs() >= z_threshold].sort_values(
        "torque_zscore", key=abs, ascending=False
    )


def head_with_most_failures(events: pd.DataFrame) -> dict:
    """Head with the most rejects (reject_signal=YES), not just
    'not successful' - which would wrongly include No Load closures."""
    failed = events[events[C.is_reject]]
    if failed.empty:
        return {"head_id": None, "failure_count": 0}
    counts = failed.groupby(C.head_id).size().sort_values(ascending=False)
    return {"head_id": counts.index[0], "failure_count": int(counts.iloc[0])}


def fault_code_breakdown(events: pd.DataFrame) -> Any:
    """Breakdown of every non-success, non-no-load status code seen in
    the data - the drill-down behind the single reject-rate number."""
    subset = events[~events[C.success] & ~events[C.is_no_load]]
    if subset.empty:
        return []
    breakdown = (
        subset.groupby([C.status, C.status_description, C.is_reject])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return breakdown.to_dict(orient="records")


def torque_status_consistency_check(events: pd.DataFrame) -> dict:
    """
    Cross-validates two independent signals that should agree: a closure
    marked 'No Load' (status 2) should have ~zero torque, and a closure
    with ~zero torque should generally be marked No Load. Rows where
    they disagree are worth a service engineer's attention - either a
    sensor/logging glitch or a mislabeled status.
    """
    no_load_nonzero_torque = events[events[C.is_no_load] & (events[C.torque] != 0)]
    zero_torque_not_no_load = events[(events[C.torque] == 0) & ~events[C.is_no_load]]
    return {
        "no_load_with_nonzero_torque_count": int(len(no_load_nonzero_torque)),
        "zero_torque_not_marked_no_load_count": int(len(zero_torque_not_no_load)),
        "note": (
            "Both counts should normally be near zero. Non-trivial counts "
            "suggest a sensor timing issue or that the No Load status "
            "isn't perfectly aligned with the torque reading on this line."
        ),
    }
