"""
Trend analysis: moving averages and drift detection on torque values,
plus time-based success-rate breakdowns.
"""
from __future__ import annotations

import pandas as pd

from capping_telemetry.config import Settings
from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C


def torque_moving_average(events: pd.DataFrame, settings: Settings, head_id: str | None = None) -> pd.DataFrame:
    """Rolling mean of torque over successive closure events, used to spot
    slow drift that a single overall average would hide."""
    subset = events[events[C.success]]
    if head_id:
        subset = subset[subset[C.head_id] == head_id]
    subset = subset.sort_values(C.timestamp)
    if subset.empty:
        return pd.DataFrame(columns=[C.timestamp, C.head_id, "torque_moving_avg"])

    window = settings.analytics.moving_average_window
    subset = subset.copy()
    subset["torque_moving_avg"] = (
        subset.groupby(C.head_id)[C.torque]
        .transform(lambda s: s.rolling(window=min(window, len(s)), min_periods=1).mean())
    )
    return subset[[C.timestamp, C.head_id, "torque_moving_avg"]]


def detect_drift(events: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Flags heads whose recent torque mean has drifted more than
    `drift_zscore_threshold` standard deviations from their own historical
    baseline (first half of the dataset vs. second half).
    """
    results = []
    threshold = settings.analytics.drift_zscore_threshold
    subset = events[events[C.success]].sort_values(C.timestamp)

    for head_id, grp in subset.groupby(C.head_id):
        if len(grp) < 10:
            continue
        midpoint = len(grp) // 2
        baseline = grp[C.torque].iloc[:midpoint]
        recent = grp[C.torque].iloc[midpoint:]
        if baseline.std() == 0 or baseline.empty or recent.empty:
            continue
        z = (recent.mean() - baseline.mean()) / baseline.std()
        results.append({
            "head_id": head_id,
            "baseline_mean_nm": round(float(baseline.mean()), 2),
            "recent_mean_nm": round(float(recent.mean()), 2),
            "zscore": round(float(z), 2),
            "drift_detected": bool(abs(z) >= threshold),
        })

    return pd.DataFrame(results).sort_values("zscore", key=abs, ascending=False)


def success_rate_over_time(events: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """Success rate per time period (daily by default). No Load closures
    are excluded from the denominator, same as kpi.overall_success_rate."""
    if events.empty:
        return pd.DataFrame(columns=["period", "attempted", "successful", "rejected", "success_rate_pct"])
    ts = pd.to_datetime(events[C.timestamp])
    attempted = events[~events[C.is_no_load]].assign(_period=ts[~events[C.is_no_load]].dt.floor(freq))
    grouped = attempted.groupby("_period").agg(
        attempted=(C.success, "size"), successful=(C.success, "sum"), rejected=(C.is_reject, "sum")
    )
    grouped["success_rate_pct"] = (100 * grouped["successful"] / grouped["attempted"]).round(1)
    return grouped.reset_index().rename(columns={"_period": "period"})


def success_rate_by_hour_of_day(events: pd.DataFrame) -> pd.DataFrame:
    """Success rate grouped by hour-of-day (0-23), pooling across every
    day - shows whether failures cluster at certain times, a pattern
    success_rate_over_time (grouped by calendar period) can't reveal."""
    if events.empty:
        return pd.DataFrame(columns=["hour", "attempted", "successful", "rejected", "success_rate_pct"])
    ts = pd.to_datetime(events[C.timestamp])
    attempted = events[~events[C.is_no_load]].assign(_hour=ts[~events[C.is_no_load]].dt.hour)
    grouped = attempted.groupby("_hour").agg(
        attempted=(C.success, "size"), successful=(C.success, "sum"), rejected=(C.is_reject, "sum")
    )
    grouped["success_rate_pct"] = (100 * grouped["successful"] / grouped["attempted"]).round(1)
    return grouped.reset_index().rename(columns={"_hour": "hour"}).sort_values("hour")


def capping_speed_summary(events: pd.DataFrame) -> dict:
    """
    Overall capping speed (pieces/hour), using the incremental/expanding
    average from closure_detection.capping_speed_pieces_per_hour. Counts
    every closure event across all heads and statuses (including No
    Load), since speed measures machine cycle rate, not quality.

    Returns the final (most converged) estimate plus the first estimate,
    so it's clear whether the running average had stabilized. For a
    day-by-day trend instead of one number, use capping_speed_over_time.
    """
    from capping_telemetry.ingestion.closure_detection import capping_speed_pieces_per_hour

    if events.empty:
        return {"overall_pieces_per_hour": None, "n_events": 0}

    speed = capping_speed_pieces_per_hour(events).dropna()
    if speed.empty:
        return {"overall_pieces_per_hour": None, "n_events": len(events)}

    return {
        "overall_pieces_per_hour": round(float(speed.iloc[-1]), 1),
        "first_estimate_pieces_per_hour": round(float(speed.iloc[0]), 1),
        "n_events": int(len(events)),
    }


def capping_speed_over_time(events: pd.DataFrame, freq: str = "1D") -> pd.DataFrame:
    """
    Capping speed (pieces/hour) broken down by time period, e.g. daily -
    unlike capping_speed_summary's single running-average figure, this
    shows whether throughput changed over the scoped period. Speed for
    each period = (events in that period) / (period's actual wall-clock
    duration in hours), across all heads combined, all statuses included
    (see capping_speed_summary for why No Load cycles are counted here).

    The first and last period in the dataset are usually partial (the
    data doesn't start/end exactly on a period boundary), so their
    duration is measured from the actual min/max event timestamp inside
    that period rather than assumed to be a full `freq`-length window.
    Using the fixed full-period length there would understate throughput
    for those two rows - e.g. a day with events only from 06:00-23:59
    would look like a ~25% slower day than an identical full 24h day,
    when nothing about the machine's actual rate changed.
    """
    if events.empty:
        return pd.DataFrame(columns=["period", "n_events", "pieces_per_hour"])

    ts = pd.to_datetime(events[C.timestamp])
    period = ts.dt.floor(freq)
    full_period_hours = pd.Timedelta(freq).total_seconds() / 3600.0

    grouped = pd.DataFrame({"period": period, "ts": ts}).groupby("period")["ts"]
    counts = grouped.size()
    span_hours = (grouped.max() - grouped.min()).dt.total_seconds() / 3600.0

    # Only the first and last period can be partial; anything in between
    # is bounded by two full period-lengths of data on either side, so
    # its true span is the full period even if no event happens to land
    # exactly on the boundary tick.
    is_boundary = pd.Series(False, index=counts.index)
    if len(is_boundary) > 0:
        is_boundary.iloc[[0, -1]] = True
    effective_hours = span_hours.where(is_boundary & (span_hours > 0), full_period_hours)

    result = pd.DataFrame({
        "period": counts.index,
        "n_events": counts.values,
        "pieces_per_hour": (counts / effective_hours).round(1).values,
    })
    return result.sort_values("period").reset_index(drop=True)