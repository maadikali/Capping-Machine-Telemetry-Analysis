"""
Idle-state detection: identifies periods where a head reports a sustained
"No Load" status, using the RAW polling data (not the closure-events
table), since idle time is by definition between closures.
"""
from __future__ import annotations

import pandas as pd

from capping_telemetry.config import Settings
from capping_telemetry.ingestion.schema import all_head_columns
from capping_telemetry.ingestion import status_codes


def detect_idle_periods(raw_df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Returns one row per contiguous "No Load" run per head that lasts at
    least `idle_no_load_seconds`, with start/end/duration. "No Load"
    (status 2) comes from the line's own status-code mapping, not a config
    placeholder.
    """
    ts_col = settings.schema_.timestamp_col
    min_duration = settings.analytics.idle_no_load_seconds

    results = []
    for hc in all_head_columns(settings):
        if hc.status_col not in raw_df.columns:
            continue
        sub = raw_df[[ts_col, hc.status_col]].copy()
        sub["is_idle"] = sub[hc.status_col].apply(status_codes.classify) == status_codes.CATEGORY_NO_LOAD

        # group consecutive True/False runs
        sub["run_id"] = (sub["is_idle"] != sub["is_idle"].shift()).cumsum()
        for _, run in sub[sub["is_idle"]].groupby("run_id"):
            start, end = run[ts_col].iloc[0], run[ts_col].iloc[-1]
            duration_s = (end - start).total_seconds()
            if duration_s >= min_duration:
                results.append({
                    "head_id": hc.head_id,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "duration_s": round(duration_s, 1),
                })

    df = pd.DataFrame(results)
    return df.sort_values("duration_s", ascending=False) if not df.empty else df


# ---------------------------------------------------------------------------
# Streaming variant, used by loader.load_pool_streaming so a whole
# multi-month pool never needs to hold every day's raw polling data in
# memory at once. Each daily file is processed on its own; an idle run
# that is still open ("No Load" continuing) at the end of one file is
# carried forward and, if it continues at the very start of the next
# file, merged back into a single run instead of being reported as two
# separate runs split exactly at the file boundary (e.g. midnight).
# ---------------------------------------------------------------------------

def detect_idle_periods_chunk(
    df: pd.DataFrame, settings: Settings, open_runs: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Process idle detection for a single chunk (typically one day's raw
    polling data).

    open_runs: {head_id: {"start": Timestamp, "last_ts": Timestamp}} for
    any run that was still ongoing at the end of the previous chunk.
    Pass {} for the first chunk.

    Returns (completed_runs_df, new_open_runs). new_open_runs must be fed
    into the call for the next chunk, and into flush_open_idle_runs()
    after the last chunk.
    """
    ts_col = settings.schema_.timestamp_col
    min_duration = settings.analytics.idle_no_load_seconds
    completed: list = []
    new_open_runs: dict = {}

    for hc in all_head_columns(settings):
        prior = open_runs.get(hc.head_id)

        if hc.status_col not in df.columns or df.empty:
            # Nothing to observe for this head in this chunk - keep
            # whatever was open carried forward unchanged.
            if prior is not None:
                new_open_runs[hc.head_id] = prior
            continue

        sub = df[[ts_col, hc.status_col]].reset_index(drop=True)
        sub["is_idle"] = sub[hc.status_col].apply(status_codes.classify) == status_codes.CATEGORY_NO_LOAD
        sub["run_id"] = (sub["is_idle"] != sub["is_idle"].shift()).cumsum()

        first_row_idle = bool(sub["is_idle"].iloc[0])
        if prior is not None and not first_row_idle:
            # The run open at the end of the previous chunk did not
            # continue into this one - close it using the last timestamp
            # we actually saw it idle at (slightly conservative: the true
            # end is somewhere in the gap between chunks, but that gap is
            # at most one polling interval).
            duration_s = (prior["last_ts"] - prior["start"]).total_seconds()
            if duration_s >= min_duration:
                completed.append({
                    "head_id": hc.head_id,
                    "start": prior["start"].isoformat(),
                    "end": prior["last_ts"].isoformat(),
                    "duration_s": round(duration_s, 1),
                })
            prior = None

        for _, run in sub[sub["is_idle"]].groupby("run_id"):
            start, end = run[ts_col].iloc[0], run[ts_col].iloc[-1]
            is_first_row = run.index[0] == 0
            is_last_row = run.index[-1] == len(sub) - 1

            if is_first_row and prior is not None:
                start = prior["start"]  # merge with the carried-in open run

            if is_last_row:
                new_open_runs[hc.head_id] = {"start": start, "last_ts": end}
            else:
                duration_s = (end - start).total_seconds()
                if duration_s >= min_duration:
                    completed.append({
                        "head_id": hc.head_id,
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "duration_s": round(duration_s, 1),
                    })

    result = pd.DataFrame(completed, columns=["head_id", "start", "end", "duration_s"])
    return result, new_open_runs


def flush_open_idle_runs(open_runs: dict, settings: Settings) -> pd.DataFrame:
    """Close out any runs still open at the very end of the whole pool
    (call once, after the last chunk has been processed)."""
    min_duration = settings.analytics.idle_no_load_seconds
    rows = []
    for head_id, run in open_runs.items():
        duration_s = (run["last_ts"] - run["start"]).total_seconds()
        if duration_s >= min_duration:
            rows.append({
                "head_id": head_id,
                "start": run["start"].isoformat(),
                "end": run["last_ts"].isoformat(),
                "duration_s": round(duration_s, 1),
            })
    return pd.DataFrame(rows, columns=["head_id", "start", "end", "duration_s"])
