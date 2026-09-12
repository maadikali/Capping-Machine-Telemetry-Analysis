"""
The core cleaning problem for this project: polling frequency is higher
than the machine's real production cycle, so a single closure event shows
up across several consecutive polling rows for the same head. Generic
"drop duplicate rows" logic does NOT solve this - most of those rows are
not exact duplicates (timestamp differs, other heads may have changed).

The correct signal is the per-head monotonically increasing Count column:
a closure event happened for head H exactly when H's Count increases
relative to the previous row. This module turns the wide polling
DataFrame into a clean, long-format "closure events" table with one row
per real closure event.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from capping_telemetry.config import Settings
from capping_telemetry.ingestion.schema import all_head_columns
from capping_telemetry.ingestion import status_codes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClosureEventColumns:
    """Column names used in the output events table (kept in one place
    so analytics modules don't hard-code strings)."""
    timestamp = "event_timestamp"
    head_id = "head_id"
    torque = "torque"
    status = "status"
    success = "success"          # True only for status 0 ("Closure OK")
    is_no_load = "is_no_load"    # True only for status 2 ("No Load")
    is_reject = "is_reject"      # True for any reject_signal=YES status
    is_fault = "is_fault"        # True for reject_signal=NO, non-success, non-no-load
    status_category = "status_category"      # success / no_load / reject / fault / unknown
    status_description = "status_description"  # human-readable, from status_codes.describe
    count = "count"
    seq_gap = "count_seq_gap"  # >1 means missed/undetected intermediate closures


def detect_closures(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Convert wide polling data into a long-format table of real closure
    events, one row per (head, closure).

    Logic per head:
      1. Compute count.diff() against the previous poll.
      2. Any row where the diff > 0 is a real closure event (or several,
         if diff > 1 - the polling loop missed some between two polls).
      3. The torque/status recorded on that row are the ones that were
         live for that closure.
      4. The event timestamp is approximated as the poll's timestamp
         (finer reconstruction is possible with cycle-time interpolation
         between the previous and current poll, if closer precision is
         required than one polling interval).
    """
    ts_col = settings.schema_.timestamp_col
    events = []

    for hc in all_head_columns(settings):
        if hc.count_col not in df.columns:
            logger.warning("Head %s missing from dataset, skipping", hc.head_id)
            continue

        sub = df[[ts_col, hc.torque_col, hc.status_col, hc.count_col]].copy()
        sub["count_diff"] = sub[hc.count_col].diff()

        # The first row is the starting counter value, not a new event.
        closures = sub[sub["count_diff"] > 0].copy()
        if closures.empty:
            continue

        closures[ClosureEventColumns.seq_gap] = closures["count_diff"].astype("Int64")
        closures[ClosureEventColumns.head_id] = hc.head_id
        closures[ClosureEventColumns.timestamp] = closures[ts_col]
        closures[ClosureEventColumns.torque] = closures[hc.torque_col]
        closures[ClosureEventColumns.status] = closures[hc.status_col]
        closures[ClosureEventColumns.count] = closures[hc.count_col]

        categories = closures[hc.status_col].apply(status_codes.classify)
        closures[ClosureEventColumns.status_category] = categories
        closures[ClosureEventColumns.status_description] = closures[hc.status_col].apply(status_codes.describe)
        closures[ClosureEventColumns.success] = categories == "success"
        closures[ClosureEventColumns.is_no_load] = categories == "no_load"
        closures[ClosureEventColumns.is_reject] = categories == "reject"
        closures[ClosureEventColumns.is_fault] = categories == "fault"

        events.append(
            closures[
                [
                    ClosureEventColumns.timestamp,
                    ClosureEventColumns.head_id,
                    ClosureEventColumns.torque,
                    ClosureEventColumns.status,
                    ClosureEventColumns.success,
                    ClosureEventColumns.is_no_load,
                    ClosureEventColumns.is_reject,
                    ClosureEventColumns.is_fault,
                    ClosureEventColumns.status_category,
                    ClosureEventColumns.status_description,
                    ClosureEventColumns.count,
                    ClosureEventColumns.seq_gap,
                ]
            ]
        )

    if not events:
        return pd.DataFrame(
            columns=[
                ClosureEventColumns.timestamp,
                ClosureEventColumns.head_id,
                ClosureEventColumns.torque,
                ClosureEventColumns.status,
                ClosureEventColumns.success,
                ClosureEventColumns.is_no_load,
                ClosureEventColumns.is_reject,
                ClosureEventColumns.is_fault,
                ClosureEventColumns.status_category,
                ClosureEventColumns.status_description,
                ClosureEventColumns.count,
                ClosureEventColumns.seq_gap,
            ]
        )

    result = pd.concat(events, ignore_index=True)
    result = result.sort_values(ClosureEventColumns.timestamp).reset_index(drop=True)

    n_gaps = (result[ClosureEventColumns.seq_gap] > 1).sum()
    if n_gaps:
        logger.warning(
            "%d closure row(s) had a count jump > 1 - polling interval may have "
            "missed intermediate closures for those heads",
            n_gaps,
        )

    logger.info(
        "Detected %d closure events across %d heads (from %d polling rows)",
        len(result), result[ClosureEventColumns.head_id].nunique(), len(df),
    )
    return result


def capping_speed_pieces_per_hour(events: pd.DataFrame) -> pd.Series:
    """
    Incremental average capping speed (pieces/hour), computed across ALL
    heads combined, using the running mean of inter-event time deltas.
    Returned as a Series indexed like `events`, so it can be attached back
    for time-series inspection.
    """
    if events.empty:
        return pd.Series(dtype=float)

    ts = pd.to_datetime(events[ClosureEventColumns.timestamp])
    order = ts.sort_values().index
    ordered_ts = ts.loc[order]

    deltas_s = ordered_ts.diff().dt.total_seconds()
    # Use the running average gap to estimate line speed.
    running_avg_gap = deltas_s.expanding().mean()
    speed = 3600.0 / running_avg_gap.replace(0, np.nan)

    return speed.reindex(events.index)
