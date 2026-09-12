"""
Generic filtering/listing over the closure-events table: pick any
combination of head, status category, torque range, and date range
without needing a bespoke function per combination.
"""
from __future__ import annotations

import pandas as pd

from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C

VALID_STATUS_CATEGORIES = {"success", "no_load", "reject", "fault"}


def resolve_end_of_range(end_date, end_ts: pd.Timestamp) -> pd.Timestamp:
    """
    Decides whether `end_date` was a bare date (extend to end-of-day) or
    an explicit timestamp (treat as an exact instant), based on the
    STRING LENGTH of the original input ("2026-03-05" is 10 chars, a
    bare date; "2026-03-05T00:00:00Z" is 20 chars, explicit) - not on
    the parsed time-of-day, since that can't tell an implicit midnight
    (meaning "the whole day") from an explicit one (an exact instant,
    where stretching to 23:59:59 would silently return 24h extra data).
    """
    if len(str(end_date)) <= 10:
        return end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return end_ts


def list_closure_events(
    events: pd.DataFrame,
    head_id: str | None = None,
    status_category: str | None = None,
    torque_min: float | None = None,
    torque_max: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Filters the closure-events table by any combination of head, status
    category (success/no_load/reject/fault), torque range, and date
    range.

    start_date/end_date accept an ISO date ("2026-03-05") or full ISO
    datetime ("2026-03-05T14:00:00Z"). A bare date is treated as that
    whole day, i.e. start_date = 00:00:00 and end_date = 23:59:59.999...
    (end-inclusive) - see resolve_end_of_range.

    Returns the matching rows (timestamp, head_id, torque,
    status_category, status_description).
    """
    result = events

    if head_id:
        result = result[result[C.head_id] == head_id]

    if status_category:
        if status_category not in VALID_STATUS_CATEGORIES:
            raise ValueError(
                f"Unknown status_category '{status_category}'. "
                f"Must be one of: {sorted(VALID_STATUS_CATEGORIES)}"
            )
        result = result[result[C.status_category] == status_category]

    if torque_min is not None:
        result = result[result[C.torque] >= torque_min]
    if torque_max is not None:
        result = result[result[C.torque] <= torque_max]

    if (start_date is not None or end_date is not None) and not result.empty:
        ts = pd.to_datetime(result[C.timestamp], utc=True)
        if start_date is not None:
            start_ts = pd.Timestamp(start_date)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            result = result[ts >= start_ts]
            ts = pd.to_datetime(result[C.timestamp], utc=True)
        if end_date is not None:
            end_ts = pd.Timestamp(end_date)
            if end_ts.tzinfo is None:
                end_ts = end_ts.tz_localize("UTC")
                end_ts = resolve_end_of_range(end_date, end_ts)
            result = result[ts <= end_ts]

    cols = [C.timestamp, C.head_id, C.torque, C.status_category, C.status_description]
    return result[cols].sort_values(C.timestamp).reset_index(drop=True)
