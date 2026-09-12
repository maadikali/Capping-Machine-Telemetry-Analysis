import pandas as pd
import pytest

from capping_telemetry.analytics import trend
from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C


def _events_at(timestamps):
    n = len(timestamps)
    return pd.DataFrame({
        C.timestamp: pd.to_datetime(timestamps, utc=True),
        C.head_id: ["H01"] * n,
        C.torque: [2.5] * n,
        C.status: [0] * n,
        C.success: [True] * n,
        C.is_no_load: [False] * n,
        C.is_reject: [False] * n,
        C.is_fault: [False] * n,
    })


def test_capping_speed_over_time_full_day_uses_24h():
    # One event/minute for a full day (00:00 to 23:59) -> 60/hour exactly.
    timestamps = pd.date_range("2026-05-20 00:00:00", "2026-05-20 23:59:00", freq="1min", tz="UTC")
    events = _events_at(timestamps)

    result = trend.capping_speed_over_time(events, freq="1D")

    assert len(result) == 1
    assert result.iloc[0]["pieces_per_hour"] == 60.0


def test_capping_speed_over_time_does_not_understate_partial_boundary_day():
    """
    Regression test: a day with events only from 06:00 to 23:59 (a
    genuinely partial first day, e.g. because that's when logging
    started) must not be divided by a full 24h when computing
    pieces_per_hour - the true rate is 1/minute = 60/hour throughout,
    and the reported rate for the partial day must reflect that, not
    a diluted ~45/hour from assuming a full 24h window.
    """
    timestamps = pd.date_range("2026-05-20 06:00:00", "2026-05-20 23:59:00", freq="1min", tz="UTC")
    events = _events_at(timestamps)

    result = trend.capping_speed_over_time(events, freq="1D")

    # The span between the first and last event (17h59m) is one polling
    # interval short of the "true" 18h window those 1,080 one-per-minute
    # events represent (a fence-post effect inherent to measuring span as
    # max-min timestamp) - pieces_per_hour comes out ~60.1 rather than an
    # exact 60.0. That's a world away from the ~45/hour the old fixed-24h
    # divisor would have reported, so a small tolerance is the right bar.
    assert len(result) == 1
    assert result.iloc[0]["pieces_per_hour"] == pytest.approx(60.0, rel=0.01)


def test_capping_speed_over_time_partial_first_and_last_day_around_full_middle_day():
    # Day 1: 18h of data (partial), Day 2: full 24h, Day 3: 6h of data (partial).
    # Rate is a constant 1 event/minute (60/hour) throughout all three.
    day1 = pd.date_range("2026-05-20 06:00:00", "2026-05-20 23:59:00", freq="1min", tz="UTC")
    day2 = pd.date_range("2026-05-21 00:00:00", "2026-05-21 23:59:00", freq="1min", tz="UTC")
    day3 = pd.date_range("2026-05-22 00:00:00", "2026-05-22 05:59:00", freq="1min", tz="UTC")
    events = _events_at(list(day1) + list(day2) + list(day3))

    result = trend.capping_speed_over_time(events, freq="1D").set_index("period")

    for day in ["2026-05-20", "2026-05-21", "2026-05-22"]:
        rate = result.loc[pd.Timestamp(day, tz="UTC"), "pieces_per_hour"]
        # Small tolerance for the boundary-day fence-post effect (see the
        # partial-day test above); the key property under test is that
        # all three days land close to the same true rate, unlike the
        # pre-fix behavior where the partial days would be ~25%+ off.
        assert rate == pytest.approx(60.0, rel=0.01), f"{day}: expected ~60.0/hour, got {rate}"


def test_capping_speed_over_time_empty_events():
    empty = pd.DataFrame(columns=[C.timestamp, C.head_id, C.torque, C.status,
                                   C.success, C.is_no_load, C.is_reject, C.is_fault])
    result = trend.capping_speed_over_time(empty)
    assert result.empty
    assert list(result.columns) == ["period", "n_events", "pieces_per_hour"]
