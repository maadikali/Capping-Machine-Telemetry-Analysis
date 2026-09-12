import pandas as pd

from capping_telemetry.ingestion import status_codes
from capping_telemetry.ingestion.closure_detection import detect_closures, ClosureEventColumns as C
from capping_telemetry.analytics import kpi, anomaly


def test_status_classification_matches_status_table():
    assert status_codes.classify(0) == "success"
    assert status_codes.classify(2) == "no_load"
    assert status_codes.classify(3) == "reject"   # reject_signal YES
    assert status_codes.classify(4) == "fault"    # reject_signal NO, not success/no_load
    assert status_codes.classify(65) == "reject"
    assert status_codes.classify(64) == "fault"
    assert status_codes.classify(999) == "unknown"


def test_no_load_excluded_from_success_and_reject_counts(settings):
    """A No Load closure (status 2) must not count as either a success
    or a reject - it's not a capping attempt at all.

    Note: the first row of any dataframe never registers as a closure
    (there's no prior row to diff the Count against), so row 0 below is
    just the baseline - the 3 real closures are rows 1-3.
    """
    rows = [
        ("2026-05-20T00:00:00Z", 2.0, 0, 100),  # baseline, not a closure
        ("2026-05-20T00:00:01Z", 2.5, 0, 101),  # closure 1: success
        ("2026-05-20T00:00:02Z", 0.0, 2, 102),  # closure 2: no_load
        ("2026-05-20T00:00:03Z", 2.6, 3, 103),  # closure 3: reject (SlowTorque)
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "H01 AppTorque", "H01 Status", "H01 Count"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    events = detect_closures(df, settings)
    result = kpi.overall_success_rate(events)

    assert result["total_closures"] == 3
    assert result["no_load"] == 1
    assert result["attempted"] == 2
    assert result["successful"] == 1
    assert result["rejected"] == 1
    assert result["success_rate_pct"] == 50.0


def test_fault_code_is_neither_successful_nor_rejected(settings):
    """Status 64 (Bad Closure) has reject_signal=NO but isn't a success -
    it must show up as a fault, not silently vanish or count as a reject."""
    rows = [
        ("2026-05-20T00:00:00Z", 2.0, 0, 100),   # baseline, not a closure
        ("2026-05-20T00:00:01Z", 1.8, 64, 101),  # closure: fault (Bad Closure)
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "H01 AppTorque", "H01 Status", "H01 Count"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    events = detect_closures(df, settings)
    assert len(events) == 1
    assert events.iloc[0][C.status_category] == "fault"
    assert events.iloc[0][C.is_reject] == False  # noqa: E712
    assert events.iloc[0][C.success] == False  # noqa: E712

    result = kpi.overall_success_rate(events)
    assert result["faults"] == 1
    assert result["rejected"] == 0
    assert result["successful"] == 0


def test_head_with_most_failures_uses_reject_not_all_non_success(settings):
    """A head with only No Load events (no rejects) should not be
    reported as the top failure contributor over a head that has a
    genuine reject, even if it has fewer total non-success events."""
    rows = [
        ("2026-05-20T00:00:00Z", 2.0, 0, 100, 2.0, 0, 50),   # baseline for both heads
        ("2026-05-20T00:00:01Z", 0.0, 2, 101, 1.9, 3, 51),   # H01 no-load, H02 reject
    ]
    cols = ["timestamp", "H01 AppTorque", "H01 Status", "H01 Count",
            "H02 AppTorque", "H02 Status", "H02 Count"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    events = detect_closures(df, settings)
    result = anomaly.head_with_most_failures(events)
    assert result["head_id"] == "H02"
    assert result["failure_count"] == 1
