from capping_telemetry.analytics import kpi
from capping_telemetry.ingestion.closure_detection import detect_closures


def test_overall_success_rate_matches_expected_shape(tiny_polling_df, settings):
    events = detect_closures(tiny_polling_df, settings)
    result = kpi.overall_success_rate(events)

    assert result["total_closures"] == 3
    assert result["no_load"] == 0  # fixture has no status-2 events
    assert result["attempted"] == 3
    assert result["successful"] == 2
    assert result["rejected"] == 1  # status 65 has reject_signal=YES
    assert result["success_rate_pct"] == round(100 * 2 / 3, 1)


def test_success_rate_per_head(tiny_polling_df, settings):
    events = detect_closures(tiny_polling_df, settings)
    per_head = kpi.success_rate_per_head(events)

    h01_row = per_head[per_head["head_id"] == "H01"].iloc[0]
    assert h01_row["total_closures"] == 2
    assert h01_row["successful"] == 2
    assert h01_row["success_rate_pct"] == 100.0

    h02_row = per_head[per_head["head_id"] == "H02"].iloc[0]
    assert h02_row["total_closures"] == 1
    assert h02_row["successful"] == 0
    assert h02_row["success_rate_pct"] == 0.0


def test_success_rate_per_head_handles_head_with_zero_attempted(settings):
    # Regression test for a real crash: a head that is 100% No Load
    # (zero attempted closures) must produce NaN for success_rate_pct,
    # not raise. `Series.replace(0, pd.NA)` on an int column upcasts to
    # object dtype in pandas, which then breaks `.round()`.
    import pandas as pd
    from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C

    rows = [
        ("2026-05-20T00:00:00Z", "H01", 2.5, 0, True, False, False, False, "success", "ok", 1, 1),
        ("2026-05-20T00:00:01Z", "H02", 0.0, 2, False, True, False, False, "no_load", "idle", 1, 1),
    ]
    cols = [C.timestamp, C.head_id, C.torque, C.status, C.success, C.is_no_load,
            C.is_reject, C.is_fault, C.status_category, C.status_description, C.count, C.seq_gap]
    df = pd.DataFrame(rows, columns=cols)
    df[C.timestamp] = pd.to_datetime(df[C.timestamp], utc=True)

    per_head = kpi.success_rate_per_head(df)  # must not raise

    h02_row = per_head[per_head["head_id"] == "H02"].iloc[0]
    assert h02_row["attempted"] == 0
    assert pd.isna(h02_row["success_rate_pct"])

    h01_row = per_head[per_head["head_id"] == "H01"].iloc[0]
    assert h01_row["success_rate_pct"] == 100.0


def test_torque_statistics_successful_only(tiny_polling_df, settings):
    events = detect_closures(tiny_polling_df, settings)
    stats = kpi.torque_statistics(events, successful_only=True)

    # only the two successful H01 closures (2.55, 2.30) should count
    assert stats["n_events"] == 2
    assert stats["mean_nm"] == round((2.55 + 2.30) / 2, 2)


def test_empty_events_do_not_crash():
    import pandas as pd
    empty = pd.DataFrame(columns=["event_timestamp", "head_id", "torque", "status", "success", "count", "count_seq_gap"])
    assert kpi.overall_success_rate(empty)["total_closures"] == 0
    assert kpi.torque_statistics(empty)["n_events"] == 0
    assert kpi.time_range(empty) == {"start": None, "end": None}