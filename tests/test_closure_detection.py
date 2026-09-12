import pandas as pd

from capping_telemetry.ingestion.closure_detection import (
    ClosureEventColumns as C,
    capping_speed_pieces_per_hour,
    detect_closures,
)


def test_polling_rows_collapse_to_real_closure_events(tiny_polling_df, settings):
    """
    6 polling rows must collapse into exactly 3 real closure events
    (H01 x2, H02 x1), NOT 6 - this is the core dedup/cleaning requirement.
    """
    events = detect_closures(tiny_polling_df, settings)

    assert len(events) == 3
    assert sorted(events[C.head_id].tolist()) == ["H01", "H01", "H02"]


def test_closure_captures_the_correct_torque_and_status(tiny_polling_df, settings):
    events = detect_closures(tiny_polling_df, settings).sort_values(C.timestamp)

    h01_events = events[events[C.head_id] == "H01"].reset_index(drop=True)
    assert h01_events.loc[0, C.torque] == 2.55
    assert h01_events.loc[0, C.success] == True  # noqa: E712 (status 0)
    assert h01_events.loc[1, C.torque] == 2.30

    h02_events = events[events[C.head_id] == "H02"].reset_index(drop=True)
    assert h02_events.loc[0, C.torque] == 1.95
    assert h02_events.loc[0, C.success] == False  # noqa: E712 (status 65 = failure)


def test_no_spurious_events_when_count_unchanged():
    """A head with a flat (unchanging) counter across all polls should
    contribute zero closure events, even though rows repeat many times."""
    rows = [
        ("2026-05-20T00:00:00Z", 2.4, 0, 10),
        ("2026-05-20T00:00:01Z", 2.4, 0, 10),
        ("2026-05-20T00:00:02Z", 2.4, 0, 10),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "H01 AppTorque", "H01 Status", "H01 Count"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    from capping_telemetry.config import Settings, DataConfig, HeadsConfig, SchemaConfig, \
        AnalyticsConfig, LoggingConfig

    minimal_settings = Settings(
        data=DataConfig(pools_dir="data/pools", default_pool="x"),
        heads=HeadsConfig(ids=["H01"]),
        schema=SchemaConfig(),
        analytics=AnalyticsConfig(torque_expected_range_nm=[1.5, 3.5]),
        logging=LoggingConfig(),
    )

    events = detect_closures(df, minimal_settings)
    assert len(events) == 0


def test_capping_speed_is_positive_and_finite(tiny_polling_df, settings):
    events = detect_closures(tiny_polling_df, settings)
    speed = capping_speed_pieces_per_hour(events)
    finite_speeds = speed.dropna()
    assert (finite_speeds > 0).all()
