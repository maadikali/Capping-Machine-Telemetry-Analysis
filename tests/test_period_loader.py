"""
Covers loading scoped by date range instead of (or spanning) a single
pool folder - e.g. "Feb 12 to March 15" when February and March live in
separate pool directories (data/pools/2026-02/, data/pools/2026-03/).
See loader.resolve_period_files / load_period_streaming.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from capping_telemetry.ingestion.loader import load_period_streaming, resolve_period_files


def _write_dated_pool(pools_dir: Path, pool_name: str, day_frames: dict[str, pd.DataFrame]) -> None:
    """day_frames: {'2026-02-27': df, '2026-02-28': df, ...} - filenames
    follow a dated filename convention so resolve_period_files can read
    the date straight out of the filename."""
    pool_dir = pools_dir / pool_name
    pool_dir.mkdir(parents=True, exist_ok=True)
    for day, frame in day_frames.items():
        out = frame.copy()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(pool_dir / f"telemetry_MACHINE_{day}.csv", index=False)


def _day_df(day: str, start_count: int) -> pd.DataFrame:
    rows = [
        (f"{day}T00:00:0{i}Z", 2.0, 0, start_count + i, 0.0, 2, 50)
        for i in range(3)
    ]
    cols = ["timestamp", "H01 AppTorque", "H01 Status", "H01 Count", "H02 AppTorque", "H02 Status", "H02 Count"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _make_settings(settings, tmp_path: Path):
    return settings.model_copy(update={
        "data": settings.data.model_copy(update={"pools_dir": str(tmp_path / "pools")}),
        "project_root": tmp_path,
    })


@pytest.fixture
def two_month_pools(settings, tmp_path):
    scoped = _make_settings(settings, tmp_path)
    pools_dir = tmp_path / "pools"
    _write_dated_pool(pools_dir, "2026-02", {
        "2026-02-20": _day_df("2026-02-20", 50),
        "2026-02-27": _day_df("2026-02-27", 100),
        "2026-02-28": _day_df("2026-02-28", 200),
    })
    _write_dated_pool(pools_dir, "2026-03", {
        "2026-03-01": _day_df("2026-03-01", 300),
        "2026-03-02": _day_df("2026-03-02", 400),
        "2026-03-15": _day_df("2026-03-15", 500),
    })
    return scoped


def test_resolve_period_files_spans_multiple_pool_dirs(two_month_pools):
    # The search window is deliberately widened by a day on each side
    # (see resolve_period_files docstring - real exports from a source system can put a
    # calendar day's data in the NEXT day's file), so requesting
    # 02-28..03-01 also picks up the immediately adjacent 02-27 and
    # 03-02 files - but not the far-outside 02-20 / 03-15 ones.
    files = resolve_period_files(two_month_pools, start_date="2026-02-28", end_date="2026-03-01")
    names = sorted(f.name for f in files)
    assert names == [
        "telemetry_MACHINE_2026-02-27.csv",
        "telemetry_MACHINE_2026-02-28.csv",
        "telemetry_MACHINE_2026-03-01.csv",
        "telemetry_MACHINE_2026-03-02.csv",
    ]


def test_resolve_period_files_excludes_files_far_outside_range(two_month_pools):
    files = resolve_period_files(two_month_pools, start_date="2026-02-28", end_date="2026-03-01")
    names = {f.name for f in files}
    assert "telemetry_MACHINE_2026-02-20.csv" not in names
    assert "telemetry_MACHINE_2026-03-15.csv" not in names


def test_resolve_period_files_can_restrict_to_named_pools(two_month_pools):
    # Even though the full date range would include both pools, an
    # explicit pools=[...] list restricts the search.
    files = resolve_period_files(
        two_month_pools, pools=["2026-02"], start_date="2026-02-01", end_date="2026-03-31"
    )
    assert all("2026-02" in f.parent.name for f in files)
    assert len(files) == 3


def test_load_period_streaming_spans_pool_boundary_and_trims_to_exact_range(two_month_pools):
    # Even though resolve_period_files widens the search and picks up
    # 4 files (02-27, 02-28, 03-01, 03-02), the returned EVENTS must be
    # trimmed to exactly [2026-02-28 00:00, 2026-03-01 23:59:59] - none
    # from 02-27 or 03-02 should leak through.
    events, idle_periods, meta = load_period_streaming(
        two_month_pools, start_date="2026-02-28", end_date="2026-03-01"
    )
    assert meta["n_files"] == 4
    assert meta["trimmed_to_exact_range"] is True
    assert not events.empty
    from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C
    ts = pd.to_datetime(events[C.timestamp], utc=True)
    assert (ts >= pd.Timestamp("2026-02-28", tz="UTC")).all()
    assert (ts <= pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).all()


def test_load_period_streaming_trims_shifted_day_data_correctly(settings, tmp_path):
    # Reproduces the real discovery: a file named for calendar day D
    # actually contains data offset into day D-1/D+1. A naive
    # file-name-only approach would silently include/exclude the wrong
    # rows; load_period_streaming must trim by actual event timestamp
    # so the result is exactly the requested calendar range regardless.
    from capping_telemetry.ingestion.closure_detection import ClosureEventColumns as C

    scoped = _make_settings(settings, tmp_path)
    pools_dir = tmp_path / "pools"
    pool_dir = pools_dir / "2026-03"
    pool_dir.mkdir(parents=True)

    # File "named" 03-05 but actually contains 03-04 16:00 -> 03-05 15:59:59,
    # mirroring a day-shift export quirk seen on real systems.
    rows = []
    for i, hour_offset in enumerate([-4, 2, 10]):  # relative to 03-05 00:00
        ts = pd.Timestamp("2026-03-05", tz="UTC") + pd.Timedelta(hours=hour_offset)
        rows.append((ts.strftime("%Y-%m-%dT%H:%M:%SZ"), 2.0, 0, 100 + i, 0.0, 2, 50))
    cols = ["timestamp", "H01 AppTorque", "H01 Status", "H01 Count", "H02 AppTorque", "H02 Status", "H02 Count"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(pool_dir / "telemetry_MACHINE_2026-03-05.csv", index=False)

    events, _, meta = load_period_streaming(scoped, start_date="2026-03-05", end_date="2026-03-05")

    ts = pd.to_datetime(events[C.timestamp], utc=True)
    # The -4h-offset row (2026-03-04 20:00) must be trimmed out even
    # though it lived inside the file named "03-05".
    assert (ts >= pd.Timestamp("2026-03-05", tz="UTC")).all()


def test_resolve_period_files_raises_for_empty_result(two_month_pools):
    with pytest.raises(FileNotFoundError):
        resolve_period_files(two_month_pools, start_date="2027-06-01", end_date="2027-06-30")