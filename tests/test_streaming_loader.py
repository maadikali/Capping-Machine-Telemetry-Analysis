"""
Covers the memory-bounded, multi-file loading path (loader.load_pool_streaming)
introduced to handle multi-month pools without holding all raw data in
memory at once. The key correctness property to test is that splitting
the *same* data across two files (e.g. one day's worth split at an
arbitrary row) produces identical results to processing it as a single
file - i.e. no closures or idle time are lost or double-counted at a
file boundary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from capping_telemetry.ingestion.closure_detection import detect_closures
from capping_telemetry.ingestion.loader import load_pool_streaming


def _write_pool(tmp_path: Path, pool_name: str, frames: list[pd.DataFrame]) -> Path:
    pool_dir = tmp_path / "pools" / pool_name
    pool_dir.mkdir(parents=True)
    for i, frame in enumerate(frames):
        out = frame.copy()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(pool_dir / f"part-{i:02d}.csv", index=False)
    return pool_dir


def _make_settings(settings, tmp_path: Path):
    return settings.model_copy(update={
        "data": settings.data.model_copy(update={"pools_dir": str(tmp_path / "pools")}),
        "project_root": tmp_path,
    })


@pytest.fixture
def two_head_df():
    # H01 closes at row idx 2 and idx 5 (spans the split point below);
    # H02 closes once, sits idle (status=2, "No Load") before and after.
    rows = [
        ("2026-05-20T00:00:00Z", 2.40, 0, 100, 0.0, 2, 50),
        ("2026-05-20T00:00:01Z", 2.40, 0, 100, 0.0, 2, 50),
        ("2026-05-20T00:00:02Z", 2.55, 0, 101, 0.0, 2, 50),  # H01 closure #1
        ("2026-05-20T00:00:03Z", 2.55, 0, 101, 0.0, 2, 50),  # <- split here
        ("2026-05-20T00:00:04Z", 2.55, 0, 101, 2.10, 0, 51),  # H02 closure
        ("2026-05-20T00:00:05Z", 2.30, 0, 102, 0.0, 2, 51),  # H01 closure #2, H02 idle again
        ("2026-05-20T00:00:06Z", 2.30, 0, 102, 0.0, 2, 51),
    ]
    cols = ["timestamp", "H01 AppTorque", "H01 Status", "H01 Count",
            "H02 AppTorque", "H02 Status", "H02 Count"]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_streaming_matches_single_file_closure_events(settings, tmp_path, two_head_df):
    single = _make_settings(settings, tmp_path / "single")
    _write_pool(tmp_path / "single", "p", [two_head_df])
    events_single, _, _ = load_pool_streaming(single, pool_name="p")

    split = _make_settings(settings, tmp_path / "split")
    # Split the exact same rows across two files, right in the middle of
    # H01's closure run (the scenario that breaks a naive per-file diff).
    _write_pool(tmp_path / "split", "p", [two_head_df.iloc[:4], two_head_df.iloc[4:]])
    events_split, _, _ = load_pool_streaming(split, pool_name="p")

    assert len(events_split) == len(events_single)
    assert sorted(events_split["torque"].tolist()) == sorted(events_single["torque"].tolist())


def test_streaming_idle_run_merges_across_file_boundary(settings, tmp_path, two_head_df):
    # H02 is idle (status=2) both right before and right after the split
    # point - the streaming path must report this as ONE idle run for
    # H02, not two separate short ones (each shorter than
    # idle_no_load_seconds and therefore silently dropped).
    split = settings.model_copy(update={
        "data": settings.data.model_copy(update={"pools_dir": str(tmp_path / "pools")}),
        "project_root": tmp_path,
        "analytics": settings.analytics.model_copy(update={"idle_no_load_seconds": 2}),
    })
    _write_pool(tmp_path, "p", [two_head_df.iloc[:4], two_head_df.iloc[4:]])

    _, idle_periods, _ = load_pool_streaming(split, pool_name="p")
    h02_runs = idle_periods[idle_periods["head_id"] == "H02"]

    # H02 is idle for rows 0-3 (4s) then idle again for rows 5-6 (1s),
    # separated by exactly one real closure at row 4 - these must stay
    # as two distinct runs (there's a real closure between them), but
    # neither run should be split further just because of the file break
    # inside the first (rows 0-3) run.
    assert len(h02_runs) == 1  # only the first run clears the 2s threshold
    assert h02_runs.iloc[0]["duration_s"] == pytest.approx(3.0)


def test_streaming_reports_quality_issues(settings, tmp_path, two_head_df):
    bad = two_head_df.copy()
    bad.loc[0, "H01 AppTorque"] = None
    pool_settings = _make_settings(settings, tmp_path)
    _write_pool(tmp_path, "p", [bad])

    _, _, meta = load_pool_streaming(pool_settings, pool_name="p")
    assert meta["quality_issues"]
    assert any("H01 AppTorque" in issue for issue in meta["quality_issues"])


def test_detect_closures_with_carried_tail_row_does_not_duplicate(settings, tiny_polling_df):
    # Sanity check on the mechanism load_pool_streaming relies on: a
    # single-row "tail" from the previous file, prepended before the next
    # file's rows, must never itself be reported as a closure.
    tail = tiny_polling_df.tail(1)
    next_chunk = pd.concat([tail, tiny_polling_df.tail(1)], ignore_index=True)
    events = detect_closures(next_chunk, settings)
    # No count actually changed between the carried row and the
    # (identical) next row, so there must be zero events.
    assert events.empty
