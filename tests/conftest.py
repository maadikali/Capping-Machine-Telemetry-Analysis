import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure src/ is importable when running pytest from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from capping_telemetry.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def settings():
    config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    return load_config(str(config_path))


@pytest.fixture
def tiny_polling_df():
    """
    Hand-built minimal polling dataframe with 2 heads. Poll rate faster
    than the production cycle on purpose (several rows share the same
    Count between real closures) - this is the exact case the closure
    detector must handle correctly.

    H01 closes at polls 2 and 5 (count 100->101, 101->102).
    H02 closes at poll 4 only (count 50->51).
    """
    rows = [
        # ts,                         H01 AppTorque, H01 Status, H01 Count, H02 AppTorque, H02 Status, H02 Count
        ("2026-05-20T00:00:00Z", 2.40, 0, 100, 2.10, 0, 50),
        ("2026-05-20T00:00:01Z", 2.40, 0, 100, 2.10, 0, 50),
        ("2026-05-20T00:00:02Z", 2.55, 0, 101, 2.10, 0, 50),  # H01 closure #1
        ("2026-05-20T00:00:03Z", 2.55, 0, 101, 2.10, 0, 50),
        ("2026-05-20T00:00:04Z", 2.55, 0, 101, 1.95, 65, 51),  # H02 closure #1 (failed)
        ("2026-05-20T00:00:05Z", 2.30, 0, 102, 1.95, 65, 51),  # H01 closure #2
    ]
    cols = [
        "timestamp",
        "H01 AppTorque", "H01 Status", "H01 Count",
        "H02 AppTorque", "H02 Status", "H02 Count",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df
