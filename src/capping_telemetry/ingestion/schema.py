"""
Understands the wide-format telemetry schema used by this project:

    timestamp, H01_AppTorque, H01_Status, H01_Count, H02_AppTorque, ...

One row per poll. Columns repeat per head using the suffixes defined in
config.yaml. This module centralizes that knowledge so nothing else in
the codebase parses column names by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from capping_telemetry.config import Settings


@dataclass(frozen=True)
class HeadColumns:
    head_id: str
    torque_col: str
    status_col: str
    count_col: str


def head_columns(settings: Settings, head_id: str) -> HeadColumns:
    s = settings.schema_
    return HeadColumns(
        head_id=head_id,
        torque_col=f"{head_id}{s.torque_suffix}",
        status_col=f"{head_id}{s.status_suffix}",
        count_col=f"{head_id}{s.count_suffix}",
    )


def all_head_columns(settings: Settings) -> List[HeadColumns]:
    return [head_columns(settings, h) for h in settings.heads.ids]


def required_columns(settings: Settings) -> List[str]:
    cols = [settings.schema_.timestamp_col]
    for hc in all_head_columns(settings):
        cols += [hc.torque_col, hc.status_col, hc.count_col]
    return cols


def validate_schema(df: pd.DataFrame, settings: Settings) -> List[str]:
    """Return a list of human-readable validation problems (empty = clean)."""
    problems: List[str] = []
    missing = [c for c in required_columns(settings) if c not in df.columns]
    if missing:
        problems.append(f"Missing expected columns: {missing}")

    ts_col = settings.schema_.timestamp_col
    if ts_col in df.columns:
        parsed = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        n_bad = parsed.isna().sum()
        if n_bad:
            problems.append(f"{n_bad} row(s) have unparseable '{ts_col}' values")
        if parsed.notna().any() and not parsed.dropna().is_monotonic_increasing:
            problems.append(f"'{ts_col}' is not monotonically increasing - check for out-of-order rows")

    for hc in all_head_columns(settings):
        for col in (hc.torque_col, hc.status_col, hc.count_col):
            if col in df.columns:
                n_missing = df[col].isna().sum()
                if n_missing:
                    problems.append(f"{n_missing} missing value(s) in '{col}'")

        if hc.torque_col in df.columns:
            bad_torque = df[hc.torque_col].dropna()
            n_negative = (bad_torque < 0).sum()
            if n_negative:
                problems.append(
                    f"{n_negative} negative torque reading(s) in '{hc.torque_col}'"
                )

        if hc.torque_col in df.columns:
            # Zero torque is expected for "No Load" cycles (status 2) -
            # only flag zero torque that is NOT accompanied by No Load,
            # since that combination is genuinely unexpected.
            torque = df[hc.torque_col]
            if hc.status_col in df.columns:
                zero_mask = (torque == 0) & df[hc.status_col].notna() & (df[hc.status_col] != 2)
            else:
                zero_mask = torque == 0
            n_zero = int(zero_mask.fillna(False).sum())
            if n_zero:
                problems.append(
                    f"{n_zero} zero torque reading(s) in '{hc.torque_col}' NOT marked 'No Load' "
                    f"(status != 2) - unlike No Load's expected zero torque, this combination is unexpected"
                )

        if hc.count_col in df.columns:
            counts = df[hc.count_col].dropna()
            n_decreasing = (counts.diff().dropna() < 0).sum()
            if n_decreasing:
                problems.append(
                    f"{n_decreasing} row(s) where '{hc.count_col}' decreases "
                    "(counter reset or out-of-order data)"
                )

    return problems
