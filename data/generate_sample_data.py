"""
Generates a fully synthetic multi-head capping-line telemetry pool that
matches the wide-format schema this project ingests, e.g.:

    timestamp,
      H01 Count, H02 Count, ..., H36 Count,
      H01 AppTorque, H02 AppTorque, ..., H36 AppTorque,
      H01 Status, H02 Status, ..., H36 Status

Note the column names are SPACE-separated ("H01 Count", not "H01_Count"),
and columns are grouped by field (all 36 Counts, then all 36 AppTorques,
then all 36 Statuses) rather than grouped by head. The loader/schema
module doesn't care about column order, but we replicate it here for
realism when eyeballing the file.

Key property this replicates on purpose: the polling frequency is faster
than the real production cycle, so several consecutive rows share the
same Count value for a head between closures - that's the exact mismatch
the ingestion pipeline has to handle.

Default size: 24 hours at 1 Hz -> 86,400 polling rows, in line with a
full daily export from a real line.

Usage:
    python data/generate_sample_data.py --out data/pools/sample_pool/telemetry.csv
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

N_HEADS = 36
HEADS = [f"H{i:02d}" for i in range(1, N_HEADS + 1)]


def _build_head_profiles(seed: int) -> dict:
    """Give each head a slightly different torque mean/std/no-load rate/
    reject rate/cycle time so per-head KPIs differ meaningfully. A handful
    of heads are deliberately made worse performers (higher reject rate),
    mirroring the "problem head" pattern often seen on real production lines.

    Status codes are sampled from the status table (see
    ingestion/status_codes.py): success=0, no_load=2, plus a mix of
    reject/fault codes for the rest. No Load is made the dominant
    non-success outcome, matching what is typically observed in real
    capping-line data (a large majority of raw "closures" tend to be No
    Load cycles, not faults).
    """
    rng = np.random.default_rng(seed)
    profiles = {}
    weak_heads = set(rng.choice(HEADS, size=max(1, N_HEADS // 9), replace=False))

    for h in HEADS:
        is_weak = h in weak_heads
        profiles[h] = dict(
            torque_mean=round(float(rng.normal(2.5 if not is_weak else 2.2, 0.08)), 2),
            torque_std=round(float(rng.uniform(0.08, 0.14 if not is_weak else 0.24)), 3),
            no_load_rate=round(float(rng.uniform(0.75, 0.85)), 3),
            reject_rate=round(float(rng.uniform(0.005, 0.02 if not is_weak else 0.05)), 4),
            fault_rate=round(float(rng.uniform(0.002, 0.01)), 4),
            cycle_s=round(float(rng.uniform(2.8, 3.4)), 2),
        )
    return profiles


# reject_signal=YES codes and reject_signal=NO fault codes,
# see config/status_codes.json / ingestion/status_codes.py
_REJECT_CODES = [3, 5, 9, 17, 33, 65]
_FAULT_CODES = [4, 8, 16, 32, 64]


def generate(
    start: datetime,
    duration_minutes: float,
    poll_hz: float,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    profiles = _build_head_profiles(seed)

    poll_interval = timedelta(seconds=1.0 / poll_hz)
    n_polls = int(duration_minutes * 60 * poll_hz)

    state = {
        h: dict(
            count=20000 + i * 137,
            torque=profiles[h]["torque_mean"],
            status=0,
            next_closure_in=rng.uniform(1, profiles[h]["cycle_s"]),
        )
        for i, h in enumerate(HEADS)
    }

    rows = []
    t = start
    for _ in range(n_polls):
        row = {"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ")}
        counts, torques, statuses = {}, {}, {}
        for h in HEADS:
            prof = profiles[h]
            s = state[h]
            s["next_closure_in"] -= poll_interval.total_seconds()
            if s["next_closure_in"] <= 0:
                s["count"] += 1
                roll = rng.random()
                if roll < prof["no_load_rate"]:
                    torque, status = 0.0, 2  # No Load
                elif roll < prof["no_load_rate"] + prof["reject_rate"]:
                    torque = round(max(0.1, float(np_rng.normal(
                        prof["torque_mean"] * 0.6, prof["torque_std"]
                    ))), 2)
                    status = int(rng.choice(_REJECT_CODES))
                elif roll < prof["no_load_rate"] + prof["reject_rate"] + prof["fault_rate"]:
                    torque = round(max(0.1, float(np_rng.normal(
                        prof["torque_mean"], prof["torque_std"]
                    ))), 2)
                    status = int(rng.choice(_FAULT_CODES))
                else:
                    torque = round(max(0.1, float(np_rng.normal(
                        prof["torque_mean"], prof["torque_std"]
                    ))), 2)
                    status = 0  # success
                s["torque"] = torque
                s["status"] = status
                s["next_closure_in"] += rng.uniform(
                    prof["cycle_s"] * 0.85, prof["cycle_s"] * 1.15
                )
            counts[h] = s["count"]
            torques[h] = s["torque"]
            statuses[h] = s["status"]

        # column order matches the real file: all Counts, then AppTorques, then Statuses
        for h in HEADS:
            row[f"{h} Count"] = counts[h]
        for h in HEADS:
            row[f"{h} AppTorque"] = torques[h]
        for h in HEADS:
            row[f"{h} Status"] = statuses[h]

        rows.append(row)
        t += poll_interval

    cols = (
        ["timestamp"]
        + [f"{h} Count" for h in HEADS]
        + [f"{h} AppTorque" for h in HEADS]
        + [f"{h} Status" for h in HEADS]
    )
    return pd.DataFrame(rows, columns=cols)


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic capping-line telemetry pool (36-head schema)")
    parser.add_argument("--out", default="data/pools/sample_pool/telemetry.csv")
    parser.add_argument("--minutes", type=float, default=1440, help="duration to simulate (default: 1440 = 24h)")
    parser.add_argument("--poll-hz", type=float, default=1.0, help="polling frequency in Hz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate(
        start=datetime(2026, 4, 1, 6, 0, 0, tzinfo=timezone.utc),
        duration_minutes=args.minutes,
        poll_hz=args.poll_hz,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} polling rows, {len(df.columns)} columns -> {out_path}")
    total_closures = sum(
        df[f"{h} Count"].iloc[-1] - df[f"{h} Count"].iloc[0] for h in HEADS
    )
    print(f"Approx. total closure events embedded: {int(total_closures)}")


if __name__ == "__main__":
    main()