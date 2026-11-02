#!/usr/bin/env python3
"""Flag CATS air mole fractions that are far from nearby air (tag 286).

A plain outlier filter on *mole fractions* (cal_step looks at abrupt cal-port
response shifts, baseline at chromatogram shape). It uses air only: no tank
values and no reference-tank noise.

A reading is flagged if it is more than `sigma` (default 5) air sigmas from
the median of the other air1+air2 readings within +/- window_days/2 (default
10 days, so +/-5). Air sigma is 1.4826 x the median absolute deviation of
those same readings. The bound is the same above and below: an asymmetric bound
removes more low points than high ones and pushes monthly means up.

The target's own value is left out of its median and sigma, so an anomalous
point can't hide itself. Air1 and air2 are pooled because both read the same
air intake through the same normalization.

The median and sigma use only air not already rejected for some OTHER reason
(cal_step, baseline, manual review, GCwerks sync). Rows carrying only this
algorithm's own tag stay in the pool, and every air row is scored whatever its
rejected state, so reruns land on the same flagged set rather than flapping.

Limit: this finds points that differ from their neighbours. An episode longer
than about half the window moves the median with it and is not flagged, so
those still need a manual tag.

The tag's dictionary name is "Mole fraction falls outside of calibration
range", which predates this rule: tank values are not consulted.

Usage::

    python3 cats_cal_window_qc.py --site brw --gas N2O_q --start 20250101 --end 20250401

    python3 cats_tagging.py --site brw --algo cal_window --analyte N2O \\
        --channel q --start 20250101 --end 20250401 --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cats_batch import CATS_batch

# The reporting channels (cats_cal_step_qc.py's roster, CATS_GCwerks2DB.UPLOAD_MOLS)
# plus the secondary "a" channel. cal_window only needs the air mole fractions
# of one channel, so it works on any channel that has been processed; a channel
# a site lacks, or has no mole fractions for (e.g. CFC11_a before a gets a cal
# method), just flags nothing.
ALL_GASES = (
    "N2O_q", "SF6_q", "CFC12_f", "CFC11_f", "CFC113_f",
    "H1211_f", "CCl4_f", "CH3CCl3_f", "CHCl3_f",
    "N2O_a", "CFC12_a", "CFC11_a", "CFC113_a", "H1211_a",
)

# "C" reject tag: "Mole fraction falls outside of calibration range, results
# certainly adversely affected" -- registered in ccgg.tag_dictionary and
# _TAG_LAYOUT's Automated Tags section (logosdata/logos_tagging.py). Kept here (the single
# source of truth cats_tagging.py's ALGORITHMS registers) so
# _already_rejected_mf_nums can exclude this algorithm's own prior tags from
# its "already rejected" pool filter without importing cats_tagging (which
# imports this module -- would be circular).
TAG_NUM = 286

# Scales a median absolute deviation to a normal standard deviation.
MAD_TO_SIGMA = 1.4826


def _already_rejected_mf_nums(db, mf_nums, batch_size: int = 1000) -> set[int]:
    """mf_nums (from the given collection) carrying a reject tag other than
    TAG_NUM.

    Used to build the air statistics from data already cleaned by earlier
    detectors (cal_step, baseline) or manual review -- excludes TAG_NUM itself
    so this algorithm's own previous tags never feed back into its own
    thresholds on rerun (see module docstring).
    """
    mf_nums = sorted({int(n) for n in mf_nums if pd.notna(n)})
    if not mf_nums:
        return set()
    other: set[int] = set()
    for i in range(0, len(mf_nums), batch_size):
        chunk = mf_nums[i:i + batch_size]
        placeholders = ",".join(["%s"] * len(chunk))
        rows = db.doquery(
            "SELECT DISTINCT t.ng_insitu_mole_fraction_num AS mf_num "
            "FROM hats.ng_insitu_mole_fraction_tags t "
            "JOIN ccgg.tag_view tv ON tv.tag_num = t.tag_num "
            f"WHERE t.ng_insitu_mole_fraction_num IN ({placeholders}) "
            "AND t.tag_num != %s AND tv.reject = 1",
            [*chunk, TAG_NUM],
        ) or []
        other.update(int(r["mf_num"]) for r in rows)
    return other


def _parse_yyyymmdd(s: str) -> str:
    """Parse YYYYMMDD (or YYYY-MM-DD) into 'YYYY-MM-DD'."""
    s = s.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date {s!r}; expected YYYYMMDD or YYYY-MM-DD."
    )


def _resolve_pnum(batch: CATS_batch, gas: str, channel: str) -> int:
    key = f"{gas} ({channel})".lower()
    lookup = {k.lower(): v for k, v in batch.analytes.items()}
    pnum = lookup.get(key)
    if pnum is None:
        raise ValueError(
            f"No analyte_list entry for {gas!r} ({channel!r}) at "
            f"{batch.inst_id} site {batch.site}"
        )
    return int(pnum)


def _windowed_air_outliers(
    air_times: np.ndarray,
    air_values: np.ndarray,
    air_in_pool: np.ndarray,
    half_window: np.timedelta64,
    sigma: float,
    min_air_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure windowed-outlier core, no pandas/DB involved (unit-testable).

    air_times must already be sorted ascending datetime64[ns]; air_values are
    the matching mole fractions. air_in_pool is a bool array aligned to them,
    True where that row may be used as a neighbour when scoring OTHER
    candidates -- every row is still itself scored regardless (see module
    docstring).

    Returns (outlier, air_sigma, air_median), each aligned to air_times. A
    point with too few in-pool neighbours in its +/-half_window, or whose
    neighbours have no spread at all, gets outlier=False: not enough context
    to judge (same convention as cats_cal_step_qc._rolling_mad_scale).
    """
    n = len(air_times)
    outlier = np.zeros(n, dtype=bool)
    air_sigma = np.full(n, np.nan)
    air_median = np.full(n, np.nan)

    air_lo = np.searchsorted(air_times, air_times - half_window, side="left")
    air_hi = np.searchsorted(air_times, air_times + half_window, side="right")

    for i in range(n):
        air_idxs = np.arange(air_lo[i], air_hi[i])
        air_idxs = air_idxs[air_idxs != i]
        air_idxs = air_idxs[air_in_pool[air_idxs]]
        if len(air_idxs) < min_air_points:
            continue
        neighbours = air_values[air_idxs]
        median = float(np.median(neighbours))
        scale = MAD_TO_SIGMA * float(np.median(np.abs(neighbours - median)))
        air_median[i] = median
        air_sigma[i] = scale
        if scale > 0 and abs(air_values[i] - median) > sigma * scale:
            outlier[i] = True

    return outlier, air_sigma, air_median


def build_cal_window_qc(
    batch: CATS_batch,
    pnum: int,
    channel: str,
    start: str,
    end: str,
    window_days: float = 10.0,
    sigma: float = 5.0,
    min_air_points: int = 20,
) -> pd.DataFrame:
    """cats_tagging.py-compatible build(): (batch, pnum, channel, start, end,
    **kwargs) -> DataFrame with at least an mf_num column for flagged rows.

    Loads window_days/2 of extra context on each side of [start, end] so
    candidates near the requested boundaries still get a full window (same
    padding idea as cats_cal_step_qc.build_cal_step_qc's scale_window_days
    lookback); those context-only rows are scored (they can act as air
    neighbours) but never themselves returned as flagged.
    """
    half_window_days = window_days / 2.0
    half_window = pd.Timedelta(days=half_window_days).to_timedelta64()

    requested_start = pd.Timestamp(start)
    requested_start = (
        requested_start.tz_localize("UTC") if requested_start.tzinfo is None
        else requested_start.tz_convert("UTC")
    )
    requested_end = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC")
    requested_end = (
        requested_end.tz_localize("UTC") if requested_end.tzinfo is None
        else requested_end.tz_convert("UTC")
    )

    load_start = (requested_start - pd.Timedelta(days=half_window_days)).strftime("%Y-%m-%d")
    load_end = (requested_end + pd.Timedelta(days=half_window_days)).strftime("%Y-%m-%d")

    df = batch.load_data(pnum, channel=channel, start_date=load_start, end_date=load_end, verbose=False)
    if df.empty or "mf_num" not in df:
        return pd.DataFrame()
    df = df.copy()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True)
    df["mole_fraction"] = pd.to_numeric(df["mole_fraction"], errors="coerce")

    air = (
        df.loc[df["port"].isin(batch.AIR_PORTS) & df["mole_fraction"].notna()]
        .sort_values("analysis_datetime")
        .reset_index(drop=True)
    )
    if air.empty:
        return pd.DataFrame()

    other_rejected = _already_rejected_mf_nums(
        batch.db, pd.to_numeric(air["mf_num"], errors="coerce").tolist()
    )
    air_in_pool = ~pd.to_numeric(air["mf_num"], errors="coerce").isin(other_rejected).to_numpy()
    air_times = air["analysis_datetime"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    outlier, air_sigma, air_median = _windowed_air_outliers(
        air_times, air["mole_fraction"].to_numpy(dtype=float), air_in_pool,
        half_window, sigma, min_air_points,
    )

    air["air_sigma"] = air_sigma
    air["air_median"] = air_median
    air["lower_bound"] = air_median - sigma * air_sigma
    air["upper_bound"] = air_median + sigma * air_sigma
    air["cal_window_outlier"] = outlier

    in_range = air["analysis_datetime"].between(requested_start, requested_end)
    return air.loc[in_range & air["cal_window_outlier"]].reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", default="brw")
    p.add_argument("--gas", default="N2O_q", help='Gas_channel (e.g. N2O_q) or "all"')
    p.add_argument("--start", required=True, type=_parse_yyyymmdd)
    p.add_argument("--end", required=True, type=_parse_yyyymmdd)
    p.add_argument("--window-days", type=float, default=10.0,
                   help="Full window width, centered on each candidate (default: 10)")
    p.add_argument("--sigma", type=float, default=5.0,
                   help="Air sigmas from the local air median to flag, either side (default: 5)")
    p.add_argument("--min-air-points", type=int, default=20,
                   help="Minimum other air readings in-window to trust the median and sigma (default: 20)")
    p.add_argument("--output", type=Path, default=Path("cats_cal_window_flags.csv"))
    args = p.parse_args()

    gases = list(ALL_GASES) if args.gas.lower() == "all" else [args.gas]
    batch = CATS_batch(args.site)

    frames = []
    for gas_channel in gases:
        gas, channel = gas_channel.rsplit("_", 1)
        try:
            pnum = _resolve_pnum(batch, gas, channel)
        except ValueError as exc:
            print(f"  Skipping {gas_channel}: {exc}")
            continue
        out = build_cal_window_qc(
            batch, pnum, channel, args.start, args.end,
            window_days=args.window_days, sigma=args.sigma,
            min_air_points=args.min_air_points,
        )
        if out.empty:
            continue
        out["gas"] = gas
        out["channel"] = channel
        out["pnum"] = pnum
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.10g")
    print(f"Wrote {len(result):,} flagged rows to {args.output}")
    if not result.empty:
        cols = ["gas", "channel", "analysis_datetime", "mole_fraction",
                "air_median", "air_sigma", "lower_bound", "upper_bound"]
        print(result[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
