#!/usr/bin/env python3
"""Flag CATS air mole fractions that fall outside their local calibration noise.

Unlike cal_step (abrupt cal-port response shifts) and baseline (abnormal
chromatogram shape), this looks at *mole fractions*, not raw response, and
judges each air1/air2 reading against its own neighbourhood rather than a
fixed threshold:

1. **Reference-tank noise as the yardstick.** CAL2_PORT (== STANDARD_PORT_NUM
   for CATS -- the near-ambient tank used as the normalization reference, see
   ../CLAUDE.md) is run repeatedly like the air ports, so its own mole
   fraction spread over a short window is a direct measurement of how noisy
   the instrument currently is -- independent of any atmospheric variability.
   `ref_std` is the plain standard deviation of the reference tank's
   mole_fraction values inside the window (ddof=0, matching
   cats_baseline_qc.py's ratio_std).
2. **Local air median as the baseline.** `air_median` is the median
   mole_fraction of every air1+air2 reading inside the same window, computed
   leave-one-out (the target's own value is excluded) -- same rationale as
   cats_baseline_qc.py's local reference: including the target lets a
   genuinely anomalous point drag its own baseline toward itself, always
   biasing toward under-detection. Air1 and air2 are pooled into one median
   (not scored separately) because both read the same air intake through the
   same normalization.
3. **Asymmetric bounds.** A reading is flagged if it is more than
   `sigma_high` (default 3) reference-tank sigmas *above* air_median, or more
   than `sigma_low` (default 2) sigmas *below* it. The asymmetry is
   deliberate -- downward excursions (partial peaks, leaks, a starved sample
   loop) are judged more strictly than upward ones (e.g. brief contamination
   spikes), matching how these failure modes actually show up in CATS data.
4. **Window.** 10 days, centered on the candidate reading (`--window-days`,
   default 10 => +/-5 days), so both statistics reflect what the instrument
   and atmosphere were doing right around that point rather than a stale
   trailing average.

Run this AFTER cal_step and baseline (and any manual review) have already
been applied, and after mole fractions have been recomputed against that
rejection state (see cats_tagging.py's recalc step, run automatically before
cal_window). The reference-tank/air statistics above are built only from
data not already rejected for some OTHER reason -- a cal-port glitch
(cal_step) or a bad chromatogram (baseline) already known to be bad
shouldn't also count toward "what normal noise/air looks like right now".
`cats_tagging.py --algo all` already sequences cal_step -> baseline ->
cal_window (dict order in ALGORITHMS), which is why this should run last.

Deliberately does NOT exclude rows rejected by this algorithm's OWN tag
(TAG_NUM) from that "already rejected" filter -- every air row is always
re-evaluated fresh against the SAME (stable) non-self pool on every rerun.
If a previously-flagged point's own rejection fed back into the pool it
would itself be judged against, a rerun could shift the median/std enough
to flip the verdict, flapping between two different flagged sets instead of
converging -- breaking cats_tagging.py's delete-over-scope + reinsert
idempotency contract.

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

# Same analyte/channel roster as cats_cal_step_qc.py (CATS_GCwerks2DB.UPLOAD_MOLS).
ALL_GASES = (
    "N2O_q", "SF6_q", "CFC12_f", "CFC11_f", "CFC113_f",
    "H1211_f", "CCl4_f", "CH3CCl3_f", "CHCl3_f",
)

# "C" reject tag: "Mole fraction falls outside of calibration range, results
# certainly adversely affected" -- registered in ccgg.tag_dictionary and
# _TAG_LAYOUT's Automated Tags section (logosdata/logos_tagging.py). Kept here (the single
# source of truth cats_tagging.py's ALGORITHMS registers) so
# _already_rejected_mf_nums can exclude this algorithm's own prior tags from
# its "already rejected" pool filter without importing cats_tagging (which
# imports this module -- would be circular).
TAG_NUM = 286


def _already_rejected_mf_nums(db, mf_nums, batch_size: int = 1000) -> set[int]:
    """mf_nums (from the given collection) carrying a reject tag other than
    TAG_NUM.

    Used to build the reference-tank/air statistics from data already
    cleaned by earlier detectors (cal_step, baseline) or manual review --
    excludes TAG_NUM itself so this algorithm's own previous tags never feed
    back into its own thresholds on rerun (see module docstring).
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
    key = f"{gas} ({channel})"
    pnum = batch.analytes.get(key)
    if pnum is None:
        raise ValueError(
            f"No analyte_list entry for {key!r} at {batch.inst_id} site {batch.site}"
        )
    return int(pnum)


def _windowed_air_outliers(
    air_times: np.ndarray,
    air_values: np.ndarray,
    air_in_pool: np.ndarray,
    ref_times: np.ndarray,
    ref_values: np.ndarray,
    half_window: np.timedelta64,
    sigma_high: float,
    sigma_low: float,
    min_ref_points: int,
    min_air_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure windowed-outlier core, no pandas/DB involved (unit-testable).

    air_times/ref_times must already be sorted ascending datetime64[ns]
    arrays; air_values/ref_values are the matching mole_fraction arrays;
    ref_times/ref_values must already be restricted to whatever pool the
    caller wants used for the reference-tank noise (e.g. not-already-rejected
    rows). air_in_pool is a bool array aligned to air_times/air_values,
    True where that air row is eligible to be used as a neighbour when
    scoring OTHER candidates -- every row in air_times is still itself
    evaluated as a candidate regardless of air_in_pool (see module
    docstring: excluding already-rejected rows from candidacy, not just from
    the pool, would make this algorithm's own prior tags feed back into
    itself and break idempotency).

    Returns (outlier, ref_std, air_median, n_ref_points), each aligned to
    air_times. Points with too few reference or (leave-one-out, in-pool) air
    neighbours in their +/-half_window get NaN stats and outlier=False,
    same "not enough context to judge" convention as
    cats_cal_step_qc._rolling_mad_scale (scale=NaN -> never flagged).
    """
    n = len(air_times)
    outlier = np.zeros(n, dtype=bool)
    ref_std = np.full(n, np.nan)
    air_median = np.full(n, np.nan)
    n_ref_points = np.zeros(n, dtype=int)

    ref_lo = np.searchsorted(ref_times, air_times - half_window, side="left")
    ref_hi = np.searchsorted(ref_times, air_times + half_window, side="right")
    air_lo = np.searchsorted(air_times, air_times - half_window, side="left")
    air_hi = np.searchsorted(air_times, air_times + half_window, side="right")

    for i in range(n):
        ref_window = ref_values[ref_lo[i]:ref_hi[i]]
        n_ref_points[i] = len(ref_window)
        if n_ref_points[i] < min_ref_points:
            continue
        sigma = float(np.std(ref_window))
        if not np.isfinite(sigma):
            continue
        ref_std[i] = sigma

        air_idxs = np.arange(air_lo[i], air_hi[i])
        air_idxs = air_idxs[air_idxs != i]
        air_idxs = air_idxs[air_in_pool[air_idxs]]
        if len(air_idxs) < min_air_points:
            continue
        median = float(np.median(air_values[air_idxs]))
        air_median[i] = median

        value = air_values[i]
        if value > median + sigma_high * sigma or value < median - sigma_low * sigma:
            outlier[i] = True

    return outlier, ref_std, air_median, n_ref_points


def build_cal_window_qc(
    batch: CATS_batch,
    pnum: int,
    channel: str,
    start: str,
    end: str,
    window_days: float = 10.0,
    sigma_high: float = 3.0,
    sigma_low: float = 2.0,
    min_ref_points: int = 4,
    min_air_points: int = 4,
) -> pd.DataFrame:
    """cats_tagging.py-compatible build(): (batch, pnum, channel, start, end,
    **kwargs) -> DataFrame with at least an mf_num column for flagged rows.

    Loads window_days/2 of extra context on each side of [start, end] so
    candidates near the requested boundaries still get a full window (same
    padding idea as cats_cal_step_qc.build_cal_step_qc's scale_window_days
    lookback); those context-only rows are scored (they can act as
    reference/air neighbours) but never themselves returned as flagged.
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

    is_ref = df["port"].eq(batch.CAL2_PORT)
    is_air = df["port"].isin(batch.AIR_PORTS)
    mf_num_numeric = pd.to_numeric(df["mf_num"], errors="coerce")
    other_rejected = _already_rejected_mf_nums(batch.db, mf_num_numeric[is_ref | is_air].tolist())
    df["already_rejected"] = mf_num_numeric.isin(other_rejected)

    ref = (
        df.loc[is_ref & df["mole_fraction"].notna() & ~df["already_rejected"]]
        .sort_values("analysis_datetime")
    )
    air = (
        df.loc[is_air & df["mole_fraction"].notna()]
        .sort_values("analysis_datetime")
        .reset_index(drop=True)
    )
    if ref.empty or air.empty:
        return pd.DataFrame()

    def _naive_times(frame: pd.DataFrame) -> np.ndarray:
        return frame["analysis_datetime"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()

    air_in_pool = ~air["already_rejected"].to_numpy()
    outlier, ref_std, air_median, n_ref_points = _windowed_air_outliers(
        _naive_times(air), air["mole_fraction"].to_numpy(dtype=float), air_in_pool,
        _naive_times(ref), ref["mole_fraction"].to_numpy(dtype=float),
        half_window, sigma_high, sigma_low, min_ref_points, min_air_points,
    )

    air = air.copy()
    air["ref_std"] = ref_std
    air["air_median"] = air_median
    air["n_ref_points"] = n_ref_points
    air["lower_bound"] = air_median - sigma_low * ref_std
    air["upper_bound"] = air_median + sigma_high * ref_std
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
    p.add_argument("--sigma-high", type=float, default=3.0,
                   help="Reference-tank sigmas above the local air median to flag (default: 3)")
    p.add_argument("--sigma-low", type=float, default=2.0,
                   help="Reference-tank sigmas below the local air median to flag (default: 2)")
    p.add_argument("--min-ref-points", type=int, default=4,
                   help="Minimum reference-tank readings in-window to trust ref_std (default: 4)")
    p.add_argument("--min-air-points", type=int, default=4,
                   help="Minimum other air readings in-window to trust air_median (default: 4)")
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
            window_days=args.window_days, sigma_high=args.sigma_high, sigma_low=args.sigma_low,
            min_ref_points=args.min_ref_points, min_air_points=args.min_air_points,
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
                "air_median", "ref_std", "lower_bound", "upper_bound"]
        print(result[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
