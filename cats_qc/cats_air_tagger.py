#!/usr/bin/env python3
"""Detect and visualize multi-day CATS air (Air1/Air2) mole-fraction
excursions that deviate from the analyte's own smooth long-term trend --
stretches where the instrument was clearly not tracking real atmospheric
values (contamination, detector upset, a starved sample loop, etc.), too
extended in duration for the existing point-local detectors to catch: cal_step
judges cal-port response only (not air), baseline judges chromatogram shape
(not the resulting mole fraction), and cal_window's own +/-5-day local window
gets swallowed whole by any excursion lasting longer than that (see
cats_qc/README.md) -- exactly the multi-day case this tool targets.

Detect + visualize ONLY -- writes a review CSV and a JPG, never touches the
database or hats.ng_insitu_mole_fraction_tags. Not yet registered in
cats_tagging.py's ALGORITHMS; meant to be tuned by eye against the output
figure first (--sigma / --median-window-days / --mad-window-days /
--min-block-hours) before anything gets wired into the automated pipeline --
the same iterate-before-wiring-in path cats_baseline_qc.py and
cats_cal_step_qc.py went through.

Algorithm
---------
1. Load AIR_PORTS-only, currently-unrejected mole_fraction for one
   analyte/channel over the requested range. Air1 and Air2 are pooled into
   one series for the detection math (same physical air intake through the
   same normalization -- same rationale cats_cal_window_qc.py already uses
   for its own air_median), split apart only for plotting.
2. baseline(t) = a centered, leave-one-out rolling MEDIAN over
   --median-window-days. Median, not mean or a fitted curve, is the whole
   design: an excursion covering less than half of any window's points
   can't pull a median toward it (~50% breakdown point), so as long as
   --median-window-days is at least roughly 2x the longest excursion you're
   hunting, the baseline stays clean straight through it -- the exact
   failure mode that defeats a fixed-form regression or a too-short local
   window (see cats_cal_window_qc.py's own +/-5-day blind spot, and
   cats_cal_method_qc.py's contamination bug from a window reaching into an
   unrelated bad stretch). A modest window (weeks, not a full year) also
   means this baseline naturally tracks the real seasonal cycle and secular
   trend without needing an explicit harmonic/polynomial model or being
   sensitive to which multi-year window you happened to pick.
3. residual(t) = value(t) - baseline(t) -- the trend-removed noise series.
4. scale(t) = 1.4826 * (centered, leave-one-out rolling MEDIAN of
   |residual| over --mad-window-days) -- a robust, self-calibrating "how
   much does this analyte's residual normally wiggle right now" (the same
   1.4826*MAD convention used throughout cats_qc/), computed on the
   ALREADY-DETRENDED residual so a real secular/seasonal change in level
   never gets counted as noise. Independently windowed from the baseline
   (--mad-window-days can differ from --median-window-days) since the two
   answer different questions: how fast the real trend moves, vs. how noisy
   the instrument currently is.
5. Flag point i where |residual(i)| > --sigma * scale(i). Flagged points
   within --max-gap-hours of each other are grouped into one block (reuses
   cats_cal_step_qc._group_periods); --min-block-hours optionally drops
   isolated single/few-point blips, keeping only the sustained multi-day
   excursions this tool is aimed at (0 keeps everything, for a first,
   unfiltered look while tuning).

Usage::

    python3 cats_air_tagger.py --site brw --gas N2O_q --output brw_n2o_q_air_tags

    # Narrower range + tighter windows for fast iteration while tuning
    python3 cats_air_tagger.py --site brw --gas N2O_q --start 20000101 --end 20031231 \\
        --median-window-days 45 --sigma 3.5 --output brw_n2o_q_air_tags_2000s
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cats_batch import CATS_batch
from cats_cal_step_qc import _group_periods


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


def _rolling_median(
    times: np.ndarray, values: np.ndarray, window_days: float, min_points: int,
) -> np.ndarray:
    """Centered, leave-one-out rolling median, pure numpy (no pandas/DB) --
    unit-testable. times must be sorted ascending datetime64[ns], values the
    matching float array (NaNs allowed and excluded, not just zero-filled).

    Leave-one-out (the candidate's own value never enters its own window) so
    a genuinely anomalous point can't drag its own baseline toward itself --
    same rationale as cats_baseline_qc.py's local reference and
    cats_cal_window_qc.py's leave-one-out air median. NaN wherever fewer
    than min_points finite neighbours fall in the window excluding the point
    itself -- not enough context to judge, the same convention used
    throughout cats_qc/.
    """
    n = len(times)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    half = pd.Timedelta(days=window_days / 2.0).to_timedelta64()
    lo = np.searchsorted(times, times - half, side="left")
    hi = np.searchsorted(times, times + half, side="right")
    finite = np.isfinite(values)
    for i in range(n):
        idx = np.concatenate([np.arange(lo[i], i), np.arange(i + 1, hi[i])])
        idx = idx[finite[idx]]
        if len(idx) < min_points:
            continue
        out[i] = np.median(values[idx])
    return out


def detect_air_excursions(
    times: np.ndarray,
    values: np.ndarray,
    median_window_days: float,
    mad_window_days: float,
    min_points: int,
    sigma: float,
) -> dict[str, np.ndarray]:
    """Core detector, pure numpy -- unit-testable, no DB/pandas-DataFrame
    dependency. See module docstring for the algorithm. Returns a dict of
    baseline/residual/scale/z/outlier arrays, each aligned to times/values.
    """
    baseline = _rolling_median(times, values, median_window_days, min_points)
    residual = values - baseline
    scale = 1.4826 * _rolling_median(times, np.abs(residual), mad_window_days, min_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = residual / scale
    outlier = np.isfinite(z) & (np.abs(z) > sigma)
    return {
        "baseline": baseline, "residual": residual, "scale": scale,
        "z": z, "outlier": outlier,
    }


def build_air_tags(
    batch: CATS_batch,
    pnum: int,
    channel: str,
    start: str,
    end: str | None,
    median_window_days: float = 60.0,
    mad_window_days: float = 60.0,
    min_points: int = 8,
    sigma: float = 4.0,
    max_gap_hours: float = 48.0,
    min_block_hours: float = 0.0,
) -> pd.DataFrame:
    """Load AIR_PORTS-only unrejected data for [start, end] (padded by half
    the longer window on each side so candidates near the requested
    boundary still get a full window), run detect_air_excursions, group
    flagged points into blocks, optionally drop blocks shorter than
    --min-block-hours.

    Returns one row per point IN THE REQUESTED RANGE, flagged or not --
    unlike the other cats_qc/ detectors this keeps unflagged rows too
    (needed to draw the full green/red figure); filter on 'outlier'
    yourself for a flagged-only view. Empty DataFrame if no air data.
    """
    pad_days = max(median_window_days, mad_window_days) / 2.0
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")
    load_end = (
        (pd.Timestamp(end) + pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")
        if end else None
    )

    df = batch.load_data(pnum, channel=channel, start_date=load_start, end_date=load_end, verbose=False)
    if df.empty:
        return pd.DataFrame()
    df = df.loc[df["port"].isin(batch.AIR_PORTS)].copy()
    if df.empty:
        return pd.DataFrame()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True).dt.tz_localize(None)
    df["mole_fraction"] = pd.to_numeric(df["mole_fraction"], errors="coerce")
    df["rejected"] = pd.to_numeric(df["rejected"], errors="coerce").fillna(0).astype(int)

    df = df.loc[df["mole_fraction"].notna() & df["rejected"].eq(0)]
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("analysis_datetime").reset_index(drop=True)

    result = detect_air_excursions(
        df["analysis_datetime"].to_numpy(),
        df["mole_fraction"].to_numpy(dtype=float),
        median_window_days, mad_window_days, min_points, sigma,
    )
    for key, arr in result.items():
        df[key] = arr
    df["upper_bound"] = df["baseline"] + sigma * df["scale"]
    df["lower_bound"] = df["baseline"] - sigma * df["scale"]

    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end) if end else pd.Timestamp.now()
    df = df.loc[df["analysis_datetime"].between(requested_start, requested_end)].reset_index(drop=True)
    if df.empty:
        return df

    df["block_id"] = -1
    flagged_times = df.loc[df["outlier"], "analysis_datetime"]
    if not flagged_times.empty:
        blocks = _group_periods(flagged_times, max_gap_hours=max_gap_hours)
        for bid, (bstart, bend) in enumerate(blocks):
            in_block = df["outlier"] & df["analysis_datetime"].between(bstart, bend)
            df.loc[in_block, "block_id"] = bid

    if min_block_hours > 0 and (df["block_id"] >= 0).any():
        span_hours = (
            df.loc[df["block_id"] >= 0]
            .groupby("block_id")["analysis_datetime"]
            .agg(lambda s: (s.max() - s.min()) / pd.Timedelta(hours=1))
        )
        short_blocks = span_hours.loc[span_hours < min_block_hours].index
        drop = df["block_id"].isin(short_blocks)
        df.loc[drop, "outlier"] = False
        df.loc[drop, "block_id"] = -1

    return df


def plot_air_tags(df: pd.DataFrame, gas: str, channel: str, site: str, sigma: float, output_jpg: Path) -> None:
    """Timeseries-style figure: green (site-shade) points for untagged data,
    red-edged points for identified excursions (mec='red', same convention
    logos_timeseries.py uses for already-rejected points), plus the rolling
    baseline and +/-sigma band for visual QC of the threshold itself.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    port_style = {4: ("Air1", "mediumseagreen"), 8: ("Air2", "darkgreen")}
    for port, (label, color) in port_style.items():
        sub = df.loc[df["port"] == port]
        if sub.empty:
            continue
        clean = sub.loc[~sub["outlier"]]
        bad = sub.loc[sub["outlier"]]
        if not clean.empty:
            ax.plot(clean["analysis_datetime"], clean["mole_fraction"], marker="o", linestyle="",
                     markersize=3, color=color, alpha=0.5, label=f"{label} (untagged)")
        if not bad.empty:
            ax.plot(bad["analysis_datetime"], bad["mole_fraction"], marker="o", linestyle="",
                     markersize=4.5, markerfacecolor=color, markeredgecolor="red",
                     markeredgewidth=1.4, alpha=1.0, label=f"{label} (excursion)")

    ordered = df.sort_values("analysis_datetime")
    ax.plot(ordered["analysis_datetime"], ordered["baseline"], color="black",
             linewidth=1.0, alpha=0.7, label="rolling median baseline")
    ax.fill_between(ordered["analysis_datetime"], ordered["lower_bound"], ordered["upper_bound"],
                     color="gray", alpha=0.15, label=f"+/-{sigma:g} sigma band")

    ax.set_xlabel("Sample datetime")
    ax.set_ylabel("Mole fraction")
    ax.set_title(f"{site.upper()} {gas} ({channel}) -- air excursion detection")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(output_jpg, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", default="brw")
    p.add_argument("--gas", required=True, help="Gas_channel, e.g. N2O_q")
    p.add_argument("--start", type=_parse_yyyymmdd, default="1998-01-01",
                    help="Start date, YYYYMMDD (default: instrument start)")
    p.add_argument("--end", type=_parse_yyyymmdd, default=None,
                    help="End date, YYYYMMDD (default: now)")
    p.add_argument("--median-window-days", type=float, default=60.0,
                    help="Rolling-median baseline window width, centered on each point "
                         "(default: 60). Should be at least ~2x the longest excursion "
                         "you're hunting -- the median is only robust to an excursion "
                         "that stays a minority of the window's points.")
    p.add_argument("--mad-window-days", type=float, default=60.0,
                    help="Rolling-MAD robust-scale window width on the detrended "
                         "residual (default: 60; independent of --median-window-days)")
    p.add_argument("--min-points", type=int, default=8,
                    help="Minimum neighbours (excluding self) in a window to trust it "
                         "(default: 8)")
    p.add_argument("--sigma", type=float, default=4.0,
                    help="Flag a point when |residual| exceeds this many robust-scale "
                         "units (default: 4)")
    p.add_argument("--max-gap-hours", type=float, default=48.0,
                    help="Max gap between flagged points to merge into one excursion "
                         "block (default: 48)")
    p.add_argument("--min-block-hours", type=float, default=0.0,
                    help="Drop excursion blocks spanning less than this many hours; 0 "
                         "keeps every flagged point including isolated singletons, "
                         "useful for a first unfiltered look while tuning (default: 0)")
    p.add_argument("--output", type=Path, required=True,
                    help="Output base path -- writes <base>.csv and <base>.jpg "
                         "(any given suffix is replaced)")
    args = p.parse_args()

    gas, channel = args.gas.rsplit("_", 1)
    batch = CATS_batch(args.site)
    try:
        pnum = _resolve_pnum(batch, gas, channel)
    except ValueError as exc:
        print(exc)
        return 1

    print(f"Scanning {gas} ({channel}) pnum={pnum} at {args.site} "
          f"{args.start} -> {args.end or 'now'} ...")
    df = build_air_tags(
        batch, pnum, channel, args.start, args.end,
        median_window_days=args.median_window_days, mad_window_days=args.mad_window_days,
        min_points=args.min_points, sigma=args.sigma,
        max_gap_hours=args.max_gap_hours, min_block_hours=args.min_block_hours,
    )
    if df.empty:
        print(f"No air data for {gas} ({channel}) at {args.site} in the requested range.")
        return 0

    out_base = args.output.with_suffix("")
    csv_path = out_base.with_suffix(".csv")
    jpg_path = out_base.with_suffix(".jpg")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    flagged = df.loc[df["outlier"]].copy()
    flagged["gas"] = gas
    flagged["channel"] = channel
    flagged["pnum"] = pnum
    cols = ["mf_num", "analysis_datetime", "port", "mole_fraction", "baseline",
            "residual", "scale", "z", "block_id", "gas", "channel", "pnum"]
    cols = [c for c in cols if c in flagged.columns]
    flagged[cols].to_csv(csv_path, index=False, float_format="%.10g")

    n_blocks = flagged["block_id"].nunique() if not flagged.empty else 0
    print(f"Flagged {len(flagged):,} of {len(df):,} points in {n_blocks} block(s)")
    print(f"Wrote {csv_path}")

    plot_air_tags(df, gas, channel, args.site, args.sigma, jpg_path)
    print(f"Wrote {jpg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
