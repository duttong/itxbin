#!/usr/bin/env python3
"""Recommend a per-period CATS calibration method (ref/cal1/cal2/cal12) that
removes non-atmospheric discontinuities from the mole-fraction record.

Detect + recommend ONLY -- this tool never writes to the database. Unlike
cal_step/baseline/cal_window it is NOT registered in cats_tagging.py's
ALGORITHMS: it produces calibration-method recommendations, not
ng_insitu_mole_fraction_tags rows, a fundamentally different kind of output.
No --dry-run flag exists because there is nothing to preview a write for.

Background: switching a species to cal12 (weekly 2-point fit through both
cal tanks) is usually the best choice -- it captures both detector gain and
offset -- but isn't always achievable across a multi-decade record. A period
where one cal tank's response is too noisy/sparse to fit reliably, or where
the tanks were simply run less consistently in an earlier era, can produce a
cal12 fit that's worse than a simpler method for that stretch, showing up as
a visible step in the mole-fraction time series where none of the parent
methods (ref/cal1/cal2/cal12) actually failed -- the WRONG one was just
chosen for that period. Real SF6/N2O have only a slow secular trend plus a
seasonal cycle; they never step. So a level discontinuity in the computed
series, found near a period boundary, is treated as evidence the assigned
method for that period is wrong -- unless NO candidate method resolves it,
in which case the culprit is more likely a bad hats.scale_assignments entry
for one of the cal tanks (see UNRESOLVED handling below), which no method
choice can fix.

Algorithm
---------
1. Load the persisted mole_fraction series (whatever methods are currently
   recorded), aggregate to CATS_batch._fit_periods() periods (normally
   calendar weeks, split at a mid-week cal-tank swap) -- the same boundary
   granularity update_fits() itself fits on, so a recommended period_start
   is already a valid cats_set_mf_method.py --start-date.
2. Score every period against a two-sided, DETRENDED local jump statistic
   (_local_level_jump): fit independent robust (Theil-Sen) trend lines to
   the window before and after the candidate, each excluding a gap around it
   so the candidate's own value can't contaminate its own comparison, then
   compare the two lines' extrapolated level AT the candidate. This is
   deliberately not point-to-point differencing (see cats_cal_step_qc.py) --
   that assumes local stationarity, which is wrong here: SF6/N2O trend
   secularly and cycle seasonally, and a naive diff would flag the trend
   itself. The ~1-year trend window instead averages the seasonal cycle out
   of each side's slope estimate, and the residual scale (median absolute
   deviation from each side's own fit) absorbs whatever seasonal wiggle a
   straight line doesn't capture -- so the z-score threshold is
   self-calibrating per analyte/era rather than a hand-tuned absolute cutoff
   or a separate STL seasonal decomposition.
3. Group flagged periods into episodes (reuses cats_cal_step_qc._group_periods)
   and, for each episode, recompute the surrounding window under every
   candidate method via CATS_batch.update_fits/update_runs(method_override=M)
   -- a non-mutating "try a method" harness that already exists for this
   exact purpose. Score each candidate the same way as step 2; the first
   candidate (in --method-preference order, default cal12 > cal2 > cal1 >
   ref) whose recomputed jump clears the same threshold is the
   recommendation. If none do, the episode is reported UNRESOLVED along with
   the cal-tank serials active at that date, for manual hats.scale_assignments
   review -- an unresolved jump is left flagged, never silently "fixed" by
   whichever method merely scores least-bad.

IMPORTANT calc contract: when recomputing under a forced method, ALWAYS pass
fits_override=fits -- the literal (possibly empty) return value of
update_fits() -- into update_runs(), never convert an empty fits DataFrame
to None. update_runs() branches on `fits_override is None`, not `.empty`:
passing None makes it fall back to calc_mole_fraction(), which reads
whatever fit is CURRENTLY PERSISTED in hats.ng_response for fit-based rows
-- silently testing stale, already-on-disk data instead of this window's
freshly-forced candidate. An empty fits table for a ref candidate (ref
stores no fit at all) is handled correctly by calc_mole_fraction_from_fits's
direct_mask branch; an empty fits table for a fit-based candidate (e.g. not
enough tank data in this window to compute cal2) correctly leaves those rows
NaN rather than reusing an unrelated stale fit.

v1 scope: only validated against CATS-BRW N2O and SF6 (channel q) -- the
record the user already knows has visible artifacts. See CATS_QC_TODO.md
before trusting recommendations on other analytes/sites/eras.

Usage::

    python3 cats_cal_method_qc.py --site brw --gas SF6_q --start 19980101 -v
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cats_batch import CATS_batch
from cats_cal_step_qc import _group_periods

# v1 scope only -- see module docstring.
ALL_GASES = ("N2O_q", "SF6_q")

METHOD_NAME_TO_NUM = {"ref": 1, "cal12": 2, "cal1": 3, "cal2": 4}


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


def _period_medians(
    period_start: pd.Series,
    mole_fraction: pd.Series,
    rejected: pd.Series,
    min_points: int = 4,
) -> pd.DataFrame:
    """Aggregate rows to one row per period: median mole_fraction and count.

    Rejected rows are excluded before aggregating. Periods with fewer than
    min_points surviving rows are dropped -- not enough to trust a period
    median, same "not enough context to judge" convention used throughout
    cats_qc/. period_mid is period_start + 3.5 days (the same week-midpoint
    convention used by ie3_cal_test.py's plotting helpers), used as the x
    coordinate for the trend fits in _local_level_jump.
    """
    df = pd.DataFrame({
        "period_start": pd.to_datetime(period_start).to_numpy(),
        "mole_fraction": pd.to_numeric(mole_fraction, errors="coerce").to_numpy(),
        "rejected": pd.to_numeric(rejected, errors="coerce").fillna(0).to_numpy(),
    })
    df = df.loc[(df["rejected"] == 0) & df["mole_fraction"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["period_start", "period_mid", "median_mf", "n"])

    grouped = df.groupby("period_start")["mole_fraction"].agg(median_mf="median", n="count")
    grouped = grouped.loc[grouped["n"] >= min_points].reset_index()
    grouped["period_mid"] = grouped["period_start"] + pd.Timedelta(days=3.5)
    return grouped.sort_values("period_start").reset_index(drop=True)[
        ["period_start", "period_mid", "median_mf", "n"]
    ]


def _local_level_jump(
    period_mid: np.ndarray,
    value: np.ndarray,
    window_days: float,
    gap_days: float,
    min_trend_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-sided, gap-excluded, detrended jump statistic at every point.

    period_mid must already be sorted ascending (datetime64[ns] or
    comparable); value is the matching series (e.g. median_mf). For each
    index i, independently Theil-Sen-fits the points in
    [t_i - window_days - gap_days, t_i - gap_days) ("before") and
    (t_i + gap_days, t_i + window_days + gap_days] ("after"), extrapolates
    each line to t_i, and takes level_after - level_before as the jump.
    Excluding the gap around t_i keeps the candidate's own value (and its
    immediate neighbors, which may already be drifting toward a real step)
    from contaminating either side's trend estimate.

    scale is 1.4826 * median(|residual|) pooled over both sides' own fits --
    a robust "how much does this analyte's mole fraction normally wiggle
    around a straight local trend" (residual absorbs whatever seasonal cycle
    a linear fit doesn't capture, deliberately, so a species with a bigger
    seasonal cycle gets a proportionally larger, self-calibrated threshold
    rather than a fixed absolute cutoff). z = jump / scale.

    Returns (jump, z, scale, n_before, n_after), each aligned to period_mid.
    A side with fewer than min_trend_points points gets NaN/0 in every
    output for that row -- never flagged, matching the "not enough context"
    convention used elsewhere in cats_qc/.
    """
    n = len(period_mid)
    jump = np.full(n, np.nan)
    z = np.full(n, np.nan)
    scale = np.full(n, np.nan)
    n_before = np.zeros(n, dtype=int)
    n_after = np.zeros(n, dtype=int)

    t = pd.to_datetime(pd.Series(period_mid)).to_numpy()
    t_days = (t - t[0]) / np.timedelta64(1, "D")
    v = np.asarray(value, dtype=float)

    for i in range(n):
        ti = t_days[i]
        before_mask = (t_days >= ti - window_days - gap_days) & (t_days < ti - gap_days)
        after_mask = (t_days > ti + gap_days) & (t_days <= ti + window_days + gap_days)
        n_before[i] = int(before_mask.sum())
        n_after[i] = int(after_mask.sum())
        if n_before[i] < min_trend_points or n_after[i] < min_trend_points:
            continue

        tb, vb = t_days[before_mask], v[before_mask]
        ta, va = t_days[after_mask], v[after_mask]
        if not (np.isfinite(vb).all() and np.isfinite(va).all()):
            continue

        slope_b, intercept_b, *_ = theilslopes(vb, tb)
        slope_a, intercept_a, *_ = theilslopes(va, ta)
        level_before = slope_b * ti + intercept_b
        level_after = slope_a * ti + intercept_a
        jump[i] = level_after - level_before

        resid_b = vb - (slope_b * tb + intercept_b)
        resid_a = va - (slope_a * ta + intercept_a)
        mad = float(np.median(np.abs(np.concatenate([resid_b, resid_a]))))
        s = 1.4826 * mad
        if s > 0:
            scale[i] = s
            z[i] = jump[i] / s

    return jump, z, scale, n_before, n_after


def _load_period_series(
    batch: CATS_batch, pnum: int, channel: str, start: str, end: str, min_period_points: int,
) -> pd.DataFrame:
    """Persisted-mole_fraction path: load_data -> _fit_periods -> _period_medians,
    plus current_method (modal mf_method_num already on each period's air rows)."""
    df = batch.load_data(pnum, channel=channel, start_date=start, end_date=end, verbose=False)
    if df.empty:
        return pd.DataFrame()
    df = df.loc[df["port"].isin(batch.AIR_PORTS)].copy()
    if df.empty:
        return pd.DataFrame()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True).dt.tz_localize(None)
    df["period_start"] = batch._fit_periods(df)

    periods = _period_medians(
        df["period_start"], df["mole_fraction"], df["rejected"], min_points=min_period_points
    )
    if periods.empty:
        return periods

    unrejected = df.loc[df["rejected"].fillna(0).astype(int).eq(0)]
    modal_method = (
        unrejected.groupby("period_start")["mf_method_num"]
        .agg(lambda s: int(pd.Series(s).mode().iat[0]) if not s.mode().empty else np.nan)
    )
    periods["current_method"] = periods["period_start"].map(modal_method)
    return periods


def _recompute_period_series(
    batch: CATS_batch, pnum: int, channel: str, start: str, end: str,
    method_override: int, min_period_points: int,
) -> pd.DataFrame:
    """Force every row in [start, end] to method_override (no DB writes) and
    aggregate the same way as _load_period_series. See module docstring for
    the fits_override=fits (never None) contract this depends on."""
    fits, _scale_num, _ref_serial, _channel_str = batch.update_fits(
        pnum, channel=channel, start_date=start, end_date=end,
        method_override=method_override, verbose=False,
    )
    df = batch.update_runs(
        pnum, channel=channel, start_date=start, end_date=end,
        fits_override=fits, method_override=method_override, verbose=False,
    )
    if df.empty:
        return pd.DataFrame()
    df = df.loc[df["port"].isin(batch.AIR_PORTS)].copy()
    if df.empty:
        return pd.DataFrame()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True).dt.tz_localize(None)
    df["period_start"] = batch._fit_periods(df)
    return _period_medians(
        df["period_start"], df["mole_fraction"], df["rejected"], min_points=min_period_points
    )


def _detect_discontinuities(
    batch: CATS_batch, pnum: int, channel: str, start: str, end: str,
    window_days: float, gap_days: float, min_trend_points: int, min_period_points: int,
    jump_z_threshold: float, min_jump: float, max_gap_days: float,
) -> tuple[pd.DataFrame, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Load the persisted-method period series, score every period, group
    flagged ones into episodes. Returns (scored periods df, episode list)."""
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=window_days + gap_days)).strftime("%Y-%m-%d")
    load_end = (pd.Timestamp(end) + pd.Timedelta(days=window_days + gap_days)).strftime("%Y-%m-%d")

    periods = _load_period_series(batch, pnum, channel, load_start, load_end, min_period_points)
    if periods.empty:
        return periods, []

    jump, z, scale, n_before, n_after = _local_level_jump(
        periods["period_mid"].to_numpy(), periods["median_mf"].to_numpy(),
        window_days, gap_days, min_trend_points,
    )
    periods = periods.assign(jump=jump, z=z, scale=scale, n_before=n_before, n_after=n_after)

    in_range = periods["period_start"].between(pd.Timestamp(start), pd.Timestamp(end))
    flagged = periods.loc[
        in_range & periods["z"].notna() & (periods["z"].abs() > jump_z_threshold)
        & (periods["jump"].abs() >= min_jump)
    ]
    episodes = _group_periods(flagged["period_mid"], max_gap_hours=max_gap_days * 24.0)
    return periods, episodes


def _evaluate_episode(
    batch: CATS_batch, pnum: int, channel: str,
    periods: pd.DataFrame, episode: tuple[pd.Timestamp, pd.Timestamp],
    window_days: float, gap_days: float, min_trend_points: int, min_period_points: int,
    resolve_z_threshold: float, method_preference: list[str],
) -> dict:
    """Recompute the window around one flagged episode under every candidate
    method and pick the first (in preference order) that resolves the jump."""
    pstart, pend = episode
    in_episode = periods.loc[periods["period_mid"].between(pstart, pend)]
    # _group_periods() built this episode from flagged (z non-null) periods,
    # so a scored row is always present; the anchor is whichever one deviated
    # most from its own local trend.
    scored = in_episode.loc[in_episode["z"].notna()]
    anchor = scored.loc[scored["z"].abs().idxmax()] if not scored.empty else in_episode.iloc[0]

    window_start = (pstart - pd.Timedelta(days=window_days + gap_days)).strftime("%Y-%m-%d")
    window_end = (pend + pd.Timedelta(days=window_days + gap_days)).strftime("%Y-%m-%d")

    result = {
        "episode_start": pstart, "episode_end": pend,
        "anchor_period_start": anchor["period_start"], "anchor_period_mid": anchor["period_mid"],
        "current_method": anchor.get("current_method"),
        "detected_jump": anchor["jump"], "detected_z": anchor["z"],
        "recommendation": "UNRESOLVED",
    }

    recommendation = None
    for method_name in method_preference:
        method_num = METHOD_NAME_TO_NUM[method_name]
        candidate = _recompute_period_series(
            batch, pnum, channel, window_start, window_end, method_num, min_period_points,
        )
        if candidate.empty:
            result[f"jump_{method_name}"] = np.nan
            result[f"z_{method_name}"] = np.nan
            continue
        jump, z, _scale, _nb, _na = _local_level_jump(
            candidate["period_mid"].to_numpy(), candidate["median_mf"].to_numpy(),
            window_days, gap_days, min_trend_points,
        )
        candidate = candidate.assign(z=z, jump=jump)
        at_anchor = candidate.loc[candidate["period_start"].eq(anchor["period_start"])]
        z_here = float(at_anchor["z"].iat[0]) if not at_anchor.empty and pd.notna(at_anchor["z"].iat[0]) else np.nan
        jump_here = float(at_anchor["jump"].iat[0]) if not at_anchor.empty and pd.notna(at_anchor["jump"].iat[0]) else np.nan
        result[f"jump_{method_name}"] = jump_here
        result[f"z_{method_name}"] = z_here
        if recommendation is None and np.isfinite(z_here) and abs(z_here) <= resolve_z_threshold:
            recommendation = method_name

    if recommendation is not None:
        result["recommendation"] = recommendation
    else:
        anchor_date = anchor["period_mid"]
        result["cal1_tank"] = batch.tank_serial_for_port(batch.CAL1_PORT, when=anchor_date)
        result["cal2_tank"] = batch.tank_serial_for_port(batch.CAL2_PORT, when=anchor_date)

    return result


def build_cal_method_qc(
    batch: CATS_batch,
    pnum: int,
    channel: str,
    start: str,
    end: str,
    window_days: float = 365.0,
    gap_days: float = 30.0,
    min_trend_points: int = 12,
    min_period_points: int = 4,
    jump_z_threshold: float = 4.0,
    resolve_z_threshold: float | None = None,
    min_jump: float = 0.0,
    max_gap_days: float = 21.0,
    method_preference: str = "cal12,cal2,cal1,ref",
) -> pd.DataFrame:
    """Detect calibration-method-induced discontinuities in [start, end] and
    recommend a method per episode. One row per detected episode; see module
    docstring for the algorithm and cats_qc/README.md for the output columns.
    Never writes to the database.
    """
    if resolve_z_threshold is None:
        resolve_z_threshold = jump_z_threshold
    prefs = [m.strip() for m in method_preference.split(",") if m.strip()]

    periods, episodes = _detect_discontinuities(
        batch, pnum, channel, start, end,
        window_days, gap_days, min_trend_points, min_period_points,
        jump_z_threshold, min_jump, max_gap_days,
    )
    if not episodes:
        return pd.DataFrame()

    rows = [
        _evaluate_episode(
            batch, pnum, channel, periods, episode,
            window_days, gap_days, min_trend_points, min_period_points,
            resolve_z_threshold, prefs,
        )
        for episode in episodes
    ]
    return pd.DataFrame(rows).sort_values("episode_start").reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", default="brw")
    p.add_argument("--gas", default="all", help='Gas_channel (e.g. SF6_q) or "all"')
    p.add_argument("--start", type=_parse_yyyymmdd, default="1998-01-01",
                    help="Start date, YYYYMMDD (default: instrument start).")
    p.add_argument("--end", type=_parse_yyyymmdd, default=None,
                    help="End date, YYYYMMDD (default: now).")
    p.add_argument("--trend-window-days", type=float, default=365.0,
                    help="Days of context per side for the local trend fit (default: 365)")
    p.add_argument("--gap-days", type=float, default=30.0,
                    help="Days excluded around a candidate from both trend fits (default: 30)")
    p.add_argument("--min-trend-points", type=int, default=12,
                    help="Minimum periods per side to trust a trend fit (default: 12)")
    p.add_argument("--min-period-points", type=int, default=4,
                    help="Minimum unrejected air rows to trust a period median (default: 4)")
    p.add_argument("--jump-z-threshold", type=float, default=4.0,
                    help="Detection threshold on the detrended jump z-score (default: 4.0)")
    p.add_argument("--resolve-z-threshold", type=float, default=None,
                    help="Threshold a candidate method's recomputed z must clear to 'resolve' "
                         "an episode (default: same as --jump-z-threshold)")
    p.add_argument("--min-jump", type=float, default=0.0,
                    help="Absolute floor (native units) on top of the z-score threshold; "
                         "0 disables it (default: 0)")
    p.add_argument("--max-gap-days", type=float, default=21.0,
                    help="Max gap between flagged periods to merge into one episode (default: 21)")
    p.add_argument("--method-preference", type=str, default="cal12,cal2,cal1,ref",
                    help="Comma-separated method names, most-preferred first "
                         "(default: cal12,cal2,cal1,ref)")
    p.add_argument("--output", type=Path, default=Path("cats_cal_method_flags.csv"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    end_date = args.end or datetime.today().strftime("%Y-%m-%d")
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
        if args.verbose:
            print(f"Scanning {gas} ({channel}) pnum={pnum} {args.start} -> {end_date} ...")
        out = build_cal_method_qc(
            batch, pnum, channel, args.start, end_date,
            window_days=args.trend_window_days, gap_days=args.gap_days,
            min_trend_points=args.min_trend_points, min_period_points=args.min_period_points,
            jump_z_threshold=args.jump_z_threshold, resolve_z_threshold=args.resolve_z_threshold,
            min_jump=args.min_jump, max_gap_days=args.max_gap_days,
            method_preference=args.method_preference,
        )
        if out.empty:
            print(f"  {gas} ({channel}): no discontinuities flagged.")
            continue
        out["gas"] = gas
        out["channel"] = channel
        out["pnum"] = pnum
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.10g")
    print(f"Wrote {len(result):,} episode(s) to {args.output}")
    if not result.empty:
        cols = ["gas", "channel", "episode_start", "episode_end", "current_method",
                "detected_jump", "detected_z", "recommendation"]
        print(result[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
