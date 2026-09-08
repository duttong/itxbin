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
   is already a valid cats_set_mf_method.py --start.
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
3. Group flagged periods into episodes (reuses cats_cal_step_qc._group_periods).
   For each candidate method (cal12/cal1/cal2/ref), build ONE period-level
   mole-fraction series for the WHOLE scan range, once -- not once per
   episode. This is exact, not an approximation: _fit_periods() defines both
   the fit table's week boundaries and the air data's period boundaries, so
   no fit boundary ever falls inside one period, and every injection in a
   period resolves to the same (slope, intercept). Median commutes with any
   per-group-constant affine map, so aggregating raw response to a period
   median first (_period_air_response) and then applying that period's own
   fit (_period_mole_fraction) reproduces exactly what median-of-per-
   injection-mole-fraction would give, without ever reconstructing every
   injection. This is also what makes the tool fast: update_fits() is called
   at most once per candidate method for the entire scan, not once per
   candidate per episode over heavily-overlapping windows.
4. Score each episode's anchor against every candidate's cached series with
   the same jump statistic as step 2, at a shorter, more local primary
   horizon (--resolve-window-days, default 180) -- this and step 5 below are
   independent defenses against the same failure mode: a genuinely bad,
   separately-flagged stretch elsewhere in the record corrupting an
   unrelated anchor's resolve verdict. The long detection horizon
   (--trend-window-days) is also recomputed and reported per candidate
   (z_{method}_long) as a cross-check, but never gates the decision. The
   first candidate (in --method-preference order, default
   cal12 > cal2 > cal1) whose primary-horizon z clears
   --resolve-z-threshold is the recommendation. If none do, the episode is
   reported UNRESOLVED along with the cal-tank serials active at that date,
   for manual hats.scale_assignments review -- an unresolved jump is left
   flagged, never silently "fixed" by whichever method merely scores
   least-bad.
5. Before scoring, periods belonging to any OTHER flagged episode are
   excluded from a candidate's trend-fit input for THIS episode
   (_exclude_other_episode_periods) -- found necessary after BRW N2O (q)'s
   2019-11-18 UNRESOLVED episode sat inside the 'after' trend window for the
   unrelated 2019-04-22 anchor and corrupted its cal12 resolve verdict, even
   though cal12 was in fact fine there (confirmed by manually forcing it and
   recomputing).

IMPORTANT calc contract: _fits_slope_intercept_for_periods forces intercept
to 0 whenever the candidate is not cal12, mirroring
CATS_batch.calc_mole_fraction_from_fits's np.where(is_cal12,
slope*x+intercept, slope*x) branch exactly -- cal1/cal2 (force-through-
origin, single-tank) must never leak an intercept. A period with no
covering fit gets NaN slope, which propagates to a non-finite median_mf and
is dropped by _period_mole_fraction -- the same "not enough information"
outcome as an empty fits table did in the old per-injection path, just
reached by NaN propagation instead of an explicit empty-DataFrame check.

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
    key = f"{gas} ({channel})".lower()
    lookup = {k.lower(): v for k, v in batch.analytes.items()}
    pnum = lookup.get(key)
    if pnum is None:
        raise ValueError(
            f"No analyte_list entry for {gas!r} ({channel!r}) at "
            f"{batch.inst_id} site {batch.site}"
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


def _period_air_response(
    period_start: pd.Series,
    normalized_resp: pd.Series,
    rejected: pd.Series,
    min_points: int = 4,
) -> pd.DataFrame:
    """Aggregate AIR-port normalized_resp to one row per period: median
    response and count. Structurally identical to _period_medians (same
    rejected/notna filter, same min_points drop, same period_mid
    convention) but aggregates raw response instead of an already-computed
    mole_fraction -- the period-level input to _period_mole_fraction, which
    applies a period's own cal fit to this median response directly instead
    of reconstructing every injection's mole fraction first and only then
    taking a median. See module docstring for why this is exact (median
    commutes with the per-period-constant affine fit), not an
    approximation, given _fit_periods()'s boundary contract.
    """
    df = pd.DataFrame({
        "period_start": pd.to_datetime(period_start).to_numpy(),
        "normalized_resp": pd.to_numeric(normalized_resp, errors="coerce").to_numpy(),
        "rejected": pd.to_numeric(rejected, errors="coerce").fillna(0).to_numpy(),
    })
    df = df.loc[(df["rejected"] == 0) & df["normalized_resp"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["period_start", "period_mid", "median_resp", "n"])

    grouped = df.groupby("period_start")["normalized_resp"].agg(median_resp="median", n="count")
    grouped = grouped.loc[grouped["n"] >= min_points].reset_index()
    grouped["period_mid"] = grouped["period_start"] + pd.Timedelta(days=3.5)
    return grouped.sort_values("period_start").reset_index(drop=True)[
        ["period_start", "period_mid", "median_resp", "n"]
    ]


def _period_mole_fraction(
    period_resp: pd.DataFrame, slope: np.ndarray, intercept: np.ndarray,
) -> pd.DataFrame:
    """median_mf = slope*median_resp + intercept, elementwise, aligned to
    period_resp's row order. The period-level analogue of
    CATS_batch.calc_mole_fraction_from_fits's per-injection arithmetic
    (slope*resp+intercept for cal12, slope*resp for cal1/cal2/ref -- pass
    an all-zero intercept for those), applied once per period instead of
    once per injection. NaN slope (no fit covers that period -- see
    _fits_slope_intercept_for_periods) propagates to a non-finite
    median_mf and is dropped here, the same "not enough context" outcome
    as today's per-injection path leaving those rows out of the median.
    Also drops any other non-finite result, mirroring
    _null_mole_fraction_for_zero_height's cross-contamination catch-all
    (a reference-port zero-height row corrupting normalized_resp on other
    rows produces inf/-inf mole fractions there too).
    """
    slope = np.asarray(slope, dtype=float)
    intercept = np.asarray(intercept, dtype=float)
    median_mf = slope * period_resp["median_resp"].to_numpy() + intercept
    out = period_resp.assign(median_mf=median_mf)
    out = out.loc[np.isfinite(out["median_mf"])]
    return out.reset_index(drop=True)[["period_start", "period_mid", "median_mf", "n"]]


def _fits_slope_intercept_for_periods(
    period_start: pd.Series, fits: pd.DataFrame, is_cal12: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward-asof each period_start against fits['week_start'] (same
    _fit_periods() tz-naive boundary convention on both sides -- do not
    re-localize either side) to resolve the (slope, intercept) in effect
    for that period -- the period-level analogue of
    calc_mole_fraction_from_fits's per-injection merge_asof. intercept is
    forced to 0 when is_cal12 is False, mirroring
    calc_mole_fraction_from_fits's np.where(is_cal12, slope*x+intercept,
    slope*x) branch exactly (see the module's "IMPORTANT calc contract"
    note) -- cal1/cal2 must never leak an intercept.

    Returns (slope, intercept, keep), all aligned to period_start's
    original order. keep[i] is False wherever no fits row has
    week_start <= period_start[i] -- not enough data to fit this candidate
    for that period, same "not enough context" convention used elsewhere.
    """
    n = len(period_start)
    if fits is None or fits.empty:
        return np.full(n, np.nan), np.zeros(n), np.zeros(n, dtype=bool)

    ps = pd.DataFrame({
        "period_start": pd.to_datetime(period_start).to_numpy(),
        "_order": np.arange(n),
    }).sort_values("period_start")

    fit_table = fits[["week_start", "slope", "intercept"]].copy()
    fit_table["week_start"] = pd.to_datetime(fit_table["week_start"])
    fit_table = fit_table.sort_values("week_start")

    merged = pd.merge_asof(
        ps, fit_table, left_on="period_start", right_on="week_start", direction="backward",
    ).sort_values("_order")

    slope = merged["slope"].to_numpy(dtype=float)
    keep = np.isfinite(slope)
    intercept = np.where(keep, merged["intercept"].to_numpy(dtype=float), 0.0)
    if not is_cal12:
        intercept = np.zeros(n)
    slope = np.where(keep, slope, np.nan)
    return slope, intercept, keep


def _exclude_other_episode_periods(
    periods: pd.DataFrame,
    own_episode: tuple[pd.Timestamp, pd.Timestamp],
    all_episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """Drop periods belonging to any OTHER flagged episode from `periods`,
    keeping periods in own_episode and periods not in any episode.
    Root-cause fix for cross-episode contamination: a separate,
    independently-bad stretch (already its own flagged episode) sitting
    inside another anchor's trend-fit window can corrupt that anchor's
    resolve-time z-score even though the two are unrelated -- e.g. BRW N2O
    (q)'s 2019-11-18 UNRESOLVED episode sat inside the 'after' trend window
    for the unrelated 2019-04-22 anchor, making a perfectly good cal12
    transition look unresolved. This strips that stretch's periods out of
    the trend-fit INPUT for every OTHER episode's evaluation, rather than
    trying to raise thresholds or shrink windows around it.
    """
    other = [ep for ep in all_episodes if ep != own_episode]
    if not other:
        return periods
    in_other = pd.Series(False, index=periods.index)
    for pstart, pend in other:
        in_other |= periods["period_start"].between(pstart, pend)
    return periods.loc[~in_other].reset_index(drop=True)


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


def _ref_slope_for_periods(batch: CATS_batch, pnum: int, period_mid: pd.Series) -> np.ndarray:
    """Per-period ref-tank coef0 (the slope in mf = coef0*resp for method
    'ref'), mirroring CATS_Instrument.calc_mole_fraction_scale_simple's
    real per-row resolution -- tank_serials_for_dates() to find the tank
    installed on STANDARD_PORT_NUM at each date, then
    scale_assignment_values_for_dates() (cached, no per-row DB query) for
    that tank's coef0 -- just evaluated once per period_mid instead of
    once per injection. A faithful period-level mirror of the real,
    already-correct method, not a new lookup.
    """
    dates = pd.to_datetime(pd.Series(period_mid).reset_index(drop=True), errors="coerce", utc=True)
    coef0 = pd.Series(np.nan, index=dates.index, dtype="float64")
    serials = batch.tank_serials_for_dates(batch.STANDARD_PORT_NUM, dates)
    for serial in serials.dropna().unique():
        mask = serials.eq(serial)
        coef0.loc[mask] = batch.scale_assignment_values_for_dates(
            serial, pnum, dates.loc[mask], key="coef0"
        )
    return coef0.to_numpy()


def _load_period_air_response(
    batch: CATS_batch, pnum: int, channel: str, start: str, end: str, min_period_points: int,
) -> pd.DataFrame:
    """load_data -> filter to AIR_PORTS & non-zero-height rows -> _fit_periods
    -> _period_air_response. Called ONCE per gas per scan range, shared by
    every candidate method: raw response doesn't depend on which method is
    being tested, only the fit applied to it does.
    """
    df = batch.load_data(pnum, channel=channel, start_date=start, end_date=end, verbose=False)
    if df.empty:
        return pd.DataFrame()
    df = df.loc[df["port"].isin(batch.AIR_PORTS)].copy()
    if "height" in df.columns:
        # Mirrors _null_mole_fraction_for_zero_height's direct case (own
        # height==0 -> not a real measurement); the cross-contamination
        # case (a reference-port zero-height corrupting OTHER rows'
        # normalized_resp into inf/-inf) is caught downstream by
        # _period_mole_fraction's finite-result filter instead.
        df = df.loc[pd.to_numeric(df["height"], errors="coerce").fillna(0) != 0]
    if df.empty:
        return pd.DataFrame()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True).dt.tz_localize(None)
    df["period_start"] = batch._fit_periods(df)
    return _period_air_response(
        df["period_start"], df["normalized_resp"], df["rejected"], min_points=min_period_points
    )


def _build_candidate_series(
    batch: CATS_batch, pnum: int, channel: str, method_num: int,
    period_resp: pd.DataFrame, load_start: str, load_end: str,
) -> pd.DataFrame:
    """One candidate method's full-range period-level mole-fraction series,
    built once and shared across every episode's evaluation -- versus the
    old per-episode-window recompute, this calls update_fits at most once
    per candidate for the WHOLE scan, not once per episode. ref has no
    weekly fit -- resolved directly via _ref_slope_for_periods. cal1/cal2/
    cal12 share one update_fits(method_override=method_num) call over the
    full padded range.
    """
    if period_resp.empty:
        return pd.DataFrame()

    if method_num == METHOD_NAME_TO_NUM["ref"]:
        slope = _ref_slope_for_periods(batch, pnum, period_resp["period_mid"])
        intercept = np.zeros(len(period_resp))
        return _period_mole_fraction(period_resp, slope, intercept)

    fits, _scale_num, _ref_serial, _channel_str = batch.update_fits(
        pnum, channel=channel, start_date=load_start, end_date=load_end,
        method_override=method_num, verbose=False,
    )
    is_cal12 = method_num == METHOD_NAME_TO_NUM["cal12"]
    slope, intercept, _keep = _fits_slope_intercept_for_periods(
        period_resp["period_start"], fits, is_cal12,
    )
    return _period_mole_fraction(period_resp, slope, intercept)


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
    # Group on period_start, not period_mid: episode_start/episode_end need
    # to be real week boundaries (valid cats_set_mf_method.py --start
    # values), not the +3.5-day midpoints used for the trend-fit x
    # coordinate above. The two series have identical relative spacing (a
    # constant offset apart), so this changes nothing about which periods
    # get grouped together -- only what the returned boundary values mean.
    episodes = _group_periods(flagged["period_start"], max_gap_hours=max_gap_days * 24.0)
    return periods, episodes


def _evaluate_episode(
    batch: CATS_batch, pnum: int,
    candidate_series: dict[str, pd.DataFrame],
    periods: pd.DataFrame, episodes: list[tuple[pd.Timestamp, pd.Timestamp]],
    episode: tuple[pd.Timestamp, pd.Timestamp],
    window_days: float, gap_days: float, resolve_window_days: float,
    min_trend_points: int, resolve_z_threshold: float, method_preference: list[str],
) -> dict:
    """Score every candidate at this episode's anchor, then recommend
    whichever clears resolve_z_threshold with the SMALLEST |z| (ties broken
    by method_preference order) -- not the first candidate in preference
    order to merely clear the bar. No DB calls here -- candidate_series was
    built once for the whole scan by build_cal_method_qc, shared across
    every episode.

    A first-to-clear rule lets a structurally weaker candidate (e.g. cal1's
    single-tank, forced-through-origin fit, versus cal12's 2-point fit) win
    outright just by narrowly ducking under the same bar a much closer-
    fitting candidate barely missed -- a weak fit's trend line is looser and
    absorbs a real step rather than seeing it, so "resolves the jump" can
    mean "can't detect the jump" rather than "is the right method". Picking
    the smallest |z| instead still requires clearing the bar (a candidate
    that doesn't isn't a candidate), but among those that do, prefers
    whichever actually flattens the discontinuity most, with
    method_preference only breaking ties.

    Periods belonging to any OTHER flagged episode are excluded from each
    candidate's trend-fit input first (_exclude_other_episode_periods), so
    a separate, independently-bad stretch can't corrupt this anchor's
    verdict. resolve_window_days (short, local) gates the recommendation;
    window_days (the long detection horizon) is recomputed and reported as
    a cross-check (z_{method}_long) but never gates -- deliberately the
    smallest change to today's single-horizon decision semantics.
    """
    pstart, pend = episode
    in_episode = periods.loc[periods["period_start"].between(pstart, pend)]
    # _group_periods() built this episode from flagged (z non-null) periods,
    # so a scored row is always present; the anchor is whichever one deviated
    # most from its own local trend.
    scored = in_episode.loc[in_episode["z"].notna()]
    anchor = scored.loc[scored["z"].abs().idxmax()] if not scored.empty else in_episode.iloc[0]

    result = {
        "episode_start": pstart, "episode_end": pend,
        "anchor_period_start": anchor["period_start"], "anchor_period_mid": anchor["period_mid"],
        "current_method": anchor.get("current_method"),
        "detected_jump": anchor["jump"], "detected_z": anchor["z"],
        "recommendation": "UNRESOLVED",
    }

    # Score every candidate first (also populates the reported jump_*/z_*
    # columns for every method regardless of who wins), then pick whichever
    # cleared resolve_z_threshold with the smallest |z| -- method_preference
    # only breaks exact ties, it no longer short-circuits the search.
    candidates_cleared = []
    for method_name in method_preference:
        series = candidate_series.get(method_name, pd.DataFrame())
        if series.empty:
            result[f"jump_{method_name}"] = np.nan
            result[f"z_{method_name}"] = np.nan
            result[f"z_{method_name}_long"] = np.nan
            continue

        local = _exclude_other_episode_periods(series, episode, episodes)

        jump_s, z_s, *_ = _local_level_jump(
            local["period_mid"].to_numpy(), local["median_mf"].to_numpy(),
            resolve_window_days, gap_days, min_trend_points,
        )
        jump_l, z_l, *_ = _local_level_jump(
            local["period_mid"].to_numpy(), local["median_mf"].to_numpy(),
            window_days, gap_days, min_trend_points,
        )
        at = local["period_start"].eq(anchor["period_start"]).to_numpy()
        z_here = float(z_s[at][0]) if at.any() and np.isfinite(z_s[at][0]) else np.nan
        jump_here = float(jump_s[at][0]) if at.any() and np.isfinite(jump_s[at][0]) else np.nan
        z_long_here = float(z_l[at][0]) if at.any() and np.isfinite(z_l[at][0]) else np.nan

        result[f"jump_{method_name}"] = jump_here
        result[f"z_{method_name}"] = z_here
        result[f"z_{method_name}_long"] = z_long_here
        if np.isfinite(z_here) and abs(z_here) <= resolve_z_threshold:
            candidates_cleared.append((abs(z_here), method_name))

    recommendation = min(candidates_cleared)[1] if candidates_cleared else None

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
    resolve_window_days: float = 180.0,
    min_trend_points: int = 12,
    min_period_points: int = 4,
    jump_z_threshold: float = 4.0,
    resolve_z_threshold: float | None = None,
    min_jump: float = 0.0,
    max_gap_days: float = 21.0,
    method_preference: str = "cal12,cal2,cal1",
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

    # One air-response load and one full-range series per candidate METHOD
    # for the whole scan -- not per episode. update_fits() is called at
    # most len(set(prefs))-1 times total here (ref needs none), versus
    # today's episodes x len(prefs) calls over heavily-overlapping windows.
    pad_days = max(window_days, resolve_window_days) + gap_days
    load_start = (pd.Timestamp(start) - pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")
    load_end = (pd.Timestamp(end) + pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")

    period_resp = _load_period_air_response(
        batch, pnum, channel, load_start, load_end, min_period_points,
    )
    candidate_series = {
        name: _build_candidate_series(
            batch, pnum, channel, METHOD_NAME_TO_NUM[name], period_resp, load_start, load_end,
        )
        for name in set(prefs)
    }

    rows = [
        _evaluate_episode(
            batch, pnum, candidate_series, periods, episodes, episode,
            window_days, gap_days, resolve_window_days, min_trend_points,
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
    p.add_argument("--resolve-window-days", type=float, default=180.0,
                    help="Primary (shorter, more local) trend-window width used only when "
                         "resolving a flagged episode's candidate methods -- less likely to "
                         "reach into an unrelated bad stretch than --trend-window-days. "
                         "--trend-window-days is reused as a reported (non-gating) cross-check "
                         "at resolution time (z_{method}_long columns) (default: 180)")
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
    p.add_argument("--method-preference", type=str, default="cal12,cal2,cal1",
                    help="Comma-separated method names, most-preferred first. "
                         "'ref' is omitted by default -- pass it explicitly "
                         "(e.g. 'cal12,cal2,cal1,ref') to consider it again "
                         "(default: cal12,cal2,cal1)")
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
            resolve_window_days=args.resolve_window_days,
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
