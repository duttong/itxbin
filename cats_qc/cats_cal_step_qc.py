#!/usr/bin/env python3
"""Flag CATS injections around rapid, sustained cal-response step changes.

Detects abrupt GC response shifts (detector/valve glitches, not real air
mole-fraction changes) using only the two calibration ports, whose assigned
values are fixed and known -- so a response change there can only be the
instrument, never the atmosphere. For each cal port (CAL1_PORT, CAL2_PORT)
independently, this looks at that port's own chronological series with the
least possible smoothing: the raw point-to-point step in log(response)
between one injection and the next. Each step is judged against that port's
own typical step size (a robust median-absolute-deviation z-score), so
"quickly" and "different than the recent median" are both expressed by the
same robust statistic rather than a hardcoded absolute threshold.

Whenever either cal port's step exceeds the threshold, the two cal ports'
flagged timestamps are merged into contiguous periods (bridging gaps under
--max-gap-hours), then each period is padded out to the nearest cal-port
reading (either port) just outside each end -- so the two cal points
bordering a flagged episode are swept in too, not just the ones that
individually crossed the threshold. Every row on *any* port (2/4/6/8) whose
analysis_datetime falls inside the padded period, inclusive of both
endpoints, is flagged -- this sweeps in the interleaved air1/air2 (4/8)
readings nearest the flagged cal points along with the cal points
themselves, without ever looking at the air data's own values. For an
isolated single-point flag this turns a single row into a 5-row window: the
neighboring cal reading before, the neighboring air reading before, the
flagged point, the neighboring air reading after, and the neighboring cal
reading after.

Worked example (matches Geoff's manual read of BRW N2O (q), April 1999):

    python3 cats_cal_step_qc.py --site brw --gas N2O_q \\
        --start 1999-04-01 --end 1999-04-20

flags all four ports from 1999-04-09 20:43 to 1999-04-10 14:46 inclusive
(the core rate-outlier episode runs 21:43 to 13:46; the extra 20:43 and
14:46 cal readings, and the air readings adjacent to them, are the
neighboring-cal padding described above).

Usage::

    python3 cats_cal_step_qc.py --site brw --gas N2O_q --start 1999-01-01 --end 1999-12-31
    python3 cats_cal_step_qc.py --site brw --gas all --start 1999-01-01 --end 1999-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cats_batch import CATS_batch

# CATS_GCwerks2DB.UPLOAD_MOLS, one channel each (the channel CATS actually
# reports that analyte on -- q for N2O/SF6, f for the halocarbon solvents).
ALL_GASES = (
    "N2O_q", "SF6_q", "CFC12_f", "CFC11_f", "CFC113_f",
    "H1211_f", "CCl4_f", "CH3CCl3_f", "CHCl3_f",
)


def _resolve_pnum(batch: CATS_batch, gas: str, channel: str) -> int:
    key = f"{gas} ({channel})"
    pnum = batch.analytes.get(key)
    if pnum is None:
        raise ValueError(
            f"No analyte_list entry for {key!r} at {batch.inst_id} site {batch.site}"
        )
    return int(pnum)


def _rolling_mad_scale(
    ptp: pd.Series, times: pd.Series, window_days: float, min_periods: int
) -> tuple[pd.Series, pd.Series]:
    """Trailing time-windowed median and 1.4826*MAD of ptp.

    Using a fixed local window (rather than every point in whatever date
    range happens to be queried) keeps "typical" meaning locally typical --
    a detector's response noise plausibly drifts over a multi-decade record,
    and results shouldn't change just because a caller queried a year instead
    of a month. Points before min_periods observations have accumulated (e.g.
    the first few days of the record, or right after a long data gap) get
    scale=NaN and so are never flagged.
    """
    s = pd.Series(ptp.to_numpy(), index=pd.DatetimeIndex(times))
    window = f"{window_days}D"
    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    scale = 1.4826 * mad
    return med.reset_index(drop=True), scale.reset_index(drop=True)


def _port_rate_outliers(
    sub: pd.DataFrame,
    mad_multiplier: float,
    secondary_multiplier: float | None = None,
    scale_window_days: float = 30.0,
    scale_min_periods: int = 20,
) -> pd.DataFrame:
    """Point-to-point (unsmoothed) log-response step size for one port's own
    chronological series, as a robust z-score against that port's own typical
    step size over the trailing scale_window_days. Adds ptp, ptp_z,
    rate_outlier; does not mutate the input.

    secondary_multiplier adds Canny-style hysteresis: a point that clears the
    lower secondary threshold is only kept if it's adjacent (in this port's
    own chronological sequence) to a point that clears the primary
    mad_multiplier -- i.e. it rides along with an already-triggered episode's
    decaying tail, rather than lowering the threshold everywhere. A weak
    point next to a normal (non-weak) point does not get pulled in, so this
    never re-opens a chain through genuinely quiet data. None disables it
    (equivalent to secondary_multiplier == mad_multiplier).
    """
    sub = sub.sort_values("analysis_datetime").reset_index(drop=True).copy()
    height = pd.to_numeric(sub["height"], errors="coerce")
    logh = np.log(height.where(height > 0))
    ptp = logh.diff()
    med, scale = _rolling_mad_scale(ptp, sub["analysis_datetime"], scale_window_days, scale_min_periods)
    sub["ptp"] = ptp
    sub["ptp_z"] = np.where((scale > 0) & scale.notna(), (ptp - med) / scale, np.nan)

    strong = sub["ptp_z"].abs() > mad_multiplier
    if secondary_multiplier is None:
        sub["rate_outlier"] = strong
        return sub

    weak = sub["ptp_z"].abs() > secondary_multiplier
    keep = pd.Series(False, index=sub.index)
    weak_positions = sub.index[weak]
    if len(weak_positions):
        runs = (pd.Series(weak_positions).diff() != 1).cumsum()
        for _, positions in pd.Series(weak_positions).groupby(runs):
            if strong.loc[positions].any():
                keep.loc[positions] = True
    sub["rate_outlier"] = keep
    return sub


def _group_periods(times, max_gap_hours: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge a set of flagged timestamps into contiguous (start, end) periods,
    bridging gaps of max_gap_hours or less."""
    ts = pd.Series(times).dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if ts.empty:
        return []
    breaks = ts.diff() > pd.Timedelta(hours=max_gap_hours)
    group_id = breaks.cumsum()
    return [(grp.iloc[0], grp.iloc[-1]) for _, grp in ts.groupby(group_id)]


def _widen_to_neighboring_cals(
    pstart: pd.Timestamp, pend: pd.Timestamp, cal_times: pd.Series
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pad a period out to the nearest cal-port reading (either cal1 or cal2)
    just outside each end, so the two cal points bordering a flagged episode
    are swept in too -- along with the air readings between the old and new
    boundary, via the normal time-window sweep. cal_times is every cal-port
    timestamp for this analyte, not just the flagged ones."""
    before = cal_times[cal_times < pstart]
    after = cal_times[cal_times > pend]
    wstart = before.max() if not before.empty else pstart
    wend = after.min() if not after.empty else pend
    return wstart, wend


def build_cal_step_qc(
    batch: CATS_batch,
    pnum: int,
    channel: str,
    start: str,
    end: str,
    mad_multiplier: float = 3.5,
    secondary_multiplier: float | None = 2.0,
    max_gap_hours: float = 24.0,
    min_cal_points: int = 1,
    scale_window_days: float = 30.0,
    scale_min_periods: int = 20,
) -> pd.DataFrame:
    """Return flagged rows (all ports) for cal-response step-change periods.

    secondary_multiplier extends a triggered cal port's own flagged points to
    adjacent (in time, same port) points that clear this lower threshold --
    catches an already-triggered episode's decaying tail (e.g. the point
    right after a jump that's still elevated but under mad_multiplier) without
    lowering the detection threshold everywhere. Pass None to disable.

    min_cal_points drops periods backed by fewer than that many rate-outlier
    cal-port points -- e.g. raise it above the default of 1 to ignore isolated
    single-injection blips and keep only sustained multi-point episodes like
    the April 1999 example in the module docstring.

    scale_window_days sets the trailing local window each cal port's "typical"
    noise scale is measured over (see _rolling_mad_scale). The load window is
    padded by this many days of lookback so results at the requested start
    date don't depend on how far back the caller happened to query -- this
    padding is only for context; no output rows are produced before start.
    """
    requested_start = pd.Timestamp(start)
    requested_start = (
        requested_start.tz_localize("UTC") if requested_start.tzinfo is None
        else requested_start.tz_convert("UTC")
    )
    load_start = (requested_start - pd.Timedelta(days=scale_window_days)).strftime("%Y-%m-%d")

    df = batch.load_data(pnum, channel=channel, start_date=load_start, end_date=end, verbose=False)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True)

    cal_frames = []
    candidate_times = []
    for port in (batch.CAL1_PORT, batch.CAL2_PORT):
        sub = df.loc[df["port"] == port]
        if len(sub) < 5:
            continue
        sub = _port_rate_outliers(
            sub, mad_multiplier, secondary_multiplier,
            scale_window_days=scale_window_days, scale_min_periods=scale_min_periods,
        )
        sub["cal_port"] = port
        cal_frames.append(sub)
        candidate_times.append(sub.loc[sub["rate_outlier"], "analysis_datetime"])

    if not cal_frames or not any(len(c) for c in candidate_times):
        return pd.DataFrame()

    diag = pd.concat(cal_frames, ignore_index=True)[
        ["analysis_datetime", "cal_port", "ptp", "ptp_z", "rate_outlier"]
    ]

    all_candidates = pd.concat(candidate_times).sort_values()
    # Candidates in the lookback padding exist only to warm up the rolling
    # scale; drop them before building periods/output for the actual request.
    all_candidates = all_candidates[all_candidates >= requested_start]
    if all_candidates.empty:
        return pd.DataFrame()
    all_cal_times = diag["analysis_datetime"].sort_values().reset_index(drop=True)
    periods = _group_periods(all_candidates, max_gap_hours)
    if not periods:
        return pd.DataFrame()

    rows = []
    for i, (pstart, pend) in enumerate(periods, start=1):
        n_candidates = int(all_candidates.between(pstart, pend).sum())
        if n_candidates < min_cal_points:
            continue
        # Widen by one cal reading (either port) on each side so the two
        # cal points bordering this episode -- and the air readings between
        # the old and new boundary -- get swept in with it.
        wstart, wend = _widen_to_neighboring_cals(pstart, pend, all_cal_times)
        in_period = df.loc[
            (df["analysis_datetime"] >= wstart) & (df["analysis_datetime"] <= wend)
        ].copy()
        if in_period.empty:
            continue
        in_period = in_period.merge(diag, on="analysis_datetime", how="left")
        in_period["rate_outlier"] = in_period["rate_outlier"].eq(True)
        in_period["period_id"] = i
        in_period["period_start"] = wstart
        in_period["period_end"] = wend
        in_period["flags"] = np.where(
            in_period["rate_outlier"],
            "cal_rate_outlier;cal_step_period",
            "cal_step_period",
        )
        rows.append(in_period)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["period_id", "analysis_datetime"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", default="brw")
    p.add_argument("--gas", default="N2O_q", help='Gas_channel (e.g. N2O_q) or "all"')
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument(
        "--mad-multiplier", type=float, default=3.5,
        help="Point-to-point log-response z-score threshold on the cal ports (default: 3.5)",
    )
    p.add_argument(
        "--secondary-multiplier", type=float, default=2.0,
        help="Lower hysteresis threshold: extends a triggered cal point to adjacent "
             "points on the same port clearing this bar (default: 2.0; pass a value "
             ">= --mad-multiplier, or a large number, to effectively disable)",
    )
    p.add_argument(
        "--max-gap-hours", type=float, default=24.0,
        help="Max gap between flagged cal points to stay in one period (default: 24)",
    )
    p.add_argument(
        "--min-cal-points", type=int, default=1,
        help="Drop periods with fewer than this many rate-outlier cal points "
             "(default: 1, i.e. keep isolated single-injection blips too)",
    )
    p.add_argument(
        "--scale-window-days", type=float, default=30.0,
        help="Trailing local window (days) each cal port's typical noise scale "
             "is measured over, so results don't depend on query span (default: 30)",
    )
    p.add_argument("--output", type=Path, default=Path("cats_cal_step_flags.csv"))
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
        out = build_cal_step_qc(
            batch, pnum, channel, args.start, args.end,
            mad_multiplier=args.mad_multiplier, secondary_multiplier=args.secondary_multiplier,
            max_gap_hours=args.max_gap_hours, min_cal_points=args.min_cal_points,
            scale_window_days=args.scale_window_days,
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
        periods = result.drop_duplicates(["gas", "channel", "period_id"])
        print(f"{len(periods):,} distinct period(s) across {result['gas'].nunique()} gas(es):")
        print(
            periods[["gas", "channel", "period_id", "period_start", "period_end"]]
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
