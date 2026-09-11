#!/usr/bin/env python3
"""Detect + recommend ONLY -- like cats_cal_method_qc.py, never writes to the
database. No cats_apply_*-style batch-apply script exists for its output
yet (apply by hand -- see cats_qc/CATS-processing.md section 2h), but the
design itself is validated, not experimental: cross-checked against BRW
CCl4(f)/N2O(q)/SF6(q), beats or matches the currently-persisted real
series' total roughness in all three, and independently lands on cal12
through the N2O 2019-04-22 discontinuity that motivated
cats_cal_method_qc.py's own redesign earlier this project -- without being
told. See cats_qc/CATS-processing.md's Viterbi section (2h) for the
validated numbers and how this relates to cats_cal_method_qc.py /
cats_cal_tank_health_qc.py.

Chooses a per-period CATS calibration method (cal12/cal2/cal1, in that
preference order) by minimizing the TOTAL first-derivative "roughness" of
the assembled mole-fraction series across the whole record at once (a
dynamic-programming / Viterbi shortest-path problem), instead of
cats_cal_method_qc.py's approach (detect local jumps against a trend
window, then greedily try candidates per flagged episode).

Reuses cats_cal_method_qc.py's period-level candidate-series machinery
(_load_period_air_response / _build_candidate_series) directly -- same
periods (CATS_batch._fit_periods()), same per-candidate mole-fraction math
-- so this is a genuine alternative READOUT of the same underlying data,
not a different data path.

DP formulation
--------------
State per period = which of {cal12, cal2, cal1} was chosen. Transition
cost from (period i-1, method a) to (period i, method b) is the actual
jump the assembled series would show there: |value_b(i) - value_a(i-1)|.
Three additions on top of that raw jump cost, each earned by a real failure
seen on real BRW data (see cats_qc/CATS-processing.md for the specific
episodes):

1. Preference penalty -- a small per-period cost added for choosing cal2 or
   cal1 (0 for cal12), so a tie or near-tie resolves toward the preferred
   method, but a real jump large enough to save more than the penalty can
   still force a fallback.
2. Switch cost -- an asymmetric, larger cost charged only when the CHOSEN
   method changes from the previous period: cheap at a real cal1/cal2 tank
   transition (--low-switch-cost), expensive everywhere else
   (--high-switch-cost). Without this, the path flickers between candidates
   period-to-period whenever one is locally a hair smoother than the other,
   even with no hardware change to justify it.
3. Rolling-median smoothing (--smooth-window) applied to each candidate
   column BEFORE the DP runs. A single anomalous period (e.g. a
   just-swapped tank's first, still-settling reading) can otherwise
   single-handedly swing the switch-cost-gated DP onto an alternate
   candidate for years, because the raw jump cost only ever compares one
   period to its immediate neighbor. A short median washes out a lone
   one-period outlier while leaving genuine multi-period problems (a
   candidate that's wrong for its ENTIRE service life, not just one week)
   fully intact -- unlike masking data outright (tried and reverted; see
   build_candidate_table's docstring), no evidence is discarded, so a
   persistently-bad candidate still reads as persistently bad.

All three penalties/costs are scaled by the record's own robust
week-to-week step (1.4826*MAD of successive differences), so they adapt to
each analyte instead of being fixed absolute numbers.

Caveat this design does NOT address: pure step-minimization can still mask
a period where NONE of the three candidates is actually right (as with the
AAL073338/AAL073214 tank pair -- see project memory) by picking whichever
is locally closest to its neighbors. The "still-rough after the best
choice" printout below is a first pass at flagging that case for review,
not a full replacement for cats_cal_method_qc.py's UNRESOLVED semantics.

Usage:
    python3 cal_method_viterbi.py --site brw --gas CCl4_f --start 19980101 \\
        --output brw_ccl4_f_viterbi.csv --plot brw_ccl4_f_viterbi.png
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cats_batch import CATS_batch
from cats_cal_method_qc import (
    METHOD_NAME_TO_NUM,
    _build_candidate_series,
    _load_period_air_response,
    _load_period_series,
    _resolve_pnum,
)

PREFERENCE = ["cal12", "cal2", "cal1"]


def _parse_yyyymmdd(s: str) -> str:
    s = s.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid date {s!r}; expected YYYYMMDD or YYYY-MM-DD.")


def build_candidate_table(
    batch: CATS_batch, pnum: int, channel: str, start: str, end: str,
    min_period_points: int = 4, smooth_window: int = 3,
) -> pd.DataFrame:
    """One row per period, one column per PREFERENCE method (NaN where that
    method has no usable fit/coverage for that period).

    Tried and reverted: masking a candidate to NaN for N days after its
    tank's install date (to blind the DP to post-transition settling
    transients, e.g. the 2023-06-16 CC456884 case where x0 took ~1.5 weeks
    to stabilize) made the overall result worse, not better -- it also hid
    cases where a candidate is persistently wrong for its ENTIRE service
    life, not just the first N days (e.g. the 2004-02-19 ALM069781 cal1
    fit, which stayed ~20 ppb off for its full ~1.5-year tenure). Blocking
    that candidate's early data removed exactly the long-run evidence the
    DP needs to correctly avoid it, and reintroduced it partway through
    with no context, creating a sharp artifact at the mask boundary. See
    project memory for the full comparison.

    smooth_window instead ROBUSTLY SMOOTHS each candidate column with a
    centered rolling median (positional, not calendar-time -- periods are
    mostly weekly already) before the DP ever sees it. This targets the
    actual mechanism directly: a single anomalous period (e.g. a
    just-transitioned tank's first, still-settling reading) can otherwise
    single-handedly swing the DP onto an alternate candidate for years,
    because the DP only compares point-to-point. A short median washes out
    a lone one-period outlier while leaving genuine multi-period problems
    (like the 2004 case above, wrong in EVERY period, not just one) fully
    intact -- unlike masking, no evidence is discarded, so a persistently-
    bad candidate still reads as persistently bad.
    """
    period_resp = _load_period_air_response(batch, pnum, channel, start, end, min_period_points)
    if period_resp.empty:
        return pd.DataFrame()

    table = period_resp[["period_start", "period_mid"]].copy()
    for name in PREFERENCE:
        series = _build_candidate_series(
            batch, pnum, channel, METHOD_NAME_TO_NUM[name], period_resp, start, end,
        )
        col = (
            series.set_index("period_start")["median_mf"]
            if not series.empty else pd.Series(dtype=float)
        )
        values = table["period_start"].map(col)
        if smooth_window > 1:
            values = values.rolling(smooth_window, center=True, min_periods=1).median()
        table[name] = values
    return table.dropna(how="all", subset=PREFERENCE).reset_index(drop=True)


def tank_transition_mask(batch: CATS_batch, period_start: pd.Series) -> np.ndarray:
    """True at period i if the cal1 or cal2 tank installed differs from
    period i-1's -- the physically 'logical' points for a calibration
    method to legitimately change (a fresh tank may need a different
    method than its predecessor). False elsewhere: the same two tanks
    stayed installed, so a method change there has no physical cause and
    should cost more to select (see solve_viterbi's switch_cost)."""
    cal1 = batch.tank_serials_for_dates(batch.CAL1_PORT, period_start).to_numpy()
    cal2 = batch.tank_serials_for_dates(batch.CAL2_PORT, period_start).to_numpy()
    mask = np.zeros(len(period_start), dtype=bool)
    mask[1:] = (cal1[1:] != cal1[:-1]) | (cal2[1:] != cal2[:-1])
    return mask


def robust_step_scale(values: np.ndarray) -> float:
    """1.4826 * MAD of successive differences -- the record's own typical
    week-to-week step, used to scale the preference penalty and the
    still-rough threshold instead of a fixed absolute number."""
    diffs = np.diff(values[np.isfinite(values)])
    if diffs.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(diffs - np.median(diffs))))


def solve_viterbi(
    table: pd.DataFrame,
    pref_penalty: dict[str, float],
    transition_mask: np.ndarray,
    low_switch_cost: float,
    high_switch_cost: float,
) -> pd.DataFrame:
    """Return table with an added 'chosen' column (method name) and
    'chosen_mf' (the value under that method) via forward DP + backtrack.

    transition_mask[i] True means the cal1 and/or cal2 tank installed at
    period i differs from period i-1's -- a physical cal-tank swap, the
    logical point for a method change. Switching methods there costs only
    low_switch_cost; switching anywhere else (same tanks still installed)
    costs high_switch_cost. No cost at all for staying on the same method.
    This is what keeps the chosen path from flickering between candidates
    period-to-period just because it's locally a hair smoother -- a real
    calibration-scheme change should track a real hardware change, not
    noise (see cats_qc/CATS-processing.md section 2h)."""
    n = len(table)
    methods = PREFERENCE
    values = {m: table[m].to_numpy(dtype=float) for m in methods}
    available = {m: np.isfinite(values[m]) for m in methods}

    INF = float("inf")
    dp = np.full((n, len(methods)), INF)
    back = np.full((n, len(methods)), -1, dtype=int)

    for j, m in enumerate(methods):
        if available[m][0]:
            dp[0, j] = pref_penalty[m]

    for i in range(1, n):
        switch_cost = low_switch_cost if transition_mask[i] else high_switch_cost
        for j, m in enumerate(methods):
            if not available[m][i]:
                continue
            best_cost, best_k = INF, -1
            for k, pm in enumerate(methods):
                if not np.isfinite(dp[i - 1, k]):
                    continue
                step = abs(values[m][i] - values[pm][i - 1])
                extra = 0.0 if pm == m else switch_cost
                cost = dp[i - 1, k] + step + pref_penalty[m] + extra
                if cost < best_cost:
                    best_cost, best_k = cost, k
            # best_k is always found here: build_candidate_table() drops any
            # period with zero available candidates (dropna how="all"), and
            # dp[i-1, :] is populated for every available[.][i-1] state --
            # inductively, starting from row 0's explicit seeding below --
            # so row i-1 always has >=1 finite entry to serve as pm.
            dp[i, j] = best_cost
            back[i, j] = best_k

    chosen_idx = np.full(n, -1, dtype=int)
    j = int(np.argmin(dp[n - 1]))
    if not np.isfinite(dp[n - 1, j]):
        raise RuntimeError("No feasible path found -- every period lacks all three candidates?")
    for i in range(n - 1, -1, -1):
        chosen_idx[i] = j
        j = back[i, j]

    out = table.copy()
    out["chosen"] = [methods[k] for k in chosen_idx]
    out["chosen_mf"] = [values[methods[k]][i] for i, k in enumerate(chosen_idx)]
    return out


def total_variation(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    if v.size < 2:
        return float("nan")
    return float(np.sum(np.abs(np.diff(v))))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", default="brw")
    p.add_argument("--gas", required=True, help="Gas_channel, e.g. CCl4_f")
    p.add_argument("--start", type=_parse_yyyymmdd, default="1998-01-01")
    p.add_argument("--end", type=_parse_yyyymmdd, default=None)
    p.add_argument("--penalty-fraction", type=float, default=0.5,
                    help="Preference-penalty step, in units of the record's own robust "
                         "week-to-week step scale, added per preference-order rung "
                         "(cal2 costs 1x this, cal1 costs 2x). Default: 0.5")
    p.add_argument("--rough-threshold", type=float, default=3.0,
                    help="Flag a chosen-path step as 'still rough' if it exceeds this "
                         "multiple of the robust step scale. Default: 3.0")
    p.add_argument("--smooth-window", type=int, default=3,
                    help="Centered rolling-median window (in periods) applied to each "
                         "candidate series before the DP runs -- washes out single-period "
                         "outliers without discarding evidence. Default: 3, 1 to disable.")
    p.add_argument("--min-period-points", type=int, default=4,
                    help="Minimum unrejected air-port injections to trust a period's "
                         "median response (same convention as cats_cal_method_qc.py). "
                         "Default: 4")
    p.add_argument("--low-switch-cost", type=float, default=1.0,
                    help="Cost to change method AT a cal1/cal2 tank transition, in units "
                         "of the robust step scale. Default: 1.0")
    p.add_argument("--high-switch-cost", type=float, default=5.0,
                    help="Cost to change method where the tanks did NOT change, in units "
                         "of the robust step scale. Default: 5.0")
    p.add_argument("--output", type=Path, default=None,
                    help="Save the solved period table (period_start, period_mid, "
                         "cal12/cal2/cal1 candidates, chosen, chosen_mf) to this CSV.")
    p.add_argument("--plot", type=Path, default=None,
                    help="Save a PNG comparing the currently-persisted period series "
                         "against the DP-chosen series (colored by chosen method).")
    args = p.parse_args()

    gas, channel = args.gas.rsplit("_", 1)
    end = args.end or datetime.now().strftime("%Y-%m-%d")

    batch = CATS_batch(args.site)
    pnum = _resolve_pnum(batch, gas, channel)

    print(f"Loading candidate series for {gas} ({channel}) pnum={pnum} {args.start} -> {end} ...")
    table = build_candidate_table(
        batch, pnum, channel, args.start, end,
        min_period_points=args.min_period_points, smooth_window=args.smooth_window,
    )
    if table.empty:
        print("No data.")
        return 0
    print(f"{len(table)} periods with at least one usable candidate.")

    scale = robust_step_scale(table["cal12"].to_numpy(dtype=float))
    if scale == 0.0:
        scale = robust_step_scale(table["cal2"].to_numpy(dtype=float))
    pref_penalty = {
        "cal12": 0.0,
        "cal2": args.penalty_fraction * scale,
        "cal1": 2 * args.penalty_fraction * scale,
    }
    print(f"Robust step scale: {scale:.4f}; preference penalties: {pref_penalty}")

    transition_mask = tank_transition_mask(batch, table["period_start"])
    low_switch_cost = args.low_switch_cost * scale
    high_switch_cost = args.high_switch_cost * scale
    print(f"{transition_mask.sum()} of {len(transition_mask)} periods are a cal1/cal2 "
          f"tank-transition boundary. Switch costs: at transition={low_switch_cost:.4f}, "
          f"elsewhere={high_switch_cost:.4f}")

    solved = solve_viterbi(table, pref_penalty, transition_mask, low_switch_cost, high_switch_cost)
    solved["is_tank_transition"] = transition_mask

    tv_cal12_only = total_variation(table["cal12"].to_numpy(dtype=float))
    tv_chosen = total_variation(solved["chosen_mf"].to_numpy(dtype=float))

    persisted = _load_period_series(
        batch, pnum, channel, args.start, end, min_period_points=args.min_period_points,
    )
    tv_persisted = total_variation(persisted["median_mf"].to_numpy(dtype=float)) if not persisted.empty else float("nan")

    print(f"\nTotal variation (sum of |step|) over the whole record:")
    print(f"  blanket cal12 only (periods where computable): {tv_cal12_only:.2f}")
    print(f"  currently persisted (real DB series):          {tv_persisted:.2f}")
    print(f"  DP-optimal chosen path:                        {tv_chosen:.2f}")

    counts = solved["chosen"].value_counts()
    print(f"\nChosen method counts: {counts.to_dict()}")

    changed = solved["chosen"].to_numpy()[1:] != solved["chosen"].to_numpy()[:-1]
    at_transition = transition_mask[1:]
    print(f"\nMethod changes: {changed.sum()} total, "
          f"{(changed & at_transition).sum()} at a tank transition, "
          f"{(changed & ~at_transition).sum()} NOT at a tank transition")

    non_cal12 = solved.loc[solved["chosen"] != "cal12"]
    print(f"\n{len(non_cal12)} period(s) where preference was overridden (not cal12):")
    for _, row in non_cal12.head(40).iterrows():
        print(f"  {row['period_start'].date()}  -> {row['chosen']}  (mf={row['chosen_mf']:.3f})")
    if len(non_cal12) > 40:
        print(f"  ... and {len(non_cal12) - 40} more")

    steps = np.abs(np.diff(solved["chosen_mf"].to_numpy(dtype=float)))
    rough_mask = steps > (args.rough_threshold * scale)
    rough_idx = np.where(rough_mask)[0]
    print(f"\n{len(rough_idx)} period-to-period step(s) still exceed "
          f"{args.rough_threshold}x the robust scale ({args.rough_threshold * scale:.3f}) "
          f"even at the DP-optimal choice -- candidates for manual review:")
    for i in rough_idx[:40]:
        a, b = solved.iloc[i], solved.iloc[i + 1]
        print(f"  {a['period_start'].date()} ({a['chosen']}, {a['chosen_mf']:.3f}) -> "
              f"{b['period_start'].date()} ({b['chosen']}, {b['chosen_mf']:.3f})  "
              f"step={steps[i]:.3f}")
    if len(rough_idx) > 40:
        print(f"  ... and {len(rough_idx) - 40} more")

    if args.output:
        solved.to_csv(args.output, index=False)
        print(f"\nWrote solved period table to {args.output}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"cal12": "tab:blue", "cal2": "tab:orange", "cal1": "tab:green"}
        fig, ax = plt.subplots(figsize=(14, 6))
        if not persisted.empty:
            ax.plot(persisted["period_mid"], persisted["median_mf"], color="0.75",
                    linewidth=1.0, label="currently persisted", zorder=1)
        for name, color in colors.items():
            sub = solved.loc[solved["chosen"] == name]
            if sub.empty:
                continue
            ax.scatter(sub["period_mid"], sub["chosen_mf"], s=14, color=color,
                       label=f"DP: {name}", zorder=2)
        ax.plot(solved["period_mid"], solved["chosen_mf"], color="0.3",
                linewidth=0.6, zorder=1.5, alpha=0.6)
        ax.set_xlabel("Period mid")
        ax.set_ylabel("Mole fraction")
        ax.set_title(f"{gas} ({channel}) at {args.site.upper()} -- DP-chosen vs persisted period medians")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"Wrote plot to {args.plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
