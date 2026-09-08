#!/usr/bin/env python3
"""Flag weekly periods where CATS' CAL1 or CAL2 tank looks unreliable, and
recommend the single-tank method that avoids the bad one.

Different question from cats_cal_method_qc.py: that tool looks for STEPS in
the computed mole-fraction series and asks "which method removes this jump"
-- any cause (detector drift, ECD response change, a real short atmospheric
excursion) can trigger it, and most flagged jumps turn out not to be a tank
problem at all. This tool instead looks directly at each cal tank's OWN raw
injection data, port by port, and asks a narrower question: is CAL1 (or
CAL2) actually delivering usable measurements this week? A tank that's
offline, mostly rejected, giving empty integrations, or unusually noisy
relative to its own history is a candidate for switching that week's method
to whichever cal tank IS healthy (cal1 if CAL2 is bad, cal2 if CAL1 is bad)
-- cal12 needs both tanks, so one going bad is exactly the situation cal12
can't handle gracefully but a single-tank method can.

cal12 remains the preferred, default method everywhere -- it captures both
detector gain and offset, which a single-tank method cannot. This tool only
ever recommends a departure from it for the specific flagged windows where
one tank looks unhealthy; an analyte/period with no output row here has
nothing wrong with either cal tank and should stay on cal12. Unlike
cats_cal_method_qc.py's cal12 > cal2 > cal1 preference order (used when
resolving a detected discontinuity by trying every method), this tool never
proposes cal12 as an outcome -- if neither tank were unhealthy there would
be nothing to flag in the first place.

Detect + recommend ONLY -- like cats_cal_method_qc.py, this never writes to
the database. Apply a recommendation with cats_set_mf_method.py directly
(there is no cats_apply_cal_method.py-style batch-apply script for this
tool's output yet -- review each row first, the health signals below are
heuristics, not a certainty of tank failure).

Algorithm
---------
1. Load raw data for the requested gas/channel, restricted to CAL1_PORT and
   CAL2_PORT, and aggregate to CATS_batch._fit_periods() periods (normally
   calendar weeks, split at a mid-week cal-tank swap) -- the same boundaries
   update_fits() itself fits on, so a flagged period_start is already a
   valid cats_set_mf_method.py --start.
2. Per port per period, compute three independent health signals:
     - relative coverage: this port's share of CAL1+CAL2's COMBINED
       injection count that period (0.5 = an even split). Compared against
       the OTHER cal port in the SAME period, deliberately not a trailing
       self-baseline -- CAL1 and CAL2 are injected on the same instrument in
       the same run cycle, so an instrument-wide event (short outage, a
       partial run, a quiet week) drops both ports' counts together, and a
       per-port trailing baseline cannot tell that apart from one tank
       specifically going bad (confirmed: an early version of this tool
       using each port's own trailing median flagged BOTH ports on nearly
       every low-cycle-count week, drowning genuine single-tank signals in
       shared-cause noise). Comparing to the sibling port in the same period
       cancels out anything instrument-wide by construction.
     - dropout: fraction of this period's injections with height==0 (no
       peak integrated at all -- see logos_instruments_insitu.py's
       _null_mole_fraction_for_zero_height for the same convention used
       downstream). A tank that's physically present but not flowing/
       detected shows up here even when coverage looks fine. Independent
       per-port signal -- not confounded by the sibling port the way raw
       coverage is, since an empty integration is specific to this port's
       own peak.
     - noise: this period's robust relative spread of normalized_resp
       (1.4826 * MAD / median) versus THIS PORT'S OWN trailing baseline
       spread (--noise-window-days, default 90; the sibling port measures a
       different tank's assigned value entirely and would not be
       comparable). A tank giving erratic, noisy responses (bad regulator,
       leak, degrading fitting) shows up here even with full coverage and
       zero dropouts.
3. A period is flagged for a port when ANY of the three signals clears its
   own threshold (--min-relative-coverage, --max-dropout-frac,
   --max-noise-ratio). Flagged periods are grouped into episodes the same
   way cats_cal_method_qc.py does (reuses cats_cal_step_qc._group_periods).
4. Recommendation: cal2 if only CAL1 is flagged that period, cal1 if only
   CAL2 is flagged, UNRESOLVED (both or neither clearly bad -- e.g. both
   flagged together suggests a shared-cause problem like a valve or
   detector issue, not one tank) for manual review. cal12 is never
   recommended here -- if neither tank looks unhealthy there is nothing to
   flag in the first place.

This is deliberately much simpler than cats_cal_method_qc.py's trend-jump
machinery: it is not trying to detect whether the atmosphere-facing output
changed, only whether the two calibration inputs are individually trustworthy.

Reference mode (--reference-gas)
---------------------------------
CAL1 and CAL2 are physical tanks/regulators/valves feeding every channel's
detector from the same injection at once -- if a tank is offline or dry,
every channel sees it simultaneously, so COVERAGE and DROPOUT are properties
of the tank/port, not of any one analyte. NOISE is not: it is a per-analyte,
per-channel response-quality signal (a noise problem on one detector's trace
does not imply another channel's detector is also noisy), so it stays
per-analyte and is NOT combined across references.

Pass --reference-gas one or more times (default: N2O_q, SF6_q, CFC11_f -- a
traditionally well-measured, high-precision set spanning both physical
channels in use) to compute ONE canonical, analyte-independent tank-health
verdict from coverage+dropout only, unioned across references: a period is
flagged bad for a port if ANY reference analyte's own coverage/dropout
signal flags it there (not requiring unanimous agreement, since an
unrelated per-analyte data gap in just one reference should not suppress a
real tank problem the others correctly caught). The resulting CSV has no
gas/channel/pnum columns -- it applies to every analyte at the site, and is
the recommended input to cats_apply_cal_tank_health.py for routine use,
rather than re-deriving (noisier, analyte-specific) episodes per analyte.

Usage::

    python3 cats_cal_tank_health_qc.py --site brw --gas CFC113_f --start 19980101 -v

    # Combined coverage+dropout reference verdict (recommended for routine use)
    python3 cats_cal_tank_health_qc.py --site brw --reference-gas N2O_q \\
        --reference-gas SF6_q --reference-gas CFC11_f --start 19980101 -v \\
        --output brw_reference_tank_health.csv
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
from cats_cal_step_qc import _group_periods

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


def _period_port_stats(
    batch: CATS_batch, port_df: pd.DataFrame, min_points: int,
) -> pd.DataFrame:
    """One row per period_start for a single port: n, n_rejected_frac,
    dropout_frac (height==0 among unrejected), and the robust relative
    spread of normalized_resp (1.4826*MAD/median) among unrejected,
    non-dropout rows. Periods with fewer than min_points unrejected rows
    get NaN spread/dropout (not enough context -- never flagged), matching
    the "not enough information" convention used throughout cats_qc/.
    """
    df = port_df.copy()
    df["rejected"] = pd.to_numeric(df["rejected"], errors="coerce").fillna(0).astype(int)
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["normalized_resp"] = pd.to_numeric(df["normalized_resp"], errors="coerce")

    rows = []
    for period_start, grp in df.groupby("period_start"):
        unrejected = grp.loc[grp["rejected"].eq(0)]
        n = len(unrejected)
        dropout = unrejected["height"].fillna(0).eq(0)
        dropout_frac = float(dropout.mean()) if n else np.nan

        good = unrejected.loc[~dropout, "normalized_resp"].dropna()
        if len(good) >= min_points:
            med = float(good.median())
            mad = float((good - med).abs().median())
            spread = (1.4826 * mad / abs(med)) if med else np.nan
        else:
            spread = np.nan

        rows.append({
            "period_start": period_start, "n": n, "dropout_frac": dropout_frac,
            "spread": spread,
        })
    return pd.DataFrame(rows).sort_values("period_start").reset_index(drop=True)


def _score_ports(
    cal1_stats: pd.DataFrame, cal2_stats: pd.DataFrame,
    noise_window_days: float,
    min_relative_coverage: float, max_dropout_frac: float, max_noise_ratio: float,
) -> pd.DataFrame:
    """Score both cal ports together for one period_start-indexed table.

    coverage is scored as THIS port's injection count relative to the OTHER
    port's count in the SAME period, not against either port's own trailing
    history. CAL1 and CAL2 are injected on the same instrument in the same
    run cycle, so an instrument-wide event (short outage, a partial run, a
    quiet week) drops both ports' counts together -- a per-port trailing
    baseline can't tell that apart from one tank specifically going bad, and
    in practice flags both ports on nearly every low-cycle-count week (see
    the module docstring's algorithm step 2). Comparing to the other port in
    the SAME period cancels out anything instrument-wide by construction: if
    both drop together the ratio stays near 1 and neither is flagged; only
    a period where one port is lopsidedly starved relative to its sibling
    port is flagged.

    dropout_frac and noise_ratio remain independent per-port signals (a
    dropout is specific to that port's own peak integration; response noise
    is specific to that port's own regulator/fitting/tank condition) --
    noise_ratio still compares against that port's OWN trailing rolling
    baseline (not the sibling port's, which measures a different tank's
    assigned value entirely and would not be comparable).
    """
    idx = sorted(set(cal1_stats["period_start"]) | set(cal2_stats["period_start"]))
    c1 = cal1_stats.set_index("period_start").reindex(idx)
    c2 = cal2_stats.set_index("period_start").reindex(idx)

    n1, n2 = c1["n"].to_numpy(dtype=float), c2["n"].to_numpy(dtype=float)
    total = n1 + n2
    with np.errstate(invalid="ignore", divide="ignore"):
        cal1_rel_coverage = np.where(total > 0, n1 / total, np.nan)
        cal2_rel_coverage = np.where(total > 0, n2 / total, np.nan)

    def _noise_ratio(spread: pd.Series) -> np.ndarray:
        noise_periods = max(4, int(noise_window_days / 7))
        baseline = spread.rolling(noise_periods, min_periods=max(4, noise_periods // 3)).median()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = spread.to_numpy() / baseline.to_numpy()
        return np.where(baseline.notna() & (baseline.to_numpy() > 0), ratio, np.nan)

    out = {}
    for label, stats, rel_coverage in (("cal1", c1, cal1_rel_coverage), ("cal2", c2, cal2_rel_coverage)):
        noise_ratio = _noise_ratio(stats["spread"])
        dropout_frac = stats["dropout_frac"].to_numpy()

        # Flag only when THIS port's share of the period's combined cal
        # injections falls well below its fair share (min_relative_coverage
        # is compared against 0.5 = perfectly even split) -- i.e. its
        # sibling got most of the cycles that period, not just "fewer than
        # usual" for an instrument-wide reason.
        low_coverage = ~np.isnan(rel_coverage) & (rel_coverage < min_relative_coverage)
        high_dropout = ~np.isnan(dropout_frac) & (dropout_frac > max_dropout_frac)
        high_noise = ~np.isnan(noise_ratio) & (noise_ratio > max_noise_ratio)
        bad = low_coverage | high_dropout | high_noise
        reasons = [
            ";".join(x for x in r if x) for r in zip(
                np.where(low_coverage, "low_coverage", ""),
                np.where(high_dropout, "high_dropout", ""),
                np.where(high_noise, "high_noise", ""),
            )
        ]

        out[f"{label}_n"] = stats["n"].to_numpy()
        out[f"{label}_rel_coverage"] = rel_coverage
        out[f"{label}_dropout_frac"] = dropout_frac
        out[f"{label}_noise_ratio"] = noise_ratio
        out[f"{label}_bad"] = bad
        out[f"{label}_reasons"] = reasons

    return pd.DataFrame(out, index=pd.Index(idx, name="period_start")).reset_index()


def _episodes_from_merged(
    batch: CATS_batch, merged: pd.DataFrame, start: str, end: str, max_gap_days: float,
) -> pd.DataFrame:
    """Shared episode-grouping/recommendation step for both the per-analyte
    path (build_cal_tank_health_qc) and the combined-reference path
    (build_reference_tank_health_qc): given a period_start-indexed table
    with cal1_bad/cal2_bad/*_reasons/*_rel_coverage/*_dropout_frac/
    *_noise_ratio columns (noise columns may be entirely NaN for the
    reference path -- see its docstring), group flagged periods into
    episodes and recommend a method per episode. One row per detected
    episode; empty DataFrame if nothing is flagged in [start, end].
    """
    in_range = merged["period_start"].between(pd.Timestamp(start), pd.Timestamp(end))
    cal1_bad = merged.get("cal1_bad", pd.Series(False, index=merged.index)).fillna(False)
    cal2_bad = merged.get("cal2_bad", pd.Series(False, index=merged.index)).fillna(False)
    flagged = merged.loc[in_range & (cal1_bad | cal2_bad)]
    if flagged.empty:
        return pd.DataFrame()

    episodes = _group_periods(flagged["period_start"], max_gap_hours=max_gap_days * 24.0)
    if not episodes:
        return pd.DataFrame()

    rows = []
    for pstart, pend in episodes:
        in_ep = merged.loc[merged["period_start"].between(pstart, pend)]
        cal1_bad_frac = float(in_ep.get("cal1_bad", pd.Series(dtype=bool)).fillna(False).mean())
        cal2_bad_frac = float(in_ep.get("cal2_bad", pd.Series(dtype=bool)).fillna(False).mean())
        # Majority rule per tank (>=half its periods in this episode flagged)
        # for the reported cal1_bad/cal2_bad summary columns.
        ep_cal1_bad = cal1_bad_frac >= 0.5
        ep_cal2_bad = cal2_bad_frac >= 0.5

        # Recommend whichever tank was bad across a CLEARLY larger share of
        # this episode's periods, not "was either ever flagged even once" --
        # an episode can span months, and one isolated blip on the
        # otherwise-healthy tank (e.g. a single dropout week) must not veto
        # a sustained, dominant signal on the other tank. Genuinely close
        # calls (both similarly bad, or both similarly clean) still fall
        # through to UNRESOLVED.
        margin = 0.34  # roughly "3x as often flagged" at typical bad_frac scales
        if cal1_bad_frac > cal2_bad_frac + margin:
            recommendation = "cal2"
        elif cal2_bad_frac > cal1_bad_frac + margin:
            recommendation = "cal1"
        else:
            recommendation = "UNRESOLVED"

        def _unique_reasons(col: str) -> str:
            # Each period's reasons cell can itself be "a;b" (multiple
            # signals tripped that period) -- split before deduping so two
            # periods sharing one reason don't survive as duplicate
            # substrings (e.g. "low_coverage;low_coverage;high_dropout").
            tokens = set()
            for cell in in_ep.get(col, pd.Series(dtype=str)).dropna():
                tokens.update(t for t in cell.split(";") if t)
            return ";".join(sorted(tokens))

        cal1_reasons = _unique_reasons("cal1_reasons")
        cal2_reasons = _unique_reasons("cal2_reasons")

        rows.append({
            "episode_start": pstart, "episode_end": pend,
            "cal1_bad": ep_cal1_bad, "cal2_bad": ep_cal2_bad,
            "cal1_bad_frac": cal1_bad_frac, "cal2_bad_frac": cal2_bad_frac,
            "cal1_reasons": cal1_reasons, "cal2_reasons": cal2_reasons,
            "cal1_min_rel_coverage": float(in_ep.get("cal1_rel_coverage", pd.Series(dtype=float)).min()),
            "cal1_max_dropout": float(in_ep.get("cal1_dropout_frac", pd.Series(dtype=float)).max()),
            "cal1_max_noise_ratio": float(in_ep.get("cal1_noise_ratio", pd.Series(dtype=float)).max()),
            "cal2_min_rel_coverage": float(in_ep.get("cal2_rel_coverage", pd.Series(dtype=float)).min()),
            "cal2_max_dropout": float(in_ep.get("cal2_dropout_frac", pd.Series(dtype=float)).max()),
            "cal2_max_noise_ratio": float(in_ep.get("cal2_noise_ratio", pd.Series(dtype=float)).max()),
            "cal1_tank": batch.tank_serial_for_port(batch.CAL1_PORT, when=pstart + pd.Timedelta(days=3.5)),
            "cal2_tank": batch.tank_serial_for_port(batch.CAL2_PORT, when=pstart + pd.Timedelta(days=3.5)),
            "recommendation": recommendation,
        })

    return pd.DataFrame(rows).sort_values("episode_start").reset_index(drop=True)


def build_cal_tank_health_qc(
    batch: CATS_batch,
    pnum: int,
    channel: str,
    start: str,
    end: str,
    min_period_points: int = 4,
    noise_window_days: float = 90.0,
    min_relative_coverage: float = 0.25,
    max_dropout_frac: float = 0.3,
    max_noise_ratio: float = 3.0,
    max_gap_days: float = 21.0,
) -> pd.DataFrame:
    """Detect periods where CAL1 and/or CAL2 look unhealthy, and recommend
    the single-tank method that avoids whichever tank is bad. One row per
    detected episode. Never writes to the database.
    """
    df = batch.load_data(pnum, channel=channel, start_date=start, end_date=end, verbose=False)
    if df.empty:
        return pd.DataFrame()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True).dt.tz_localize(None)
    df["period_start"] = batch._fit_periods(df)

    cal1_df = df.loc[df["port"] == batch.CAL1_PORT]
    cal2_df = df.loc[df["port"] == batch.CAL2_PORT]
    if cal1_df.empty and cal2_df.empty:
        return pd.DataFrame()
    cal1_stats = _period_port_stats(batch, cal1_df, min_period_points)
    cal2_stats = _period_port_stats(batch, cal2_df, min_period_points)
    if cal1_stats.empty and cal2_stats.empty:
        return pd.DataFrame()

    merged = _score_ports(
        cal1_stats, cal2_stats, noise_window_days,
        min_relative_coverage, max_dropout_frac, max_noise_ratio,
    )
    return _episodes_from_merged(batch, merged, start, end, max_gap_days)


def build_reference_tank_health_qc(
    batch: CATS_batch,
    reference_gases: list[str],
    start: str,
    end: str,
    min_period_points: int = 4,
    min_relative_coverage: float = 0.25,
    max_dropout_frac: float = 0.3,
    max_gap_days: float = 21.0,
) -> pd.DataFrame:
    """Combined, analyte-independent tank-health verdict from coverage +
    dropout ONLY (no noise -- see module docstring's Reference mode
    section), unioned across reference_gases (each an 'Analyte_channel'
    token, e.g. 'N2O_q'). A period is bad for a port if ANY reference
    analyte's own coverage/dropout signal flags it there. One row per
    detected episode; no gas/channel/pnum columns, since the result applies
    to every analyte at this site. Never writes to the database.
    """
    per_gas_merged = []
    for gas_channel in reference_gases:
        gas, channel = gas_channel.rsplit("_", 1)
        try:
            pnum = _resolve_pnum(batch, gas, channel)
        except ValueError as exc:
            print(f"  Skipping reference {gas_channel}: {exc}")
            continue

        df = batch.load_data(pnum, channel=channel, start_date=start, end_date=end, verbose=False)
        if df.empty:
            continue
        df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True).dt.tz_localize(None)
        df["period_start"] = batch._fit_periods(df)

        cal1_df = df.loc[df["port"] == batch.CAL1_PORT]
        cal2_df = df.loc[df["port"] == batch.CAL2_PORT]
        if cal1_df.empty and cal2_df.empty:
            continue
        cal1_stats = _period_port_stats(batch, cal1_df, min_period_points)
        cal2_stats = _period_port_stats(batch, cal2_df, min_period_points)
        if cal1_stats.empty and cal2_stats.empty:
            continue

        # max_noise_ratio=np.inf disables the high_noise flag entirely
        # (noise_ratio > inf is always False), matching this mode's
        # coverage+dropout-only contract -- noise_window_days is still
        # computed internally but its result never gates anything here.
        scored = _score_ports(
            cal1_stats, cal2_stats, noise_window_days=90.0,
            min_relative_coverage=min_relative_coverage,
            max_dropout_frac=max_dropout_frac, max_noise_ratio=np.inf,
        )
        per_gas_merged.append(scored)

    if not per_gas_merged:
        return pd.DataFrame()

    idx = sorted(set().union(*(set(m["period_start"]) for m in per_gas_merged)))
    combined = {"period_start": idx}
    for label in ("cal1", "cal2"):
        bad_any = pd.Series(False, index=idx)
        reasons_union: dict = {p: set() for p in idx}
        rel_coverage_min = pd.Series(np.nan, index=idx)
        dropout_max = pd.Series(np.nan, index=idx)
        for m in per_gas_merged:
            s = m.set_index("period_start").reindex(idx)
            bad_any = bad_any | s[f"{label}_bad"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            for p, cell in s[f"{label}_reasons"].items():
                if isinstance(cell, str) and cell:
                    # Noise never appears here (max_noise_ratio=np.inf above
                    # means high_noise can't trigger), so only
                    # low_coverage/high_dropout tokens ever reach the union.
                    reasons_union[p].update(cell.split(";"))
            rel_coverage_min = np.fmin(rel_coverage_min, s[f"{label}_rel_coverage"].to_numpy())
            dropout_max = np.fmax(dropout_max, s[f"{label}_dropout_frac"].to_numpy())

        combined[f"{label}_bad"] = bad_any
        combined[f"{label}_reasons"] = [";".join(sorted(reasons_union[p])) for p in idx]
        combined[f"{label}_rel_coverage"] = rel_coverage_min
        combined[f"{label}_dropout_frac"] = dropout_max
        combined[f"{label}_noise_ratio"] = np.nan  # not scored in reference mode

    merged = pd.DataFrame(combined)
    return _episodes_from_merged(batch, merged, start, end, max_gap_days)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", default="brw")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--gas", help='Gas_channel (e.g. CFC113_f) or "all" -- '
                         "per-analyte mode (coverage+dropout+noise, this analyte only).")
    target.add_argument(
        "--reference-gas", action="append", dest="reference_gas", metavar="GAS_CHANNEL",
        help="Repeatable. Combined, analyte-independent coverage+dropout-only "
             "verdict unioned across these reference analytes (e.g. "
             "--reference-gas N2O_q --reference-gas SF6_q --reference-gas CFC11_f). "
             "See module docstring's Reference mode section. Not used with --gas.",
    )
    p.add_argument("--start", type=_parse_yyyymmdd, default="1998-01-01",
                    help="Start date, YYYYMMDD (default: instrument start).")
    p.add_argument("--end", type=_parse_yyyymmdd, default=None,
                    help="End date, YYYYMMDD (default: now).")
    p.add_argument("--min-period-points", type=int, default=4,
                    help="Minimum unrejected cal-port injections to trust a period's "
                         "noise estimate (default: 4)")
    p.add_argument("--noise-window-days", type=float, default=90.0,
                    help="Trailing days used to establish a port's typical response "
                         "noise (default: 90)")
    p.add_argument("--min-relative-coverage", type=float, default=0.25,
                    help="Flag a period if this port's share of the period's combined "
                         "CAL1+CAL2 injection count falls below this fraction (0.5 = "
                         "an even split between the two ports; default: 0.25, i.e. "
                         "this port got less than a third as many injections as its "
                         "sibling). Compared against the OTHER cal port in the same "
                         "period, not a trailing baseline -- see build_cal_tank_health_qc "
                         "docstring for why (instrument-wide coverage drops must not "
                         "flag both ports at once)")
    p.add_argument("--max-dropout-frac", type=float, default=0.3,
                    help="Flag a period if more than this fraction of unrejected "
                         "injections have height==0 (default: 0.3)")
    p.add_argument("--max-noise-ratio", type=float, default=3.0,
                    help="Flag a period if its response noise exceeds this multiple "
                         "of the port's trailing baseline noise (default: 3.0)")
    p.add_argument("--max-gap-days", type=float, default=21.0,
                    help="Max gap between flagged periods to merge into one episode "
                         "(default: 21)")
    p.add_argument("--output", type=Path, default=Path("cats_cal_tank_health_flags.csv"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    end_date = args.end or datetime.today().strftime("%Y-%m-%d")
    batch = CATS_batch(args.site)

    if args.reference_gas:
        if args.verbose:
            print(f"Scanning references {args.reference_gas} "
                  f"{args.start} -> {end_date} (coverage+dropout only) ...")
        result = build_reference_tank_health_qc(
            batch, args.reference_gas, args.start, end_date,
            min_period_points=args.min_period_points,
            min_relative_coverage=args.min_relative_coverage,
            max_dropout_frac=args.max_dropout_frac,
            max_gap_days=args.max_gap_days,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, float_format="%.6g")
        print(f"Wrote {len(result):,} episode(s) to {args.output}")
        if not result.empty:
            cols = ["episode_start", "episode_end", "cal1_bad", "cal2_bad", "recommendation"]
            print(result[cols].to_string(index=False))
        else:
            print("  No unhealthy cal-tank periods flagged across references.")
        return 0

    if args.gas.lower() == "all":
        rows = batch.db.doquery(
            f"SELECT DISTINCT display_name, channel FROM hats.analyte_list "
            f"WHERE inst_num = {batch.inst_num}"
        )
        gases = [f"{r['display_name']}_{r['channel']}" for r in rows if r.get("channel")]
    else:
        gases = [args.gas]

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
        out = build_cal_tank_health_qc(
            batch, pnum, channel, args.start, end_date,
            min_period_points=args.min_period_points,
            noise_window_days=args.noise_window_days,
            min_relative_coverage=args.min_relative_coverage,
            max_dropout_frac=args.max_dropout_frac,
            max_noise_ratio=args.max_noise_ratio,
            max_gap_days=args.max_gap_days,
        )
        if out.empty:
            print(f"  {gas} ({channel}): no unhealthy cal-tank periods flagged.")
            continue
        out["gas"] = gas
        out["channel"] = channel
        out["pnum"] = pnum
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.6g")
    print(f"Wrote {len(result):,} episode(s) to {args.output}")
    if not result.empty:
        cols = ["gas", "channel", "episode_start", "episode_end",
                "cal1_bad", "cal2_bad", "recommendation"]
        print(result[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
