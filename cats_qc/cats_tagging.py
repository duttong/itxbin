#!/usr/bin/env python3
"""Apply CATS QC-algorithm flags to hats.ng_insitu_mole_fraction_tags.

Each registered QC algorithm owns its own tag_num (see ALGORITHMS below).
Applying a tag is idempotent per (site, analyte, channel, start, end, algorithm):
it deletes that algorithm's tag_num from every mf_num loaded in the requested
window before reinserting it on the currently-flagged subset. This means a
point that stops being flagged after retuning a threshold loses its tag on
the next run instead of keeping it forever -- rerun freely while tuning.

NOTE on tag_num: all three registered algorithms write real reject tags in
ccgg.tag_dictionary (reject=1, automated=1) -- cal_step writes 328
("Detector cal-response rapid change"), baseline writes 329 ("Abnormal
chromatogram"), cal_window writes 286 ("Mole fraction falls outside of
calibration range, results certainly adversely affected" -- the "C" tag
under _TAG_LAYOUT's Automated Tags section in logos_data.py; its 287
companion is the non-reject "c" variant and is not written by cal_window).
A future algorithm can start on an unregistered placeholder number the same
way cal_step/baseline did; update ALGORITHMS once a real number is
assigned, and move any rows already written under the placeholder with
UPDATE hats.ng_insitu_mole_fraction_tags SET tag_num = <new> WHERE tag_num
= <placeholder>.

This module is meant to grow: as more CATS QC algorithms are added (peak
integration, ratio-based, etc.), register each as another ALGORITHMS entry
with its own tag_num and a build() callable of the same shape as
cats_cal_step_qc.build_cal_step_qc -- (batch, pnum, channel, start, end,
**kwargs) -> DataFrame with at least an mf_num column for the flagged rows.

ORDERING: cal_window must run after cal_step and baseline, and after mole
fractions have been recomputed against their rejection state -- its
reference-tank/air statistics exclude already-rejected rows (see
cats_cal_window_qc.py), and tagging alone never recomputes mole fractions
(see ../CLAUDE.md). `--algo all` handles both automatically: ALGORITHMS'
insertion order runs cal_step -> baseline -> cal_window, and main() calls
recalc_mole_fractions() (the cats_batch.py `-i --fits` recompute+upsert
path) for each analyte/channel right before cal_window's build() -- unless
--dry-run (preview only, no DB writes at all) or --skip-recalc (mole
fractions already known current). Running `--algo cal_window` on its own in
a separate invocation still gets this recalc; it is NOT conditional on
cal_step/baseline having just run in the same command.

Usage::

    # See what would be tagged, without writing anything
    python3 cats_tagging.py --site brw --algo cal_step --analyte N2O \\
        --channel q --start 19990101 --end 19991231 --dry-run

    # Actually write the tags
    python3 cats_tagging.py --site brw --algo cal_step --analyte N2O \\
        --channel q --start 19990101 --end 19991231

    # All registered algorithms, all analytes, defaults to Jan 1 of the
    # current year through now
    python3 cats_tagging.py --site brw --algo all --analyte all
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cats_batch import CATS_batch
from cats_cal_step_qc import ALL_GASES, build_cal_step_qc
from cats_baseline_qc import build_baseline_tag_qc
from cats_cal_window_qc import TAG_NUM as CAL_WINDOW_TAG_NUM, build_cal_window_qc


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


@dataclass(frozen=True)
class QcAlgorithm:
    tag_num: int
    build: Callable[..., pd.DataFrame]


# tag_num=328 ("Detector cal-response rapid change"), tag_num=329 ("Abnormal
# chromatogram"), and tag_num=286 ("Mole fraction falls outside of
# calibration") are all registered in ccgg.tag_dictionary as reject=1,
# automated=1 -- cal_step, baseline, and cal_window write these as real
# reject tags.
ALGORITHMS: dict[str, QcAlgorithm] = {
    "cal_step": QcAlgorithm(tag_num=328, build=build_cal_step_qc),
    # "Abnormal chromatogram": pre-peak shape deviates from its own local
    # (time-nearest) neighbors' median shape -- see cats_baseline_qc.py.
    "baseline": QcAlgorithm(tag_num=329, build=build_baseline_tag_qc),
    # "Mole fraction falls outside of calibration": air1/air2 mole fraction
    # deviates from its local air median by more than a reference-tank-noise
    # multiple (asymmetric: 3 sigma high, 2 sigma low) -- see
    # cats_cal_window_qc.py. tag_num is imported (not hardcoded here) since
    # cats_cal_window_qc.py's own logic needs the same number to exclude its
    # own prior tags from its "already rejected" pool filter.
    "cal_window": QcAlgorithm(tag_num=CAL_WINDOW_TAG_NUM, build=build_cal_window_qc),
}

# Algorithms whose build() reads mole_fraction values that depend on
# up-to-date rejection state (cal_window's reference-tank/air statistics
# exclude already-rejected rows -- see cats_cal_window_qc.py). Tagging alone
# never recomputes mole fractions (see ../CLAUDE.md's calibrations/
# tag-propagation notes), so main() recomputes+upserts them for these
# algorithms' pnum/channel/window right before calling their build(). Only
# matters in practice for cal_step/baseline rejections applied in an EARLIER
# invocation -- ALGORITHMS' own insertion order already sequences
# cal_step -> baseline -> cal_window within a single `--algo all` run.
_RECALC_BEFORE_BUILD = {"cal_window"}

# analyte display_name -> its one reporting channel, per ALL_GASES
# (CATS_GCwerks2DB.UPLOAD_MOLS: q for N2O/SF6, f for the halocarbon solvents).
ALL_ANALYTES: dict[str, str] = {
    gas: channel
    for gas, channel in (gc.rsplit("_", 1) for gc in ALL_GASES)
}


def resolve_analyte(batch: CATS_batch, analyte: str, channel: str | None) -> tuple[int, str]:
    """Resolve an analyte display name (+ optional channel) to (pnum, channel).

    Exits with a helpful message if the analyte is unknown or the channel is
    required but missing/ambiguous.
    """
    rows = batch.db.doquery(
        "SELECT display_name, param_num, channel FROM hats.analyte_list "
        f"WHERE inst_num = {batch.inst_num} "
        f"AND display_name = '{analyte}';"
    ) or []
    if not rows:
        available = sorted(batch.analytes.keys())
        sys.exit(
            f"No analyte named {analyte!r} for CATS-{batch.site.upper()}. "
            f"Available: {available}"
        )

    by_channel = {(r['channel'] or None): int(r['param_num']) for r in rows}
    if channel is not None:
        channel = channel.lower().strip()
        if channel not in by_channel:
            sys.exit(
                f"Analyte {analyte!r} has no channel {channel!r} on "
                f"CATS-{batch.site.upper()}. Available channels: "
                f"{sorted(c for c in by_channel if c)}"
            )
        return by_channel[channel], channel

    if len(by_channel) > 1:
        sys.exit(
            f"Analyte {analyte!r} is present on multiple channels for "
            f"CATS-{batch.site.upper()}: {sorted(c for c in by_channel if c)}. "
            f"Pass --channel to disambiguate."
        )
    (only_channel, pnum), = by_channel.items()
    return pnum, only_channel


def _scope_mf_nums(batch: CATS_batch, pnum: int, channel: str, start: str, end: str) -> set[int]:
    """Every mf_num loaded for this analyte/window -- the sync scope for the
    delete half of the idempotent apply (not just the currently-flagged rows,
    so a point that stops being flagged loses its tag on the next run)."""
    df = batch.load_data(pnum, channel=channel, start_date=start, end_date=end, verbose=False)
    if df.empty or "mf_num" not in df:
        return set()
    return set(pd.to_numeric(df["mf_num"], errors="coerce").dropna().astype(int))


def recalc_mole_fractions(
    batch: CATS_batch, pnum: int, channel: str, start: str, end: str, verbose: bool = False,
) -> None:
    """Recompute weekly cal fits + mole fractions and upsert to DB.

    Mirrors cats_batch.py's `-i --fits` path (CATS_batch._process_one with
    args.fits=True, args.insert=True): update_fits() computes+upserts the
    weekly ng_response fits, then update_runs() applies them and
    upsert_mole_fractions() writes the resulting air/tank mole fractions.

    Rejecting rows (cal_step, baseline, or manual review) changes which
    points feed a week's response fit and which get a recomputed mole
    fraction, but applying a tag never triggers this itself -- see
    ../CLAUDE.md's calibrations/tag-propagation notes. Called for
    _RECALC_BEFORE_BUILD algorithms (cal_window) so their build() always
    reads mole fractions consistent with the CURRENT rejection state,
    including any cal_step/baseline tags applied in an earlier invocation.
    """
    fits, scale_num, ref_serial, channel_str = batch.update_fits(
        pnum, channel=channel, start_date=start, end_date=end, verbose=verbose,
    )
    fits_override = None
    if not fits.empty:
        fits_override = fits
        if scale_num is not None:
            batch._upsert_fits(fits, scale_num, ref_serial, channel_str, verbose=verbose)

    df = batch.update_runs(
        pnum, channel=channel, start_date=start, end_date=end, verbose=verbose,
        fits_override=fits_override,
    )
    if df.empty:
        return
    empty_peak = (
        pd.to_numeric(df["height"], errors="coerce").eq(0)
        & pd.to_numeric(df["retention_time"], errors="coerce").eq(0)
    )
    batch.upsert_mole_fractions(df.loc[~empty_peak])


def sync_tags(
    batch: CATS_batch,
    algo_name: str,
    pnum: int,
    channel: str,
    start: str,
    end: str,
    dry_run: bool = False,
    batch_size: int = 1000,
    **algo_kwargs,
) -> dict:
    """Idempotently sync one algorithm's tag_num for one analyte/window."""
    algo = ALGORITHMS[algo_name]
    scope = _scope_mf_nums(batch, pnum, channel, start, end)
    flagged = algo.build(batch, pnum, channel, start, end, **algo_kwargs)
    flagged_mf_nums = (
        set(pd.to_numeric(flagged["mf_num"], errors="coerce").dropna().astype(int))
        if not flagged.empty and "mf_num" in flagged else set()
    )
    # Never tag something outside this window's own scope.
    flagged_mf_nums &= scope

    summary = {
        "algo": algo_name, "tag_num": algo.tag_num,
        "scope": len(scope), "flagged": len(flagged_mf_nums),
    }
    if dry_run or not scope:
        return summary

    db = batch.db
    scope_list = sorted(scope)
    for i in range(0, len(scope_list), batch_size):
        chunk = scope_list[i:i + batch_size]
        placeholders = ",".join(["%s"] * len(chunk))
        db.doquery(
            "DELETE FROM hats.ng_insitu_mole_fraction_tags "
            f"WHERE tag_num = %s AND ng_insitu_mole_fraction_num IN ({placeholders})",
            [algo.tag_num, *chunk],
        )

    insert_sql = (
        "INSERT INTO hats.ng_insitu_mole_fraction_tags "
        "(ng_insitu_mole_fraction_num, tag_num) VALUES (%s, %s)"
    )
    flagged_list = sorted(flagged_mf_nums)
    for i in range(0, len(flagged_list), batch_size):
        chunk = flagged_list[i:i + batch_size]
        params = [(mf_num, algo.tag_num) for mf_num in chunk]
        db.doMultiInsert(insert_sql, params, all=True)

    return summary


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", default="brw")
    p.add_argument("--algo", default="cal_step", choices=[*ALGORITHMS, "all"])
    p.add_argument("--analyte", default="N2O",
                   help='Analyte display_name from hats.analyte_list (e.g. N2O), or "all".')
    p.add_argument("--channel", default=None,
                   help="GC channel (q, f, cc, ...). Required if --analyte is "
                        "ambiguous across channels. Ignored when --analyte all.")
    p.add_argument("--start", type=_parse_yyyymmdd, default=None,
                   help="Start date, YYYYMMDD (default: Jan 1 of the current year).")
    p.add_argument("--end", type=_parse_yyyymmdd, default=None,
                   help="End date, YYYYMMDD (default: now).")
    p.add_argument("--mad-multiplier", type=float, default=3.5,
                   help="cal_step: point-to-point log-response z-score threshold (default: 3.5)")
    p.add_argument("--secondary-multiplier", type=float, default=2.0,
                   help="cal_step: lower hysteresis threshold for a triggered point's "
                        "neighbors on the same cal port (default: 2.0)")
    p.add_argument("--max-gap-hours", type=float, default=24.0,
                   help="cal_step: max gap between flagged cal points in one period (default: 24)")
    p.add_argument("--min-cal-points", type=int, default=1,
                   help="cal_step: drop periods with fewer than this many rate-outlier cal points (default: 1)")
    p.add_argument("--scale-window-days", type=float, default=30.0,
                   help="cal_step: trailing local window (days) for the typical-noise scale (default: 30)")
    p.add_argument("--window-days", type=float, default=10.0,
                   help="cal_window: full window width, centered on each candidate (default: 10)")
    p.add_argument("--sigma-high", type=float, default=3.0,
                   help="cal_window: reference-tank sigmas above the local air median to flag (default: 3)")
    p.add_argument("--sigma-low", type=float, default=2.0,
                   help="cal_window: reference-tank sigmas below the local air median to flag (default: 2)")
    p.add_argument("--min-ref-points", type=int, default=4,
                   help="cal_window: minimum in-window reference-tank readings to trust ref_std (default: 4)")
    p.add_argument("--min-air-points", type=int, default=4,
                   help="cal_window: minimum other in-window air readings to trust air_median (default: 4)")
    p.add_argument("--dry-run", action="store_true", help="Report counts; write nothing")
    p.add_argument("--skip-recalc", action="store_true",
                   help="cal_window: skip the automatic cats_batch-style mole-fraction "
                        "recalc that normally runs first (use if fits/mole fractions are "
                        "already known current for this window).")
    args = p.parse_args()

    start_date = args.start or f"{datetime.today().year}-01-01"
    end_date = args.end or datetime.today().strftime("%Y-%m-%d")

    algos = list(ALGORITHMS) if args.algo == "all" else [args.algo]
    batch = CATS_batch(args.site)

    if args.analyte.lower() == "all":
        analytes = list(ALL_ANALYTES.items())
    else:
        analytes = [(args.analyte, args.channel)]

    algo_kwargs = {
        "cal_step": dict(
            mad_multiplier=args.mad_multiplier,
            secondary_multiplier=args.secondary_multiplier,
            max_gap_hours=args.max_gap_hours,
            min_cal_points=args.min_cal_points,
            scale_window_days=args.scale_window_days,
        ),
        "cal_window": dict(
            window_days=args.window_days,
            sigma_high=args.sigma_high,
            sigma_low=args.sigma_low,
            min_ref_points=args.min_ref_points,
            min_air_points=args.min_air_points,
        ),
    }

    results = []
    for algo_name in algos:
        for analyte, channel in analytes:
            pnum, resolved_channel = resolve_analyte(batch, analyte, channel)
            if algo_name in _RECALC_BEFORE_BUILD and not args.dry_run and not args.skip_recalc:
                recalc_mole_fractions(batch, pnum, resolved_channel, start_date, end_date)
            summary = sync_tags(
                batch, algo_name, pnum, resolved_channel, start_date, end_date,
                dry_run=args.dry_run, **algo_kwargs.get(algo_name, {}),
            )
            summary.update(analyte=analyte, channel=resolved_channel)
            results.append(summary)

    out = pd.DataFrame(results)
    if not out.empty:
        cols = ["algo", "analyte", "channel", "tag_num", "scope", "flagged"]
        print(out[cols].to_string(index=False))
    print("Dry run -- no writes." if args.dry_run else "Synced tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
