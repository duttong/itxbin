#!/usr/bin/env python3
"""Apply cal_method_viterbi.py's per-period method choices to the database.

cal_method_viterbi.py only detects and recommends -- it never writes
anything. This is the manual "apply" step: reads its CSV output (one row
per period, every period assigned a chosen method -- a DENSE table, unlike
cats_cal_method_qc.py's SPARSE episode list where most of the timeline is
untouched) and, for each contiguous run of the same chosen method (oldest
first), runs

    cats_set_mf_method.py --site <site> --start <run_start> [--end <run_end>] \\
        --gas <gas>_<channel> --method <chosen>

then, once, unless --skip-retag:

    cats_tagging.py --site <site> --algo cal_window --gas <gas>_<channel> \\
        --start <earliest run_start>

Bounded vs open-ended --end: every run except the LAST gets an explicit
--end (the day before the next run's start -- Viterbi's periods already
tile the requested range with no gaps, so this never leaves a hole).
The last run omits --end, so it naturally extends to cover any new data
ingested after this CSV was generated -- matching cats_apply_cal_method.py's
convention for its own final state.

Deliberately does NOT call cats_batch.py directly -- same reasoning as
cats_apply_cal_method.py: a plain recompute would leave the cal_window
(286) reject tag stale against the freshly-recomputed values, since it
never touches hats.ng_insitu_mole_fraction_tags. cats_tagging.py --algo
cal_window already performs an equivalent recalc internally
(recalc_mole_fractions()) before it retags, and that recalc reads
get_week_mf_method() per week -- which picks up whatever
cats_set_mf_method.py just set for that week -- so ONE call spanning the
whole affected range correctly recomputes every run's own method in a
single pass and retags.

--skip-retag exists because cal_window is not always wanted: BRW CHCl3 has
real seasonality that cal_window's local-median outlier test can mistake
for instrument noise (see cats_qc/CATS-processing.md and project memory) --
pass --skip-retag there and run cats_batch.py -i --fits by hand instead if
you need mole fractions recomputed without the retag.

There is no UNRESOLVED/skip concept here -- unlike cats_cal_method_qc.py,
every period in a Viterbi CSV already has a definite chosen candidate. The
closest analogue is the detection step's own "still rough" printout
(period-to-period steps that remain large even at the best available
choice) -- that's a manual-review flag on the INPUT, not something this
apply script encounters.

Usage::

    # Preview the exact commands without running them
    python3 cats_apply_viterbi.py --site brw --gas CCl4_f --input brw_ccl4_f_viterbi.csv --dry-run

    # Apply for real
    python3 cats_apply_viterbi.py --site brw --gas CCl4_f --input brw_ccl4_f_viterbi.csv

    # Apply without the cal_window retag (e.g. BRW CHCl3)
    python3 cats_apply_viterbi.py --site brw --gas CHCl3_f --input brw_chcl3_f_viterbi.csv --skip-retag
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _build_apply_plan(df: pd.DataFrame) -> list[dict]:
    """Turn a cal_method_viterbi.py period table into an ordered list of
    {'method', 'start_date', 'end_date'} runs -- one per contiguous stretch
    of the same 'chosen' value, oldest first. end_date is None for the
    last run (open-ended). Pure function (no subprocess/DB calls), so the
    ordering is unit-testable without a database.
    """
    df = df.sort_values("period_start").reset_index(drop=True)
    # format="mixed": most periods are plain dates, but a mid-week
    # cal-tank-swap split period carries a real HH:MM:SS -- a single
    # inferred format would choke on the mix (see test coverage).
    df["period_start"] = pd.to_datetime(df["period_start"], format="mixed")

    is_new_run = df["chosen"] != df["chosen"].shift(1)
    run_id = is_new_run.cumsum()
    runs = df.groupby(run_id).agg(
        method=("chosen", "first"),
        start=("period_start", "first"),
    ).reset_index(drop=True)

    plan = []
    for i, row in runs.iterrows():
        start_date = row["start"].strftime("%Y-%m-%d")
        if i + 1 < len(runs):
            end_date = (runs.loc[i + 1, "start"].normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            end_date = None
        plan.append({"method": row["method"], "start_date": start_date, "end_date": end_date})
    return plan


def _run(cmd: list[str], dry_run: bool) -> None:
    print("  $ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", required=True, help="CATS site code (e.g. brw).")
    p.add_argument("--gas", required=True,
                    help="Analyte_channel (e.g. CCl4_f) -- the Viterbi CSV carries no "
                         "gas/channel columns of its own (one CSV is always one analyte).")
    p.add_argument("--input", type=Path, required=True,
                    help="CSV produced by cal_method_viterbi.py --output.")
    p.add_argument("--skip-retag", action="store_true",
                    help="Apply the method changes but skip the final cats_tagging.py "
                         "--algo cal_window retag (e.g. BRW CHCl3 -- real seasonality "
                         "that cal_window can mistake for outliers).")
    p.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would run; execute nothing.")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print(f"{args.input}: no periods, nothing to apply.")
        return 0

    plan = _build_apply_plan(df)
    for step in plan:
        end_label = step["end_date"] or "now"
        print(f"APPLY {args.gas} {step['start_date']} -> {end_label}: {step['method']}")
        cmd = [
            sys.executable, str(HERE / "cats_set_mf_method.py"),
            "--site", args.site, "--start", step["start_date"],
            "--gas", args.gas, "--method", step["method"],
        ]
        if step["end_date"]:
            cmd += ["--end", step["end_date"]]
        _run(cmd, args.dry_run)

    if args.skip_retag:
        print(f"\n{'Dry run -- ' if args.dry_run else ''}"
              f"{len(plan)} run(s) applied, retag skipped (--skip-retag).")
        return 0

    earliest_start = plan[0]["start_date"]
    print(f"\nRETAG {args.gas} {earliest_start} -> now "
          f"(recomputes mole fractions for every method set above, then cal_window)")
    _run([
        sys.executable, str(HERE / "cats_tagging.py"),
        "--site", args.site, "--algo", "cal_window",
        "--gas", args.gas, "--start", earliest_start,
    ], args.dry_run)

    print(f"\n{'Dry run -- ' if args.dry_run else ''}{len(plan)} run(s) applied, 1 group retagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
