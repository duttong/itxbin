#!/usr/bin/env python3
"""Apply cats_cal_method_qc.py recommendations to the database.

cats_cal_method_qc.py only detects and recommends -- it never writes
anything. This is the manual "apply" step: reads its CSV output and, for
each RESOLVED episode (oldest first), runs

    cats_set_mf_method.py --site <site> --start-date <episode_start> \\
        --pnum <pnum> --channel <channel> --method <recommendation>

then, once per (gas, channel) that had at least one applied change,

    cats_tagging.py --site <site> --algo cal_window \\
        --analyte <gas> --channel <channel> --start <earliest episode_start>

cats_set_mf_method.py --start-date has no end -- it labels everything from
that date forward -- so applying oldest first means each later episode
naturally supersedes the previous one from its own start date on, with no
explicit "clear the old range" step needed.

Deliberately does NOT call cats_batch.py directly, even though that's what
actually recomputes mole fractions -- it never touches
hats.ng_insitu_mole_fraction_tags, so a plain recompute would leave the
cal_window (286) reject tag stale against the freshly-recomputed values
(cal_step/baseline don't need this: they're pure functions of raw
response/chromatogram data, unaffected by a method change). cats_tagging.py
--algo cal_window already performs an equivalent recalc internally
(recalc_mole_fractions(), the same update_fits/_upsert_fits/update_runs/
upsert_mole_fractions sequence) before it retags, and that recalc calls
get_week_mf_method() per week -- which picks up whatever
cats_set_mf_method.py just set for that week -- so ONE call spanning the
whole affected range correctly recomputes every episode's own method in a
single pass and retags, rather than a separate untagged recompute per
episode.

UNRESOLVED episodes are skipped with a printed warning -- whatever method
was already in effect continues to apply through that stretch. This script
never guesses a method for an episode cats_cal_method_qc.py couldn't
resolve; see its cal1_tank/cal2_tank columns for where to look instead.

Usage::

    # Preview the exact commands without running them
    python3 cats_apply_cal_method.py --site brw --input brw_sf6_calmethod.csv --dry-run

    # Apply for real
    python3 cats_apply_cal_method.py --site brw --input brw_sf6_calmethod.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ITXBIN = HERE.parent


def _build_apply_plan(df: pd.DataFrame) -> list[dict]:
    """Turn a cats_cal_method_qc.py episodes DataFrame into an ordered list
    of steps -- {'action': 'apply', ...} or {'action': 'skip', ...} for
    UNRESOLVED episodes. Pure function (no subprocess/DB calls), so the
    ordering is unit-testable without a database.
    """
    df = df.sort_values("episode_start").reset_index(drop=True)
    plan = []
    for _, row in df.iterrows():
        start_date = pd.Timestamp(row["episode_start"]).strftime("%Y-%m-%d")
        if row["recommendation"] == "UNRESOLVED":
            plan.append({
                "action": "skip",
                "gas": row["gas"], "channel": row["channel"], "start_date": start_date,
                "cal1_tank": row.get("cal1_tank"), "cal2_tank": row.get("cal2_tank"),
                "anchor_period_mid": row.get("anchor_period_mid"),
            })
            continue
        plan.append({
            "action": "apply",
            "gas": row["gas"], "channel": row["channel"], "pnum": int(row["pnum"]),
            "start_date": start_date, "method": row["recommendation"],
        })
    return plan


def _retag_groups(plan: list[dict]) -> list[dict]:
    """One retag step per (gas, channel) with >=1 applied step, spanning
    from that group's earliest applied start_date through now. Pure
    function over an already-built plan -- unit-testable without a
    database."""
    groups: dict[tuple[str, str], str] = {}
    for step in plan:
        if step["action"] != "apply":
            continue
        key = (step["gas"], step["channel"])
        if key not in groups or step["start_date"] < groups[key]:
            groups[key] = step["start_date"]
    return [
        {"gas": gas, "channel": channel, "start_date": start_date}
        for (gas, channel), start_date in sorted(groups.items())
    ]


def _run(cmd: list[str], dry_run: bool) -> None:
    print("  $ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", required=True, help="CATS site code (e.g. brw).")
    p.add_argument("--input", type=Path, required=True,
                    help="CSV produced by cats_cal_method_qc.py --output.")
    p.add_argument("--dry-run", action="store_true",
                    help="Print the commands that would run; execute nothing.")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    if df.empty:
        print(f"{args.input}: no episodes, nothing to apply.")
        return 0

    plan = _build_apply_plan(df)
    n_applied = n_skipped = 0
    for step in plan:
        if step["action"] == "skip":
            print(
                f"SKIP  {step['gas']} ({step['channel']}) {step['start_date']}: UNRESOLVED -- "
                f"check hats.scale_assignments around {step['anchor_period_mid']} for "
                f"cal1_tank={step['cal1_tank']} cal2_tank={step['cal2_tank']}. "
                f"Whatever method was already in effect continues to apply."
            )
            n_skipped += 1
            continue

        print(f"APPLY {step['gas']} ({step['channel']}) {step['start_date']} -> now: "
              f"{step['method']}")
        _run([
            sys.executable, str(ITXBIN / "cats_set_mf_method.py"),
            "--site", args.site, "--start-date", step["start_date"],
            "--pnum", str(step["pnum"]), "--channel", step["channel"],
            "--method", step["method"],
        ], args.dry_run)
        n_applied += 1

    for group in _retag_groups(plan):
        print(f"RETAG {group['gas']} ({group['channel']}) {group['start_date']} -> now "
              f"(recomputes mole fractions for every method set above, then cal_window)")
        _run([
            sys.executable, str(HERE / "cats_tagging.py"),
            "--site", args.site, "--algo", "cal_window",
            "--analyte", group["gas"], "--channel", group["channel"],
            "--start", group["start_date"],
        ], args.dry_run)

    print(f"\n{'Dry run -- ' if args.dry_run else ''}"
          f"{n_applied} episode(s) applied, {n_skipped} UNRESOLVED skipped, "
          f"{len(_retag_groups(plan))} analyte/channel group(s) retagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
