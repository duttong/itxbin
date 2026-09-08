#!/usr/bin/env python3
"""Apply cats_cal_tank_health_qc.py recommendations to the database.

cats_cal_tank_health_qc.py only detects and recommends -- it never writes
anything. This is the manual "apply" step, but unlike
cats_apply_cal_method.py it does NOT apply each recommendation from its
episode_start forward indefinitely: a tank-health episode is a TEMPORARY
outage (the tank was offline, dry, or noisy for that specific window), not
a permanent calibration-scheme change, so the applied method is bounded to
[episode_start, episode_end] only.

Deliberately does NOT reset the surrounding range to cal12 first -- unlike
cats_cal_method_qc.py's episodes (which describe the calibration scheme
that should apply from that date forward, so cats_apply_cal_method.py's
un-bounded --start naturally supersedes whatever came before), a tank
outage says nothing about what method is correct outside its own window.
Composing on top of whatever cats_apply_cal_method.py (or manual review)
already set elsewhere is the safe default: a resetting version would
silently undo that separate, already-reviewed work for every date not
covered by one of these episodes. Run cats_set_mf_method.py by hand first
if a date range genuinely needs to be forced back to cal12.

For each (gas, channel) group with at least one RESOLVED (non-UNRESOLVED)
episode, in order:

    1. For each RESOLVED episode, oldest first:
       cats_set_mf_method.py --start <episode_start> --end <episode_end> \\
           --method <recommendation>
       -- overwrites just that bounded window with the tank-health verdict.
       Episodes for the same (gas, channel) are not expected to overlap
       (each comes from a distinct _group_periods() episode) but if they
       did, oldest-first ordering means the later one wins.
    2. cats_tagging.py --algo cal_window --start <earliest episode_start>
       -- one retag covering from the earliest applied episode through now,
       recomputing mole fractions for every method set above (see
       cats_apply_cal_method.py's docstring for why this step -- not a
       plain cats_batch.py call -- is required to keep the cal_window (286)
       reject tag in sync).

UNRESOLVED episodes are skipped with a printed warning -- both tanks (or
neither clearly) looked bad that period, so no method choice reliably
avoids the problem; whatever method is already in effect continues to
apply. This script never guesses a method for an episode
cats_cal_tank_health_qc.py couldn't resolve.

Usage::

    # Preview the exact commands without running them
    python3 cats_apply_cal_tank_health.py --site brw --input brw_health_flags.csv --dry-run

    # Apply for real
    python3 cats_apply_cal_tank_health.py --site brw --input brw_health_flags.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _build_apply_plan(df: pd.DataFrame) -> list[dict]:
    """Turn a cats_cal_tank_health_qc.py episodes DataFrame into an ordered
    list of steps -- {'action': 'apply', ...} (one bounded override per
    RESOLVED episode) or {'action': 'skip', ...} for UNRESOLVED episodes.
    No reset/baseline step: composes on top of whatever method is already
    set for dates outside these episodes (see module docstring for why).
    Pure function (no subprocess/DB calls), so the ordering is
    unit-testable without a database.
    """
    df = df.sort_values("episode_start").reset_index(drop=True)

    plan = []
    for _, row in df.iterrows():
        start_date = pd.Timestamp(row["episode_start"]).strftime("%Y-%m-%d")
        end_date = pd.Timestamp(row["episode_end"]).strftime("%Y-%m-%d")

        if row["recommendation"] == "UNRESOLVED":
            plan.append({
                "action": "skip",
                "gas": row["gas"], "channel": row["channel"],
                "start_date": start_date, "end_date": end_date,
                "cal1_tank": row.get("cal1_tank"), "cal2_tank": row.get("cal2_tank"),
                "cal1_reasons": row.get("cal1_reasons"), "cal2_reasons": row.get("cal2_reasons"),
            })
            continue

        plan.append({
            "action": "apply",
            "gas": row["gas"], "channel": row["channel"], "pnum": int(row["pnum"]),
            "start_date": start_date, "end_date": end_date, "method": row["recommendation"],
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
                    help="CSV produced by cats_cal_tank_health_qc.py --output.")
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
                f"SKIP  {step['gas']} ({step['channel']}) "
                f"{step['start_date']} -> {step['end_date']}: UNRESOLVED -- "
                f"cal1_tank={step['cal1_tank']} ({step['cal1_reasons']}) "
                f"cal2_tank={step['cal2_tank']} ({step['cal2_reasons']}). "
                f"Whatever method was already in effect continues to apply."
            )
            n_skipped += 1
            continue

        print(f"APPLY {step['gas']} ({step['channel']}) "
              f"{step['start_date']} -> {step['end_date']}: {step['method']}")
        _run([
            sys.executable, str(HERE / "cats_set_mf_method.py"),
            "--site", args.site, "--start", step["start_date"], "--end", step["end_date"],
            "--gas", f"{step['gas']}_{step['channel']}",
            "--method", step["method"],
        ], args.dry_run)
        n_applied += 1

    for group in _retag_groups(plan):
        print(f"RETAG {group['gas']} ({group['channel']}) {group['start_date']} -> now "
              f"(recomputes mole fractions for every method set above, then cal_window)")
        _run([
            sys.executable, str(HERE / "cats_tagging.py"),
            "--site", args.site, "--algo", "cal_window",
            "--gas", f"{group['gas']}_{group['channel']}",
            "--start", group["start_date"],
        ], args.dry_run)

    print(f"\n{'Dry run -- ' if args.dry_run else ''}"
          f"{n_applied} episode(s) applied, {n_skipped} UNRESOLVED skipped, "
          f"{len(_retag_groups(plan))} group(s) retagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
