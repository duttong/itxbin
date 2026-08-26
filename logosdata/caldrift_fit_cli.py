#!/usr/bin/env python3
"""Standalone CLI wrapper around ccg_cal_db.Calibrations + ccg_calfit.fitCalibrations.

Prints one JSON line to stdout: the caldrift drift-fit for one tank/gas/fill,
in the same shape the desktop logos_tanks GUI computes for its single-tank
plot overlay (see _overlay_caldrift_fit in
/nfs/hats/gc/itxbin/logosdata/logos_tanks.py:1809-2009). Read-only: never
writes to the database. Intended to be exec()'d from PHP
(flask_logostanks_dev3.php) so the web prototype gets bit-for-bit parity
with the desktop tool's fit math instead of a reimplementation.

Must run under the prod6 conda environment, since ccg_cal_db/ccg_calfit
live under /ccg/src/python3/nextgen and are only importable there.

Usage:
    caldrift_fit_cli.py --tank SERIAL --gas SPECIES --fillcode CODE
                         --syslist 'M1,M3,M4' [--degree auto|mean|linear|quadratic]
                         [--include-flagged]
                         [--cal-level all|tertiary|secondary|primary|gravimetric]

Output (stdout, single JSON line):
    {"ok": true, "fit": {"tzero":..., "coef0":..., "coef1":..., "coef2":...,
                          "unc_c0":..., "unc_c1":..., "unc_c2":...,
                          "sd_resid":..., "n":..., "chisq":...,
                          "n_excluded":..., "n_wrong_level":...}}
    {"ok": false, "message": "..."}

Exit code is 0 for a successful fit (even if fit.n == 0 — caller checks
"ok"/"n"), non-zero only on an unexpected exception.
"""
import argparse
import json
import sys
import warnings

_CCG_NEXTGEN = "/ccg/src/python3/nextgen"
if _CCG_NEXTGEN not in sys.path:
    sys.path.append(_CCG_NEXTGEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tank", required=True)
    ap.add_argument("--gas", required=True)
    ap.add_argument("--fillcode", required=True)
    ap.add_argument("--syslist", required=True, help="comma-separated instrument ids, e.g. M1,M3,M4")
    ap.add_argument("--degree", default="auto", choices=["auto", "mean", "linear", "quadratic"])
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--cal-level", default="all",
                     choices=["all", "tertiary", "secondary", "primary", "gravimetric"],
                     help="restrict the fit to one hats.calibrations.cal_level "
                          "(default: all levels, unfiltered)")
    args = ap.parse_args()

    try:
        import ccg_cal_db
        import ccg_calfit
    except Exception as exc:
        print(json.dumps({"ok": False, "message": f"import failed: {exc}"}))
        return 1

    try:
        cals = ccg_cal_db.Calibrations(
            tank=args.tank,
            gas=args.gas,
            fillingcode=args.fillcode,
            syslist=args.syslist,
            database="hats",
            quiet=True,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "message": f"DB error: {exc}"}))
        return 0

    # Mirrors logos_tanks.py:1860-1880: unflagged ('.') cals only, unless
    # --include-flagged also allows manually flagged ('M') episodes; drop
    # 9'd/None mixratios and zero/blank-uncertainty rows (which make
    # ccg_calfit's 1/unc**2 weighting blow up to NaN and crash the fit).
    # Also require num >= 3 for instruments where num is a meaningful
    # quality signal, matching the plot's own calibrations query
    # (LOGOSTANKS_GetCalibrations / _fetch_calibration_df) — a
    # single/double-injection M4 episode that never appears as a plotted
    # point could otherwise silently skew the fit. The legacy 'm3' system
    # is exempted: it recorded num=1 on every row it ever wrote (its whole
    # 1994-2022 history), so for m3 that value carries no quality
    # information and a uniform threshold would discard all of it.
    allowed_flags = (".", "M") if args.include_flagged else (".",)
    candidates = [
        d for d in cals.cals
        if d.get("flag") in allowed_flags
        and d.get("mixratio") is not None
        and d.get("mixratio") > -800
        and (d.get("inst") == "m3" or (d.get("num") or 0) >= 3)
    ]

    # cal_level filter: applied only when the caller asked for a specific
    # level. A tank/fillcode's calibration history can mix cal_level values
    # over time (e.g. a PRS tank calibrated as 'secondary' for a while, then
    # 'tertiary' afterward, interleaved rather than a clean cutover) — since
    # the two levels aren't necessarily on the same footing, blending them
    # into one fit silently mixes methodologies. Left as "all" by default
    # for instruments where cal_level is never populated (M4/FE3/m3 are all
    # NULL in hats.calibrations) so this filter is a no-op for them. Counted
    # separately from n_excluded (data-quality drops) since this is a
    # deliberate scope choice, not a rejected/bad point.
    n_wrong_level = 0
    if args.cal_level != "all":
        level_filtered = [d for d in candidates if d.get("cal_level") == args.cal_level]
        n_wrong_level = len(candidates) - len(level_filtered)
        candidates = level_filtered

    fit_data = [
        d for d in candidates
        if d.get("episode_unc") is not None and d.get("episode_unc") > 0
    ]
    n_excluded = len(candidates) - len(fit_data)

    if not fit_data:
        print(json.dumps({
            "ok": True, "fit": None,
            "message": "No usable calibration points."
                       + (f" ({n_wrong_level} excluded by cal_level={args.cal_level!r}.)" if n_wrong_level else ""),
        }))
        return 0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = ccg_calfit.fitCalibrations(fit_data, degree=args.degree)
    except Exception as exc:
        print(json.dumps({"ok": False, "message": f"fit failed: {exc}"}))
        return 0

    if fit is None or fit.n < 1 or fit.coef0 <= -800:
        print(json.dumps({"ok": True, "fit": None, "message": "Fit did not converge."}))
        return 0

    print(json.dumps({
        "ok": True,
        "fit": {
            "tzero":     fit.tzero,
            "coef0":     fit.coef0,
            "coef1":     fit.coef1,
            "coef2":     fit.coef2,
            "unc_c0":    fit.unc_c0,
            "unc_c1":    fit.unc_c1,
            "unc_c2":    fit.unc_c2,
            "sd_resid":  fit.sd_resid,
            "n":         fit.n,
            "chisq":     fit.chisq,
            "n_excluded": n_excluded,
            "n_wrong_level": n_wrong_level,
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
