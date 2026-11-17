"""Weekly cal-fit helpers for IE3 and CATS, shared by logos_data (desktop) and logos_data_service (web).

These were methods of logos_data.MainWindow (_ie3_week_fit, _ie3_cal_tank_coefs_for_run, _ie3_tank_assigned,
_ie3_tank_unc, _ie3_ref_pred_unc); they use only the instrument and the ie3_cal_test fitting code, so they live here
without Qt and the window's methods call them.
"""
import os
import sys

import numpy as np
import pandas as pd

_itxbin = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _itxbin not in sys.path:
    sys.path.insert(0, _itxbin)
try:
    from ie3_cal_test import (
        weekly_cal_fits as _ie3_weekly_cal_fits,
        weekly_tank_data as _ie3_weekly_tank_data,
        _fill_value_for_date as _ie3_fill_value_for_date,
    )
    IE3_CAL_AVAILABLE = True
except ImportError:
    IE3_CAL_AVAILABLE = False

def week_fit(inst, unflagged, pnum, method, weekly=None, coefs=None):
    """Compute the single-week cal fit for the given method from the
    unflagged cal/ref data. ``weekly`` and ``coefs`` may be passed in to
    avoid recomputing them (they must already be scoped to the same
    sub-period picked below -- see callers).

    A calendar week straddling a ref/cal1/cal2 tank swap is split into
    sub-periods by weekly_tank_data()/fit_periods() (same machinery as
    ie3_batch.py's update_fits()); this displays/saves the LAST such
    sub-period within the loaded window, matching "this week's fit" for
    a GUI that shows one week at a time -- for a non-boundary week
    there's only one period, so behavior is unchanged. Returns
    (fit_row or None, coefs dict for that period's tanks).
    """
    if not IE3_CAL_AVAILABLE or unflagged.empty:
        return None, (coefs or {})
    force_zero, single_port = inst.fit_params_for_method(method)
    if weekly is None:
        cal_ports = tuple(sorted({
            inst.CAL1_PORT, inst.CAL2_PORT,
            inst.STANDARD_PORT_NUM,
        }))
        weekly = _ie3_weekly_tank_data(inst, unflagged, cal_ports)
    if weekly.empty:
        return None, (coefs or {})
    period_start = weekly['week_start'].max()
    wk = weekly.loc[weekly['week_start'] == period_start]
    if coefs is None:
        coefs = tank_coefs_for_run(inst, wk, pnum)
    if not coefs:
        return None, coefs
    if force_zero:
        if single_port not in coefs:
            return None, coefs
        fits = _ie3_weekly_cal_fits(
            wk, {single_port: coefs[single_port]}, force_zero=True
        )
    else:
        if len(coefs) < 2:
            return None, coefs
        fits = _ie3_weekly_cal_fits(wk, coefs)
    if fits.empty:
        return None, coefs
    return fits.iloc[-1], coefs


def tank_coefs_for_run(inst, data, pnum):
    """Resolve cal-tank assignment histories for the tank(s) present in
    *data* -- a weekly_tank_data()/fit_periods() period table, already
    scoped to one fit sub-period by the caller (see _ie3_week_fit) and
    carrying a 'tank_serial' column per (port, period) row rather than
    raw per-injection 'analysis_datetime'. Requires exactly one serial
    per cal port within that scope; a genuinely mixed-tank slice (should
    not happen once data is period-scoped) is left without a fit for
    that port rather than blending two tanks' responses.
    """
    coefs = {}
    for port in (inst.CAL1_PORT, inst.CAL2_PORT):
        rows = data.loc[data['port'].eq(port)]
        if rows.empty:
            continue
        serials = rows['tank_serial'].dropna().unique()
        if len(serials) != 1:
            continue
        history = inst.scale_assignment_history(serials[0], pnum)
        if history:
            coefs[port] = history
    return coefs


def tank_assigned(inst, port, pnum, coefs, week_start):
    """Assigned mole fraction (coef0) for a standard port in this week.
    Cal tanks (including CATS, where CAL2_PORT == STANDARD_PORT_NUM) use
    the dated fill active on week_start; a genuinely separate ref port
    (IE3's port 5, never in coefs) falls back to ref_tank_coef0, passing
    week_start so it resolves the tank actually on port 5 at that date
    (tank_serials_for_dates) rather than whatever is on it today. Returns
    None if unavailable."""
    fills = (coefs or {}).get(port)
    if fills:
        return _ie3_fill_value_for_date(fills, pd.Timestamp(week_start), 'coef0')
    if port == inst.STANDARD_PORT_NUM:
        return inst.ref_tank_coef0(pnum, when=week_start)
    return None


def tank_unc(inst, port, pnum, coefs, week_start):
    """Uncertainty (unc_c0) of the assigned mole fraction for a standard
    port in this week. Mirrors _ie3_tank_assigned. Returns None if
    unavailable."""
    fills = (coefs or {}).get(port)
    if fills:
        return _ie3_fill_value_for_date(fills, pd.Timestamp(week_start), 'unc_c0')
    if port == inst.STANDARD_PORT_NUM:
        return inst.ref_tank_unc_c0(pnum, when=week_start)
    return None


def ref_pred_unc(fit_row, ref_x, force_zero):
    """Propagated uncertainty of the cal-fit-predicted y-value at ref_x,
    from the assigned-value uncertainties (uc0/uc1) of the cal tank(s)
    used in the fit. Returns None if the needed uncertainties aren't
    available.

    force_zero (single-tank-through-origin, y = slope*x): the fit's
    only uncertainty is unc_slope, so unc(y) = |ref_x| * unc_slope.

    Two-tank fit (line through (x0,y0±uc0) and (x1,y1±uc1)): y(x) is a
    linear interpolation between the two assigned values, so its
    uncertainty is the weighted quadrature sum of uc0 and uc1 at ref_x.
    """
    if force_zero:
        unc_slope = fit_row.get('unc_slope')
        if unc_slope is None or pd.isna(unc_slope):
            return None
        return abs(ref_x) * float(unc_slope)

    x0, x1 = fit_row.get('x0'), fit_row.get('x1')
    uc0, uc1 = fit_row.get('uc0'), fit_row.get('uc1')
    if any(v is None or pd.isna(v) for v in (x0, x1, uc0, uc1)):
        return None
    dx = x1 - x0
    if dx == 0:
        return None
    return float(np.hypot(uc0 * (x1 - ref_x), uc1 * (ref_x - x0)) / abs(dx))
