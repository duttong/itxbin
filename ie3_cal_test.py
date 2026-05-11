#!/usr/bin/env python3
"""Investigate IE3 calibration scenarios.

Plot normalized response time series for the reference tank (port 5) and the
two calibration tanks (ports 1, 9): raw points plus weekly (Mon-Mon) means
with ±1σ error bars.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from pathlib import Path

from logos_instruments import IE3_Instrument


CAL_PORTS = (1, 9)
REF_PORT = 5
PLOT_PORTS = (REF_PORT,) + CAL_PORTS


def split_display_name(name: str) -> tuple[str, str]:
    """Return (base, channel) for a display name like 'CFC12 (b)' → ('CFC12','b')."""
    if '(' in name and ')' in name:
        base = name.split(' (')[0]
        ch = name.split('(')[1].split(')')[0]
        return base, ch
    return name, ''


def resolve_pnum(
    instrument: IE3_Instrument, mol: str, channel: Optional[str]
) -> tuple[int, str]:
    """Resolve (mol, channel) → (pnum, display_name).

    Matches case-insensitively. Requires --channel only when the molecule
    name is ambiguous across channels.
    """
    mol_lc = mol.strip().lower()
    channel_lc = channel.strip().lower() if channel else None

    matches = []
    for display_name, pnum in instrument.analytes.items():
        base, ch = split_display_name(display_name)
        if base.lower() != mol_lc:
            continue
        if channel_lc is not None and ch.lower() != channel_lc:
            continue
        matches.append((display_name, int(pnum), ch))

    if not matches:
        available = sorted({split_display_name(k)[0] for k in instrument.analytes})
        raise typer.BadParameter(
            f"No analyte found for mol={mol!r} channel={channel!r}. "
            f"Available molecules: {', '.join(available)}"
        )
    if len(matches) > 1:
        chans = ', '.join(m[2] for m in matches if m[2])
        raise typer.BadParameter(
            f"Analyte {mol!r} is ambiguous across channels ({chans}). Pass --channel."
        )

    display_name, pnum, _ = matches[0]
    return pnum, display_name


def filter_tanks(df: pd.DataFrame) -> pd.DataFrame:
    tanks = df[df['port'].isin(PLOT_PORTS)].copy()
    tanks = tanks[tanks['normalized_resp'].notna()]
    return tanks[tanks['rejected'] == 0]


def drop_first_in_burst(df: pd.DataFrame, max_gap_minutes: int = 60) -> pd.DataFrame:
    """Drop the first sample of each cal-tank burst (ports 1, 9).

    A burst is a run of same-port injections with consecutive gaps smaller
    than max_gap_minutes; within a single cal event same-port samples are
    spaced ~30 min while gaps between events are several hours.
    """
    if df.empty:
        return df
    cal_mask = df['port'].isin(CAL_PORTS)
    other = df[~cal_mask]
    kept = [other]
    gap = pd.Timedelta(minutes=max_gap_minutes)
    for port in CAL_PORTS:
        pdf = df[df['port'] == port].sort_values('analysis_datetime').copy()
        if pdf.empty:
            continue
        gaps = pdf['analysis_datetime'].diff()
        burst_id = (gaps.isna() | (gaps > gap)).cumsum()
        rank = pdf.groupby(burst_id).cumcount()
        kept.append(pdf[rank > 0])
    return pd.concat(kept).sort_values('analysis_datetime')


def weekly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mon-Sun week; week_start is the Monday of that week."""
    df = df.copy()
    df['week_start'] = df['analysis_datetime'].dt.tz_localize(None).dt.to_period('W-SUN').dt.start_time
    return (
        df.groupby(['port', 'port_label', 'week_start'])['normalized_resp']
          .agg(mean='mean', std='std', count='count')
          .reset_index()
    )


def plot_time_series(
    raw: pd.DataFrame,
    weekly: pd.DataFrame,
    display_name: str,
    site: str,
    savepath: Optional[str],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ports_present = sorted(raw['port'].unique())
    cmap = plt.get_cmap('tab10')
    color_for = {p: cmap(i) for i, p in enumerate(ports_present)}
    week_midpoint = pd.Timedelta(days=3, hours=12)

    for port in ports_present:
        color = color_for[port]
        port_label = raw.loc[raw['port'] == port, 'port_label'].iloc[0]

        r = raw[raw['port'] == port]
        w = weekly[weekly['port'] == port]

        ax.scatter(
            r['analysis_datetime'], r['normalized_resp'],
            s=8, alpha=0.3, color=color,
        )
        ax.errorbar(
            w['week_start'] + week_midpoint, w['mean'], yerr=w['std'],
            fmt='o', markersize=8, color=color, ecolor=color,
            markeredgecolor='dimgray', markeredgewidth=0.8,
            capsize=3, elinewidth=1.2,
            label=f"{port_label} weekly mean ±1σ",
        )

    ax.set_title(f"IE3 @ {site.upper()} — {display_name} normalized response")
    ax.set_xlabel("Date")
    ax.set_ylabel("normalized_resp")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"Saved plot → {savepath}")


def cal_tank_serials(instrument: IE3_Instrument) -> dict[int, str]:
    """Return {port: serial_number} for the cal tanks at the current site."""
    pc = instrument.port_config
    if pc is None:
        return {}
    mask = (pc['site_num'] == instrument.site_num) & (pc['port_num'].isin(CAL_PORTS))
    return {int(p): s for p, s in zip(pc.loc[mask, 'port_num'], pc.loc[mask, 'label'])}


def cal_tank_coefs(
    instrument: IE3_Instrument, pnum: int, serials: dict[int, str]
) -> dict[int, float]:
    """Look up coef0 from hats.scale_assignments for each cal tank serial."""
    coefs: dict[int, float] = {}
    for port, serial in serials.items():
        rec = instrument.scale_assignments(serial, pnum)
        if rec is None:
            print(f"  ⚠  No scale_assignments row for tank {serial} (port {port}), pnum {pnum}")
            continue
        coefs[port] = float(rec['coef0'])
        print(f"  port {port}: tank {serial} coef0 = {coefs[port]:.5g}")
    return coefs


def weekly_cal_fits(
    weekly: pd.DataFrame,
    coefs: dict[int, float],
    force_zero: bool = False,
) -> pd.DataFrame:
    """Fit y = m*x + b per week.

    Default (two-tank mode): requires both cal tanks in a week, fits a line
    through their (normalized_resp_mean, coef0) pairs.

    Single-tank mode (force_zero=True): requires only one tank per week and
    fits a line constrained through the origin — equivalent to a single-point
    cal using just that tank.
    """
    cal = weekly[weekly['port'].isin(coefs.keys())].copy()
    cal['mole_fraction'] = cal['port'].astype(int).map(coefs)

    rows = []
    for week, grp in cal.groupby('week_start'):
        grp = grp.sort_values('port')
        xs = grp['mean'].to_numpy()
        ys = grp['mole_fraction'].to_numpy()

        if force_zero:
            if len(xs) < 1:
                continue
            xs_fit = np.concatenate([[0.0], xs])
            ys_fit = np.concatenate([[0.0], ys])
        else:
            if len(xs) < 2:
                continue
            xs_fit = xs
            ys_fit = ys

        slope, intercept = np.polyfit(xs_fit, ys_fit, 1)
        rows.append({
            'week_start': week,
            'slope': slope,
            'intercept': intercept,
            'x0': xs[0], 'y0': ys[0],
            'x1': xs[-1], 'y1': ys[-1],
        })
    return pd.DataFrame(rows)


def plot_weekly_cal_fits(
    fits: pd.DataFrame,
    display_name: str,
    site: str,
    savepath: Optional[str],
    ref_coef0: Optional[float] = None,
    ref_label: Optional[str] = None,
    ref_xerr: Optional[float] = None,
    ref_yerr: Optional[float] = None,
    mode_label: str = "2-point",
) -> None:
    if fits.empty:
        print("No weeks with both cal tanks present; skipping cal-fit figure.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    xs_fit = np.concatenate([fits['x0'].to_numpy(), fits['x1'].to_numpy()])
    x_all = np.concatenate([xs_fit, np.array([1.0])]) if ref_coef0 is not None else xs_fit
    x_pad = 0.1 * (x_all.max() - x_all.min() or 1.0)
    x_line = np.array([x_all.min() - x_pad, x_all.max() + x_pad])

    cmap = plt.get_cmap('viridis')
    week_nums = fits['week_start'].map(mdates.date2num).to_numpy()
    norm = plt.Normalize(vmin=week_nums.min(), vmax=week_nums.max())

    for _, row in fits.sort_values('week_start').iterrows():
        color = cmap(norm(mdates.date2num(row['week_start'])))
        y_line = row['slope'] * x_line + row['intercept']
        ax.plot(x_line, y_line, color=color, alpha=0.7, linewidth=1.1)
        ax.plot(
            [row['x0'], row['x1']], [row['y0'], row['y1']],
            'o', color=color, markersize=5, alpha=0.9,
        )

    if ref_coef0 is not None:
        ref_text = ref_label or "ref tank"
        predicted_at_ref = fits['slope'] * 1.0 + fits['intercept']
        pmin = predicted_at_ref.min()
        pmax = predicted_at_ref.max()
        pmean = predicted_at_ref.mean()
        pstd = predicted_at_ref.std()
        xerr_txt = f"1σ={ref_xerr:.3g}" if ref_xerr is not None else "—"
        yerr_txt = f"sd_resid={ref_yerr:.3g}" if ref_yerr is not None else "—"
        legend_label = (
            f"{ref_text} predicted @ norm_resp=1.0\n"
            f"  min={pmin:.4g}  mean={pmean:.4g}  max={pmax:.4g}\n"
            f"  std={pstd:.4g}  (assigned={ref_coef0:.4g})\n"
            f"  xerr: {xerr_txt}   yerr: {yerr_txt}"
        )
        ax.errorbar(
            1.0, ref_coef0,
            xerr=ref_xerr, yerr=ref_yerr,
            marker='*', markersize=11, color='red',
            markeredgecolor='black', markeredgewidth=0.8,
            ecolor='black', elinewidth=1.0, capsize=3,
            linestyle='none', zorder=5,
            label=legend_label,
        )
        ax.legend(loc='upper left', fontsize=9)

    mean_A = fits['slope'].mean()
    mean_B = fits['intercept'].mean()
    std_A = fits['slope'].std()
    std_B = fits['intercept'].std()
    mean_fit_text = (
        f"Mean weekly fit (n={len(fits)})\n"
        f"  mf = {mean_A:.4g}·resp + {mean_B:.4g}\n"
        f"  A_std={std_A:.3g}  B_std={std_B:.3g}"
    )
    ax.text(
        0.98, 0.02, mean_fit_text,
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9),
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    cbar.set_label("Week start (Mon)")

    ax.set_title(
        f"IE3 @ {site.upper()} — {display_name} weekly {mode_label} cal fits "
        f"({len(fits)} weeks)"
    )
    ax.set_xlabel("normalized_resp (weekly mean per cal tank)")
    ax.set_ylabel("Mole fraction (coef0, ppt or ppb)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"Saved cal-fit plot → {savepath}")


def _suffix_path(path: str, suffix: str) -> str:
    p = Path(path)
    return str(p.with_name(f"{p.stem}{suffix}{p.suffix}"))


app = typer.Typer(help=__doc__, add_completion=False)


@app.command()
def main(
    mol: str = typer.Option(..., "--mol", help="Analyte name (e.g. N2O, CFC11)."),
    channel: Optional[str] = typer.Option(
        None, "--channel",
        help="Channel (a/b/c…) — required only when --mol is ambiguous.",
    ),
    site: str = typer.Option("smo", "--site", help="IE3 deployment site."),
    start: Optional[str] = typer.Option(
        None, "--start", help="Start date YYYY-MM-DD (default: instrument start_date)."
    ),
    end: Optional[str] = typer.Option(
        None, "--end", help="End date YYYY-MM-DD (default: today)."
    ),
    save: Optional[str] = typer.Option(
        None, "--save", help="Write figure to this path instead of showing it."
    ),
    remove_first: bool = typer.Option(
        False, "--remove-first",
        help="Drop the first sample in each burst of three cal-tank injections "
             "(useful when the CCl4 regulator needs more flush time).",
    ),
    cal1: bool = typer.Option(
        False, "--cal1",
        help="Fit through origin using only port 9 (single-point cal).",
    ),
    cal2: bool = typer.Option(
        False, "--cal2",
        help="Fit through origin using only port 1 (single-point cal).",
    ),
):
    if cal1 and cal2:
        raise typer.BadParameter("--cal1 and --cal2 are mutually exclusive.")

    inst = IE3_Instrument(site=site)
    pnum, display_name = resolve_pnum(inst, mol, channel)
    print(f"Resolved {mol!r} → pnum={pnum} ({display_name})")

    if start is None:
        s = inst.start_date
        start = f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 else s
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    df = inst.load_data(
        pnum=pnum,
        channel=channel,
        start_date=start,
        end_date=end,
        verbose=True,
    )
    if df.empty:
        print("No data loaded; nothing to plot.")
        raise typer.Exit(code=1)

    tanks = filter_tanks(df)
    if tanks.empty:
        print(f"No data for ports {PLOT_PORTS} in the loaded range.")
        raise typer.Exit(code=1)

    if remove_first:
        before = (tanks['port'].isin(CAL_PORTS)).sum()
        tanks = drop_first_in_burst(tanks)
        after = (tanks['port'].isin(CAL_PORTS)).sum()
        print(f"--remove-first: dropped {before - after} first-in-burst cal samples "
              f"({before} → {after})")

    present = sorted(tanks['port'].unique())
    print(f"Tank observations: {len(tanks)} across ports {present}")

    weekly = weekly_aggregate(tanks)

    ts_path = save
    cal_path = _suffix_path(save, "_calfits") if save else None
    plot_time_series(tanks, weekly, display_name, site, savepath=ts_path)

    print("Cal tank coefficients:")
    serials = cal_tank_serials(inst)
    coefs = cal_tank_coefs(inst, pnum, serials)

    ref_serial = None
    ref_coef0 = None
    ref_sd_resid = None
    ref_resp_std = None
    pc = inst.port_config
    if pc is not None:
        rmask = (pc['site_num'] == inst.site_num) & (pc['port_num'] == REF_PORT)
        rows = pc.loc[rmask, 'label']
        if not rows.empty:
            ref_serial = rows.iat[0]
            sql = f"""
                SELECT coef0, sd_resid FROM hats.scale_assignments
                WHERE serial_number = '{ref_serial}'
                  AND scale_num = (
                    SELECT idx FROM reftank.scales
                    WHERE parameter_num = {pnum} AND current = 1
                  );
            """
            rec = inst.db.doquery(sql)
            if rec:
                ref_coef0 = float(rec[0]['coef0'])
                ref_sd_resid = (
                    float(rec[0]['sd_resid']) if rec[0]['sd_resid'] is not None else None
                )
                print(f"  ref port {REF_PORT}: tank {ref_serial} "
                      f"coef0={ref_coef0:.5g}  sd_resid={ref_sd_resid}")
            else:
                print(f"  ⚠  No scale_assignments for ref tank {ref_serial}, pnum {pnum}")

            ref_vals = tanks.loc[tanks['port'] == REF_PORT, 'normalized_resp']
            if len(ref_vals) > 1:
                ref_resp_std = float(ref_vals.std())
                print(f"  ref normalized_resp 1σ = {ref_resp_std:.4g} (n={len(ref_vals)})")

    if cal1 or cal2:
        use_port = 9 if cal1 else 1
        if use_port not in coefs:
            print(f"--cal{1 if cal1 else 2}: no scale_assignments for port {use_port}; skipping fit.")
        else:
            single = {use_port: coefs[use_port]}
            fits = weekly_cal_fits(weekly, single, force_zero=True)
            mode_label = f"single-point (port {use_port} through 0)"
            print(f"Fitting in {mode_label} mode")
            plot_weekly_cal_fits(
                fits, display_name, site, savepath=cal_path,
                ref_coef0=ref_coef0, ref_label=ref_serial,
                ref_xerr=ref_resp_std, ref_yerr=ref_sd_resid,
                mode_label=mode_label,
            )
    elif len(coefs) < 2:
        print("Need coef0 for both cal tanks to fit weekly lines; skipping.")
    else:
        fits = weekly_cal_fits(weekly, coefs)
        plot_weekly_cal_fits(
            fits, display_name, site, savepath=cal_path,
            ref_coef0=ref_coef0, ref_label=ref_serial,
            ref_xerr=ref_resp_std, ref_yerr=ref_sd_resid,
        )

    if not save:
        plt.show()


if __name__ == "__main__":
    app()
