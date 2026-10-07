#!/usr/bin/env python3
"""Compare the two ways of weighting the M* hemispheric and global means.

For a gas with an entry in ``gas_bins`` (gml_global_means_config.yaml) this
builds the monthly means three ways from the same site data:

  A  latitude   cos(latitude) bands, with the sites the config lists for the gas
                (gas_background_overrides, else background_sites)
  B  latitude   the same bands, restricted to the sites the bins use
  C  bins       Montzka's hand-chosen bins and weights

so that B - A is the effect of the site choice, C - B the effect of the
weighting scheme alone, and C - A the total.

    global_means_compare.py CH3Br CH3Cl CH3CCl3
    global_means_compare.py CH3Br --start 2005 --plot figs/
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'logosdata'))
sys.path.append('/ccg/src/db/db_utils')

from data_export import MstarGlobalMeansExporter  # noqa: E402
from global_means import GlobalMeansConfig  # noqa: E402
from hats_db import HATSdb  # noqa: E402

MEANS = ('Global', 'NH', 'SH')
LABELS = {'A': 'latitude, config sites', 'B': 'latitude, bin sites', 'C': 'bins'}


def parameter_num(db: HATSdb, gas: str) -> int:
    rows = db.doquery('SELECT param_num FROM hats.analyte_list '
                      'WHERE inst_num = 192 AND display_name = %s', [gas])
    if not rows:
        sys.exit(f'{gas}: not an M4 analyte in hats.analyte_list')
    return int(rows[0]['param_num'])


def build(db: HATSdb, gas: str, pnum: int, start: int, end: int,
          method: str, sites: list[str] | None = None) -> pd.DataFrame:
    """Monthly means for *gas* with the config switched to *method*."""
    cfg = GlobalMeansConfig.load()
    cfg.weighting_method = method
    if sites is not None:
        cfg.gas_background_overrides[cfg.gas_key(gas)] = sites
    exporter = MstarGlobalMeansExporter(db, gas, pnum, start, end, config=cfg)
    df = exporter.query_data()
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')[list(MEANS)]


def compare(db: HATSdb, gas: str, start: int, end: int):
    cfg = GlobalMeansConfig.load()
    cfg.weighting_method = 'bins'
    if not cfg.bins_for(gas):
        sys.exit(f'{gas}: no gas_bins entry in the config')
    bin_sites = cfg.sites_for(gas)
    pnum = parameter_num(db, gas)
    return {
        'A': build(db, gas, pnum, start, end, 'latitude'),
        'B': build(db, gas, pnum, start, end, 'latitude', bin_sites),
        'C': build(db, gas, pnum, start, end, 'bins'),
    }


def summarise(gas: str, res: dict[str, pd.DataFrame]) -> None:
    print(f'\n{gas}   months with all of A, B and C: '
          f'{len(pd.concat(res, axis=1).dropna())}')
    rows = []
    for name, (x, y) in {'B - A  site choice': ('B', 'A'),
                         'C - B  weighting': ('C', 'B'),
                         'C - A  total': ('C', 'A')}.items():
        diff = (res[x] - res[y]).dropna()
        for m in MEANS:
            d = diff[m]
            rows.append({'effect': name, 'mean': m, 'n': len(d), 'bias': d.mean(),
                         'mean|d|': d.abs().mean(), 'sd': d.std(),
                         'max|d|': d.abs().max()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    annual = pd.concat({k: v['Global'] for k, v in res.items()}, axis=1).dropna()
    annual = annual.groupby(annual.index.year).mean()
    annual['C-A'] = annual['C'] - annual['A']
    print('\nannual-mean Global (every 3rd year) and C - A:')
    print(annual.iloc[::3].round(3).to_string())


def plot(gas: str, res: dict[str, pd.DataFrame], outdir: Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    colours = {'A': 'tab:gray', 'B': 'tab:blue', 'C': 'tab:red'}
    for j, m in enumerate(MEANS):
        for k in 'ABC':
            axes[0, j].plot(res[k].index, res[k][m], color=colours[k], lw=1.2,
                            label=f'{k}: {LABELS[k]}')
        axes[0, j].set_title(f'{gas} {m}')
        axes[1, j].axhline(0, color='k', lw=0.6)
        axes[1, j].plot(res['B'].index, res['B'][m] - res['A'][m], color='tab:blue',
                        lw=1, label='B - A  site choice')
        axes[1, j].plot(res['C'].index, res['C'][m] - res['B'][m], color='tab:red',
                        lw=1, label='C - B  weighting')
        axes[1, j].plot(res['C'].index, res['C'][m] - res['A'][m], color='k',
                        lw=1.4, label='C - A  total')
        axes[1, j].set_ylabel('ppt')
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f'{gas}_means_compare.png'
    fig.savefig(path, dpi=120)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('gases', nargs='+', help='gases with a gas_bins entry, e.g. CH3Br')
    ap.add_argument('--start', type=int, default=1991)
    ap.add_argument('--end', type=int, default=2026)
    ap.add_argument('--plot', metavar='DIR', type=Path, help='write a PNG per gas here')
    args = ap.parse_args()

    db = HATSdb()
    for gas in args.gases:
        res = compare(db, gas, args.start, args.end)
        summarise(gas, res)
        if args.plot:
            print('wrote', plot(gas, res, args.plot))


if __name__ == '__main__':
    main()
