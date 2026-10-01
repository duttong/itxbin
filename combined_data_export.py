#!/usr/bin/env python3
"""
Write the HATS combined data sets from hats.ng_logos_combined_data as text
files with the same columns as the published /aftp/hats/<gas>/combined/
HATS_global_<gas>.txt files, but with an updated header and metadata.

Columns (unchanged): <P>_<gas>_YYYY, <P>_<gas>_MM, then NH, SH, Global and the
twelve background sites, each as <mean> <mean>_sd, and <P>_<gas>_Programs.
Differences: the Programs value is a flag string with one digit per row of
hats.ng_logos_combined_programs (bit_pos order); published files had 6 digits
(oldGC RITS otto CATS CCGG MSD).

Usage:
    python3 combined_data_export.py [gases] [-o DIR] [--compare]

Writes DIR/<P>_global_<gas>.txt (default DIR is ~/combined_data, never /aftp).
--compare diffs the columns/values against the published file.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'logosdata'))
sys.path.append('/ccg/src/db/')

from combined_data import CombinedConfig, load_program_lookup  # noqa: E402
import db_utils.db_conn as db_conn  # noqa: E402

TABLE = 'hats.ng_logos_combined_data'
AFTP_ROOT = Path('/aftp/hats')

SITES = ['alt', 'sum', 'brw', 'mhd', 'thd', 'nwr', 'kum', 'mlo', 'smo', 'cgo', 'psa', 'spo']
SITE_DESC = {
    'alt': 'Alert, North West Territories, Canada (82.5N, 62.3W) (Atmospheric Environment Service Site) (flask only)',
    'sum': 'Summit, Greenland (72.6N, 38.4W, 3210m)-GLASS FLASKS ONLY (flask and in insitu)',
    'brw': 'Pt. Barrow, Alaska, USA (71.3N, 156.6W) (flask and in insitu)',
    'mhd': 'Mace Head, Ireland (53N, 10W) (flask only)',
    'thd': 'Trinidad Head, USA (41N, 124W, 120m) (flask only)',
    'nwr': 'Niwot Ridge, Colorado, USA (40.052N, 105.585W) (University of Colorado site)(flask and in insitu)',
    'kum': 'Cape Kumukahi, Hawaii, USA (19.5N, 154.8W) (flask only)',
    'mlo': 'Mauna Loa, Hawaii, USA (19.5N, 155.6W)(flask and in insitu)',
    'smo': 'Cape Matatula, American Samoa (14.3S, 170.6W)(flask and in insitu)',
    'cgo': 'Cape Grim, Tasmania, Australia (40.7S, 144.8E) (CSIRO/Australia site) (flask only)',
    'psa': 'Palmer Station, Antarctica (64.6S, 64.0W) (GLASS FLASKS ONLY)',
    'spo': 'South Pole (90S)(flask and in insitu)',
}

UNITS = {'ppt': 'parts-per-trillion, ppt', 'ppb': 'parts-per-billion, ppb'}
MEAN_ORDER = ['NH', 'SH', 'Global']


def names_to_bits(names: str | None, order: list[str]) -> str:
    """Comma separated abbrs -> '0'/'1' string in the lookup's bit order."""
    on = set((names or '').split(','))
    return ''.join('1' if p in on else '0' for p in order)


def build_header(gas: str, cfg: dict, meta: dict, programs: list[dict], filename: str,
                 first_year: int, last_year: int, today: date) -> list[str]:
    short, p = meta['short'], meta['prefix']
    h = [
        f"File: {filename}",
        f"Date: {today:%Y-%m-%d}",
        "",
        "Data use policy see https://gml.noaa.gov/hats/hats_datause.html",
        "",
        f"{meta['name']} ({short}) hemispheric and global monthly means from the NOAA/GML",
        "halocarbons program.  The following data is a combined data set from two or more measurement programs.",
        "Monthly means from each program are combined with inverse-variance weights, gap-filled,",
        "and smoothed for each sampling location.  Hemispheric and global means are estimated",
        "from the site series using cosine-of-latitude weighting.",
        "",
        "This work was funded in part by the Atmospheric Chemistry Project of NOAA's Climate and Global Change Program.",
        "",
        "Citation:",
        f"{meta['authors']} ({today.year}),",
        f"   Combined Atmospheric {meta['name']} Dry Air Mole Fractions from the",
        f"   NOAA GML Halocarbons Sampling Network, {first_year}-{last_year}, Version: {today:%Y-%m-%d},",
        f"   https://doi.org/{meta['doi']}",
        "",
        "For contact information see https://gml.noaa.gov/about/stafflist.html",
        "",
        f"Calibration scale used:  {meta['scale']}",
        "More information about the calibration scale can be found at: https://gml.noaa.gov/ccl/",
        "",
        f"Monthly data are provided in {UNITS[cfg['units']]}.",
        "",
        "See (https://gml.noaa.gov/dv/site/?program=hats&active=1) more information on current",
        "background air halocarbon sampling sites.",
        "",
    ]
    h += [f"{s} = {SITE_DESC[s]}" for s in SITES]
    h += [
        "",
        "See https://gml.noaa.gov/obop/ for station statistics and personnel.",
        "",
        f"Columns: year, month, then the mean and 1-sigma error ({p}_<NH|SH|Global|site>_{short} and _sd)",
        "for the northern hemisphere, southern hemisphere, global mean and each site.",
        "nan = not a number or no data, and space(s) is the delimiter.",
        "",
        f"The last column ({p}_{short}_Programs) is an {len(programs)}-digit binary number listing the",
        "measurement programs used in the combined hemispheric and global mean for that month.",
        "A 1 means data from the program was used.  Digits are ordered left to right:",
        "   " + ", ".join(r['abbr'] for r in programs),
        "",
    ]
    h += [f"{r['abbr']} = {r['description']}" for r in programs]
    h += [
        "",
        "Not every program applies to every gas; its digit is then always 0.",
        "",
        "Error estimates are 1-standard deviation values derived from the measured noise of each",
        "program (from differences between programs at shared site-months), combined across",
        "programs, plus a term for smoothing and interpolation.  Months filled by interpolation",
        "carry an inflated error that grows with the distance to the nearest measurement.",
        "",
        "All programs are on the same NOAA scale and known biases between programs are corrected;",
        "small differences may remain.  When comparing site to site, be aware that not all sites are",
        "composed of the same programs.  The hemispheric and global means are our best measure of",
        "long-term trends for background air.",
        "",
    ]
    if meta.get('notes'):
        h += ["Notes:"] + textwrap.wrap(str(meta['notes']), 92) + [""]
    return ['#  ' + line if line else '#  ' for line in h]


def build_file(gas: str, df: pd.DataFrame, cfg: dict, meta: dict, programs: list[dict],
               today: date) -> tuple[str, str]:
    short, p = meta['short'], meta['prefix']
    df = df[df.location.isin(MEAN_ORDER + SITES)].copy()
    df['month'] = pd.to_datetime(df['month'])
    order = [r['abbr'] for r in programs]
    df['programs'] = [names_to_bits(n, order) for n in df['programs']]
    means = df.pivot(index='month', columns='location', values='mean')
    sds = df.pivot(index='month', columns='location', values='sd')
    # Like the published files, start and stop at the months that have a
    # hemispheric or global mean (a site alone, e.g. 1977-08 or the newest month, is left out).
    means = means.dropna(subset=[c for c in MEAN_ORDER if c in means], how='all')
    sds = sds.reindex(means.index)
    # Programs flag comes from the Global row, else any row that month.
    prog = df[df.location == 'Global'].set_index('month')['programs']
    prog = prog.reindex(means.index).fillna(
        df.drop_duplicates('month').set_index('month')['programs'].reindex(means.index))
    locs = MEAN_ORDER + SITES

    cols = [f'{p}_{short}_YYYY', f'{p}_{short}_MM']
    for loc in locs:
        cols += [f'{p}_{loc}_{short}', f'{p}_{loc}_{short}_sd']
    cols.append(f'{p}_{short}_Programs')

    def fmt(v):
        return f"{'nan':>10}" if pd.isna(v) else f"{v:10.3f}"

    lines = [' '.join(cols)]
    for month, row in means.iterrows():
        parts = [f"{month.year:4d} {month.month:3d}"]
        for loc in locs:
            m = row.get(loc, float('nan'))
            s = sds.loc[month].get(loc, float('nan')) if loc in sds else float('nan')
            parts += [fmt(m), fmt(s)]
        parts.append(f"{prog.get(month, '0' * len(order)):>10}")
        lines.append(' '.join(parts))

    filename = f'{p}_global_{short}.txt'
    header = build_header(gas, cfg, meta, programs, filename, means.index.min().year,
                          means.index.max().year, today)
    return filename, '\n'.join(header + lines) + '\n'


def compare(path_new: Path, path_pub: Path) -> None:
    def load(p):
        lines = p.read_text().splitlines()
        i = next(i for i, l in enumerate(lines) if not l.startswith('#'))
        cols = lines[i].split()
        df = pd.read_csv(p, sep=r'\s+', skiprows=i + 1, header=None, names=cols,
                         na_values=['nan'], dtype={cols[-1]: str})
        return cols, df
    cn, dn = load(path_new)
    cp, dp = load(path_pub)
    print(f"  columns identical: {cn == cp}")
    if cn != cp:
        print("   only new:", [c for c in cn if c not in cp], " only published:", [c for c in cp if c not in cn])
    n = dn.set_index(dn.columns[:2].tolist())
    q = dp.set_index(dp.columns[:2].tolist())
    common_c = [c for c in n.columns[:-1] if c in q.columns]
    d = (n[common_c] - q[common_c]).dropna(how='all')
    print(f"  rows new/published: {len(n)}/{len(q)}; overlap {len(d)}")
    for loc in ('Global', 'NH', 'SH'):
        c = next(c for c in common_c if f'_{loc}_' in c and not c.endswith('_sd'))
        x = d[c].dropna()
        if len(x):
            print(f"  {loc}: mean diff {x.mean():+.3f}, max |diff| {x.abs().max():.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gases', nargs='*', help="Gas keys; default all.")
    ap.add_argument('-o', '--outdir', default=str(Path.home() / 'combined_data'))
    ap.add_argument('--compare', action='store_true')
    args = ap.parse_args()

    config = CombinedConfig.load()
    gases = args.gases or [g for g, c in config.gases.items() if c.get('publish')]
    db = db_conn.HATS_ng()
    programs = load_program_lookup(db)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    today = date.today()
    for gas in gases:
        cfg = config.gases[gas]
        rows = db.doquery(f"SELECT location, month, mean, sd, n, programs FROM {TABLE} "
                          "WHERE gas = %s", [gas])
        df = pd.DataFrame(rows)
        name, text = build_file(gas, df, cfg, cfg['publish'], programs, today)
        (out / name).write_text(text)
        print(f"{gas}: wrote {out / name}")
        if args.compare:
            pub = AFTP_ROOT / cfg['gas_dir'] / 'combined' / name
            if pub.exists():
                compare(out / name, pub)
            else:
                print(f"  no published {pub}")


if __name__ == '__main__':
    main()
