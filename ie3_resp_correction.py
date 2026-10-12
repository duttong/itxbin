#!/usr/bin/env python3
"""Build per-injection response corrections for IE3 from room/loop temperature.

While IE3 sat in its temporary SMO location (2026-05-11 onward), the AC unit
next to the GC cycles roughly every 22 minutes. Injections made at the warm
peak of each cycle read high, by an amount that depends on the compound
(H1211 > CFC-113 > CCl4 > CFC-11 > CFC-12; N2O not at all). That adds noise
and a positive skew to the halocarbon mole fractions. The oven and ECDs are
held to about 0.001 C, so they are not the cause.

For each analyte, a gradient-boosted model predicts the ref tank's (port 5)
injection-to-injection height deviation from the 1 Hz engineering data:
temp_top and that channel's loop temperature (temp_s1/2/3), each as the
deviation from its 31-minute centred mean, at lags 0-14 minutes before the
injection. The model is trained on the ref tank only, so it cannot learn
atmospheric signal. Every row in the window (air, ref and cal ports) gets

    factor = exp(-prediction / 100)

stored in hats.ng_insitu_resp_corrections. IE3_Instrument.load_data() and
load_cal_week_data() multiply height by factor before ref smoothing.

An analyte is written only if the held-out (week-blocked CV) ref noise drops
by at least --min-gain; otherwise its rows in the window are removed. Rows
whose injection has no engineering data get no factor (1.0). Training always
uses raw heights from ng_insitu_data_view, so reruns do not compound.

Only apply this to periods with the AC problem: on the calm Mar-Apr 2026 data
the same model adds 5-10% noise. Set DEFAULT_END when IE3 moves back; after
that the daily --new-only step writes nothing.

--new-only (run daily by ie3_ingest.py before ie3_batch.py) refits on the whole
window but writes factors only for rows that do not have one yet, so factors
already used for stored mole fractions never change. A full rerun (without
--new-only) refits and rewrites everything; follow it with the full recalc.

After writing, recompute mole fractions and monthly means:

    ie3_batch.py -p all -s 2605 -e end -i
    insitu_monthly_means_batch.py ie3 -i

Usage:
    python3 ie3_resp_correction.py                      # dry run, all analytes
    python3 ie3_resp_correction.py -p 114 -c c          # one analyte
    python3 ie3_resp_correction.py -i                   # (re)write all factors in the window
    python3 ie3_resp_correction.py --new-only -i        # daily (ie3_ingest): only rows without one
    python3 ie3_resp_correction.py --clear -i           # remove this model's rows
"""

from __future__ import annotations

import argparse
import glob
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from hats_db import HATSdb

warnings.filterwarnings('ignore', category=FutureWarning)

INST_NUM = 236
SITE = 'smo'
REF_PORT = 5
DEFAULT_START = '2026-05-11'
DEFAULT_END = None   # set to the date IE3 leaves the temporary location (AC period ends)
MODEL_TAG = 'ie3_temp_gbm_v1'
LAGS = (0, 2, 4, 6, 8, 10, 14)            # minutes before injection
HF_WINDOW = 31                             # minutes, centred rolling mean
LOOP_TEMP = {'a': 'temp_s1', 'b': 'temp_s2', 'c': 'temp_s3'}
ENG_COLS = ['ie3_time', 'temp_top', *LOOP_TEMP.values()]


def _read_eng(path):
    try:
        d = pd.read_csv(path, usecols=lambda c: c in ENG_COLS, low_memory=False)
        if 'ie3_time' not in d.columns:
            return None                    # early headerless files
        d['ie3_time'] = pd.to_datetime(d['ie3_time'], utc=True, format='mixed')
        return (d.set_index('ie3_time').apply(pd.to_numeric, errors='coerce')
                 .resample('1min').mean())
    except Exception as e:                 # unreadable/partial file
        print(f'  skipping {path}: {e}')
        return None


def load_eng_hf(start, end, workers=8):
    """1-minute engineering temperatures as deviations from a 31-min mean."""
    days = pd.date_range(start.normalize() - pd.Timedelta(days=1), end.normalize())
    files = sorted(f for d in days
                   for f in glob.glob(f'/hats/gc/{SITE}/{d:%y}/incoming/{d:%Y%m%d}/eng_*.csv*'))
    with ProcessPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(_read_eng, files) if p is not None]
    if not parts:
        return pd.DataFrame()
    eng = pd.concat(parts).sort_index()
    eng = eng[~eng.index.duplicated()].asfreq('1min')
    print(f'Engineering data: {len(files)} files, {eng.index.min()} to {eng.index.max()}')
    return eng - eng.rolling(HF_WINDOW, center=True, min_periods=20).mean()


def features(times, hf, channel):
    t = pd.DatetimeIndex(times).floor('min')
    X = {}
    for lag in LAGS:
        tl = t - pd.Timedelta(minutes=lag)
        X[f'top{lag}'] = hf['temp_top'].reindex(tl).to_numpy()
        X[f'loop{lag}'] = hf[LOOP_TEMP[channel]].reindex(tl).to_numpy()
    return pd.DataFrame(X)


def ref_noise(t, h):
    """Successive-difference noise (%) of a ref height series."""
    s = pd.Series(np.log(h), index=t).sort_index()
    return s.diff().std() * 100 / np.sqrt(2)


def new_model():
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                         max_leaf_nodes=15, min_samples_leaf=40)


class RespCorrection:
    def __init__(self, start, end, insert, min_gain, model_tag):
        self.db = HATSdb()
        self.start, self.end = start, end
        self.insert, self.min_gain, self.model_tag = insert, min_gain, model_tag

    def analytes(self):
        rows = self.db.doquery(f"""
            SELECT DISTINCT param_num, channel, display_name FROM hats.analyte_list
            WHERE inst_num = {INST_NUM} ORDER BY param_num, channel""")
        return [(int(r['param_num']), r['channel'], r['display_name']) for r in rows]

    def load(self, pnum, channel):
        rows = self.db.doquery(f"""
            SELECT mf_num, analysis_time, port, height, rejected
            FROM hats.ng_insitu_data_view
            WHERE inst_num = {INST_NUM} AND parameter_num = {pnum} AND channel = '{channel}'
              AND analysis_time BETWEEN '{self.start:%Y-%m-%d %H:%M:%S}' AND '{self.end:%Y-%m-%d %H:%M:%S}'
              AND height > 0 AND height <> -999""")
        df = pd.DataFrame(rows or [])
        if df.empty:
            return df
        df['analysis_time'] = pd.to_datetime(df['analysis_time'], utc=True)
        df['height'] = df['height'].astype(float)
        df['rejected'] = df['rejected'].fillna(0).astype(int)
        return df.sort_values('analysis_time').reset_index(drop=True)

    def fit(self, df, hf, channel):
        """Return (factors Series indexed like df, stats dict)."""
        X = features(df['analysis_time'], hf, channel)
        ref = (df['port'] == REF_PORT) & (df['rejected'] == 0)
        rs = df.loc[ref].set_index('analysis_time')['height']
        y = (np.log(rs / rs.rolling('12h', center=True).median()) * 100).to_numpy()
        Xr = X[ref.to_numpy()]
        ok = np.isfinite(y) & Xr.notna().all(axis=1).to_numpy()
        stats = {'n_ref': int(ok.sum())}
        weeks = df.loc[ref, 'analysis_time'].dt.isocalendar().week.to_numpy()[ok]
        n_splits = min(5, len(np.unique(weeks)))
        if ok.sum() < 200 or n_splits < 3:
            stats['skip'] = 'too little ref data'
            return None, stats
        oof = cross_val_predict(new_model(), Xr[ok], y[ok], groups=weeks,
                                cv=GroupKFold(n_splits))
        t_ref = rs.index[ok]
        h_ref = rs.to_numpy()[ok]
        before = ref_noise(t_ref, h_ref)
        after = ref_noise(t_ref, h_ref * np.exp(-oof / 100))
        stats.update(noise_before=before, noise_after=after, gain=1 - after / before)
        model = new_model().fit(Xr[ok], y[ok])
        good = X.notna().all(axis=1).to_numpy()
        factor = pd.Series(np.nan, index=df.index)
        factor[good] = np.exp(-model.predict(X[good]) / 100)
        stats.update(n_factor=int(good.sum()), n_rows=len(df),
                     factor_sd_pct=float(np.log(factor.dropna()).std() * 100))
        return factor, stats

    def delete_window(self, pnum, channel, only_model=True):
        model_str = f"AND rc.model = '{self.model_tag}'" if only_model else ''
        return self.db.doquery(f"""
            DELETE rc FROM hats.ng_insitu_resp_corrections rc
            JOIN hats.ng_insitu_mole_fractions mf ON mf.num = rc.ng_insitu_mole_fraction_num
            JOIN hats.ng_insitu_analysis a ON a.num = mf.analysis_num
            WHERE a.inst_num = {INST_NUM} AND mf.parameter_num = {pnum} AND mf.channel = '{channel}'
              AND a.analysis_time BETWEEN '{self.start:%Y-%m-%d %H:%M:%S}' AND '{self.end:%Y-%m-%d %H:%M:%S}'
              {model_str}""")

    def existing(self, df):
        """mf_nums in df that already have a factor."""
        mf = df['mf_num'].astype(int)
        rows = self.db.doquery(f"""
            SELECT ng_insitu_mole_fraction_num AS mf_num FROM hats.ng_insitu_resp_corrections
            WHERE ng_insitu_mole_fraction_num BETWEEN {mf.min()} AND {mf.max()}""") or []
        return {int(r['mf_num']) for r in rows}

    def write(self, df, factor):
        sql = """INSERT INTO hats.ng_insitu_resp_corrections
                 (ng_insitu_mole_fraction_num, factor, model) VALUES (%s, %s, %s)
                 ON DUPLICATE KEY UPDATE factor = VALUES(factor), model = VALUES(model),
                 created = CURRENT_TIMESTAMP"""
        params = [(int(m), float(f), self.model_tag)
                  for m, f in zip(df['mf_num'], factor) if np.isfinite(f)]
        for i in range(0, len(params), 5000):
            self.db.db.doMultiInsert(sql, params[i:i + 5000], all=True)
        return len(params)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-p', '--parameter-num', default='all', help="pnum or 'all'")
    ap.add_argument('-c', '--channel', help='channel (a, b, c)')
    ap.add_argument('-s', '--start', default=DEFAULT_START, help='window start (UTC date)')
    ap.add_argument('-e', '--end', default=DEFAULT_END,
                    help='window end (UTC date, default DEFAULT_END or now)')
    ap.add_argument('--min-gain', type=float, default=0.05,
                    help='minimum held-out ref noise reduction to write (default 0.05)')
    ap.add_argument('--model-tag', default=MODEL_TAG)
    ap.add_argument('--new-only', action='store_true',
                    help='write only rows without a factor; never change existing ones (daily use)')
    ap.add_argument('--clear', action='store_true', help="remove this model's rows in the window")
    ap.add_argument('-i', '--insert', action='store_true', help='write to the DB (default: dry run)')
    args = ap.parse_args()

    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC') if args.end else pd.Timestamp.now(tz='UTC')
    if args.new_only and DEFAULT_END and end < pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=60):
        print(f'AC correction window ended {end:%Y-%m-%d}; nothing to do')
        return
    rc = RespCorrection(start, end, args.insert, args.min_gain, args.model_tag)

    todo = rc.analytes()
    if args.parameter_num != 'all':
        todo = [a for a in todo if a[0] == int(args.parameter_num)]
    if args.channel:
        todo = [a for a in todo if a[1] == args.channel]

    if args.clear:
        for pnum, ch, name in todo:
            n = rc.delete_window(pnum, ch) if args.insert else 'dry run'
            print(f'{name} ({ch}): removed {n}')
        return

    t0 = time.time()
    hf = load_eng_hf(start, end)
    if hf.empty:
        raise SystemExit('No engineering data in window')

    print(f'\nWindow {start:%Y-%m-%d} to {end:%Y-%m-%d %H:%M}, model {args.model_tag}, '
          f'min gain {args.min_gain:.0%}, {"INSERT" if args.insert else "dry run"}\n')
    print(f'{"analyte":14s} {"n_ref":>6s} {"ref noise %":>18s} {"gain":>6s} '
          f'{"factor sd %":>11s} {"rows":>13s}  action')
    for pnum, ch, name in todo:
        df = rc.load(pnum, ch)
        label = f'{name} ({ch})'
        if df.empty:
            print(f'{label:14s} no data')
            continue
        factor, st = rc.fit(df, hf, ch)
        if factor is None:
            action = st['skip']
            if args.insert:
                rc.delete_window(pnum, ch)
            print(f'{label:14s} {st["n_ref"]:6d}  {action}')
            continue
        keep = st['gain'] >= args.min_gain
        if args.new_only:
            new = ~df['mf_num'].astype(int).isin(rc.existing(df)).to_numpy()
            if not keep:
                action = 'below gain, skip'
            elif args.insert:
                action = f'wrote {rc.write(df[new], factor[new])} new'
            else:
                action = f'would write {int(np.isfinite(factor[new]).sum())} new'
        elif args.insert:
            rc.delete_window(pnum, ch)
            action = f'wrote {rc.write(df, factor)}' if keep else 'below gain, cleared'
        else:
            action = 'would write' if keep else 'below gain, skip'
        print(f'{label:14s} {st["n_ref"]:6d} {st["noise_before"]:8.3f} -> {st["noise_after"]:6.3f} '
              f'{st["gain"]:6.1%} {st["factor_sd_pct"]:11.3f} {st["n_factor"]:6d}/{st["n_rows"]:<6d}  {action}')
    print(f'\nDone in {time.time() - t0:.0f}s')
    if args.insert and not args.new_only:
        print('Now run: ie3_batch.py -p all -s 2605 -e end -i && insitu_monthly_means_batch.py ie3 -i')


if __name__ == '__main__':
    main()
