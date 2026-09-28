#!/usr/bin/env python3
"""Find probable unrecorded refills in cal tanks used by IE3 and CATS.

A refill with no reftank.fill record leaves two contents under one fill
code. caldrift then fits one assignment (and a coef1 "drift") through both,
which is wrong for the period the tank was actually on an instrument
(ALM067679 at MLO, ALM-066023 at SMO/BRW/NWR, AAL070764-style reactive loss
excepted -- see below).

For each tank that has sat on an IE3/CATS cal or ref port, calibrations
within 30 days are grouped into one episode. A refill is reported when, in
two or more of the conservative tracers SF6, N2O, HCFC-22, HFC-134a and
CFC-12, an episode differs from that tracer's previous value in the same
fill by more than its threshold, and the next episode (if any) confirms
the new value. Tracers must be above an ambient-like floor, so scrubbed
and zero-air tanks are skipped. Reactive gases (CCl4, CH3CCl3, CH3Cl,
COS) are deliberately not tracers: they can fall in a cylinder without a
refill.

Usage::

    python3 cats_qc/tank_refill_check.py            # print the table
    python3 cats_qc/tank_refill_check.py --csv out.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hats_db import HATSdb

INSITU_INSTS = {236: 'IE3', 239: 'BRW', 240: 'SUM', 241: 'NWR',
                242: 'MLO', 243: 'SMO', 244: 'SPO'}
# Values at or below the floor are ignored (scrubbed/zero-air tanks).
FLOOR = {'SF6': 1, 'N2O': 100, 'HCFC22': 20, 'HFC134a': 3, 'CFC12': 50}
# Relative change that counts as a jump, well above calibration noise.
THRESHOLD = {'SF6': 0.04, 'N2O': 0.01, 'HCFC22': 0.05, 'HFC134a': 0.08, 'CFC12': 0.015}
EPISODE_DAYS = 30
MIN_TRACERS = 2


def load(db):
    ports = db.to_df(f"""
        SELECT inst_num, port_num, serial_number s, start_datetime
        FROM hats.ng_port_info
        WHERE inst_num IN ({','.join(map(str, INSITU_INSTS))})
        ORDER BY inst_num, port_num, start_datetime""")
    ports['start_datetime'] = pd.to_datetime(ports['start_datetime'])
    ports['end'] = (ports.groupby(['inst_num', 'port_num'])['start_datetime']
                    .shift(-1).fillna(pd.Timestamp.now()))
    serials = ports['s'].dropna().unique().tolist()
    in_list = ','.join(f"'{s}'" for s in serials)
    cals = pd.concat([
        db.to_df(f"""SELECT serial_number s, date, species sp, mixratio v
                     FROM hats.calibrations
                     WHERE serial_number IN ({in_list})
                       AND species IN ('CFC12', 'HCFC22', 'HFC134a')"""),
        db.to_df(f"""SELECT serial_number s, date, UPPER(species) sp, mixratio v
                     FROM reftank.calibrations
                     WHERE serial_number IN ({in_list})
                       AND LOWER(species) IN ('n2o', 'sf6')
                       AND inst IN ('HP', 'LGR2', 'stdgc', 'VC')"""),
    ])
    cals['v'] = cals['v'].astype(float)
    cals['date'] = pd.to_datetime(cals['date'])
    cals = cals[cals['v'] > cals['sp'].map(FLOOR).fillna(0)]
    fills = db.to_df(f"""SELECT serial_number s, date, code FROM reftank.fill
                         WHERE serial_number IN ({in_list}) ORDER BY date""")
    fills['date'] = pd.to_datetime(fills['date'])
    return ports, cals, fills


def fill_code_at(fills, when):
    earlier = fills[fills['date'] <= when]
    return earlier['code'].iloc[-1] if not earlier.empty else '?'


def find_refills(ports, cals, fills):
    rows = []
    for serial, tank_cals in cals.groupby('s'):
        tank_fills = fills[fills['s'] == serial].reset_index(drop=True)
        tank_cals = tank_cals.sort_values('date').copy()
        tank_cals['fill'] = [fill_code_at(tank_fills, d) for d in tank_cals['date']]
        new_episode = ((tank_cals['date'].diff() > pd.Timedelta(days=EPISODE_DAYS))
                       | (tank_cals['fill'] != tank_cals['fill'].shift()))
        tank_cals['ep'] = new_episode.cumsum().to_numpy()
        grouped = tank_cals.groupby('ep')
        episodes = tank_cals.groupby(['ep', 'sp'])['v'].median().unstack()
        episodes['fill'] = grouped['fill'].first()
        episodes['d0'] = grouped['date'].min()
        episodes['d1'] = grouped['date'].max()

        for i in range(1, len(episodes)):
            cur = episodes.iloc[i]
            before = episodes.iloc[:i]
            before = before[before['fill'] == cur['fill']]
            if before.empty:
                continue
            hits = []
            for sp, thr in THRESHOLD.items():
                if sp not in episodes or pd.isna(cur.get(sp)):
                    continue
                prev = before[sp].dropna() if sp in before else pd.Series(dtype=float)
                if prev.empty or abs(cur[sp] / prev.iloc[-1] - 1) <= thr:
                    continue
                later = episodes.iloc[i + 1:][sp].dropna()
                if not later.empty and abs(later.iloc[0] / cur[sp] - 1) > thr:
                    continue  # one-off blip, not confirmed by the next episode
                hits.append(f"{sp} {prev.iloc[-1]:.4g}->{cur[sp]:.4g}")
            if len(hits) < MIN_TRACERS:
                continue

            this_fill = tank_fills[tank_fills['code'] == cur['fill']]
            fill_start = this_fill['date'].iloc[0] if not this_fill.empty else pd.Timestamp('1900-01-01')
            next_fill = tank_fills[tank_fills['date'] > fill_start]
            fill_end = next_fill['date'].iloc[0] if not next_fill.empty else pd.Timestamp('2100-01-01')
            used = ports[(ports['s'] == serial) & (ports['start_datetime'] < fill_end)
                         & (ports['end'] > fill_start)]
            if used.empty:
                continue
            last_before = before.loc[before[[h.split()[0] for h in hits]].notna().any(axis=1), 'd1'].iloc[-1]
            rows.append({
                'tank': serial,
                'fill': cur['fill'],
                'last_cal_before': last_before.date(),
                'first_cal_after': cur['d0'].date(),
                'evidence': '; '.join(hits),
                'used_on': '; '.join(
                    f"{INSITU_INSTS[u.inst_num]} p{u.port_num} "
                    f"{u.start_datetime:%Y-%m}..{u.end:%Y-%m}" for u in used.itertuples()),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--csv', type=Path, help='also write the table to this CSV')
    args = parser.parse_args()
    ports, cals, fills = load(HATSdb())
    result = find_refills(ports, cals, fills)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 100)
    print(f"{len(result)} probable unrecorded refills in {result['tank'].nunique() if len(result) else 0} tanks")
    if len(result):
        print(result.to_string(index=False))
    if args.csv:
        result.to_csv(args.csv, index=False)


if __name__ == '__main__':
    main()
