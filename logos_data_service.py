#!/usr/bin/env python3
"""Calculation service behind the omi page hats/logos_data.php.

A long-running process that keeps pandas, statsmodels and the LOGOS
instrument classes loaded, so the page gets the same smoothing,
normalization and mole fractions as logos_data and the batch scripts
without paying Python's start-up on every request.

Protocol: one JSON object per line in, one JSON object per line out, over a
unix socket (default /dev/shm/logos_data.sock; /tmp is private to php-fpm on
omi). Only the web server's user and the user running the service may
connect (SO_PEERCRED).

The page decides who may do what (hats.omv2_users 'integ:<inst>' rights, as
logos_integ.php) and sends the signed-in person as "user" and their role on
the instrument as "role" (rw, admin or null = view only). Anything that
writes must require a role here too.

    {"op": "ping"}
    {"op": "init", "inst": "m4"}                 analytes, run types, tag names
    {"op": "runs", "inst": "m4", "start": "2026-10-01", "end": "2026-11-30"}
    {"op": "load", "inst": "m4", "pnum": 29, "channel": null, "run": "2026-10-22 09:28:00"}
    {"op": "smooth", ...as load..., "detrend": 5}   the run with another smoothing (preview)
    {"op": "chrom", "inst": "m4", "channel": null, "t": "2026-10-22 10:01:00"}

This version reads only: nothing here writes to the database.

Run on omi:  logos_data_service.py [--socket PATH] [--web-user apache]
"""

import argparse
import configparser
import json
import os
import pwd
import re
import socket
import socketserver
import struct
import sys
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'logosdata'))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors

import logos_instruments as li
from gcwerks_chromatogram import find_gcwerks_chromatogram, gcwerks_channel_number
from logos_tagging import _INFO_TAG_DESCRIPTIONS, _INFO_TAG_NUMS

DEFAULT_SOCKET = '/dev/shm/logos_data.sock'
MAX_LINE = 1 << 20

_lock = threading.Lock()        # one request at a time: the instruments share DB connections
_instruments = {}
_started = time.time()


def instrument_for(inst, site=None):
    key = (inst, site or '')
    if key not in _instruments:
        try:
            cls = getattr(li, f'{inst.upper()}_Instrument')
        except AttributeError:
            raise ValueError(f'unknown instrument {inst!r}')
        _instruments[key] = cls(site=site) if inst in ('ie3', 'cats') and site else cls()
    return _instruments[key]


# Tags applied by loaders and QC scripts, not by hand; a point rejected only by these is drawn
# with an x.  A copy of MainWindow.AUTO_TAG_NUMS in logosdata/logos_data.py: keep in step.
AUTO_TAG_NUMS = [316, 26, 25, 2, 32, 324, 328, 329, 286]

# name on the page: (DataFrame column, kind)
POINT_COLUMNS = {
    't': ('analysis_datetime', 'time'), 'run': ('run_time', 'time'), 'rt': ('run_type_num', 'num'),
    'port': ('port', 'num'), 'pidx': ('port_idx', 'num'), 'plabel': ('port_label', 'str'),
    'pcolor': ('port_color', 'color'), 'pmarker': ('port_marker', 'str'),
    'smoothed': ('smoothed', 'num'), 'ratio': ('normalized_resp', 'num'), 'mf': ('mole_fraction', 'num'),
    'rej': ('rejected', 'num'), 'det': ('detrend_method_num', 'num'), 'site': ('site', 'str'),
    'sample_id': ('sample_id', 'str'), 'pair_id': ('pair_id_num', 'num'), 'port_info': ('port_info', 'str'),
    'tank': ('tank_serial_num', 'str'), 'netp': ('net_pressure', 'num'), 'sample_t': ('sample_datetime', 'time'),
    'comments': ('status_comments', 'str'), 'flask_port': ('flask_port', 'num'), 'cal_date': ('cal_date', 'str'),
    'anum': ('analysis_num', 'num'), 'mfnum': ('ng_mole_fraction_num', 'num'), 'channel': ('channel', 'str'),
}


def _column(s, kind):
    if kind == 'time':
        t = pd.to_datetime(s, errors='coerce', utc=True)
        return [None if pd.isna(v) else v.strftime('%Y-%m-%d %H:%M:%S') for v in t]
    if kind == 'num':
        a = pd.to_numeric(s, errors='coerce').to_numpy(dtype=float)
        return [None if not np.isfinite(v) else (int(v) if v == int(v) else float(v)) for v in a]
    if kind == 'color':
        cache = {}
        out = []
        for v in s:
            k = v if isinstance(v, str) else tuple(v) if v is not None and not np.isscalar(v) else v
            if k not in cache:
                try:
                    cache[k] = mcolors.to_hex(k)
                except (ValueError, TypeError):
                    cache[k] = '#808080'
            out.append(cache[k])
        return out
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in ('', 'nan', 'None', 'NaT', '<NA>')
            else str(v) for v in s]


def points_json(instrument, df):
    """The run's rows as columns, for the page."""
    out = {}
    resp = instrument.response_type
    if resp in df.columns:
        out['resp'] = _column(df[resp], 'num')
    for name, (col, kind) in POINT_COLUMNS.items():
        if col in df.columns:
            out[name] = _column(df[col], kind)
    return out


def tags_for(instrument, df):
    """{row position: [tag_num, ...]} from the instrument's tag table (M4, FE3, BLD1)."""
    if 'ng_mole_fraction_num' not in df.columns:
        return {}
    nums = pd.to_numeric(df['ng_mole_fraction_num'], errors='coerce')
    ids = sorted({int(n) for n in nums.dropna()})
    if not ids:
        return {}
    rows = instrument.doquery(
        'SELECT ng_mole_fraction_num n, tag_num t, comment c FROM hats.ng_mole_fraction_tags '
        f"WHERE ng_mole_fraction_num IN ({','.join(map(str, ids))}) ORDER BY tag_num") or []
    by_mf = {}
    for r in rows:
        by_mf.setdefault(int(r['n']), []).append([int(r['t']), r['c'] or ''])
    return {str(i): by_mf[int(n)] for i, n in enumerate(nums) if pd.notna(n) and int(n) in by_mf}


_tagdict = None


def tag_dictionary(instrument):
    """tag_num -> [flag, name, reject]; read once."""
    global _tagdict
    if _tagdict is None:
        rows = instrument.doquery('SELECT tag_num, flag, name, reject FROM ccgg.tag_view') or []
        _tagdict = {str(int(r['tag_num'])): [r['flag'] or '', r['name'] or '', int(r['reject'] or 0)] for r in rows}
        for num, desc in _INFO_TAG_DESCRIPTIONS.items():        # the panel's own wording for info tags
            _tagdict.setdefault(str(num), ['', desc, 0])
    return _tagdict


def analyte_list(instrument):
    out = []
    for name, pnum in instrument.analytes.items():
        m = re.search(r'\((\w+)\)\s*$', name)
        out.append({'name': name, 'pnum': int(pnum), 'channel': m.group(1) if m else None})
    return out


def op_ping(req):
    return {'ok': True, 'pid': os.getpid(), 'up_s': round(time.time() - _started, 1),
            'instruments': [f'{i}{":" + s if s else ""}' for i, s in _instruments]}


def op_init(req):
    """What the page needs once per instrument."""
    inst = instrument_for(req['inst'], req.get('site'))
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logosdata', 'logos_data.conf'))
    section = f'instrument.{req["inst"]}'
    return {
        'analytes': analyte_list(inst),
        'default_analyte': cfg.get(section, 'default_analyte', fallback=None),
        'run_types': [[k, v] for k, v in inst.RUN_TYPE_MAP.items()],
        'response': inst.response_type,
        'start': str(inst.start_date)[:8],
        'legend_chronological': bool(getattr(inst, 'LEGEND_CHRONOLOGICAL', False)),
        'standard_run_type': getattr(inst, 'STANDARD_RUN_TYPE', None),
        'marker_size': inst.BASE_MARKER_SIZE,
        'tags': tag_dictionary(inst),
        'auto_tags': AUTO_TAG_NUMS,
        'info_tags': sorted(_INFO_TAG_NUMS),
    }


def op_runs(req):
    """Every run_time in a date range with its run types: [[run_time, [types]], ...]."""
    inst = instrument_for(req['inst'], req.get('site'))
    start, end = _date(req['start']), _date(req['end'])
    rows = inst.doquery(
        'SELECT run_time, GROUP_CONCAT(DISTINCT run_type_num) types FROM hats.ng_analysis '
        f"WHERE inst_num = {int(inst.inst_num)} AND run_time BETWEEN '{start}' AND '{end}' "
        'GROUP BY run_time ORDER BY run_time') or []
    return {'runs': [[pd.Timestamp(r['run_time']).strftime('%Y-%m-%d %H:%M:%S'),
                      [int(t) for t in str(r['types']).split(',') if t.strip().lstrip('-').isdigit()]]
                     for r in rows if r['run_time'] is not None]}


def _date(text):
    """A date or timestamp from the page, checked, as 'YYYY-MM-DD HH:MM:SS'."""
    return pd.Timestamp(str(text)).strftime('%Y-%m-%d %H:%M:%S')


def _load(req):
    inst = instrument_for(req['inst'], req.get('site'))
    run = _date(req['run'])
    df = inst.load_data(pnum=int(req['pnum']), channel=req.get('channel') or None,
                        start_date=run, end_date=run, verbose=False)
    if df is None or df.empty:
        return inst, None
    return inst, df.reset_index(drop=True)


def op_load(req):
    """One run of one analyte, as logos_data loads it."""
    t0 = time.time()
    inst, df = _load(req)
    t1 = time.time()
    if df is None:
        return {'points': None, 'load_ms': round((t1 - t0) * 1000)}
    out = {'points': points_json(inst, df), 'tags': tags_for(inst, df)}
    out['load_ms'] = round((t1 - t0) * 1000)
    out['tags_ms'] = round((time.time() - t1) * 1000)
    return out


def op_smooth(req):
    """The run recalculated with another smoothing method.  A preview: nothing is saved."""
    inst, df = _load(req)
    if df is None:
        return {'points': None}
    df['detrend_method_num'] = int(req['detrend'])
    t0 = time.time()
    df = inst.norm.merge_smoothed_data(df)
    df = inst.calc_mole_fraction(df)
    df = df.reset_index(drop=True)
    return {'points': points_json(inst, df), 'tags': tags_for(inst, df),
            'calc_ms': round((time.time() - t0) * 1000, 1)}


def op_chrom(req):
    """The GCwerks chromatogram behind an injection: its channel number and file name."""
    inst = instrument_for(req['inst'], req.get('site'))
    channel = gcwerks_channel_number(inst.inst_id, req.get('channel'), getattr(inst, 'site', ''))
    gc_dir = str(inst.gc_dir)
    if not os.path.isdir(gc_dir) and os.path.isdir('/nfs' + gc_dir):      # omi mounts /hats under /nfs
        gc_dir = '/nfs' + gc_dir
    path = find_gcwerks_chromatogram(gc_dir, _date(req['t']), channel)
    return {'c': channel, 'f': os.path.basename(str(path))}


OPS = {'ping': op_ping, 'init': op_init, 'runs': op_runs, 'load': op_load, 'smooth': op_smooth,
       'chrom': op_chrom}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        creds = self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i'))
        _pid, uid, _gid = struct.unpack('3i', creds)
        if uid not in self.server.allowed_uids:
            self.wfile.write(b'{"error": "not allowed"}\n')
            return
        while True:
            line = self.rfile.readline(MAX_LINE)
            if not line.strip():
                return
            t0 = time.time()
            try:
                req = json.loads(line)
                with _lock:
                    out = OPS[req['op']](req)
            except KeyError as e:
                out = {'error': f'missing or unknown {e}'}
            except Exception as e:      # the page shows it; the log has the trace
                traceback.print_exc()
                out = {'error': f'{type(e).__name__}: {e}'}
            out['svc_ms'] = round((time.time() - t0) * 1000, 1)
            self.wfile.write(json.dumps(out, allow_nan=False, separators=(',', ':')).encode() + b'\n')
            self.wfile.flush()


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--socket', default=DEFAULT_SOCKET)
    ap.add_argument('--web-user', default='apache', help='user the web server connects as')
    args = ap.parse_args()

    allowed = {os.getuid()}
    try:
        allowed.add(pwd.getpwnam(args.web_user).pw_uid)
    except KeyError:
        print(f'no user {args.web_user!r} here; only uid {os.getuid()} may connect', file=sys.stderr)

    # Refuse to replace a socket another running service is answering on.
    if os.path.exists(args.socket):
        probe = socket.socket(socket.AF_UNIX)
        try:
            probe.connect(args.socket)
            sys.exit(f'a service is already listening on {args.socket}')
        except OSError:
            os.unlink(args.socket)
        finally:
            probe.close()

    server = Server(args.socket, Handler)
    os.chmod(args.socket, 0o666)        # the uid check above is the gate
    server.allowed_uids = allowed
    print(f'logos_data_service pid {os.getpid()} on {args.socket}, uids {sorted(allowed)}', flush=True)
    try:
        server.serve_forever()
    finally:
        try:
            os.unlink(args.socket)
        except OSError:
            pass


if __name__ == '__main__':
    main()
