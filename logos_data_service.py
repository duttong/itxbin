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
    {"op": "chrom", "inst": "m4", "channel": null, "t": "2026-10-22 10:01:00",
     "data": true, "analyte": "HFC-134a", "mf_num": 123}   the chromatogram trace, peak window and
                                                 the stored integration (GCwerks' or logosint's)

    {"op": "tag", ...as load..., "mf_nums": [..], "tag": 141, "apply": true, "detrend": null}
    {"op": "comment", ...as load..., "mf_nums": [..], "comment": "text"}
    {"op": "save", ...as load..., "detrend": null}
    {"op": "copyall", ...as load..., "adds": {"<analysis_num>": [tags]}, "removes": {...}}

The last four change the database and need "role" rw or admin. They call the
same instrument and tagging code as logos_data. The account running the
service decides what it can write: its ~/.db_rw.cnf login.

Run on omi:  logos_data_service.py [--socket PATH] [--web-user apache] [--dry-run FILE]
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
import subprocess
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
import integ_source
from gcwerks_chromatogram import (GCWerksPeakIntegration, _peak_result_analyte_prefix, find_gcwerks_chromatogram,
                                  gcwerks_channel_number, gcwerks_focus_limits,
                                  gcwerks_ms_quantitation_mass, gcwerks_peak_window, gcwerks_run_integrations,
                                  read_gcwerks_chromatogram, read_gcwerks_ms_chromatogram)
from logos_tagging import (TagCRUDMixin, _INFO_TAG_DESCRIPTIONS, _INFO_TAG_NUMS, _TAG_LAYOUT,
                           _USER_REMOVABLE_AUTO_TAGS)

DEFAULT_SOCKET = '/dev/shm/logos_data.sock'
MAX_LINE = 1 << 20

_lock = threading.Lock()        # one request at a time: the instruments share DB connections
_instruments = {}
_started = time.time()


DRY_RUN = None      # with --dry-run: the file that statements which would change the database go to


class DryRunDB:
    """Stands in for an instrument's database connection: reads pass through,
    anything else is written to the --dry-run file and not run."""

    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    @staticmethod
    def _note(sql, params):
        with open(DRY_RUN, 'a') as f:
            f.write(json.dumps({'sql': ' '.join(str(sql).split()), 'params': params}, default=str) + '\n')

    def doquery(self, sql, params=None, *args, **kwargs):
        if str(sql).lstrip().split(None, 1)[0].upper() in ('SELECT', 'SHOW', 'WITH'):
            return self._db.doquery(sql, params, *args, **kwargs)
        self._note(sql, params)
        return []

    def doMultiInsert(self, sql, params, all=False, *args, **kwargs):
        if all:
            self._note(sql, params)
            return True
        return False        # keep batching; the caller's final all=True call is the one noted


# The page's instrument ids (logos_integ's, where it has the instrument) -> (class id, site).
PAGE_INSTS = {
    'm4': ('m4', None), 'fe3': ('fe3', None), 'bld1': ('bld1', None),
    'ie3-smo': ('ie3', 'smo'),
    'brw': ('cats', 'brw'), 'sum': ('cats', 'sum'), 'nwr': ('cats', 'nwr'), 'mlo': ('cats', 'mlo'),
    'spo': ('cats', 'spo'), 'smo_cats': ('cats', 'smo'),
}


def instrument_for(page_id, site=None):
    if page_id not in PAGE_INSTS:
        raise ValueError(f'unknown instrument {page_id!r}')
    if page_id not in _instruments:
        cls_id, site = PAGE_INSTS[page_id]
        cls = getattr(li, f'{cls_id.upper()}_Instrument')
        obj = cls(site=site) if site else cls()
        if DRY_RUN:
            obj.db = DryRunDB(obj.db)
            obj.doquery = obj.db.doquery
        _instruments[page_id] = obj
    return _instruments[page_id]


def is_insitu(inst):
    return inst.inst_id in ('ie3', 'cats')


def id_column(inst):
    """The DataFrame column holding each row's mole fraction primary key."""
    return _TagHost(inst)._tag_table_info()[3]


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
    'anum': ('analysis_num', 'num'), 'channel': ('channel', 'str'),
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
    idc = id_column(instrument)
    out['mfnum'] = _column(df[idc], 'num') if idc in df.columns else [None] * len(df)
    out['extra'] = {c: _column(df[c], 'num') for c in getattr(instrument, 'ADDITIONAL_DATA_COLUMNS', ()) if c in df.columns}
    return out


def tags_for(instrument, df):
    """{row position: [[tag_num, comment], ...]} from the instrument's tag table."""
    tag_table, tag_key, _mf_table, idc = _TagHost(instrument)._tag_table_info()
    if idc not in df.columns:
        return {}
    nums = pd.to_numeric(df[idc], errors='coerce')
    ids = sorted({int(n) for n in nums.dropna()})
    by_mf = {}
    for i in range(0, len(ids), 5000):
        rows = instrument.doquery(
            f'SELECT {tag_key} n, tag_num t, comment c FROM {tag_table} '
            f"WHERE {tag_key} IN ({','.join(map(str, ids[i:i + 5000]))}) ORDER BY tag_num") or []
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
            'instruments': sorted(_instruments)}


def op_init(req):
    """What the page needs once per instrument."""
    inst = instrument_for(req['inst'])
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logosdata', 'logos_data.conf'))
    section = f'instrument.{inst.inst_id}'
    by_type = inst.inst_id in ('m4', 'prs')      # what Samples/Standard autoscaling keys on, as logos_data
    return {
        'insitu': is_insitu(inst),
        'additional': [c for c in getattr(inst, 'ADDITIONAL_DATA_COLUMNS', ())],
        'autoscale': {
            'by': 'rt' if by_type else 'port',
            'exclude': list(getattr(inst, 'EXCLUDE', []) or []),
            'standard': getattr(inst, 'STANDARD_RUN_TYPE', None) if by_type else getattr(inst, 'STANDARD_PORT_NUM', None),
            'standard_ports': list(getattr(inst, 'AUTOSCALE_STANDARD_PORTS', None) or []),
        },
        'analytes': analyte_list(inst),
        'default_analyte': cfg.get(section, 'default_analyte', fallback=None),
        'run_types': [['Air Samples', None]] if is_insitu(inst) else [[k, v] for k, v in inst.RUN_TYPE_MAP.items()],
        'response': inst.response_type,
        'start': str(inst.start_date)[:8],
        'legend_chronological': bool(getattr(inst, 'LEGEND_CHRONOLOGICAL', False)),
        'standard_run_type': getattr(inst, 'STANDARD_RUN_TYPE', None),
        'marker_size': inst.BASE_MARKER_SIZE,
        'tags': tag_dictionary(inst),
        'auto_tags': AUTO_TAG_NUMS,
        'info_tags': sorted(_INFO_TAG_NUMS),
        'tag_layout': [[section, [list(e) for e in entries]] for section, entries in _TAG_LAYOUT],
        'removable_auto_tags': sorted(_USER_REMOVABLE_AUTO_TAGS),
        'dry_run': bool(DRY_RUN),
    }


def op_runs(req):
    """Every run in a date range with its run types: [[run_time, [types]], ...].  The in situ
    instruments have no runs: theirs are calendar months, ["YYYY-MM", []]."""
    inst = instrument_for(req['inst'])
    start, end = _date(req['start']), _date(req['end'])
    if is_insitu(inst):
        first = max(pd.Timestamp(start), pd.Timestamp(str(inst.start_date)[:8])).to_period('M')
        last = min(pd.Timestamp(end), pd.Timestamp.utcnow().tz_localize(None)).to_period('M')
        return {'chunks': True, 'runs': [[str(m), []] for m in pd.period_range(first, last, freq='M')] if first <= last else []}
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


def run_range(inst, run):
    """(start, end) for load_data: a run_time for both ends, or an in situ month's first and last second."""
    if is_insitu(inst):
        if not re.fullmatch(r'\d{4}-\d{2}', str(run)):
            raise ValueError(f'bad month {run!r}')
        month = pd.Period(run, freq='M')
        return (month.start_time.strftime('%Y-%m-%d %H:%M:%S'), month.end_time.strftime('%Y-%m-%d %H:%M:%S'))
    run = _date(run)
    return run, run


def _load(req):
    inst = instrument_for(req['inst'])
    start, end = run_range(inst, req['run'])
    df = inst.load_data(pnum=int(req['pnum']), channel=req.get('channel') or None,
                        start_date=start, end_date=end, verbose=False)
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
    t0 = time.time()
    inst, df = _recalculated(req)
    return _view(inst, df, calc_ms=round((time.time() - t0) * 1000, 1))


class _TagHost(TagCRUDMixin):
    """The tag table reads and writes of logos_data, for one instrument."""

    def __init__(self, instrument):
        self.instrument = instrument


# Tags a person may set by hand (the Sampling and Measurement sections), and the automatic
# ones they may only take off.
_HAND_TAGS = frozenset(t for section, entries in _TAG_LAYOUT if section != 'Automated Tags'
                       for e in entries for t in e[2:] if t)


def _may_write(req):
    if req.get('role') not in ('rw', 'admin'):
        raise PermissionError(f"{req.get('user') or 'you'} cannot change {req.get('inst')} data")


def _audit(req, what):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {req.get('user')} {req['op']} {req.get('inst')} "
          f"pnum={req.get('pnum')} ch={req.get('channel') or ''} run={req.get('run')} {what}", flush=True)


def _check_tag(tag, apply):
    tag = int(tag)
    if tag in _HAND_TAGS or (not apply and tag in _USER_REMOVABLE_AUTO_TAGS):
        return tag
    raise ValueError(f'tag {tag} cannot be {"set" if apply else "removed"} by hand')


def _own_mf_nums(df, wanted, idc):
    """The requested mole fraction rows, which must all be in this run and analyte."""
    have = {int(n) for n in pd.to_numeric(df[idc], errors='coerce').dropna()} if idc in df.columns else set()
    nums = sorted({int(n) for n in wanted or []})
    if not nums or not set(nums) <= have:
        raise ValueError('those points are not in this run; reload and try again')
    return nums


def _recalculated(req):
    """The run as it stands in the database, smoothed and with mole fractions recalculated
    in memory (with another smoothing method if asked): what Save would write."""
    inst, df = _load(req)
    if df is None:
        return inst, None
    if req.get('detrend') is not None:
        df['detrend_method_num'] = int(req['detrend'])
    df = inst.norm.merge_smoothed_data(df)
    df = inst.calc_mole_fraction(df)
    return inst, df.reset_index(drop=True)


def _view(inst, df, **more):
    if df is None:
        return {'points': None, **more}
    return {'points': points_json(inst, df), 'tags': tags_for(inst, df), **more}


def op_tag(req):
    """Put a tag on, or take it off, some of the run's points for this analyte.  Written at
    once, as in logos_data; the reply is the run recalculated but not saved."""
    _may_write(req)
    apply = bool(req['apply'])
    tag = _check_tag(req['tag'], apply)
    inst, df = _load(req)
    if df is None:
        raise ValueError('no data in that run')
    nums = _own_mf_nums(df, req.get('mf_nums'), id_column(inst))
    host = _TagHost(inst)
    if apply:
        host._insert_tag_for_mf_nums(nums, tag)
    else:
        host._delete_tag_for_mf_nums(nums, tag)
    _audit(req, f"tag={tag} {'on' if apply else 'off'} mf_nums={nums}")
    inst, df = _recalculated(req)
    return _view(inst, df, unsaved=True)


def op_comment(req):
    """Set the comment on every tag of the selected points (empty clears it)."""
    _may_write(req)
    inst, df = _load(req)
    if df is None:
        raise ValueError('no data in that run')
    nums = _own_mf_nums(df, req.get('mf_nums'), id_column(inst))
    comment = str(req.get('comment') or '').strip()[:255] or None
    _TagHost(inst)._save_comment_for_mf_nums(nums, comment)
    _audit(req, f'comment={comment!r} mf_nums={nums}')
    return {'tags': tags_for(inst, df)}


def _save(inst, df, pnum, response_id='own'):
    """Write a recalculated run's mole fractions and its calibration rows, as logos_data's Save."""
    if response_id == 'own':
        response_id = None
        if 'ng_response_id' in df.columns:
            val = df['ng_response_id'].iat[0]
            if pd.notna(val) and val != 0:
                response_id = int(val)
    inst.upsert_mole_fractions(df, response_id=response_id)
    inst.upsert_calibrations(df, int(pnum))


def op_save(req):
    """Save this analyte's run: recalculated mole fractions, smoothing method, calibrations."""
    _may_write(req)
    inst, df = _recalculated(req)
    if df is None:
        raise ValueError('no data in that run')
    _save(inst, df, req['pnum'])
    _audit(req, f"saved {len(df)} rows detrend={req.get('detrend')}")
    return op_load(req)


def op_copyall(req):
    """Copy this session's tag changes to every analyte of the same injections, then
    recalculate and save every analyte of the run (logos_data's Copy Tags to all Analytes)."""
    _may_write(req)
    inst, df = _load(req)
    if df is None:
        raise ValueError('no data in that run')
    in_run = {int(a) for a in pd.to_numeric(df['analysis_num'], errors='coerce').dropna()}

    def pending(d, apply):
        out = {}
        for anum, tags in (d or {}).items():
            if int(anum) in in_run and tags:
                out[int(anum)] = {_check_tag(t, apply) for t in tags}
        return out

    adds, removes = pending(req.get('adds'), True), pending(req.get('removes'), False)
    inst.update_flags_all_analytes(adds, removes)
    start, end = run_range(inst, req['run'])
    failed = []
    analytes = analyte_list(inst)
    for a in analytes:
        try:
            one = inst.load_data(pnum=a['pnum'], channel=a['channel'], start_date=start, end_date=end, verbose=False)
            if one is None or one.empty:
                continue
            one = inst.calc_mole_fraction(one)
            _save(inst, one, a['pnum'], response_id=None)       # as the desktop's copy-to-all saves
        except Exception as e:
            traceback.print_exc()
            failed.append(f"{a['name']}: {e}")
    _audit(req, f'copied adds={ {k: sorted(v) for k, v in adds.items()} } removes={ {k: sorted(v) for k, v in removes.items()} } '
                f'analytes={len(analytes)} failed={len(failed)}')
    out = op_load(req)
    out.update(analytes=len(analytes), failed=failed)
    return out


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def logosint_peaks(inst, req):
    """The stored logosint integrations of the channel this injection's analyte is on, as
    (this analyte's peak, every peak of the channel), or None where the stored results are
    GCwerks'.  logosint feeds a CATS instrument from hats.ng_integ_source's logosint_from on;
    the peak bounds are in hats.ng_insitu_peak_details and the retention time and height in
    ng_insitu_mole_fractions (what logos_integ.php draws for those runs)."""
    mf_num = req.get('mf_num')
    if inst.inst_id != 'cats' or mf_num is None:
        return None
    src = integ_source.current(int(inst.inst_num), inst.db)
    if not src.logosint:
        return None
    rows = inst.doquery(
        "SELECT m.num, a.analysis_time, m.retention_time AS rt, m.height AS ht, d.width AS w, "
        "d.start_time AS st, d.start_level AS sl, d.end_time AS et, d.end_level AS el, d.codes, d.source "
        "FROM hats.ng_insitu_mole_fractions m0 "
        "JOIN hats.ng_insitu_analysis a ON a.num = m0.analysis_num "
        "JOIN hats.ng_insitu_mole_fractions m ON m.analysis_num = m0.analysis_num AND m.channel = m0.channel "
        "LEFT JOIN hats.ng_insitu_peak_details d ON d.ng_insitu_mole_fraction_num = m.num "
        "WHERE m0.num = %s", [int(mf_num)]) or []
    if not rows or (src.logosint_from is not None and pd.Timestamp(rows[0]['analysis_time']) < src.logosint_from):
        return None
    peaks, mine = [], None
    for r in rows:
        pk = {k: _finite(r[k]) for k in ('rt', 'ht', 'w', 'st', 'sl', 'et', 'el')}
        if r.get('source') not in (None, 'logosint') or any(pk[k] is None for k in ('st', 'sl', 'et', 'el')):
            pk = None
        else:
            pk.update(codes=r.get('codes'), src='logosint')
            peaks.append(pk)
        if int(r['num']) == int(mf_num):
            mine = pk
    # A run logosint has no peak for is drawn without one: GCwerks' would describe a different integration.
    return mine, peaks


def stored_peak(inst, req, gc_dir, channel, path, start_time, analyte, elapsed, signal):
    """The stored integration of the analyte in this chromatogram, from the program whose
    results the instrument uses (logosint or GCwerks), in seconds: rt, ht, st, et, the signal's
    level at each end (sl, el), the baseline there (bs, be) and the half-height crossings (wl, wr)."""
    got = logosint_peaks(inst, req)
    if got is not None:
        peak, others = got
    else:
        runs = gcwerks_run_integrations(gc_dir, path, start_time)
        prefix = _peak_result_analyte_prefix({f'{p.analyte}_start_time' for p in runs}, analyte)
        as_peak = lambda p: {'rt': p.retention_time, 'ht': p.height, 'w': p.width, 'st': p.start_time, 'sl': p.start_level,
                             'et': p.end_time, 'el': p.end_level, 'src': 'gcwerks', 'name': p.analyte}
        others = [as_peak(p) for p in runs]
        peak = next((p for p in others if p['name'] == prefix), None)
        # The GC/MS row holds every ion's peaks; only this trace's own can be fused with it.
        if inst.inst_id == 'm4':
            others = [peak] if peak else []
    if peak is None:
        return None
    peak['bs'], peak['be'] = (logosint_baseline(inst, peak, gc_dir, channel, os.path.basename(str(path)))
                              or peak_baseline(peak, others, elapsed, signal))
    peak['wl'], peak['wr'] = half_height_crossings(peak, elapsed, signal)
    return peak


def peak_baseline(peak, others, t, y):
    """The baseline level at a peak's start and end.

    The stored start and end levels are the signal's at those times.  That is the baseline at a
    resolved end, but not at a valley shared with a fused neighbour or a shoulder (the baseline
    runs on under it, and the area is cut by a vertical drop to it) nor under a horizontal
    baseline (FH, BH, MH).  Under each crest the baseline is the signal less the stored height, so
    the line is checked against those points: the peaks joined by shared boundaries keep the line
    between the group's outer ends when it passes through them; otherwise an end above the line
    from the other end through the crests is a valley, and that line is the baseline.  As
    fusedBaselines() in omi logos_integ.js, with the group's own valleys given their drop too.
    The stored height is only good to a few percent of the crest, so smaller drops are missed:
    logosint_baseline() is tried first and this is the fallback (and the GC/MS's only source)."""
    usable = lambda p: (p.get('rt') is not None and p.get('ht') is not None and p['ht'] > 0 and p['st'] < p['rt'] < p['et'])
    bs, be = peak['sl'], peak['el']
    if not usable(peak) or len(t) < 2:
        return bs, be
    pool = sorted((p for p in others if usable(p) and p is not peak), key=lambda p: p['st'])
    group = [peak]
    grew = True
    while grew:
        grew = False
        for p in pool:
            if p in group:
                continue
            if abs(p['et'] - group[0]['st']) < 0.05:
                group.insert(0, p); grew = True
            elif abs(p['st'] - group[-1]['et']) < 0.05:
                group.append(p); grew = True
    level = lambda x: float(np.interp(x, t, y))
    pts = [(p['rt'], level(p['rt']) - p['ht'], 0.03 * p['ht']) for p in group]
    A = (group[0]['st'], group[0]['sl'], 0.03 * group[0]['ht'])
    B = (group[-1]['et'], group[-1]['el'], 0.03 * group[-1]['ht'])

    def through(P, Q):
        return lambda x: P[1] + (Q[1] - P[1]) * (x - P[0]) / ((Q[0] - P[0]) or 1.0)

    def anchored(P):
        num = sum((q[0] - P[0]) * (q[1] - P[1]) for q in pts)
        den = sum((q[0] - P[0]) ** 2 for q in pts)
        m = num / den if den else 0.0
        return lambda x: P[1] + m * (x - P[0])

    ab = through(A, B)
    if all(abs(q[1] - ab(q[0])) <= q[2] for q in pts):
        line = ab
    else:
        from_a, from_b = anchored(A), anchored(B)
        b_valley = B[1] - from_a(B[0]) > B[2]
        a_valley = A[1] - from_b(A[0]) > A[2]
        if b_valley and not a_valley:
            line = from_a
        elif a_valley and not b_valley:
            line = from_b
        elif a_valley and b_valley and len(pts) >= 2:
            line = through(pts[0], pts[-1])
        else:
            return bs, be
    return float(line(peak['st'])), float(line(peak['et']))


LOGOSINT = next((p for p in ('/hats/gc/logosint/logosint', '/nfs/hats/gc/logosint/logosint') if os.path.isfile(p)), None)


def logosint_baseline(inst, peak, gc_dir, channel, run):
    """The baseline at a stored peak's start and end as logosint integrates the run, or None.

    The result tables keep only the signal's level at each end; logosint also reports the
    baseline there, for named and unnamed peaks alike.  It is run on the chromatogram with the
    parameter set behind the stored results (logosint's own, or its copy of GCwerks' for
    GCwerks results, as logos_integ.php does for unnamed peaks), and its peak is used when it
    is the stored one (the crest within a quarter width and each end within 2 s: GCwerks and
    logosint can put an end a few points apart).  As gwBaselines() in omi logos_integ.js.
    None for the GC/MS, which logosint does not integrate, and on any failure."""
    if LOGOSINT is None or inst.inst_id == 'm4':
        return None
    pset = 'logosint' if peak.get('src') == 'logosint' else 'gcwerks'
    try:
        done = subprocess.run([LOGOSINT, 'peaks', '--dir', gc_dir, '--channel', str(int(channel)), '--run', run,
                               '--inst', str(int(inst.inst_num)), '--set', pset],
                              capture_output=True, text=True, timeout=20, cwd='/tmp')
        res = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if not isinstance(res, dict):
        return None
    rows = list((res.get('peaks') or {}).values()) + list(res.get('unnamed') or [])
    rt, w = peak.get('rt'), peak.get('w') or 0.0
    for r in rows:
        if len(r) <= 10 or abs(r[5] - peak['st']) > 2 or abs(r[7] - peak['et']) > 2:
            continue
        if rt is not None and abs(r[0] - rt) > max(1.0, 0.25 * w):
            continue
        bs = r[9] if r[9] is not None else r[6]
        be = r[10] if r[10] is not None else r[8]
        # An end logosint has off the baseline takes the level of its baseline line at the stored time.
        line = lambda x: bs + (be - bs) * (x - r[5]) / ((r[7] - r[5]) or 1.0)
        eps = 1e-4 * max(1.0, peak.get('ht') or 0.0)
        return (float(line(peak['st'])) if abs(bs - r[6]) > eps else peak['sl'],
                float(line(peak['et'])) if abs(be - r[8]) > eps else peak['el'])
    return None


def half_height_crossings(peak, t, y):
    """Where the signal crosses half height on each side of the crest, above the baseline, as
    GCwerks draws the width; (None, None) when they can't be found or don't match the stored width."""
    none = (None, None)
    if peak.get('rt') is None or not peak.get('ht') or peak['ht'] <= 0 or not peak['et'] > peak['st']:
        return none
    st, et, bs, be, ht = peak['st'], peak['et'], peak['bs'], peak['be'], peak['ht']
    above = y - (bs + (be - bs) * (t - st) / (et - st)) - ht / 2.0
    k = int(np.clip(np.searchsorted(t, peak['rt']), 0, len(t) - 1))
    if k > 0 and above[k] <= 0 < above[k - 1]:
        k -= 1
    if above[k] <= 0:
        return none
    i = j = k
    while i > 0 and t[i] > st and above[i] > 0:
        i -= 1
    while j < len(t) - 1 and t[j] < et and above[j] > 0:
        j += 1
    if above[i] > 0 or above[j] > 0:
        return none
    cross = lambda a, b: float(t[a] + (t[b] - t[a]) * above[a] / ((above[a] - above[b]) or 1.0))
    wl, wr = cross(i, i + 1), cross(j, j - 1)
    w = peak.get('w')
    if w and w > 0 and abs(wr - wl - w) > 0.25 * w:
        return none
    return wl, wr


def op_chrom(req):
    """The chromatogram behind an injection, as logos_data's Chrom View shows it: the trace
    (the analyte's quantitation ion on the GC/MS), the peak window, GCwerks' integration
    baseline and the limits that frame the peak.  c and f name it for logos_integ."""
    inst = instrument_for(req['inst'])
    channel = gcwerks_channel_number(inst.inst_id, req.get('channel'), getattr(inst, 'site', ''))
    gc_dir = str(inst.gc_dir)
    if not os.path.isdir(gc_dir) and os.path.isdir('/nfs' + gc_dir):      # omi mounts /hats under /nfs
        gc_dir = '/nfs' + gc_dir
    path = find_gcwerks_chromatogram(gc_dir, _date(req['t']), channel)
    out = {'c': channel, 'f': os.path.basename(str(path))}
    if not req.get('data'):
        return out
    analyte = str(req.get('analyte') or '')
    mass = None
    if inst.inst_id == 'm4':
        chrom = read_gcwerks_ms_chromatogram(path)
        window = gcwerks_peak_window(gc_dir, channel, chrom.start_time, analyte)
        mass = window.mass if window is not None else None
        if mass is None:
            mass = gcwerks_ms_quantitation_mass(gc_dir, analyte)
        if req.get('mass') is not None:
            mass = float(req['mass'])
        trace = chrom.trace_for_mass(float(mass) if mass is not None else 0.0)
        elapsed, signal, mass = trace.elapsed_seconds, trace.signal, float(trace.mass)
        out['masses'] = [float(m) for m in chrom.masses]
        out['ylabel'] = 'Ion signal (counts)'
    else:
        chrom = read_gcwerks_chromatogram(path)
        window = gcwerks_peak_window(gc_dir, channel, chrom.start_time, analyte)
        signal = chrom.signal
        elapsed = chrom.elapsed_seconds if chrom.elapsed_seconds is not None else (
            np.arange(signal.size) / chrom.sample_rate + chrom.inject_time_offset)
        out['ylabel'] = 'Signal'
    elapsed = np.asarray(elapsed, dtype=float)
    signal = np.asarray(signal, dtype=float)
    try:
        peak = stored_peak(inst, req, gc_dir, channel, path, chrom.start_time, analyte, elapsed, signal)
    except (OSError, ValueError):        # an absent or older result table: the trace is still worth showing
        peak = None
    integ = peak and GCWerksPeakIntegration(analyte, peak['st'], peak['sl'], peak['et'], peak['el'], path)
    out.update(start=f'{chrom.start_time:%Y-%m-%d %H:%M} UTC', mass=mass,
               x=[round(float(v) / 60.0, 5) for v in elapsed], y=[float(v) for v in signal])
    if window is not None:
        out['window'] = [window.center_seconds / 60.0, window.width_seconds / 60.0]
    if integ is not None:
        # x, y: the baseline as integrated.  sl, el: the signal's level at each end, which is off the
        # baseline at a fused (shoulder) or horizontal-baseline end: the page joins the two with a drop line.
        out['integ'] = {'x': [peak['st'] / 60.0, peak['et'] / 60.0], 'y': [peak['bs'], peak['be']],
                        'sl': peak['sl'], 'el': peak['el'], 'src': peak['src'], 'codes': peak.get('codes')}
        if peak.get('rt') and peak.get('ht') and peak['ht'] > 0:
            out['integ'].update(rt=peak['rt'] / 60.0, ht=peak['ht'])
            if peak.get('wl') is not None:
                out['integ'].update(wl=peak['wl'] / 60.0, wr=peak['wr'] / 60.0)
    focus = gcwerks_focus_limits(elapsed, signal, window, integ)
    if focus is not None:
        out['focus'] = {'x': [focus[0][0] / 60.0, focus[0][1] / 60.0], 'y': [float(focus[1][0]), float(focus[1][1])]}
    return out


OPS = {'ping': op_ping, 'init': op_init, 'runs': op_runs, 'load': op_load, 'smooth': op_smooth,
       'chrom': op_chrom, 'tag': op_tag, 'comment': op_comment, 'save': op_save, 'copyall': op_copyall}


READ_OPS = {'ping', 'init', 'runs', 'load', 'smooth', 'chrom'}


def _lost_connection(e):
    """True for a MySQL error meaning the connection itself is gone."""
    inner = e.args[0] if e.args and isinstance(e.args[0], BaseException) else e   # the wrapper nests it
    code = inner.args[0] if inner.args and isinstance(inner.args[0], int) else None
    return code in (2006, 2013, 2055)


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
                    try:
                        out = OPS[req['op']](req)
                    except Exception as e:
                        # A cached instrument's MySQL connection goes stale while the page sits idle
                        # (2006 gone away / 2013 lost connection). Reads reconnect and run once more;
                        # writes do not, as a commit may have gone through before the drop.
                        if not (_lost_connection(e) and req['op'] in READ_OPS):
                            raise
                        print(f"{req['op']}: {e}; reconnecting and retrying", flush=True)
                        _instruments.clear()
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
    ap.add_argument('--dry-run', metavar='FILE',
                    help='write statements that would change the database to FILE instead of running them')
    args = ap.parse_args()
    global DRY_RUN
    DRY_RUN = args.dry_run

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
