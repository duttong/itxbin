"""The logos_data Timeseries tab for the web page (omi hats/logos_data.php?view=ts), without Qt.

TimeseriesWidget (logos_timeseries.py) keeps its queries and dataset rules on
the widget, reading the analyte, years and sites from its Qt controls.
HeadlessTimeseries stands in for the widget: plain values take the controls'
place and every other attribute is borrowed from TimeseriesWidget, so the web
page plots exactly what the desktop figure does, from the same SQL.

logos_data_service.py calls the op_* functions here:

    {"op": "tsinit", "inst": "m4"}       analytes, sites (by latitude, with colours), datasets, exports
    {"op": "ts", "inst": "m4", "analyte": "CFC-11", "start": 2022, "end": 2026}
                                         every dataset for every site, flagged rows included (marked)
    {"op": "tstags", "inst": "m4", "mf_nums": [..]}            the tags on those mole fractions
    {"op": "tstag", "inst": "m4", "mf_nums": [..], "tag": 141, "apply": true}    (rw/admin)
    {"op": "tscomment", "inst": "m4", "mf_nums": [..], "comment": "text"}        (rw/admin)
    {"op": "tsrelstd", "inst": "m4", "analyte": "CFC-11", "start": 2022, "end": 2026}
                                         the Relative Stddev figure's flask means (flask runs, unrejected)
    {"op": "tsexport", "inst": "m4", "kind": "mstar_sel", "analyte": "CFC-11", "start": 2022, "end": 2026,
     "sites": [..], "all_analytes": false}    -> {"name": file name, "b64": contents (a .zip for several files)}

Tagging from the time series only writes tags, as on the desktop: mole
fractions and means follow at the next batch run.
"""

import base64
import io
import os
import re
import tempfile
import types
import zipfile

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import logos_timeseries as lts
from data_export import MstarDataExporter, MstarMonthlyExporter, MstarGlobalMeansExporter, FecdDataExporter

PALETTES = ('Latitude', 'tab10', 'Dark2')


class _Value:
    """A spin box's value()."""
    def __init__(self, v):
        self._v = int(v)

    def value(self):
        return self._v


class _Combo:
    """The analyte combo's currentText() (and the setters the widget calls on it)."""
    def __init__(self, text):
        self.text = text

    def currentText(self):
        return self.text

    def findText(self, text, *a):
        return 0

    def setCurrentIndex(self, i):
        pass


class _Check:
    """A site check box."""
    def __init__(self, site, on):
        self._site, self._on = site, on

    def text(self):
        return self._site

    def isChecked(self):
        return self._on


class HeadlessTimeseries:
    """TimeseriesWidget's query and dataset code with plain values for its controls."""

    def __init__(self, instrument, analyte=None, start=None, end=None, sites=None):
        self.instrument = instrument
        self._cached_df = None
        self._last_query_params = None
        self._cached_insitu_df = None
        self._last_insitu_params = None
        inst_analytes = instrument.analytes or {}
        # As TimeseriesWidget.__init__: IE3/CATS show one date-aware preferred stream per compound.
        self.force_preferred_channel = instrument.inst_id in ('cats', 'ie3')
        if self.force_preferred_channel:
            self.analytes = {}
            for name, pnum in inst_analytes.items():
                self.analytes.setdefault(re.sub(r'\s+\([^()]+\)\s*$', '', name), pnum)
        else:
            self.analytes = dict(inst_analytes)
        try:
            info = self.get_site_info().sort_values('lat', ascending=False)
            self.sites_by_lat = info['code'].tolist()
            self.site_lat_map = {c: float(l) for c, l in zip(info['code'], info['lat'])}
        except Exception:
            self.sites_by_lat = sorted(lts.LOGOS_sites)
            self.site_lat_map = {}
        self.current_analyte = analyte or (next(iter(self.analytes)) if self.analytes else None)
        self.current_channel = lts._explicit_channel(self.current_analyte)
        self.analyte_combo = _Combo(self.current_analyte)
        self.start_year = _Value(start or 2000)
        self.end_year = _Value(end or pd.Timestamp.now().year)
        on = set(self.sites_by_lat if sites is None else sites)
        self.site_checks = [_Check(s, s in on) for s in self.sites_by_lat]

    def set_current_analyte(self, analyte_name):
        self.current_analyte = analyte_name or None
        self.current_channel = lts._explicit_channel(analyte_name) if analyte_name else None
        self.analyte_combo.text = analyte_name

    def __getattr__(self, name):
        # Everything else is the widget's own: its methods bound to this object, its class attributes as they are.
        try:
            attr = getattr(lts.TimeseriesWidget, name)
        except AttributeError:
            raise AttributeError(name) from None
        if isinstance(attr, types.FunctionType):
            return types.MethodType(attr, self)
        return attr


def _hex(c):
    return mcolors.to_hex(c, keep_alpha=False)


def site_colours(sites):
    return {p: {s: _hex(c) for s, c in lts.build_site_colors(sites, p).items()} for p in PALETTES}


def _ts(v):
    """Timestamps as the page's strings (UTC, to the second)."""
    s = pd.to_datetime(pd.Series(v))
    return [None if pd.isna(t) else t.strftime('%Y-%m-%d %H:%M:%S') for t in s]


def _num(v, digits=6):
    a = pd.to_numeric(pd.Series(v), errors='coerce').to_numpy(dtype=float)
    return [None if not np.isfinite(x) else round(float(x), digits) for x in a]


def _ints(v):
    a = pd.to_numeric(pd.Series(v), errors='coerce')
    return [None if pd.isna(x) else int(x) for x in a]


def _strs(v):
    return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x) for x in v]


def _means(df, x, mean, std, extra=None):
    """A binned or averaged dataset: {site, t, y, sd} columns, rows with no mean dropped."""
    if df is None or df.empty:
        return None
    m = pd.to_numeric(df[mean], errors='coerce')
    df = df[np.isfinite(m)]
    if df.empty:
        return None
    out = {'site': _strs(df['site']), 't': _ts(df[x]), 'y': _num(df[mean]), 'sd': _num(df[std])}
    for k, col in (extra or {}).items():
        out[k] = _ints(df[col]) if col == 'port' else _strs(df[col])
    return out


def _insitu_binned(ts, insitu, period):
    """Per (port, site, bin) mean and spread of the unflagged in situ air data; bins of one value are dropped (as the desktop)."""
    if insitu.empty:
        return None
    clean = insitu[insitu['rejected'] == 0].copy()
    if clean.empty:
        return None
    clean['_period'] = period.loc[clean.index]
    agg = clean.groupby(['port', 'site', '_period'])['mole_fraction'].agg(['mean', 'std', 'count']).reset_index()
    agg = agg[(agg['count'] >= 2) & np.isfinite(agg['std'])]
    return _means(agg, '_period', 'mean', 'std', {'port': 'port'})


def _merge(a, b):
    if not a:
        return b
    if not b:
        return a
    return {k: a.get(k, [None] * len(a['t'])) + b.get(k, [None] * len(b['t'])) for k in set(a) | set(b)}


def timeseries(inst, analyte, start, end):
    """Every dataset of the desktop Timeseries figure for one analyte and year range, for all of the instrument's sites.
    The page filters by site and by flag, so those toggles need no new request."""
    ts = HeadlessTimeseries(inst, analyte, start, end)
    if analyte not in ts.analytes and ts._resolve_pnum(analyte) is None:
        raise ValueError(f'unknown analyte {analyte!r}')
    is_m4, is_fe3 = inst.inst_num == 192, inst.inst_num == 193
    out = {'analyte': analyte, 'start': int(start), 'end': int(end), 'sets': {}}
    sets = out['sets']

    df = ts.query_flask_data(force=True, analyte=analyte)
    if not df.empty:
        sets['samples'] = {
            't': _ts(df['sample_datetime']), 'y': _num(df['mole_fraction']), 'site': _strs(df['site']),
            'rej': _ints(df['rejected'].fillna(0)), 'mf': _ints(df['ng_mole_fraction_num']),
            'run': _ts(df['run_time']), 'at': _ts(df['analysis_datetime']), 'rt': _ints(df['run_type_num']),
            'sid': _strs(df['sample_id']), 'pair': _ints(df['pair_id_num']), 'ch': _strs(df['channel']),
        }
    ds = ts.build_datasets(df, show_flagged=True)
    fm = ds.get('Flask mean', pd.DataFrame())
    if is_m4:
        mfm = ts.query_mstar_flask_data(analyte)       # M1/M3 store flask means: they join "Flask mean"
        if not mfm.empty:
            fm = pd.concat([d for d in (fm, mfm) if not d.empty], ignore_index=True)
    sets['flask'] = _means(fm, 'sample_datetime', 'mean', 'std', {'sid': 'sample_id'})
    sets['pair'] = _means(ds.get('Pair mean', pd.DataFrame()), 'sample_datetime', 'mean', 'std', {'pair': 'pair_id_num'})

    if is_m4:      # M1/M3 pair means: the pre-2022 half of "All samples", with the mole fractions behind each
        mp = ts.query_mstar_pair_data(analyte, include_tagged=True)
        if not mp.empty:
            sets['mstar'] = {
                't': _ts(mp['sample_datetime']), 'y': _num(mp['pair_avg']), 'site': _strs(mp['site']),
                'rej': _ints(mp['rejected'].fillna(0)), 'pair': _ints(mp['pair_id_num']),
                'flasks': [list(x) for x in mp['flask_ids']], 'mfs': [list(x) for x in mp['mf_nums']],
            }

    insitu = ts.query_insitu_data(analyte, force=True)
    if not insitu.empty:
        ports = list(getattr(inst, 'AIR_PORTS', [3, 7]))
        sets['insitu'] = {
            't': _ts(insitu['analysis_time']), 'y': _num(insitu['mole_fraction']), 'site': _strs(insitu['site']),
            'rej': _ints(insitu['rejected']), 'mf': _ints(insitu['mf_num']), 'port': _ints(insitu['port']),
            'run': _ts(insitu['run_time']), 'ch': _strs(insitu['channel']),
        }
        out['air_ports'] = ports
        month = insitu['analysis_time'].dt.to_period('M').dt.to_timestamp()
        day = insitu['analysis_time'].dt.day
        tenday = month + pd.to_timedelta(np.where(day <= 10, 0, np.where(day <= 20, 10, 20)), unit='D')
        sets['tenday'] = _insitu_binned(ts, insitu, tenday)
        sets['monthly'] = _insitu_binned(ts, insitu, month)

    sets['tenday'] = _merge(_means(ts.query_10day_mean_data(analyte), 'period_start', 'period_avg', 'period_std'), sets.get('tenday'))
    sets['monthly'] = _merge(_means(ts.query_monthly_mean_data(analyte), 'month_start', 'monthly_avg', 'monthly_std'), sets.get('monthly'))

    if is_fe3:
        op = ts.query_otto_pair_data(analyte)
        if not op.empty:
            sets['otto'] = {'t': _ts(op['sample_datetime']), 'y': _num(op['pair_avg']), 'site': _strs(op['site']),
                            'sd': _num(op['pair_stdv'])}
        sets['otto10'] = _means(ts.query_otto_10day_mean_data(analyte), 'period_start', 'period_avg', 'period_std')
        sets['ottomon'] = _means(ts.query_otto_monthly_mean_data(analyte), 'month_start', 'monthly_avg', 'monthly_std')

    out['sets'] = {k: v for k, v in sets.items() if v}
    return out


def relstd(inst, analyte, start, end):
    """The desktop's Relative Stddev figure data (query_rel_stddev_data): per run and flask, the mean, s.d., relative
    s.d. (%) and count of the unrejected injections of flask runs, for all of the instrument's sites."""
    ts = HeadlessTimeseries(inst, analyte, start, end)
    df = ts.query_rel_stddev_data(analyte)
    if df is None or df.empty:
        return {'rows': None}
    return {'rows': {
        'run': _ts(df['run_time']), 't': _ts(df['sample_datetime']), 'site': _strs(df['site']), 'sid': _strs(df['sample_id']),
        'pair': _ints(df['pair_id_num']), 'y': _num(df['mixratio']), 'sd': _num(df['stddev']), 'rel': _num(df['relstd'], 4),
        'n': _ints(df['n']),
    }}


EXPORTS = {
    # kind: (instrument number, exporter, every site and all time?, sites: 'all' | 'selected', label)
    'mstar_all': (192, MstarDataExporter, True, 'all', 'All sites and time'),
    'mstar_sel': (192, MstarDataExporter, False, 'selected', 'Selected sites and time'),
    'mstar_monthly': (192, MstarMonthlyExporter, False, 'selected', 'Selected sites, times, monthly means'),
    'mstar_global': (192, MstarGlobalMeansExporter, False, 'all', 'Global means at selected time'),
    'fecd_all': (193, FecdDataExporter, True, 'all', 'All sites and time'),
    'fecd_sel': (193, FecdDataExporter, False, 'selected', 'Selected sites and time'),
}


def export_kinds(inst):
    return [[k, v[4], v[3]] for k, v in EXPORTS.items() if v[0] == inst.inst_num]


def export(inst, kind, analyte, start, end, sites, all_analytes=False):
    """Write one export (or one per analyte) as the desktop's Save buttons do, into a scratch directory, and return
    the file, or a zip of the files."""
    if kind not in EXPORTS or EXPORTS[kind][0] != inst.inst_num:
        raise ValueError(f'no {kind} export for this instrument')
    _, cls, all_time, which, label = EXPORTS[kind]
    ts = HeadlessTimeseries(inst, analyte, start, end, sites)
    if which == 'all':
        skip = lts.PFP_SITES if cls is FecdDataExporter else lts.MSTAR_EXPORT_EXCLUDE
        use_sites = [s for s in ts.sites_by_lat if s not in skip]
    else:
        use_sites = [s for s in ts.sites_by_lat if s in set(sites or [])]
        if cls is FecdDataExporter:
            use_sites = [s for s in use_sites if s not in lts.PFP_SITES]
        if not use_sites:
            raise ValueError('no sites selected')
    names = list(ts.analytes) if all_analytes and cls is not FecdDataExporter else [analyte]
    with tempfile.TemporaryDirectory(prefix='logos_ts_') as tmp:
        written, empty, failed = [], [], []
        for name in names:
            exp = cls.from_timeseries_widget(ts, sites=use_sites, all_time=all_time, analyte=name)
            try:
                if getattr(exp, 'WRITES_DIRECTORY', False):
                    if exp.export_all(tmp):
                        written = sorted(os.listdir(tmp))
                    else:
                        empty.append(name)
                else:
                    path = os.path.join(tmp, exp.default_filename())
                    (written.append(os.path.basename(path)) if exp.export(path) else empty.append(name))
            except Exception as e:      # one bad analyte doesn't stop the others
                failed.append(f'{name}: {e}')
        if not written:
            raise ValueError('no data for that selection' + (f" ({'; '.join(failed)})" if failed else ''))
        if len(written) == 1:
            fname = written[0]
            with open(os.path.join(tmp, fname), 'rb') as f:
                data = f.read()
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
                for fname in written:
                    z.write(os.path.join(tmp, fname), fname)
            data = buf.getvalue()
            fname = f"{inst.inst_id}_{kind}_{'all_analytes' if all_analytes else re.sub(r'[^A-Za-z0-9]+', '_', analyte)}.zip"
    return {'name': fname, 'b64': base64.b64encode(data).decode('ascii'), 'files': len(written),
            'empty': empty, 'failed': failed, 'label': label}
