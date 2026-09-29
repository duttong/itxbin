"""
LOGOS combined data sets: several measurement programs blended into one
monthly record per background site, then into semi-hemispheric, hemispheric
and global means.

This replaces the Igor Pro "HATS combined" calculation
(HATS-Igor-code/CATS/Global Means.ipf).  combined_data_config.yaml is the
roadmap -- which programs feed each gas and how each is read -- and describes
where this version deliberately differs from Igor.

Classes
-------
CombinedConfig
    Parsed combined_data_config.yaml.
CombinedDataBuilder
    Builds one gas: loads every program, combines them per site, and computes
    the band, hemispheric and global means with global_means.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess

from global_means import GlobalMeansCalculator, GlobalMeansConfig
from mstar_pairs import MSTAR_INST_IDS, MSTAR_PAIR_AVG_SQL

CONFIG_FILE = Path(__file__).parent / 'combined_data_config.yaml'

# Locations written besides the sites, in output order.
MEAN_LOCATIONS = ('Global', 'NH', 'SH', 'HN', 'LN', 'LS', 'HS')


@dataclass
class CombinedConfig:
    """Parsed combined_data_config.yaml."""

    sites: list[str]
    phi: float
    weight_lat_overrides: dict[str, float]
    interpolation_method: str
    max_interpolation_months: int
    smoothing_window: int
    smoothing_order: int
    programs: dict[str, dict]
    site_se_scale: dict[str, float]
    gases: dict[str, dict] = field(default_factory=dict)
    pfp_sites: dict[str, str] = field(default_factory=dict)
    se_method: str = 'igor'
    downweight_filled: bool = False
    site_combine: str = 'igor'
    se_fallback: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = CONFIG_FILE) -> 'CombinedConfig':
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        smoothing = cfg.get('smoothing') or {}
        return cls(
            sites=[s.lower() for s in cfg['sites']],
            phi=float(cfg.get('phi', 30.0)),
            weight_lat_overrides={k.lower(): float(v)
                                  for k, v in (cfg.get('weight_lat_overrides') or {}).items()},
            interpolation_method=str(cfg.get('interpolation_method', 'seasonal')),
            max_interpolation_months=int(cfg.get('max_interpolation_months', 3)),
            smoothing_window=int(smoothing.get('window', 0) or 0),
            smoothing_order=int(smoothing.get('order', 2)),
            programs=cfg['programs'],
            site_se_scale={k.lower(): float(v)
                           for k, v in (cfg.get('site_se_scale') or {}).items()},
            gases=cfg['gases'],
            pfp_sites={k.lower(): v.lower() for k, v in (cfg.get('pfp_sites') or {}).items()},
            se_method=str(cfg.get('se_method', 'igor')),
            downweight_filled=bool(cfg.get('downweight_filled', False)),
            site_combine=str(cfg.get('site_combine', 'igor')),
            se_fallback=dict(cfg.get('se_fallback') or {}),
        )

    @property
    def program_order(self) -> list[str]:
        """Program names in bit order."""
        return list(self.programs)

    def global_means_config(self) -> GlobalMeansConfig:
        """The global_means.py settings for the band means and gap filling."""
        return GlobalMeansConfig(
            phi=self.phi,
            background_sites=list(self.sites),
            weight_lat_overrides=dict(self.weight_lat_overrides),
            interpolate_site_gaps=True,
            interpolation_method=self.interpolation_method,
            max_interpolation_months=self.max_interpolation_months,
        )


def programs_bitstring(present: set[str], order: list[str]) -> str:
    """'1'/'0' per program in *order*, e.g. '0101001'."""
    return ''.join('1' if p in present else '0' for p in order)


def _stack_programs(frames: dict[str, pd.DataFrame]):
    """Programs side by side on their shared month index: (x, se, ok), each a
    frame with one column per program; ok marks usable values."""
    months = sorted(set().union(*[f.index for f in frames.values()])) if frames else []
    idx = pd.DatetimeIndex(months, name='date')
    x = pd.DataFrame({p: f['mf'].reindex(idx) for p, f in frames.items()}, index=idx)
    se = pd.DataFrame({p: f['se'].reindex(idx) for p, f in frames.items()}, index=idx)
    return x, se, x.notna() & se.gt(0)


def _combined_frame(x: pd.DataFrame, ok: pd.DataFrame, mean: pd.Series,
                    sd: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({'mf': mean, 'sd': sd}, index=x.index)
    out['programs'] = [set(ok.columns[r]) for r in ok.to_numpy()]
    return out


def combine_programs(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weighted mean of several programs' series at one site.

    *frames* maps program name to a frame indexed by month with columns
    ``mf`` and ``se``.  Follows Igor WeightedMean_ofasite: each program is
    weighted by 1/se, the mean is sum(x/se)/sum(1/se) and the base error is
    N/sum(1/se) for N contributing programs.  Returns columns ``mf``, ``sd``
    (base error, before the mismatch term) and ``programs`` (a set per month).
    """
    x, se, ok = _stack_programs(frames)
    w = (1.0 / se).where(ok, 0.0)
    wsum = w.sum(axis=1).where(lambda v: v > 0)
    mean = (w * x.where(ok, 0.0)).sum(axis=1) / wsum
    return _combined_frame(x, ok, mean, ok.sum(axis=1) / wsum)


def combine_inverse_variance(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inverse-variance weighted mean of several programs at one site.

    mean = sum(x/se^2)/sum(1/se^2).  The error is the internal error
    1/sqrt(sum(1/se^2)) scaled up by the Birge ratio sqrt(chi^2/(N-1)) when
    the programs scatter more than their errors allow (never scaled down).
    Same columns as combine_programs(); ``sd`` is the final error.
    """
    x, se, ok = _stack_programs(frames)
    w = (1.0 / se ** 2).where(ok, 0.0)
    wsum = w.sum(axis=1).where(lambda v: v > 0)
    count = ok.sum(axis=1)
    mean = (w * x.where(ok, 0.0)).sum(axis=1) / wsum
    chi2 = (w * (x.sub(mean, axis=0) ** 2).where(ok, 0.0)).sum(axis=1)
    birge = np.sqrt(chi2 / (count - 1).where(count > 1)).fillna(1.0).clip(lower=1.0)
    return _combined_frame(x, ok, mean, birge / np.sqrt(wsum))


def add_mismatch(combined: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.Series:
    """Igor's mismatch error: where a program sits further from the combined
    mean than the two errors allow, add the excess to the site error."""
    sd = combined['sd'].copy()
    for f in frames.values():
        f = f.reindex(combined.index)
        excess = (combined['mf'] - f['mf']).abs() - (combined['sd'] + f['se'])
        sd = sd + excess.clip(lower=0).fillna(0)
    return sd.where(combined['mf'].notna())


def _decimal_year(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (d.year + (d.month - 0.5) / 12).to_numpy()


def loess_site_series(obs: pd.DataFrame, window_months: float) -> pd.DataFrame:
    """Igor-style Loess of one site's monthly record (columns date, mf, se, n).

    Returns every month from the first to the last observation: mf is the
    lowess curve (window about *window_months* wide, at least 5 points), se
    is interpolated in time, and n is the month's sample count (0 where the
    month was filled).
    """
    obs = obs.sort_values('date')
    grid = pd.date_range(obs['date'].iloc[0], obs['date'].iloc[-1], freq='MS', name='date')
    if len(obs) < 3:
        return obs.set_index('date')[['mf', 'se', 'n']]
    frac = min(1.0, max(window_months / len(grid), 5 / len(obs)))
    fit = lowess(obs['mf'].to_numpy(), _decimal_year(obs['date']), frac=frac, it=0,
                 xvals=_decimal_year(grid))
    by_date = obs.set_index('date')
    out = pd.DataFrame({'mf': fit}, index=grid)
    out['se'] = by_date['se'].reindex(grid).interpolate(method='time').ffill().bfill()
    out['n'] = by_date['n'].reindex(grid).fillna(0).astype(int)
    return out


def inflate_filled_se(frame: pd.DataFrame) -> pd.Series:
    """se with filled months (n == 0) down-weighted: multiplied by
    sqrt(1 + d), d = months to the nearest measured month, so an estimate
    one month from data counts about half as much as a measurement, and one
    a year away about a thirteenth."""
    measured = frame['n'].to_numpy() > 0
    if measured.all() or not measured.any():
        return frame['se']
    month = frame.index.year.to_numpy() * 12 + frame.index.month.to_numpy()
    obs = month[measured]
    pos = np.searchsorted(obs, month)
    before = np.abs(month - obs[np.clip(pos - 1, 0, len(obs) - 1)])
    after = np.abs(obs[np.clip(pos, 0, len(obs) - 1)] - month)
    d = np.where(measured, 0, np.minimum(before, after))
    return frame['se'] * np.sqrt(1.0 + d)


def smooth_series(series: pd.Series, window: int, order: int) -> pd.Series:
    """Savitzky-Golay smoothing of each contiguous run of a monthly series.

    Runs shorter than *window* are returned unchanged, as are the first and
    last points of each run (Igor re-inserted them after smoothing).
    """
    if window <= 1:
        return series
    out = series.copy()
    valid = series.notna()
    run_id = (valid != valid.shift()).cumsum()
    for _, run in series[valid].groupby(run_id[valid]):
        if len(run) < window:
            continue
        smoothed = savgol_filter(run.to_numpy(), window, order, mode='interp')
        smoothed[0], smoothed[-1] = run.iloc[0], run.iloc[-1]
        out.loc[run.index] = smoothed
    return out


def pairwise_log_ratios(series: dict[str, pd.Series], min_overlap: int = 12) -> pd.DataFrame:
    """Median 100*ln(a/b) over the site-months two series share.

    *series* maps a key to a Series of mole fractions indexed by (site, date).
    Returns one row per pair with at least *min_overlap* shared site-months:
    columns a, b, pct, n.
    """
    keys = list(series)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            j = pd.concat([series[a], series[b]], axis=1, keys=['a', 'b'], join='inner')
            j = j[(j['a'] > 0) & (j['b'] > 0)]
            if len(j) < min_overlap:
                continue
            rows.append({'a': a, 'b': b, 'n': len(j),
                         'pct': float(np.median(100 * np.log(j['a'] / j['b'])))})
    return pd.DataFrame(rows, columns=['a', 'b', 'pct', 'n'])


def solve_offsets(pairs: pd.DataFrame, reference: str) -> pd.Series:
    """Each key's level relative to *reference*, in percent (log ratio x 100).

    Weighted least squares on pct(a, b) = x_a - x_b with x_reference = 0 and
    weights equal to the shared site-months, so a program with no direct
    overlap with the reference is tied to it through the programs it does
    overlap.  Keys not connected to the reference are left out.
    """
    linked, frontier = {reference}, [reference]
    while frontier:
        k = frontier.pop()
        for r in pairs.itertuples():
            other = r.b if r.a == k else r.a if r.b == k else None
            if other is not None and other not in linked:
                linked.add(other)
                frontier.append(other)
    free = sorted(linked - {reference})
    if not free:
        return pd.Series({reference: 0.0})
    col = {k: i for i, k in enumerate(free)}
    use = pairs[pairs['a'].isin(linked) & pairs['b'].isin(linked)]
    A = np.zeros((len(use), len(free)))
    for row, r in enumerate(use.itertuples()):
        if r.a in col:
            A[row, col[r.a]] = 1.0
        if r.b in col:
            A[row, col[r.b]] = -1.0
    w = np.sqrt(use['n'].to_numpy(float))
    x, *_ = np.linalg.lstsq(A * w[:, None], use['pct'].to_numpy() * w, rcond=None)
    return pd.Series({reference: 0.0, **dict(zip(free, x))})


def program_noise(series: dict[str, pd.Series], min_overlap: int = 24) -> pd.Series:
    """Each program's monthly-mean noise from how it differs from the others.

    *series* maps a program to mole fractions indexed by (site, date).  For
    two programs at the same site-months, var(a - b) = s_a^2 + s_b^2 when
    their errors are independent (a three-cornered hat).  Each pair's
    difference has its median per site removed (constant offsets are not
    noise) and its variance taken robustly (1.4826 x MAD)^2; the s^2 are then
    solved by non-negative least squares on relative residuals, weighted by
    the square root of the shared site-months.
    Programs with no pair of at least *min_overlap* site-months are left out.
    """
    keys = list(series)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            j = pd.concat([series[a], series[b]], axis=1, keys=['a', 'b'], join='inner').dropna()
            if len(j) < min_overlap:
                continue
            d = j['a'] - j['b']
            d = d - d.groupby(level='site').transform('median')
            var = (1.4826 * (d - d.median()).abs().median()) ** 2
            rows.append((a, b, var, len(j)))
    used = sorted({k for r in rows for k in r[:2]}, key=keys.index)
    if not rows:
        return pd.Series(dtype=float)
    col = {k: i for i, k in enumerate(used)}
    A = np.zeros((len(rows), len(used)))
    y = np.zeros(len(rows))
    w = np.zeros(len(rows))
    for r, (a, b, var, n) in enumerate(rows):
        A[r, col[a]] = A[r, col[b]] = 1.0
        # Relative residuals, so the large variances of the early programs
        # don't swamp the small ones.
        y[r], w[r] = var, np.sqrt(n) / max(var, 1e-12)
    s2, _ = nnls(A * w[:, None], y * w)
    return pd.Series(np.sqrt(s2), index=used)


class CombinedDataBuilder:
    """Build one combined data set.

    Parameters
    ----------
    gas :
        Key under ``gases`` in the config (e.g. 'CFC11').
    db :
        Object with ``doquery(sql, params)`` returning a list of dicts.
    config :
        A :class:`CombinedConfig`; the packaged config is loaded if omitted.
    """

    def __init__(self, gas: str, db, config: Optional[CombinedConfig] = None):
        self.config = config or CombinedConfig.load()
        if gas not in self.config.gases:
            raise KeyError(f"No combined data set configured for {gas!r}")
        self.gas = gas
        self.gas_cfg = self.config.gases[gas]
        self.db = db
        self.gm_config = self.config.global_means_config()
        self.calculator = GlobalMeansCalculator(gas, config=self.gm_config)
        self.program_data: dict[str, pd.DataFrame] = {}
        self._loaded: dict[str, pd.DataFrame] = {}
        self._noise: Optional[pd.Series] = None

    # ── loading ──────────────────────────────────────────────────────────────

    def pnum_for(self, program: str) -> int:
        return int((self.gas_cfg.get('pnums') or {}).get(program, self.gas_cfg['parameter_num']))

    def program_monthly(self, program: str) -> pd.DataFrame:
        """load_program(), read once per builder."""
        if program not in self._loaded:
            self._loaded[program] = self.load_program(program)
        return self._loaded[program].copy()

    def load_program(self, program: str) -> pd.DataFrame:
        """Monthly mean/sd/n per site for one program: columns site, date,
        mf, sd, n.  Only the configured background sites are kept."""
        spec = self.config.programs[program]
        loader = {
            'oldgc': self._load_oldgc,
            'insitu': self._load_insitu,
            'pairs': self._load_pairs,
            'ccgg': self._load_ccgg,
        }[spec['source']]
        df = loader(program, spec)
        if df.empty:
            return pd.DataFrame(columns=['site', 'date', 'mf', 'sd', 'n'])
        df['site'] = df['site'].str.lower()
        df = df[df['site'].isin(self.config.sites)]
        df['date'] = pd.to_datetime(df['date'])
        for col in ('mf', 'sd', 'n'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        limits = (self.gas_cfg.get('program_limits') or {}).get(program) or {}
        if limits.get('start'):
            df = df[df['date'] >= pd.Timestamp(limits['start'])]
        if limits.get('end'):
            df = df[df['date'] <= pd.Timestamp(limits['end'])]
        return df.dropna(subset=['mf'])[['site', 'date', 'mf', 'sd', 'n']].reset_index(drop=True)

    def _load_oldgc(self, program: str, spec: dict) -> pd.DataFrame:
        """oldGC monthly means from hats.fecd_oldgc (fecd_oldgc_import.py)."""
        rows = self.db.doquery(
            """SELECT LOWER(s.code) AS site, o.month AS date, o.mean AS mf, o.sd AS sd, o.n AS n
               FROM hats.fecd_oldgc o JOIN gmd.site s ON s.num = o.site_num
               WHERE o.parameter_num = %s""",
            [self.pnum_for(program)]) or []
        return pd.DataFrame(rows)

    def _load_insitu(self, program: str, spec: dict) -> pd.DataFrame:
        insts = [int(i) for i in spec['inst_nums']]
        rows = self.db.doquery(
            f"""SELECT LOWER(s.code) AS site, m.month AS date, m.mean AS mf, m.std AS sd, m.n AS n
                FROM hats.ng_insitu_monthly_means m JOIN gmd.site s ON s.num = m.site_num
                WHERE m.inst_num IN ({','.join(['%s'] * len(insts))}) AND m.parameter_num = %s
                  AND m.mean IS NOT NULL""",
            insts + [self.pnum_for(program)]) or []
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        # Two instruments of one program at a site in the same month: average.
        return df.groupby(['site', 'date'], as_index=False).agg(
            mf=('mf', 'mean'), sd=('sd', 'mean'), n=('n', 'sum'))

    # Preferred channel per (inst, parameter, date); falls back to the
    # earliest preference, then to the row's own channel (single-channel
    # instruments have no preference rows).
    _PREFERRED_CHANNEL_SQL = """
        v.channel = COALESCE(
            (SELECT pc.channel FROM hats.ng_preferred_channel pc
             WHERE pc.inst_num = v.inst_num AND pc.parameter_num = v.parameter_num
               AND pc.start_date <= COALESCE(v.sample_datetime, v.analysis_datetime)
             ORDER BY pc.start_date DESC LIMIT 1),
            (SELECT pc.channel FROM hats.ng_preferred_channel pc
             WHERE pc.inst_num = v.inst_num AND pc.parameter_num = v.parameter_num
             ORDER BY pc.start_date ASC LIMIT 1),
            v.channel)"""

    def _pfp_label_sql(self) -> str:
        """CASE expression putting PFP pairs (pair_id_num = 0 at a base site)
        on their pseudo-site, as data_export._site_label_sql() does."""
        whens = ' '.join(
            f"WHEN LOWER(v.site) = '{base}' AND v.pair_id_num = 0 THEN '{pseudo}'"
            for pseudo, base in self.config.pfp_sites.items())
        return f"CASE {whens} ELSE LOWER(v.site) END" if whens else "LOWER(v.site)"

    def _load_pairs(self, program: str, spec: dict) -> pd.DataFrame:
        """Flask pair means pooled across the program's instruments, then
        aggregated to monthly mean/sd/n.  PFP pairs are kept only by a
        `pfp: only` program, at their base site.
        OTTO pairs carry no site/date in the view (flask_id = 0), so they come
        from hatsflask_pair_info."""
        pnum = self.pnum_for(program)
        inst_ids = list(spec['inst_ids'])
        frames = []
        regular = [i for i in inst_ids if i.upper() != 'OTTO']
        # M* pairs need two flasks (Montzka's rule), so they come from
        # MSTAR_PAIR_AVG_SQL; other instruments keep ng_pair_avg_view.
        mstar = [i for i in regular if i.upper() in MSTAR_INST_IDS]
        other = [i for i in regular if i.upper() not in MSTAR_INST_IDS]
        for ids, source in ((mstar, MSTAR_PAIR_AVG_SQL), (other, 'hats.ng_pair_avg_view')):
            if not ids:
                continue
            rows = self.db.doquery(
                f"""SELECT {self._pfp_label_sql()} AS site, v.sample_datetime AS dt,
                           v.pair_avg AS value
                    FROM {source} v
                    WHERE v.inst_id IN ({','.join(['%s'] * len(ids))})
                      AND v.parameter_num = %s
                      AND v.sample_datetime IS NOT NULL AND {self._PREFERRED_CHANNEL_SQL}""",
                ids + [pnum]) or []
            frames.append(pd.DataFrame(rows))
        if any(i.upper() == 'OTTO' for i in inst_ids):
            rows = self.db.doquery(
                f"""SELECT LOWER(s.code) AS site, pi.datetime AS dt, v.pair_avg AS value
                    FROM hats.ng_pair_avg_view v
                    JOIN hats.hatsflask_pair_info pi ON pi.pair_id = v.pair_id_num
                    JOIN gmd.site s ON s.num = pi.site_num
                    WHERE v.inst_id = 'OTTO' AND v.parameter_num = %s
                      AND {self._PREFERRED_CHANNEL_SQL}""",
                [pnum]) or []
            frames.append(pd.DataFrame(rows))
        pairs = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else pd.DataFrame()
        if pairs.empty:
            return pairs
        # PFP pairs feed only a program marked `pfp: only`, at their base site;
        # every other flask program leaves them out.
        is_pfp = pairs['site'].isin(self.config.pfp_sites)
        if spec.get('pfp') == 'only':
            pairs = pairs[is_pfp].assign(site=lambda d: d['site'].map(self.config.pfp_sites))
        else:
            pairs = pairs[~is_pfp]
        if pairs.empty:
            return pairs
        pairs['value'] = pd.to_numeric(pairs['value'], errors='coerce')
        pairs['date'] = pd.to_datetime(pairs['dt']).dt.to_period('M').dt.to_timestamp()
        return pairs.dropna(subset=['value']).groupby(['site', 'date'], as_index=False).agg(
            mf=('value', 'mean'), sd=('value', 'std'), n=('value', 'size'))

    def _load_ccgg(self, program: str, spec: dict) -> pd.DataFrame:
        sites = self.config.sites
        rows = self.db.doquery(
            f"""SELECT LOWER(site) AS site, ev_datetime AS dt, value, unc
                FROM ccgg.flask_data_view
                WHERE parameter_num = %s AND program = 'ccgg' AND strategy = 'flask'
                  AND flag LIKE '..%%' AND LOWER(site) IN ({','.join(['%s'] * len(sites))})""",
            [self.pnum_for(program)] + sites) or []
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        for col in ('value', 'unc'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Pair (same event time) means first, as ~/CCGG/N2O_comparison.py does.
        pairs = df.groupby(['site', 'dt'], as_index=False).agg(value=('value', 'mean'),
                                                                unc=('unc', 'mean'))
        pairs['date'] = pd.to_datetime(pairs['dt']).dt.to_period('M').dt.to_timestamp()
        monthly = pairs.groupby(['site', 'date'], as_index=False).agg(
            mf=('value', 'mean'), sd=('value', 'std'), n=('value', 'size'),
            unc=('unc', lambda u: u[u > 0].mean()))
        monthly['sd'] = monthly['sd'].fillna(monthly['sd'].mean())
        monthly['sd'] = np.sqrt(monthly['sd'] ** 2 + monthly['unc'].fillna(0) ** 2)
        return monthly[['site', 'date', 'mf', 'sd', 'n']]

    # ── per-program processing ───────────────────────────────────────────────

    def program_noise(self) -> pd.Series:
        """Monthly-mean noise for each program piece of this gas (keys as in
        offset_segments(), so OTTO and FE3, M1 and M3/M4 get their own), from
        the offset-corrected measured months (see program_noise()).  A piece
        with too little overlap, or solved as zero, takes the mean of its
        program's other pieces, else the latest piece of its se_fallback
        program."""
        if self._noise is None:
            series, program_of = {}, {}
            for key, program, start, end in self.offset_segments():
                df = self.program_monthly(program)
                if start:
                    df = df[df['date'] >= pd.Timestamp(start)]
                if end:
                    df = df[df['date'] <= pd.Timestamp(end)]
                if not df.empty:
                    series[key] = df.assign(mf=self.apply_offsets(program, df)).set_index(
                        ['site', 'date'])['mf']
                    program_of[key] = program
            # A zero means the overlaps can't separate this piece's noise from
            # its partners' (few or one-sided pairs), not that it has none.
            noise = program_noise(series)
            noise = noise[noise > 0]
            for key, program in program_of.items():
                if key in noise:
                    continue
                same = [noise[k] for k, p in program_of.items() if p == program and k in noise]
                other = self.config.se_fallback.get(program)
                borrowed = [noise[k] for k, p in program_of.items() if p == other and k in noise]
                if same or borrowed:
                    noise[key] = float(np.mean(same)) if same else borrowed[-1]
            self._noise = noise
        return self._noise

    def _row_noise(self, program: str, dates: pd.Series) -> pd.Series:
        """program_noise() for each row, by the program piece its date is in."""
        noise = self.program_noise()
        out = pd.Series(np.nan, index=dates.index)
        for key, prog, start, end in self.offset_segments():
            if prog != program or key not in noise:
                continue
            mask = pd.Series(True, index=dates.index)
            if start:
                mask &= dates >= pd.Timestamp(start)
            if end:
                mask &= dates <= pd.Timestamp(end)
            out[mask] = noise[key]
        return out

    def standard_errors(self, program: str, df: pd.DataFrame) -> pd.Series:
        """se for each row of one program's monthly frame.

        se_method 'igor': sd divided by the program's se_divisor.
        se_method 'overlap': sqrt(sampling^2 + floor^2), where sampling is
        sd / sqrt(n) (n capped at the program's max_n for autocorrelated in
        situ data) and the floor is set so the program's median se equals its
        measured monthly-mean noise (program_noise()).
        """
        spec = self.config.programs[program]
        if self.config.se_method == 'overlap':
            n = df['n'].where(df['n'] > 0)
            if spec.get('max_n'):
                n = n.clip(upper=float(spec['max_n']))
            se = (df['sd'] / np.sqrt(n)).where(lambda x: x > 0)
            se = se.fillna(se.groupby(df['site']).transform('median')).fillna(se.median())
            # Floor per program piece, so its median se equals its noise.
            noise = self._row_noise(program, df['date'])
            for level in noise.dropna().unique():
                piece = noise == level
                floor2 = max(level ** 2 - float(np.nanmedian(se[piece] ** 2)), 0.0)
                se[piece] = np.sqrt(se[piece] ** 2 + floor2)
        else:
            divisor = spec.get('se_divisor', 1)
            if divisor == 'sqrt_n':
                se = df['sd'] / np.sqrt(df['n'].where(df['n'] > 0))
            else:
                se = df['sd'] / float(divisor)
        cap = (self.gas_cfg.get('se_cap') or {}).get(program)
        if cap is not None:
            se = se.clip(upper=float(cap))
        se = se.where(se > 0)
        # One-sample months have no spread: use that site's median, then the
        # program's.
        se = se.fillna(se.groupby(df['site']).transform('median')).fillna(se.median())
        scale = df['site'].map(self.config.site_se_scale).fillna(1.0)
        return se * scale

    def apply_offsets(self, program: str, df: pd.DataFrame) -> pd.Series:
        """mf with the gas's configured percent offset(s) for *program*."""
        spec = (self.gas_cfg.get('offsets_pct') or {}).get(program)
        if spec is None:
            return df['mf']
        entries = spec if isinstance(spec, list) else [{'pct': spec}]
        factor = pd.Series(1.0, index=df.index)
        for e in entries:
            mask = pd.Series(True, index=df.index)
            if e.get('start'):
                mask &= df['date'] >= pd.Timestamp(e['start'])
            if e.get('end'):
                mask &= df['date'] <= pd.Timestamp(e['end'])
            factor[mask] *= (100.0 + float(e['pct'])) / 100.0
        return df['mf'] * factor

    def _fill_long_gaps_loess(self, grp: pd.DataFrame, obs: pd.DataFrame, cfg: dict,
                              site: str) -> pd.DataFrame:
        """Fill the interior gaps the seasonal fill left empty (longer than
        max_interpolation_months) from a broad Loess of the site's measured
        months.  Measured and seasonally filled months are untouched; the
        Loess supplies only the months inside those long gaps, marked n = 0."""
        hole = grp['mf'].isna()
        if not hole.any():
            return grp
        windows = cfg.get('site_window_months') or {}
        curve = loess_site_series(obs.rename(columns={'sd': 'se'}),
                                  float(windows.get(site, cfg['window_months'])))
        grp = grp.copy()
        fill = hole & grp.index.isin(curve.index)
        grp.loc[fill, 'mf'] = curve['mf'].reindex(grp.index[fill]).to_numpy()
        se = grp['se'].interpolate(method='time', limit_area='inside')
        grp.loc[fill, 'se'] = se[fill]
        return grp

    def prepare_program(self, program: str) -> dict[str, pd.DataFrame]:
        """{site: frame(mf, se, n) indexed by month} after gap filling."""
        df = self.program_monthly(program)
        self.program_data[program] = df
        if df.empty:
            return {}
        df = df.assign(mf=self.apply_offsets(program, df))
        df = df.assign(sd=self.standard_errors(program, df))
        loess_cfg = self.config.programs[program].get('loess')
        if loess_cfg:
            # Igor oldGC: a Loess curve replaces the monthly values and fills
            # every gap from the first to the last sample.
            site_windows = loess_cfg.get('site_window_months') or {}
            out = {site: loess_site_series(g.rename(columns={'sd': 'se'}),
                                           float(site_windows.get(site, loess_cfg['window_months'])))
                   for site, g in df.groupby('site')}
        else:
            box = int(self.config.programs[program].get('box_smooth_months') or 0)
            filled = self.calculator.fill_site_gaps(df)
            long_cfg = self.config.programs[program].get('long_gap_loess')
            out = {}
            for site, grp in filled.groupby('site'):
                grp = grp.set_index('date').rename(columns={'sd': 'se'})
                if long_cfg:
                    grp = self._fill_long_gaps_loess(grp, df[df['site'] == site], long_cfg, site)
                if box > 1:
                    # Seasonal gap fill, then a centred box mean of the filled
                    # series.  min_periods=2 keeps the first and last month of
                    # each record (and the edges of any unfilled gap) smoothed
                    # over what is there instead of dropping them.
                    grp['mf'] = grp['mf'].rolling(box, center=True, min_periods=2).mean()
                out[site] = grp[['mf', 'se', 'n']].dropna(subset=['mf'])
        if self.config.downweight_filled:
            out = {site: f.assign(se=inflate_filled_se(f)) for site, f in out.items()}
        return out

    # ── offsets ──────────────────────────────────────────────────────────────

    def offset_segments(self) -> list[tuple[str, str, Optional[str], Optional[str]]]:
        """(key, program, start, end) for every program piece whose offset is
        estimated separately.  A program is split at its ``offset_breaks``
        dates; start is inclusive, end is the day before the next break."""
        breaks = self.gas_cfg.get('offset_breaks') or {}
        segs = []
        for program in self.gas_cfg['programs']:
            cuts = [pd.Timestamp(b) for b in breaks.get(program, [])]
            edges = [None] + cuts + [None]
            for lo, hi in zip(edges[:-1], edges[1:]):
                start = lo.strftime('%Y-%m-%d') if lo is not None else None
                end = (hi - pd.Timedelta(days=1)).strftime('%Y-%m-%d') if hi is not None else None
                key = program if not cuts else f"{program}[{start or ''}..{end or ''}]"
                segs.append((key, program, start, end))
        return segs

    def estimate_offsets(self, min_overlap: int = 12) -> dict[str, pd.DataFrame]:
        """Estimate each program's scale offset from the site-months it shares
        with the other programs (measured months only, before gap filling and
        before any configured offset).

        The gas's ``offset_reference`` program is held at zero.  Returns
        ``offsets`` (key, program, start, end, level_pct, offset_pct, n) where
        offset_pct is the value for ``offsets_pct`` that puts the piece on the
        reference, and ``pairs`` (a, b, n, pct, fitted, resid) to judge how
        consistent the pairwise differences are.
        """
        reference = self.gas_cfg.get('offset_reference')
        if reference not in self.gas_cfg['programs']:
            raise ValueError(f"{self.gas}: offset_reference must be one of its programs")
        raw = {p: self.program_monthly(p) for p in self.gas_cfg['programs']}
        series = {}
        meta = {}
        for key, program, start, end in self.offset_segments():
            df = raw[program]
            if start:
                df = df[df['date'] >= pd.Timestamp(start)]
            if end:
                df = df[df['date'] <= pd.Timestamp(end)]
            if not df.empty:
                series[key] = df.set_index(['site', 'date'])['mf']
                meta[key] = (program, start, end)
        ref_keys = [k for k in series if meta[k][0] == reference]
        ref_date = self.gas_cfg.get('offset_reference_date')
        if ref_date:
            d = pd.Timestamp(ref_date)
            ref_keys = [k for k in ref_keys
                        if (not meta[k][1] or d >= pd.Timestamp(meta[k][1]))
                        and (not meta[k][2] or d <= pd.Timestamp(meta[k][2]))]
        if len(ref_keys) != 1:
            raise ValueError(f"{self.gas}: offset_reference needs data, and an "
                             f"offset_reference_date if it has offset_breaks")
        pairs = pairwise_log_ratios(series, min_overlap)
        level = solve_offsets(pairs, ref_keys[0])
        pairs['fitted'] = pairs['a'].map(level) - pairs['b'].map(level)
        pairs['resid'] = pairs['pct'] - pairs['fitted']
        shared = pd.concat([pairs.groupby('a')['n'].sum(), pairs.groupby('b')['n'].sum()],
                           axis=1).sum(axis=1)
        offsets = pd.DataFrame(
            [{'key': k, 'program': meta[k][0], 'start': meta[k][1], 'end': meta[k][2],
              'level_pct': level.get(k, np.nan),
              'offset_pct': 100 * (np.exp(-level[k] / 100) - 1) if k in level else np.nan,
              'n': int(shared.get(k, 0))}
             for k in series])
        return {'reference': ref_keys[0], 'offsets': offsets, 'pairs': pairs,
                'blocks': self._offset_residual_blocks(series, level)}

    @staticmethod
    def _offset_residual_blocks(series: dict[str, pd.Series], level: pd.Series,
                                years: int = 5) -> pd.DataFrame:
        """Once every piece is moved to the reference by its estimated level,
        each piece's median % difference from all the other pieces it shares
        site-months with, per *years*-year block.  A trend here is drift that
        a constant offset does not remove."""
        adj = {k: 100 * np.log(s[s > 0]) - level[k] for k, s in series.items() if k in level}
        out = {}
        for k, s in adj.items():
            diffs = [pd.concat([s, o], axis=1, join='inner').pipe(lambda j: j.iloc[:, 0] - j.iloc[:, 1])
                     for other, o in adj.items() if other != k]
            d = pd.concat(diffs) if diffs else pd.Series(dtype=float)
            if d.empty:
                continue
            block = (d.index.get_level_values('date').year // years) * years
            out[k] = d.groupby(block).median()
        return pd.DataFrame(out).T.sort_index(axis=1)

    # ── combining ────────────────────────────────────────────────────────────

    def program_site_frames(self) -> dict[str, dict[str, pd.DataFrame]]:
        """{site: {program: frame(mf, se, n)}} for every program, prepared."""
        by_site: dict[str, dict[str, pd.DataFrame]] = {}
        for program in self.gas_cfg['programs']:
            for site, frame in self.prepare_program(program).items():
                by_site.setdefault(site, {})[program] = frame
        return by_site

    def site_series(self, by_site: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Combined, smoothed monthly series for every site from
        program_site_frames(): columns site, date, mf, sd, n, programs."""
        order = self.config.program_order
        drop_only = set(self.gas_cfg.get('drop_if_only') or [])
        rows = []
        for site, frames in sorted(by_site.items()):
            inputs = {p: f[['mf', 'se']] for p, f in frames.items()}
            if self.config.site_combine == 'inverse_variance':
                combined = combine_inverse_variance(inputs)
            else:
                combined = combine_programs(inputs)
            if drop_only:
                only = combined['programs'].apply(lambda p: bool(p) and p <= drop_only)
                combined = combined[~only]
                if combined.empty:
                    continue
            # Mismatch against the unsmoothed mean, so smoothing residuals
            # don't count as program disagreement.
            if self.config.site_combine != 'inverse_variance':
                combined['sd'] = add_mismatch(combined, inputs)
            combined['mf'] = smooth_series(combined['mf'], self.config.smoothing_window,
                                           self.config.smoothing_order)
            n = sum(f['n'].reindex(combined.index).fillna(0) for f in frames.values())
            out = combined.assign(site=site, n=n.astype(int))
            out['programs'] = [programs_bitstring(p, order) for p in out['programs']]
            rows.append(out.reset_index())
        if not rows:
            return pd.DataFrame(columns=['site', 'date', 'mf', 'sd', 'n', 'programs'])
        return pd.concat(rows, ignore_index=True)[['site', 'date', 'mf', 'sd', 'n', 'programs']]

    def site_latitudes(self) -> dict[str, float]:
        """{site: lat} from gmd.site."""
        sites = sorted(self.config.sites)
        rows = self.db.doquery(
            f"SELECT LOWER(code) AS code, lat FROM gmd.site WHERE LOWER(code) IN "
            f"({','.join(['%s'] * len(sites))})", sites) or []
        return {r['code']: float(r['lat']) for r in rows}

    def program_global_means(self, by_site: dict[str, dict[str, pd.DataFrame]],
                             lats: dict[str, float]) -> list[pd.DataFrame]:
        """Each program's own global mean, from its gap-filled site series
        alone (no combining or smoothing), as location 'prog:<program>'.

        These show how the programs overlap and agree (the provenance figure);
        a program with too few sites for all four bands has no global mean.
        """
        order = self.config.program_order
        frames = []
        for program in self.gas_cfg['programs']:
            parts = [f.assign(site=site).rename(columns={'se': 'sd'}).reset_index()
                     for site, progs in by_site.items()
                     for p, f in progs.items() if p == program]
            if not parts:
                continue
            site_df = pd.concat(parts, ignore_index=True)[['site', 'date', 'mf', 'sd', 'n']]
            means = self.calculator.compute_prepared(self.calculator.prepare(site_df, lats))
            if means.empty or 'Global' not in means:
                continue
            m = means[['Global', 'Global_sd']].rename(columns={'Global': 'mean', 'Global_sd': 'sd'})
            m = m.dropna(subset=['mean']).reset_index()
            if m.empty:
                continue
            m['location'] = f'prog:{program}'
            m['programs'] = programs_bitstring({program}, order)
            m['n'] = m['date'].map(site_df.groupby('date')['n'].sum()).fillna(0).astype(int)
            frames.append(m)
        return frames

    def build(self) -> pd.DataFrame:
        """Tidy result: columns location, date, mean, sd, n, programs.

        Locations are the sites, Global, NH, SH and the four bands, and each
        program's own global mean as 'prog:<program>'.
        """
        by_site = self.program_site_frames()
        sites = self.site_series(by_site)
        if sites.empty:
            return pd.DataFrame(columns=['location', 'date', 'mean', 'sd', 'n', 'programs'])
        lats = self.site_latitudes()
        prepared = self.calculator.prepare(sites[['site', 'date', 'mf', 'sd', 'n']], lats)
        means = self.calculator.compute_prepared(prepared)

        # Programs contributing to any site each month, for the mean rows.
        order = self.config.program_order
        bits = sites.groupby('date')['programs'].agg(
            lambda s: ''.join('1' if any(b[i] == '1' for b in s) else '0'
                              for i in range(len(order))))
        n_total = sites.groupby('date')['n'].sum()

        out = [sites.rename(columns={'site': 'location', 'mf': 'mean'})]
        for loc in MEAN_LOCATIONS:
            if loc not in means:
                continue
            m = means[[loc, f'{loc}_sd']].rename(columns={loc: 'mean', f'{loc}_sd': 'sd'})
            m = m.dropna(subset=['mean']).reset_index()
            m['location'] = loc
            m['programs'] = m['date'].map(bits)
            m['n'] = m['date'].map(n_total).fillna(0).astype(int)
            out.append(m)
        out.extend(self.program_global_means(by_site, lats))
        result = pd.concat(out, ignore_index=True)
        return result[['location', 'date', 'mean', 'sd', 'n', 'programs']].sort_values(
            ['location', 'date']).reset_index(drop=True)
