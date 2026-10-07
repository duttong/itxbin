"""
Hemispheric and global mean mole fractions from site monthly means.

The math follows https://github.com/duttong/GML_means (gml_annualmeans.py's
``semi_hemispheric_means``): sites are grouped into four semi-hemispheric bands
by latitude, averaged with cos(latitude) weights, and the global mean is the
unweighted average of the four bands.  Two differences from that repository:

* input is per-site *monthly* means from the HATS database rather than the
  published GML website files, so this module stops at monthly resolution and
  leaves the annual-mean rollup to GML_means;
* the ``_sd`` columns propagate the site standard deviations through the same
  weights, which GML_means does not report.

Classes
-------
GlobalMeansConfig
    Parsed ``gml_global_means_config.yaml``: background sites, per-gas
    overrides, weighting-latitude overrides and the output header template.
GlobalMeansCalculator
    Turns a tidy frame of site monthly means into band, hemispheric and global
    means with propagated uncertainties.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CONFIG_FILE = Path(__file__).parent / 'gml_global_means_config.yaml'

# Band order used throughout, and the combinations built from them.
BANDS = ('HN', 'LN', 'LS', 'HS')
COMBOS = {'NH': ('HN', 'LN'), 'SH': ('LS', 'HS'), 'Global': BANDS}


METHODS = ('latitude', 'bins')


def _check_method(method: str) -> str:
    method = str(method).lower()
    if method not in METHODS:
        raise ValueError(f"weighting_method must be one of {METHODS}, got {method!r}")
    return method


def _parse_gas_bins(raw: dict) -> dict[str, dict[str, list[dict]]]:
    """Normalise the gas_bins config into {gas: {hemisphere: [bin, ...]}}.

    Each bin is ``{name, sites (lower case), weight}``.  Hemispheres are NH and
    SH; bin names have to be unique within a gas, since they become columns.
    """
    out: dict[str, dict[str, list[dict]]] = {}
    for gas, hemis in raw.items():
        names: set[str] = set()
        out[gas] = {}
        for hemi, bins in hemis.items():
            if hemi not in ('NH', 'SH'):
                raise ValueError(f"gas_bins {gas}: hemisphere must be NH or SH, got {hemi!r}")
            out[gas][hemi] = []
            for name, spec in bins.items():
                if name in names or name in ('Global', 'NH', 'SH') + BANDS:
                    raise ValueError(f"gas_bins {gas}: bin name {name!r} is not unique")
                names.add(name)
                weight = float(spec['weight'])
                if weight <= 0 or not spec.get('sites'):
                    raise ValueError(f"gas_bins {gas}/{name}: needs sites and a positive weight")
                out[gas][hemi].append({'name': name, 'weight': weight,
                                       'sites': [x.lower() for x in spec['sites']]})
        if set(out[gas]) != {'NH', 'SH'}:
            raise ValueError(f"gas_bins {gas}: needs both NH and SH")
    return out


def normalize_gas_name(name: str) -> str:
    """Return *name* with hyphens stripped, for matching display names to config keys.

    logos_data analytes are display names from ``hats.analyte_list`` ('CFC-11',
    'HFC-134a'); the config uses GML gas names ('CFC11', 'HFC134a').
    """
    return name.replace('-', '').strip()


@dataclass
class GlobalMeansConfig:
    """Parsed configuration for the global-means calculation."""

    phi: float = 30.0
    background_sites: list[str] = field(default_factory=list)
    gas_background_overrides: dict[str, list[str]] = field(default_factory=dict)
    weight_lat_overrides: dict[str, float] = field(default_factory=dict)
    gas_weight_lat_overrides: dict[str, dict] = field(default_factory=dict)
    interpolate_site_gaps: bool = True
    interpolation_method: str = 'seasonal'
    max_interpolation_months: int = 3
    combined_source_gases: list[str] = field(default_factory=list)
    analyte_aliases: dict[str, str] = field(default_factory=dict)
    weighting_method: str = 'latitude'
    gas_bins: dict[str, dict[str, list[dict]]] = field(default_factory=dict)
    header_template: str = ''
    method_text: dict[str, str] = field(default_factory=dict)
    columns_text: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = CONFIG_FILE) -> 'GlobalMeansConfig':
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        return cls(
            phi=float(cfg.get('phi', 30.0)),
            background_sites=[s.lower() for s in cfg.get('background_sites', [])],
            gas_background_overrides={
                k: [s.lower() for s in v]
                for k, v in (cfg.get('gas_background_overrides') or {}).items()
            },
            weight_lat_overrides={
                k.lower(): float(v)
                for k, v in (cfg.get('weight_lat_overrides') or {}).items()
            },
            gas_weight_lat_overrides=cfg.get('gas_weight_lat_overrides') or {},
            interpolate_site_gaps=bool(cfg.get('interpolate_site_gaps', True)),
            interpolation_method=str(cfg.get('interpolation_method', 'seasonal')),
            max_interpolation_months=int(cfg.get('max_interpolation_months', 3)),
            combined_source_gases=cfg.get('combined_source_gases') or [],
            analyte_aliases=cfg.get('analyte_aliases') or {},
            weighting_method=_check_method(cfg.get('weighting_method', 'latitude')),
            gas_bins=_parse_gas_bins(cfg.get('gas_bins') or {}),
            header_template=cfg.get('global_means_file_header', ''),
            method_text={m: cfg.get(f'global_means_method_{m}', '') for m in METHODS},
            columns_text={m: cfg.get(f'global_means_columns_{m}', '') for m in METHODS},
        )

    # ── per-gas lookups ──────────────────────────────────────────────────────

    def gas_key(self, analyte: str) -> str:
        """Map a logos_data analyte display name onto a config gas key.

        Explicit ``analyte_aliases`` win; otherwise hyphens are stripped and the
        match is case-insensitive against the keys that appear in the config.
        """
        if analyte in self.analyte_aliases:
            return self.analyte_aliases[analyte]
        candidate = normalize_gas_name(analyte)
        known = (set(self.gas_background_overrides) | set(self.combined_source_gases)
                 | set(self.gas_bins))
        for entry in self.gas_weight_lat_overrides.values():
            known.update(entry.get('gases', []))
        for key in known:
            if key.lower() == candidate.lower():
                return key
        return candidate

    def is_combined_source(self, analyte: str) -> bool:
        """True when GML publishes this gas from blended fECD + MSD data.

        This export is M-system only, so those gases won't match the published
        global means exactly.
        """
        gas = self.gas_key(analyte)
        return any(g.lower() == gas.lower() for g in self.combined_source_gases)

    def bins_for(self, analyte: str) -> dict[str, list[dict]]:
        """Bins for *analyte* as {hemisphere: [{name, sites, weight}, ...]}.

        Empty unless ``weighting_method`` is ``bins`` and the gas has an entry in
        ``gas_bins``; an empty result means the latitude method is in force.
        """
        if self.weighting_method != 'bins':
            return {}
        return self.gas_bins.get(self.gas_key(analyte), {})

    def sites_for(self, analyte: str) -> list[str]:
        """Background sites to use for *analyte*."""
        bins = self.bins_for(analyte)
        if bins:
            sites: list[str] = []
            for hemi_bins in bins.values():
                for b in hemi_bins:
                    sites.extend(s for s in b['sites'] if s not in sites)
            return sites
        return self.gas_background_overrides.get(self.gas_key(analyte),
                                                 self.background_sites)

    def weight_lats_for(self, analyte: str) -> dict[str, float]:
        """Weighting-latitude overrides in force for *analyte*, as {site: lat}."""
        overrides = dict(self.weight_lat_overrides)
        gas = self.gas_key(analyte)
        for site, entry in self.gas_weight_lat_overrides.items():
            gases = entry.get('gases', [])
            if any(g.lower() == gas.lower() for g in gases):
                overrides[site.lower()] = float(entry['lat'])
        return overrides


class GlobalMeansCalculator:
    """Compute band, hemispheric and global monthly means from site monthly means.

    Parameters
    ----------
    config :
        A :class:`GlobalMeansConfig`; the packaged config is loaded if omitted.
    analyte :
        Display name of the compound, used to resolve per-gas config overrides.
    """

    def __init__(self, analyte: str, config: Optional[GlobalMeansConfig] = None):
        self.config = config or GlobalMeansConfig.load()
        self.analyte = analyte
        self.weight_lats = self.config.weight_lats_for(analyte)
        self.skipped_sites: list[str] = []
        self.bins = self.config.bins_for(analyte)
        self.mean_labels = (
            ['Global', 'NH', 'SH'] + [b['name'] for h in ('NH', 'SH') for b in self.bins[h]]
            if self.bins else ['Global', 'NH', 'SH', *BANDS]
        )

    @property
    def uses_bins(self) -> bool:
        """True when this gas is weighted with gas_bins rather than by latitude."""
        return bool(self.bins)

    # ── site frame preparation ───────────────────────────────────────────────

    def band_of(self, lat: float) -> str:
        """Semi-hemispheric band for a (possibly overridden) weighting latitude."""
        phi = self.config.phi
        if lat >= phi:
            return 'HN'
        if lat >= 0:
            return 'LN'
        if lat <= -phi:
            return 'HS'
        return 'LS'

    def fill_site_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reindex each site onto a continuous monthly series.

        Interior gaps are filled when the config asks for it -- ``mf`` by
        :meth:`_fill_series` (seasonal Holt-Winters by default), ``sd`` by a plain
        time interpolation.  Leading and trailing gaps are never extrapolated.
        Filled months carry ``n = 0``, which is what marks them as inferred.

        A run of missing months longer than ``max_interpolation_months`` is left
        empty instead of being bridged.  Straight-lining a long outage produces a
        confident-looking ramp that tracks nothing, and where a co-located
        surrogate is running -- MLO's PFPs during the 2022 eruption outage, a
        33-month hole in the programmatic flask record -- the fabricated series
        would also be weighted alongside the real one, double-counting the
        location.  Note pandas' own ``limit`` would fill the first N months of a
        longer run; the choice here is all-or-nothing per run.
        """
        out = []
        for site, grp in df.groupby('site', sort=True):
            grp = grp.set_index('date').sort_index()
            months = pd.date_range(grp.index.min(), grp.index.max(), freq='MS')
            observed = grp.index
            grp = grp.reindex(months)
            grp['site'] = site
            grp['n'] = grp['n'].fillna(0).astype(int)
            if self.config.interpolate_site_gaps:
                grp['mf'] = self._fill_series(grp['mf'])
                # sd gets a plain time interpolation, as in the GML_means loader:
                # a seasonal cycle in the mole fraction says nothing useful about
                # the scatter of that month's flask pairs.
                grp['sd'] = grp['sd'].interpolate(method='time', limit_area='inside')
                too_long = self._overlong_gaps(grp.index, observed)
                grp.loc[too_long, ['mf', 'sd']] = np.nan
            out.append(grp.rename_axis('date').reset_index())
        if not out:
            return pd.DataFrame()
        return pd.concat(out, ignore_index=True)

    # Monthly data, so one seasonal cycle is 12 points.  Holt-Winters needs at
    # least two full cycles before an additive seasonal term means anything.
    SEASONAL_PERIODS = 12
    MIN_SEASONAL_POINTS = 2 * SEASONAL_PERIODS

    def _fill_series(self, series: pd.Series) -> pd.Series:
        """Fill interior gaps in one site's monthly mole fractions.

        ``interpolation_method: seasonal`` (the default, and what the GML_means
        loader uses for MSD gases) fits additive Holt-Winters -- level, trend and
        a 12-month seasonal term -- and takes the model value only where an
        observation is missing.  Real observations are never replaced.  Falls back
        to a straight line for a series too short to carry two seasonal cycles, or
        if the fit will not converge.

        Leading and trailing gaps are left alone either way: the fit only spans
        the site's own first to last observation, so nothing is extrapolated.
        """
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            return series
        span = series.loc[first:last]
        linear = span.interpolate(method='time')

        if self.config.interpolation_method != 'seasonal':
            return linear.reindex(series.index)
        if span.notna().sum() < self.MIN_SEASONAL_POINTS:
            return linear.reindex(series.index)

        try:
            with warnings.catch_warnings():
                # A site with a ragged record often fits without fully converging;
                # the result is still usable and this is a GUI, not a log file.
                warnings.simplefilter('ignore')
                fit = ExponentialSmoothing(
                    linear,                      # the fit cannot run through NaNs
                    trend='add',
                    seasonal='add',
                    seasonal_periods=self.SEASONAL_PERIODS,
                    initialization_method='estimated',
                ).fit(optimized=True)
                model = fit.predict(start=span.index[0], end=span.index[-1])
        except (ValueError, np.linalg.LinAlgError):
            return linear.reindex(series.index)

        if not np.isfinite(model).all():
            return linear.reindex(series.index)
        return span.fillna(model).reindex(series.index)

    def _overlong_gaps(self, months: pd.DatetimeIndex,
                       observed: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask of months inside a missing run longer than the cap."""
        cap = self.config.max_interpolation_months
        missing = pd.Series(~months.isin(observed), index=months)
        run = missing.ne(missing.shift()).cumsum()
        run_len = missing.groupby(run).transform('size')
        return (missing & run_len.gt(cap)).to_numpy()

    def prepare(self, site_df: pd.DataFrame, site_lats: dict[str, float]) -> pd.DataFrame:
        """Attach weighting latitude, band and weight to a tidy site-month frame.

        *site_df* needs columns ``site``, ``date``, ``mf``, ``sd``, ``n``.
        Sites absent from *site_lats* are dropped and recorded in
        :attr:`skipped_sites`.
        """
        df = site_df.copy()
        df['site'] = df['site'].str.lower()

        known = set(site_lats)
        missing = sorted(set(df['site']) - known)
        if missing:
            self.skipped_sites.extend(missing)
            df = df[df['site'].isin(known)]
        if df.empty:
            return df

        df = self.fill_site_gaps(df)
        # Weighting latitudes: config overrides win over the site registry.
        df['weight_lat'] = [
            self.weight_lats.get(s, site_lats[s]) for s in df['site']
        ]
        df['band'] = df['weight_lat'].apply(self.band_of)
        df['weight'] = np.cos(np.deg2rad(df['weight_lat']))
        return df

    # ── the means ────────────────────────────────────────────────────────────

    @staticmethod
    def _band_stats(grp: pd.DataFrame) -> pd.Series:
        """cos(lat)-weighted mean of one band-month, plus its propagated variance."""
        w_sum = grp['weight'].sum()
        mean = (grp['mf'] * grp['weight']).sum() / w_sum
        # Independent sites: var = sum( (w_i/sum_w)^2 * sd_i^2 ).  A site whose
        # own sd is unknown contributes nothing rather than poisoning the sum.
        var = (((grp['weight'] / w_sum) * grp['sd'].fillna(0.0)) ** 2).sum()
        return pd.Series({'mean': mean, 'var': var})

    def band_means(self, prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (means, variances) per date for the four bands."""
        data = prepared.dropna(subset=['mf'])
        if data.empty:
            empty = pd.DataFrame(columns=list(BANDS))
            return empty, empty
        stats = (data.groupby(['date', 'band'])
                     .apply(self._band_stats, include_groups=False)
                     .unstack('band'))
        means = stats['mean'].reindex(columns=list(BANDS))
        variances = stats['var'].reindex(columns=list(BANDS))
        return means, variances

    def compute(self, site_df: pd.DataFrame, site_lats: dict[str, float]) -> pd.DataFrame:
        """Prepare *site_df* and compute the monthly means frame."""
        return self.compute_prepared(self.prepare(site_df, site_lats))

    def compute_prepared(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Compute the monthly means from an already-:meth:`prepare`\ d frame.

        Returns a DataFrame indexed by month start with columns
        ``Global, Global_sd, NH, NH_sd, SH, SH_sd`` followed by each band and
        its ``_sd``.  A combined mean is NaN unless every band it needs has a
        value that month.  Callers that also need the prepared frame should
        prepare once and use this, so the site frame is only built once.
        """
        if prepared.empty:
            return pd.DataFrame()
        if self.uses_bins:
            return self._compute_bins(prepared)

        means, variances = self.band_means(prepared)
        out = pd.DataFrame(index=means.index)

        for label, cols in COMBOS.items():
            cols = list(cols)
            k = len(cols)
            out[label] = means[cols].mean(axis=1, skipna=False)
            # sd of an unweighted average of k independent band means.
            out[f'{label}_sd'] = np.sqrt(variances[cols].sum(axis=1, min_count=k)) / k
        for band in BANDS:
            out[band] = means[band]
            out[f'{band}_sd'] = np.sqrt(variances[band])

        out.index.name = 'date'
        return out

    def _compute_bins(self, prepared: pd.DataFrame) -> pd.DataFrame:
        """Monthly means from the configured bins (see ``gas_bins`` in the config).

        A bin is the unweighted mean of the sites in it that have data that
        month; its variance is ``sum(sd_i^2) / k^2`` for the *k* sites present.
        A hemisphere is the weighted mean of its bins and is left NaN unless
        every bin has a value, matching how the latitude bands behave.  Global is
        ``(NH + SH) / 2``.  Returns the same columns as the latitude path, with
        the bins in place of HN/LN/LS/HS.
        """
        data = prepared.dropna(subset=['mf'])
        if data.empty:
            return pd.DataFrame()
        mf = data.pivot_table(index='date', columns='site', values='mf')
        sd = data.pivot_table(index='date', columns='site', values='sd')
        sd = sd.reindex(index=mf.index, columns=mf.columns)

        bin_mean, bin_var = {}, {}
        for hemi in ('NH', 'SH'):
            for b in self.bins[hemi]:
                cols = [s for s in b['sites'] if s in mf.columns]
                if not cols:
                    bin_mean[b['name']] = pd.Series(np.nan, index=mf.index)
                    bin_var[b['name']] = pd.Series(np.nan, index=mf.index)
                    continue
                have = mf[cols].notna()
                k = have.sum(axis=1).replace(0, np.nan)
                bin_mean[b['name']] = mf[cols].sum(axis=1, min_count=1) / k
                # A site without its own sd contributes nothing, not NaN.
                var = (sd[cols].where(have).fillna(0.0) ** 2).sum(axis=1) / k ** 2
                bin_var[b['name']] = var.where(k.notna())
        bin_mean = pd.DataFrame(bin_mean)
        bin_var = pd.DataFrame(bin_var)

        out = pd.DataFrame(index=mf.index)
        hemi_var = {}
        for hemi in ('NH', 'SH'):
            names = [b['name'] for b in self.bins[hemi]]
            w = pd.Series({b['name']: b['weight'] for b in self.bins[hemi]})
            w = w / w.sum()
            out[hemi] = (bin_mean[names] * w).sum(axis=1, min_count=len(names))
            hemi_var[hemi] = (bin_var[names] * w ** 2).sum(axis=1, min_count=len(names))
            out[f'{hemi}_sd'] = np.sqrt(hemi_var[hemi])
        out['Global'] = out[['NH', 'SH']].mean(axis=1, skipna=False)
        out['Global_sd'] = np.sqrt((hemi_var['NH'] + hemi_var['SH']).where(
            out['Global'].notna())) / 2
        for name in bin_mean.columns:
            out[name] = bin_mean[name]
            out[f'{name}_sd'] = np.sqrt(bin_var[name])
        out = out[['Global', 'Global_sd', 'NH', 'NH_sd', 'SH', 'SH_sd']
                  + [c for n in bin_mean.columns for c in (n, f'{n}_sd')]]
        out.index.name = 'date'
        return out
