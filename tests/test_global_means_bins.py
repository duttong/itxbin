"""Bin-weighted hemispheric means (gas_bins in gml_global_means_config.yaml).

Montzka builds CH3Br, CH3Cl and CH3CCl3 means from hand-chosen bins rather than
latitude bands: a bin is the plain mean of its sites, a hemisphere is the
weighted mean of its bins, and Global is (NH + SH) / 2.
"""
import copy
import unittest

import numpy as np
import pandas as pd

BINS = {'TESTGAS': {
    'NH': {'NHt': {'sites': ['mlo', 'kum'], 'weight': 0.5},
           'NHh': {'sites': ['brw'], 'weight': 0.25},
           'NHm': {'sites': ['lef'], 'weight': 0.25}},
    'SH': {'SHt': {'sites': ['smo'], 'weight': 0.5},
           'SHm': {'sites': ['cgo'], 'weight': 0.25},
           'SHh': {'sites': ['spo'], 'weight': 0.25}},
}}
LATS = {'mlo': 19.5, 'kum': 19.5, 'brw': 71.3, 'lef': 45.9,
        'smo': -14.2, 'cgo': -40.7, 'spo': -90.0}


def _config(method='bins'):
    from global_means import GlobalMeansConfig, _parse_gas_bins
    cfg = copy.deepcopy(GlobalMeansConfig.load())
    cfg.gas_bins = _parse_gas_bins(BINS)
    cfg.weighting_method = method
    cfg.interpolate_site_gaps = False
    return cfg


def _frame(values, sd=0.1):
    """One month of site values, {site: mf}; None leaves the site out."""
    rows = [{'site': s, 'date': pd.Timestamp('2020-01-01'), 'mf': v, 'sd': sd, 'n': 3}
            for s, v in values.items() if v is not None]
    return pd.DataFrame(rows)


def _calc(values, method='bins'):
    from global_means import GlobalMeansCalculator
    calc = GlobalMeansCalculator('TESTGAS', config=_config(method))
    return calc.compute(_frame(values), LATS).iloc[0]


FULL = {'mlo': 10.0, 'kum': 12.0, 'brw': 8.0, 'lef': 9.0,
        'smo': 5.0, 'cgo': 4.0, 'spo': 3.0}


class BinMeansTests(unittest.TestCase):
    def test_hand_calculation(self):
        out = _calc(FULL)
        # NHt = (10+12)/2 = 11 ; NH = .5*11 + .25*8 + .25*9 = 9.75
        self.assertAlmostEqual(out['NHt'], 11.0)
        self.assertAlmostEqual(out['NH'], 9.75)
        # SH = .5*5 + .25*4 + .25*3 = 4.25
        self.assertAlmostEqual(out['SH'], 4.25)
        self.assertAlmostEqual(out['Global'], 7.0)

    def test_weights_need_not_sum_to_one(self):
        # CH3CCl3's bin weights are cosines (0.970, 0.751, 0.402 ...), not shares.
        from global_means import _parse_gas_bins
        cfg = _config()
        cfg.gas_bins = _parse_gas_bins({'TESTGAS': {
            'NH': {'a': {'sites': ['mlo'], 'weight': 0.97},
                   'b': {'sites': ['brw'], 'weight': 0.40}},
            'SH': {'c': {'sites': ['smo'], 'weight': 0.97},
                   'd': {'sites': ['spo'], 'weight': 0.40}}}})
        from global_means import GlobalMeansCalculator
        out = GlobalMeansCalculator('TESTGAS', config=cfg).compute(
            _frame(FULL), LATS).iloc[0]
        self.assertAlmostEqual(out['NH'], (0.97 * 10 + 0.40 * 8) / 1.37)

    def test_missing_site_in_bin_averages_the_rest(self):
        out = _calc({**FULL, 'kum': None})
        self.assertAlmostEqual(out['NHt'], 10.0)
        self.assertAlmostEqual(out['NH'], .5 * 10 + .25 * 8 + .25 * 9)

    def test_missing_bin_blanks_its_hemisphere_and_global(self):
        out = _calc({**FULL, 'brw': None})
        self.assertTrue(np.isnan(out['NH']))
        self.assertTrue(np.isnan(out['Global']))
        self.assertAlmostEqual(out['SH'], 4.25)

    def test_uncertainty_propagation(self):
        out = _calc(FULL)
        # NHt: two sites sd .1 -> sqrt(2*.1^2)/2 ; others a single site sd .1.
        var_nht = 2 * 0.1 ** 2 / 4
        var_nh = (0.5 ** 2) * var_nht + 0.25 ** 2 * 0.01 + 0.25 ** 2 * 0.01
        var_sh = 0.5 ** 2 * 0.01 + 0.25 ** 2 * 0.01 + 0.25 ** 2 * 0.01
        self.assertAlmostEqual(out['NH_sd'], np.sqrt(var_nh))
        self.assertAlmostEqual(out['Global_sd'], np.sqrt(var_nh + var_sh) / 2)

    def test_site_without_sd_contributes_no_variance(self):
        frame = _frame(FULL)
        frame.loc[frame.site == 'smo', 'sd'] = np.nan
        from global_means import GlobalMeansCalculator
        out = GlobalMeansCalculator('TESTGAS', config=_config()).compute(frame, LATS).iloc[0]
        self.assertAlmostEqual(out['SHt_sd'], 0.0)
        self.assertFalse(np.isnan(out['SH_sd']))


class MethodSwitchTests(unittest.TestCase):
    def test_latitude_method_ignores_bins(self):
        from global_means import GlobalMeansCalculator
        calc = GlobalMeansCalculator('TESTGAS', config=_config('latitude'))
        self.assertFalse(calc.uses_bins)
        out = calc.compute(_frame(FULL), LATS).iloc[0]
        self.assertIn('HN', out.index)
        self.assertNotIn('NHt', out.index)

    def test_bins_take_the_sites_from_the_bins(self):
        cfg = _config()
        self.assertEqual(set(cfg.sites_for('TESTGAS')), set(LATS))
        self.assertEqual(len(cfg.sites_for('TESTGAS')), len(LATS))   # no duplicates
        cfg.weighting_method = 'latitude'
        self.assertEqual(cfg.sites_for('TESTGAS'), cfg.background_sites)

    def test_gas_without_bins_stays_on_latitude(self):
        from global_means import GlobalMeansCalculator
        cfg = _config('bins')
        calc = GlobalMeansCalculator('HCFC-22', config=cfg)
        self.assertFalse(calc.uses_bins)
        self.assertEqual(calc.mean_labels[-4:], ['HN', 'LN', 'LS', 'HS'])

    def test_shipped_config_defines_the_three_gases(self):
        from global_means import GlobalMeansConfig
        cfg = GlobalMeansConfig.load()
        cfg.weighting_method = 'bins'
        for gas in ('CH3Br', 'CH3Cl', 'CH3CCl3'):
            bins = cfg.bins_for(gas)
            self.assertEqual(set(bins), {'NH', 'SH'}, gas)
            self.assertTrue(all(len(v) == 3 for v in bins.values()), gas)
            # MLO's PFP sits in the MLO bin so it is not counted twice.
            self.assertIn('mlo_pfp', cfg.sites_for(gas), gas)


class ValidationTests(unittest.TestCase):
    def test_bad_method(self):
        from global_means import _check_method
        with self.assertRaises(ValueError):
            _check_method('sine')

    def test_duplicate_bin_name(self):
        from global_means import _parse_gas_bins
        bad = {'G': {'NH': {'a': {'sites': ['mlo'], 'weight': 1}},
                     'SH': {'a': {'sites': ['smo'], 'weight': 1}}}}
        with self.assertRaises(ValueError):
            _parse_gas_bins(bad)

    def test_reserved_name_and_missing_hemisphere(self):
        from global_means import _parse_gas_bins
        with self.assertRaises(ValueError):
            _parse_gas_bins({'G': {'NH': {'HN': {'sites': ['mlo'], 'weight': 1}},
                                   'SH': {'b': {'sites': ['smo'], 'weight': 1}}}})
        with self.assertRaises(ValueError):
            _parse_gas_bins({'G': {'NH': {'a': {'sites': ['mlo'], 'weight': 1}}}})


if __name__ == '__main__':
    unittest.main()
