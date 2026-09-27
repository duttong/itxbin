import dataclasses
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / 'logosdata'))

from combined_data import (  # noqa: E402
    CombinedConfig,
    CombinedDataBuilder,
    add_mismatch,
    combine_inverse_variance,
    combine_programs,
    inflate_filled_se,
    loess_site_series,
    pairwise_log_ratios,
    program_noise,
    programs_bitstring,
    smooth_series,
    solve_offsets,
)

MONTHS = pd.date_range('2000-01-01', periods=3, freq='MS', name='date')


def _frame(mf, se):
    return pd.DataFrame({'mf': mf, 'se': se}, index=MONTHS)


class CombineProgramsTests(unittest.TestCase):
    def test_inverse_se_weighting(self):
        a = _frame([100.0, 100.0, np.nan], [1.0, 1.0, 1.0])
        b = _frame([103.0, np.nan, 103.0], [2.0, 2.0, 2.0])

        out = combine_programs({'A': a, 'B': b})

        # weights 1 and 0.5: (100*1 + 103*0.5) / 1.5 = 101
        self.assertAlmostEqual(out['mf'].iloc[0], 101.0)
        self.assertAlmostEqual(out['sd'].iloc[0], 2 / 1.5)
        self.assertEqual(out['mf'].iloc[1], 100.0)
        self.assertEqual(out['mf'].iloc[2], 103.0)
        self.assertEqual(out['programs'].iloc[0], {'A', 'B'})
        self.assertEqual(out['programs'].iloc[2], {'B'})

    def test_mismatch_adds_excess_disagreement(self):
        a = _frame([100.0, 100.0, 100.0], [0.1, 0.1, 0.1])
        b = _frame([100.0, 110.0, 100.0], [0.1, 0.1, 0.1])
        combined = combine_programs({'A': a, 'B': b})

        sd = add_mismatch(combined, {'A': a, 'B': b})

        # month 1: mean 105, base sd 0.1; each program is 5 away, allowing 0.2
        self.assertAlmostEqual(sd.iloc[1], 0.1 + 2 * (5 - 0.2))
        self.assertAlmostEqual(sd.iloc[0], 0.1)

    def test_bitstring(self):
        self.assertEqual(programs_bitstring({'CATS', 'MSD'}, ['oldGC', 'CATS', 'MSD']), '011')


class SmoothSeriesTests(unittest.TestCase):
    def test_short_runs_and_endpoints_untouched(self):
        idx = pd.date_range('2000-01-01', periods=12, freq='MS')
        vals = [1, 5, 2, 6, 3, 7, 4, 8, np.nan, 10, 11, 12]
        s = pd.Series(vals, index=idx, dtype=float)

        out = smooth_series(s, 7, 4)

        self.assertEqual(out.iloc[0], 1)
        self.assertEqual(out.iloc[7], 8)
        self.assertTrue(np.isnan(out.iloc[8]))
        pd.testing.assert_series_equal(out.iloc[9:], s.iloc[9:])
        self.assertNotEqual(out.iloc[3], 6)

    def test_window_zero_is_noop(self):
        s = pd.Series([1.0, 2.0, 3.0])
        self.assertIs(smooth_series(s, 0, 4), s)


class LoessTests(unittest.TestCase):
    def test_fills_every_month_and_follows_a_trend(self):
        # Summer-only sampling, like SPO before 1992, on a straight line.
        dates = pd.to_datetime([f'{y}-{m:02d}-01' for y in range(1980, 1986) for m in (1, 2, 12)])
        mf = 100 + 2.0 * (dates.year - 1980 + (dates.month - 0.5) / 12)
        obs = pd.DataFrame({'date': dates, 'mf': mf, 'se': 0.5, 'n': 3})

        out = loess_site_series(obs, 30)

        self.assertEqual(len(out), (1985 - 1980) * 12 + 12)
        self.assertEqual(out.index[0], dates[0])
        july = out.loc['1983-07-01']
        self.assertAlmostEqual(july['mf'], 100 + 2.0 * (3 + 6.5 / 12), places=6)
        self.assertEqual(july['n'], 0)
        self.assertEqual(out.loc['1983-01-01', 'n'], 3)
        self.assertTrue(out['se'].notna().all())


class OffsetEstimateTests(unittest.TestCase):
    def test_chain_through_a_middle_program(self):
        # C never overlaps A (the reference) but both overlap B.
        idx = pd.MultiIndex.from_product([['brw'], pd.date_range('2000-01-01', periods=24,
                                                                 freq='MS')],
                                         names=['site', 'date'])
        a = pd.Series(100.0, index=idx[:16])
        b = pd.Series(102.0, index=idx)
        c = pd.Series(101.0, index=idx[16:])
        pairs = pairwise_log_ratios({'A': a, 'B': b, 'C': c}, min_overlap=6)
        self.assertEqual(set(zip(pairs.a, pairs.b)), {('A', 'B'), ('B', 'C')})

        level = solve_offsets(pairs, 'A')

        self.assertAlmostEqual(level['B'], 100 * np.log(1.02), places=6)
        self.assertAlmostEqual(level['C'], 100 * np.log(1.01), places=6)

    def test_unlinked_program_left_out(self):
        pairs = pd.DataFrame({'a': ['A'], 'b': ['B'], 'pct': [1.0], 'n': [20]})
        level = solve_offsets(pairs, 'A')
        self.assertNotIn('C', level)
        self.assertAlmostEqual(level['B'], -1.0)


class InverseVarianceTests(unittest.TestCase):
    def test_weights_and_birge(self):
        a = _frame([100.0, 100.0, np.nan], [1.0, 1.0, 1.0])
        b = _frame([100.5, 110.0, 103.0], [2.0, 2.0, 2.0])

        out = combine_inverse_variance({'A': a, 'B': b})

        # month 0: weights 1 and 0.25 -> 100.1; chi2 = 0.1^2 + 0.4^2/4 < 1
        self.assertAlmostEqual(out['mf'].iloc[0], 100.1)
        self.assertAlmostEqual(out['sd'].iloc[0], 1 / np.sqrt(1.25))
        # month 1: 10 ppt apart, far outside the errors -> Birge-scaled
        chi2 = (2.0 ** 2) + (8.0 ** 2) / 4
        self.assertAlmostEqual(out['sd'].iloc[1], np.sqrt(chi2) / np.sqrt(1.25))
        self.assertEqual(out['mf'].iloc[2], 103.0)
        self.assertEqual(out['sd'].iloc[2], 2.0)
        self.assertEqual(out['programs'].iloc[2], {'B'})


class InflateFilledTests(unittest.TestCase):
    def test_se_grows_with_distance_from_data(self):
        idx = pd.date_range('2000-01-01', periods=6, freq='MS')
        f = pd.DataFrame({'se': 1.0, 'n': [3, 0, 0, 0, 2, 0]}, index=idx)
        np.testing.assert_allclose(inflate_filled_se(f),
                                   [1, np.sqrt(2), np.sqrt(3), np.sqrt(2), 1, np.sqrt(2)])


class ProgramNoiseTests(unittest.TestCase):
    def test_three_cornered_hat_recovers_noise(self):
        rng = np.random.default_rng(1)
        idx = pd.MultiIndex.from_product(
            [['brw', 'mlo'], pd.date_range('1990-01-01', periods=600, freq='MS')],
            names=['site', 'date'])
        truth = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        sigma = {'A': 0.2, 'B': 0.3, 'C': 0.4}
        series = {k: truth + 5.0 * (k == 'C') + rng.normal(0, s, len(idx))
                  for k, s in sigma.items()}

        noise = program_noise(series)

        for k, s in sigma.items():
            self.assertAlmostEqual(noise[k], s, delta=0.15 * s)


class BuilderUnitTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CombinedConfig.load()

    def _builder(self, gas_cfg):
        b = CombinedDataBuilder.__new__(CombinedDataBuilder)
        b.config = self.cfg
        b.gas_cfg = gas_cfg
        return b

    def test_offsets_scalar_and_ranged(self):
        b = self._builder({'offsets_pct': {
            'CATS': -2.0,
            'MSD': [{'pct': 1.0, 'end': '2009-12-31'}],
        }})
        df = pd.DataFrame({'date': pd.to_datetime(['2005-01-01', '2015-01-01']),
                           'mf': [100.0, 100.0]})
        np.testing.assert_allclose(b.apply_offsets('CATS', df), [98.0, 98.0])
        np.testing.assert_allclose(b.apply_offsets('MSD', df), [101.0, 100.0])
        np.testing.assert_allclose(b.apply_offsets('fECD', df), [100.0, 100.0])

    def test_standard_errors_igor(self):
        b = self._builder({'se_cap': {'fECD': 0.7}})
        b.config = dataclasses.replace(self.cfg, se_method='igor')
        df = pd.DataFrame({'site': ['brw', 'smo', 'brw'], 'sd': [2.0, 2.0, np.nan],
                           'n': [4, 4, 1]})
        se = b.standard_errors('fECD', df)
        # sqrt_n: 2/2 = 1 -> capped 0.7; smo doubled; the NaN gets brw's median
        np.testing.assert_allclose(se, [0.7, 1.4, 0.7])

    def test_offset_segments(self):
        b = self._builder({'programs': ['fECD', 'MSD'],
                           'offset_breaks': {'fECD': ['2019-09-01']}})
        self.assertEqual(b.offset_segments(), [
            ('fECD[..2019-08-31]', 'fECD', None, '2019-08-31'),
            ('fECD[2019-09-01..]', 'fECD', '2019-09-01', None),
            ('MSD', 'MSD', None, None),
        ])

    def test_pfp_pairs_only_in_pfp_program(self):
        class FakeDb:
            def doquery(self, sql, params=None):
                return [{'site': 'mlo', 'dt': '2023-01-05', 'value': 200.0},
                        {'site': 'mlo_pfp', 'dt': '2023-01-09', 'value': 210.0},
                        {'site': 'brw', 'dt': '2023-01-02', 'value': 205.0}]
        b = self._builder({'parameter_num': 22})
        b.db = FakeDb()
        msd = b._load_pairs('MSD', {'inst_ids': ['M3']})
        pfp = b._load_pairs('PFP', {'inst_ids': ['M3'], 'pfp': 'only'})
        self.assertEqual(dict(zip(msd.site, msd.mf)), {'brw': 205.0, 'mlo': 200.0})
        self.assertEqual(dict(zip(pfp.site, pfp.mf)), {'mlo': 210.0})

    def test_pfp_label(self):
        b = self._builder({})
        sql = b._pfp_label_sql()
        self.assertIn("WHEN LOWER(v.site) = 'mlo' AND v.pair_id_num = 0 THEN 'mlo_pfp'", sql)


if __name__ == '__main__':
    unittest.main()
