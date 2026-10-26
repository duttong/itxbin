import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'logosdata'), str(ROOT)]
from logos_instruments_insitu import IE3_Instrument


class ScaleAssignmentFactorsTest(unittest.TestCase):
    """IE3_Instrument._apply_scale_assignment_factors (no database needed)."""

    def setUp(self):
        self.inst = IE3_Instrument.__new__(IE3_Instrument)   # skip __init__ (DB)

    def test_named_fill_is_scaled(self):
        hist = [{'fill_code': 'A', 'coef0': 4.0, 'unc_c0': 0.02, 'start_date': '2012-05-02'}]
        out = self.inst._apply_scale_assignment_factors('AAL072269', 26, hist)
        self.assertAlmostEqual(out[0]['coef0'], 3.88)
        self.assertAlmostEqual(out[0]['unc_c0'], 0.0194)
        self.assertEqual(out[0]['factor'], 0.97)
        self.assertEqual(hist[0]['coef0'], 4.0)           # input not modified

    def test_other_fill_gas_or_tank_untouched(self):
        hist = [{'fill_code': 'B', 'coef0': 3.5, 'unc_c0': 0.01}]
        self.assertEqual(self.inst._apply_scale_assignment_factors('AAL072269', 26, hist), hist)
        hist = [{'fill_code': 'A', 'coef0': 500.0, 'unc_c0': 1.0}]
        self.assertEqual(self.inst._apply_scale_assignment_factors('AAL072269', 22, hist), hist)
        self.assertEqual(self.inst._apply_scale_assignment_factors('CC000000', 26, hist), hist)

    def test_drift_terms_scaled(self):
        hist = [{'fill_code': 'A', 'coef0': 4.0, 'unc_c0': 0.0,
                 'drift': {'tzero': 2013.0, 'coef1': 0.1, 'coef2': 0.01}}]
        out = self.inst._apply_scale_assignment_factors('AAL072101', 26, hist)
        self.assertAlmostEqual(out[0]['drift']['coef1'], 0.097)
        self.assertAlmostEqual(out[0]['drift']['coef2'], 0.0097)
        self.assertAlmostEqual(IE3_Instrument.fill_value_at(out[0], '2014-01-01'),
                               0.97 * IE3_Instrument.fill_value_at(hist[0], '2014-01-01'))


if __name__ == '__main__':
    unittest.main()
