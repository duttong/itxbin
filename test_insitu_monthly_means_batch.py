from datetime import date
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

from insitu_monthly_means_batch import _resolve_gases, refresh


class MonthlyGasSelectionTest(TestCase):
    def test_resolve_comma_separated_gases(self):
        db = Mock()
        db.doquery.return_value = [
            {'num': 22, 'gas': 'cfc12'},
            {'num': 32, 'gas': 'cfc113'},
        ]

        self.assertEqual(_resolve_gases(db, ' CFC-12, cfc113,cfc12 '), (22, 32))
        self.assertEqual(db.doquery.call_args.args[1], ('cfc12', 'cfc113'))

    def test_unknown_gas_rejects_the_list(self):
        db = Mock()
        db.doquery.return_value = [{'num': 22, 'gas': 'cfc12'}]

        with self.assertRaisesRegex(ValueError, 'Unknown gas formula.*typo'):
            _resolve_gases(db, 'cfc12,typo')

    def test_selected_gases_scope_aggregation_and_cleanup(self):
        existing = [
            {'site_num': 92, 'parameter_num': 22, 'month': date(2017, 7, 1)},
            {'site_num': 92, 'parameter_num': 32, 'month': date(2017, 8, 1)},
        ]
        row = dict(existing[0], mean=507.9, std=0.5, n=95)
        for computed in ([], [row]):
            with self.subTest(computed=computed):
                db = Mock()
                db.doquery.side_effect = ([computed, None, existing, None] if computed
                                          else [computed, existing, None])
                inst = SimpleNamespace(inst_id='cats', site='smo', inst_num=243, AIR_PORTS=[4, 8], db=db)

                refresh(inst, write=True, verbose=False, parameter_nums=(22, 32))

                calls = db.doquery.call_args_list
                self.assertIn('AND v.parameter_num IN (22, 32)', calls[0].args[0])
                self.assertIn('hats.ng_preferred_channel', calls[0].args[0])
                self.assertIn('WHERE inst_num = 243 AND parameter_num IN (22, 32)', calls[-2].args[0])
                self.assertEqual(calls[-1].args[1], [
                    (243, item['site_num'], item['parameter_num'], item['month'])
                    for item in (existing[1:] if computed else existing)
                ])
                self.assertEqual(any('INSERT INTO' in call.args[0] for call in calls), bool(computed))
