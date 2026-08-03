import unittest

import numpy as np
import pandas as pd


class M4PfpLabelTests(unittest.TestCase):
    def setUp(self):
        from logos_instruments import M4_Instrument

        self.instrument = object.__new__(M4_Instrument)

    def test_second_pfp_falls_back_to_port_info(self):
        df = pd.DataFrame(
            [
                {
                    'run_type_num': 5, 'port': 1, 'flask_port': 5,
                    'site': 'MLO', 'pair_id_num': 0,
                    'sample_id': '3930-05', 'port_info': '5-3930',
                },
                {
                    'run_type_num': 5, 'port': 12, 'flask_port': 5,
                    'site': None, 'pair_id_num': 0,
                    'sample_id': '0', 'port_info': '5-3129',
                },
                {
                    'run_type_num': 8, 'port': 14, 'flask_port': None,
                    'site': None, 'pair_id_num': 0,
                    'sample_id': '0', 'port_info': 'esx-3608',
                },
                {
                    'run_type_num': 7, 'port': 16, 'flask_port': None,
                    'site': None, 'pair_id_num': 0,
                    'sample_id': '0', 'port_info': 'sx-3579_Arch_A',
                },
            ]
        )

        result = self.instrument.add_port_labels(df)

        self.assertEqual(result.loc[0, 'port_label'], 'MLO 3930-05 (5)')
        self.assertEqual(result.loc[1, 'port_label'], '3129-05 (5)')
        self.assertEqual(result.loc[2, 'port_label'], 'esx-3608 (14)')
        self.assertEqual(result.loc[3, 'port_label'], 'sx-3579_Arch_A (16)')
        self.assertFalse(result['port_label'].str.contains('nan', case=False).any())

    def test_port_info_identity_keeps_missing_metadata_packages_distinct(self):
        pfp_df = pd.DataFrame(
            {
                'sample_id': ['0', '0'],
                'port_info': ['5-3129', '5-4001'],
            }
        )

        identities = self.instrument._pfp_sample_identity(pfp_df)

        self.assertEqual(identities.tolist(), ['3129-05', '4001-05'])

    def test_second_same_site_pfp_uses_darker_site_color_per_run(self):
        rows = []
        for analysis_time, package, flask in [
            ('2026-07-31 14:00:00', '3930', 1),
            ('2026-07-31 14:22:00', '3930', 2),
            ('2026-07-31 19:56:00', '3129', 1),
            ('2026-07-31 20:19:00', '3129', 2),
        ]:
            rows.append(
                {
                    'analysis_datetime': analysis_time,
                    'run_time': '2026-07-31 11:02:00',
                    'run_type_num': 5,
                    'port': 1,
                    'flask_port': flask,
                    'site': 'MLO',
                    'pair_id_num': 0,
                    'sample_id': f'{package}-{flask:02d}',
                    'port_info': f'{flask}-{package}',
                }
            )
        rows.append(
            {
                'analysis_datetime': '2026-08-02 14:00:00',
                'run_time': '2026-08-02 11:00:00',
                'run_type_num': 5,
                'port': 1,
                'flask_port': 1,
                'site': 'MLO',
                'pair_id_num': 0,
                'sample_id': '3129-01',
                'port_info': '1-3129',
            }
        )

        result = self.instrument.add_port_labels(pd.DataFrame(rows))

        first_color = result.loc[0, 'port_color']
        second_color = result.loc[2, 'port_color']
        self.assertEqual(result.loc[1, 'port_color'], first_color)
        self.assertEqual(result.loc[3, 'port_color'], second_color)
        self.assertNotEqual(second_color, first_color)
        self.assertTrue(all(b < a for a, b in zip(first_color[:3], second_color[:3])))
        self.assertEqual(result.loc[4, 'port_color'], first_color)


class M4PfpRmsTests(unittest.TestCase):
    def setUp(self):
        from logos_instruments import Normalizing

        self.normalizing = Normalizing('m4', 8, 'run_type_num', 'area')

    @staticmethod
    def pfp_run():
        return pd.DataFrame(
            [
                # First PFP inlet: two complete pairs.
                {'port': 1, 'flask_port': 3, 'normalized_resp': 0.98},
                {'port': 1, 'flask_port': 1, 'normalized_resp': 1.00},
                {'port': 1, 'flask_port': 4, 'normalized_resp': 0.99},
                {'port': 1, 'flask_port': 2, 'normalized_resp': 1.02},
                # A repeat injection is averaged before forming the pair.
                {'port': 1, 'flask_port': 2, 'normalized_resp': 1.02},
                # Second PFP inlet: one complete and one incomplete pair.
                {'port': 12, 'flask_port': 1, 'normalized_resp': 1.10},
                {'port': 12, 'flask_port': 2, 'normalized_resp': 1.13},
                {'port': 12, 'flask_port': 3, 'normalized_resp': 1.01},
            ]
        ).assign(
            run_time=pd.Timestamp('2026-07-31 11:02:00'),
            run_type_num=5,
            rejected=0,
            sample_id='0',
            pair_id_num=0,
            port_info='',
        )

    def test_m4_pfp_uses_complete_odd_even_pairs_per_inlet(self):
        rms, count = self.normalizing.sample_diffs(
            self.pfp_run(), verbose=False
        )

        self.assertEqual(count, 3)
        self.assertAlmostEqual(
            rms,
            np.sqrt((0.02**2 + 0.01**2 + 0.03**2) / 3),
        )

    def test_m4_pfp_outlier_rule_applies_to_pair_differences(self):
        rms, count = self.normalizing.sample_diffs(
            self.pfp_run(), verbose=False, drop_outlier=True
        )

        self.assertEqual(count, 2)
        self.assertAlmostEqual(rms, np.sqrt((0.02**2 + 0.01**2) / 2))

    def test_non_m4_pfp_rows_do_not_change_existing_flask_rms(self):
        from logos_instruments import Normalizing

        normalizing = Normalizing('fe3', 8, 'run_type_num', 'area')
        df = self.pfp_run()
        flask_rows = pd.DataFrame(
            [
                {
                    'port': 1, 'flask_port': np.nan,
                    'normalized_resp': 1.00, 'sample_id': '1001',
                },
                {
                    'port': 2, 'flask_port': np.nan,
                    'normalized_resp': 1.04, 'sample_id': '1002',
                },
            ]
        ).assign(
            run_time=pd.Timestamp('2026-07-31 11:02:00'),
            run_type_num=1,
            rejected=0,
            pair_id_num=77,
            port_info='',
        )

        rms, count = normalizing.sample_diffs(
            pd.concat([df, flask_rows], ignore_index=True), verbose=False
        )

        self.assertEqual(count, 1)
        self.assertAlmostEqual(rms, 0.04)


class M4PfpEventResolverTests(unittest.TestCase):
    class FakeDB:
        def __init__(self, exact=None, recent=None, exact_sequence=None):
            self.exact = exact or []
            self.recent = recent or []
            self.exact_sequence = list(exact_sequence) if exact_sequence is not None else None
            self.queries = []

        def doquery(self, sql):
            self.queries.append(sql)
            if 'DATE_SUB' in sql:
                return self.recent
            if self.exact_sequence is not None:
                return self.exact_sequence.pop(0)
            return self.exact

    def make_instrument(self, db):
        from m4_samplogs import M4_SampleLogs

        instrument = object.__new__(M4_SampleLogs)
        instrument.db = db
        return instrument

    def test_site_date_lookup_is_case_insensitive_and_not_site_limited(self):
        db = self.FakeDB(
            exact=[
                {
                    'num': 123,
                    'id': '4000-05',
                    'date': '2024-10-25',
                    'site': 'BLD',
                }
            ]
        )
        instrument = self.make_instrument(db)

        match = instrument.resolve_pfp_event(
            {
                'samptype': 'pfp',
                'tank': '5-xxxx',
                'site': 'bld',
                'sample_time': '241025',
                'dt_run': '2024-10-31 13:10:00',
            }
        )

        self.assertEqual(match['event_num'], 123)
        self.assertEqual(match['method'], 'site_date')
        self.assertIn("LOWER(s.code) = LOWER('bld')", db.queries[0])

    def test_numeric_package_uses_recent_exact_event_fallback(self):
        db = self.FakeDB(
            recent=[
                {
                    'num': 566747,
                    'id': '3129-05',
                    'date': '2026-07-12',
                    'site': 'MLO',
                },
                {
                    'num': 562869,
                    'id': '3129-05',
                    'date': '2025-11-30',
                    'site': 'AMT',
                },
            ]
        )
        instrument = self.make_instrument(db)

        match = instrument.resolve_pfp_event(
            {
                'samptype': 'pfp',
                'tank': '5-3129',
                'site': None,
                'sample_time': None,
                'dt_run': '2026-07-31 19:56:00',
            }
        )

        self.assertEqual(match['event_num'], 566747)
        self.assertEqual(match['site'], 'MLO')
        self.assertEqual(match['method'], 'recent_package')
        self.assertIn("e.id = '3129-05'", db.queries[0])
        self.assertIn('INTERVAL 120 DAY', db.queries[0])

    def test_numeric_package_typo_uses_unique_same_site_date_flask_match(self):
        db = self.FakeDB(
            exact_sequence=[
                [],
                [],
                [
                    {
                        'num': 564963,
                        'id': '3937-05',
                        'date': '2026-05-25',
                        'site': 'MKO',
                    }
                ],
            ]
        )
        instrument = self.make_instrument(db)

        match = instrument.resolve_pfp_event(
            {
                'samptype': 'pfp',
                'tank': '5-3947',
                'site': 'mko',
                'sample_time': '260525',
                'dt_run': '2026-06-26 20:26:00',
            }
        )

        self.assertEqual(match['event_num'], 564963)
        self.assertEqual(match['event_id'], '3937-05')
        self.assertEqual(match['site'], 'MKO')
        self.assertEqual(match['method'], 'site_date_flask')
        self.assertEqual(len(db.queries), 3)
        self.assertIn("e.id = '3947-05'", db.queries[0])
        self.assertIn("e.id = '3947-05'", db.queries[1])
        self.assertIn("e.id LIKE '%-05'", db.queries[2])

    def test_numeric_site_typo_uses_unique_same_date_package_match(self):
        db = self.FakeDB(
            exact_sequence=[
                [],
                [
                    {
                        'num': 566747,
                        'id': '3129-05',
                        'date': '2026-07-12',
                        'site': 'MLO',
                    }
                ],
                [],
            ]
        )
        instrument = self.make_instrument(db)

        match = instrument.resolve_pfp_event(
            {
                'samptype': 'pfp',
                'tank': '5-3129',
                'site': 'mko',
                'sample_time': '260712',
                'dt_run': '2026-07-31 19:56:00',
            }
        )

        self.assertEqual(match['event_num'], 566747)
        self.assertEqual(match['event_id'], '3129-05')
        self.assertEqual(match['site'], 'MLO')
        self.assertEqual(match['method'], 'site_date_package')

    def test_numeric_package_with_source_metadata_does_not_fall_back_to_other_site(self):
        db = self.FakeDB(
            exact_sequence=[[], [], []],
            recent=[
                {
                    'num': 564552,
                    'id': '3947-05',
                    'date': '2026-05-07',
                    'site': 'WKT',
                }
            ],
        )
        instrument = self.make_instrument(db)

        match = instrument.resolve_pfp_event(
            {
                'samptype': 'pfp',
                'tank': '5-3947',
                'site': 'MKO',
                'sample_time': '260525',
                'dt_run': '2026-06-26 20:26:00',
            }
        )

        self.assertIsNone(match)
        self.assertFalse(any('DATE_SUB' in query for query in db.queries))

    def test_unknown_package_does_not_guess_without_exact_metadata_match(self):
        db = self.FakeDB()
        instrument = self.make_instrument(db)

        match = instrument.resolve_pfp_event(
            {
                'samptype': 'pfp',
                'tank': '5-xxxx',
                'site': 'MKO',
                'sample_time': '240326',
                'dt_run': '2024-04-23 20:00:00',
            }
        )

        self.assertIsNone(match)
        self.assertEqual(len(db.queries), 1)

    def test_run_index_parser_preserves_pfp_source_metadata(self):
        from m4_samplogs import parse_run_index_sample_info

        parsed = parse_run_index_sample_info('MLO_15_Sep_24_#5-xxxx')

        self.assertEqual(
            parsed,
            {'site': 'MLO', 'sample_time': '240915', 'tank': '5-xxxx'},
        )

    def test_backfill_update_is_limited_to_matched_missing_rows(self):
        from m4_pfp_event_backfill import apply_backfill

        class BackfillDB:
            def __init__(self):
                self.queries = []

            def doquery(self, sql):
                self.queries.append(sql)
                if 'SELECT COUNT(*) AS updated' in sql:
                    return [{'updated': 2}]
                return None

        class Instrument:
            inst_num = 192
            db = BackfillDB()

        audit = pd.DataFrame(
            {
                'num': [10, 11, 12],
                'event_num': [100, 101, pd.NA],
            }
        )

        updated = apply_backfill(Instrument(), audit)

        self.assertEqual(updated, 2)
        update_sql = Instrument.db.queries[0]
        self.assertIn('WHEN 10 THEN 100', update_sql)
        self.assertIn('WHEN 11 THEN 101', update_sql)
        self.assertNotIn('WHEN 12', update_sql)
        self.assertIn('(ccgg_event_num IS NULL OR ccgg_event_num = 0)', update_sql)


if __name__ == '__main__':
    unittest.main()
