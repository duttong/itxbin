from unittest import TestCase

import logos_jobs


class MeansJobPlanTest(TestCase):
    def tail(self, job):
        argv, what = logos_jobs.plan(job)
        self.assertTrue(argv[1].endswith('insitu_monthly_means_batch.py'))
        return argv[2:], what

    def test_one_analyte_of_a_cats_site(self):
        args, what = self.tail({'inst': 'smo_cats', 'action': 'means', 'parameter_num': 22})
        self.assertEqual(args, ['cats', 'smo', '-p', '22', '-i'])
        self.assertIn('parameter 22', what)

    def test_every_analyte_of_ie3(self):
        for pnum in (None, 0, ''):
            with self.subTest(pnum=pnum):
                args, what = self.tail({'inst': 'ie3-smo', 'action': 'means', 'parameter_num': pnum})
                self.assertEqual(args, ['ie3', '-i'])
                self.assertIn('every analyte', what)

    def test_means_is_for_in_situ_only(self):
        for inst in ('m4', 'fe3', 'bld1', None):
            with self.subTest(inst=inst), self.assertRaisesRegex(ValueError, 'in situ'):
                logos_jobs.plan({'inst': inst, 'action': 'means'})

    def test_bad_parameter_num(self):
        for pnum in ('cfc12', -3):
            with self.subTest(pnum=pnum), self.assertRaisesRegex(ValueError, 'parameter_num'):
                logos_jobs.plan({'inst': 'brw', 'action': 'means', 'parameter_num': pnum})

    def test_means_instruments_match_the_service(self):
        # The page ids must be the service's in situ PAGE_INSTS (logos_data_service), with their inst_nums.
        nums = {'ie3-smo': 236, 'brw': 239, 'sum': 240, 'nwr': 241, 'mlo': 242, 'smo_cats': 243, 'spo': 244}
        self.assertEqual({k: v['inst_num'] for k, v in logos_jobs.MEANS_INSTS.items()}, nums)

    def test_other_actions_unchanged(self):
        argv, _ = logos_jobs.plan({'inst': 'fe3', 'action': 'recalc', 'start_time': '2026-10-01'})
        self.assertEqual(argv[2:], ['--steps', 'batch', '-s', '2026-10-01 00:00:00'])
