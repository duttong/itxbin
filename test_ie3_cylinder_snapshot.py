import unittest

from ie3_cylinder_snapshot import TANK_POPUP_INCLUDE, build_variant


HELPERS = r'''<?php
function LT_FillBadge($f) {
    $title = trim("Filled {$f['date']}. " . preg_replace('/\s+/', ' ', (string)$f['notes']));
    return $title;
}
function LT_TankDetails($f) {
    return ['code' => $f['code'], 'date' => $f['date'],
            'notes' => trim(preg_replace('/\s+/', ' ', (string)$f['notes']))];
}
'''
PAGE = '''<?php
require_once(WWW_ROOT . UTILS_ROOT_RELPATH . '/dbutils.php');
db_connect_ro('hats');
__HELPERS__
?>Fill = reftank.fill code for the gas in the tank (hover for fill date and notes).
<?php
$bs_adjContent = ob_get_clean();
'''


class SnapshotVariantTests(unittest.TestCase):
    def assert_public_variant(self, variant):
        self.assertNotIn("$f['notes']", variant)
        self.assertIn("'notes' => ''", variant)
        self.assertIn('$title = "Filled {$f[\'date\']}";', variant)
        self.assertIn("'site'] === 'SMO'", variant)
        self.assertIn('require __DIR__ . \'/replay_db.php\';', variant)
        self.assertIn('SMO IE3 cylinders', variant)
        self.assertNotIn(TANK_POPUP_INCLUDE, variant)

    def test_shared_helper_is_inlined_and_fill_notes_are_removed(self):
        variant = build_variant(PAGE.replace('__HELPERS__', TANK_POPUP_INCLUDE),
                                'SMO', HELPERS)
        self.assert_public_variant(variant)

    def test_legacy_inline_helpers_remain_supported(self):
        variant = build_variant(PAGE.replace('__HELPERS__', HELPERS[5:]), 'SMO')
        self.assert_public_variant(variant)

    def test_missing_shared_helper_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'shared tank popup source'):
            build_variant(PAGE.replace('__HELPERS__', TANK_POPUP_INCLUDE), 'SMO')

    def test_changed_tooltip_fails_before_publishing(self):
        with self.assertRaisesRegex(RuntimeError, 'fill-notes tooltip'):
            build_variant(PAGE.replace('__HELPERS__', TANK_POPUP_INCLUDE),
                          'SMO', HELPERS.replace('$title = trim(', '$tooltip = trim('))

    def test_changed_popup_notes_fail_before_publishing(self):
        with self.assertRaisesRegex(RuntimeError, 'page layout changed'):
            build_variant(PAGE.replace('__HELPERS__', TANK_POPUP_INCLUDE),
                          'SMO', HELPERS.replace("'notes' => trim", "'fill_notes' => trim"))


if __name__ == '__main__':
    unittest.main()
