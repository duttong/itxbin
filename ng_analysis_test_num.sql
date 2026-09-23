-- Add test_num to ng_analysis so Test runs (run_type_num=10) can be joined to
-- the equipment test that produced them. Filled by fe3_gcwerks2db.py from the
-- test_id in FE3 meta_*.json / testinfo_*.json; 0 for non-test rows.
--
-- Example:
--   SELECT a.*, t.pi, t.type_description, t.test_comment
--     FROM hats.ng_analysis a
--     JOIN (SELECT DISTINCT test_num, pi, type_description, test_comment
--             FROM ccgg_equip.equip_tests_view) t ON t.test_num = a.test_num
--    WHERE a.test_num = 1090;

ALTER TABLE hats.ng_analysis
    ADD COLUMN test_num INT(11) NOT NULL DEFAULT 0
        COMMENT 'fk to ccgg_equip.equip_tests_view.test_num',
    ADD KEY t (test_num);
