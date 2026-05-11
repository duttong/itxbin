-- Create mole fraction method lookup table and add mf_method_num to ng_insitu_mole_fractions.
--
-- Methods:
--   1  ref    Reference tank (port 5 × coef0) — current default
--   2  cal12  2-point calibration (ports 1 & 9)
--   3  cal1   Single-point calibration (port 9)
--   4  cal2   Single-point calibration (port 1)

CREATE TABLE hats.ng_insitu_mf_methods (
    num  INT(11)     NOT NULL AUTO_INCREMENT,
    abbr VARCHAR(10) NOT NULL,
    name VARCHAR(45) NOT NULL,
    PRIMARY KEY (num)
);

INSERT INTO hats.ng_insitu_mf_methods (abbr, name) VALUES
    ('ref',   'Reference tank'),
    ('cal12', '2-point calibration'),
    ('cal1',  'Single-point calibration (port 9)'),
    ('cal2',  'Single-point calibration (port 1)');

-- Existing rows get the default value of 1 (ref).
ALTER TABLE hats.ng_insitu_mole_fractions
    ADD COLUMN mf_method_num INT(5) NULL DEFAULT 1
    AFTER detrend_method_num;
