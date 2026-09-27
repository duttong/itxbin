-- LOGOS combined data sets: per-site monthly means blended from several
-- measurement programs, plus semi-hemispheric, hemispheric and global means.
-- Written by combined_data_batch.py from logosdata/combined_data_config.yaml;
-- each run replaces all rows of the gases it builds.
--
--   gas            config key (CFC11, CFC12, CFC113, CCl4, SF6, N2O)
--   location       site code (brw, mlo_pfp, ...) or Global, NH, SH, HN, LN, LS, HS
--   month          first day of the month
--   mean, sd       mole fraction and its 1-sigma uncertainty (DOUBLE, never NUMERIC)
--   n              samples behind the value (0 for a month filled by interpolation)
--   programs       one character per program in config order, '1' where the
--                  program contributed (oldGC, RITS, fECD, CATS, IE3, CCGG, MSD)

CREATE TABLE hats.ng_logos_combined_data (
    num            INT(11)     NOT NULL AUTO_INCREMENT,
    gas            VARCHAR(16) NOT NULL,
    parameter_num  INT(11)     NOT NULL,
    location       VARCHAR(16) NOT NULL,
    month          DATE        NOT NULL,
    mean           DOUBLE      NULL,
    sd             DOUBLE      NULL,
    n              INT(11)     NOT NULL DEFAULT 0,
    programs       VARCHAR(16) NULL,
    mod_date       TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (num),
    UNIQUE KEY u_gas_location_month (gas, location, month),
    KEY i_parameter (parameter_num, month)
);
