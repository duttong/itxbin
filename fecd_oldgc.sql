-- Monthly means from the original HATS flask GC-ECD ("oldGC", 1977-1995).
-- Only the published monthly means survive, in
-- /aftp/hats/{cfcs/cfc11,cfcs/cfc12,n2o}/flasks/OldGC/monthly/{SITE}_{gas}_MM.dat
-- (ALT, BRW, CGO, MLO, NWR, SMO, SPO).  Loaded by fecd_oldgc_import.py; read
-- by the combined data sets (logosdata/combined_data.py).
--
--   inst_num   251 (ccgg.inst_description: the oldGC flask ECD, aka FE1)
--   month      first day of the month
--   mean, sd   monthly mean and its standard deviation (DOUBLE, never NUMERIC)
--   n          number of samples in the month
--   scale      calibration scale named in the source file header

CREATE TABLE hats.fecd_oldgc (
    num            INT(11)     NOT NULL AUTO_INCREMENT,
    site_num       INT(11)     NOT NULL,
    inst_num       INT(11)     NOT NULL DEFAULT 251,
    parameter_num  INT(11)     NOT NULL,
    month          DATE        NOT NULL,
    mean           DOUBLE      NOT NULL,
    sd             DOUBLE      NULL,
    n              INT(11)     NOT NULL,
    scale          VARCHAR(32) NULL,
    mod_date       TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (num),
    UNIQUE KEY u_site_param_month (site_num, parameter_num, month),
    KEY i_param_month (parameter_num, month)
);
