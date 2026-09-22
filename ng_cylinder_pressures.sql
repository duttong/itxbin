-- Field-instrument cylinder pressure log (cal tanks, carrier gas, dopant).
--
-- Loaded from the IE3 operator log (logs/cylinder_pressures.csv, Cylinders tab
-- in IE3_acquisition/display.py) by ie3_cylinders2db.py, run from
-- ie3_ingest.py.  Read by omi:/var/www/html/hats/logos_field_cylinders.php.
--
-- `cylinder` is the cal tank serial number (spelled as in reftank.fill /
-- ng_port_info) or the gas name ('N2', 'CO2') for utility cylinders, which
-- operators don't log by serial.  A bad reading is flagged rejected=1 rather
-- than deleted, so the insert-only loader never brings it back.
--
-- Cylinder types:
--   1  cal      Calibration tank  (pressure-based change projection)
--   2  carrier  Carrier gas       (pressure-based change projection)
--   3  dopant   ECD dopant        (liquid CO2: psi doesn't predict empty)

CREATE TABLE hats.ng_cylinder_types (
    num            INT(11)     NOT NULL AUTO_INCREMENT,
    abbr           VARCHAR(10) NOT NULL,
    description    VARCHAR(45) NOT NULL,
    project_change TINYINT(1)  NOT NULL DEFAULT 1,
    PRIMARY KEY (num)
);

INSERT INTO hats.ng_cylinder_types (num, abbr, description, project_change) VALUES
    (1, 'cal',     'Calibration tank', 1),
    (2, 'carrier', 'Carrier gas',      1),
    (3, 'dopant',  'ECD dopant',       0);

CREATE TABLE hats.ng_cylinder_pressures (
    num               INT(11)          NOT NULL AUTO_INCREMENT,
    inst_num          INT(11)          NOT NULL,
    site_num          INT(11)          NOT NULL,
    cylinder_type_num INT(11)          NOT NULL COMMENT 'FK to ng_cylinder_types',
    cylinder          VARCHAR(45)      NOT NULL COMMENT 'cal tank serial, or gas name for utility cylinders',
    serial_number     VARCHAR(45)      NULL     COMMENT 'real serial for utility cylinders, if recorded',
    reading_datetime  DATETIME         NOT NULL COMMENT 'UTC',
    pressure          DOUBLE PRECISION NOT NULL COMMENT 'psi',
    rejected          TINYINT(1)       NOT NULL DEFAULT 0,
    comment           VARCHAR(200)     NULL,
    PRIMARY KEY (num),
    UNIQUE KEY inst_cyl_time (inst_num, cylinder, reading_datetime),
    KEY site_time (site_num, reading_datetime)
);
