# CLAUDE.md — itxbin project context

## Environment

- Production conda env: `prod6`
- Run scripts as: `python3 <script>.py` or via the installed entry points
- Database: HATS next-generation MySQL database, accessed via
  `/ccg/src/db/db_utils/db_conn.py` → `db_conn.HATS_ng()`
- The helper wrapper `hats_db.py` provides `HATSdb` for quick one-off queries
- **Do not use `NUMERIC` for float columns** — the DB driver maps it to
  `decimal(10,0)` (scale=0), silently truncating decimal values. Use
  `DOUBLE PRECISION` or `FLOAT8` instead.

## Key instrument/parameter numbers

| Instrument | inst_num | inst_id |
|---|---|---|
| M4 (mass spec) | 192 | `m4` |
| FE3 (flask ECD) | 193 | `fe3` |
| BLD1 | 220 | `bld1` |
| PR1 (Perseus 1) | 58 | `pr1` |
| PR2 (Perseus 2) | 238 | `pr2` |
| Perseus combined tanks view | 58/238 | `prs` |
| IE3 | 236 | `ie3` |

| Compound | parameter_num |
|---|---|
| CFC-113 | 32 |
| CFC-113a | 178 |

## Key database tables

- `hats.ng_data_processing_view` — main analysis view (areas, normalized
  responses, mole fractions, run metadata)
- `hats.ng_insitu_data_view` — IE3 analysis/mole-fraction view with
  `rejected`, `rej_flags`, and `background`; used by
  `IE3_Instrument.load_data()`
- `hats.ng_mole_fractions` — computed mole fraction output table (upserted by
  batch scripts and logos_data)
- `hats.ng_mole_fraction_tags` — tag table for M4/FE3/BLD1 mole fractions
- `hats.ng_insitu_mole_fraction_tags` — tag table for IE3 mole fractions
- `hats.calibrations` — per-run aggregated tank calibration values (avg
  mole fraction, stddev, n, run_number, scale_num) keyed on
  `(serial_number, date, time, species, inst, parameter_num)`; kept current
  by `upsert_calibrations()` (see below); read by the Tanks tab in logos_data.
  Tank plots filter out sentinel values with `mixratio <= -99`.
- `hats.ng_cfc113a` — M4 molar response factors (R1–R4) for CFC-113/113a
  deconvolution, windowed by reference tank and date
- `hats.scale_assignments` — calibration scale coefficients (coef0, coef1)
  used by `calc_mole_fraction_scalevalues` for most M4 analytes
- `hats.ng_insitu_analysis` — IE3 GC run metadata (`num, run_time,
  analysis_time, site_num, inst_num, port`); `run_time` groups all port
  injections for one GC run, `analysis_time` is per-injection
- `hats.ng_insitu_mole_fractions` — IE3 computed mole fractions joined to
  `ng_insitu_analysis` via `analysis_num`; air ports are 3 and 7;
  `sample_loop_temp/pressure/flow` upserted by `ie3_eng2db.py`
- `hats.ng_preferred_channel` — preferred channel per `(inst_num,
  parameter_num, start_date)`; used by `return_preferred_channel()` on FE3
  and IE3 instruments (e.g. IE3 CFC12→`b`, CFC11→`c`)

## CFC-113 / CFC-113a coupling (M4)

These two compounds co-elute on M4 and must always be processed as a pair
using the Montzka two-equation deconvolution. Key points:

- `M4_Instrument.calc_mole_fraction()` automatically routes pnum 32 and 178
  through the deconvolution — all logos_data call sites are covered
- `m4_batch.py -p 32` and `-p 178` are equivalent: both recalculate and
  upsert pnum=32 and pnum=178 together
- After any logos_data edits to CFC-113 or CFC-113a, run
  `m4_batch.py -p 32 -i` to write correct deconvolved values back to DB
- Reference data and the Montzka email are in `cfc113a/`

## Class hierarchy

```
LOGOS_Instruments
  └── HATS_DB_Functions        # DB connection, shared methods
        ├── M4_Instrument      # inst_num=192, scale-value mole fractions
        ├── FE3_Instrument     # inst_num=193, polynomial response inversion
        ├── IE3_Instrument     # inst_num=236
        ├── Perseus_Instrument # inst_id=prs, PR1/PR2 tank facade
        └── BLD1_Instrument    # inst_num=220
```

`Perseus_Instrument` is currently for tanks/calibrations. It uses PR1
analytes from `hats.analyte_list` and queries calibration records for both
`PR1` and `PR2`.

`M4_Instrument` is also the tank/calibration facade for the M-system. The
standalone `logos_tanks m4` entry queries `hats.calibrations` for `M1`, `m1`,
`M3`, `m3`, and `M4`, while keeping M4 as the user-facing entry point and
analyte list.

## Tagging / rejection model

- Current NG rejection state comes from tag tables, not the legacy
  three-character `flag` columns.
- `rejected` is the dataframe column used by `logos_data`, `logos_timeseries`,
  and `logos_tanks` filtering.
- Manual reject tags default to `tag_num=141`.
- GCwerks reject tags use `tag_num=324`; GCwerks DB loaders synchronize this
  tag for the rows in the current flagged export by deleting stale 324 tags and
  reinserting current ones.
- M4 first-reference tags use `tag_num=316`.

## IE3 in-situ timeseries (logos_timeseries.py)

- `TimeseriesWidget.query_insitu_data()` queries `ng_insitu_analysis` ⋈
  `ng_insitu_mole_fractions` ⋈ `gmd.site` for unflagged air-port data
  (ports 3 & 7); only runs when `instrument.inst_num == 236`
- Channel is extracted from the analyte display name (e.g. `"CFC12 (b)"` →
  `channel='b'`) and applied as a SQL filter — avoids mixing channels that
  share a `parameter_num`
- Air ports are split into **Air1** (port 3) and **Air2** (port 7) on the
  plot, both as filled circles at different brightness levels of the site color
- `TimeseriesWidget` defaults to the site passed via `--site` (stored as
  `instrument.site`); other instruments default to BRW/MLO/SMO/SPO
- Right-click a point → navigates main window to that GC `run_time`

## logos_data.py (GUI)

- Requires a display (`$DISPLAY` must be set); connect with `ssh -Y`
- Single-parameter context: `self.run` holds data for one pnum at a time
- Save action calls `upsert_mole_fractions(self.run)` — mole fractions must
  be correctly computed before saving
- For M4 CFC-113/113a, the GUI recalc paths load the partner pnum from DB
  internally; authoritative recalc should be done with `m4_batch.py -p 32 -i`

## logos_tanks.py / logos_tanks

- Standalone launcher: `logos_tanks [instrument]`
- Valid direct-launch instruments: `m4`, `fe3`, `bld1`, `prs`
- If omitted, the default instrument is read from `~/.logos-tanks.conf` as
  JSON key `default_inst`; if missing, the launcher prompts once and saves it.
- The same config file stores saved tank sets under `sets_by_analyte`.
