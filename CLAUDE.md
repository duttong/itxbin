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
| M2 (predecessor mass spec) | 47 | `m2` |
| FE3 (flask ECD) | 193 | `fe3` |
| BLD1 | 220 | `bld1` |
| PR1 (Perseus 1) | 58 | `pr1` |
| PR2 (Perseus 2) | 238 | `pr2` |
| Perseus combined tanks view | 58/238 | `prs` |
| IE3 | 236 | `ie3` |
| CATS-BRW | 239 | `cats` (site=brw) |
| CATS-SUM | 240 | `cats` (site=sum) |
| CATS-NWR | 241 | `cats` (site=nwr) |
| CATS-MLO | 242 | `cats` (site=mlo) |
| CATS-SMO | 243 | `cats` (site=smo) |
| CATS-SPO | 244 | `cats` (site=spo) |

| Compound | parameter_num |
|---|---|
| CFC-113 | 32 |
| CFC-113a | 178 |

## Key database tables

- `hats.ng_data_processing_view` — main analysis view (areas, normalized
  responses, mole fractions, run metadata)
- `hats.ng_insitu_data_view` — IE3 analysis/mole-fraction view with
  `rejected`, `rej_flags`, `background`, and `mf_num`
  (`ng_insitu_mole_fractions.num`); used by `IE3_Instrument.load_data()`
- `hats.ng_mole_fractions` — computed mole fraction output table (upserted by
  batch scripts and logos_data)
- `hats.ng_mole_fraction_tags` — tag table for M4/FE3/BLD1 mole fractions
- `hats.ng_insitu_mole_fraction_tags` — tag table for IE3 mole fractions
- `hats.calibrations` — per-run aggregated tank calibration values (avg
  mole fraction, stddev, n, run_number, scale_num) keyed on
  `(serial_number, date, time, species, inst, parameter_num)`; kept current
  by `upsert_calibrations()` (see below); read by the Tanks tab in logos_data.
  Tank plots filter out sentinel values with `mixratio <= -99` and `flag != '.'`.
  `flag` column: `'.'` = all injections in the group were unrejected; `'M'`
  reserved for partially-rejected groups (not yet written by `upsert_calibrations()`).
  Groups where **all** injections are rejected are **deleted** from this table
  by `upsert_calibrations()` rather than left stale — re-run the batch script
  after rejections to keep this table current.
  `hats.calibrations_view` re-derives the same data live from `ng_data_view` /
  `prs_data_view` (always reflects current rejection state, but ~100× slower);
  note: PR2 is currently absent from `calibrations_view` due to a `level`
  filter bug in the view (tertiary standards excluded); awaiting DB fix.
- `hats.ng_cfc113a` — M4 molar response factors (R1–R4) for CFC-113/113a
  deconvolution, windowed by reference tank and date
- `hats.scale_assignments` — calibration scale coefficients (coef0, coef1)
  used by `calc_mole_fraction_scalevalues` for most M4 analytes
- `hats.ng_insitu_analysis` — IE3 GC run metadata (`num, run_time,
  analysis_time, site_num, inst_num, port`); `run_time` groups all port
  injections for one GC run, `analysis_time` is per-injection
- `hats.ng_insitu_mole_fractions` — IE3/CATS computed mole fractions joined to
  `ng_insitu_analysis` via `analysis_num`; IE3 air ports are 3 and 7;
  CATS air ports are 4 and 8; `sample_loop_temp/pressure/flow` upserted by
  `ie3_eng2db.py` (IE3 only)
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

## M2 program (logos_compare)

M2 (inst_num=47, inst_id `m2`) is the predecessor mass-spec program, stored
in the `prs` tables alongside PR1/PR2 (`hats.prs_data_view`, `hats.analysis` ⋈
`hats.mole_fractions`). It is wired into `logos_compare` as a selectable
program. Key points:

- Data spans **1994–2015** by sample date (analyzed 2007–2015; old archived
  flasks were run later). `PROGRAM_YEAR_LIMITS["m2"] = (1994, 2015)` greys the
  checkbox out when the selected range doesn't overlap.
- Air samples use **`sample_type='Flask'`** (PFP pseudo-sites use `'PFP'`),
  unlike PR1 which uses `'HATS'`. The data path is
  `TimeseriesWidget.query_m2_monthly_mean_data()` — a near-copy of the PR1
  method with `inst_num=47` and the Flask/PFP filters.
- M2 shares most `parameter_num`s with PR1/M4, so those compare on the same
  panels automatically. **Exception (resolved by a one-time DB edit):** M2's
  CFC-11 was originally pnum **28** (`CFC11_A`) while M4/PR1 use 29 and FE3
  uses 114. `hats.mole_fractions.parameter_num` was updated **28→29 scoped to
  inst_num=47 only** (inst 46/54 still use 28) so M2 CFC-11 unifies on pnum 29.
  This relabels the quantitation (`_A`→`_B`) without re-quantifying values, so
  watch for a CFC-11 offset vs other programs in the diff panel.
- `hats.analyte_list` for inst_num=47 holds one row per data-param (CFC-11 as
  pnum 29), mostly copied from PR1's rows; this is what gates M2's selectable
  analytes in `logos_compare`.

## Class hierarchy

```
LOGOS_Instruments
  └── HATS_DB_Functions        # DB connection, shared methods
        ├── M4_Instrument      # inst_num=192, scale-value mole fractions
        ├── FE3_Instrument     # inst_num=193, polynomial response inversion
        ├── IE3_Instrument     # inst_num=236
        │     └── CATS_Instrument  # inst_num=239-244 (per site), scale-simple mole fractions
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
- M4 first-reference tags use `tag_num=316`. `flag_first_reference_run()` in
  `m4_gcwerks2db.py` applies 316 and simultaneously sets `qc_status='F'`;
  reapplication only targets `qc_status='P'` rows, so manually removing 316
  in logos_data is safe — it will not be reapplied on the next batch run.

## IE3/CATS in-situ timeseries (logos_timeseries.py)

- `TimeseriesWidget.query_insitu_data()` queries `ng_insitu_analysis` ⋈
  `ng_insitu_mole_fractions` ⋈ `gmd.site` for unflagged air-port data;
  runs for IE3 (inst_num=236) and all CATS inst_nums (239-244)
- Air ports: IE3 uses ports 3 & 7; CATS uses ports 4 & 8 (from `AIR_PORTS`
  class attribute on each instrument)
- Channel is extracted from the analyte display name (e.g. `"CFC12 (b)"` →
  `channel='b'`) and applied as a SQL filter — avoids mixing channels that
  share a `parameter_num`
- Air ports are split into **Air1** and **Air2** on the plot, both as filled
  circles at different brightness levels of the site color
- `TimeseriesWidget` defaults to the site passed via `--site` (stored as
  `instrument.site`); other instruments default to BRW/MLO/SMO/SPO
- Right-click a point → navigates main window to that GC `run_time`

## CATS ingest pipeline

- `cats_export.py` (in `~/bin/`) — exports GCwerks data to per-molecule CSVs
  in `/hats/gc/cats_results/` (e.g. `brw_N2O.csv`, `brw_F12.csv`)
- `cats_gcwerks2db.py` — merges per-molecule CSVs and loads into
  `ng_insitu_analysis` + `ng_insitu_mole_fractions`; handles CATS channel
  suffixes (q, f, cc); syncs GCwerks flags (F, *, B) as tag 324
- `cats_aftp2db.py` — imports published mole fractions from
  `/aftp/hats/.../insituGCs/CATS/hourly/{site}_{compound}_All.dat` into
  `ng_insitu_mole_fractions.mole_fraction`; requires gcwerks2db to run first
- `cats_ingest.py` — orchestrator: export → gcwerks2db → aftp2db
- Port layout (all sites): port 2=cal1 (Std), port 4=air1, port 6=cal2 (Ref),
  port 8=air2; cal2 is near-ambient and used as the normalization reference
- Mole fractions: `mf = normalized_resp × coef0` from `hats.scale_assignments`
  keyed on (cal2 serial number, parameter_num)

## logos_data.py (GUI)

- Requires a display (`$DISPLAY` must be set); connect with `ssh -Y`
- Single-parameter context: `self.run` holds data for one pnum at a time
- Save action calls `upsert_mole_fractions(self.run)` — mole fractions must
  be correctly computed before saving
- For M4 CFC-113/113a, the GUI recalc paths load the partner pnum from DB
  internally; authoritative recalc should be done with `m4_batch.py -p 32 -i`

## sample_sheets_account.py

Audits archived sample sheets against `hats.Status_MetData`.

- Archive root: `/hats/gc/sample_sheets/archived/{site}/`
- Filename pattern matched: `logos_{site}_{pairid}_*`
- Takes a single site argument, or `--all` to audit every archived site at
  once (the two are mutually exclusive; one is required). `--all` joins DB
  records to archive dirs per `Station`.
- Default output: CSV (`pairid, sample_datetime, site`) of DB records with no
  archived PDF; sorted by pairid unless `--sort-datetime` is given
- `--orphans`: instead report archived PDFs whose PairID has no DB record
  (sample_datetime will be empty since these are absent from the DB)
- Summary line (record counts) is written to stderr so CSV redirects are clean

## logos_tanks.py / logos_tanks

- Standalone launcher: `logos_tanks [instrument]`
- Valid direct-launch instruments: `m4`, `fe3`, `bld1`, `prs`
- If omitted, the default instrument is read from `~/.logos-tanks.conf` as
  JSON key `default_inst`; if missing, the launcher prompts once and saves it.
- The same config file stores saved tank sets under `sets_by_analyte`.
