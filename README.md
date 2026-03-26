# itxbin

Utilities for NOAA/GML LOGOS gas chromatography workflows that use GCwerks.

This repository contains command-line and GUI tools used to:
- import chromatograms into GCwerks,
- export GCwerks results,
- load processed data into HATS next-generation database tables,
- run instrument-specific batch processing,
- inspect and edit operational metadata (for example tank use history).

## Instruments and domains

The codebase includes utilities for:
- `m4` (Mass Spectrometer, generation 4),
- `fe3` (Flask Electron Capture GC, generation 3),
- `pr1` (Perseus-1),
- `bld1` (Boulder instrument workflow),
- `ie3` (import workflow support).

Main shared modules:
- `logos_instruments.py`: common instrument/database classes and methods.
- `logos_data.py`: interactive PyQt data processing/inspection UI.
- `logos_tanks.py`: tank analysis UI helpers.
- `logos_timeseries.py`: interactive timeseries plotting helpers.

## Environment assumptions

These scripts are designed for the NOAA/GML internal runtime environment and expect:
- GCwerks binaries at `/hats/gc/gcwerks-3/bin/` (`gcimport`, `gcupdate`, `gccalc`, `gcexport`, `run-index`, `gcwerks`).
- instrument directories under `/hats/gc/<instrument>/`.
- database helper module at `/ccg/src/db/db_utils/db_conn.py` (used by `logos_instruments.py`).
- access to HATS/GMD database schemas used in SQL queries (for example `hats.ng_*`, `gmd.site`, `reftank.*`).

If you run this outside the NOAA environment, many scripts will fail without path and DB adaptation.

## Python dependencies

Common dependencies used across scripts:
- `python3`
- `pandas`
- `numpy`
- `matplotlib`
- `PyQt5`
- `statsmodels`

Install example:

```bash
python3 -m pip install pandas numpy matplotlib PyQt5 statsmodels
```

## Common workflows

### 1) Import chromatograms into GCwerks

Generic importer:

```bash
python3 gcwerks_import.py --help
```

Instrument wrappers:

```bash
python3 fe3_import.py --help
python3 bld1_import.py --help
python3 ie3_import.py --help
```

### 2) Export GCwerks results

Instrument export scripts:

```bash
python3 m4_export.py --help
python3 fe3_export.py --help
python3 pr1_export.py --help
python3 bld1_export.py --help
```

### 3) Load GCwerks results into HATS database

```bash
python3 m4_gcwerks2db.py --help
python3 fe3_gcwerks2db.py --help
python3 pr1_gcwerks2db.py --help
python3 bld1_gcwerks2db.py --help
```

### 4) Batch mole-fraction processing

```bash
python3 m4_batch.py --help
python3 fe3_batch.py --help
```

### CFC-113 and CFC-113a joint processing (M4)

CFC-113 (pnum=32, ion 103) and CFC-113a (pnum=178, ion 117) co-elute on
M4 and contribute to both ion signals. Their mole fractions must be solved
simultaneously using the Montzka (Jan 2026) two-equation deconvolution:

```
RX = MFA·R1 + MFB·R2    (ion 103)
RY = MFA·R3 + MFB·R4    (ion 117)

MFA = (RX - RY·R2/R4) / (R1 - R3·R2/R4)   [CFC-113]
MFB = (RY - MFA·R3) / R4                    [CFC-113a]
```

Molar response factors R1–R4 are stored in `hats.ng_cfc113a`, windowed by
reference tank and date. Source data and the original email are in `cfc113a/`.

**Because the two compounds are coupled, always process them as a pair.**
`m4_batch.py -p 32` and `-p 178` are equivalent — either one recalculates
and upserts both pnum=32 and pnum=178. Running `-p all` also handles the
pair correctly at the end of the loop.

```bash
# Recalculate both CFC-113 and CFC-113a over a date range and save to DB
python3 m4_batch.py -p 32 -s 2501 -e 2503 -i

# Full M4 record
python3 m4_batch.py -p 32 -i
```

After making changes to CFC-113 or CFC-113a in `logos_data.py` (flagging,
smoothing adjustments), run `m4_batch.py -p 32 -i` to propagate the correct
deconvolved values back to the database. The normal ingest pipeline
(`m4_ingest.py`) handles this automatically for new data.

### 5) Ingest pipelines

Automated ingest/process wrappers:

```bash
python3 m4_ingest.py --help
python3 fe3_ingest.py
```

### 6) GUIs

GCwerks launcher UI:

```bash
python3 gcwerks.py
```

Database processing UIs:

```bash
python3 m4db-gui.py --gui
python3 pr1db-gui.py --gui
```

Tank history editor:

```bash
python3 tank_hist_editor.py --help
```

## First-day operator checklist

Use this as a practical daily startup sequence for M4/FE3 operations.

1. Confirm environment access
- Verify you are on a host with `/hats/gc/...` mounted and database access available.
- Confirm GCwerks binaries exist:
```bash
ls /hats/gc/gcwerks-3/bin
```

2. Verify instrument input paths
- Check incoming/raw directories exist and are updating (especially for M4):
```bash
ls -la /hats/gc/m4/chemstation
ls -la /hats/gc/m4/MassHunter/GCMS/1/data
```

3. Run ingest pipeline
- M4 end-to-end ingest/process:
```bash
python3 m4_ingest.py
```
- FE3 ingest/process chain:
```bash
python3 fe3_ingest.py
```

4. Re-run database loaders when needed
- Use `-x` to force fresh export/extract before DB load:
```bash
python3 m4_gcwerks2db.py -x
python3 fe3_gcwerks2db.py
```

5. Run batch mole-fraction updates
- Process all analytes and insert outputs:
```bash
python3 m4_batch.py -p all -i
python3 fe3_batch.py -p all -i
```

6. Spot-check results in GUI tools
- Open data review UI:
```bash
python3 logos_data.py m4
```
- Optional: launch GCwerks directly for manual inspection:
```bash
python3 gcwerks.py m4
python3 gcwerks.py fe3
```

7. Operational sanity checks
- Confirm recent runs appear in expected date window.
- Confirm no unexpected missing analytes/channels in batch output.
- If mole fractions look stale, re-run export + `*_gcwerks2db.py` + batch for the affected instrument/date range.

## Script map (quick reference)

- `gcwerks.py`: launch GCwerks for a selected site/instrument.
- `gcwerks_import.py`: generic ITX/chromatogram import utilities.
- `gcwerks_export.py`: shared GCwerks export logic.
- `m4_samplogs.py`: ingest and sync M4 sample log / analysis metadata.
- `m4_ingest.py`: end-to-end M4 ingest and processing chain.
- `*_gcwerks2db.py`: instrument-specific DB upsert/load flows.
- `*_batch.py`: per-analyte mole fraction recalculation and optional DB insert.
- `logos_data.py`: main PyQt analysis/review application.
- `tank_hist_editor.py`: edit `hats.ng_tank_use_history`.

## Notes for contributors

- Prefer small, focused changes.
- Keep public behavior stable unless intentionally changing workflow.
- Use `--help` on scripts to verify CLI behavior before updating docs.
- Most scripts target operational production paths, so testing often requires NOAA-specific data and DB access.
