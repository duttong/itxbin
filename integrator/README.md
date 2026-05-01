# Replacement for GCwerks GUI Chromatography Integration

## Goals

1. Replace GCwerks for integrating gas chromatography results
2. Integrate thousands of injections extremely fast and reproducibly
3. Use incoming raw chromatograms (`.ITX` files)
4. Support NOAA instruments: IE3, FE3, BLD1. Raw `.ITX` file locations:
   - IE3:  `/hats/gc/smo/26/incoming`
   - FE3:  `/hats/gc/fe3/26/incoming`
   - BLD1: `/hats/gc/bld1/25/incoming`

   FE3 has the most historical data — start with one year to test feasibility.

5. Explore new techniques for peak detection and integration — in particular, instead of relying on a single injection or chromatogram, use information from neighboring chromatograms to build robustness and fault detection into the integration process

## Background

1. Familiarize AI with gas chromatography — search the internet for basic terms
2. Learn about GCwerks: https://gcwerks.com
3. Research new techniques for peak detection and integration

## Work Outline

1. ✅ Survey the `itxbin` codebase
2. ✅ Use `itx_import.py` and chromatogram smoothing techniques
3. ✅ Build a matrix / dataframe of chromatograms for cross-run analysis
4. ✅ Determine a basic set of parameters for peak detection
5. ✅ Implement baseline estimation + scipy peak detection + trapezoidal integration
6. ✅ Parse `peak.list`; RT discovery via auto-detection + elution-order matching
7. ✅ Build a CLI headless interface for core functions
8. ✅ Windowed integration per compound using verified RTs in `fe3_peaks.conf`
9. ✅ HDF5 persistent store (`store.py`) — raw signals, ~2.6 s full-year load, incremental update
10. ✅ RT alignment via cross-correlation; `batch` command integrates full year in ~51 s
11. Compare integrated areas to GCwerks results (mole-fraction level; requires calibration)
12. PyQt GUI for visual review and parameter tuning

## Module Structure

```
integrator/
  store.py       — ChromStore: HDF5-backed persistent store for raw signals
  chrom.py       — Chromatogram class: baseline, peak detection, windowed integration
  run_loader.py  — Load one run directory of ITX files → Chromatogram objects
  matrix.py      — ChromMatrix: 2-D array of runs for cross-injection analysis
  peak_list.py   — Parse GCwerks peak.list / fe3_peaks.conf → Compound objects
  rt_align.py    — RT shift estimation via cross-correlation; reference-peak selection
  pipeline.py    — batch_integrate(): store → smooth → RT-align → integrate → DataFrame
  __main__.py    — CLI entry point (run / matrix / discover / integrate / store / batch)
  fe3_peaks.conf — FE3 compound RT windows (6-column: name ch report rt win ref)
```

## Usage

**Build the annual raw-signal store (run once, then update as new data arrives):**
```bash
python -m integrator store build /hats/gc/fe3/26/incoming /hats/gc/fe3/26/fe3_chroms.h5
python -m integrator store update /hats/gc/fe3/26/fe3_chroms.h5 /hats/gc/fe3/26/incoming
python -m integrator store info   /hats/gc/fe3/26/fe3_chroms.h5
```
2026 store: 4,229 injections · 172 runs · 181 MB · loads in ~2.6 s

**Batch integrate a full year from the store (with RT alignment) → CSV:**
```bash
python -m integrator batch /hats/gc/fe3/26/fe3_chroms.h5 --output fe3_areas_2026.csv
```
2026: 4,229 injections · 12 compounds · RT-aligned per channel · ~51 s · 3,009 fully detected

**Single-run windowed integration using `fe3_peaks.conf`:**
```bash
python -m integrator integrate /hats/gc/fe3/26/incoming/20260105-224906
```

**Free peak detection on one run directory:**
```bash
python -m integrator run /hats/gc/fe3/26/incoming/20260105-195518
```

**Discover nominal retention times from a batch of runs:**
```bash
python -m integrator discover /hats/gc/fe3/26/incoming --limit 10
```

## FE3 Signal Characteristics (2026 data)

- 3 channels, 10 Hz, ~600 s (~6000 points) per injection
- Raw counts: ch0/ch1 baseline ~350k, peaks to ~6M; ch2 baseline ~1M, O₂ peak ~1B
- Port 9 = push port (different sample gas), port 10 = initial push-port run (excluded);
  port sequence and tank labels in `meta_*.json`
- Peak-area RSD across runs: ~3–4% for major peaks (10 injections, ch0/ch1)

## FE3 Retention Times (2026, verified)

All RTs verified against raw chromatograms. See `fe3_peaks.conf` for integration windows.
Channel suffix: `a` = ch0 (ECD A), `c` = ch2 (ECD C). Same compound, different detector.
Normalized response (ratio to port-1 reference tank fit) is used for mole fractions — not raw areas.

| Compound | Channel | Nominal RT | Window ±s |
|----------|---------|-----------|-----------|
| CFC11a   | 0 | 37.6 s | 8 |
| CFC113a  | 0 | 48.5 s | 5 |
| CHCl3    | 0 | 78.4 s | 10 |
| CH3CCl3  | 0 | 95.8 s | 10 |
| CCl4     | 0 | 206.5 s | 10 |
| TCE      | 0 | 267.9 s | 10 |
| N2O      | 1 | 87.5 s | 8 |
| SF6      | 1 | 271.0 s | 12 |
| CFC12    | 2 | 52.1 s | 5 |
| h1211    | 2 | 177.0 s | 15 |
| CFC11c   | 2 | 333.0 s | 20 |
| CFC113c  | 2 | 485.0 s | 20 |

Note: ch2 also has a large O₂ peak at ~63s and a CFC12 shoulder at ~52s — both
ignored since only peak-list compounds are integrated.

## Open Items

- RT alignment across injections (cross-correlation on the largest stable peak per channel)
- Compare integrated areas to GCwerks results (mole-fraction level; requires calibration)
- PyQt GUI for visual review and parameter tuning
- Verify CHCl3, CH3CCl3 RTs more carefully (high σ in auto-discovery)
