# integrator — CLAUDE.md

## What this is

Python replacement for GCwerks chromatogram integration.  Reads raw `.ITX`
signal files, stores them in HDF5, and batch-integrates thousands of injections
with RT alignment.  Currently built and validated against FE3 (2026 data).

## Data paths

| Item | Path |
|------|------|
| FE3 2026 store | `/nfs/hats/gc/fe3/26/fe3_chroms.h5` |
| FE3 incoming ITX | `/nfs/hats/gc/fe3/26/incoming/` |
| Peak config | `integrator/fe3_peaks.conf` |

Store stats (as of May 2026): 4,229 injections · 3 channels · ~181 MB · loads in ~2.6 s.

## FE3 signal basics

- 3 channels, 10 Hz, ~600 s per injection (~6000 samples)
- Channels: ch0 = ECD A (a-suffix), ch1 = ECD B, ch2 = ECD C (c-suffix)
- Port 9 and port 10 are push-port runs (different sample gas, no analyte signal)
  — port 10 is excluded by default; port 9 causes ~430 alignment failures on ch1
  but the carry-forward fill handles it
- Compound RTs and windows are in `fe3_peaks.conf` — **always edit that file for
  RT or window changes**, not any hardcoded values

## Integration methods

**meth=1 (fixed window)** — `integrate_window()`

Places integration boundaries at `[nominal_rt - win_lo, nominal_rt + win_hi]`.
Apex is searched within `±min(win_lo, win_hi)/2` of nominal RT to prevent a
large adjacent compound from stealing the argmax (critical for CFC11a/CFC113a
which are ~17s apart and differ by 21×).  Baseline is the rolling-minimum
estimate (NOT a local linear baseline between the endpoints — local linear
breaks when window edges fall on adjacent peak tails).  All 12 FE3 compounds
currently use meth=1.

**meth=2 (tangent skim)** — `integrate_tangent_skim()`

Iteratively finds peak boundaries via first-derivative zero crossings, subtracts
the linear baseline between them, repeats (default n_iter=2).  Apex is
restricted to `±min(win_lo, win_hi)/2` in every iteration and in the final
apex computation.

**Known limitation of meth=2:** does not work reliably for closely-spaced peaks
where a large neighbor's tail extends into the apex search window.  Specifically:
CFC11a (ch0) fails because CFC113a is 17s later and 21× taller — its rising
signal dominates the right half of the apex window after baseline subtraction.
Leave CFC11a (and CFC113a) as meth=1.  Meth=2 gives correct results for
well-separated compounds.

## RT alignment

Cross-correlation against a median template built from the reference compound
(`ref=1` in fe3_peaks.conf).  One ref compound per channel:
- ch0: CFC11a
- ch1: N2O
- ch2: CFC12

Shifts beyond `max_shift_sec=5.0` are treated as failures (reference absent,
cross-correlation latched onto noise).  Failures are filled with the most
recent valid shift from up to `shift_lookback=3` prior injections.  Injections
with no valid predecessor within 3 back default to 0.0 (nominal RT).

This carry-forward is important for integrating near-zero peaks accurately:
those injections (push ports, zero-air) still get a sensible RT correction from
the surrounding valid injections.

## Batch performance (2026, FE3)

~51 s total for 4,229 injections × 12 compounds.  Key optimizations:
- Batch SG smoothing: `savgol_filter(sigs_ch, ..., axis=1)` over [N, MAX_SAMPLES]
- Batch baseline: `minimum_filter1d` + `savgol_filter` over same array
- Pre-inject baseline into `Chromatogram._baseline` (skips per-object recompute)
- `_noise_std` cached per Chromatogram (one compute per channel per injection,
  shared across all compounds on that channel)

## Baseline strategy

Rolling minimum (`minimum_filter1d`, 60 s window) → SG-smoothed.  This tracks
slow detector drift and is robust against narrow peaks because the 60 s window
spans many peak widths.  The SG smooth removes the staircase artifact from the
raw running minimum.

**Do not switch to local linear baseline for meth=1.**  Local linear breaks
whenever a window edge falls on an adjacent peak tail (e.g. CFC12 right edge
at 57.1 s clips O₂ rising at ~63 s; CH3CCl3 left edge clips CHCl3 tail).

## CLI quick reference

```bash
# Build / update / inspect the store
python -m integrator store build  /nfs/hats/gc/fe3/26/incoming /nfs/hats/gc/fe3/26/fe3_chroms.h5
python -m integrator store update /nfs/hats/gc/fe3/26/fe3_chroms.h5 /nfs/hats/gc/fe3/26/incoming
python -m integrator store info   /nfs/hats/gc/fe3/26/fe3_chroms.h5

# Full-year batch integration → CSV
python -m integrator batch /nfs/hats/gc/fe3/26/fe3_chroms.h5 --output fe3_areas_2026.csv

# Exclude push ports
python -m integrator batch /nfs/hats/gc/fe3/26/fe3_chroms.h5 --exclude-ports 9,10

# Box smoothing instead of Savitzky-Golay
python -m integrator batch /nfs/hats/gc/fe3/26/fe3_chroms.h5 --smooth-method box

# Single-run integration (uses fe3_peaks.conf)
python -m integrator integrate /nfs/hats/gc/fe3/26/incoming/20260424-XXXXXX
```

## Next steps

- Compare batch areas/heights to GCwerks output at mole-fraction level
- Extend to IE3 and BLD1 instruments
- PyQt GUI for visual review and parameter tuning
