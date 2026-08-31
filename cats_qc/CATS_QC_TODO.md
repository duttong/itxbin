# CATS QC TODO

Pending follow-ups for the cats_qc work. See `README.md` for how the
`cal_step`, `baseline`, and `cal_method_qc` algorithms and `cats_tagging.py`
work.

## Validate `cats_cal_method_qc.py` thresholds against known BRW discontinuities

`--trend-window-days=365`, `--gap-days=30`, `--min-trend-points=12`,
`--jump-z-threshold=4.0`, `--resolve-z-threshold` (defaults to
`jump-z-threshold`) are reasoned starting points, not validated ones -- same
caveat this file already documents for `cal_step`'s `mad_multiplier=3.5` and
`baseline`'s `ratio_threshold=0.05`, both tuned against real data before
being trusted. Run against BRW N2O/SF6 (channel `q`) first and check:

- Does it actually fire near the discontinuities already visible by eye in
  the full-record SF6 plot (roughly 2001, 2004-2005, 2007, 2009-2010, 2012,
  2017, 2019-2020, 2022), and not in the smooth stretches between them?
- For `UNRESOLVED` episodes, do the reported cal-tank serials/dates line up
  with anything suspicious in `hats.scale_assignments`?
- Does `--jump-z-threshold=4.0` need to be different pre- vs. post- the
  2025 switch to `cal12` (sparser/noisier cal-tank data in earlier eras
  could inflate or deflate the self-calibrated scale differently)?

Only validated for BRW N2O/SF6 (q) -- do not extend to other analytes/sites
without re-checking the same way, same caution as the two items below.

## Scan the chromatogram archive with `baseline`

Only a small validation window is in `hats.ng_chromatogram_qc` so far. The
full archive is ~29 years x 6 sites x 4 channels; at ~40 min per
site-channel (threaded) that is roughly a day of wall-clock in total.

Agreed approach: **one site/channel at a time**, reviewing results before
committing more compute -- the detector has only been validated against 2026
data, and older eras may differ in noise character or run length. Check
coverage with the query in `README.md` before starting a new range.

Worth confirming on the first pre-2020 scan:

- Do older chromatograms share the ~30 min / 4 Hz / ~7150-sample shape? (2015
  and 2011-2012 spot checks did.)
- Does GCwerks' dated peakid file coverage reach back far enough, and are
  the windows still sane, for the earliest eras? `find_gcwerks_peakid_file`
  falls back to the earliest available file if the target predates all of
  them, so masking degrades gracefully rather than failing outright, but a
  badly-mismatched fallback window would silently under- or over-mask peaks.
  `n_masked_samples` in `hats.ng_chromatogram_qc` is a cheap sanity check --
  compare it across eras for surprises.
- Does `--ratio-threshold 0.05` still separate cleanly, or does the
  quiet-run spread differ enough by era to need a per-era value? Unflagged
  rows are stored, so a different threshold can be re-derived with SQL
  rather than a rescan.

## Possible: reusable CATS chromatogram store

The baseline scan holds its chromatogram matrix in memory and discards it.
If repeated whole-trace analysis becomes a need (peak re-integration, RT
alignment, a second detector), the `integrator/` package already has the
right pattern -- `store.py`'s HDF5 `ChromStore`, raw int32 full traces, so
analysis parameters can change without a rebuild.

Extending it to CATS means a new build/update path for the
`chromatograms/channelN/YYMMDD.HHMM.ext` layout using
`read_gcwerks_chromatogram` (FE3's uses `incoming/RUNDIR/*.itx` via
`itx_import`). Rough size: ~1.2 GB per site-year, ~200 GB for all sites and
years gzipped. Not worth doing for the baseline scan alone -- threaded
re-reads already make a rescan ~40 min/channel.
