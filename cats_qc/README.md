# cats_qc

Experimental/exploratory CATS QC code. Not tracked in git (see `../.gitignore`).

## cats_cal_step_qc.py + cats_tagging.py

Detects abrupt GC response shifts (detector/valve glitches) using only the
two calibration ports, whose assigned values are fixed and known -- so a
response change there can only be the instrument, never the atmosphere.
Deliberately insensitive to slow drift, since that's exactly what
normalization corrects for when everything is working well.

### Reproduce the current BRW N2O (q) 1999 tagging

```
cd cats_qc
python3 cats_tagging.py --site brw --algo cal_step --analyte N2O --channel q --start 19990101 --end 19991231
```

This is idempotent: it deletes tag 401 from every mf_num in the requested
scope before reinserting it on whatever's currently flagged, so it's always
safe to rerun (e.g. after retuning a threshold below).

To extend to other sites/years/analytes: change `--site`, `--start`/`--end`
(YYYYMMDD; `--start` defaults to Jan 1 of the current year, `--end` to now),
or `--analyte`/`--channel` (`--channel` is only required when the analyte is
on multiple channels; `--analyte all` runs all 9 CATS analytes: N2O (q),
SF6 (q), CFC12 (f), CFC11 (f), CFC113 (f), H1211 (f), CCl4 (f), CH3CCl3 (f),
CHCl3 (f)).

Add `--dry-run` to preview scope/flagged counts without writing anything.

### How detection works (cats_cal_step_qc.py)

For each cal port (CAL1_PORT, CAL2_PORT) independently, on that port's own
chronological series:

1. **Point-to-point step**: the raw, unsmoothed step in log(response)
   between one injection and the next (`ptp`).
2. **Robust z-score**: each step is judged against that port's own typical
   step size -- median absolute deviation (MAD) computed over a **trailing
   30-day local window**, not the whole requested date range. This matters:
   a global scale over "however much data happens to be queried" makes
   results depend on query span (a bug we hit and fixed -- a full-year query
   and a 1-week query used to give different flags for the same points). The
   load window is padded with 30 days of lookback so results at the
   requested start date don't depend on how far back the caller queried.
3. **Primary threshold** (`--mad-multiplier`, default 3.5): `|ptp_z| >
   3.5` flags a point outright.
4. **Hysteresis** (`--secondary-multiplier`, default 2.0): Canny-style
   double threshold. A point clearing the lower bar (2.0) is kept only if
   it's adjacent, on that same cal port's own timeline, to a point that
   cleared the primary bar -- lets an already-triggered episode's decaying
   tail ride along without lowering the threshold everywhere. A weak point
   next to a normal point is never pulled in, so this can't reopen a chain
   through genuinely quiet data.
5. **Period grouping** (`--max-gap-hours`, default 24): flagged cal
   timestamps within this gap of each other merge into one period.
6. **Neighboring-cal widening**: each period is padded out to the nearest
   cal-port reading (either port) just past each end, so the two cal points
   bordering an episode are swept in too -- not just the ones that
   individually crossed the threshold.
7. **Sweep**: every row on *any* port (2/4/6/8) whose `analysis_datetime`
   falls inside the padded period, inclusive of both endpoints, is flagged.
   This is how the interleaved air1/air2 readings get tagged along with the
   cal points, without ever looking at the air data's own values.
8. **min_cal_points** (`--min-cal-points`, default 1): drop periods backed
   by fewer than this many individually-triggered cal points, to ignore
   isolated single-injection blips if desired.

Worked example: BRW N2O (q), April 1999 -- flags all four ports from
1999-04-09 20:43 to 1999-04-10 21:47 (core rate-outlier episode 21:43 to
19:47, widened by one cal reading on each end).

### Tagging (cats_tagging.py)

- **tag_num 401 is a placeholder.** It's not yet registered in
  `ccgg.tag_dictionary` -- pending assignment by whoever can edit that table
  (`automated=1, reject=0, information=1` agreed). Until then the tag has no
  effect on `rejected` (the view's `rejected` column is a live join against
  `tag_dictionary.reject`, and with no row for 401 the join finds nothing).
  Once assigned, update `ALGORITHMS` in `cats_tagging.py` with the real
  number; already-written 401 rows can be moved with a single
  `UPDATE hats.ng_insitu_mole_fraction_tags SET tag_num = <new> WHERE
  tag_num = 401`.
- `_TAG_LAYOUT` in `logosdata/logos_data.py` also has an entry for 401
  ("Detector cal-response rapid change (auto)") under Automatic Tags, so
  flagged points show up as the info-tag overlay (hollow purple diamond) when
  "Show Info Tags" is checked, and in the point tooltip.
- **Applying a tag doesn't recompute mole fractions.** `rejected` only
  changes once a tag's `reject` flag is 1 (globally, via `tag_dictionary`,
  or per-point by promoting to a reject tag like 141 after review in
  logos_data). Only then does the weekly cal fit (`hats.ng_response`) and
  downstream air mole fractions need recomputing via `cats_batch.py`.
- **Built to grow**: `ALGORITHMS` in `cats_tagging.py` is a small registry
  (`name -> (tag_num, build_fn)`). Each future QC algorithm gets its own
  entry with its own tag_num and a `build()` function shaped like
  `build_cal_step_qc` -- `(batch, pnum, channel, start, end, **kwargs) ->
  DataFrame` with at least an `mf_num` column for the flagged rows.
  `--algo all` runs everything registered.

### Other files in this directory

- `cats_cal_ratio_qc.py` / `cats_peak_qc.py` / `cats_peak_qc_plot.py` /
  `cats_compare.py` / `cats_qc.py` -- earlier/separate QC exploration, not
  covered above.
- `test_cats_cal_step_qc.py` -- unit tests for the pure helper functions
  (`_port_rate_outliers`, `_group_periods`) using synthetic data; run with
  `python3 -m unittest test_cats_cal_step_qc.py`.
