# cats_qc

CATS QC detection algorithms and the tagging tool that applies their results
to `hats.ng_insitu_mole_fraction_tags`.

Only the active pipeline is tracked in git (`cats_tagging.py`,
`cats_cal_step_qc.py`, `cats_baseline_qc.py`, `cats_cal_window_qc.py`,
`cats_cal_method_qc.py`, `test_cats_cal_step_qc.py`,
`test_cats_cal_window_qc.py`, `test_cats_cal_method_qc.py`, and these docs) --
the rest of this directory is earlier exploratory work and is gitignored.
See the `cats_qc/` allowlist in `../.gitignore`.

Three detectors are registered today (apply a real reject tag via
`cats_tagging.py`), plus one standalone recommendation tool that writes
nothing:

| Algorithm | Tag | What it finds | Scope |
|---|---|---|---|
| `cal_step` | 328 (reject) | Abrupt cal-port response shifts | per analyte + channel |
| `baseline` | 329 (reject) | Abnormal chromatogram shape | per **physical channel** |
| `cal_window` | 286 (reject) | Mole fraction outside local calibration noise | per analyte + channel |
| `cal_method_qc` (not a `cats_tagging.py` algorithm) | n/a -- recommendation only | Non-atmospheric discontinuities caused by the wrong per-period calibration method | per analyte + channel, BRW N2O/SF6 (q) only for v1 |

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

This is idempotent: it deletes tag 328 from every mf_num in the requested
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

- **Both tag_num 328 ("Detector cal-response rapid change") and tag_num 329
  ("Abnormal chromatogram") are real reject tags**, registered in
  `ccgg.tag_dictionary` with `reject=1, automated=1`. Points either tags
  become `rejected` immediately, because the view's `rejected` column is a
  live join against `tag_dictionary.reject`. Both started life as
  unregistered placeholders (401, then 402) before their real numbers were
  assigned; each transition was a single
  `UPDATE hats.ng_insitu_mole_fraction_tags SET tag_num = <new> WHERE
  tag_num = <placeholder>` plus updating `ALGORITHMS`, and the placeholder
  rows were fully migrated (none left at 401 or 402).
- `_TAG_LAYOUT` in `logosdata/logos_data.py` carries both under **Automated
  Tags** as reject tags (letters `S` and `X`). `AUTO_TAG_NUMS` and the
  tag-dropdown sort order both include 328 and 329 so they render correctly
  as automated, not manual, tags.
- **Applying a tag doesn't recompute mole fractions.** Once a tag with
  `reject=1` lands (or a point is promoted to a reject tag like 141 after
  review in logos_data), the weekly cal fit (`hats.ng_response`) and
  downstream air mole fractions need recomputing via `cats_batch.py`:
  `cats_batch.py --site <site> -p <pnum> -c <channel> -i --fits -s <start>`.
- **Built to grow**: `ALGORITHMS` in `cats_tagging.py` is a small registry
  (`name -> (tag_num, build_fn)`). Each future QC algorithm gets its own
  entry with its own tag_num and a `build()` function shaped like
  `build_cal_step_qc` -- `(batch, pnum, channel, start, end, **kwargs) ->
  DataFrame` with at least an `mf_num` column for the flagged rows.
  `--algo all` runs everything registered.

## cats_baseline_qc.py -- "Abnormal chromatogram" (tag 329)

Finds chromatograms whose **overall shape** departs from what its own
neighbours were doing at the time: contamination carryover, detector upsets,
ECD instability, baseline sag/ramp during elution -- anything that disturbs
the trace outside the analyte peaks themselves, wherever in the run it
happens.

Unlike `cal_step`, this reads **raw chromatogram files**, never DB response
values, and is scoped to a **physical GCwerks channel** rather than an
analyte -- one chromatogram covers every analyte reported on that channel.

### Two phases: scan once, tag many times

Decoding chromatograms is the expensive part, and its result doesn't depend
on which analyte you're tagging, so the two are separated:

```
# Phase 1 -- slow, run once per site/channel/range. Scans raw files and
# upserts every scored chromatogram into hats.ng_chromatogram_qc.
python3 cats_baseline_qc.py --site spo --channel f --start 20260101 --end 20260801

# Phase 2 -- fast, re-runnable. Reads the persisted results, decodes nothing,
# tags every analyte on that channel.
python3 cats_tagging.py --site spo --algo baseline --analyte all \
    --start 20260101 --end 20260801
```

Add `--dry-run` to either phase to preview without writing.

Running Phase 2 per analyte against a live scan (the original design) redid
an identical chromatogram scan 5-8x -- CATS channel `f` alone carries eight
analytes. Phase 1 also stores **unflagged** rows, so a different
`--ratio-threshold` can be re-derived later with a SQL query instead of a
rescan.

Note `--analyte all` in Phase 2 iterates every CATS analyte across *all*
channels, so analytes on channels you haven't scanned yet simply report zero
flags. That's harmless, just expected noise in the summary table.

### Channel mapping (non-SMO CATS sites)

`--channel` takes the **DB channel letter** (resolved via
`gcwerks_channel_number()`) or the **physical GCwerks channel number**
directly (`0`-`3`) -- `gcwerks_channel_number()` returns a bare digit
argument as-is, so `--channel 0` and `--channel q` are equivalent at SPO:

| DB letter | `--channel` number | GCwerks dir | Analytes at SPO |
|---|---|---|---|
| `q` | `0` | `channel0` | N2O, SF6 |
| `a` | `1` | `channel1` | CFC11, CFC113, CFC12, H1211, N2O |
| `f` | `2` | `channel2` | CCl4, CFC11, CFC113, CFC12, CH3CCl3, CHCl3, H1211, TCE |
| `c` / `cc` | `3` | `channel3` | CFC12, CH3Br, CH3Cl, H1211, H1301, HCFC142b, HCFC22, OCS |

**SMO/IE3 uses a different letter mapping** (`a/b/c` → 0/1/2) -- the numeric
form sidesteps that ambiguity entirely, since the channel number is the same
physical thing regardless of site. See `gcwerks_channel_number()` in
`logosdata/gcwerks_chromatogram.py`.

### How detection works (cats_baseline_qc.py)

1. **Tail normalization.** Each chromatogram is divided by the mean of its
   own signal over 29.0-29.5 min -- a quiet, settled region after everything
   has eluted. Every trace is then expressed relative to its own steady
   state, which removes the large day-to-day baseline drift (routinely
   ±100% at SPO) that makes absolute levels useless as a feature.
2. **Local median reference.** For each target, take the pointwise **median**
   of the `--neighbor-window` (default 10) chromatograms before and after it
   on the same channel. Median, not mean: during a multi-run contamination
   event several neighbours are themselves bad, and a mean reference gets
   visibly dragged toward the anomaly (measured ~1.2x baseline instead of a
   flat 1.0 near the 2026-07-17 SPO event), understating exactly the runs
   being hunted. The reference also **excludes the target row** -- including
   it lets a strongly anomalous run pull its own reference toward itself,
   always biasing toward under-detection.
3. **Mask every analyte's peak.** For each analyte reported on the channel,
   GCwerks' own dated peakid file (`find_gcwerks_peakid_file` /
   `read_gcwerks_peak_windows`) gives that era's real retention-time window;
   each window is padded by `--peak-margin-min` (default 0.3) and excluded
   from scoring. This is not a fixed minute range -- peak retention times
   drift over a 25+ year archive and differ by channel, and GCwerks'
   per-era files already track that drift, so the mask adapts automatically
   instead of needing hand-maintained config per channel/era. Masking (not
   just a fixed early window) is what lets the score cover the *whole*
   trace: 260713.0258.8 (SPO channel0) has a normal 0-5 min baseline but a
   real post-peak sag around minute 22-23 that an earlier pre-peak-only
   design couldn't see.
4. **Score as a ratio, over the whole non-peak trace.** `ratio = target /
   reference` at every remaining (unmasked) sample, then `ratio_mean` and
   `ratio_std` summarize the entire trace as two numbers. Ratio, not
   difference, is what makes one mean meaningful across the whole trace
   instead of needing separate thresholds per region. Masking is required
   for this to work, not optional: an *unmasked* ratio blows up at peak
   edges, where a small height or timing mismatch between target and
   reference produces a huge ratio error because the reference is large and
   changing fast there. Tested directly on 2026-07-13 (SPO channel0): an
   entirely ordinary tall-peak run (ratio_mean 1.6, peak height 7.65x its
   own tail) outranked the real anomaly (ratio_mean 0.89) when peaks weren't
   masked. With peaks masked, both are still correctly flagged --
   the oversized peak's base still extends past its own masked window into
   the baseline being scored, so it isn't hidden by masking, just no longer
   the dominant noise source everywhere else.
5. **Flag** when `|ratio_mean - 1| > --ratio-threshold` (default 0.05). This
   is an absolute cutoff, not a z-score -- the local-median design already
   removes the need for a separately tuned statistic. On 60 quiet SPO
   channel0 runs (June 2026) `ratio_mean` sits at 0.986-1.007 (std 0.004);
   the validated 2026-07-13 event scores 0.89-0.90 (and 1.6 for the
   tall-peak run one injection prior) -- a clean separation.
6. **Period grouping** (`--max-gap-hours`, default 1): flagged runs within
   this gap merge into one `period_id`, so a multi-run event reads as one
   episode rather than a scatter of points.

**Known limitation:** the local median degrades if a majority of the
neighbours on one side are themselves anomalous -- a median only resists a
*minority* of outliers. Keep the window modest and review flagged clusters
together rather than trusting a single isolated run's score.

**What it does not catch:** peak-height dropouts (the peak region is masked
out of the score entirely by design). See `cats_peak_qc.py` for peak-height
features.

### Performance

Reads are threaded (`--workers`, default 16) because they come off NFS and
are latency-bound, not CPU- or bandwidth-bound. Measured on a cold month of
SPO channel0: **51.5 s → 7.8 s (6.6x)**, bit-identical output. That puts a
full ~29-year single-channel scan at roughly 40 minutes rather than several
hours. Lower `--workers` if the NFS server objects.

### hats.ng_chromatogram_qc

Phase 1's output table -- one row per **scored chromatogram**, flagged or
not. Created 2026-08 (Geoff + Claude), columns renamed 2026-08 when the
algorithm changed from pre-peak-window difference to whole-trace masked
ratio; the DDL lives here because it is not otherwise captured in the repo:

```sql
CREATE TABLE hats.ng_chromatogram_qc (
    num INT AUTO_INCREMENT PRIMARY KEY,
    inst_num INT NOT NULL,
    gcwerks_channel_num INT NOT NULL,
    analysis_time DATETIME NOT NULL,
    algo_name VARCHAR(32) NOT NULL,
    ratio_mean DOUBLE PRECISION,
    ratio_std DOUBLE PRECISION,
    n_masked_samples INT,
    outlier TINYINT NOT NULL DEFAULT 0,
    period_id INT NOT NULL DEFAULT 0,
    scanned_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_scan (inst_num, gcwerks_channel_num, analysis_time, algo_name),
    KEY idx_outlier (inst_num, gcwerks_channel_num, algo_name, outlier)
);
```

| Column | Meaning |
|---|---|
| `inst_num` | CATS site instrument number (SPO=244 etc.; see `../CLAUDE.md`) |
| `gcwerks_channel_num` | **Physical** channel dir (0-3), not the DB letter -- one GCwerks channel maps to different letters at SMO vs other sites, so the table records physical reality |
| `analysis_time` | Chromatogram start, matching `ng_insitu_analysis.analysis_time` exactly (both derive from the file's `YYMMDD.HHMM`) |
| `algo_name` | Detector that produced the row (`baseline` today) |
| `ratio_mean` | Mean of target/reference over every non-peak sample -- the value `--ratio-threshold` tests against 1.0 |
| `ratio_std` | Std of the same ratio; not used to flag today, kept for review (a real event's std is typically far larger than a quiet run's, e.g. 0.52 vs ~0.01-0.08) |
| `n_masked_samples` | How many samples this row's peakid-window mask excluded; a sanity check that the mask found the expected analyte windows |
| `outlier` | 1 if it exceeded the threshold at scan time |
| `period_id` | Groups consecutive flagged runs into one episode; 0 for unflagged |
| `scanned_at` | When the row was last written |

Floats are `DOUBLE PRECISION` deliberately -- `NUMERIC` maps to
`decimal(10,0)` through this driver and would silently truncate to integers
(see `../CLAUDE.md`).

The unique key makes re-scanning an overlapping range an in-place update
rather than a duplicate, so ranges can be extended or re-run freely.

Useful queries:

```sql
-- Coverage: what has been scanned, and how much was flagged
SELECT inst_num, gcwerks_channel_num, algo_name,
       COUNT(*) AS scanned, SUM(outlier) AS flagged,
       MIN(analysis_time) AS first, MAX(analysis_time) AS last
FROM hats.ng_chromatogram_qc
GROUP BY inst_num, gcwerks_channel_num, algo_name;

-- Flagged episodes for one site/channel
SELECT period_id, COUNT(*) AS n,
       MIN(analysis_time) AS start, MAX(analysis_time) AS end,
       MAX(ABS(ratio_mean - 1.0)) AS worst
FROM hats.ng_chromatogram_qc
WHERE inst_num = 244 AND gcwerks_channel_num = 0 AND outlier = 1
GROUP BY period_id ORDER BY start;

-- Re-derive a different threshold without rescanning
SELECT COUNT(*) FROM hats.ng_chromatogram_qc
WHERE inst_num = 244 AND gcwerks_channel_num = 0
  AND ABS(ratio_mean - 1.0) > 0.10;
```

## cats_cal_window_qc.py -- "Mole fraction falls outside of calibration" (tag 286)

Finds air1/air2 mole fractions that drift beyond what the instrument's own
current noise level can explain, using the reference tank (CAL2_PORT, the
near-ambient tank normalization is anchored to -- see `../CLAUDE.md`) as a
live noise gauge rather than a fixed tolerance.

**Must run after `cal_step` and `baseline`** -- see "Run order" below.

Reproduce/tune:

```
cd cats_qc
python3 cats_tagging.py --site brw --algo cal_window --analyte N2O \
    --channel q --start 20250101 --end 20250401 --dry-run
```

Same idempotency contract as `cal_step`/`baseline`: `cats_tagging.py` deletes
tag 286 from every `mf_num` in the requested scope before reinserting it on
whatever's currently flagged, so it's always safe to rerun after retuning a
threshold.

### How detection works

1. **Reference-tank noise as the yardstick.** For each candidate air1/air2
   reading, `ref_std` is the plain standard deviation (ddof=0) of the
   reference tank's `mole_fraction` values inside a window centered on that
   reading -- a direct, contemporaneous measurement of how noisy the
   instrument currently is, independent of atmospheric variability.
2. **Local air median as the baseline.** `air_median` is the median
   `mole_fraction` of every air1+air2 reading in the same window, computed
   leave-one-out (the candidate's own value is excluded) -- same rationale
   as `cats_baseline_qc.py`'s local reference: including the candidate lets
   an anomalous point drag its own baseline toward itself, always biasing
   toward under-detection. Air1 and air2 are pooled into one median, not
   scored separately, since both read the same air intake through the same
   normalization.
3. **Asymmetric bounds** (`--sigma-high`/`--sigma-low`, default 3/2): flag
   when the reading is more than `sigma_high` reference-tank sigmas *above*
   `air_median`, or more than `sigma_low` sigmas *below* it. Downward
   excursions (partial peaks, leaks, a starved sample loop) are judged more
   strictly than upward ones (e.g. brief contamination spikes).
4. **Window** (`--window-days`, default 10, i.e. +/-5 days): wide enough to
   collect a stable reference/air sample at CATS' cal/air cadence, narrow
   enough that both statistics reflect what the instrument and atmosphere
   were doing right around that point rather than a stale trailing average.
5. **Minimum context** (`--min-ref-points`/`--min-air-points`, default 4
   each): a candidate whose window has too few reference or (leave-one-out)
   air neighbors gets `NaN` stats and is never flagged -- same "not enough
   context to judge" convention as `cal_step`'s `scale=NaN` rows.

### Already-rejected data is excluded from the statistics, not from candidacy

`ref_std` and `air_median` are built only from rows **not already rejected
for some other reason** (a cal-port glitch caught by `cal_step`, a bad
chromatogram caught by `baseline`, manual review, GCwerks sync, etc.) --
already-known-bad points shouldn't count toward "what normal noise/air looks
like right now". Every air1/air2 row is still itself evaluated as a
candidate regardless of its own rejected status; only the neighbor *pool*
used to judge other points is filtered.

Crucially, this filter deliberately excludes rows rejected **only** by
`cal_window`'s own tag (286) -- every rerun re-evaluates every candidate
against the same pool of *other* algorithms' rejects, never its own from a
previous run. If a flagged point's own rejection fed back into the pool it
would itself be judged against, a rerun could shift the median/std enough to
flip the verdict, flapping between two different flagged sets instead of
converging. That is what keeps `cats_tagging.py`'s delete-over-scope +
reinsert cycle landing on the same result every time.

### Run order

`cal_window` needs `cal_step` and `baseline` to have *already tagged*, and
mole fractions to have been *recomputed* against that rejection state --
tagging alone never recomputes a mole fraction (see `../CLAUDE.md`). Running
`cal_window` against stale mole fractions (computed before those rejects
existed) defeats the "already-rejected" filter above.

`cats_tagging.py` handles both automatically:

- `--algo all` already sequences `cal_step -> baseline -> cal_window`
  (`ALGORITHMS`' dict insertion order), each fully applied across every
  requested analyte/channel before the next algorithm starts.
- Immediately before calling `cal_window`'s `build()` for each
  analyte/channel, `main()` calls `recalc_mole_fractions()` -- the same
  `update_fits()` + `_upsert_fits()` + `update_runs()` +
  `upsert_mole_fractions()` sequence as `cats_batch.py -i --fits` -- so its
  statistics always reflect the current rejection state, whether
  `cal_step`/`baseline` were just run in the same command or in an earlier
  one. Skipped under `--dry-run` (no DB writes at all) or `--skip-recalc`
  (mole fractions already known current for this window).

## cats_cal_method_qc.py -- calibration-method discontinuity recommendations

**Detect + recommend only -- never writes to the database.** Unlike the three
detectors above, this is **not** registered in `cats_tagging.py`'s
`ALGORITHMS`: it produces calibration-method recommendations (which of
`ref`/`cal1`/`cal2`/`cal12` a period should use), not
`ng_insitu_mole_fraction_tags` rows -- a fundamentally different kind of
output that doesn't belong in that framework. There is no `--dry-run` flag
since there is nothing to preview a write for.

**v1 scope: BRW N2O + SF6 (channel `q`) only**, validated against a record
the user already knew had visible artifacts (see below). Treat thresholds as
unvalidated starting points before trusting recommendations on other
analytes/sites/eras -- see `CATS_QC_TODO.md`.

### Motivation

Switching a species to `cal12` (weekly 2-point fit through both cal tanks)
is usually the best choice -- it captures both detector gain and offset --
but isn't always achievable across a multi-decade record: a period where one
cal tank's response is too noisy/sparse to fit reliably, or where the tanks
were simply run less consistently in an earlier era, can produce a `cal12`
fit that's worse than a simpler method for that stretch, showing up as a
visible step in the mole-fraction time series. This happened after switching
BRW SF6/N2O to `cal12` for their whole 1998-2026 record: real SF6/N2O have
only a slow secular trend plus a seasonal cycle -- they never step -- so a
level discontinuity found near a period boundary is evidence the WRONG
method was chosen for that period, not that the underlying data is bad.

A secondary hypothesis: if NO candidate method resolves a discontinuity, the
culprit is more likely a bad `hats.scale_assignments` entry for one of the
cal tanks -- no method choice can fix that, so those periods are reported
`UNRESOLVED` (with the cal-tank serials active at that date) for manual
review, never silently "fixed" by whichever method merely scores least-bad.

### How detection works

1. **Period-level series.** Loads the currently-persisted `mole_fraction`
   for air1/air2 rows, aggregated to `CATS_batch._fit_periods()` periods
   (normally calendar weeks, split at a mid-week cal-tank swap -- the same
   boundary granularity `update_fits()` itself fits on, so a recommended
   period's `period_start` is already a valid `cats_set_mf_method.py
   --start-date`). Each period's representative value is the **median**
   (not mean) of its unrejected rows -- same rationale as `baseline`/
   `cal_window`'s local references, one bad week shouldn't skew the level.
2. **Two-sided detrended jump statistic.** Point-to-point differencing
   (`cal_step`'s approach) doesn't work here -- SF6/N2O trend secularly and
   cycle seasonally, and a naive diff would flag the trend itself. Instead,
   at each candidate period, independent robust (Theil-Sen) trend lines are
   fit to the ~1-year windows *before* and *after* it, each excluding a
   `--gap-days` buffer around the candidate so its own value (and its
   immediate neighbors, which may already be drifting toward a real step)
   can't contaminate either side's trend estimate. Both lines are
   extrapolated to the candidate's own time; `jump` is the difference
   between them. The scale is `1.4826 * median(|residual|)` pooled over
   both sides' own fits -- deliberately absorbing whatever seasonal wiggle a
   straight line doesn't capture, so the resulting z-score threshold is
   self-calibrated per analyte/era instead of a hand-tuned absolute cutoff
   or a separate seasonal decomposition. A period needs
   `>= --min-trend-points` on **both** sides or gets `NaN` (never flagged).
3. **Grouping.** Flagged periods are merged into episodes with
   `cats_cal_step_qc._group_periods` (reused, not reimplemented), bridging
   gaps up to `--max-gap-days`. Each episode's anchor is its most-deviant
   period.
4. **Candidate evaluation.** For each episode, the surrounding window is
   recomputed under every method in `--method-preference` (default
   `cal12,cal2,cal1,ref`) via `CATS_batch.update_fits`/`update_runs(...,
   method_override=M)` -- an existing **non-mutating** "try a method"
   harness, no DB writes. The same jump statistic is recomputed at the
   anchor under each candidate; the first method (in preference order) that
   clears `--resolve-z-threshold` is the recommendation. None clear it ->
   `UNRESOLVED`.

   **Contract detail:** always pass `fits_override=fits` -- the literal
   (possibly empty) return of `update_fits()` -- into `update_runs()`, never
   convert an empty DataFrame to `None`. `update_runs()` branches on
   `fits_override is None`, not `.empty`: `None` falls back to reading
   whatever fit is *currently persisted* in `hats.ng_response`, silently
   testing stale on-disk data instead of the freshly-forced candidate. An
   empty `fits` table for a `ref` candidate (which stores no fit at all) is
   handled correctly by `calc_mole_fraction_from_fits`'s `direct_mask`
   branch regardless.

### Output

One row per episode: `episode_start`/`episode_end`, `current_method`
(modal `mf_method_num` already recorded), `detected_jump`/`detected_z`,
`jump_<method>`/`z_<method>` for every candidate, `recommendation` (or
`UNRESOLVED` + `cal1_tank`/`cal2_tank` serials).

### Applying a recommendation (manual, by design)

```
python3 cats_set_mf_method.py --site brw --start-date <episode's period_start> \
    --pnum <pnum> --channel q --method <recommendation>
python3 cats_batch.py --analyte <gas> -c q --site brw -s <period_start> -i --fits
```

Each episode's `period_start` is already a valid `--start-date` for both
commands; the next episode's `period_start` implicitly bounds how far the
previous recommendation should extend, so no explicit end-date concept is
needed in the output.

### Usage

```
python3 cats_cal_method_qc.py --site brw --gas SF6_q --start 19980101 -v
```

## Other files in this directory

- `cats_cal_ratio_qc.py` / `cats_peak_qc.py` / `cats_peak_qc_plot.py` /
  `cats_compare.py` / `cats_qc.py` -- earlier/separate QC exploration, not
  covered above and not tracked in git.
- `test_cats_cal_step_qc.py` -- unit tests for the pure helper functions
  (`_port_rate_outliers`, `_group_periods`) using synthetic data; run with
  `python3 -m unittest test_cats_cal_step_qc.py`.
- `test_cats_cal_window_qc.py` -- unit tests for the pure windowed-outlier
  core (`_windowed_air_outliers`) using synthetic data; run with
  `python3 -m unittest test_cats_cal_window_qc.py`.
- `test_cats_cal_method_qc.py` -- unit tests for the pure period-aggregation
  and detrended-jump core (`_period_medians`, `_local_level_jump`) using
  synthetic data (including a trend+seasonal-cycle case validating the
  "one-year window averages out the seasonal cycle" design assumption); run
  with `python3 -m unittest test_cats_cal_method_qc.py`.
