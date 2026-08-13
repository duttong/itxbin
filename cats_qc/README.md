# cats_qc

CATS QC detection algorithms and the tagging tool that applies their results
to `hats.ng_insitu_mole_fraction_tags`.

Only the active pipeline is tracked in git (`cats_tagging.py`,
`cats_cal_step_qc.py`, `cats_baseline_qc.py`, `test_cats_cal_step_qc.py`, and
these docs) -- the rest of this directory is earlier exploratory work and is
gitignored. See the `cats_qc/` allowlist in `../.gitignore`.

Two detectors are registered today:

| Algorithm | Tag | What it finds | Scope |
|---|---|---|---|
| `cal_step` | 328 (reject) | Abrupt cal-port response shifts | per analyte + channel |
| `baseline` | 402 (placeholder) | Abnormal chromatogram shape | per **physical channel** |

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

- **tag_num 328 ("Detector cal-response rapid change") is a real reject
  tag**, registered in `ccgg.tag_dictionary` with `reject=1, automated=1`.
  Points it tags become `rejected` immediately, because the view's `rejected`
  column is a live join against `tag_dictionary.reject`.
- **tag_num 402 ("Abnormal chromatogram") is still a placeholder** -- not yet
  in `ccgg.tag_dictionary`, so it currently has no effect on `rejected` (the
  join finds no row). Once a real number is assigned, update `ALGORITHMS` in
  `cats_tagging.py`; rows already written under the placeholder move with a
  single `UPDATE hats.ng_insitu_mole_fraction_tags SET tag_num = <new> WHERE
  tag_num = 402`. (An earlier 401 placeholder for `cal_step` was retired this
  way when 328 was registered, and all 401 rows were deleted.)
- `_TAG_LAYOUT` in `logosdata/logos_data.py` carries both, under **Automated
  Tags**: 328 as a reject tag paired with the 401 slot, and 402 as
  "Abnormal chromatogram". Informational tags show up as the info-tag overlay
  (hollow purple diamond) when "Show Info Tags" is checked, and in the point
  tooltip.
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

## cats_baseline_qc.py -- "Abnormal chromatogram" (tag 402)

Finds chromatograms whose **pre-peak baseline shape** departs from what its
own neighbours were doing at the time: contamination carryover, detector
upsets, ECD instability, runs that start before the detector has recovered
from whatever preceded them.

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
`--diff-threshold` can be re-derived later with a SQL query instead of a
rescan.

Note `--analyte all` in Phase 2 iterates every CATS analyte across *all*
channels, so analytes on channels you haven't scanned yet simply report zero
flags. That's harmless, just expected noise in the summary table.

### Channel mapping (non-SMO CATS sites)

`--channel` takes the **DB channel letter**; the scan resolves it to the
GCwerks channel directory via `gcwerks_channel_number()`:

| DB letter | GCwerks dir | Analytes at SPO |
|---|---|---|
| `q` | `channel0` | N2O, SF6 |
| `a` | `channel1` | CFC11, CFC113, CFC12, H1211, N2O |
| `f` | `channel2` | CCl4, CFC11, CFC113, CFC12, CH3CCl3, CHCl3, H1211, TCE |
| `c` / `cc` | `channel3` | CFC12, CH3Br, CH3Cl, H1211, H1301, HCFC142b, HCFC22, OCS |

**SMO/IE3 uses a different mapping** (`a/b/c` → 0/1/2) -- see
`gcwerks_channel_number()` in `logosdata/gcwerks_chromatogram.py`.

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
3. **Score the pre-peak window only.** Mean and max absolute deviation from
   that reference over the first `--pre-peak-min` (default 5) minutes.
   Scoping to pre-peak is deliberate: baseline changes there can occur
   without touching the downstream analyte peaks at all (heart-cutting), so
   a whole-trace comparison would conflate two different questions.
4. **Flag** when `pre_mean_abs_diff > --diff-threshold` (default 0.15). This
   is an absolute cutoff, not a z-score -- the local-median design already
   removes the need for a separately tuned statistic. On the validated
   2026-07-17 SPO event, affected runs score 0.2-1.4 against ~0.001-0.06 for
   quiet runs the same day, a clean order-of-magnitude separation.
5. **Period grouping** (`--max-gap-hours`, default 1): flagged runs within
   this gap merge into one `period_id`, so a multi-run event reads as one
   episode rather than a scatter of points.

**Known limitation:** the local median degrades if a majority of the
neighbours on one side are themselves anomalous -- a median only resists a
*minority* of outliers. Keep the window modest and review flagged clusters
together rather than trusting a single isolated run's score.

**What it does not catch:** peak-height dropouts with an otherwise-normal
baseline, and anomalies confined to the post-peak region (only pre-peak is
scored, per point 3). See `cats_peak_qc.py` for peak-height features.

### Performance

Reads are threaded (`--workers`, default 16) because they come off NFS and
are latency-bound, not CPU- or bandwidth-bound. Measured on a cold month of
SPO channel0: **51.5 s → 7.8 s (6.6x)**, bit-identical output. That puts a
full ~29-year single-channel scan at roughly 40 minutes rather than several
hours. Lower `--workers` if the NFS server objects.

### hats.ng_chromatogram_qc

Phase 1's output table -- one row per **scored chromatogram**, flagged or
not. Created 2026-08 (Geoff + Claude); the DDL lives here because it is not
otherwise captured in the repo:

```sql
CREATE TABLE hats.ng_chromatogram_qc (
    num INT AUTO_INCREMENT PRIMARY KEY,
    inst_num INT NOT NULL,
    gcwerks_channel_num INT NOT NULL,
    analysis_time DATETIME NOT NULL,
    algo_name VARCHAR(32) NOT NULL,
    pre_mean_abs_diff DOUBLE PRECISION,
    pre_max_abs_diff DOUBLE PRECISION,
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
| `pre_mean_abs_diff` | Mean abs deviation from the local median over the pre-peak window -- the value `--diff-threshold` tests |
| `pre_max_abs_diff` | Max abs deviation over the same window |
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
       MAX(pre_mean_abs_diff) AS worst
FROM hats.ng_chromatogram_qc
WHERE inst_num = 244 AND gcwerks_channel_num = 0 AND outlier = 1
GROUP BY period_id ORDER BY start;

-- Re-derive a different threshold without rescanning
SELECT COUNT(*) FROM hats.ng_chromatogram_qc
WHERE inst_num = 244 AND gcwerks_channel_num = 0
  AND pre_mean_abs_diff > 0.25;
```

## Other files in this directory

- `cats_cal_ratio_qc.py` / `cats_peak_qc.py` / `cats_peak_qc_plot.py` /
  `cats_compare.py` / `cats_qc.py` -- earlier/separate QC exploration, not
  covered above and not tracked in git.
- `test_cats_cal_step_qc.py` -- unit tests for the pure helper functions
  (`_port_rate_outliers`, `_group_periods`) using synthetic data; run with
  `python3 -m unittest test_cats_cal_step_qc.py`.
