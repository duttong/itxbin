# CATS Data Processing Cheat Sheet

Quick-reference commands. Deep rationale for the QC algorithms lives in
`cats_qc/README.md` and `../CLAUDE.md` — this file is the "what do I type"
version.

---

## 1. Ongoing daily ingest — BRW, SPO (cron-driven, automatic)

Runs unattended via crontab (`crontab -l`). Chain per site each morning:

```
cats_incoming.py <site>          # 03:20 — pull incoming GC data
catseng.py --output <site>       # 07:4x — engineering data (sample loop flow/temp/pressure)
cats_export.py <site>            # 09:0x — export GCwerks -> /hats/gc/cats_results/*.csv
cats_ingest.py <site> --skip-export   # 09:1x — load CSVs, sync GCwerks flags, assign
                                       #         cal method, recompute weekly fits + MF
```

`cats_ingest.py <site>` (no `--skip-export`) does all of export -> load ->
mf-method-assign -> recompute in one call — use this for a manual re-run.
Add `--run-aftp` only for a one-time legacy-MF backfill; add `--no-mf` to
skip the auto cal-method/recompute step (rare).

Nothing to do here normally — this is the "is it still running" reference,
not a workflow you execute by hand.

---

## 2. Legacy/historical reprocessing template

Use this whenever redoing or updating tagging or mole-fraction calculation
for a site/date range (any of `brw mlo nwr smo spo sum`) — the Igor Pro ->
logos_data transition work.

### 2a. One-time: bulk-load a site that hasn't been ingested into `ng_insitu_*` yet

```
cats_ingest.py --all <site>          # or: cats_ingest.py <site> --year YYYY
```

Skip this if the site/range is already loaded (check `ng_insitu_analysis`
coverage, or just try step 2b with `--dry-run`).

### 2b. Automated QC tagging

```
cd cats_qc

# One-time per site/channel (slow, ~40 min/channel/site for full history):
# scans raw chromatograms, upserts hats.ng_chromatogram_qc
python3 cats_baseline_qc.py --site <site> --channel <0-3|q|a|f|c> --start YYYYMMDD --end YYYYMMDD

# Apply all registered algorithms (cal_step -> baseline -> cal_window, in that
# order; recomputes mole fractions automatically right before cal_window runs)
python3 cats_tagging.py --site <site> --algo all --gas all \
    --start YYYYMMDD --end YYYYMMDD --dry-run     # preview first
python3 cats_tagging.py --site <site> --algo all --gas all \
    --start YYYYMMDD --end YYYYMMDD                # then write
```

Idempotent per (site, analyte, channel, start, end, algo) — safe to rerun
after retuning a threshold flag (`--mad-multiplier`, `--ratio-threshold`,
`--sigma-high/low`, etc.); each rerun deletes-then-reinserts that algorithm's
tag across the requested scope.

### 2c. Manual review — logos_data GUI

```
logos_data cats --site <site>
```

- Hollow circle = manual reject, circle+`x` = auto reject, hollow purple
  diamond = info tag.
- Multi-Tag panel (`G` or button) for reject/info tags on selected point(s);
  manual reject defaults to tag 141.
- "Copy Tags to all Analytes" if a change (e.g. a bad chromatogram) should
  reject the same injection across every analyte sharing that `analysis_num`.

### 2d. Assign/force the mole-fraction calculation method

`cats_set_mf_method.py` sets `mf_method_num` on `ng_insitu_mole_fractions`
air-port rows (relabels which cal ports normalize the mole fraction — it
does not itself recompute values). By default it applies each analyte's
`CATS_Instrument.default_mf_method(pnum)` (cal12, or cal1 for CCl4,
parameter_num=37), falling back to ref if the needed cal tank(s) have no
`hats.scale_assignments` entry. `--start` is required (no baked-in date
floor — CATS has decades of legacy history populated from published /aftp
mole fractions, not weekly cal12 fits); only tag the window you're about to
recompute in 2e.

```
cd cats_qc

# Default method per analyte, one gas/channel:
python3 cats_set_mf_method.py --site <site> --gas <ANALYTE>_<channel> \
    --start YYYYMMDD --dry-run     # preview counts first
python3 cats_set_mf_method.py --site <site> --gas <ANALYTE>_<channel> \
    --start YYYYMMDD               # then write

# Force a specific method instead of the default:
python3 cats_set_mf_method.py --site <site> --gas <ANALYTE>_<channel> \
    --start YYYYMMDD --method {ref,cal1,cal2,cal12} --dry-run
```

`--pnum` (comma-separated parameter_nums) is an alternative to `--gas` when
you want every channel a pnum reports on, or `--channel` to restrict
further; omit both to touch every analyte for the instrument. `--end` is
optional (defaults to "from --start onward"). Idempotent — safe to rerun.
This is also run automatically per episode by `cats_apply_cal_method.py`
(2f) and daily by `cats_ingest.py`'s auto cal-method/recompute step (1); run
it by hand for a one-off relabel outside those flows.

### 2e. Recompute mole fractions after any tag change

Required after 2c (GUI tag edits are not auto-recomputed), after retagging
via 2b if you skipped its auto-recalc (`--skip-recalc`), and after 2d
(method assignment alone does not recompute values):

```
cats_batch.py --site <site> -p all -i --fits -s <start> -e <end>
# or one analyte, by parameter_num:
cats_batch.py --site <site> -p <pnum> -c <channel> -i --fits -s <start> -e <end>
# or by Analyte_channel token (same format as cats_set_mf_method.py's --gas):
cats_batch.py --site <site> --gas <ANALYTE>_<channel> -i --fits -s <start> -e <end>
```

### 2f. Calibration-method review (optional — redo *how* MF is calculated)

For suspected non-atmospheric steps from a bad per-period cal method choice
(ref/cal1/cal2/cal12). **v1 only validated for BRW N2O/SF6 (channel q)** —
see `CATS_QC_TODO.md` before trusting elsewhere.

```
cd cats_qc
python3 cats_cal_method_qc.py --site <site> --gas <ANALYTE>_<channel> \
    --start YYYYMMDD -v --output <site>_<analyte>_calmethod.csv

# review CSV, then apply RESOLVED episodes oldest-first:
python3 cats_apply_cal_method.py --site <site> --input <csv> --dry-run   # preview commands
python3 cats_apply_cal_method.py --site <site> --input <csv>             # apply for real
```

`cats_apply_cal_method.py` runs `cats_set_mf_method.py` (relabels
`mf_method_num`) + `cats_batch.py -i --fits` per episode for you.
`UNRESOLVED` episodes are skipped/printed for manual `hats.scale_assignments`
review — no method choice fixes a bad scale assignment.

To force a method directly without the discontinuity-detection step, use
`cats_set_mf_method.py --method` directly (2d) followed by `cats_batch.py
--fits` (2e).

### 2g. Cal-tank health review (optional — find/route around a bad CAL1 or CAL2 tank)

Different question from 2f: instead of detecting mole-fraction jumps and
asking which method removes them, this looks directly at each cal tank's
own raw injection data and asks whether the tank was actually delivering
usable measurements — offline, dry (empty integrations), or heavily
starved relative to its sibling port that period. See
`cats_cal_tank_health_qc.py`'s module docstring for the full algorithm.

Coverage and dropout are properties of the shared CAL1/CAL2 tanks (every
channel sees the same injection at once), not of any one analyte, so the
recommended workflow detects episodes once from a small set of
traditionally well-measured reference analytes and applies the result to
every analyte:

```
cd cats_qc

# Combined, analyte-independent verdict from coverage+dropout only
# (noise is intentionally excluded — it's per-analyte/per-channel, not a
# shared tank property; see the script's Reference mode docstring)
python3 cats_cal_tank_health_qc.py --site <site> \
    --reference-gas N2O_q --reference-gas SF6_q --reference-gas CFC11_f \
    --start YYYYMMDD -v --output <site>_reference_tank_health.csv

# review CSV, then apply to every analyte (or a specific list):
python3 cats_apply_cal_tank_health.py --site <site> \
    --input <site>_reference_tank_health.csv --gas all --dry-run   # preview
python3 cats_apply_cal_tank_health.py --site <site> \
    --input <site>_reference_tank_health.csv --gas all             # apply
```

`cats_apply_cal_tank_health.py` runs, per analyte with a RESOLVED episode:
`cats_set_mf_method.py --start <episode_start> --end <episode_end>`
(bounded to just the outage window, unlike 2f's open-ended apply) for each
episode, then ONE `cats_tagging.py --algo cal_window` retag at the end.
That retag call is itself what recomputes mole fractions against the
method(s) just set (`cal_window` is in `cats_tagging.py`'s
`_RECALC_BEFORE_BUILD` set — see §3) — **no separate `cats_batch.py` or
extra `cats_tagging.py` call is needed afterward**; the apply script's own
output above is the complete recompute-and-retag step. `UNRESOLVED`
episodes are skipped — both tanks (or neither clearly) looked bad that
period, so no single-tank method reliably avoids the problem.

**Full per-analyte processing sequence** (recommended order when
onboarding/reprocessing an analyte with this workflow):

1. **Fresh analyte** (no prior 2f `cats_apply_cal_method.py` history for
   it): `cats_set_mf_method.py --site <site> --gas <ANALYTE>_<channel>
   --start YYYYMMDD` (2d) to set the default method for the whole range.
   **Analyte with prior 2f work**: skip this step — it would reset the
   whole range and erase those discontinuity-based fixes.
2. `cats_apply_cal_tank_health.py --gas <ANALYTE>_<channel> --input
   <reference CSV>` — layers the bounded tank-outage overrides on top and
   recomputes+retags in the same call (see above); safe either way since
   it never resets.
3. Review in `logos_data` (2c) — mole fractions are already current at
   this point, nothing further to recompute first.

Cal-tank health (2g) and calibration-method review (2f) are independent
and compose: run either or both, in whichever order fits — 2g's apply
never resets, so an episode from one doesn't need to "come after" the
other except by your own judgment about which fix should win where they'd
overlap in time.

### 2h. Viterbi method-per-period picker (alternative to 2f — detect + recommend only)

A different way to answer the same question as 2f: instead of detecting
local jumps against a trend window and greedily trying candidates per
flagged episode, `cal_method_viterbi.py` looks at the *whole*
record at once and picks the per-period method (cal12/cal2/cal1) that
minimizes total first-derivative roughness of the resulting series —
subject to a strong preference for cal12 and a switch cost that's cheap
right at a real cal1/cal2 tank transition and expensive everywhere else.
Validated against BRW CCl4(f)/N2O(q)/SF6(q): beats or matches the
currently-persisted series' total roughness in all three, and
independently lands on `cal12` through the N2O 2019-04-22 discontinuity
that motivated 2f's own algorithm redesign — without being told. See the
script's module docstring for the full DP formulation.

```
cd cats_qc
python3 cal_method_viterbi.py --site <site> --gas <ANALYTE>_<channel> \
    --start YYYYMMDD --output <site>_<analyte>_viterbi.csv --plot <site>_<analyte>_viterbi.png
```

Detect + recommend only, like 2f/2g — never writes to the database, and
there is no `cats_apply_*`-style batch-apply script for its output yet
(apply by hand: for each contiguous run of the same `chosen` method in the
output CSV, `cats_set_mf_method.py --start <run_start> --method <chosen>`,
oldest-first, same as 2f's apply pattern — see 2d). The printed summary
reports total variation (blanket cal12 / currently persisted / DP-chosen),
how many method changes landed on a real tank transition vs. not, and any
period-to-period step still large even at the best available choice — the
Viterbi analogue of 2f's `UNRESOLVED`.

**Two known false leads, worth knowing before treating a divergence as a
tank/method problem:**

- A **single anomalous period** (e.g. a freshly-swapped tank's first,
  still-settling reading) can look like strong evidence for switching away
  from cal12 even though the tank is fine days later. `--smooth-window`
  (default 3, a centered rolling median applied to each candidate before
  the DP runs) exists specifically to wash this out without discarding the
  long-run evidence needed to catch a candidate that's *actually* wrong for
  its whole service life — see the script's docstring for the BRW
  `CC456884` (settling, real) vs. `ALM069781` (persistently wrong, real,
  fixed via a bad-fill-code split — see below) contrast that motivated this.
- **A persistent divergence between candidates doesn't by itself tell you
  which one is right** — both can be individually smooth. Before concluding
  a tank's scale_assignments value is wrong, check whether the value is
  independently corroborated (e.g. by `cats_cal_tank_health_qc.py`'s
  per-tank noise/coverage signal, or by comparing to what's already
  persisted in the database) — see the BRW N2O `ALM-052752` case (2023-06
  to 2025-11: cal2 read ~1 ppb low the whole time, but the tank's assigned
  value itself checked out fine against two independent M3 measurements
  6 years apart) vs. the `ALM069781` case below (an *actually* bad
  assignment, confirmed and fixed at the source).

**Real bug this workflow found and fixed**: BRW's `ALM069781` (CATS cal1
tank, 2004-2009) had its `reftank.fill` record lumping two distinct real
fills under one fill code — 2003/2006 calibration points from the original
fill, 2007/2009 points from an undocumented refill — so `caldrift.py` was
fitting a spurious quadratic "drift" curve through what was actually a
step change, corrupting the scale assignment for CCl4, CFC11, CFC12,
CFC113, CHCl3, N2O, and SF6 alike (every gas measured on that tank). Fixed
at the source: split `reftank.fill` into two fill codes at the true refill
date, then re-ran `caldrift.py --mean` per gas per fill (`--fit_inst STDGC`
for the CFC/CCl4 group, `--fit_inst M3` for CHCl3, `--database reftank` for
N2O/SF6, which predate `hats.scale_assignments_view`). This is the
general pattern for chasing a Viterbi- or 2f-flagged divergence back to its
root cause rather than papering over it with a method choice — a bad
scale_assignments value can look exactly like a discontinuity, and no
method choice fixes it (2f would have reported this as `UNRESOLVED`, and
did, historically, in an earlier `cats_cal_method_qc.py` run for this
same tank/episode).

---

## 3. Tagging algorithms — quick reference

**Automated, registered in `cats_tagging.py`** (write real reject tags to
`hats.ng_insitu_mole_fraction_tags`; run order matters — `--algo all`
sequences these correctly):

- **`cal_step`** (tag 328, *Detector cal-response rapid change*) — per
  analyte+channel. Flags abrupt point-to-point step changes in cal-port
  (2/6) response vs. that port's own trailing-30-day MAD noise; sweeps in
  the interleaved air points bordering the flagged episode.
- **`baseline`** (tag 329, *Abnormal chromatogram*) — per physical channel
  (covers all analytes on it). Two-phase: `cats_baseline_qc.py` scans raw
  chromatograms (target vs. local median-of-neighbors trace, analyte peaks
  masked out) and persists to `hats.ng_chromatogram_qc`; `cats_tagging.py`
  then tags every analyte on that channel from the persisted scores.
- **`cal_window`** (tag 286, *Mole fraction falls outside of calibration
  range*) — per analyte+channel. **Runs last.** Flags air1/air2 mole
  fractions >3σ above / >2σ below the local (±5 day) air median, using
  reference-tank (cal2) noise as the yardstick; requires mole fractions
  already recomputed against `cal_step`/`baseline` rejections (handled
  automatically by `--algo all`).

**Recommendation-only, not a `cats_tagging.py` algorithm (no DB writes):**

- **`cal_method_qc`** — detects non-atmospheric level discontinuities via a
  detrended (Theil-Sen) trend-jump test, recommends the best per-period
  cal method (ref/cal1/cal2/cal12) to remove them, or reports `UNRESOLVED`
  (bad `scale_assignments`, not a method problem). Apply recommendations
  with `cats_apply_cal_method.py`. v1: BRW N2O/SF6 (q) only.
- **`cal_tank_health_qc`** — scores each cal port's own raw injections per
  week (coverage relative to its sibling port, dropout fraction, response
  noise vs. its own trailing baseline) and recommends switching to the
  single-tank method (cal1/cal2) that avoids whichever tank looks
  unhealthy, bounded to just that episode's window, or `UNRESOLVED`. Run
  in `--reference-gas` mode (coverage+dropout only, combined across
  N2O_q/SF6_q/CFC11_f) for an analyte-independent verdict applied to every
  analyte via `cats_apply_cal_tank_health.py --gas all`. See 2g.
- **Viterbi method-per-period picker** (`cal_method_viterbi.py`)
  — alternative to `cal_method_qc`: picks the per-period method that
  minimizes total roughness of the whole assembled series at once
  (dynamic programming), instead of detecting local jumps and greedily
  trying candidates per episode. Validated on BRW CCl4(f)/N2O(q)/SF6(q).
  No batch-apply script yet — apply by hand per 2h. See 2h.

**Manual / legacy (outside `cats_qc/`):**

- **Manual reject** — logos_data GUI, tag 141 (default).
- **GCwerks flag sync** — `F`/`*`/`B` chars from GCwerks, tag 324; synced by
  `cats_gcwerks2db.py` on every ingest (stale 324 tags deleted, current ones
  reinserted).

**Exploratory, not wired into `cats_tagging.py`:** `cats_cal_ratio_qc.py`
(cal-pair response-ratio deviations), `cats_peak_qc.py` (peak-integration/RT
outliers) — useful standalone, but produce no tags today.
