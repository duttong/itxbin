# CATS QC TODO

Pending follow-ups for the cats_qc work. See `README.md` for how the
`cal_step` and `baseline` algorithms and `cats_tagging.py` work.

## Register a real tag number for "Abnormal chromatogram" (currently 402)

`baseline` writes tag **402**, a placeholder that is not in
`ccgg.tag_dictionary`, so it has no effect on `rejected` today -- the view's
`rejected` column is a live join against `tag_dictionary.reject`, and with no
row for 402 the join finds nothing. Flagged points still show up in
logos_data as the info-tag overlay via `_TAG_LAYOUT`.

Once a real number is assigned:

1. Update `ALGORITHMS` in `cats_tagging.py` with the real number.
2. Update the 402 entry in `_TAG_LAYOUT` (`logosdata/logos_data.py`).
3. Move already-written rows:
   ```sql
   UPDATE hats.ng_insitu_mole_fraction_tags SET tag_num = <new> WHERE tag_num = 402;
   ```

This is the same path `cal_step` took: it ran under placeholder 401 until
**328** ("Detector cal-response rapid change", `reject=1, automated=1`) was
registered, at which point `ALGORITHMS` was pointed at 328 and every 401 row
was deleted.

**Decide reject vs informational before registering.** If 402 lands with
`reject=1`, every algorithm-flagged point becomes rejected globally and at
once -- not just ones reviewed and confirmed. If a curated approach is
preferred, register it as informational and promote reviewed episodes
individually to a reject tag (141) via the MultiTagPanel. Reverting means
re-reviewing what should go back to unrejected, so the choice is easier made
up front.

**Mole fractions need recomputing afterward**, whenever a reject tag lands on
points that feed a cal fit. Rerun
`cats_batch.py --site <site> -p <pnum> -c <channel> -i --fits -s <week>`
(or use "Update Method" / "Update MF" in logos_data) for the affected weeks.

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
- Does `--diff-threshold 0.15` still separate cleanly, or does the
  quiet-run baseline spread differ enough by era to need a per-era value?
  Unflagged rows are stored, so a different threshold can be re-derived with
  SQL rather than a rescan.

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
