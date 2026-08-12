# CATS QC TODO

Pending follow-ups for the cats_qc tagging work. See `README.md` for how the
`cal_step` algorithm and `cats_tagging.py` work.

## Promote tag 401 from information to reject

Once tag 401 (Detector cal-response rapid change) is registered in
`ccgg.tag_dictionary` and reviewed, switching it from informational to an
actual reject tag is a single field flip -- `rejected` is a live join against
`tag_dictionary.reject`, not a stored value, so nothing in
`ng_insitu_mole_fraction_tags` or `cats_tagging.py` needs to change:

```sql
UPDATE ccgg.tag_dictionary SET reject = 1, information = 0 WHERE num = 401;
```

Requires write access to `ccgg.tag_dictionary` (coordinate with whoever has
it). The moment this runs, every row currently tagged 401 becomes
`rejected=1` everywhere at once -- logos_data, timeseries, etc. -- with no
rerun of `cats_tagging.py` needed.

Two things to resolve before doing this:

1. **This is all-or-nothing.** Flipping `reject` on tag 401 promotes every
   algorithm-flagged point globally, not just ones reviewed and confirmed.
   If a curated approach is preferred instead -- keep 401 purely
   informational and only reject the specific episodes reviewed in
   logos_data -- promote those points individually to a real reject tag
   (141) via the MultiTagPanel instead of flipping the registry entry.
   Decide which workflow before flipping it, since reverting means
   re-reviewing what should go back to unrejected.
2. **Mole fractions need recomputing afterward.** Once cal points are
   actually excluded, the weekly cal fit changes for any week touching them.
   Rerun `cats_batch.py --site <site> -p <pnum> -c <channel> -i --fits -s
   <week>` (or use "Update Method"/"Update MF" in logos_data) for the
   affected weeks.
