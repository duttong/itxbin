# LOGOS Compare

Run:

```bash
logos_compare
```

This first version compares LOGOS measurement programs using monthly mean mole
fractions as the common data product.  It reuses the database query methods from
`logosdata/logos_timeseries.py` through hidden `TimeseriesWidget` loaders.

Current comparison rules:

- The analyte selector is keyed by `hats.analyte_list.param_num` and labeled
  from `gmd.parameter`, so program-specific `display_name` values do not need
  to match.
- Parameter aliases can be grouped in one selector entry.  For example, CFC11
  displays as `CFC11 (29,114)` and each program uses whichever of those
  parameters it measures.
- The analyte list can be filtered with one-click category buttons:
  `All`, `CFCs`, `HCFCs`, `HFCs`, `Halons`, `Solvents`, `Hydrocarbons`, and
  `Other`.  The analyte combo remains editable/searchable within the selected
  category.
- Programs are offered when the selected parameter number is present for the
  program instrument number. M1, M3, and M4 are combined as `M*`; OTTO and FE3
  are combined as `fECD`.
- Selected programs with no data in the selected time period/sites are omitted
  from the plots and legends.
- `fECD` uses the FE3 analyte list; `M*` uses the M4 analyte list.
- PR1 is loaded from legacy `hats.analysis`/`hats.mole_fractions` tables through
  `logos_timeseries.py`: regular sites use `sample_type='HATS'`, `_PFP`
  pseudo-sites use `sample_type='PFP'`, and rows with non-empty
  `hats.flags_internal.iflag` are excluded. PR1 is averaged by flask event
  first, then by month. HATS sample dates come from
  `hats.Status_MetData.sample_datetime_utc` via `PairID`; PFP sample dates come
  from `ccgg.flask_event`.
- The lower plot shows monthly means for each selected site.
- Real site colors use the same full-network `build_site_colors()` assignment
  as `logos_timeseries.py`, so a site keeps the same color as selections change.
- The upper plot shows the first plotted program minus each other plotted
  program, matched by site and month, with a legend showing the subtraction.
  If only one program has data, the upper plot stays empty and the lower
  timeseries plot still displays that program.
