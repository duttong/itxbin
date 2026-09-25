# CLAUDE.md — logosdata package

This package contains the logos_data GUI application and all supporting
modules. See the parent `itxbin/CLAUDE.md` for database tables, instrument
numbers, and compound parameter numbers.

## Package layout

```
logosdata/
  logos_data.py        # Main PyQt5 GUI — MainWindow, tabs, plotting
  logos_instruments.py # Facade re-exporting the instrument classes below
  logos_instruments_core.py   # LOGOS_Instruments, HATS_DB_Functions, Normalizing
  logos_instruments_flask.py  # M4_Instrument, FE3_Instrument, Perseus_Instrument
  logos_instruments_insitu.py # IE3_Instrument, CATS_Instrument, BLD1_Instrument
  logos_timeseries.py  # TimeseriesWidget and TimeseriesFigure
  data_export.py       # Mstar/Fecd file exporters used by the Timeseries tab
  global_means.py      # cos(lat)-weighted hemispheric and global mean math
  gml_global_means_config.yaml  # background sites and weighting rules for it
  logos_tanks.py       # TanksWidget — tank history and reference tank UI
  logos_ai_agent.py    # LOGOSChatAgent — free-form chat agent
  logos_agent_tools.py # LOGOSDataAgentTools — read-only DB query helpers
  logos_data.conf      # Author-managed UI config (tabs, defaults per instrument)
  __init__.py
```

Root shims `logos_instruments.py` and `logos_agent_tools.py` exist so that
batch scripts (`fe3_batch.py`, `m4_batch.py`, etc.) keep working without
import changes.

All instrument classes should be imported from `logos_instruments` (the
facade), never from the `_core`/`_flask`/`_insitu` modules directly — the
split is an internal layout detail.

## Launching

Use the `logos_data` script in the itxbin root (not `logos_data.py`):

```bash
logos_data fe3
logos_data ie3 --site smo
logos_data            # reads preferred_instrument from ~/.logos_data_user.conf
```

Requires a display: connect with `ssh -Y` or set `$DISPLAY`.

## Configuration files

### logos_data.conf (author-managed)
Lives in this directory alongside the code. Controls per-instrument UI:

| Key | Type | Effect |
|---|---|---|
| `tabs` | comma list | Which tabs are visible (`processing, timeseries, tanks, ai`) |
| `default_analyte` | string | Analyte selected on startup |
| `csv_export` | bool | Show "Save run to .csv" button on Processing tab |
| `change_run_type` | bool | Show "Change Run Type" control on Processing tab |

Edit this file to change UI behaviour for all users. Do not use it for
personal preferences.

### ~/.logos_data_user.conf (user-managed)
Created automatically on first run if absent. Currently supports:

```ini
[user]
preferred_instrument = fe3
```

## Tab structure (MainWindow)

Tabs are built conditionally from `logos_data.conf`. The four possible tabs:

1. **Processing** — always present; date range, run/analyte selection, plotting
2. **Timeseries** — `TimeseriesWidget`; long-term mole fraction trends
3. **Tanks** — `TanksWidget`; reference tank history
4. **LOGOS AI** — `LOGOSAITab`; chat agent backed by `LOGOSChatAgent`

`self.timeseries_tab`, `self.tanks_tab`, and `self.logos_ai_tab` are `None`
when their tab is disabled — guard before use.

## Key MainWindow state

- `self.instrument` — Instrument instance (M4_Instrument, etc.)
- `self._inst_cfg` — `configparser.SectionProxy` for the current instrument
- `self.run` — `pd.DataFrame` for the currently loaded run (one pnum at a time)
- `self.current_pnum` / `self.current_channel` — active analyte
- `self.current_run_time` — selected run timestamp
- `self.madechanges` — dirty flag; cleared on save

`self.run` carries several computed columns beyond the DB view columns:

| Column | Type | Set by |
|---|---|---|
| `rejected` | int 0/1 | view + `_sync_rejected_state()` |
| `auto_rejected` | bool | `_update_auto_rejected()` |
| `has_info_tag` | bool | `_update_info_tagged()` |

Pending "copy to all analytes" state is **not** stored in `self.run` (a reload
would wipe it). It lives in two MainWindow dicts keyed by `analysis_num`:
`self._pending_tag_adds` / `self._pending_tag_removes`, each mapping
`analysis_num -> set(tag_nums)` of tags (reject **and** info) applied/removed
this session (recorded by `_record_pending_tag()`; multiple tags per injection
supported). A single-analyte Save clears both dicts — saved tags are final for
that analyte and can no longer be copied to the others.

## Save workflow

Save (`s` key or Save button) calls `upsert_mole_fractions(self.run)`.
Mole fractions must be recomputed before saving. For M4 CFC-113/113a,
run `m4_batch.py -p 32 -i` after any GUI edits for authoritative values.
IE3 calibration weeks save differently — see Update Method / Update MF /
Revert below, not this Save button.

## Tagging model (logos_data.py)

### Module-level tag constants

```python
_TAG_LAYOUT       # list of (section, [(letter, desc, r_tag, i_tag), ...])
_INFO_TAG_NUMS    # frozenset of all i_tag values from _TAG_LAYOUT (non-zero)
_INFO_TAG_DESCRIPTIONS  # {i_tag: desc} — same description as paired R tag
_USER_REMOVABLE_AUTO_TAGS  # frozenset — auto tags the user may manually remove
                           # currently {316} (M4 first-reference-run)
```

### MultiTagPanel

Floating panel showing R (reject) and I (informational) checkboxes for each
tag in `_TAG_LAYOUT`, grouped into Sampling, Measurement, and Automatic
sections. Opened via the **Multi-Tag** button or the `G` key cycle.

- **Auto tags** — checkboxes are read-only; state reflects DB but cannot be
  changed by the user.
- **`_USER_REMOVABLE_AUTO_TAGS`** — exception: the R checkbox is enabled only
  when the tag is already applied to *all* selected points (remove-only).
  Tag 316 (`qc_status` is set to `'F'` when applied, so it won't be
  reapplied by the batch loader after manual removal).
- **Save/Update Comment** button is disabled unless the selected point(s)
  carry at least one tag.
- **Selection gestures**: plain click selects one point; dragging a box
  selects a region (both replace the selection). Region select is **always
  active while the panel is open** — there is no Select Region toggle; the
  RectangleSelector is (re)built on panel open and after every `gc_plot`,
  keyed on panel visibility. **SHIFT+click** toggles a point in/out of the
  current selection; **SHIFT+drag** adds the box contents (union, never
  removes). SHIFT+click on empty space is a no-op (a plain empty-space click
  clears the selection). All shift paths funnel through
  `MainWindow._apply_multi_tag_selection()`, which dedupes, drops stale index
  labels, and re-highlights. SHIFT detection uses Qt keyboard modifiers
  (`_shift_held()`), not matplotlib key events (canvas focus is unreliable);
  the RectangleSelector is built with `state_modifier_keys=dict(square='')`
  to unbind matplotlib's default shift→square-box behavior.
- **Toolbar coexistence**: Pan/Zoom are no longer disabled by the tagging
  modes. An engaged nav tool holds the canvas widgetlock (selectors go
  dormant automatically), `_on_pick_point_inner` returns early while
  `toolbar.mode` is set, and `_on_click_tooltip` already guards likewise —
  so drags pan/zoom while a tool is engaged and go back to selecting when
  it's toggled off. `FastNavigationToolbar` wraps `canvas.set_cursor` to
  keep the tagging crosshair (the base toolbar resets to a pointer on every
  mouse move) and restores it in its `pan()`/`zoom()` overrides.

### Copy Tags to all Analytes

Tag clicks (MultiTagPanel checkboxes or G-mode `_toggle_flags`, reject and
info alike) write the tag to the DB for the **current analyte only** and
record the change in `_pending_tag_adds` / `_pending_tag_removes`. The
**Copy Tags to all Analytes** button (and the plot-legend "Save flags to all
gases" button) call `update_all_analytes()`, which:

1. Propagates pending adds **and removals** to every analyte's mole-fraction
   rows sharing each `analysis_num`, via
   `instrument.update_flags_all_analytes(adds, removes)` (core version for
   M4/FE3/BLD1; insitu override for IE3/CATS tables). Only injections present
   in the currently loaded window are propagated — pending entries from runs
   the user navigated away from are dropped with a console note, so tags are
   never copied without the matching recalc.
2. Reloads every analyte for the current window, recalculates mole fractions
   (fresh smoothing/normalization now excludes newly rejected points), and
   upserts `ng_mole_fractions` + `calibrations`.
3. Clears the pending dicts.

**Save vs. Copy-to-all are mutually exclusive intents**: a single-analyte
Save (`S` key, Save legend button, IE3 "Update Method") finalizes the tags
for that analyte and clears the pending dicts, so a later copy-to-all cannot
accidentally drag previously saved tags along. Use Copy-to-all *instead of*
Save when tags should reach every analyte (it saves all analytes itself,
including the current one). Pending state does survive analyte/run switching
— only Save and copy-to-all clear it.

### Informational tag overlay

"Show Info Tags" checkbox (Processing tab, below "Hide Rejected Data"):

- `_update_info_tagged()` populates `run['has_info_tag']` by querying the
  tag table for any `tag_num` in `_INFO_TAG_NUMS`.
  - **M4/FE3/BLD1**: uses `ng_mole_fraction_num` directly from the DataFrame.
  - **IE3**: uses `mf_num` (added to `ng_insitu_data_view` 2026-05-08); falls
    back to `analysis_num + parameter_num + channel` join if `mf_num` is absent.
- Called on initial data load and whenever the checkbox is toggled on.
- Plot overlay: hollow purple diamond (`marker='D'`, `edgecolors='mediumpurple'`,
  `facecolors='none'`), `zorder=3` — above normal markers, below rejection
  overlays.
- Tooltip shows **Info Tag: \<description\>** for informational tags using
  `_INFO_TAG_DESCRIPTIONS` (does not require the tag to be in `ccgg.tag_view`).

### Rejection overlay markers

Drawn in `_gc_plot_impl` when "Hide Rejected Data" is unchecked:

| Condition | Marker |
|---|---|
| Manual reject | Hollow circle, port-color edge, `zorder=4` |
| Auto-only reject | Hollow circle + `x` overlay, port-color edge, `zorder=5` |
| Informational tag | Hollow purple diamond, `zorder=3` |

`auto_rejected` is True when all of a point's reject tags are in
`AUTO_TAG_NUMS = {316, 26, 25, 2, 32, 324}`.

### Port legend toggle (declutter)

Clicking a port entry in the right-side sample legend (the marker or its
text) hides/shows all of that port's data in the GC plot — main scatter,
rejection overlays, and info-tag overlays together — like a per-port version
of "Hide Rejected Data". Used to uncover a point obscured by other ports.

- State lives in `self._hidden_ports` (a set of `port_idx`), honored by
  `_gc_plot_impl` when drawing. The plot fully rebuilds each `gc_plot()`, so
  this persistent set — not matplotlib `set_visible` — is what survives redraws.
- Hidden ports stay in the legend, dimmed (`alpha=0.35`), so they can be
  toggled back on. Their mean/std/count stats are still computed from full
  data (hiding is purely visual; it does not change other ports' stats).
- `_gc_plot_impl` builds `label_to_port` while making the legend handles;
  after `ax.legend()` it maps each legend Text + handle artist to its port in
  `self._port_legend_artists` and marks them pickable. `_on_legend_pick`
  checks that dict first (before its `Text`-only guard, since handle markers
  are `Line2D`), toggles `_hidden_ports`, preserves the current x/y view via
  `_pending_xlim/ylim`, and redraws.
- `_hidden_ports` is reset in `load_selected_run()` (fresh per loaded run) and
  pruned to `ports_in_run` on each draw. `_port_legend_artists` is cleared in
  `clear_plot()` and at the top of `calibration_plot()` so stale artist refs
  can't match a pick on a non-GC plot.

## Hide Panel (`w`)

The plot toolbar's leftmost button, **◀ Hide Panel** (shortcut `w`), hides
`left_container` (tabs and Processing controls) so the plot fills the window.
`_on_left_pane_toggled()` only allows hiding on the Processing tab, then
redraws via `_redraw_gc_plot_keep_view()` so the legend margin re-fits the new
width. The window-level QShortcuts (r/t/m, d, a, g, s, Ctrl+Shift+arrows) keep
working while the pane is hidden.

## "Additional" data panel (GC plot)

The plot toolbar (next to Chrom View) has an **Additional:** combo box. Picking a
column draws a small panel (1:4 height, `sharex`) above the Response/Ratio/Mole
Fraction plot showing that per-injection value in port colors (rejected points
hollow, or hidden with "Hide Rejected Data"; hidden legend ports hide too).

- Options come from the instrument's `ADDITIONAL_DATA_COLUMNS` class attribute
  (empty in `LOGOS_Instruments` → combo not shown). Currently only M4 sets it,
  to the `ng_data_processing_view` engineering columns (`init_p`, `net_pressure`,
  `cryocount`, `pfp_press1`, ...), which `load_data`'s `SELECT *` already loads.
  Add it to another instrument to enable it there. Columns missing from
  `self.run` are ignored.
- The main axes is created first so `self.figure.axes[0]` is still the main
  plot, and `figure.sca(ax)` keeps `gca()` on it (tagging, tooltips, highlights
  and y-lock depend on this). With the panel on, the title and sub_info banner
  move onto the panel.
- `_adjust_layout_for_legend()` widens the left margin when y tick labels would
  clip, and lines up the two y-labels.
- Keyboard: `d` steps to the next variable and `Shift+D` to the previous one,
  wrapping through "None" (panel off). Set up by
  `_setup_additional_data_shortcuts()` only when the combo has choices. (`a` is
  already the autoscale cycle.)
- The panel is display-only: no tagging or selection. Its scatters use
  `picker=False`, both RectangleSelectors attach to `axes[0]` only, and
  `_on_click_tooltip` ignores clicks in `self._ax_additional` so they can't
  clear a Multi-Tag selection.

## Timeseries figure datasets (M4)

The Mole Fractions figure's dataset legend toggles these. M4 (inst_num=192)
only runs from **2022**; M1 covers 1991-2009 and M3 2009-2023, so anything
M4-only shows the last four years of a 32-year record.

| Dataset | Source | Coverage |
|---|---|---|
| All samples | `ng_data_processing_view` (M4 injections) **+** `query_mstar_pair_data()` (M1/M3 pair means) | **1991-2026** |
| Flask mean / Pair mean | `ng_data_processing_view`, inst_num=192 | M4 only |
| 10-day mean / Monthly mean | `ng_pair_avg_view` via `_binned_inst_filter()` | **M1+M3+M4** |

**`All samples` spans the whole M-system**, carrying the finest per-sample data
available at each date: M4 injections from 2022, M1/M3 flask pair means before
that. M4 is the only one of the three with per-injection rows in
`ng_data_processing_view`, so before 2022 the pair mean *is* the raw datum. The
two halves are drawn with different markers (`o` vs `P`, the M* half dimmed to
0.75 brightness) because the granularity genuinely changes, but they share the
`_dataset_label` so **one legend click clears the entire raw record** — which is
the point, when you want to read the binned means. `_draw_mstar_artists()`
appends into the existing `All samples` handle list via `setdefault().extend()`.

There was briefly a separate `M* pair` legend entry. It meant toggling off
`All samples` still left the pre-2022 scatter on the plot, which is not what
anyone wants. Note M* pair means are conceptually closer to `Pair mean` than to
`All samples`; they sit here because they are the only raw data that exists
before 2022, not because they are per-injection.

- `_binned_inst_filter()` returns `inst_id IN ('M1','M3','M4')` for M4 and
  `inst_num = %s` for everything else, so the binned aggregates pool the whole
  M-system and match what the Export M* Data buttons write. FE3/IE3/CATS are
  unaffected.
- There used to be separate `Mstar 10-day mean` / `Mstar monthly mean` datasets
  (M1/M3 only) alongside M4-only `10-day mean` / `Monthly mean`. Toggling
  "Monthly mean" then showed nothing before 2022, which reads as a bug. They
  were merged; the two M*-only queries were deleted.
- Pooling M3 and M4 is safe: compared like for like (same site, same month) they
  agree to **-0.02 ppt mean, sd 0.37** over their 2022-2023 overlap. A naive
  pooled-across-sites comparison suggests a -7.6 ppt step, but that is a
  site-mix artifact -- early M4 has 1-4 pairs/month from different sites than
  M3's 13-22.
- Only 3 site-months have both M3 and M4 for a given analyte, so merging changes
  almost nothing structurally (1489 rows across the two old datasets -> 1486
  pooled). Four more site-months overlap M1/M3, but those were already pooled.
- **`All samples` stays M4-only** and can't be extended: M1/M3 have no
  per-injection rows in `ng_data_processing_view`, only pair averages in
  `ng_pair_avg_view`. Same reason the Relative Stddev figure and right-click
  navigation don't work for M*.

## Pale-yellow "staged or running" cue

`_PENDING_STYLE` / `set_button_pending()` in `logos_timeseries.py` are the one
definition of the `#f6e7a1` pale yellow, used wherever a button's action has
been asked for but not yet applied, or is running. A test asserts the literal
appears exactly once, so it stays that way.

- `_set_button_loading_state()` uses it for "Loading..." states. Its
  `loading_text` argument keeps the label where a button is too narrow for
  anything longer — the **Plot** buttons pass `loading_text="Plot"` and just
  change colour, restored in a `finally` so a failed preview can't strand them.
- `TimeseriesFigure._on_year_changed()` marks **Reload** when the year range is
  staged, since nothing re-queries there and the plot would otherwise disagree
  with the spinboxes silently. `_on_reload_clicked()` clears it via the loading
  state's own reset.
- `ExportPreviewFigure._mark_pending()` / `_clear_pending()` do the same for the
  preview window.

## Timeseries tab export buttons (SAVE group)

Instrument-specific; built in `TimeseriesWidget.__init__` and handled by
`_run_mstar_export()` / `_run_fecd_export()`. The M* buttons all share
`_run_mstar_export()`, which takes an `exporter_cls` argument.

| Button (M4) | Exporter | Output |
|---|---|---|
| All Sites and Time | `MstarDataExporter` | one file, per-pair rows, all sites |
| Selected Sites and Time | `MstarDataExporter` | same, checked sites + year range |
| Selected Sites, Times, Monthly Means | `MstarMonthlyExporter` | one file, one row per site per month |
| Global Means | `MstarGlobalMeansExporter` | one file, monthly global/hemispheric means + the site means behind them |

FE3 gets two fECD buttons (`FecdDataExporter`), which prompt for a directory
and write one file per site.

### Preview ("Plot") buttons

Each export row is `[Export] [Plot] [ⓘ]`. **Plot** opens an
`ExportPreviewFigure` showing exactly what that export would write — built by
`_preview_export()`, which constructs the exporter the same way the export
button does and plots its own `query_data()`, so the preview cannot drift from
the file.

The preview's toolbar carries a **year range**, an **analyte** combo,
**Reload** and **Save**, mirroring the main figure's toolbar.

- The **year range stages** (`_mark_pending()`) and **Reload** applies it.
  Walking the spinboxes otherwise fired one query per step — 12 steps = 12
  queries, seconds each for global means. Reload goes pale yellow while staged.
- An **analyte change reloads immediately**, as in the main figure: it's one
  discrete event rather than a run of them.
- **Ctrl+Shift+Up / Ctrl+Shift+Down** step the analyte, the same sequences the
  main figure uses, so they reload too.

Overrides go through `from_timeseries_widget(..., analyte=, start_year=,
end_year=)`, so the Timeseries tab's own selection is **left untouched**. An
all-time export has no range to pick, so its spinboxes are omitted
(`start_year is None`).

**Save applies a staged change first** (`_on_save` calls `reload()` when
`_pending`), so it can never write a different selection from the one drawn —
the plot, the footer and the file always agree.

`_preview_export()` takes the sites as a **callable**, not a list, so each
reload re-reads the site checkboxes rather than freezing them at open time.

**Save** calls `TimeseriesWidget.save_exporter()`, the same path the export
button uses, so saving from a preview and pressing Export are identical.
`save_exporter()` dispatches on the exporter's `WRITES_DIRECTORY` class
attribute — False writes one file via `export()`, True prompts for a directory
and calls `export_all()` (fECD only).

Each exporter describes its own preview through two hooks, keeping the shape
knowledge with the format rather than in the plotting code:

- `preview_series(df)` → `[{label, site, x, y, yerr, marker, …}]`. `site` lets
  the figure colour a series with `build_site_colors()`, matching the main
  timeseries figure; it is None for a series that isn't one site (Global/NH/SH).
  Optional keys: `colour`, `linestyle`, `linewidth`, `markersize`, `zorder`,
  `background` (drawn faint and behind).
- `preview_title()` → the window title and axes title.

The global-means preview draws the background sites faintly with the three
means as lines on top, so the means can be read against the data behind them.
`FecdDataExporter` gained a `query_data()` that concatenates its per-site
frames purely for this — its export still writes one file per site, and the
preview footer says so.

An empty selection warns and opens no window. Adding an export means adding
those two hooks and passing `on_plot=` to `_export_row()`.

`mstar_header.txt` holds the shared GML header for the first three; its
COLUMN DESCRIPTIONS block is a `{columns}` placeholder filled from
`mstar_columns_pairs.txt` or `mstar_columns_monthly.txt`. The standalone
`mstar-export.py` reads the same pair of files — update both readers if the
placeholder changes.

### Standard-deviation floor (`apply_sd_floor`)

Shared by the **monthly-means** and **global-means** exports, so a site-month's
sd reads the same in both (verified: 3,844/3,844 identical for HFC-134a).

A month of one or two flask pairs reports `STDDEV` near zero whenever those
pairs happen to agree — an artefact of sample size, not a tightly constrained
month. Median sd scales straight with n (n=2 → 0.198, n=3 → 0.285, n=4 → 0.395,
n=5 → 0.538), which is the tell.

- **Floor = mean of the usable monthly sds over the trailing 12 months** (the
  month itself plus the 11 before it), per site. `SD_FLOOR_WINDOW`/
  `SD_FLOOR_MIN_MONTHS` in `data_export.py`.
- **Only months with n ≤ 2 are floored** (`SD_FLOOR_MAX_N`). Tested, not
  assumed: at n=2 a month's sd barely predicts the next month's (lag-1
  correlation of log sd 0.20–0.41 across analytes) — it's sampling noise. At
  n ≥ 3 it does (0.32–0.73), so flooring would discard real information. It
  would also bias the record upward, since a floor raises and never lowers:
  flooring every month inflates the mean sd **21–33%** across analytes, versus
  **5–12%** confined to thin months. An earlier revision floored everything;
  55% of what it moved was n ≥ 3.
- **Averaging spreads, not recomputing an sd across the window**, is what keeps
  it trend-free. A recomputed 12-month sd is ~1.5 ppt for HFC-134a purely
  because the gas rises ~5 ppt/yr, and would raise 100% of months. The mean of
  spreads gives HFC-134a (+4.57/yr) 0.365 and CFC-11 (−1.72/yr) 0.358 — opposite
  trends, same floor. It tracks the measurement, not the signal.
- **Mean, not median** (user's choice). Mean sits above median on a right-skewed
  sd distribution, so the floor is slightly conservative: raises ~61% of months,
  median 1.11×, p75 1.79×, with 5.7% above 5× (the near-zero ones).
- **Single-pair months are excluded from the window average** — a lone pair has
  no spread, and counting its zero would drag the floor toward zero, defeating
  the point. They still *receive* the floor, which is the only sd available to
  them. Months with no data keep NaN; a window under `SD_FLOOR_MIN_MONTHS`
  yields no floor and the measured value passes through untouched.
- The rolling window is **calendar-based**: each site is reindexed to continuous
  months internally so a gap costs a month rather than being skipped. Rows are
  never added — the result carries exactly the input's rows.
- **Watch the NaN-floor case.** `sd.where(sd >= floor, floor)` looks right but
  silently NaNs the measurement when the floor is NaN, since any comparison with
  NaN is False. It is written `sd.where(~(sd < floor), floor)` for that reason.
- Effect on the global product: the mean is **unchanged** (verified — the floor
  writes only `sd`/`sd_floor`, leaving `mf` and `n` identical on key). Against an
  unfloored `Global_sd` median of 0.233, flooring n ≤ 2 gives 0.239 and flooring
  everything gave 0.282. The global path no longer uses the old
  `AVG(pair_stdv)` fallback for n=1.
- A caution when comparing rules: `apply_sd_floor` sorts by site, so positional
  comparisons of the in/out frames misalign. Merge on `(site, date)`.

### PFP pseudo-sites in the M* exports

PFP (programmable flask package) samples are a **different kind of flask**: they
carry a `ccgg_event_num` instead of a `pair_id_num`. `ng_pair_avg_view` admits
them through the `ccgg_event_num > 0` branch of its `WHERE` clause — they are
*not* missing from the view, contrary to what older comments said — but files
them under the **base site**, so raw `MLO` rows blend programmatic HatsFlask
pairs with PFP pairs.

The view exposes no `run_type_num`, so the pseudo-site the Timeseries plot shows
(`TimeseriesWidget.query_data()` relabels `run_type_num = 5`) cannot be rebuilt
the same way in the exporters. `_site_label_sql()` in `data_export.py` keys off
`pair_id_num = 0` instead. That is sound rather than empirical: the view only
admits a row when `pair_id_num > 0 OR ccgg_event_num > 0`, so `pair_id_num = 0`
*inside the view* necessarily means a CCGG event flask.

- Scoped to `_PFP_SITES = {'MLO_PFP': 'MLO', 'MKO_PFP': 'MKO'}` — only MLO and
  MKO carry such rows, on M3/M4 only. OTTO/FE3 have none, so the fECD exporter
  is unaffected.
- All three M* exports split them. `_base_sites()` expands pseudo-sites to the
  base site for the `WHERE`, then a subquery filters on the relabelled
  `export_site` (MySQL can't reference a SELECT alias from `WHERE`).
- The split is **conservative**: relabelling only. Verified row-for-row —
  `mlo` 1903 → `mlo` 1243 + `mlo_pfp` 660, `mko` 467 → `mko_pfp` 467, all 18
  other sites byte-identical, 16177 rows in and out.
- MKO is 100% PFP, which is why `MSTAR_EXPORT_EXCLUDE` holds `MKO_PFP`; after
  the split plain `MKO` has no rows at all.

**Why it matters:** PFP runs ~0.45 ppt below programmatic flask at MLO (median
−0.50, sd 0.74 for HFC-134a) and outnumbers it ~3:1, so a blended monthly mean
is pulled toward PFP. Splitting shifts the global mean by −0.077 ppt on average
(max 0.21, 0.06% of signal) and moves agreement with the published GML product
slightly *closer* (HFC-134a 0.021 → 0.015 ppt).

**History:** PFPs were deployed at MLO in 2021 and became the only sampling
there after the Nov 2022 Mauna Loa eruption cut power and access to the
observatory — no programmatic flask analyses at all in 2023–2024. They resumed
in 2025, so the two programmes run concurrently again.

**Open question:** a pseudo-site inherits its base site's latitude, so during
the concurrent stretches (2021–22, 2025 onward) the MLO location contributes two
weighted values to the LN band. Whether to instead prefer the flask value and
drop `mlo_pfp` where both exist is undecided — see the note in
`gml_global_means_config.yaml`. Nothing implements it yet.

### Global Means export

The math follows <https://github.com/duttong/GML_means>
(`gml_annualmeans.py`'s `semi_hemispheric_means`), but runs off
`ng_pair_avg_view` instead of the published GML website files and stops at
monthly resolution:

- Sites are bucketed into `HN / LN / LS / HS` at ±`phi` (30°) and averaged with
  `cos(lat)` weights; `Global` is the unweighted mean of the four bands, NaN
  unless all four are present that month.
- `weight_lat_overrides` move SPO to −65 (and PSA to −80 for the gases listed
  in `gas_weight_lat_overrides`) **for the weights only** — Steve's method.
  The data and `gmd.site` are untouched.
- `_sd` columns propagate the site standard deviations through the same
  weights (`var = Σ (wᵢ/Σw)² sdᵢ²`), which GML_means does not report. A site
  month whose own sd is unknown contributes 0.
- A site's monthly sd is the spread of its pair means, falling back to the
  single pair's own `pair_stdv` when `n = 1` (a one-pair month has no spread).
- Interior gaps in a site's monthly series are filled when
  `interpolate_site_gaps` is on; those months carry `n = 0`. Leading/trailing
  gaps are never extrapolated.
- **`interpolation_method` (`seasonal`)** fits additive Holt–Winters per site —
  level, trend, 12-month seasonal term (`statsmodels ExponentialSmoothing`) — and
  takes the model value *only* where an observation is missing; observed months
  are never replaced. This is what the GML_means loader uses for MSD gases (only
  its `oldgc` program uses linear). Falls back to linear for a series under
  `MIN_SEASONAL_POINTS` (24, two cycles) or a fit that won't converge. `sd` is
  always time-interpolated, never modelled — a seasonal cycle in the mole
  fraction says nothing about a month's pair scatter. Costs ~3s vs ~1s per
  export. Set to `linear` to revert without code changes.
- **`max_interpolation_months` (3) caps the bridge.** A longer run of missing
  months is left empty rather than straight-lined, all-or-nothing per run — note
  pandas' own `limit` would instead fill the first N months of a longer gap.
  Without the cap, MLO's 33-month eruption outage (2022-12 → 2025-08) was filled
  with a linear ramp biased up to 3.4 ppt above the real co-located `mlo_pfp`
  measurement, and then weighted in LN *alongside* it — double-counting MLO with
  one value fabricated. Capping costs nothing in coverage (Global still 382/391
  months for HFC-134a) because the surrogate covers the outage. Every run the cap
  suppresses is long: 5, 6, 7, 13, 16, 25, 33, 42 months; short operational gaps
  are unaffected.
- Note `MstarMonthlyExporter` does **not** interpolate — its gaps are written as
  `nan` with `n = 0`. Only the global-means path fills.
- **Divergence from upstream:** the seasonal fit is now ported, so the only
  remaining difference is that GML_means fills arbitrarily long gaps while the cap
  here refuses them. Agreement with the published product across the three
  configurations (mean |diff|, ppt):

  | analyte | linear, uncapped | linear + cap | seasonal + cap |
  |---|---|---|---|
  | HFC-134a | 0.015 | 0.018 | 0.017 |
  | HCFC-22 | 0.046 | 0.061 | 0.052 |
  | CH3Br | 0.027 | 0.030 | 0.027 |
  | CFC-11 | 0.437 | 0.442 | 0.423 |
  | HFC-152a | 0.035 | 0.072 | 0.071 |

  The seasonal fit recovers most of what the cap cost; HFC-152a's residual is the
  cap refusing long holes (kum has a 42-month gap), which no method choice
  recovers. Neither change touches the mlo/mlo_pfp double-weighting, which
  upstream has too and which remains the open question above.
- Seasonal vs linear changes filled site-months by mean 0.006 ppt (max 2.07 where
  a chord cut across a seasonal turn) and the global mean by mean 0.007 ppt
  (max 0.18). Coverage is identical either way.
- **The site checkboxes are ignored** — the site list comes from
  `background_sites` / `gas_background_overrides` in the yaml, so the file
  always contains every site feeding the means. `ush` is in that list but has no
  M* data; it's dropped and named in the file header. (`mlo_pfp` also used to
  be dropped, until the PFP split above gave it its own rows.)
- Config keys use GML gas names (`HFC134a`); analytes are display names
  (`HFC-134a`). `GlobalMeansConfig.gas_key()` matches by stripping hyphens,
  with `analyte_aliases` for the rest (`PCE` → `C2Cl4`).
- Verified against the published `GML_jul_annual_means.csv` by rolling the
  monthly output up to calendar-year means: MSD-sourced gases agree to
  0.03–0.8%. Gases in `combined_source_gases` (CFC11/12/113, CCl4, SF6) are
  published from blended fECD+MSD data, so an M*-only file won't match them
  exactly — the header says so.

## IE3 Calibration view (`_ie3_cal_plot`)

Shown when the Calibration radio is selected and the loaded run is a weekly
cal week (`current_run_time` contains `'(Cal)'`). Plots normalized_resp
(weekly mean) vs. assigned mole fraction for the cal2/ref/cal1 tanks, the fit
line for the selected method, and the fit's predicted value at the ref
tank's response (crimson diamond).

- **Assigned-value error bars**: y-error bars on the cal2/ref/cal1 points use
  `unc_c0` from `hats.scale_assignments_view` (`_ie3_tank_unc()` for cal
  tanks, `ref_tank_unc_c0()` for the ref tank).
- **Predicted-point error bar**: the diamond's y-error bar comes from
  `_ie3_ref_pred_unc()`, which propagates the cal tank(s)' `unc_c0` through
  the fit — weighted linear interpolation for the two-tank fit, or
  `|x|·unc_slope` for the single-tank-through-origin methods (cal1/cal2).
- **Click tooltips**: left-click a point for `Assigned: val ± unc` (cal2/
  ref/cal1) or `Predicted: val ± unc` + `Diff from assigned: val ± unc` (the
  diamond). Routed through `_is_ie3_cal_plot_active()` inside
  `_on_click_tooltip` — this plot's markers are Line2D artists (from
  `errorbar`/`plot`), not the PathCollection scatter the main gc-plot tooltip
  logic expects, so they're handled separately via `_ie3_cal_tooltip_click()`.
  `self._ie3_cal_tooltip_points` holds `{'artist', 'lines': [{'title', 'val',
  'unc'}, ...]}` per point, reset at the top of every `_ie3_cal_plot()` call.
- **`hats.scale_assignments.coef1` (drift) is not applied anywhere in
  logos_data.** `ref_tank_coef0()`/`ref_tank_unc_c0()` and `cal_tank_coefs()`
  (in `ie3_cal_test.py`, imported by `logos_data.py`) only read `coef0`/
  `unc_c0` and silently ignore `coef1`. Contrast with the M4/FE3/Perseus path
  (`populate_cal_mf()`), which detects a non-zero `coef1` and raises a
  `RuntimeWarning` but still uses flat `coef0` — no path applies drift yet.
- **Don't confuse with `hats.ng_response`**: that table stores the weekly
  cal-fit's own `coef0`/`coef1` (intercept/slope of the fit line, written by
  `upsert_ng_response()` / `ie3_batch.py`) — a different quantity from
  `scale_assignments.coef0`/`coef1` (tank assigned value / drift), despite
  the same column names. Both flow through `_ie3_cal_plot`.

### Update Method / Update MF / Revert buttons

Below the Calibration view, IE3-only, visible only for cal weeks. State is
managed together in `_refresh_ie3_update_button()`, called after any change
to `self.madechanges` / `self._ie3_mf_dirty`:

- **Update Method** (yellow when `self.madechanges`): saves the fit-method
  combo selection to `hats.ng_response` and recomputes the week's fit. Sets
  `_ie3_mf_dirty` and flips the button to...
- **Update MF** (lightgreen, when `self._ie3_mf_dirty`): recomputes and
  upserts the week's air-port mole fractions from the saved fit (in-GUI
  equivalent of `ie3_batch.py -i` for this one week).
- **Revert** (light red `#ffcdd2`, enabled only alongside "Update Method"):
  discards unsaved local edits (fit-method selection, rejection toggles) by
  calling `load_selected_run()` again and re-rendering the active plot — no
  DB writes. Lets you try a method/rejection change and see the resulting
  fit without committing it. Disabled once past "Update Method" — a change
  already saved to `hats.ng_response` can't be undone by a local reload.
