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
