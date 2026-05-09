# CLAUDE.md — logosdata package

This package contains the logos_data GUI application and all supporting
modules. See the parent `itxbin/CLAUDE.md` for database tables, instrument
numbers, and compound parameter numbers.

## Package layout

```
logosdata/
  logos_data.py        # Main PyQt5 GUI — MainWindow, tabs, plotting
  logos_instruments.py # Instrument class hierarchy and DB query methods
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
| `_pending_tag_num` | nullable int | tag toggle helpers |

## Save workflow

Save (`s` key or Save button) calls `upsert_mole_fractions(self.run)`.
Mole fractions must be recomputed before saving. For M4 CFC-113/113a,
run `m4_batch.py -p 32 -i` after any GUI edits for authoritative values.

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
