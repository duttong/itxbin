# CLAUDE.md — ng_engplot

## Purpose

PyQt5 GUI for viewing instrument engineering (housekeeping) data as timeseries.
Launched via `engplot_main.py` — a tabbed window with one tab per instrument.
Requires a display (`$DISPLAY`); run with `ssh -Y` or locally.

UI state (last tab, last date range, trace selections, resample, site) is
persisted to `~/.engplot.json` keyed by `config_key`.

## Class hierarchy

```
EngPlotWidget  (engplot_base.py)   — shared UI, plotting, state persistence
  ├── IE3EngWidget  (ie3eng.py)    — site-selectable; reads /hats/gc/<site>/
  ├── FE3EngWidget  (fe3eng.py)    — reads /hats/gc/fe3/
  └── BLD1EngWidget (bld1eng.py)   — reads /hats/gc/bld1/
```

Each subclass implements three abstract methods:
- `scan_date_range()` → `(YYYYMMDD_min, YYYYMMDD_max)` — sets date picker bounds
- `get_columns()` → `list[str]` — populates trace dropdowns on startup
- `load_data(end_date, n_days, resample)` → `pd.DataFrame | None` — indexed by time

Optional hooks for instrument-specific controls and state:
`_build_extra_controls`, `_build_right_controls`, `_extra_save_state`,
`_extra_restore_state`, `_connect_extra_signals`

## Data roots and file formats

| Instrument | Root | File pattern | Time column |
|---|---|---|---|
| IE3 | `/hats/gc/<site>/` | `<yy>/incoming/<YYYYMMDD>/eng*.csv[.gz]` | `ie3_time` (UTC string) |
| FE3 | `/hats/gc/fe3/` | `<yy>/incoming/<YYYYMMDD-HHMMSS>/eng_<N>.csv[.gz]` | `fe3_time` (UTC string) |
| BLD1 | `/hats/gc/bld1/` | `<yy>/incoming/<YYYYMMDD-HHMMSS>/<YYYYbldDDDHHMM>.<N>.eng` | derived from `Tsec*100` (local Denver time → UTC) |

IE3 column names: files from 2026-04-01 onward have an embedded CSV header row
(first field `ie3_time`); earlier files fall back to the legacy
`eng_header_smo_early.txt` in the site root.

## IE3 specifics

- Valid sites: `smo`, `mlo`, `spo`, `brw`
- `GSV1/2/3` columns are valve-state categoricals — converted to integer codes for plotting
- "Sample Flow Report" button shows `flow_samp` stats per SSV0 port, excluding
  the first 5 s after each port transition
- Port descriptions queried from `hats.ng_port_info` via `HATS_DB_Functions`

## BLD1 specifics

- Filename pattern: `YYYYbldDDDHHMM.NN.eng`; time derived from `Tsec*100`
  (centiseconds since local Denver midnight) converted to UTC
- Files have 2 skip rows before the header

## Plotting

- Dual-axis: left trace (blue, ax1) and optional right trace (red, twinx ax2)
- Legend labels show `mean ± std` of the visible x-range, updating on pan/zoom
- Home button autoscales to full loaded data (custom `_EngToolbar`)
- Auto-resample: switches to `1min` when n_days ≥ 3, else `1s`
- X-axis uses `AutoDateLocator` + `AutoDateFormatter` with custom `scaled` dict:
  days→`MM-DD`, hours→`MM-DD HH:MM`, seconds→`MM-DD HH:MM:SS`
- Markers toggle (QPushButton, checkable): updates existing line objects via
  `line.set_marker()` — no data reload needed; state stored in `_plot_lines`

## Lazy loading

- Only the active tab loads on startup (`QTimer.singleShot` in `EngPlotWindow`)
- `_has_loaded` flag on each widget; `_load_tab_if_needed` fires on `currentChanged`
- Directory filter change calls `_load_and_plot_home` (load + toolbar.home())
