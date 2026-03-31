# ng_engplot — Engineering Data Viewer

A PyQt5 GUI for viewing instrument housekeeping (engineering) data as
interactive timeseries plots. One tabbed window covers all three instruments.

## Requirements

- Python 3.10+, conda env `prod6`
- PyQt5, matplotlib, pandas
- A display: run locally or via `ssh -Y`

## Launch

```bash
cd ng_engplot
python3 engplot_main.py
```

## Instruments / tabs

| Tab | Instrument | Data root |
|-----|-----------|-----------|
| IE3 | Flask ECD (sites: smo, mlo, spo, brw) | `/hats/gc/<site>/` |
| FE3 | Flask ECD | `/hats/gc/fe3/` |
| BLD1 | Boulder 1 | `/hats/gc/bld1/` |

Tabs load lazily: only the selected tab loads on startup; switching to a new
tab loads it automatically on first visit.

## UI controls

| Control | Description |
|---------|-------------|
| **Most Recent** | Jump the end-date picker to the latest available data |
| **End date** | Calendar date picker for the last day of the loaded window |
| **Last N days** | How many days to load (auto-sets resample to `1min` at ≥ 3 days) |
| **Directory** | Filter to a single run directory, or show all in the date window |
| **Resample** | Time-averaging interval: `1s`, `10s`, `1min`, `5min`, `10min` |
| **Load** | Fetch and plot data for the current settings |
| **Left / Right trace** | Choose which engineering column to plot on each y-axis |
| **Markers** | Toggle circle markers on/off for all traces (off by default) |

## Plot features

- **Dual y-axis**: left trace in blue, optional right trace in red (independent scales)
- **Legend stats**: `mean ± std` computed over the visible x-range; updates live on pan/zoom
- **Dynamic x-axis labels**: format adapts to zoom level (days → `MM-DD`, hours → `MM-DD HH:MM`, seconds → `MM-DD HH:MM:SS`)
- **Home button**: autoscales both axes to the full loaded range; also triggered automatically when switching the directory filter
- **Markers toggle**: add/remove circle markers on existing traces without reloading

## State persistence

All UI selections (tab, date, N days, resample, trace choices, directory,
site for IE3) are saved to `~/.engplot.json` and restored on next launch.
