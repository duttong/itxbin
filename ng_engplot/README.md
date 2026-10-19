# ng_engplot — Engineering Data Viewer

A PyQt5 GUI for viewing instrument housekeeping (engineering) data as
interactive timeseries plots. One tabbed window covers all three instruments.

## Requirements

- Python 3.10+, conda env `prod6`
- PyQt5, matplotlib, pandas
- A display: run locally or via `ssh -Y`

## Launch

```bash
engplot
engplot --end-date 2026-08-09
engplot --end-date 2026-08-09 --y1 water_trap_1_temp --y2 water_trap_2_temp
```

`--end-date YYYY-MM-DD` sets the **End date** field on every instrument tab
before the first data load. The selected date is the last day included in the
**Last N days** window. Dates outside a tab's available range are clamped to its
nearest available date, with a message on stderr. Without this option, startup
continues to use the latest available date.

`--y1 VARIABLE` and `--y2 VARIABLE` select the left and right traces using exact
column names. Use `--y2 None` to hide the right trace. Each option overrides the
corresponding saved selection on each tab's first successful data load. If a
variable is unavailable in that tab's loaded data, the current selection is kept
and a message is printed on stderr. Omitted options keep the saved selections.

The `engplot` script at `/hats/gc/itxbin/engplot` is the preferred entry point.
It uses the `prod6` conda interpreter directly and adds `ng_engplot/` to the
path automatically. No `cd` or manual `python3` invocation needed.

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
| **Lock Y axes** | Give both y-axes the same scale, using the combined min/max of the traces in the visible time range, with 5% padding |

## Plot features

- **Dual y-axis**: left trace in blue, optional right trace in red; independent scales by default, or a shared scale with **Lock Y axes**. While locked, pan/zoom keeps the y-axes aligned, and changing the time range rescales to the visible data. Turning the lock off restores independent autoscaling.
- **Legend stats**: `mean ± std` computed over the visible x-range; updates live on pan/zoom
- **Dynamic x-axis labels**: format adapts to zoom level (days → `MM-DD`, hours → `MM-DD HH:MM`, seconds → `MM-DD HH:MM:SS`)
- **Home button**: autoscales both axes to the full loaded range; also triggered automatically when switching the directory filter
- **Markers toggle**: add/remove circle markers on existing traces without reloading

## State persistence

All UI selections (tab, date, N days, resample, trace choices, directory, y-axis lock,
site for IE3) are saved to `~/.engplot.json` and restored on next launch.
