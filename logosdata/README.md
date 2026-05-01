# logosdata

Interactive GUI for reviewing and processing HATS GC instrument data.
Connects to the HATS next-generation MySQL database and supports M4, FE3,
IE3, BLD1, and Perseus tank views.

## Requirements

- Python 3 in the `prod6` conda environment
- PyQt5, matplotlib, pandas
- A display (`ssh -Y` or local session with `$DISPLAY` set)
- Access to the HATS ng database via `/ccg/src/db/db_utils/db_conn.py`

## Usage

Run via the `logos_data` launcher script in the itxbin root:

```bash
logos_data <instrument> [--site SITE]
```

**Instruments:** `m4`, `fe3`, `bld1`, `ie3`

**Examples:**
```bash
logos_data m4
logos_data fe3
logos_data ie3 --site smo
logos_data bld1
logos_data          # uses preferred instrument from ~/.logos_data_user.conf
```

On first run without an argument, you will be prompted to choose a preferred
instrument. The choice is saved to `~/.logos_data_user.conf` and used for
subsequent no-argument calls.

Run the tank browser directly via the `logos_tanks` launcher script in the
itxbin root:

```bash
logos_tanks [instrument]
```

**Tank instruments:** `m4`, `fe3`, `bld1`, `prs`

**Examples:**
```bash
logos_tanks prs
logos_tanks m4
logos_tanks          # uses default_inst from ~/.logos-tanks.conf
```

`m4` is the M-system tank/calibration view. It uses the M4 analyte list and
queries `hats.calibrations` for M1, M3, and M4 calibration records so older
tank history appears when the date range reaches back before M4.

`prs` is the combined Perseus tank/calibration view. It uses PR1 analytes and
queries `hats.calibrations` for both PR1 and PR2 calibration records.
On first no-argument run, `logos_tanks` prompts for a default instrument and
saves it as `default_inst` in `~/.logos-tanks.conf`.

## Tabs

| Tab | Instruments | Description |
|---|---|---|
| Processing | all | Date range, run/analyte selection, response/ratio/mole-fraction plots, flagging |
| Timeseries | all | Long-term mole fraction trends with site overlays and IE3 in-situ data |
| Tanks | m4, fe3, ie3, bld1, prs | Reference tank history and tank assignment |
| LOGOS AI | m4, fe3, ie3 | Chat interface for querying HATS data |

Tab visibility is controlled per instrument in `logos_data.conf`.

## Keyboard shortcuts

| Key | Action |
|---|---|
| `r` / `t` / `m` | Response / Ratio / Mole Fraction plot |
| `Ctrl+Shift+←/→` | Previous / Next run |
| `Ctrl+Shift+↑/↓` | Previous / Next analyte |
| `p` / `l` | Point-to-point / Lowess smoothing |
| `a` | Cycle autoscale mode (Samples → Standard → Fullscale) |
| `s` | Save current gas results to database |
| `n` | Edit / view run notes |
| `g` | Toggle flagging/tagging mode |

## Configuration

### logos_data.conf
Author-managed UI settings. Lives in this directory. Controls which tabs are
visible, the startup analyte, and whether instrument-specific controls
(CSV export, change-run-type) appear. Edit this file to change UI behaviour
for all users.

### ~/.logos_data_user.conf
Per-user settings. Created automatically on first run. Example:

```ini
[user]
preferred_instrument = fe3
```

### ~/.logos-tanks.conf
Per-user tank browser settings. JSON file created by `logos_tanks` and the
Tanks tab. It stores the direct-launch default instrument plus saved tank sets.

```json
{
  "default_inst": "prs",
  "sets_by_analyte": {}
}
```

Tank calibration queries read `hats.calibrations` and exclude sentinel
mixing ratios with `mixratio <= -99`.

## Files

| File | Purpose |
|---|---|
| `logos_data.py` | Main application — `MainWindow`, all tab and plot logic |
| `logos_instruments.py` | Instrument classes — DB queries, mole fraction calculation |
| `logos_timeseries.py` | `TimeseriesWidget` and `TimeseriesFigure` |
| `logos_tanks.py` | `TanksWidget` — tank history UI |
| `../logos_tanks` | Launcher for the standalone tank browser |
| `logos_ai_agent.py` | `LOGOSChatAgent` — Anthropic-backed chat agent |
| `logos_agent_tools.py` | `LOGOSDataAgentTools` — read-only query helpers for the agent |
| `logos_data.conf` | Author-managed UI configuration |
