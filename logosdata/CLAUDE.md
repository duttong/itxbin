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

## Save workflow

Save (`s` key or Save button) calls `upsert_mole_fractions(self.run)`.
Mole fractions must be recomputed before saving. For M4 CFC-113/113a,
run `m4_batch.py -p 32 -i` after any GUI edits for authoritative values.
