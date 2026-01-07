#! /usr/bin/env python

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip
)
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt

from matplotlib.widgets import Button
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import colorsys
import time
import sys


LOGOS_sites = ['SUM', 'PSA', 'SPO', 'SMO', 'AMY', 'MKO', 'ALT', 'CGO', 'NWR',
            'LEF', 'BRW', 'RPB', 'KUM', 'MLO', 'WIS', 'THD', 'MHD', 'HFM',
            'BLD', 'MKO']


class TanksPlotter:
    
    
    def __init__(self, db, inst_num):
        self.db = db
        self.inst_num = inst_num
        
        
    def return_active_tanks(self, start_year, end_year, parameter_num=None, channel=None):
        """
        Return a list of tank records (serial + latest fill metadata) active in the window.
        """
        if parameter_num is None:
            return []

        start_ts = f"{start_year}-01-01"
        end_ts = f"{end_year + 1}-01-01"  # half-open interval on year boundary

        # Query 1: find the latest fill_idx per tank observed in the window for this instrument/parameter/channel.
        fills_sql = f"""
            SELECT
                f.idx
            FROM (
                SELECT DISTINCT
                    a.tank_serial_num
                FROM hats.ng_analysis a
                JOIN hats.ng_mole_fractions mf
                  ON mf.analysis_num = a.num
                WHERE a.inst_num = {self.inst_num}
                  AND mf.parameter_num = {parameter_num}
                  {f"AND mf.channel = '{channel}'" if channel else ""}
                  AND a.tank_serial_num IS NOT NULL
                  AND a.run_time >= '{start_ts}'
                  AND a.run_time <  '{end_ts}'
            ) AS t
            LEFT JOIN (
                SELECT
                    serial_number,
                    MAX(`date`) AS max_date
                FROM reftank.fill
                GROUP BY serial_number
            ) AS latest
              ON latest.serial_number = t.tank_serial_num
            LEFT JOIN reftank.fill AS f
              ON f.serial_number = latest.serial_number
             AND f.`date` = latest.max_date
            WHERE f.idx IS NOT NULL
            ORDER BY
                t.tank_serial_num;
        """

        fills_df = (
            self.db.query_to_df(fills_sql)
            if hasattr(self.db, "query_to_df")
            else pd.DataFrame(self.db.doquery(fills_sql))
        )
        fill_ids = [str(idx) for idx in fills_df["idx"].dropna().unique()] if not fills_df.empty else []
        if not fill_ids:
            return []

        fill_list = ",".join(fill_ids)

        # Query 2: fetch metadata for those fill_idx values.
        sql = f"""
            SELECT
                h.num,
                h.fill_idx,
                f.serial_number,
                f.`date`,
                f.code,
                f.location,
                u.abbr        AS use_short,
                g.species,
                g.species_num AS parameter_num,
                g.mf_value,
                h.site_num,
                h.start,
                h.end,
                h.comment,
                f.notes AS fill_notes,
                g.notes AS grav_notes
            FROM hats.ng_tank_use_history h
            JOIN reftank.fill f
              ON f.idx = h.fill_idx
            LEFT JOIN hats.ng_tank_uses u
              ON u.num = h.ng_tank_uses_num
            LEFT JOIN reftank.gravs_corrected g
              ON g.fill_num = h.fill_idx
            WHERE h.fill_idx IN ({fill_list})
            ORDER BY f.serial_number, f.`date` DESC;
        """

        if hasattr(self.db, "query_to_df"):
            df = self.db.query_to_df(sql)
        else:
            df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            return []

        # Only keep Grav rows matching the selected parameter; keep Archive/other uses regardless.
        if "use_short" in df.columns and "parameter_num" in df.columns:
            use_lower = df["use_short"].fillna("").str.lower()
            is_grav = use_lower.str.startswith("grav")
            param_series = pd.to_numeric(df["parameter_num"], errors="coerce")
            grav_match = param_series == pd.to_numeric(parameter_num)
            grav_total = int(is_grav.sum())
            grav_kept = int((is_grav & grav_match).sum())
            grav_dropped = grav_total - grav_kept
            non_grav = ~is_grav
            df = df[non_grav | grav_match]

        if "date" in df.columns:
            df = df.rename(columns={"date": "fill_date"})
        # Prefer Grav rows (matching parameter) for a serial when present; otherwise take latest.
        if "serial_number" in df.columns:
            df = df.copy()
            df["__is_grav"] = df["use_short"].fillna("").str.lower() == "grav"
            df = df.sort_values(
                ["serial_number", "__is_grav", "fill_date"],
                ascending=[True, False, False],
            )
            df = df.drop_duplicates(subset=["serial_number"], keep="first")
            df = df.drop(columns=["__is_grav"], errors="ignore")

        return df.to_dict(orient="records")
    

class TanksWidget(QWidget):
    """
    Simple control pane for selecting a year range + gas and showing tanks
    as a 3-column grid of checkboxes.
    """
    def __init__(self, instrument=None, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.tanks_plotter = TanksPlotter(self.instrument.db, self.instrument.inst_num) if self.instrument else None
        self.tank_checks: list[QCheckBox] = []
        self.analyte_checks: list[QCheckBox] = []
        self._ready = False
        self._reload_dirty = False

        # --- Layout scaffold (mirror logos_timeseries style) ---
        controls = QVBoxLayout()

        # Year range selection
        i_start = int(self.instrument.start_date[0:4]) if self.instrument else 2020
        i_end = pd.Timestamp.now().year
        year_group = QGroupBox("Year Range")
        year_layout = QHBoxLayout()
        self.start_year = QSpinBox()
        self.start_year.setRange(i_start, i_end)
        self.start_year.setValue(max(i_start, i_end - 2))
        self.end_year = QSpinBox()
        self.end_year.setRange(i_start, i_end)
        self.end_year.setValue(i_end)
        self.reload_btn = QPushButton("Reload Tanks")
        self.reload_btn.clicked.connect(self._on_reload)
        year_layout.addWidget(QLabel("Start"))
        year_layout.addWidget(self.start_year)
        year_layout.addWidget(QLabel("End"))
        year_layout.addWidget(self.end_year)
        year_layout.addWidget(self.reload_btn)
        year_group.setLayout(year_layout)
        controls.addWidget(year_group)

        # Analyte selector
        analyte_group = QGroupBox("Gas / Parameter")
        analyte_layout = QVBoxLayout()
        analyte_container = QWidget()
        analyte_container.setStyleSheet("background-color: #fffbe6; border: 1px solid #f2e6b3;")
        analyte_checks_layout = QGridLayout()
        analyte_checks_layout.setContentsMargins(6, 6, 6, 6)
        analyte_checks_layout.setHorizontalSpacing(8)
        analyte_checks_layout.setVerticalSpacing(4)
        cols_analyte = 5
        for idx, name in enumerate((self.instrument.analytes or {}).keys() if self.instrument else []):
            cb = QCheckBox(name)
            if idx == 0:
                cb.setChecked(True)
            cb.toggled.connect(lambda checked, cb=cb: self._on_analyte_toggled(cb, checked))
            self.analyte_checks.append(cb)
            row, col = divmod(idx, cols_analyte)
            analyte_checks_layout.addWidget(cb, row, col)
        analyte_container.setLayout(analyte_checks_layout)
        analyte_layout.addWidget(analyte_container)
        self.tanks_status = QLabel("Select a year range and gas to load tanks.")
        self.tanks_status.setWordWrap(True)
        self.tank_grid = QGridLayout()
        self.tank_grid.setContentsMargins(0, 0, 0, 0)
        self.tank_grid.setHorizontalSpacing(8)
        self.tank_grid.setVerticalSpacing(4)
        analyte_layout.addWidget(self.tanks_status)
        analyte_layout.addLayout(self.tank_grid)
        analyte_group.setLayout(analyte_layout)
        controls.addWidget(analyte_group)

        controls.addStretch()
        self.setLayout(controls)

        # Wire date change after widgets exist
        self.start_year.valueChanged.connect(self._mark_reload_needed)
        self.end_year.valueChanged.connect(self._mark_reload_needed)

        # Populate tanks initially if any analyte starts checked
        self._ready = True
        if any(cb.isChecked() for cb in self.analyte_checks):
            self.refresh_tanks()

    # --- Slots / helpers ---
    def _mark_reload_needed(self):
        """Mark that dates changed and a reload is needed."""
        if not getattr(self, "_ready", False):
            return
        if self._reload_dirty:
            return
        self._reload_dirty = True
        self.reload_btn.setStyleSheet("background-color: 'lightgreen';")  # light green

    def _on_reload(self):
        """Reload tanks and clear the pending visual cue."""
        self.refresh_tanks()
        self._reload_dirty = False
        self.reload_btn.setStyleSheet("")

    def _on_analyte_toggled(self, cb: QCheckBox, checked: bool):
        """Keep analyte selection single-choice and refresh tanks when changed."""
        if not getattr(self, "_ready", False):
            return
        if checked:
            for other in self.analyte_checks:
                if other is cb:
                    continue
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
            self.refresh_tanks()
        else:
            if not any(c.isChecked() for c in self.analyte_checks):
                self.refresh_tanks()

    def _selected_analytes(self) -> list[tuple[str, int, str | None]]:
        """Return list of (name, parameter_num, channel)."""
        if not self.instrument:
            return []
        selected = []
        for cb in self.analyte_checks:
            if not cb.isChecked():
                continue
            name = cb.text()
            pnum = (self.instrument.analytes or {}).get(name)
            channel = None
            if "(" in name and ")" in name:
                _, ch = name.split("(", 1)
                channel = ch.strip(") ").strip()
            if pnum is not None:
                selected.append((name, pnum, channel))
        return selected

    def refresh_tanks(self):
        """Query DB for tanks matching the selected year range + analyte."""
        if not self.instrument or not self.tanks_plotter:
            self._rebuild_tank_checks([], "Select a gas to load tanks.")
            return

        start = self.start_year.value()
        end = self.end_year.value()
        if start > end:
            start, end = end, start
            self.start_year.setValue(start)
            self.end_year.setValue(end)

        selections = self._selected_analytes()
        if not selections:
            self._rebuild_tank_checks([], "Select at least one gas to load tanks.")
            return

        tanks_info: dict[str, str | None] = {}
        for _, pnum, channel in selections:
            tanks = self.tanks_plotter.return_active_tanks(
                start, end, parameter_num=pnum, channel=channel
            )
            for entry in tanks:
                serial = None
                use_short = None
                if isinstance(entry, dict):
                    serial = entry.get("serial_number") or entry.get("tank_serial_num")
                    use_short = entry.get("use_short")
                else:
                    serial = str(entry)
                if not serial:
                    continue
                serial_str = str(serial)
                if serial_str not in tanks_info:
                    tanks_info[serial_str] = use_short if use_short else None

        tanks_list = [(serial, tanks_info[serial]) for serial in sorted(tanks_info.keys())]
        self._rebuild_tank_checks(tanks_list)

    def _rebuild_tank_checks(self, tanks, empty_msg: str = None):
        """Clear and rebuild the checkbox grid."""
        # Remove existing widgets from the grid
        for i in reversed(range(self.tank_grid.count())):
            item = self.tank_grid.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)

        self.tank_checks.clear()

        if not tanks:
            self.tanks_status.setText(empty_msg or "No tanks found for that selection.")
            return

        self.tanks_status.setText(
            f"{len(tanks)} tanks found. "
            f"<span style='color: darkred;'>Grav</span> "
            f"| <span style='color: darkgreen;'>Archive</span>"
        )
        cols = 5
        for idx, tank in enumerate(tanks):
            serial = None
            use_short = None
            if isinstance(tank, dict):
                serial = tank.get("serial_number") or tank.get("tank_serial_num")
                use_short = tank.get("use_short")
            elif isinstance(tank, tuple):
                if len(tank) >= 1:
                    serial = tank[0]
                if len(tank) >= 2:
                    use_short = tank[1]
            label = str(serial or tank)
            use_lower = str(use_short).lower() if use_short is not None else ""
            cb = QCheckBox(label)
            cb.setChecked(False)
            if use_lower:
                if use_lower.startswith("grav"):
                    cb.setStyleSheet("color: darkred;")
                elif use_lower.startswith("archive"):
                    cb.setStyleSheet("color: darkgreen;")
            self.tank_checks.append(cb)
            row, col = divmod(idx, cols)
            self.tank_grid.addWidget(cb, row, col)

    def selected_tanks(self) -> list[str]:
        """Return checked tank serial numbers."""
        return [cb.text() for cb in self.tank_checks if cb.isChecked()]


if __name__ == "__main__":
    """
    Minimal harness to exercise the widget standalone using the real M4 instrument/DB.
    Falls back to fake data if initialization fails (e.g., DB connectivity issues).
    """
    from PyQt5.QtWidgets import QApplication

    try:
        from logos_instruments import M4_Instrument
    except Exception as exc:  # pragma: no cover - import-time failure path
        print(f"Could not import M4_Instrument: {exc}", file=sys.stderr)
        M4_Instrument = None  # type: ignore

    class _FakeDB:
        def query_to_df(self, sql):
            # Pretend the query returned a small tank list.
            return pd.DataFrame(
                [
                    {"tank_serial_num": "TNK-001"},
                    {"tank_serial_num": "TNK-002"},
                    {"tank_serial_num": "TNK-003"},
                    {"tank_serial_num": "TNK-004"},
                    {"tank_serial_num": "TNK-005"},
                ]
            )

    class _FakeInstrument:
        def __init__(self):
            self.db = _FakeDB()
            self.inst_num = 999
            self.analytes = {"CO2 (C)": 44, "CH4 (W)": 77, "N2O": 88}
            self.start_date = "2018-01-01"

    def build_instrument():
        if M4_Instrument is None:
            return None
        try:
            return M4_Instrument()
        except Exception as exc:
            print(f"Failed to initialize M4_Instrument (using fake data): {exc}", file=sys.stderr)
            return None

    app = QApplication(sys.argv)
    instrument = build_instrument() or _FakeInstrument()
    widget = TanksWidget(instrument=instrument)
    widget.setWindowTitle("TanksWidget Test Harness (M4)")
    widget.resize(320, 420)
    widget.show()
    sys.exit(app.exec_())
    
