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
        """Return a list of tank serial numbers active in the given window."""
        if parameter_num is None:
            return []

        start_ts = f"{start_year}-01-01"
        end_ts = f"{end_year + 1}-01-01"  # half-open interval on year boundary

        sql = f"""
            SELECT a.tank_serial_num
            FROM hats.ng_analysis a
            JOIN hats.ng_mole_fractions mf
              ON mf.analysis_num = a.num
            WHERE a.inst_num = {self.inst_num}
              AND mf.parameter_num = {parameter_num}
              AND a.tank_serial_num IS NOT NULL
              AND a.run_time >= '{start_ts}'
              AND a.run_time <  '{end_ts}'
              {f"AND mf.channel = '{channel}'" if channel else ""}
            GROUP BY a.tank_serial_num
            ORDER BY a.tank_serial_num;
        """

        if hasattr(self.db, "query_to_df"):
            df = self.db.query_to_df(sql)
        else:
            df = pd.DataFrame(self.db.doquery(sql))
        return df['tank_serial_num'].dropna().tolist()
    

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

        tanks_set: set[str] = set()
        for _, pnum, channel in selections:
            tanks = self.tanks_plotter.return_active_tanks(
                start, end, parameter_num=pnum, channel=channel
            )
            tanks_set.update(tanks)

        self._rebuild_tank_checks(sorted(tanks_set))

    def _rebuild_tank_checks(self, tanks: list[str], empty_msg: str = None):
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

        self.tanks_status.setText(f"{len(tanks)} tanks found.")
        cols = 5
        for idx, tank in enumerate(tanks):
            cb = QCheckBox(str(tank))
            cb.setChecked(False)
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
    
