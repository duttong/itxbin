#! /usr/bin/env python

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
del _here, _sys, _os

import json
import os
import math
import warnings

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip, QApplication, QInputDialog, QSizePolicy, QShortcut,
    QLineEdit, QRadioButton, QButtonGroup, QMessageBox, QScrollArea, QFrame
)
from PyQt5.QtGui import QCursor, QKeySequence
from PyQt5.QtCore import Qt, QTimer

from matplotlib.widgets import Button
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import sys
from collections import defaultdict


LOGOS_sites = ['SUM', 'PSA', 'SPO', 'SMO', 'AMY', 'MKO', 'ALT', 'CGO', 'NWR',
            'LEF', 'BRW', 'RPB', 'KUM', 'MLO', 'WIS', 'THD', 'MHD', 'HFM',
            'BLD', 'MKO']

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".logos-tanks.conf")
MAX_SAVED_SETS = 5

# Directory holding the shared CCG calibration/fit modules that `caldrift`
# uses (ccg_cal_db.Calibrations + ccg_calfit.fitCalibrations + the
# ccg_refgasdb.refgas.insertFromFit() write path used by caldrift's -u/
# --update option). We import and reuse these directly rather than shelling
# out to or editing caldrift itself. Matches the path in the /ccg/bin/caldrift
# wrapper -- saving from the panel uses the exact same insert code caldrift's
# CLI would call, not a hand-rolled INSERT.
_CCG_NEXTGEN = "/ccg/src/python3/nextgen"


def _import_ccg_fit():
    """Lazily import the CCG cal/fit/refgasdb modules from the nextgen dir.

    Returns a (ccg_cal_db, ccg_calfit, ccg_dates, ccg_refgasdb) tuple, or
    None if the modules are unavailable (keeps logos_tanks usable without
    them).
    """
    import importlib
    if _CCG_NEXTGEN not in sys.path:
        sys.path.append(_CCG_NEXTGEN)
    try:
        return (
            importlib.import_module("ccg_cal_db"),
            importlib.import_module("ccg_calfit"),
            importlib.import_module("ccg_dates"),
            importlib.import_module("ccg_refgasdb"),
        )
    except Exception:
        return None


def _read_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_config(data: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


class TanksPlotter:
        
    def __init__(self, db, inst_num, calibration_inst_ids=None):
        self.db = db
        self.inst_num = inst_num
        self.calibration_inst_ids = calibration_inst_ids

    def _active_tanks_dataframe(self, start_year, end_year) -> pd.DataFrame:
        """Return raw DataFrame of active tanks across all analytes for a window."""
        start_ts = f"{start_year}-01-01"
        end_ts = f"{end_year + 1}-01-01"
        
        inst_map = {193: "fe3", 192: "m4", 220: "bld1", 236: "ie3", 58: "pr1", 238: "pr2"}
        calibration_inst_ids = self.calibration_inst_ids or (inst_map.get(self.inst_num, ""),)
        inst_list = ",".join(
            f"'{str(inst).replace(chr(39), chr(39) + chr(39))}'"
            for inst in calibration_inst_ids
            if inst
        )
        if not inst_list:
            return pd.DataFrame()

        fills_sql_old = f"""
            SELECT
                DISTINCT r.idx
            FROM hats.ng_analysis a
            JOIN reftank.fill r
              ON r.serial_number = a.tank_serial_num
             AND r.`date` = (
                  SELECT MAX(f2.`date`)
                  FROM reftank.fill f2
                  WHERE f2.serial_number = a.tank_serial_num
                    AND f2.`date` <= a.analysis_time
              )
            WHERE a.inst_num = {self.inst_num}
              AND a.tank_serial_num IS NOT NULL
              AND a.run_time >= '{start_ts}'
              AND a.run_time <  '{end_ts}'
            ORDER BY r.serial_number;
        """
        
        # working on a new query using updated hats.calibrations table.
        fills_sql = f"""
            select 
                DISTINCT r.idx
            from hats.calibrations c
                JOIN reftank.fill r
                 ON r.serial_number = c.serial_number
                AND r.`date` = (
                    SELECT MAX(f2.`date`)
                    FROM reftank.fill f2
                    WHERE f2.serial_number = c.serial_number
                    AND f2.`date` <= c.date
                )
            WHERE inst IN ({inst_list})
            AND c.mixratio IS NOT NULL
            AND c.mixratio > -99
            AND c.date > '{start_ts}'
            AND c.date <  '{end_ts}';
        """
        fills_df = pd.DataFrame(self.db.doquery(fills_sql))
        fill_ids = [str(idx) for idx in fills_df["idx"].dropna().unique()] if not fills_df.empty else []
        if not fill_ids:
            return pd.DataFrame()

        fill_list = ",".join(fill_ids)
        
        sql = f"""
            SELECT
                h.num,
                h.fill_idx,
                f.serial_number,
                f.`date`,
                f.code,
                f.location,
                h.ng_tank_uses_num,
                u.abbr         AS use_short,
                u.description  AS use_desc,
                g.species,
                g.species_num AS parameter_num,
                g.mf_value,
                h.site_num,
                h.start,
                h.end,
                h.comment,
                f.notes AS fill_notes,
                g.notes AS grav_notes
            FROM reftank.fill f
            LEFT JOIN hats.ng_tank_use_history h
              ON h.fill_idx = f.idx
            LEFT JOIN hats.ng_tank_uses u
              ON u.num = h.ng_tank_uses_num
            LEFT JOIN reftank.grav_view g
              ON g.fill_num = f.idx
            WHERE f.idx IN ({fill_list})
            ORDER BY f.serial_number, f.`date` DESC;
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if "date" in df.columns:
            df = df.rename(columns={"date": "fill_date"})
        return df

    def _filter_active_tanks(self, df: pd.DataFrame, parameter_num=None) -> list[dict]:
        """Filter a tank DataFrame down to parameter-specific rows."""
        if df is None or df.empty:
            return []

        df = df.copy()

        if "ng_tank_uses_num" in df.columns:
            other_mask = df["ng_tank_uses_num"].isna()
            if other_mask.any():
                df.loc[other_mask, "use_short"] = "Other"

        if "use_short" in df.columns and "parameter_num" in df.columns:
            use_lower = df["use_short"].fillna("").str.lower()
            is_grav = use_lower.str.startswith("grav")
            param_series = pd.to_numeric(df["parameter_num"], errors="coerce")
            param_value = (
                pd.to_numeric(parameter_num, errors="coerce")
                if parameter_num is not None
                else None
            )
            grav_match = param_series == param_value if param_value is not None else True
            non_grav = ~is_grav
            grav_other = is_grav & ~grav_match
            df_other = df[grav_other].copy()
            if not df_other.empty:
                df_other["use_short"] = "Other Gravs"
            df = pd.concat([df[non_grav | grav_match], df_other], ignore_index=True)

        use_lower = df["use_short"].fillna("").str.lower()
        df["__is_grav"] = use_lower.str.contains("grav")
        df = df.sort_values(
            ["serial_number", "__is_grav", "fill_date"],
            ascending=[True, False, False],
        )
        df = df.drop_duplicates(subset=["fill_idx"], keep="first")
        df = df.drop(columns=["__is_grav"], errors="ignore")

        return df.to_dict(orient="records")
        
        
    def return_active_tanks(self, start_year, end_year, parameter_num=None, channel=None):
        """
        Return a list of tank records (serial + latest fill metadata) active in the window.
        """
        df = self._active_tanks_dataframe(start_year, end_year)
        return self._filter_active_tanks(df, parameter_num)

    def return_active_tanks_df(self, start_year, end_year) -> pd.DataFrame:
        """Expose raw tank dataframe (all analytes) for caching."""
        return self._active_tanks_dataframe(start_year, end_year)

    def filter_active_tanks(self, df: pd.DataFrame, parameter_num=None) -> list[dict]:
        """Public wrapper to filter cached tank data by parameter."""
        return self._filter_active_tanks(df, parameter_num)
    

# Explicit fit degrees selectable in the caldrift panel, passed straight
# through to ccg_calfit.fitCalibrations(degree=...). "Auto" is caldrift's
# own behavior: start at the highest degree the data supports and fall back
# through a significance test on the top coefficient. The other three force
# that exact degree and skip the significance test/fallback entirely (see
# fitCalibrations() in ccg_calfit.py) -- e.g. "Linear" always draws a sloped
# line even if the slope isn't statistically significant.
CALDRIFT_FIT_DEGREES = [
    ("Auto", "auto"),
    ("Mean", "mean"),
    ("Linear", "linear"),
    ("Quadratic", "quadratic"),
]

# hats.scale_assignments.level enum values, in the order caldrift's own
# --level CLI flag documents them. Only meaningful for a first-time
# assignment -- an update to an existing assignment always carries the
# existing row's level forward untouched (see update_assignment_db() in
# caldrift.py, which refuses to change level on an existing assignment).
CALDRIFT_LEVELS = ["Primary", "Secondary", "Tertiary", "Other"]


class CaldriftPanel(QWidget):
    """Floating panel for the single-tank caldrift figure.

    Flag/unflag calibration episodes (hats.calibrations.flag = 'M' / '.'),
    toggle whether flagged episodes feed the fit, and refresh the fit. Styled
    to match the logos_data Multi-Tag panel. All actions call back into the
    owning TanksWidget (``controller``).
    """

    def __init__(self, controller):
        super().__init__(None, Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.controller = controller
        self.setWindowTitle("Caldrift")
        self._serial = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Tank identity + copy-to-clipboard.
        header_row = QHBoxLayout()
        self._tank_label = QLabel("Tank: —")
        self._tank_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self._tank_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._copy_serial_btn = QPushButton("Copy")
        self._copy_serial_btn.setToolTip("Copy this tank's serial number to the clipboard.")
        self._copy_serial_btn.setStyleSheet(
            "QPushButton { padding: 2px 8px; border: 1px solid #aaa; border-radius: 4px; }"
        )
        self._copy_serial_btn.clicked.connect(self.controller._caldrift_copy_serial)
        header_row.addWidget(self._tank_label)
        header_row.addStretch()
        header_row.addWidget(self._copy_serial_btn)
        layout.addLayout(header_row)

        instructions = QLabel(
            "<b>Flag episodes:</b>"
            "<ul style='-qt-list-indent:0; margin-left:8px; margin-top:2px; margin-bottom:0px;'>"
            "<li>Click a calibration point to select it; <b>SHIFT+click</b> adds/removes more.</li>"
            "<li><b>Flag Selected</b> flagged points show as open circles.</li>"
            "<li>With <b>Exclude Flagged</b> checked, the caldrift fit ignores flagged episodes.</li>"
            "<li><b>Fit degree</b> forces Mean/Linear/Quadratic instead of caldrift's "
            "auto significance test.</li>"
            "<li><b>Refresh Fit</b> re-reads the flags and re-runs the fit.</li>"
            "</ul>"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "background-color: #eef3fb; border: 1px solid #c8d4e8; "
            "border-radius: 6px; padding: 4px 6px; font-size: 11px; color: #1a2a4a;"
        )
        layout.addWidget(instructions)

        self._info_label = QLabel("No episode selected")
        self._info_label.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        layout.addWidget(self._info_label)

        _btn_style = "QPushButton { padding: 4px 8px; border: 1px solid #aaa; border-radius: 4px; }"
        flag_row = QHBoxLayout()
        self._flag_btn = QPushButton("Flag Selected (M)")
        self._flag_btn.setStyleSheet(_btn_style)
        self._flag_btn.clicked.connect(lambda: self.controller._caldrift_flag_selected("M"))
        self._unflag_btn = QPushButton("Unflag Selected")
        self._unflag_btn.setStyleSheet(_btn_style)
        self._unflag_btn.clicked.connect(lambda: self.controller._caldrift_flag_selected("."))
        flag_row.addWidget(self._flag_btn)
        flag_row.addWidget(self._unflag_btn)
        layout.addLayout(flag_row)

        self._clear_btn = QPushButton("Clear Selection")
        self._clear_btn.setStyleSheet(_btn_style)
        self._clear_btn.clicked.connect(self.controller._caldrift_clear_selection)
        layout.addWidget(self._clear_btn)

        self._exclude_cb = QCheckBox("Exclude Flagged")
        self._exclude_cb.setToolTip("Leave flagged episodes out of the caldrift fit.")
        self._exclude_cb.setChecked(True)
        self._exclude_cb.toggled.connect(self.controller._caldrift_set_exclude)
        layout.addWidget(self._exclude_cb)

        self._hide_cb = QCheckBox("Hide Flagged")
        self._hide_cb.setToolTip(
            "Remove flagged points from the figure so autoscale uses only unflagged data."
        )
        self._hide_cb.toggled.connect(self.controller._caldrift_set_hide)
        layout.addWidget(self._hide_cb)

        fit_group_box = QGroupBox("FIT DEGREE")
        fit_row = QHBoxLayout()
        fit_tooltip = (
            "Auto picks the highest-degree fit whose top coefficient is "
            "statistically significant (caldrift's default). Mean/Linear/"
            "Quadratic force that exact degree, skipping the significance test."
        )
        self._fit_buttons: dict[str, QRadioButton] = {}
        self._fit_group = QButtonGroup(self)
        for label, value in CALDRIFT_FIT_DEGREES:
            rb = QRadioButton(label)
            rb.setToolTip(fit_tooltip)
            rb.setChecked(value == "auto")
            self._fit_group.addButton(rb)
            self._fit_buttons[value] = rb
            fit_row.addWidget(rb)
        fit_row.addStretch()
        fit_group_box.setLayout(fit_row)
        layout.addWidget(fit_group_box)
        self._fit_group.buttonToggled.connect(self._on_fit_degree_toggled)

        self._refresh_btn = QPushButton("Refresh Fit")
        self._refresh_btn.setStyleSheet(
            "QPushButton { padding: 4px 8px; border: 1px solid #8bbf8b; "
            "border-radius: 4px; background-color: #e7f6e7; }"
        )
        self._refresh_btn.clicked.connect(self.controller._caldrift_refresh_fit)
        layout.addWidget(self._refresh_btn)

        self._level_group_box = QGroupBox("LEVEL (first-time assignments only)")
        level_row = QHBoxLayout()
        level_tooltip = (
            "Level for a brand-new scale assignment. Disabled when the tank "
            "already has one -- the existing level is always carried forward "
            "on an update, matching caldrift's own behavior."
        )
        self._level_buttons: dict[str, QRadioButton] = {}
        self._level_group = QButtonGroup(self)
        for label in CALDRIFT_LEVELS:
            rb = QRadioButton(label)
            rb.setToolTip(level_tooltip)
            rb.setChecked(label == "Tertiary")
            self._level_group.addButton(rb)
            self._level_buttons[label] = rb
            level_row.addWidget(rb)
        level_row.addStretch()
        self._level_group_box.setLayout(level_row)
        self._level_group_box.setToolTip(level_tooltip)
        layout.addWidget(self._level_group_box)
        self._level_group.buttonToggled.connect(self._on_level_toggled)

        self._save_assignment_btn = QPushButton("Save to Scale Assignments")
        self._save_assignment_btn.setToolTip(
            "Write the currently plotted caldrift fit to hats.scale_assignments, "
            "recorded under this panel's instrument."
        )
        self._save_assignment_btn.setStyleSheet(
            "QPushButton { padding: 4px 8px; border: 1px solid #d9975a; "
            "border-radius: 4px; background-color: #ffe0b2; }"
        )
        self._save_assignment_btn.clicked.connect(
            self.controller._caldrift_save_scale_assignment
        )
        layout.addWidget(self._save_assignment_btn)

        self.setMinimumWidth(300)
        self.set_selected_count(0)

    def _on_fit_degree_toggled(self, button, checked):
        if not checked:
            return
        for value, rb in self._fit_buttons.items():
            if rb is button:
                self.controller._caldrift_set_fit_degree(value)
                break

    def _on_level_toggled(self, button, checked):
        if not checked:
            return
        for value, rb in self._level_buttons.items():
            if rb is button:
                self.controller._caldrift_set_level(value)
                break

    def set_level_state(self, enabled: bool, level: str | None = None):
        """Enable the level picker for a first-time assignment (showing the
        user's last chosen level), or disable it and show the level being
        carried forward from an existing assignment."""
        self._level_group_box.setEnabled(enabled)
        if level in self._level_buttons:
            rb = self._level_buttons[level]
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)

    def set_tank(self, serial, fillcode=None):
        """Update the tank-identity header (serial + optional fill code)."""
        self._serial = str(serial) if serial else None
        if serial and fillcode:
            self._tank_label.setText(f"Tank: {serial} (fill {fillcode})")
        elif serial:
            self._tank_label.setText(f"Tank: {serial}")
        else:
            self._tank_label.setText("Tank: —")
        self._copy_serial_btn.setEnabled(bool(serial))

    def set_selected_count(self, n: int):
        self._info_label.setText("No episode selected" if n == 0 else f"{n} episode(s) selected")
        has = n > 0
        self._flag_btn.setEnabled(has)
        self._unflag_btn.setEnabled(has)
        self._clear_btn.setEnabled(has)

    def closeEvent(self, event):
        try:
            self.controller._caldrift_on_panel_closed()
        except Exception:
            pass
        super().closeEvent(event)


class TanksWidget(QWidget):
    """
    Simple control pane for selecting a year range + gas and showing tanks
    as a 3-column grid of checkboxes.
    """
    def __init__(self, instrument=None, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.main_window = self.parent()
        self.tanks_plotter = (
            TanksPlotter(
                self.instrument.db,
                self.instrument.inst_num,
                getattr(self.instrument, "calibration_inst_ids", None),
            )
            if self.instrument else None
        )
        self.tank_checks: list[QCheckBox] = []
        self.analyte_checks: list[QCheckBox] = []
        self._ready = False
        self._reload_dirty = False
        self.saved_sets: dict[str, list[dict | None]] = {}
        self.active_set_idx: int | None = None
        self.set_buttons: list[QPushButton] = []
        self._loading_set = False
        self._tank_metadata: dict[str, dict] = {}
        self._tank_cache: dict[str, list[dict]] = {}
        self._tank_cache_range: tuple[int, int] | None = None
        # (tank_cache_range, pnum, serials) -> {serial: [timestamp, ...]}. A
        # category-filter checkbox triggers refresh_tanks() without changing
        # the analyte, year range, or tank set, so this key stays the same
        # across those calls -- avoids reissuing the hats.calibrations query
        # in _annotate_recent_analysis() on every filter click.
        self._recent_analysis_cache: dict[tuple, dict[str, list]] = {}
        self._resize_timer: QTimer | None = None
        # --- caldrift panel state (single-tank figure) ---
        self._caldrift_panel = None            # CaldriftPanel | None
        self._caldrift_ctx: dict | None = None  # live single-tank figure context
        self._caldrift_last_fit: dict | None = None  # last-plotted fit (for Save to Scale Assignments)
        self._caldrift_refresh = None           # callable: re-run _plot_for for current analyte
        self._caldrift_selected: set[int] = set()   # df row indices selected for flagging
        self._caldrift_exclude_flagged: bool = True  # flagged episodes omitted from the fit
        self._caldrift_hide_flagged: bool = False    # flagged episodes removed from the figure
        self._caldrift_fit_degree: str = "auto"      # degree passed to ccg_calfit.fitCalibrations
        self._caldrift_level: str = "Tertiary"       # level for a first-time scale assignment
        self._caldrift_suppress_close_refresh = False  # True while tearing down the whole figure
        self.analytes = self.instrument.analytes or {} if self.instrument else {}
        self._analyte_names = list((self.instrument.analytes or {}).keys() if self.instrument else [])
        self._preferred_channels = self._load_preferred_channels()
        if self._preferred_channels:
            self._analyte_names = [
                name for name in self._analyte_names
                if self._is_preferred_channel(name)
            ]

        # --- Layout scaffold (mirror logos_timeseries style) ---
        controls = QVBoxLayout()

        # Year range selection
        i_start = int(self.instrument.start_date[0:4]) if self.instrument else 2020
        i_end = pd.Timestamp.now().year
        year_group = QGroupBox("YEAR RANGE")
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
        analyte_group = QGroupBox("GAS / PARAMETER")
        analyte_layout = QVBoxLayout()
        alpha_row = QHBoxLayout()
        self.alpha_sort_cb = QCheckBox("List Alphabetically")
        self.alpha_sort_cb.toggled.connect(self._on_alpha_sort_toggled)
        alpha_row.addWidget(self.alpha_sort_cb)
        alpha_row.addStretch()
        analyte_layout.addLayout(alpha_row)
        analyte_container = QWidget()
        analyte_container.setStyleSheet("background-color: #fffbe6; border: 1px solid #f2e6b3;")
        analyte_checks_layout = QGridLayout()
        analyte_checks_layout.setContentsMargins(6, 6, 6, 6)
        analyte_checks_layout.setHorizontalSpacing(8)
        analyte_checks_layout.setVerticalSpacing(4)
        self._analyte_checks_layout = analyte_checks_layout
        cols_analyte = 5
        default_analyte = None
        if self.instrument and getattr(self.instrument, "inst_id", None) == "fe3":
            for name in self._analyte_names:
                if name == "N2O" or name.startswith("N2O"):
                    default_analyte = name
                    break
        for idx, name in enumerate(self._analyte_names):
            cb = QCheckBox(self._display_label(name))
            cb.setProperty("analyte_name", name)
            if default_analyte is not None:
                if name == default_analyte:
                    cb.setChecked(True)
            elif idx == 0:
                cb.setChecked(True)
            cb.toggled.connect(lambda checked, cb=cb: self._on_analyte_toggled(cb, checked))
            self.analyte_checks.append(cb)
        self._reflow_analyte_grid(cols_analyte)
        analyte_container.setLayout(analyte_checks_layout)
        analyte_scroll = QScrollArea()
        analyte_scroll.setWidgetResizable(True)
        analyte_scroll.setFrameShape(QFrame.NoFrame)
        analyte_scroll.setMinimumHeight(160)
        analyte_scroll.setMaximumHeight(360)
        analyte_scroll.setWidget(analyte_container)
        analyte_layout.addWidget(analyte_scroll)
        self.tanks_status = QLabel("Select a year range and gas to load tanks.")
        self.tanks_status.setWordWrap(True)
        analyte_layout.addWidget(self.tanks_status)

        # --- Filter / sort / search box ---
        filter_group = QGroupBox("FILTER / SORT")
        filter_layout = QVBoxLayout()
        category_bar = QHBoxLayout()
        self.show_grav_cb = QCheckBox("Gravimetric")
        self.show_grav_cb.setChecked(True)
        self.show_grav_cb.setStyleSheet("color: darkred;")
        self.show_grav_cb.toggled.connect(self._on_category_toggle)
        self.show_other_grav_cb = QCheckBox("Other Gravs")
        self.show_other_grav_cb.setChecked(False)
        self.show_other_grav_cb.setStyleSheet("color: darkblue;")
        self.show_other_grav_cb.toggled.connect(self._on_category_toggle)
        self.show_other_cb = QCheckBox("Other")
        self.show_other_cb.setChecked(True)
        self.show_other_cb.setStyleSheet("color: dimgray;")
        self.show_other_cb.toggled.connect(self._on_category_toggle)
        self.show_cal_cb = QCheckBox("Cal Tanks")
        self.show_cal_cb.setChecked(True)
        self.show_cal_cb.setStyleSheet("color: #66aadd;")
        self.show_cal_cb.toggled.connect(self._on_category_toggle)
        self.show_archive_cb = QCheckBox("Archive")
        self.show_archive_cb.setChecked(True)
        self.show_archive_cb.setStyleSheet("color: darkgreen;")
        self.show_archive_cb.toggled.connect(self._on_category_toggle)
        category_bar.addWidget(self.show_grav_cb)
        category_bar.addWidget(self.show_other_grav_cb)
        category_bar.addWidget(self.show_cal_cb)
        category_bar.addWidget(self.show_archive_cb)
        category_bar.addWidget(self.show_other_cb)
        category_bar.addStretch()
        filter_layout.addLayout(category_bar)
        sort_bar = QHBoxLayout()
        self.sort_alpha_rb = QRadioButton("Alphabetical Sort")
        self.sort_alpha_rb.setChecked(True)
        self.sort_recent_rb = QRadioButton("Analysis Date Sort")
        self.sort_recent_rb.setToolTip(
            "List tanks with the most recent hats.calibrations analysis first."
        )
        self.sort_button_group = QButtonGroup(self)
        self.sort_button_group.addButton(self.sort_alpha_rb)
        self.sort_button_group.addButton(self.sort_recent_rb)
        self.sort_button_group.buttonToggled.connect(self._on_sort_mode_changed)
        sort_bar.addWidget(self.sort_alpha_rb)
        sort_bar.addWidget(self.sort_recent_rb)
        sort_bar.addStretch()
        filter_layout.addLayout(sort_bar)
        search_bar = QHBoxLayout()
        search_bar.addWidget(QLabel("Search for tank in list"))
        self.tank_search = QLineEdit()
        self.tank_search.setPlaceholderText("type to filter tanks (e.g. CC)")
        self.tank_search.setClearButtonEnabled(False)
        self.tank_search.textChanged.connect(self._on_tank_search_changed)
        search_bar.addWidget(self.tank_search)
        self.clear_search_btn = QPushButton("Clear")
        self.clear_search_btn.clicked.connect(self._on_clear_tank_search)
        search_bar.addWidget(self.clear_search_btn)
        filter_layout.addLayout(search_bar)
        filter_group.setLayout(filter_layout)
        analyte_layout.addWidget(filter_group)

        # --- Tank checkbox grid box ---
        tanks_group = QGroupBox("TANKS")
        tanks_group_layout = QVBoxLayout()
        self.tank_grid = QGridLayout()
        self.tank_grid.setContentsMargins(0, 0, 0, 0)
        self.tank_grid.setHorizontalSpacing(8)
        self.tank_grid.setVerticalSpacing(4)
        tank_grid_container = QWidget()
        tank_grid_container.setLayout(self.tank_grid)
        tank_scroll = QScrollArea()
        tank_scroll.setWidgetResizable(True)
        tank_scroll.setFrameShape(QFrame.NoFrame)
        tank_scroll.setMinimumHeight(160)
        tank_scroll.setMaximumHeight(420)
        tank_scroll.setWidget(tank_grid_container)
        tanks_group_layout.addWidget(tank_scroll)
        tanks_group.setLayout(tanks_group_layout)
        analyte_layout.addWidget(tanks_group)

        selection_bar = QHBoxLayout()
        self.deselect_btn = QPushButton("Deselect All")
        self.deselect_btn.clicked.connect(self._on_deselect_all)
        self.all_gravs_btn = QPushButton("All Gravs")
        self.all_gravs_btn.clicked.connect(self._on_select_all_gravs)
        selection_bar.addWidget(self.deselect_btn)
        selection_bar.addWidget(self.all_gravs_btn)
        selection_bar.addStretch()
        analyte_layout.addLayout(selection_bar)
        sets_bar = QHBoxLayout()
        self.save_set_btn = QPushButton("Save Tank Set")
        self.save_set_btn.clicked.connect(self._on_save_set)
        sets_bar.addWidget(self.save_set_btn)
        self.set_buttons_container = QWidget()
        self.set_buttons_layout = QHBoxLayout()
        self.set_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.set_buttons_layout.setSpacing(6)
        self.set_buttons_container.setLayout(self.set_buttons_layout)
        sets_bar.addWidget(self.set_buttons_container)
        self.delete_set_btn = QPushButton("Delete Tank Set")
        self.delete_set_btn.clicked.connect(self._on_delete_set)
        sets_bar.addWidget(self.delete_set_btn)
        sets_bar.addStretch()
        analyte_layout.addLayout(sets_bar)
        analyte_group.setLayout(analyte_layout)
        controls.addWidget(analyte_group)

        plot_bar = QHBoxLayout()
        plot_bar.addStretch()
        self.plot_tanks_btn = QPushButton("Plot Tanks")
        self.plot_tanks_btn.clicked.connect(self._on_plot_tanks)
        plot_bar.addWidget(self.plot_tanks_btn)
        controls.addLayout(plot_bar)

        controls.addStretch()
        self.setLayout(controls)

        # Wire date change after widgets exist
        self.start_year.valueChanged.connect(self._mark_reload_needed)
        self.end_year.valueChanged.connect(self._mark_reload_needed)

        # Populate tanks initially if any analyte starts checked
        self._ready = True
        self._load_saved_sets()
        self._refresh_set_buttons()
        if any(cb.isChecked() for cb in self.analyte_checks):
            self.refresh_tanks(force_reload=True)

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
        btn = getattr(self, "reload_btn", None)
        default_text = "Reload Tanks"
        if btn:
            btn.setText("Loading...")
            btn.setStyleSheet(
                "background-color: #f6e7a1; "
                "border: 1px solid #524b2f; "
                "padding: 3px 6px; "
                "color: #524b2f;")
            btn.setEnabled(False)
        try:
            self.refresh_tanks(force_reload=True)
            self._reload_dirty = False
        finally:
            if btn:
                btn.setText(default_text)
                btn.setStyleSheet("")
                btn.setEnabled(True)

    def _on_analyte_toggled(self, cb: QCheckBox, checked: bool):
        """Keep analyte selection single-choice and refresh tanks when changed."""
        if not getattr(self, "_ready", False):
            return
        if checked:
            self._clear_active_set_selection()
            for other in self.analyte_checks:
                if other is cb:
                    continue
                other.blockSignals(True)
                other.setChecked(False)
                other.blockSignals(False)
            self.refresh_tanks()
            self._refresh_set_buttons()
        else:
            if not any(c.isChecked() for c in self.analyte_checks):
                self.refresh_tanks()
                self._refresh_set_buttons()

    def _selected_analytes(self) -> list[tuple[str, int, str | None]]:
        """Return list of (name, parameter_num, channel)."""
        if not self.instrument:
            return []
        selected = []
        for cb in self.analyte_checks:
            if not cb.isChecked():
                continue
            name = self._cb_analyte_name(cb)
            pnum = (self.instrument.analytes or {}).get(name)
            channel = self._analyte_channel(name)
            if pnum is not None:
                selected.append((name, pnum, channel))
        return selected

    def _grid_col_count(self, checks: list[QCheckBox], default: int = 5,
                         min_cols: int = 3, max_cols: int = 10) -> int:
        """Pick a checkbox-grid column count from this widget's current
        width, so a wider window (more screen reclaimed from the old fixed
        left-column layout) shows fewer, more legible rows instead of
        leaving the extra space unused. Falls back to ``default`` before
        the widget has been laid out (width still 0)."""
        width = self.width()
        if width <= 0 or not checks:
            return default
        col_width = max((cb.sizeHint().width() for cb in checks), default=140) + 12
        usable = max(width - 64, col_width)
        cols = usable // col_width
        return int(max(min_cols, min(max_cols, cols)))

    def _reflow_analyte_grid(self, cols: int | None = None):
        """Re-layout analyte checkboxes based on sort toggle and available width."""
        layout = getattr(self, "_analyte_checks_layout", None)
        if layout is None:
            return
        if cols is None:
            cols = self._grid_col_count(self.analyte_checks, default=5)
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                layout.removeWidget(w)
        if self.alpha_sort_cb.isChecked():
            ordered = sorted(self.analyte_checks, key=lambda cb: cb.text().lower())
        else:
            name_to_cb = {self._cb_analyte_name(cb): cb for cb in self.analyte_checks}
            ordered = []
            for name in self._analyte_names:
                cb = name_to_cb.get(name)
                if cb:
                    ordered.append(cb)
            # append any new analytes not in original list
            for cb in self.analyte_checks:
                if cb not in ordered:
                    ordered.append(cb)
        self.analyte_checks = ordered
        for idx, cb in enumerate(ordered):
            row, col = divmod(idx, cols)
            layout.addWidget(cb, row, col)

    def _on_alpha_sort_toggled(self, _checked: bool):
        """Handle alphabetical sort toggle."""
        self._reflow_analyte_grid()

    def _cache_key(self, parameter_num: int | None, channel: str | None) -> str:
        """Return a stable cache key for an analyte."""
        channel_str = channel or ""
        return f"{parameter_num}:{channel_str}"

    def _analyte_channel(self, name: str) -> str | None:
        """Extract channel from analyte display name if present."""
        if "(" in name and ")" in name:
            _, ch = name.split("(", 1)
            return ch.strip(") ").strip()
        return None

    def _display_label(self, name: str) -> str:
        """Analyte name with any trailing channel suffix dropped for display
        ('CFC11 (c)' -> 'CFC11'; 'N2O' -> 'N2O'). Tank calibrations are already
        keyed on the preferred channel, so the suffix is redundant on screen;
        the full name is kept internally for parameter/channel lookups."""
        if name and self._analyte_channel(name):
            return name.split("(", 1)[0].strip()
        return name

    def _cb_analyte_name(self, cb) -> str:
        """Full analyte name (channel preserved) backing a checkbox, falling
        back to the visible text."""
        return cb.property("analyte_name") or cb.text()

    def _load_preferred_channels(self) -> dict[int, set[str]]:
        """Return parameter_num -> preferred channels mapping."""
        if not self.instrument or not getattr(self.instrument, "db", None):
            return {}
        inst_num = getattr(self.instrument, "inst_num", None)
        if inst_num is None:
            return {}
        sql = (
            "SELECT parameter_num, channel "
            "FROM hats.ng_preferred_channel "
            f"WHERE inst_num = {int(inst_num)};"
        )
        try:
            rows = self.instrument.db.doquery(sql)
        except Exception:
            return {}
        prefs: dict[int, set[str]] = {}
        for row in rows or []:
            try:
                pnum = row.get("parameter_num")
                channel = row.get("channel")
            except AttributeError:
                # Non-dict row; skip
                continue
            if pnum is None or channel in (None, ""):
                continue
            try:
                pnum_int = int(pnum)
            except (TypeError, ValueError):
                continue
            prefs.setdefault(pnum_int, set()).add(str(channel).strip())
        return prefs

    def _is_preferred_channel(self, analyte_name: str) -> bool:
        """Return True if analyte channel is in preferred list or no filter."""
        if not self.instrument:
            return True
        pnum = (self.instrument.analytes or {}).get(analyte_name)
        if pnum is None:
            return True
        preferred = self._preferred_channels.get(int(pnum))
        if not preferred:
            return True
        channel = self._analyte_channel(analyte_name)
        if not channel:
            return True
        return channel in preferred

    def _build_tank_cache(self, start: int, end: int):
        """Populate cached tanks for all analytes for the given date range."""
        # A new active-tanks fetch invalidates any cached recent-analysis
        # results -- they're keyed on tank_cache_range, so stale entries
        # would just go unused, but clearing avoids growing this unbounded
        # across a long session of year-range changes.
        self._recent_analysis_cache = {}
        if not self.instrument or not self.tanks_plotter:
            self._tank_cache = {}
            self._tank_cache_range = None
            return
        cache: dict[str, list[dict]] = {}
        df_all = self.tanks_plotter.return_active_tanks_df(start, end)
        for name, pnum in (self.instrument.analytes or {}).items():
            channel = self._analyte_channel(name)
            cache_key = self._cache_key(pnum, channel)
            cache[cache_key] = self.tanks_plotter.filter_active_tanks(
                df_all, parameter_num=pnum
            )
        self._tank_cache = cache
        self._tank_cache_range = (start, end)

    def refresh_tanks(self, force_reload: bool = False):
        """Query DB for tanks matching the selected year range + analyte."""
        if not self.instrument or not self.tanks_plotter:
            self._rebuild_tank_checks([], "Select a gas to load tanks.")
            return

        prev_selected = self.selected_tanks()

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

        should_rebuild = force_reload or self._tank_cache_range is None
        if not should_rebuild and self._tank_cache_range != (start, end) and not self._reload_dirty:
            should_rebuild = True
        if should_rebuild:
            self._build_tank_cache(start, end)

        def _fill_key(serial_val, fill_idx_val, fill_code_val):
            return f"{serial_val}::{fill_idx_val if fill_idx_val is not None else 'na'}::{fill_code_val or ''}"

        tanks_list: list[dict] = []
        tanks_seen: set[str] = set()
        self._tank_metadata = {}
        for _, pnum, channel in selections:
            cache_key = self._cache_key(pnum, channel)
            tanks = self._tank_cache.get(cache_key, [])
            for entry in tanks:
                serial = None
                use_short = None
                fill_idx = None
                fill_code = None
                fill_date = None
                if isinstance(entry, dict):
                    serial = entry.get("serial_number") or entry.get("tank_serial_num")
                    use_short = entry.get("use_short")
                    fill_idx = entry.get("fill_idx")
                    fill_code = entry.get("code") or entry.get("fill_code")
                    fill_date = entry.get("fill_date") or entry.get("date")
                else:
                    serial = str(entry)
                if not serial:
                    continue
                serial_str = str(serial)
                fill_key = _fill_key(serial_str, fill_idx, fill_code)
                if fill_key in tanks_seen:
                    continue
                tanks_seen.add(fill_key)
                if isinstance(entry, dict):
                    self._tank_metadata[fill_key] = entry
                tanks_list.append(
                    {
                        "serial": serial_str,
                        "use_short": use_short,
                        "fill_idx": fill_idx,
                        "fill_code": fill_code,
                        "fill_key": fill_key,
                        "fill_date": fill_date,
                    }
            )

        self._annotate_next_fill_dates()
        self._annotate_recent_analysis()

        if getattr(self, "sort_recent_rb", None) is not None and self.sort_recent_rb.isChecked():
            def _recent_sort_key(t):
                meta = self._tank_metadata.get(t.get("fill_key"), {})
                recent = meta.get("recent_analysis")
                return recent if pd.notnull(recent) else pd.Timestamp.min
            tanks_list = sorted(tanks_list, key=_recent_sort_key, reverse=True)
        else:
            tanks_list = sorted(
                tanks_list,
                key=lambda t: (t.get("serial") or "", str(t.get("fill_date") or t.get("fill_idx") or "")),
            )
        tanks_list = self._filter_tanks_by_category(tanks_list)
        empty_msg = None
        if not tanks_list and self._reload_dirty and self._tank_cache_range:
            cached_start, cached_end = self._tank_cache_range
            empty_msg = (
                f"No cached tanks for this analyte. Reload tanks for {cached_start}-{cached_end} "
                f"or press Reload Tanks after adjusting the date range."
            )
        self._rebuild_tank_checks(tanks_list, empty_msg)
        if prev_selected:
            self._apply_tank_selection(prev_selected)

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

        for idx, tank in enumerate(tanks):
            serial = None
            use_short = None
            fill_code = None
            fill_key = None
            if isinstance(tank, dict):
                serial = tank.get("serial") or tank.get("serial_number") or tank.get("tank_serial_num")
                use_short = tank.get("use_short")
                fill_code = tank.get("fill_code") or tank.get("code")
                fill_key = tank.get("fill_key")
            elif isinstance(tank, tuple):
                if len(tank) >= 1:
                    serial = tank[0]
                if len(tank) >= 2:
                    use_short = tank[1]
            serial_str = str(serial or tank)
            label = f"{serial_str} ({fill_code})" if fill_code else serial_str
            use_lower = str(use_short).lower() if use_short is not None else ""
            cb = QCheckBox(label)
            cb.setProperty("serial", serial_str)
            cb.setProperty("fill_key", fill_key or serial_str)
            cb.setProperty("use_short", use_short)
            cb.setChecked(False)
            cb.toggled.connect(self._on_tank_toggled)
            cb.setContextMenuPolicy(Qt.CustomContextMenu)
            cb.customContextMenuRequested.connect(
                lambda pos, cb=cb, key=(fill_key or serial_str): self._on_tank_context(cb, pos, key)
            )
            if use_lower:
                if use_lower.startswith("other gravs") or use_lower.startswith("other_gravs"):
                    cb.setStyleSheet("color: darkblue;")
                elif use_lower in ("cal", "second", "tert"):
                    cb.setStyleSheet("color: #66aadd;")
                elif use_lower.startswith("grav"):
                    cb.setStyleSheet("color: darkred;")
                elif use_lower.startswith("archive"):
                    cb.setStyleSheet("color: darkgreen;")
            self.tank_checks.append(cb)

        self._populate_tank_grid()

    def _tank_search_text(self) -> str:
        """Current lowercased search filter text (empty if none)."""
        widget = getattr(self, "tank_search", None)
        if widget is None:
            return ""
        return widget.text().strip().lower()

    def _tank_cb_matches_search(self, cb: QCheckBox) -> bool:
        """Return True if a tank checkbox matches the current search filter."""
        search = self._tank_search_text()
        return (not search) or (search in cb.text().lower())

    def _populate_tank_grid(self):
        """Lay out only the search-matching tank checkboxes into the grid."""
        # Detach all currently-laid-out widgets (keeps the checkbox objects).
        for i in reversed(range(self.tank_grid.count())):
            item = self.tank_grid.itemAt(i)
            w = item.widget()
            if w:
                self.tank_grid.removeWidget(w)

        visible = []
        for cb in self.tank_checks:
            if self._tank_cb_matches_search(cb):
                visible.append(cb)
            else:
                # Safe on a parentless widget; only ever hides, never shows.
                cb.setVisible(False)

        num_visible = len(visible)
        cols = self._grid_col_count(self.tank_checks, default=6)
        for idx, cb in enumerate(visible):
            row, col = divmod(idx, cols)
            # Reparent into the grid first; only then make visible. Calling
            # setVisible(True) on a still-parentless checkbox makes it flash as
            # its own top-level window before it lands in the layout.
            self.tank_grid.addWidget(cb, row, col)
            cb.setVisible(True)

        total = len(self.tank_checks)
        search = self._tank_search_text()
        if search:
            self.tanks_status.setText(
                f"{num_visible} of {total} tanks match '{self.tank_search.text().strip()}'."
            )
        else:
            self.tanks_status.setText(f"{total} tanks found.")

    def _on_tank_search_changed(self, _text: str):
        """Re-filter the displayed tank checkboxes as the user types."""
        self._populate_tank_grid()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not getattr(self, "_ready", False):
            return
        if self._resize_timer is None:
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._on_resize_settled)
        # Debounce: only reflow once the resize (e.g. a window drag) settles,
        # rather than rebuilding both grids on every intermediate event.
        self._resize_timer.start(120)

    def _on_resize_settled(self):
        self._reflow_analyte_grid()
        self._populate_tank_grid()

    def _on_clear_tank_search(self):
        """Clear the search field and show all tanks again."""
        widget = getattr(self, "tank_search", None)
        if widget is None:
            return
        widget.blockSignals(True)
        widget.clear()
        widget.blockSignals(False)
        self._populate_tank_grid()

    def selected_tanks(self) -> list[str]:
        """Return checked tank identifiers keyed by fill (fill_key).

        Only tanks currently displayed (matching the search filter) are
        returned, so plotting and saving act on the visible results.
        """
        selected = []
        for cb in self.tank_checks:
            if not cb.isChecked():
                continue
            if not self._tank_cb_matches_search(cb):
                continue
            fill_key = cb.property("fill_key")
            if fill_key is None:
                serial = cb.property("serial")
                fill_key = serial if serial is not None else cb.text().split(" ", 1)[0]
            selected.append(str(fill_key))
        return selected

    # --- Tank set persistence helpers ---
    def _load_saved_sets(self):
        """Load saved tank sets from disk."""
        try:
            data = _read_config()
            sets_by_analyte = {}
            if isinstance(data, dict) and isinstance(data.get("sets_by_analyte"), dict):
                for key, lst in data["sets_by_analyte"].items():
                    trimmed = [
                        entry if isinstance(entry, dict) else None
                        for entry in (lst if isinstance(lst, list) else [])
                    ][:MAX_SAVED_SETS]
                    trimmed += [None] * max(0, MAX_SAVED_SETS - len(trimmed))
                    sets_by_analyte[key] = trimmed
            elif isinstance(data, dict) and isinstance(data.get("sets"), list):
                # Legacy format: assign sets to their inferred keys.
                for entry in data["sets"]:
                    if not isinstance(entry, dict):
                        continue
                    key = self._key_for_saved_entry(entry)
                    if not key:
                        continue
                    lst = sets_by_analyte.setdefault(key, [None] * MAX_SAVED_SETS)
                    for idx in range(MAX_SAVED_SETS):
                        if lst[idx] is None:
                            lst[idx] = entry
                            break
            self.saved_sets = sets_by_analyte
        except Exception:
            # Ignore malformed files; keep defaults.
            self.saved_sets = {}

    def _persist_sets(self):
        """Persist saved sets to a JSON config."""
        payload = _read_config()
        payload["sets_by_analyte"] = self.saved_sets
        try:
            _write_config(payload)
        except Exception:
            # Silent failure to avoid UI crash; caller may re-attempt.
            return

    def _refresh_set_buttons(self):
        """Rebuild the dynamic set buttons for compatible saved sets."""
        key = self._current_analyte_key()
        sets = self._sets_for_key(key)
        if self.active_set_idx is not None:
            if self.active_set_idx >= len(sets) or not self._is_set_available(sets[self.active_set_idx]):
                self.active_set_idx = None
        # Clear existing buttons
        for i in reversed(range(self.set_buttons_layout.count())):
            item = self.set_buttons_layout.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.set_buttons.clear()

        if not sets:
            self.delete_set_btn.setEnabled(False)
            return

        for idx, saved in enumerate(sets):
            if not self._is_set_available(saved):
                continue
            label = saved.get("name") or f"Set {idx + 1}"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(idx == self.active_set_idx)
            btn.setStyleSheet("background-color: lightgreen;" if idx == self.active_set_idx else "")
            btn.clicked.connect(lambda checked, i=idx: self._on_set_clicked(i))
            self.set_buttons.append(btn)
            self.set_buttons_layout.addWidget(btn)
        self.delete_set_btn.setEnabled(self.active_set_idx is not None)

    def _on_save_set(self):
        """Save current selection to a slot."""
        selections = self._selected_analytes()
        if not selections:
            self._toast("Select a gas before saving a tank set.")
            return
        tanks = self.selected_tanks()
        if not tanks:
            self._toast("Select at least one tank before saving.")
            return
        name, pnum, channel = selections[0]
        key = self._current_analyte_key()
        if not key:
            self._toast("Unable to determine analyte key for saving.")
            return
        sets = self._sets_for_key(key, ensure=True)
        # Default target slot for prompt; may change after naming to overwrite by name.
        if self.active_set_idx is not None:
            target_idx = self.active_set_idx
        else:
            empty_idx = next((i for i, val in enumerate(sets) if val is None), None)
            target_idx = empty_idx if empty_idx is not None else 0
        existing_name = sets[target_idx].get("name") if isinstance(sets[target_idx], dict) else ""
        prompt_default = existing_name or f"Set {target_idx + 1}"
        save_name, ok = QInputDialog.getText(
            self,
            "Save Tank Set",
            "Enter a name for this tank set:",
            text=prompt_default,
        )
        if not ok:
            return
        save_name = save_name.strip() or prompt_default
        # If a set with this name already exists, overwrite it.
        matching_idx = next(
            (i for i, val in enumerate(sets) if isinstance(val, dict) and val.get("name") == save_name),
            None,
        )
        if matching_idx is not None:
            target_idx = matching_idx
        entry = {
            "instrument": getattr(self.instrument, "inst_num", None),
            "parameter_name": name,
            "parameter_num": pnum,
            "channel": channel,
            "tanks": tanks,
            "name": save_name,
        }
        sets[target_idx] = entry
        self.saved_sets[key] = sets
        self.active_set_idx = target_idx
        self._persist_sets()
        self._refresh_set_buttons()
        self._toast(f"Saved tank set '{save_name}'.")

    def _on_set_clicked(self, idx: int):
        """Load a saved set and mark it active."""
        self._apply_saved_set(idx)

    def _apply_saved_set(self, idx: int):
        key = self._current_analyte_key()
        sets = self._sets_for_key(key)
        if idx >= len(sets):
            return
        saved = sets[idx]
        if not saved:
            return
        inst_num = getattr(self.instrument, "inst_num", None)
        saved_inst = saved.get("instrument")
        if inst_num is not None and saved_inst not in (None, inst_num):
            self._toast("Saved set is for a different instrument.")
            return
        if not self._apply_parameter_selection(saved):
            self._toast("Saved parameter not available; cannot load set.")
            return
        self.active_set_idx = idx
        self.refresh_tanks()
        self._apply_tank_selection(saved.get("tanks", []))
        self._refresh_set_buttons()
        self._update_set_button_styles()

    def _find_matching_analyte_cb(self, saved: dict) -> QCheckBox | None:
        """Return checkbox matching saved analyte if available."""
        target_name = saved.get("parameter_name")
        target_num = saved.get("parameter_num")
        target_channel = saved.get("channel")
        for cb in self.analyte_checks:
            name = self._cb_analyte_name(cb)
            pnum = (self.instrument.analytes or {}).get(name) if self.instrument else None
            channel = self._analyte_channel(name)
            if target_name and name == target_name:
                return cb
            if target_num is not None and pnum == target_num and (target_channel is None or channel == target_channel):
                return cb
        return None

    def _apply_parameter_selection(self, saved: dict) -> bool:
        """Select the saved analyte if present."""
        matched_cb = self._find_matching_analyte_cb(saved)
        if not matched_cb:
            return False
        for cb in self.analyte_checks:
            cb.blockSignals(True)
            cb.setChecked(cb is matched_cb)
            cb.blockSignals(False)
        return True

    def _apply_tank_selection(self, tanks: list[str]):
        """Apply saved tank selections without triggering clears."""
        self._loading_set = True
        desired = set(str(t) for t in tanks)
        for cb in self.tank_checks:
            cb.blockSignals(True)
            fill_key = cb.property("fill_key") or cb.text()
            serial_base = cb.property("serial")
            should_check = (
                str(fill_key) in desired
                or (serial_base is not None and str(serial_base) in desired)
                or cb.text() in desired
            )
            cb.setChecked(should_check)
            cb.blockSignals(False)
        self._loading_set = False

    def _on_delete_set(self):
        """Delete the active saved set."""
        if self.active_set_idx is None:
            self._toast("Select a saved set before deleting.")
            return
        key = self._current_analyte_key()
        sets = self._sets_for_key(key)
        if not sets:
            self._toast("No saved sets for this analyte.")
            return
        sets[self.active_set_idx] = None
        deleted_idx = self.active_set_idx
        self.saved_sets[key] = sets
        self.active_set_idx = None
        self._persist_sets()
        self._refresh_set_buttons()
        self._toast("Deleted saved tank set.")

    def _on_tank_toggled(self, _checked: bool):
        """Tank clicks clear the active-set highlight."""
        if self._loading_set:
            return
        self._clear_active_set_selection()

    def _clear_active_set_selection(self):
        """Reset active set highlight and delete enablement."""
        self.active_set_idx = None
        self._update_set_button_styles()

    def _on_deselect_all(self):
        """Uncheck all tank boxes."""
        self._loading_set = True
        for cb in self.tank_checks:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._loading_set = False
        self._clear_active_set_selection()

    def _on_select_all_gravs(self):
        """Select only gravimetric tanks."""
        desired_keys = set()
        for key, meta in self._tank_metadata.items():
            use_lower = str(meta.get("use_short") or "").lower()
            if use_lower.startswith("grav"):
                desired_keys.add(str(key))
        if not desired_keys and self.tank_checks:
            # Fallback to checkbox property if metadata missing
            for cb in self.tank_checks:
                use_lower = str(cb.property("use_short") or "").lower()
                if use_lower.startswith("grav"):
                    fill_key = cb.property("fill_key") or cb.property("serial") or cb.text()
                    desired_keys.add(str(fill_key))
        self._loading_set = True
        for cb in self.tank_checks:
            cb.blockSignals(True)
            fill_key = cb.property("fill_key") or cb.property("serial") or cb.text()
            cb.setChecked(str(fill_key) in desired_keys)
            cb.blockSignals(False)
        self._loading_set = False
        self._clear_active_set_selection()

    def _filter_tanks_by_category(self, tanks: list[dict]) -> list[dict]:
        """Apply category toggle filters to tank list."""
        filtered: list[dict] = []
        show_grav = self.show_grav_cb.isChecked() if hasattr(self, "show_grav_cb") else True
        show_other = self.show_other_grav_cb.isChecked() if hasattr(self, "show_other_grav_cb") else True
        show_other_misc = self.show_other_cb.isChecked() if hasattr(self, "show_other_cb") else True
        show_cal = self.show_cal_cb.isChecked() if hasattr(self, "show_cal_cb") else True
        show_archive = self.show_archive_cb.isChecked() if hasattr(self, "show_archive_cb") else True
        for tank in tanks:
            use_lower = str(tank.get("use_short") or "").lower()
            is_other_grav = use_lower.startswith("other grav")
            is_other = use_lower == "other"
            is_grav = use_lower.startswith("grav") and not is_other_grav
            is_cal = use_lower in ("cal", "second", "tert")
            is_archive = use_lower.startswith("archive")
            if is_other_grav and not show_other:
                continue
            if is_other and not show_other_misc:
                continue
            if is_cal and not show_cal:
                continue
            if is_grav and not show_grav:
                continue
            if is_archive and not show_archive:
                continue
            filtered.append(tank)
        return filtered

    def _on_category_toggle(self, _checked: bool):
        """Refresh tank view when category filters change."""
        self.refresh_tanks()

    def _on_sort_mode_changed(self, _button, checked: bool):
        """Refresh tank view when the sort-mode radio selection changes."""
        if not getattr(self, "_ready", False):
            return
        if not checked:
            return
        self.refresh_tanks()

    def _update_set_button_styles(self):
        for idx, btn in enumerate(self.set_buttons):
            btn.blockSignals(True)
            btn.setChecked(idx == self.active_set_idx)
            btn.setStyleSheet("background-color: lightgreen;" if idx == self.active_set_idx else "")
            btn.blockSignals(False)
        self.delete_set_btn.setEnabled(self.active_set_idx is not None)

    def _toast(self, message: str):
        """Show a small tooltip-style notification."""
        QToolTip.showText(QCursor.pos(), message, self)

    def _on_tank_context(self, cb: QCheckBox, pos, serial: str):
        """Show tank metadata tooltip on right-click."""
        meta = self._tank_metadata.get(serial, {})
        fallback_serial = meta.get("serial_number") or meta.get("tank_serial_num") or serial.split("::")[0]
        tooltip = self._build_tank_tooltip(meta, fallback_serial)
        if tooltip:
            global_pos = cb.mapToGlobal(pos)
            QToolTip.showText(global_pos, tooltip, cb)

    def _annotate_next_fill_dates(self):
        """Add next_fill_date to metadata so plotting can bracket fills."""
        per_serial: dict[str, list[tuple[pd.Timestamp, str]]] = defaultdict(list)
        for key, meta in self._tank_metadata.items():
            serial_val = meta.get("serial_number") or meta.get("tank_serial_num") or str(key).split("::")[0]
            fill_date = pd.to_datetime(meta.get("fill_date") or meta.get("date"), errors="coerce")
            if pd.notnull(fill_date):
                per_serial[str(serial_val)].append((fill_date, key))

        for serial, items in per_serial.items():
            items_sorted = sorted(items, key=lambda t: t[0])
            for idx, (fill_dt, key) in enumerate(items_sorted):
                next_dt = items_sorted[idx + 1][0] if idx + 1 < len(items_sorted) else None
                if key in self._tank_metadata:
                    self._tank_metadata[key]["next_fill_date"] = next_dt

    def _annotate_recent_analysis(self):
        """Add recent_analysis (most recent hats.calibrations timestamp within
        each tank's fill window, for the currently selected analyte) to metadata.
        Used by the tank tooltip and the "Analysis Date Sort" toggle."""
        for meta in self._tank_metadata.values():
            meta["recent_analysis"] = None
        if not self.instrument or not getattr(self.instrument, "db", None):
            return
        selections = self._selected_analytes()
        if not selections:
            return
        _, pnum, _channel = selections[0]
        inst_id = self._resolve_inst_id()
        if not inst_id or pnum is None or not self._tank_metadata:
            return

        serials = sorted({
            str(meta.get("serial_number") or meta.get("tank_serial_num") or key.split("::")[0])
            for key, meta in self._tank_metadata.items()
        })
        if not serials:
            return

        cache_key = (self._tank_cache_range, int(pnum), tuple(serials))
        by_serial = self._recent_analysis_cache.get(cache_key)
        if by_serial is None:
            serial_list = ",".join("'{}'".format(s.replace("'", "''")) for s in serials)
            inst_filter = self._calibration_inst_filter(inst_id)
            sql = f"""
                SELECT c.serial_number, CONCAT(c.date, ' ', c.time) AS run_time
                FROM hats.calibrations c
                WHERE c.serial_number IN ({serial_list})
                  AND {inst_filter}
                  AND c.parameter_num = {int(pnum)}
                  AND c.mixratio IS NOT NULL
                  AND c.mixratio > -99
                  AND c.mixratio != 0
                  AND c.num >= 3
                  AND c.flag = '.'
                ORDER BY c.serial_number, run_time;
            """
            try:
                rows = self.instrument.db.doquery(sql)
            except Exception:
                return

            by_serial: dict[str, list[pd.Timestamp]] = defaultdict(list)
            for row in rows or []:
                serial = row.get("serial_number")
                dt = pd.to_datetime(row.get("run_time"), errors="coerce")
                if serial is not None and pd.notnull(dt):
                    by_serial[str(serial)].append(dt)
            self._recent_analysis_cache[cache_key] = by_serial

        for key, meta in self._tank_metadata.items():
            serial = str(meta.get("serial_number") or meta.get("tank_serial_num") or key.split("::")[0])
            times = by_serial.get(serial)
            if not times:
                continue
            fill_dt = pd.to_datetime(meta.get("fill_date") or meta.get("date"), errors="coerce")
            next_dt = meta.get("next_fill_date")
            candidates = times
            if pd.notnull(fill_dt):
                candidates = [t for t in candidates if t >= fill_dt]
            if next_dt is not None and pd.notnull(next_dt):
                candidates = [t for t in candidates if t < next_dt]
            if candidates:
                meta["recent_analysis"] = max(candidates)

    def _build_tank_tooltip(self, meta: dict, serial_fallback: str) -> str:
        """Build an HTML tooltip with tank metadata."""
        parts = []
        serial_val = meta.get("serial_number") or serial_fallback
        parts.append(f"<b>Serial Number:</b> {serial_val}")
        fill_code = meta.get("code") or meta.get("fill_code")
        if fill_code:
            parts.append(f"<b>Fill code:</b> {fill_code}")
        use_desc = meta.get("use_desc") or meta.get("use_short")
        if use_desc:
            parts.append(f"<b>Use:</b> {use_desc}")
        species = meta.get("species")
        if species:
            parts.append(f"<b>Species:</b> {species}")
        mf_value = meta.get("mf_value")
        if mf_value is not None:
            try:
                mf_val = float(mf_value)
                if not math.isnan(mf_val):
                    parts.append(f"<b>Mole Fraction:</b> {mf_val:.2f}")
            except (ValueError, TypeError):
                if mf_value not in ("", None):
                    parts.append(f"<b>Mole Fraction:</b> {mf_value}")
        fill_date = meta.get("fill_date")
        if fill_date:
            parts.append(f"<b>Fill date:</b> {fill_date}")
        recent_analysis = meta.get("recent_analysis")
        recent_str = (
            recent_analysis.strftime("%Y-%m-%d %H:%M")
            if pd.notnull(recent_analysis) else "(no data)"
        )
        parts.append(f"<b>Recent Analysis:</b> {recent_str}")
        fill_note = meta.get("fill_notes")
        if fill_note:
            parts.append(f"<b>Fill Note:</b> {fill_note}")
        grav_note = meta.get("grav_notes")
        if grav_note:
            parts.append(f"<b>Grav Note:</b> {grav_note}")
        return "<br>".join(parts)

    def _is_set_available(self, saved: dict | None) -> bool:
        """Return True if set matches current instrument and analytes."""
        if not saved:
            return False
        inst_num = getattr(self.instrument, "inst_num", None)
        saved_inst = saved.get("instrument")
        if inst_num is not None and saved_inst not in (None, inst_num):
            return False
        if not self.instrument:
            return False
        return self._find_matching_analyte_cb(saved) is not None

    def _current_analyte_key(self) -> str | None:
        """Return a stable key for the selected analyte."""
        selections = self._selected_analytes()
        if not selections or not self.instrument:
            return None
        _, pnum, channel = selections[0]
        inst_num = getattr(self.instrument, "inst_num", None)
        if inst_num is None or pnum is None:
            return None
        channel_str = channel if channel is not None else ""
        return f"{inst_num}:{pnum}:{channel_str}"

    def _sets_for_key(self, key: str | None, ensure: bool = False) -> list[dict | None]:
        """Get the list of sets for a key; optionally create."""
        if not key:
            return []
        if key not in self.saved_sets and ensure:
            self.saved_sets[key] = [None] * MAX_SAVED_SETS
        return self.saved_sets.get(key, [None] * MAX_SAVED_SETS)

    def _key_for_saved_entry(self, entry: dict) -> str | None:
        """Compute key from saved entry."""
        inst_num = entry.get("instrument")
        pnum = entry.get("parameter_num")
        channel = entry.get("channel") or ""
        if inst_num is None or pnum is None:
            return None
        return f"{inst_num}:{pnum}:{channel}"

    # --- Plotting ---
    def _resolve_inst_id(self) -> str | None:
        """Return string instrument id (e.g., 'm4'), falling back from inst_num."""
        if not self.instrument:
            return None
        inst_id = getattr(self.instrument, "inst_id", None)
        if inst_id:
            return str(inst_id)
        inst_num = getattr(self.instrument, "inst_num", None)
        mapping = getattr(self.instrument, "INSTRUMENTS", {})
        if isinstance(mapping, dict):
            for key, val in mapping.items():
                if val == inst_num:
                    return str(key)
        return None

    def _calibration_inst_filter(self, inst_id: str) -> str:
        """Return a SQL WHERE fragment (against alias c) filtering hats.calibrations by instrument."""
        calibration_inst_ids = getattr(self.instrument, "calibration_inst_ids", None)
        if calibration_inst_ids:
            return "c.inst IN ({})".format(
                ",".join(
                    f"'{str(inst).upper().replace(chr(39), chr(39) + chr(39))}'"
                    for inst in calibration_inst_ids
                )
            )
        inst_upper = str(inst_id).upper().replace("'", "''")
        return f"c.inst = '{inst_upper}'"

    def _fetch_calibration_df(
        self,
        serial: str,
        parameter_num: int,
        inst_id: str,
        include_flagged: bool = False,
    ) -> pd.DataFrame:
        """Query calibration mole fractions for a tank/parameter from hats.calibrations.

        include_flagged: when True, also return manually flagged episodes
        (flag='M') alongside the flag column, so the caller can render them as
        flagged points. The caldrift single-tank figure uses this.
        """
        serial_safe = str(serial).replace("'", "''")
        inst_filter = self._calibration_inst_filter(inst_id)
        flag_filter = "c.flag IN ('.', 'M')" if include_flagged else "c.flag = '.'"
        sql = f"""
            SELECT
                CONCAT(c.date, ' ', c.time) AS run_time,
                c.mixratio,
                c.stddev,
                c.num,
                c.run_number,
                c.inst,
                c.species,
                c.flag
            FROM hats.calibrations c
            WHERE c.serial_number = '{serial_safe}'
              AND {inst_filter}
              AND c.parameter_num = {int(parameter_num)}
              AND c.mixratio IS NOT NULL
              AND c.mixratio > -99
              AND c.mixratio != 0
              AND c.num >= 3
              AND {flag_filter}
            ORDER BY c.date, c.time;
        """
        try:
            df = pd.DataFrame(self.instrument.db.doquery(sql))
            return df
        except Exception as exc:
            self._toast(f"DB error for tank {serial}: {exc}")
            return pd.DataFrame()

    def _inst_abbr(self, inst_num) -> str:
        """Map an instrument number to its abbreviation (193 -> 'FE3'); cached."""
        if inst_num is None:
            return "?"
        if not hasattr(self, "_inst_abbr_cache"):
            self._inst_abbr_cache = {}
        if inst_num in self._inst_abbr_cache:
            return self._inst_abbr_cache[inst_num]
        abbr = str(inst_num)
        try:
            rows = self.instrument.db.doquery(
                "SELECT abbr, id FROM ccgg.inst_description WHERE num=%s LIMIT 1",
                (int(inst_num),),
            )
            if rows:
                abbr = rows[0].get("abbr") or rows[0].get("id") or str(inst_num)
        except Exception:
            pass
        self._inst_abbr_cache[inst_num] = abbr
        return abbr

    def _fetch_scale_assignment(self, serial, parameter_num, fillcode):
        """Return the current scale assignment for a tank/parameter/fill, or None.

        Joins scale_assignments_view (coefs, uncertainties, fill code, current
        flag) to the base scale_assignments table for the assigning inst_num.
        """
        if not serial or parameter_num is None or not fillcode:
            return None
        sql = """
            SELECT v.coef0, v.coef1, v.coef2, v.unc_c0, v.unc_c1, v.unc_c2,
                   v.tzero, v.scale, v.n, sa.inst_num, v.level, v.start_date
            FROM hats.scale_assignments_view v
            JOIN hats.scale_assignments sa ON sa.num = v.scale_assignment_num
            WHERE v.serial_number=%s AND v.parameter_num=%s
              AND v.fill_code=%s AND v.current_assignment=1
            ORDER BY v.assign_date DESC
            LIMIT 1
        """
        try:
            rows = self.instrument.db.doquery(
                sql, (str(serial), int(parameter_num), str(fillcode))
            )
            return rows[0] if rows else None
        except Exception:
            return None

    @staticmethod
    def _assignment_matches_fit(assign, fit) -> bool:
        """True if the stored assignment equals the current caldrift fit — same
        tzero and coefficients within tolerance (i.e. the assignment is up to
        date with the data). Tolerances are heuristic and easily tuned."""
        try:
            def close(a, b, tol):
                return abs(float(a or 0) - float(b or 0)) <= tol
            return (
                close(assign.get("tzero"), fit.tzero, 0.02)
                and close(assign.get("coef0"), fit.coef0, max(0.01, 0.001 * abs(fit.coef0)))
                and close(assign.get("coef1"), fit.coef1, 1e-3)
                and close(assign.get("coef2"), fit.coef2, 1e-4)
            )
        except Exception:
            return False

    def _overlay_caldrift_fit(self, ax, serial, species, fillcode, plot_df,
                              exclude_flagged: bool = True, parameter_num=None,
                              degree: str = "auto"):
        """Overlay caldrift's drift fit (curve + legend entry) for one tank.

        Reuses ccg_cal_db.Calibrations (the same cal source caldrift reads) and
        ccg_calfit.fitCalibrations (the fit engine caldrift calls); it does not
        shell out to or modify caldrift. The fit is restricted to this panel's
        calibration-instrument family, so it uses the same system history as
        the visible series. Called only when a single tank/fill is plotted.

        exclude_flagged: when True (default, matching caldrift) only unflagged
        ('.') cals feed the fit; when False, manually flagged ('M') episodes are
        included too. Driven by the caldrift panel's "Exclude Flagged" checkbox.

        degree: passed straight to ccg_calfit.fitCalibrations(). "auto"
        (default) matches standalone caldrift's significance-cascade
        behavior; "mean"/"linear"/"quadratic" force that exact degree with
        no fallback. Driven by the caldrift panel's "Fit degree" radio buttons.
        """
        if not species or not fillcode:
            return
        mods = _import_ccg_fit()
        if mods is None:
            return
        ccg_cal_db, ccg_calfit, ccg_dates, _ccg_refgasdb = mods

        # Calibrations() otherwise includes every instrument that measured the
        # tank.  That made an M4 plot's red fit include PR1/PR2 results which
        # were not blue points on the figure.  Keep the source selection in
        # lockstep with _fetch_calibration_df().
        calibration_inst_ids = getattr(self.instrument, "calibration_inst_ids", None)
        if calibration_inst_ids:
            syslist = ",".join(str(inst) for inst in calibration_inst_ids)
        else:
            inst_id = self._resolve_inst_id()
            syslist = str(inst_id) if inst_id else None

        try:
            cals = ccg_cal_db.Calibrations(
                tank=str(serial),
                gas=str(species),
                fillingcode=str(fillcode),
                syslist=syslist,
                database="hats",
                quiet=True,
            )
        except Exception as exc:
            self._toast(f"caldrift fit: DB error for {serial}: {exc}")
            return

        # Include unflagged cals ('.') — plus flagged ('M') episodes when the
        # panel asks to include them — and drop 9'd/None values.  The selected
        # system list above keeps these candidates aligned with the plot.
        allowed_flags = (".",) if exclude_flagged else (".", "M")
        candidates = [
            d for d in cals.cals
            if d.get("flag") in allowed_flags
            and d.get("mixratio") is not None
            and d.get("mixratio") > -800
        ]
        # Guard the shared ccg_calfit engine: it weights points by 1/unc**2,
        # so a cal with zero/blank uncertainty (single-injection num=1 rows,
        # degenerate 0.0 values) yields an infinite weight -> NaN -> a LAPACK
        # failure that crashes even the standalone caldrift. Drop those here
        # (rather than editing caldrift) and report how many so the plot can
        # tell the user the fit used fewer points than caldrift's full set.
        fit_data = [
            d for d in candidates
            if d.get("episode_unc") is not None and d.get("episode_unc") > 0
        ]
        n_excluded = len(candidates) - len(fit_data)
        if not fit_data:
            return
        try:
            # Suppress numeric RuntimeWarnings from the third-party fit engine
            # (we can't edit it); genuine failures still raise and are caught.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fit = ccg_calfit.fitCalibrations(fit_data, degree=degree)
        except Exception as exc:
            self._toast(f"caldrift fit failed for {serial}: {exc}")
            return
        if fit is None or fit.n < 1 or fit.coef0 <= -800:
            return

        # Existing scale assignment (if any) for this tank/fill/parameter --
        # looked up once here and reused both for the legend overlay below
        # and for "Save to Scale Assignments" (existing -> update, carrying
        # its start_date/level forward; none -> first-time assignment).
        assign = self._fetch_scale_assignment(serial, parameter_num, fillcode)

        # Fill date for this fillcode, needed as start_date for a first-time
        # assignment (no existing row to inherit start_date from). Mirrors
        # caldrift.py's own lookup: scan cals.fill (all fills for the tank,
        # sorted by date) for the matching code, keeping the last match if
        # a code was reused across multiple fills.
        fill_date = None
        for line in cals.fill:
            if line.get("code") == fillcode:
                fill_date = line.get("date")

        # Remember exactly what's plotted so "Save to Scale Assignments"
        # writes the same fit the user is looking at, rather than
        # recomputing (and possibly drifting from the visible curve).
        self._caldrift_last_fit = {
            "fit": fit,
            "serial": serial,
            "species": species,
            "fillcode": fillcode,
            "parameter_num": parameter_num,
            "degree": degree,
            "existing_assignment": assign,
            "fill_date": fill_date,
        }
        self._caldrift_sync_level_control()

        # Build a smooth fit curve across the plotted datetime span. Work in
        # decimal-date space (as caldrift does) so we don't round-trip the
        # interpolated points through datetime (avoids nanosecond warnings).
        dts = plot_df["datetime"].dropna()
        if dts.empty:
            return
        dd_min = ccg_dates.decimalDateFromDatetime(dts.min().to_pydatetime())
        dd_max = ccg_dates.decimalDateFromDatetime(dts.max().to_pydatetime())
        if dd_max <= dd_min:
            dd_max = dd_min + 1e-6
        steps = 200
        line_dd = [dd_min + i * (dd_max - dd_min) / (steps - 1) for i in range(steps)]
        x_dates = [ccg_dates.datetimeFromDecimalDate(dd) for dd in line_dd]
        ys = [
            fit.coef0 + fit.coef1 * (dd - fit.tzero) + fit.coef2 * (dd - fit.tzero) ** 2
            for dd in line_dd
        ]

        # Auto fit collapses to the highest significant degree.  Format its
        # coefficients exactly like the stored scale assignment below, so the
        # two results can be compared directly in the legend.
        excl = f", {n_excluded} excl." if n_excluded else ""
        parts = [f"c0={fit.coef0:.3f}±{fit.unc_c0:.3f}"]
        if fit.coef2 != 0.0:
            fit_type = "quad"
            parts.extend([
                f"c1={fit.coef1:.5f}±{fit.unc_c1:.5f}",
                f"c2={fit.coef2:.6f}±{fit.unc_c2:.6f}",
            ])
        elif fit.coef1 != 0.0:
            fit_type = "linear"
            parts.append(f"c1={fit.coef1:.5f}±{fit.unc_c1:.5f}")
        else:
            fit_type = "mean"
        label = f"caldrift ({fit_type}): {', '.join(parts)}  (n={fit.n}{excl})"

        ax.plot(
            x_dates, ys,
            color="red", linestyle="--", linewidth=1.5, zorder=1, label=label,
        )

        # When degenerate cals were dropped, surface it as a legend entry (an
        # invisible proxy line, ordered/tinted by the caller): the fit used
        # fewer points than caldrift's full unflagged set, and caldrift would
        # have errored on those points rather than skipping them.
        if n_excluded:
            ax.plot(
                [], [], linestyle="none", marker="none",
                label=(
                    f"⚠ {n_excluded} cal(s) with zero uncertainty excluded "
                    f"from fit (caldrift would error)"
                ),
            )

        # Existing scale assignment overlay, shown under the caldrift result
        # (looked up once, above). A green ✓ (tinted by the caller) marks an
        # assignment that matches this fit; otherwise it's out of date.
        if assign is not None:
            abbr = self._inst_abbr(assign.get("inst_num"))
            parts = [
                f"c0={float(assign['coef0']):.3f}±{float(assign.get('unc_c0') or 0):.3f}"
            ]
            if assign.get("coef1"):
                parts.append(
                    f"c1={float(assign['coef1']):.5f}±{float(assign.get('unc_c1') or 0):.5f}"
                )
            if assign.get("coef2"):
                parts.append(
                    f"c2={float(assign['coef2']):.6f}±{float(assign.get('unc_c2') or 0):.6f}"
                )
            n_assign = assign.get("n")
            n_str = f"(n={int(n_assign)})" if n_assign is not None else "(n=?)"
            # DejaVu Sans (matplotlib's font) has no colour-emoji glyphs, so use
            # the heavy check / heavy X and tint them green / red in the caller.
            mark = " ✔" if self._assignment_matches_fit(assign, fit) else ""
            ax.plot(
                [], [], linestyle="none", marker="none",
                label=f"scale_assignment: {', '.join(parts)}  {n_str} [{abbr}]{mark}",
            )
        else:
            ax.plot(
                [], [], linestyle="none", marker="none",
                label="scale_assignment: none on file  ✖",
            )

    # --- Caldrift flag panel (single-tank figure) ---
    def _open_caldrift_panel(self):
        """Open (or raise) the caldrift flag panel for the current figure."""
        if self._caldrift_ctx is None:
            self._toast("Plot a single tank to use caldrift flagging.")
            return
        if self._caldrift_panel is None:
            self._caldrift_panel = CaldriftPanel(self)
        cb = self._caldrift_panel._exclude_cb
        cb.blockSignals(True)
        cb.setChecked(self._caldrift_exclude_flagged)
        cb.blockSignals(False)
        hb = self._caldrift_panel._hide_cb
        hb.blockSignals(True)
        hb.setChecked(self._caldrift_hide_flagged)
        hb.blockSignals(False)
        fit_buttons = self._caldrift_panel._fit_buttons
        target_rb = fit_buttons.get(self._caldrift_fit_degree) or fit_buttons["auto"]
        target_rb.blockSignals(True)
        target_rb.setChecked(True)
        target_rb.blockSignals(False)
        self._caldrift_sync_level_control()
        ctx = self._caldrift_ctx or {}
        self._caldrift_panel.set_tank(ctx.get("serial"), ctx.get("fillcode"))
        self._caldrift_update_panel_info()
        self._caldrift_panel.show()
        self._caldrift_panel.raise_()
        self._caldrift_panel.activateWindow()

    def _caldrift_panel_open(self) -> bool:
        return self._caldrift_panel is not None and self._caldrift_panel.isVisible()

    def _caldrift_copy_serial(self):
        """Copy the current single-tank serial number to the clipboard."""
        serial = (self._caldrift_ctx or {}).get("serial")
        if not serial:
            self._toast("No tank serial available.")
            return
        try:
            QApplication.clipboard().setText(str(serial))
            self._toast(f"Copied {serial}")
        except Exception as exc:
            self._toast(f"Copy failed: {exc}")

    def _caldrift_update_panel_info(self):
        if self._caldrift_panel is not None:
            self._caldrift_panel.set_selected_count(len(self._caldrift_selected))

    def _caldrift_toggle_selection(self, idx: int, additive: bool):
        """Add/remove a plotted episode (df row index) in the flag selection."""
        if self._caldrift_ctx is None:
            return
        if additive:
            self._caldrift_selected.symmetric_difference_update({idx})
        else:
            # plain click selects only this point, or clears if re-clicked
            self._caldrift_selected = set() if self._caldrift_selected == {idx} else {idx}
        self._caldrift_redraw_highlight()
        self._caldrift_update_panel_info()

    def _caldrift_clear_selection(self):
        self._caldrift_selected = set()
        self._caldrift_redraw_highlight()
        self._caldrift_update_panel_info()

    def _caldrift_redraw_highlight(self):
        """Draw a gold ring around currently selected episodes (no full replot)."""
        ctx = self._caldrift_ctx
        if not ctx:
            return
        old = ctx.get("highlight")
        if old is not None:
            try:
                old.remove()
            except Exception:
                pass
            ctx["highlight"] = None
        ax = ctx.get("ax")
        df = ctx.get("df")
        if ax is not None and df is not None and self._caldrift_selected:
            idxs = [i for i in self._caldrift_selected if 0 <= i < len(df)]
            if idxs:
                sub = df.iloc[idxs]
                ctx["highlight"] = ax.scatter(
                    sub["datetime"], sub["mixratio"],
                    marker="o", facecolors="none", edgecolors="gold",
                    linewidths=2.0, s=140, zorder=6,
                )
        fig = ctx.get("fig")
        if fig is not None:
            fig.canvas.draw_idle()

    def _caldrift_flag_selected(self, flag_value: str):
        """Write hats.calibrations.flag for the selected episodes, then refresh."""
        ctx = self._caldrift_ctx
        if not ctx or not self._caldrift_selected:
            self._toast("Select one or more episodes first.")
            return
        df = ctx.get("df")
        if df is None:
            return
        sql = (
            "UPDATE hats.calibrations SET flag=%s "
            "WHERE serial_number=%s AND date=%s AND time=%s "
            "AND species=%s AND inst=%s AND parameter_num=%s"
        )
        n = 0
        for idx in sorted(self._caldrift_selected):
            if idx < 0 or idx >= len(df):
                continue
            row = df.iloc[idx]
            rt = pd.Timestamp(row["datetime"])
            species = row.get("species") or ctx.get("species")
            inst = row.get("inst") or str(ctx.get("inst_id", "")).upper()
            try:
                self.instrument.db.doquery(sql, (
                    flag_value, ctx.get("serial"), rt.date(), rt.time(),
                    species, inst, int(ctx.get("param_num")),
                ))
                n += 1
            except Exception as exc:
                self._toast(f"Flag write failed: {exc}")
                return
        self._caldrift_selected = set()
        self._toast(f"{'Flagged' if flag_value == 'M' else 'Unflagged'} {n} episode(s).")
        self._caldrift_refresh_fit()

    def _caldrift_set_exclude(self, checked: bool):
        self._caldrift_exclude_flagged = bool(checked)
        self._caldrift_refresh_fit()

    def _caldrift_set_hide(self, checked: bool):
        self._caldrift_hide_flagged = bool(checked)
        self._caldrift_refresh_fit()

    def _caldrift_set_fit_degree(self, degree):
        self._caldrift_fit_degree = degree or "auto"
        self._caldrift_refresh_fit()

    def _caldrift_set_level(self, level):
        # Doesn't affect the plotted fit, so no refresh -- just remember the
        # choice for the next "Save to Scale Assignments" on a new tank.
        self._caldrift_level = level or "Tertiary"

    def _caldrift_sync_level_control(self):
        """Push first-time-vs-update level state to the panel, if open.

        Enabled + the user's last-picked level when this tank/fill has no
        existing assignment (a first-time save); disabled + the existing
        assignment's own level when it does (an update always carries that
        level forward, matching caldrift's own behavior).
        """
        if self._caldrift_panel is None:
            return
        existing = (self._caldrift_last_fit or {}).get("existing_assignment")
        if existing is not None:
            self._caldrift_panel.set_level_state(False, existing.get("level"))
        else:
            self._caldrift_panel.set_level_state(True, self._caldrift_level)

    def _caldrift_refresh_fit(self):
        """Re-fetch data (picking up flag changes) and re-run the fit/redraw."""
        if self._caldrift_refresh is not None:
            self._caldrift_refresh()

    def _caldrift_save_scale_assignment(self):
        """Write the currently plotted caldrift fit to hats.scale_assignments.

        Reuses ccg_refgasdb.refgas.insertFromFit() -- the exact write path
        caldrift's own -u/--update CLI option calls -- rather than a
        hand-rolled INSERT. That call has no concept of inst_num (the
        column defaults to 58/PR1 regardless of which system's calibration
        history fed the fit), so we patch inst_num in with a follow-up
        UPDATE on the same connection right after the insert, recording
        which instrument this panel's fit actually came from.

        For a first-time assignment (no existing row for this tank/fill),
        start_date comes from the tank's fill record instead (cals.fill,
        looked up in _overlay_caldrift_fit -- the same source caldrift.py's
        CLI uses) and level comes from the panel's LEVEL picker, which is
        only enabled in this case.
        """
        info = self._caldrift_last_fit
        if not info:
            self._toast("No caldrift fit to save. Plot a single tank first.")
            return
        fit = info["fit"]
        serial = info["serial"]
        species = info["species"]
        fillcode = info["fillcode"]
        degree = info["degree"]
        existing = info.get("existing_assignment")

        inst_num = getattr(self.instrument, "inst_num", None)
        inst_abbr = self._inst_abbr(inst_num)

        if existing is not None and existing.get("start_date"):
            start_date = existing["start_date"]
            level = existing.get("level") or "Tertiary"
            kind = "Updating existing assignment"
        else:
            fill_date = info.get("fill_date")
            if not fill_date:
                QMessageBox.warning(
                    self, "Save to Scale Assignments",
                    f"No existing scale assignment AND no fill record found for "
                    f"{serial} (fill {fillcode}), {species} -- can't determine a "
                    "start date. Use the caldrift CLI (-u) instead.",
                )
                return
            start_date = fill_date
            level = self._caldrift_level
            kind = "First-time assignment"

        fit_type = "quad" if fit.coef2 != 0.0 else ("linear" if fit.coef1 != 0.0 else "mean")
        parts = [f"c0={fit.coef0:.3f}±{fit.unc_c0:.3f}"]
        if fit.coef1:
            parts.append(f"c1={fit.coef1:.5f}±{fit.unc_c1:.5f}")
        if fit.coef2:
            parts.append(f"c2={fit.coef2:.6f}±{fit.unc_c2:.6f}")
        summary = (
            f"{kind}\n"
            f"Tank: {serial}  (fill {fillcode})\n"
            f"Species: {species}\n"
            f"Fit ({fit_type}, degree={degree}):  {', '.join(parts)}\n"
            f"n={fit.n}\n"
            f"Level: {level}   Start date: {start_date}\n"
            f"Instrument: {inst_abbr} (inst_num={inst_num})"
        )
        reply = QMessageBox.question(
            self, "Save to Scale Assignments",
            f"Save this caldrift fit as the new scale assignment?\n\n{summary}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        mods = _import_ccg_fit()
        if mods is None:
            QMessageBox.critical(
                self, "Save to Scale Assignments",
                "CCG modules unavailable; cannot save.",
            )
            return
        _ccg_cal_db, _ccg_calfit, _ccg_dates, ccg_refgasdb = mods

        try:
            ref = ccg_refgasdb.refgas(
                sp=species, sn=[serial], database="hats", readonly=False,
            )
            comment = f"logos_tanks caldrift panel ({degree} fit, {inst_abbr})"
            idx = ref.insertFromFit(
                serial, start_date, fit, level=level, comment=comment
            )
            ref.db.doquery(
                "UPDATE hats.scale_assignments SET inst_num=%s WHERE num=%s",
                (inst_num, idx),
            )
        except (Exception, SystemExit) as exc:
            # SystemExit is caught too: the CCG refgasdb/dbutils modules call
            # sys.exit() on some lookup failures, which would otherwise take
            # the whole logos_tanks process down with it.
            QMessageBox.critical(
                self, "Save to Scale Assignments",
                f"Save failed for {serial}:\n{exc}",
            )
            return

        QMessageBox.information(
            self, "Save to Scale Assignments",
            f"Saved scale assignment #{idx} for {serial} (fill {fillcode}), {species}.\n"
            f"Instrument: {inst_abbr}",
        )
        self._caldrift_refresh_fit()

    def _caldrift_on_panel_closed(self):
        """Panel closed: drop selection + highlight, and revert to
        caldrift's own auto fit -- overriding the fit degree is only in
        effect while the panel is open."""
        self._caldrift_selected = set()
        self._caldrift_redraw_highlight()
        self._caldrift_fit_degree = "auto"
        if not self._caldrift_suppress_close_refresh:
            self._caldrift_refresh_fit()

    def _on_plot_tanks(self):
        """Pop up a matplotlib figure with mole fraction history for selected tanks."""
        if not self.instrument:
            self._toast("Instrument not configured.")
            return
        selections = self._selected_analytes()
        if not selections:
            self._toast("Select a gas before plotting.")
            return
        parameter_name, parameter_num, channel = selections[0]
        if parameter_num is None:
            self._toast("Invalid parameter number for selection.")
            return
        tank_keys = self.selected_tanks()
        if not tank_keys:
            self._toast("Select at least one tank to plot.")
            return
        inst_id = self._resolve_inst_id()
        if not inst_id:
            self._toast("Instrument id unavailable; cannot query calibrations.")
            return

        # A previous caldrift figure's panel/context is now stale. Suppress
        # the close-triggered refresh -- a brand-new plot is about to replace
        # it, so redrawing the old one first would just be wasted/glitchy.
        if self._caldrift_panel is not None:
            self._caldrift_suppress_close_refresh = True
            try:
                self._caldrift_panel.close()
            except Exception:
                pass
            finally:
                self._caldrift_suppress_close_refresh = False
            self._caldrift_panel = None
        self._caldrift_ctx = None
        self._caldrift_last_fit = None
        self._caldrift_selected = set()

        btn = getattr(self, "plot_tanks_btn", None)
        if btn:
            btn.setText("Loading...")
            btn.setStyleSheet(
                "background-color: #f6e7a1; "
                "border: 1px solid #524b2f; "
                "padding: 3px 6px; "
                "color: #524b2f;")
            btn.setEnabled(False)
        # Force a repaint so the user sees the busy state before the DB call.
        try:
            QApplication.processEvents()
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=(9, 5))
        pick_map: dict[mlines.Line2D, dict] = {}
        state = {
            "parameter_name": parameter_name,
            "parameter_num": parameter_num,
            "channel": channel,
            "pick_map": pick_map,
        }
        _saved_xlim: list = [None]   # persists across analyte switches

        try:
            def _plot_for(param_name: str, param_num: int, param_channel: str | None,
                          *, close_on_empty: bool = False) -> bool:
                nonlocal pick_map
                state["parameter_name"] = param_name
                state["parameter_num"] = param_num
                state["channel"] = param_channel

                # Capture the live xlim (reflects any user pan/zoom since last plot).
                # _saved_xlim[0] is None only before the first successful plot.
                prev_xlim = ax.get_xlim() if _saved_xlim[0] is not None else None
                ax.clear()
                pick_map = {}
                any_data = False
                all_inst_dts: list[tuple] = []   # (datetime, inst) across all tanks
                label_to_serial: dict = {}       # legend label -> serial (copy on click)
                # When exactly one tank/fill is plotted, remember what we need
                # to overlay caldrift's drift fit after the loop.
                single_ctx = None
                if len(tank_keys) == 1:
                    # Reset per-replot caldrift state; repopulated below.
                    self._caldrift_ctx = None
                    self._caldrift_last_fit = None
                    self._caldrift_selected = set()

                for fill_key in tank_keys:
                    meta = self._tank_metadata.get(str(fill_key), {})
                    serial = (
                        meta.get("serial_number")
                        or meta.get("tank_serial_num")
                        or str(fill_key).split("::")[0]
                    )
                    df = self._fetch_calibration_df(
                        serial, param_num, inst_id,
                        include_flagged=(len(tank_keys) == 1),
                    )
                    fill_date = meta.get("fill_date") or meta.get("date")
                    fill_code = meta.get("code") or meta.get("fill_code")
                    next_fill_date = meta.get("next_fill_date")
                    if df is None or df.empty:
                        continue
                    df = df.copy()
                    # This logic is for the old calibration table; the new view already has a run_time timestamp column
                    #date_part = pd.to_datetime(df["date"], errors="coerce")
                    #time_part = pd.to_timedelta(df["time"].astype(str).str.strip(), errors="coerce")
                    #if time_part.isna().all():
                    #    time_part = pd.to_timedelta(0)
                    #df["datetime"] = date_part + time_part
                    df["datetime"] = pd.to_datetime(df["run_time"], errors="coerce")
                    df["mixratio"] = pd.to_numeric(df["mixratio"], errors="coerce")
                    df["stddev"] = pd.to_numeric(df["stddev"], errors="coerce")
                    if fill_date:
                        fill_dt = pd.to_datetime(fill_date, errors="coerce")
                        if pd.notnull(fill_dt):
                            df = df[df["datetime"] >= fill_dt]
                    if next_fill_date:
                        next_dt = pd.to_datetime(next_fill_date, errors="coerce")
                        if pd.notnull(next_dt):
                            df = df[df["datetime"] < next_dt]
                    df = df.dropna(subset=["datetime", "mixratio"]).sort_values("datetime")
                    if df.empty:
                        continue
                    df = df.reset_index(drop=True)

                    # Hide-flagged: drop flagged episodes entirely so the line,
                    # legend, selection and autoscale reflect only unflagged data.
                    if (len(tank_keys) == 1 and self._caldrift_hide_flagged
                            and "flag" in df.columns):
                        df = df[df["flag"] == "."].reset_index(drop=True)
                        if df.empty:
                            continue

                    if "inst" in df.columns:
                        all_inst_dts.extend(zip(df["datetime"], df["inst"]))

                    err_label = f"{serial} ({fill_code})" if fill_code else str(serial)
                    label_to_serial[err_label] = serial
                    err_container = ax.errorbar(
                        df["datetime"],
                        df["mixratio"],
                        yerr=df["stddev"] if "stddev" in df.columns else None,
                        fmt="o-",
                        markersize=4,
                        linewidth=1,
                        capsize=3,
                        label=err_label,
                    )
                    line = err_container.lines[0] if err_container.lines else None
                    line_color = line.get_color() if line is not None else "C0"
                    if line is not None:
                        line.set_picker(5)
                        pick_map[line] = {
                            "df": df,
                            "serial": serial,
                            "fill_code": fill_code,
                        }
                    any_data = True

                    if len(tank_keys) == 1:
                        # Overlay manually flagged (flag='M') episodes as
                        # logos_data-style rejected points (hollow whitesmoke
                        # circle, coloured edge). Hidden from the legend.
                        if "flag" in df.columns:
                            flagged = df[df["flag"] == "M"]
                            if not flagged.empty:
                                ax.scatter(
                                    flagged["datetime"], flagged["mixratio"],
                                    marker="o", facecolors="whitesmoke",
                                    edgecolors=line_color, linewidths=1.5,
                                    s=55, zorder=4, label="_flagged",
                                )
                        cal_species = None
                        if "species" in df.columns and not df["species"].dropna().empty:
                            cal_species = str(df["species"].dropna().iloc[0])
                        single_ctx = {
                            "serial": serial,
                            "species": cal_species,
                            "fillcode": fill_code,
                            "df": df,
                        }
                        # Publish live context for the caldrift panel.
                        self._caldrift_ctx = {
                            "fig": fig, "ax": ax, "df": df,
                            "serial": serial, "species": cal_species,
                            "fillcode": fill_code, "param_num": param_num,
                            "channel": param_channel, "inst_id": inst_id,
                            "highlight": None,
                        }

                state["pick_map"] = pick_map

                # Draw instrument-transition vertical lines when the calibration
                # data spans multiple instruments (e.g. PR1→PR2, m3→M4).
                if all_inst_dts:
                    inst_df = pd.DataFrame(all_inst_dts, columns=["datetime", "inst"])
                    inst_df = inst_df.dropna().sort_values("datetime")
                    # Normalise case so m3/M3 etc. are treated as the same group
                    inst_df["inst_upper"] = inst_df["inst"].str.upper()
                    seen: set = set()
                    for _, row in inst_df.iterrows():
                        key = row["inst_upper"]
                        if key not in seen:
                            if seen:   # not the very first instrument
                                label = "Transition" if not any(
                                    isinstance(a, mlines.Line2D)
                                    and a.get_label() == "Transition"
                                    for a in ax.lines
                                ) else "_"
                                ax.axvline(row["datetime"], color="darkblue",
                                           linewidth=1.5, linestyle="--", alpha=0.4,
                                           zorder=0, label=label)
                                ax.text(row["datetime"], 1.0, f" {row['inst']}",
                                        transform=ax.get_xaxis_transform(),
                                        color="darkblue", fontsize=8,
                                        va="bottom", ha="left", clip_on=True)
                            seen.add(key)

                species = None
                try:
                    pnum_int = int(param_num)
                    species = getattr(self.instrument, "analytes_inv", {}).get(pnum_int)
                except Exception:
                    species = None
                # Channel-stripped labels for on-screen text.
                disp_name = self._display_label(param_name)
                label_species = self._display_label(species or param_name or "Species")
                ax.set_xlabel("Datetime")
                ax.set_ylabel(f"Mole Fraction ({label_species})")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
                ax.grid(True, alpha=0.3)

                if not any_data:
                    ax.set_title(f"No calibration data for {disp_name} ({inst_id.upper()})")
                    ax.text(
                        0.5,
                        0.5,
                        "No calibration data found.",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    _saved_xlim[0] = None
                    fig.canvas.draw_idle()
                    if close_on_empty:
                        plt.close(fig)
                    return False

                ax.set_title(f"Calibrations for {disp_name} ({inst_id.upper()})")
                if single_ctx is not None:
                    self._overlay_caldrift_fit(
                        ax,
                        single_ctx["serial"],
                        single_ctx["species"],
                        single_ctx["fillcode"],
                        single_ctx["df"],
                        exclude_flagged=self._caldrift_exclude_flagged,
                        parameter_num=param_num,
                        degree=self._caldrift_fit_degree,
                    )

                # Legend order: tank series first, then the caldrift fit, then
                # the zero-uncertainty warning (if _overlay_caldrift_fit added
                # one). Stable within each rank, so multi-tank order is kept.
                handles, labels = ax.get_legend_handles_labels()

                def _legend_rank(lbl):
                    if lbl.startswith("caldrift"):
                        return 1
                    if lbl.startswith("scale_assignment"):
                        return 2
                    if lbl.startswith("⚠"):
                        return 3
                    return 0

                order = sorted(range(len(labels)), key=lambda i: _legend_rank(labels[i]))
                legend = ax.legend(
                    [handles[i] for i in order],
                    [labels[i] for i in order],
                )
                fig._legend_serials = {}
                for text in legend.get_texts():
                    t = text.get_text()
                    if t.startswith("⚠"):
                        text.set_color("darkorange")
                    elif t.startswith("scale_assignment"):
                        if "✔" in t:
                            text.set_color("green")
                        elif "✖" in t:
                            text.set_color("red")
                    # Make tank legend entries clickable to copy their serial.
                    s = label_to_serial.get(text.get_text())
                    if s:
                        text.set_picker(True)
                        fig._legend_serials[text.get_text()] = s
                fig.autofmt_xdate()
                fig.tight_layout()

                # Compute autoscaled limits for the new data.
                ax.relim()
                ax.autoscale()
                auto_ylim = ax.get_ylim()

                # Push autoscaled state as the toolbar Home view so pressing
                # Home always resets to natural autoscale for the current data.
                toolbar = getattr(getattr(fig.canvas, "manager", None), "toolbar", None)
                if toolbar is not None and hasattr(toolbar, "update"):
                    toolbar.update()
                    if hasattr(toolbar, "push_current"):
                        toolbar.push_current()

                # Restore the user's x-zoom (y always rescales).
                if prev_xlim is not None:
                    ax.set_xlim(prev_xlim)
                ax.set_ylim(auto_ylim)

                # Mark that at least one successful plot has been drawn.
                _saved_xlim[0] = True

                fig.canvas.draw_idle()
                return True

            if not _plot_for(parameter_name, parameter_num, channel, close_on_empty=True):
                self._toast("No calibration data found for the selected tanks/parameter.")
                return

            # Use the preferred-channel-filtered analyte list (same as the
            # checkbox grid) so a species measured on two channels isn't listed
            # twice; show the channel-stripped label but keep the full name as
            # item data for parameter/channel lookups.
            analyte_full_names = list(self._analyte_names) if self._analyte_names else [parameter_name]
            analyte_combo = QComboBox()
            for full in analyte_full_names:
                analyte_combo.addItem(self._display_label(full), full)
            idx = analyte_combo.findData(parameter_name)
            if idx < 0:
                idx = analyte_combo.findText(self._display_label(parameter_name), Qt.MatchExactly)
            if idx >= 0:
                analyte_combo.setCurrentIndex(idx)

            reload_button = QPushButton("Reload")

            def _prev_analyte():
                idx = analyte_combo.currentIndex()
                if idx > 0:
                    analyte_combo.setCurrentIndex(idx - 1)

            def _next_analyte():
                idx = analyte_combo.currentIndex()
                if idx < analyte_combo.count() - 1:
                    analyte_combo.setCurrentIndex(idx + 1)

            fig._prev_sc = QShortcut(QKeySequence("Ctrl+Shift+Up"), fig.canvas)
            fig._prev_sc.activated.connect(_prev_analyte)
            fig._next_sc = QShortcut(QKeySequence("Ctrl+Shift+Down"), fig.canvas)
            fig._next_sc.activated.connect(_next_analyte)

            def _do_plot_for_analyte(name: str):
                if not name:
                    return
                pnum = (self.instrument.analytes or {}).get(name)
                if pnum is None:
                    self._toast("Invalid parameter number for selection.")
                    return
                channel = self._analyte_channel(name)
                analyte_combo.setEnabled(False)
                reload_button.setText("Loading...")
                reload_button.setStyleSheet(
                    "background-color: #f6e7a1; "
                    "border: 1px solid #524b2f; "
                    "padding: 3px 6px; "
                    "color: #524b2f;")
                reload_button.setEnabled(False)

                try:
                    QApplication.processEvents()
                except Exception:
                    pass
                try:
                    if not _plot_for(name, pnum, channel):
                        self._toast("No calibration data found for the selected tanks/parameter.")
                finally:
                    analyte_combo.setEnabled(True)
                    reload_button.setText("Reload")
                    reload_button.setStyleSheet("")
                    reload_button.setEnabled(True)

            def _on_combo_changed(_text=None):
                # Use item data (full name incl. channel), not the shown text.
                _do_plot_for_analyte(analyte_combo.currentData())

            def _on_reload_clicked():
                _do_plot_for_analyte(analyte_combo.currentData())

            analyte_combo.currentTextChanged.connect(_on_combo_changed)
            reload_button.clicked.connect(_on_reload_clicked)
            fig._analyte_combo = analyte_combo

            combo_container = QWidget()
            combo_layout = QHBoxLayout()
            combo_layout.setContentsMargins(0, 0, 0, 0)
            combo_layout.setSpacing(4)
            combo_layout.addWidget(analyte_combo)
            combo_layout.addWidget(reload_button)
            # Caldrift flag-panel button, to the left of the analyte combo
            # (single-tank figures only — that's where caldrift results exist).
            if len(tank_keys) == 1:
                caldrift_btn = QPushButton("caldrift")
                caldrift_btn.setToolTip("Open the caldrift panel to flag/exclude episodes.")
                caldrift_btn.clicked.connect(self._open_caldrift_panel)
                combo_layout.insertWidget(0, caldrift_btn)
                fig._caldrift_button = caldrift_btn
                self._caldrift_refresh = lambda: _do_plot_for_analyte(analyte_combo.currentData())
            combo_container.setLayout(combo_layout)

            toolbar = getattr(getattr(fig.canvas, "manager", None), "toolbar", None)
            if toolbar is not None and hasattr(toolbar, "addWidget"):
                spacer = QWidget()
                spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                toolbar.addWidget(spacer)
                toolbar.addWidget(combo_container)
                fig._analyte_combo_spacer = spacer
            else:
                combo_container.setParent(fig.canvas)
                combo_container.show()

                def _reposition_combo(_event=None):
                    margin = 12
                    x = max(margin, fig.canvas.width() - combo_container.sizeHint().width() - margin)
                    combo_container.move(x, margin)

                fig.canvas.mpl_connect("resize_event", _reposition_combo)
                _reposition_combo()

            def _format_value(value, digits: int = 6) -> str:
                if value is None:
                    return "N/A"
                try:
                    if isinstance(value, float) and math.isnan(value):
                        return "N/A"
                except TypeError:
                    pass
                if isinstance(value, (int,)):
                    return str(value)
                if isinstance(value, (float,)):
                    return f"{value:.{digits}f}"
                return str(value)

            def _format_run_dt(row: pd.Series) -> str:
                dt_val = row.get("datetime")
                dt = pd.to_datetime(dt_val, errors="coerce")
                if pd.isna(dt):
                    date_part = row.get("date")
                    time_part = row.get("time")
                    if date_part is not None or time_part is not None:
                        return f"{date_part} {time_part}".strip()
                    return "N/A"
                return dt.strftime("%Y-%m-%d %H:%M:%S")

            def _line_numeric_points(line: mlines.Line2D):
                if not hasattr(line, "get_xdata") or not hasattr(line, "get_ydata"):
                    return None
                try:
                    xdata = line.get_xdata(orig=False)
                except TypeError:
                    xdata = line.get_xdata()
                try:
                    ydata = line.get_ydata(orig=False)
                except TypeError:
                    ydata = line.get_ydata()
                if len(xdata) == 0:
                    return None
                x_series = pd.Series(xdata)
                y_series = pd.Series(ydata)
                x_num = pd.to_numeric(x_series, errors="coerce")
                y_num = pd.to_numeric(y_series, errors="coerce")
                mask = x_num.notna() & y_num.notna()
                if not mask.any():
                    return None
                x_vals = x_num[mask].tolist()
                y_vals = y_num[mask].tolist()
                idxs = list(x_num[mask].index)
                return x_vals, y_vals, idxs

            def _line_xy_at(line: mlines.Line2D, idx: int):
                try:
                    xdata = line.get_xdata(orig=False)
                except TypeError:
                    xdata = line.get_xdata()
                try:
                    ydata = line.get_ydata(orig=False)
                except TypeError:
                    ydata = line.get_ydata()
                if idx >= len(xdata) or idx >= len(ydata):
                    return None
                x_val = pd.to_numeric(pd.Series([xdata[idx]]), errors="coerce").iloc[0]
                y_val = pd.to_numeric(pd.Series([ydata[idx]]), errors="coerce").iloc[0]
                if pd.isna(x_val) or pd.isna(y_val):
                    return None
                return x_val, y_val

            def _nearest_point(line: mlines.Line2D, mouseevent, max_px: int = 10):
                if not line.get_visible():
                    return None
                pts_data = _line_numeric_points(line)
                if pts_data is None:
                    return None
                x_vals, y_vals, idxs = pts_data
                pts = ax.transData.transform(list(zip(x_vals, y_vals)))
                mx, my = mouseevent.x, mouseevent.y
                best_idx = None
                best_dist = None
                for i, (px, py) in enumerate(pts):
                    dx = px - mx
                    dy = py - my
                    dist = (dx * dx + dy * dy) ** 0.5
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_dist is None or best_dist > max_px:
                    return None
                return idxs[best_idx]

            def _on_press(event):
                if event.button not in (1, 3):
                    return
                best_line = None
                best_idx = None
                best_dist = None
                for line in state.get("pick_map", {}):
                    idx = _nearest_point(line, event, max_px=10)
                    if idx is None:
                        continue
                    xy = _line_xy_at(line, idx)
                    if xy is None:
                        continue
                    px, py = ax.transData.transform(xy)
                    dx = px - event.x
                    dy = py - event.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best_line = line
                        best_idx = idx

                # Caldrift flag-selection mode: while the panel is open, a
                # left-click selects/toggles episodes for flagging (SHIFT adds
                # more) instead of showing a tooltip. Right-click still navigates.
                if event.button == 1 and len(tank_keys) == 1 and self._caldrift_panel_open():
                    additive = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
                    if best_idx is None:
                        if not additive:
                            self._caldrift_clear_selection()
                        return
                    self._caldrift_toggle_selection(best_idx, additive)
                    return

                if best_line is None or best_idx is None:
                    QToolTip.hideText()
                    return
                data = state.get("pick_map", {}).get(best_line)
                if not data:
                    QToolTip.hideText()
                    return
                df_line = data["df"]
                if best_idx < 0 or best_idx >= len(df_line):
                    QToolTip.hideText()
                    return
                row = df_line.iloc[best_idx]
                serial_val = data.get("serial")
                fill_code = data.get("fill_code")
                run_dt = _format_run_dt(row)
                mixratio = _format_value(row.get("mixratio"), 3)
                stddev = _format_value(row.get("stddev"), 3)
                num_samples = _format_value(row.get("num"))
                run_time = row.get("datetime", None)
                text = (
                    f"<b>Serial number:</b> {serial_val}<br>"
                    f"<b>Fill code:</b> {fill_code}<br>"
                    f"<b>Run date/time:</b> {run_dt}<br>"
                    f"<b>Mixing ratio:</b> {mixratio}<br>"
                    f"<b>Standard deviation:</b> {stddev}<br>"
                    f"<b>Number of samples:</b> {num_samples}"
                )
                QToolTip.showText(QCursor.pos(), text)
                # Right mouse button: set main window to this tank/run
                if event.button == 3 and self.main_window is not None:
                    analyte = state.get("parameter_name")
                    try:
                        self.main_window.current_run_time = str(run_time)
                    except Exception:
                        self.main_window.current_run_time = str(run_dt)
                    try:
                        self.main_window.current_pnum = int(state.get("parameter_num"))
                    except Exception:
                        pass
                    self.main_window.current_channel = state.get("channel")

                    if hasattr(self.main_window, "radio_group") and self.main_window.radio_group:
                        for rb in self.main_window.radio_group.buttons():
                            if rb.text() == analyte:
                                rb.setChecked(True)
                                break
                    elif hasattr(self.main_window, "analyte_combo"):
                        idx = self.main_window.analyte_combo.findText(analyte, Qt.MatchExactly)
                        if idx >= 0:
                            self.main_window.analyte_combo.setCurrentIndex(idx)

                    if hasattr(self.main_window, "tabs"):
                        self.main_window.tabs.setCurrentIndex(0)

                    if not isinstance(run_time, pd.Timestamp):
                        run_time = pd.to_datetime(run_time)

                    end_year = run_time.year
                    end_month = run_time.month
                    start_dt = (run_time - pd.DateOffset(months=1))
                    start_year = start_dt.year
                    start_month = start_dt.month

                    self.main_window.end_year_cb.setCurrentText(str(end_year))
                    self.main_window.end_month_cb.setCurrentIndex(end_month - 1)
                    self.main_window.start_year_cb.setCurrentText(str(start_year))
                    self.main_window.start_month_cb.setCurrentIndex(start_month - 1)

                    self.main_window.runTypeCombo.blockSignals(True)
                    self.main_window.runTypeCombo.setCurrentText("All")
                    self.main_window.runTypeCombo.blockSignals(False)

                    self.main_window.set_runlist(initial_date=run_time)
                    self.main_window.on_plot_type_changed(self.main_window.current_plot_type)
                    self.main_window.current_run_time = str(run_time)
                    self.main_window.apply_date_btn.setStyleSheet("")

            def _on_release(event):
                if event.button == 1:
                    QToolTip.hideText()

            def _on_legend_pick(event):
                # Click a tank's legend entry to copy its serial to the clipboard.
                artist = event.artist
                if not hasattr(artist, "get_text"):
                    return
                serial = getattr(fig, "_legend_serials", {}).get(artist.get_text())
                if not serial:
                    return
                try:
                    QApplication.clipboard().setText(str(serial))
                    self._toast(f"Copied {serial}")
                except Exception:
                    pass

            def _on_fig_close(_evt):
                # Closing the figure dismisses its caldrift panel + context.
                # Suppress the close-triggered refresh -- the figure being
                # closed is exactly what a refresh would try to redraw into.
                if self._caldrift_panel is not None:
                    self._caldrift_suppress_close_refresh = True
                    try:
                        self._caldrift_panel.close()
                    except Exception:
                        pass
                    finally:
                        self._caldrift_suppress_close_refresh = False
                    self._caldrift_panel = None
                self._caldrift_ctx = None
                self._caldrift_last_fit = None
                self._caldrift_selected = set()

            fig.canvas.mpl_connect("button_press_event", _on_press)
            fig.canvas.mpl_connect("button_release_event", _on_release)
            fig.canvas.mpl_connect("close_event", _on_fig_close)
            fig.canvas.mpl_connect("pick_event", _on_legend_pick)
            fig.show()
        finally:
            if btn:
                btn.setText("Plot Tanks")
                btn.setStyleSheet("")
                btn.setEnabled(True)


if __name__ == "__main__":
    """
    Minimal harness to exercise the widget standalone using the real M4 instrument/DB.
    Falls back to fake data if initialization fails (e.g., DB connectivity issues).
    """
    import argparse
    from PyQt5.QtWidgets import QApplication

    try:
        from logos_instruments import M4_Instrument, FE3_Instrument, BLD1_Instrument, Perseus_Instrument
    except Exception as exc:  # pragma: no cover - import-time failure path
        print(f"Could not import instrument classes: {exc}", file=sys.stderr)
        M4_Instrument = FE3_Instrument = BLD1_Instrument = Perseus_Instrument = None  # type: ignore

    VALID_INSTRUMENTS = ["m4", "fe3", "bld1", "prs"]

    def _default_instrument() -> str | None:
        inst = _read_config().get("default_inst")
        return str(inst).lower() if inst else None

    def _save_default_instrument(inst_key: str) -> None:
        data = _read_config()
        data["default_inst"] = inst_key
        try:
            _write_config(data)
        except Exception as exc:
            print(f"Could not save default_inst to {CONFIG_PATH}: {exc}", file=sys.stderr)

    def _prompt_and_save_instrument() -> str:
        opts = ", ".join(VALID_INSTRUMENTS)
        while True:
            try:
                choice = input(
                    f"No default instrument set.\n"
                    f"Choose one [{opts}]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                sys.exit(1)
            if choice in VALID_INSTRUMENTS:
                _save_default_instrument(choice)
                print(f"Saved default_inst = {choice} to {CONFIG_PATH}")
                return choice
            print(f'  Invalid choice "{choice}". Pick from: {opts}')

    def build_instrument(inst_key: str):
        """Factory to build an instrument by key, with fallback."""
        inst_key = inst_key.lower()
        class_map = {
            "m4": M4_Instrument,
            "fe3": FE3_Instrument,
            "bld1": BLD1_Instrument,
            "prs": Perseus_Instrument,
        }
        cls = class_map.get(inst_key)
        if cls is None:
            print(f"Unknown instrument '{inst_key}', using fake instrument.", file=sys.stderr)
            return None
        try:
            return cls()
        except Exception as exc:
            print(f"Failed to initialize {cls.__name__}: {exc}", file=sys.stderr)
            return None

    parser = argparse.ArgumentParser(prog="logos_tanks", description="TanksWidget test harness")
    parser.add_argument(
        "instrument",
        nargs="?",
        choices=VALID_INSTRUMENTS,
        help=f"Instrument to load: {VALID_INSTRUMENTS}. Defaults to default_inst in {CONFIG_PATH}.",
    )
    args = parser.parse_args()
    inst_key = args.instrument or _default_instrument()
    if inst_key not in VALID_INSTRUMENTS:
        inst_key = _prompt_and_save_instrument()
    elif args.instrument:
        _save_default_instrument(inst_key)

    app = QApplication(sys.argv)
    instrument = build_instrument(inst_key)
    if instrument is None:
        print("Could not initialize instrument; exiting.", file=sys.stderr)
        sys.exit(1)
    widget = TanksWidget(instrument=instrument)
    widget.setWindowTitle(f"TanksWidget Test Harness ({inst_key.upper()})")
    # 320x420 was cramped even before the responsive checkbox grids (which
    # need real width to reflow into more than their 3-column floor). Size
    # against the screen like logos_data's main window does.
    screen = app.primaryScreen()
    avail = screen.availableGeometry() if screen else None
    if avail is not None:
        target_w = min(1000, avail.width())
        target_h = min(800, avail.height())
        widget.resize(target_w, target_h)
        widget.move(avail.x() + (avail.width() - target_w) // 2,
                    avail.y() + (avail.height() - target_h) // 2)
    else:
        widget.resize(1000, 800)
    widget.show()
    sys.exit(app.exec_())
    
