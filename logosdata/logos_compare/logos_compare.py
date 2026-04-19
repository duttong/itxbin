#!/usr/bin/env python3
"""Compare LOGOS measurement programs for a selected analyte.

This is an initial PyQt5 application built around the data-loading methods in
``logosdata/logos_timeseries.py``.  It intentionally uses monthly means as the
common comparison product so flask, insitu, and predecessor records can share a
single plotting path.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from pathlib import Path

# Suppress Mesa GLX warnings on hosts with limited X/GL.
os.environ.setdefault("QT_XCB_GL_INTEGRATION", "none")

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QButtonGroup,
    QCompleter,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.lines as mlines


LOGOSDATA = Path(__file__).resolve().parents[1]
ROOT = LOGOSDATA.parent
if str(LOGOSDATA) not in sys.path:
    sys.path.insert(0, str(LOGOSDATA))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logos_timeseries import LOGOS_sites, PFP_SITES, TimeseriesWidget, build_site_colors  # noqa: E402
from logos_instruments import (  # noqa: E402
    FE3_Instrument,
    HATS_DB_Functions,
    IE3_Instrument,
    M4_Instrument,
)


PROGRAMS = {
    "mstar": {"label": "M*", "inst_id": "m4", "inst_num": 192, "loader": "m4"},
    "fecd": {"label": "fECD", "inst_id": "fe3", "inst_num": 193, "loader": "fe3"},
    "ie3": {"label": "IE3", "inst_id": "ie3", "inst_num": 236, "loader": "ie3"},
    "pr1": {"label": "PR1", "inst_id": "pr1", "inst_num": 58, "loader": "pr1"},
}

DEFAULT_SITE_SELECTION = {"BRW", "MLO", "SMO", "SPO"}
PROGRAM_MARKERS = {
    "mstar": "o",
    "fecd": "s",
    "ie3": "^",
    "pr1": "D",
}
MARKER_EDGE_COLOR = "0.55"
PROGRAM_YEAR_LIMITS = {}
ANALYTE_CATEGORIES = [
    "All",
    "CFCs",
    "HCFCs",
    "HFCs",
    "Halons",
    "Solvents",
    "Hydrocarbons",
    "Other",
]
HALON_ANALYTES = {"H1211", "H1301", "H2402", "HALON1211", "HALON1301", "HALON2402"}
SOLVENT_ANALYTES = {
    "CCL4",
    "CH3CCL3",
    "CH2CL2",
    "CHCL3",
    "C2CL4",
    "C2HCL3",
    "12DCE",
    "CH3CL",
    "CH3BR",
    "CH3I",
}
HYDROCARBON_ANALYTES = {
    "CH4",
    "C2H2",
    "C2H4",
    "C2H6",
    "C3H6",
    "C3H8",
    "C4H10",
    "C5H12",
    "C6H6",
    "C7H8",
    "BENZENE",
    "TOLUENE",
}
ANALYTE_PARAMETER_GROUPS = {
    29: ("CFC11", (29, 114)),
    114: ("CFC11", (29, 114)),
}


@dataclass
class ProgramSelection:
    key: str
    analyte_key: str
    parameter_num: int


class BasicInstrument(HATS_DB_Functions):
    """Small instrument wrapper for programs without a dedicated UI class."""

    def __init__(self, inst_id: str):
        super().__init__(inst_id)
        self.molecules = self.query_molecules()
        self.analytes = self.query_analytes()
        self.analytes_inv = {int(v): k for k, v in self.analytes.items()}
        self.site = None


class CompareDataLoader(TimeseriesWidget):
    """Hidden TimeseriesWidget used only for its existing DB load methods."""

    def __init__(self, instrument):
        super().__init__(instrument=instrument, parent=None)
        self.force_preferred_channel = True
        self.hide()

    def configure(self, analyte: str, start_year: int, end_year: int, sites: list[str]) -> None:
        self.start_year.setValue(start_year)
        self.end_year.setValue(end_year)
        idx = self.analyte_combo.findText(analyte, Qt.MatchExactly)
        if idx < 0:
            self.analyte_combo.addItem(analyte)
            idx = self.analyte_combo.findText(analyte, Qt.MatchExactly)
        self.analyte_combo.setCurrentIndex(idx)
        self.set_current_analyte(analyte)
        site_set = set(sites)
        for cb in self.site_checks:
            cb.setChecked(cb.text() in site_set)


def _parameter_analyte_keys(analytes: dict[str, int]) -> dict[int, str]:
    """Return the local TimeseriesWidget analyte key to use for each parameter."""
    by_parameter: dict[int, list[str]] = {}
    for name, pnum in analytes.items():
        by_parameter.setdefault(int(pnum), []).append(name)
    return {pnum: sorted(names)[0] for pnum, names in by_parameter.items()}


def _compact_analyte_name(name: object) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def _is_hydrocarbon_formula(compact_name: str) -> bool:
    if not compact_name.startswith("C") or "H" not in compact_name:
        return False
    parts = compact_name[1:].split("H", 1)
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def analyte_category(name: object) -> str:
    compact = _compact_analyte_name(name)
    if compact.startswith("HCFC"):
        return "HCFCs"
    if compact.startswith("CFC"):
        return "CFCs"
    if compact.startswith(("HFC", "HFO")):
        return "HFCs"
    if compact in HALON_ANALYTES:
        return "Halons"
    if compact in SOLVENT_ANALYTES:
        return "Solvents"
    if compact in HYDROCARBON_ANALYTES or _is_hydrocarbon_formula(compact):
        return "Hydrocarbons"
    return "Other"


def parameter_group_key(pnum: int) -> tuple[int, ...]:
    group = ANALYTE_PARAMETER_GROUPS.get(int(pnum))
    return group[1] if group else (int(pnum),)


def parameter_group_name(pnum: int, fallback_name: object) -> str:
    group = ANALYTE_PARAMETER_GROUPS.get(int(pnum))
    return group[0] if group else str(fallback_name)


class LogosCompareWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LOGOS Program Compare")
        self.resize(1400, 900)

        self.loaders = self._build_loaders()
        self.loader_analytes_by_parameter = {
            key: _parameter_analyte_keys(loader.analytes)
            for key, loader in self.loaders.items()
        }
        self.site_info = self._load_site_info()
        self.site_order = self._site_order()
        self.site_lats = dict(zip(self.site_info["code"], self.site_info["lat"]))
        self.analyte_programs = self._load_analyte_programs()
        self.current_category = "All"
        self.current_plot_df = pd.DataFrame()
        self.current_plot_selections: list[ProgramSelection] = []
        self.site_visibility: dict[str, bool] = {}
        self.site_artist_map: dict[str, list[object]] = {}
        self.site_legend = None

        self._build_ui()
        self._setup_shortcuts()
        self._refresh_program_checks(reset_checked=True)

    def _build_loaders(self) -> dict[str, CompareDataLoader]:
        instruments = {
            "m4": M4_Instrument(),
            "fe3": FE3_Instrument(),
            "ie3": IE3_Instrument(),
            "pr1": BasicInstrument("pr1"),
        }
        return {key: CompareDataLoader(inst) for key, inst in instruments.items()}

    def _load_site_info(self) -> pd.DataFrame:
        try:
            return self.loaders["m4"].get_site_info().sort_values("lat", ascending=False)
        except Exception:
            return pd.DataFrame({"code": LOGOS_sites, "lat": np.nan})

    def _site_order(self) -> list[str]:
        ordered = [s for s in self.site_info["code"].tolist() if s in LOGOS_sites]
        for site in LOGOS_sites:
            if site not in ordered:
                ordered.append(site)
        return ordered

    def _load_analyte_programs(self) -> dict[tuple[int, ...], dict[str, object]]:
        inst_nums = sorted({meta["inst_num"] for meta in PROGRAMS.values()})
        rows = self._query_analyte_parameter_rows(inst_nums)
        mapping: dict[tuple[int, ...], dict[str, object]] = {}
        for row in rows:
            pnum = int(row["parameter_num"])
            inst_num = int(row["inst_num"])
            for key, meta in PROGRAMS.items():
                if meta["inst_num"] == inst_num:
                    name = parameter_group_name(pnum, row.get("analyte_name") or str(pnum))
                    group_key = parameter_group_key(pnum)
                    entry = mapping.setdefault(
                        group_key,
                        {
                            "name": name,
                            "category": analyte_category(name),
                            "programs": set(),
                            "parameters": set(group_key),
                        },
                    )
                    entry["programs"].add(key)
                    entry["parameters"].add(pnum)
        return mapping

    def _query_analyte_parameter_rows(self, inst_nums: list[int]) -> list[dict]:
        placeholders = ",".join(["%s"] * len(inst_nums))
        sql = f"""
            SELECT DISTINCT al.param_num AS parameter_num, al.inst_num, p.formula AS analyte_name
            FROM hats.analyte_list al
            JOIN gmd.parameter p ON p.num = al.param_num
            WHERE al.inst_num IN ({placeholders})
            ORDER BY p.formula, al.param_num, al.inst_num;
        """
        return self.loaders["m4"].instrument.doquery(sql, inst_nums)

    def _build_ui(self) -> None:
        root = QSplitter(Qt.Horizontal)
        self.setCentralWidget(root)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)

        date_group = QGroupBox("YEAR RANGE")
        date_layout = QGridLayout(date_group)
        current_year = pd.Timestamp.now().year
        self.start_year = QSpinBox()
        self.start_year.setRange(1990, 2035)
        self.start_year.setValue(current_year - 2)
        self.start_year.valueChanged.connect(self._on_year_range_changed)
        self.end_year = QSpinBox()
        self.end_year.setRange(1990, 2035)
        self.end_year.setValue(current_year)
        self.end_year.valueChanged.connect(self._on_year_range_changed)
        date_layout.addWidget(QLabel("Start"), 0, 0)
        date_layout.addWidget(self.start_year, 0, 1)
        date_layout.addWidget(QLabel("End"), 1, 0)
        date_layout.addWidget(self.end_year, 1, 1)
        controls_layout.addWidget(date_group)

        analyte_group = QGroupBox("ANALYTE")
        analyte_layout = QVBoxLayout(analyte_group)

        self.category_buttons = QButtonGroup(self)
        self.category_buttons.setExclusive(True)
        category_grid = QGridLayout()
        category_grid.setContentsMargins(0, 0, 0, 0)
        category_grid.setHorizontalSpacing(4)
        category_grid.setVerticalSpacing(4)
        for i, category in enumerate(ANALYTE_CATEGORIES):
            btn = QPushButton(category)
            btn.setCheckable(True)
            btn.setChecked(category == self.current_category)
            btn.setToolTip(f"{self._category_count(category)} analytes")
            btn.setProperty("categoryButton", True)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.category_buttons.addButton(btn)
            row, col = divmod(i, 2)
            category_grid.addWidget(btn, row, col)
        self.category_buttons.buttonClicked.connect(self._on_category_changed)
        analyte_layout.addLayout(category_grid)

        self.analyte_combo = QComboBox()
        self.analyte_combo.setEditable(True)
        self.analyte_combo.setInsertPolicy(QComboBox.NoInsert)
        self._populate_analyte_combo()
        self.analyte_combo.currentTextChanged.connect(self._on_analyte_changed)

        analyte_row = QHBoxLayout()
        analyte_row.setContentsMargins(0, 0, 0, 0)
        analyte_row.setSpacing(4)
        analyte_row.addWidget(self.analyte_combo, stretch=1)
        analyte_step_layout = QVBoxLayout()
        analyte_step_layout.setContentsMargins(0, 0, 0, 0)
        analyte_step_layout.setSpacing(2)
        self.prev_analyte_button = QToolButton()
        self.prev_analyte_button.setArrowType(Qt.UpArrow)
        self.prev_analyte_button.setToolTip("Previous analyte")
        self.prev_analyte_button.clicked.connect(self._prev_analyte)
        self.next_analyte_button = QToolButton()
        self.next_analyte_button.setArrowType(Qt.DownArrow)
        self.next_analyte_button.setToolTip("Next analyte")
        self.next_analyte_button.clicked.connect(self._next_analyte)
        analyte_step_layout.addWidget(self.prev_analyte_button)
        analyte_step_layout.addWidget(self.next_analyte_button)
        analyte_row.addLayout(analyte_step_layout)
        analyte_layout.addLayout(analyte_row)
        controls_layout.addWidget(analyte_group)

        program_group = QGroupBox("PROGRAMS")
        program_layout = QGridLayout(program_group)
        program_layout.setContentsMargins(10, 10, 10, 10)
        program_layout.setHorizontalSpacing(12)
        program_layout.setVerticalSpacing(6)
        self.program_checks: dict[str, QCheckBox] = {}
        for i, (key, meta) in enumerate(PROGRAMS.items()):
            cb = QCheckBox(meta["label"])
            if key in PROGRAM_YEAR_LIMITS:
                first, last = PROGRAM_YEAR_LIMITS[key]
                cb.setToolTip(f"Available for selected ranges overlapping {first}-{last}.")
            self.program_checks[key] = cb
            row, col = divmod(i, 2)
            program_layout.addWidget(cb, row, col)
        controls_layout.addWidget(program_group)

        site_group = QGroupBox("SITES")
        site_layout = QVBoxLayout(site_group)
        site_grid = QGridLayout()
        self.site_checks: dict[str, QCheckBox] = {}
        cols = 4
        for i, site in enumerate(self.site_order):
            cb = QCheckBox(site)
            cb.setChecked(site in DEFAULT_SITE_SELECTION)
            self.site_checks[site] = cb
            row, col = divmod(i, cols)
            site_grid.addWidget(cb, row, col)
        site_layout.addLayout(site_grid)

        button_row = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(lambda: self._set_all_sites(True))
        none_btn = QPushButton("None")
        none_btn.clicked.connect(lambda: self._set_all_sites(False))
        button_row.addWidget(all_btn)
        button_row.addWidget(none_btn)
        site_layout.addLayout(button_row)
        controls_layout.addWidget(site_group)

        self.plot_button = QPushButton("Plot Comparison")
        self.plot_button.setObjectName("primaryButton")
        self.plot_button.clicked.connect(self.plot_comparison)
        controls_layout.addWidget(self.plot_button)

        figure_group = QGroupBox("FIGURE CONTROL")
        figure_layout = QGridLayout(figure_group)
        figure_layout.setContentsMargins(10, 10, 10, 10)
        figure_layout.setHorizontalSpacing(8)
        figure_layout.setVerticalSpacing(6)

        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 20)
        self.marker_size_spin.setValue(5)
        self.marker_size_spin.setSuffix(" pt")
        self.marker_size_spin.valueChanged.connect(self._redraw_current_plot)
        figure_layout.addWidget(QLabel("Marker size"), 0, 0)
        figure_layout.addWidget(self.marker_size_spin, 0, 1)

        self.connect_lines_check = QCheckBox("Connect monthly points")
        self.connect_lines_check.setChecked(False)
        self.connect_lines_check.stateChanged.connect(self._redraw_current_plot)
        figure_layout.addWidget(self.connect_lines_check, 1, 0, 1, 2)

        self.show_site_legend_check = QCheckBox("Show site legend")
        self.show_site_legend_check.setChecked(True)
        self.show_site_legend_check.stateChanged.connect(self._redraw_current_plot)
        figure_layout.addWidget(self.show_site_legend_check, 2, 0, 1, 2)
        controls_layout.addWidget(figure_group)
        controls_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        root.addWidget(scroll)

        display = QWidget()
        display_layout = QVBoxLayout(display)
        self.figure = Figure(figsize=(11, 8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("pick_event", self._on_pick_event)
        self.toolbar = NavigationToolbar(self.canvas, self)
        display_layout.addWidget(self.toolbar)
        display_layout.addWidget(self.canvas, stretch=1)
        root.addWidget(display)
        root.setSizes([340, 1060])

    def _setup_shortcuts(self) -> None:
        self.prev_analyte_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Up"), self)
        self.prev_analyte_shortcut.activated.connect(self._prev_analyte)
        self.next_analyte_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Down"), self)
        self.next_analyte_shortcut.activated.connect(self._next_analyte)

    def _prev_analyte(self) -> None:
        idx = self.analyte_combo.currentIndex()
        if idx > 0:
            self.analyte_combo.setCurrentIndex(idx - 1)

    def _next_analyte(self) -> None:
        idx = self.analyte_combo.currentIndex()
        if idx < self.analyte_combo.count() - 1:
            self.analyte_combo.setCurrentIndex(idx + 1)

    def _category_count(self, category: str) -> int:
        return len(self._filtered_analyte_items(category))

    def _filtered_analyte_items(self, category: str | None = None) -> list[tuple[tuple[int, ...], dict[str, object]]]:
        category = category or self.current_category
        items = []
        for parameter_key, info in self.analyte_programs.items():
            if category == "All" or info.get("category") == category:
                items.append((parameter_key, info))
        return sorted(items, key=lambda item: (str(item[1]["name"]).lower(), item[0]))

    def _populate_analyte_combo(self, preferred_key: tuple[int, ...] | None = None) -> None:
        previous_key = preferred_key if preferred_key is not None else self._current_parameter_key()
        items = self._filtered_analyte_items()

        self.analyte_combo.blockSignals(True)
        self.analyte_combo.clear()
        for parameter_key, info in items:
            pnum_label = ",".join(str(pnum) for pnum in parameter_key)
            self.analyte_combo.addItem(f"{info['name']} ({pnum_label})", parameter_key)

        target_idx = -1
        if previous_key is not None:
            target_idx = self.analyte_combo.findData(previous_key)
        if target_idx < 0 and self.analyte_combo.count() > 0:
            target_idx = 0
        if target_idx >= 0:
            self.analyte_combo.setCurrentIndex(target_idx)

        completer = self.analyte_combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setCompletionMode(QCompleter.PopupCompletion)
        self.analyte_combo.blockSignals(False)

    def _on_category_changed(self, button: QPushButton) -> None:
        self.current_category = button.text()
        self._populate_analyte_combo()
        self._refresh_program_checks(reset_checked=True)

    def _set_all_sites(self, checked: bool) -> None:
        for cb in self.site_checks.values():
            cb.setChecked(checked)

    def _on_analyte_changed(self) -> None:
        self._refresh_program_checks(reset_checked=True)

    def _on_year_range_changed(self) -> None:
        self._refresh_program_checks(reset_checked=False)

    def _refresh_program_checks(self, reset_checked: bool = False) -> None:
        parameter_key = self._current_parameter_key()
        available = self._available_programs_for_parameter_key(parameter_key)
        for key, cb in self.program_checks.items():
            has_analyte = key in available
            in_year_range = self._program_available_in_year_range(key)
            is_available = has_analyte and in_year_range
            cb.setEnabled(is_available)
            if reset_checked:
                cb.setChecked(is_available)
            elif not is_available:
                cb.setChecked(False)
            if has_analyte and not in_year_range:
                first, last = PROGRAM_YEAR_LIMITS[key]
                cb.setToolTip(
                    f"No {PROGRAMS[key]['label']} data in selected years. "
                    f"Available for ranges overlapping {first}-{last}."
                )
            elif key in PROGRAM_YEAR_LIMITS:
                first, last = PROGRAM_YEAR_LIMITS[key]
                cb.setToolTip(f"Available for selected ranges overlapping {first}-{last}.")
            elif not has_analyte:
                cb.setToolTip("This program does not measure the selected analyte.")
            else:
                cb.setToolTip("")

    def _program_available_in_year_range(self, program_key: str) -> bool:
        limits = PROGRAM_YEAR_LIMITS.get(program_key)
        if limits is None:
            return True
        first, last = limits
        start = min(self.start_year.value(), self.end_year.value())
        end = max(self.start_year.value(), self.end_year.value())
        return start <= last and end >= first

    def _current_parameter_key(self) -> tuple[int, ...] | None:
        parameter_key = self.analyte_combo.currentData()
        if parameter_key is None:
            return None
        if isinstance(parameter_key, tuple):
            return tuple(int(pnum) for pnum in parameter_key)
        return (int(parameter_key),)

    def _current_analyte_name(self) -> str:
        parameter_key = self._current_parameter_key()
        if parameter_key is None:
            return self.analyte_combo.currentText()
        info = self.analyte_programs.get(parameter_key, {})
        return str(info.get("name") or ",".join(str(pnum) for pnum in parameter_key))

    def _available_programs_for_parameter_key(self, parameter_key: tuple[int, ...] | None) -> dict[str, tuple[str, int]]:
        if parameter_key is None:
            return {}
        available: dict[str, tuple[str, int]] = {}
        for key, meta in PROGRAMS.items():
            loader_params = self.loader_analytes_by_parameter[meta["loader"]]
            for pnum in parameter_key:
                analyte_key = loader_params.get(pnum)
                if analyte_key is not None:
                    available[key] = (analyte_key, pnum)
                    break
        return available

    def _selected_programs(self) -> list[ProgramSelection]:
        available = self._available_programs_for_parameter_key(self._current_parameter_key())
        selected = []
        for key, cb in self.program_checks.items():
            if cb.isEnabled() and cb.isChecked() and key in available:
                analyte_key, pnum = available[key]
                selected.append(
                    ProgramSelection(
                        key=key,
                        analyte_key=analyte_key,
                        parameter_num=int(pnum),
                    )
                )
        return selected

    def _selected_sites(self) -> list[str]:
        return [site for site, cb in self.site_checks.items() if cb.isChecked()]

    def plot_comparison(self) -> None:
        selections = self._selected_programs()
        if not selections:
            QMessageBox.warning(self, "Plot Comparison", "Select at least one program.")
            return

        sites = self._selected_sites()
        if not sites:
            QMessageBox.warning(self, "Plot Comparison", "Select at least one site.")
            return

        self.site_visibility = {site: True for site in sites}
        self.plot_button.setText("Loading data...")
        self.plot_button.setEnabled(False)
        QApplication.processEvents()
        try:
            df = self._load_comparison_data(selections, sites)
            if df.empty:
                QMessageBox.warning(self, "Plot Comparison", "No data found for this selection.")
                return
            active_selections = self._active_selections_with_data(df, selections)
            self.current_plot_df = df
            self.current_plot_selections = active_selections
            self._draw_comparison(df, active_selections)
        finally:
            self.plot_button.setText("Plot Comparison")
            self.plot_button.setEnabled(True)

    def _redraw_current_plot(self) -> None:
        if self.current_plot_df.empty or not self.current_plot_selections:
            return
        self._draw_comparison(self.current_plot_df, self.current_plot_selections)

    def _active_selections_with_data(
        self,
        df: pd.DataFrame,
        selections: list[ProgramSelection],
    ) -> list[ProgramSelection]:
        present = set(df["program"].dropna().unique())
        return [selection for selection in selections if selection.key in present]

    def _load_comparison_data(
        self,
        selections: list[ProgramSelection],
        sites: list[str],
    ) -> pd.DataFrame:
        frames = []
        for selection in selections:
            program_df = self._load_program_monthly(selection, sites)
            program_df = self._clean_monthly_frame(program_df)
            if program_df.empty:
                continue
            program_df["program"] = selection.key
            frames.append(program_df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["site", "program", "month"])

    def _clean_monthly_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["site", "month", "value", "std", "n"])
        out = df.copy()
        for col in ("value", "std", "n"):
            if col in out:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out["month"] = pd.to_datetime(out["month"], errors="coerce")
        out = out.dropna(subset=["site", "month", "value"])
        return out[["site", "month", "value", "std", "n"]]

    def _load_program_monthly(self, selection: ProgramSelection, sites: list[str]) -> pd.DataFrame:
        if not self._program_available_in_year_range(selection.key):
            return pd.DataFrame(columns=["site", "month", "value", "std", "n"])

        meta = PROGRAMS[selection.key]
        loader = self.loaders[meta["loader"]]
        loader.configure(
            selection.analyte_key,
            self.start_year.value(),
            self.end_year.value(),
            sites,
        )

        if selection.key == "mstar":
            df = self._query_mstar_monthly_mean_data(selection, sites)
        elif selection.key == "fecd":
            df = self._query_fecd_monthly_mean_data(selection, sites)
        elif selection.key == "ie3":
            df = loader.query_insitu_data(selection.analyte_key, force=True)
            return self._monthly_from_insitu(df, sites)
        elif selection.key == "pr1":
            df = loader.query_pr1_monthly_mean_data(selection.analyte_key)
        else:
            df = loader.query_monthly_mean_data(selection.analyte_key)

        if df.empty:
            return pd.DataFrame(columns=["site", "month", "value", "std", "n"])
        out = df.rename(
            columns={
                "month_start": "month",
                "monthly_avg": "value",
                "monthly_std": "std",
            }
        ).copy()
        out["site"] = out["site"].astype(str).str.upper()
        out["month"] = pd.to_datetime(out["month"])
        if "n" not in out:
            out["n"] = np.nan
        return out[["site", "month", "value", "std", "n"]]

    def _query_mstar_monthly_mean_data(
        self,
        selection: ProgramSelection,
        sites: list[str],
    ) -> pd.DataFrame:
        return self._query_combined_pair_monthly_mean_data(
            selection=selection,
            sites=sites,
            loader_key="m4",
            regular_condition="v.inst_id IN ('M1', 'M3', 'M4')",
            pfp_condition="v.inst_id = 'M4'",
        )

    def _query_fecd_monthly_mean_data(
        self,
        selection: ProgramSelection,
        sites: list[str],
    ) -> pd.DataFrame:
        loader = self.loaders["fe3"]
        preferred_filter = self._sql_condition_from_and_filter(
            loader._preferred_channel_filter_sql("v.channel", "v.parameter_num", "v.sample_datetime")
        )
        fe3_condition = "v.inst_num = 193"
        if preferred_filter:
            fe3_condition = f"({fe3_condition} AND {preferred_filter})"
        return self._query_combined_pair_monthly_mean_data(
            selection=selection,
            sites=sites,
            loader_key="fe3",
            regular_condition=f"(v.inst_id = 'OTTO' OR {fe3_condition})",
            pfp_condition=fe3_condition,
        )

    def _query_combined_pair_monthly_mean_data(
        self,
        selection: ProgramSelection,
        sites: list[str],
        loader_key: str,
        regular_condition: str,
        pfp_condition: str,
    ) -> pd.DataFrame:
        pnum = int(selection.parameter_num)
        start = self.start_year.value()
        end = self.end_year.value()
        frames = []
        loader = self.loaders[loader_key]
        regular_sites = [s for s in sites if s not in PFP_SITES]
        pfp_pseudo_sites = [s for s in sites if s in PFP_SITES]

        pfp_base_sites = set(PFP_SITES.values())
        has_pfp_base = any(s in pfp_base_sites for s in regular_sites)
        pfp_exclusion = "AND v.sample_type IN ('S', 'G')" if has_pfp_base else ""

        if regular_sites:
            sql = f"""
            SELECT UPPER(v.site) AS site,
                DATE_FORMAT(v.sample_datetime, '%%Y-%%m-01') AS month_start,
                AVG(v.pair_avg) AS monthly_avg,
                STDDEV(v.pair_avg) AS monthly_std,
                COUNT(*) AS n
            FROM hats.ng_pair_avg_view v
            WHERE {regular_condition}
              AND v.parameter_num = %s
              AND UPPER(v.site) IN ({",".join(["%s"] * len(regular_sites))})
              {pfp_exclusion}
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
            GROUP BY site, month_start ORDER BY site, month_start;
            """
            params = [pnum] + regular_sites + [start, end]
            frames.append(pd.DataFrame(loader.instrument.doquery(sql, params)))

        for pfp_site in pfp_pseudo_sites:
            base_site = PFP_SITES[pfp_site]
            sql = f"""
            SELECT %s AS site,
                DATE_FORMAT(v.sample_datetime, '%%Y-%%m-01') AS month_start,
                AVG(v.pair_avg) AS monthly_avg,
                STDDEV(v.pair_avg) AS monthly_std,
                COUNT(*) AS n
            FROM hats.ng_pair_avg_view v
            WHERE {pfp_condition}
              AND v.parameter_num = %s
              AND v.sample_type = 'PFP'
              AND v.site = %s
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
            GROUP BY month_start ORDER BY month_start;
            """
            params = [pfp_site, pnum, base_site, start, end]
            frames.append(pd.DataFrame(loader.instrument.doquery(sql, params)))

        non_empty = [frame for frame in frames if not frame.empty]
        df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        if not df.empty:
            df["month_start"] = pd.to_datetime(df["month_start"])
        return df

    def _sql_condition_from_and_filter(self, sql_filter: str) -> str:
        condition = sql_filter.strip()
        if condition.upper().startswith("AND "):
            condition = condition[4:].strip()
        return condition

    def _monthly_from_insitu(self, df: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["site", "month", "value", "std", "n"])
        insitu = df[df["site"].isin(sites)].copy()
        if insitu.empty:
            return pd.DataFrame(columns=["site", "month", "value", "std", "n"])
        insitu["month"] = insitu["analysis_time"].dt.to_period("M").dt.to_timestamp()
        monthly = (
            insitu.groupby(["site", "month"])["mole_fraction"]
            .agg(value="mean", std="std", n="count")
            .reset_index()
        )
        monthly["site"] = monthly["site"].astype(str).str.upper()
        return monthly[["site", "month", "value", "std", "n"]]

    def _draw_comparison(self, df: pd.DataFrame, selections: list[ProgramSelection]) -> None:
        self.figure.clear()
        self.site_artist_map = {}
        self.site_legend = None
        ax_diff, ax_data = self.figure.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 2]}
        )

        selected_sites = [site for site in self.site_order if site in set(df["site"])]
        for site in selected_sites:
            self.site_visibility.setdefault(site, True)
        colors = self._site_colors()
        marker_size = self.marker_size_spin.value()
        line_style = "-" if self.connect_lines_check.isChecked() else ""

        for (site, program), group in df.groupby(["site", "program"], sort=False):
            yerr = group["std"].to_numpy(dtype=float)
            yerr = None if not np.isfinite(yerr).any() else yerr
            container = ax_data.errorbar(
                group["month"],
                group["value"],
                yerr=yerr,
                fmt=PROGRAM_MARKERS.get(program, "o"),
                linestyle=line_style,
                markersize=marker_size,
                capsize=2,
                alpha=0.85,
                color=colors.get(site, "gray"),
                markeredgecolor=MARKER_EDGE_COLOR,
                markeredgewidth=0.7,
                label=f"{site} {PROGRAMS[program]['label']}",
            )
            self._register_site_artists(site, container)

        ref_program = selections[0].key if selections else None
        if ref_program is not None and len(selections) > 1:
            diff_df = self._difference_data(df, ref_program)
            for (site, program), group in diff_df.groupby(["site", "program"], sort=False):
                yerr = group["diff_std"].to_numpy(dtype=float)
                yerr = None if not np.isfinite(yerr).any() else yerr
                ax_diff.axhline(0, color="0.6", linewidth=0.8, zorder=0)
                container = ax_diff.errorbar(
                    group["month"],
                    group["diff"],
                    yerr=yerr,
                    marker=PROGRAM_MARKERS.get(program, "o"),
                    linestyle=line_style,
                    markersize=marker_size,
                    capsize=2,
                    alpha=0.9,
                    color=colors.get(site, "gray"),
                    markeredgecolor=MARKER_EDGE_COLOR,
                    markeredgewidth=0.7,
                )
                self._register_site_artists(site, container)
        else:
            ax_diff.text(
                0.5,
                0.5,
                "Select at least two programs with data for differences",
                ha="center",
                va="center",
                transform=ax_diff.transAxes,
                color="0.35",
            )

        analyte = self._current_analyte_name()
        ax_diff.set_title(f"{analyte}: monthly program differences")
        ax_diff.set_ylabel("Difference")
        ax_diff.grid(True, alpha=0.3)

        ax_data.set_title(f"{analyte}: monthly program means")
        ax_data.set_xlabel("Sample month")
        ax_data.set_ylabel("Mole fraction")
        ax_data.grid(True, alpha=0.3)

        if ref_program is not None and len(selections) > 1:
            self._add_difference_legend(ax_diff, diff_df, ref_program, selections, line_style)
        self._add_legends(ax_data, selected_sites, selections, colors, line_style)
        self._apply_site_visibility()
        self.canvas.draw_idle()

    def _register_site_artists(self, site: str, container: object) -> None:
        self.site_artist_map.setdefault(site, []).extend(self._flatten_artists(container))

    def _flatten_artists(self, item: object) -> list[object]:
        if item is None:
            return []
        if isinstance(item, (list, tuple)):
            artists = []
            for child in item:
                artists.extend(self._flatten_artists(child))
            return artists
        if hasattr(item, "lines"):
            return self._flatten_artists(getattr(item, "lines"))
        if hasattr(item, "set_visible"):
            return [item]
        return []

    def _apply_site_visibility(self) -> None:
        for site, artists in self.site_artist_map.items():
            visible = self.site_visibility.get(site, True)
            for artist in artists:
                if hasattr(artist, "set_visible"):
                    artist.set_visible(visible)

        if self.site_legend is not None:
            for text in self.site_legend.get_texts():
                site = text.get_text()
                alpha = 1.0 if self.site_visibility.get(site, True) else 0.2
                text.set_alpha(alpha)
            for handle in self._legend_handles(self.site_legend):
                site = handle.get_label()
                alpha = 1.0 if self.site_visibility.get(site, True) else 0.2
                handle.set_alpha(alpha)

    def _legend_handles(self, legend) -> list[object]:
        return list(getattr(legend, "legend_handles", getattr(legend, "legendHandles", [])))

    def _on_pick_event(self, event) -> None:
        if self.site_legend is None:
            return

        site = None
        artist = event.artist
        for text in self.site_legend.get_texts():
            if artist == text:
                site = text.get_text()
                break
        if site is None:
            for handle in self._legend_handles(self.site_legend):
                if artist == handle:
                    site = handle.get_label()
                    break

        if site not in self.site_artist_map:
            return

        self.site_visibility[site] = not self.site_visibility.get(site, True)
        self._apply_site_visibility()
        self.canvas.draw_idle()

    def _site_colors(self) -> dict[str, object]:
        """Use the same full-network site color assignment as logos_timeseries."""
        return build_site_colors(self.site_order)

    def _difference_data(self, df: pd.DataFrame, ref_program: str) -> pd.DataFrame:
        ref = df[df["program"] == ref_program][["site", "month", "value", "std"]].rename(
            columns={"value": "ref_value", "std": "ref_std"}
        )
        other = df[df["program"] != ref_program].copy()
        merged = other.merge(ref, on=["site", "month"], how="inner")
        merged["diff"] = merged["ref_value"] - merged["value"]
        merged["diff_std"] = np.sqrt(
            np.square(pd.to_numeric(merged["ref_std"], errors="coerce"))
            + np.square(pd.to_numeric(merged["std"], errors="coerce"))
        )
        return merged

    def _add_difference_legend(
        self,
        ax,
        diff_df: pd.DataFrame,
        ref_program: str,
        selections: list[ProgramSelection],
        line_style: str,
    ) -> None:
        handles = []
        for selection in selections:
            if selection.key == ref_program:
                continue
            stats_df = diff_df.loc[diff_df["program"] == selection.key, ["diff", "diff_std"]].copy()
            stats_df["diff"] = pd.to_numeric(stats_df["diff"], errors="coerce")
            stats_df["diff_std"] = pd.to_numeric(stats_df["diff_std"], errors="coerce")
            stats_df = stats_df.dropna(subset=["diff"])
            if stats_df.empty:
                stats = "mean nan, std nan"
            else:
                mean = self._weighted_mean_from_std(stats_df["diff"], stats_df["diff_std"])
                stats = f"weighted mean {mean:.3f}, std {stats_df['diff'].std():.3f}"
            label = (
                f"{PROGRAMS[ref_program]['label']} - {PROGRAMS[selection.key]['label']} "
                f"({stats})"
            )
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="black",
                    marker=PROGRAM_MARKERS.get(selection.key, "o"),
                    linestyle=line_style,
                    label=label,
                )
            )
        if handles:
            ax.legend(handles=handles, title="Difference", loc="best")

    def _weighted_mean_from_std(self, values: pd.Series, stds: pd.Series) -> float:
        values = pd.to_numeric(values, errors="coerce")
        stds = pd.to_numeric(stds, errors="coerce")
        valid = values.notna() & stds.notna() & np.isfinite(stds) & (stds > 0)
        if not valid.any():
            return float(values.mean())
        weights = 1.0 / np.square(stds[valid])
        return float(np.average(values[valid], weights=weights))

    def _add_legends(
        self,
        ax,
        sites: list[str],
        selections: list[ProgramSelection],
        colors: dict[str, tuple],
        line_style: str,
    ) -> None:
        site_handles = [
            mlines.Line2D([], [], color=colors.get(site, "gray"), marker="o", linestyle=line_style, label=site)
            for site in sites
        ]
        program_handles = [
            mlines.Line2D(
                [],
                [],
                color="black",
                marker=PROGRAM_MARKERS.get(selection.key, "o"),
                linestyle=line_style,
                label=PROGRAMS[selection.key]["label"],
            )
            for selection in selections
        ]

        site_legend = ax.legend(
            handles=site_handles,
            title="Sites",
            loc="upper left",
        )
        site_legend.set_visible(self.show_site_legend_check.isChecked())
        ax.add_artist(site_legend)
        self.site_legend = site_legend
        for handle in self._legend_handles(site_legend):
            handle.set_picker(5)
        for text in site_legend.get_texts():
            text.set_picker(5)
        ax.legend(
            handles=program_handles,
            title="Programs",
            loc="upper right",
        )


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            font-size: 12px;
        }
        QScrollArea {
            border: none;
            background: #f6f7f9;
        }
        QGroupBox {
            border: 1px solid #d1d5db;
            border-radius: 8px;
            margin-top: 14px;
            padding: 10px 12px 12px 12px;
            font-weight: 600;
            color: #2255aa;
            background: #ffffff;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 4px;
            background: #ffffff;
        }
        QLabel {
            color: #374151;
        }
        QComboBox, QSpinBox {
            min-height: 24px;
            padding: 2px 6px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            background: #ffffff;
            color: #111827;
        }
        QComboBox:focus, QSpinBox:focus {
            border-color: #2255aa;
        }
        QCheckBox {
            spacing: 5px;
            color: #111827;
        }
        QPushButton {
            min-height: 24px;
            padding: 4px 8px;
            border: 1px solid #cbd5e1;
            border-radius: 6px;
            background: #f8fafc;
            color: #111827;
        }
        QPushButton:hover {
            background: #eef4ff;
            border-color: #93b4e8;
        }
        QPushButton:pressed {
            background: #dbeafe;
        }
        QPushButton:disabled {
            color: #9ca3af;
            background: #f3f4f6;
            border-color: #e5e7eb;
        }
        QPushButton[categoryButton="true"] {
            min-height: 22px;
            padding: 2px 6px;
            font-weight: 500;
        }
        QPushButton[categoryButton="true"]:checked {
            background: #2255aa;
            border-color: #2255aa;
            color: #ffffff;
            font-weight: 700;
        }
        QPushButton#primaryButton {
            min-height: 30px;
            background: #2255aa;
            border-color: #2255aa;
            color: #ffffff;
            font-weight: 700;
        }
        QPushButton#primaryButton:hover {
            background: #1d4d99;
            border-color: #1d4d99;
        }
        QToolTip {
            background-color: #fff59d;
            color: #111827;
            border: 1px solid #d6c95a;
        }
    """)
    win = LogosCompareWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
