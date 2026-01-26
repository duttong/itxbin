#! /usr/bin/env python

import json
import os
import math

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip, QApplication, QInputDialog
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
from collections import defaultdict


LOGOS_sites = ['SUM', 'PSA', 'SPO', 'SMO', 'AMY', 'MKO', 'ALT', 'CGO', 'NWR',
            'LEF', 'BRW', 'RPB', 'KUM', 'MLO', 'WIS', 'THD', 'MHD', 'HFM',
            'BLD', 'MKO']

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".logos-tanks.conf")
MAX_SAVED_SETS = 5


class TanksPlotter:
    
    
    def __init__(self, db, inst_num):
        self.db = db
        self.inst_num = inst_num

    def _active_tanks_dataframe(self, start_year, end_year) -> pd.DataFrame:
        """Return raw DataFrame of active tanks across all analytes for a window."""
        start_ts = f"{start_year}-01-01"
        end_ts = f"{end_year + 1}-01-01"  # half-open interval on year boundary

        fills_sql = f"""
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
            FROM hats.ng_tank_use_history h
            JOIN reftank.fill f
              ON f.idx = h.fill_idx
            LEFT JOIN hats.ng_tank_uses u
              ON u.num = h.ng_tank_uses_num
            LEFT JOIN reftank.grav_view g
              ON g.fill_num = h.fill_idx
            WHERE h.fill_idx IN ({fill_list})
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
        self.saved_sets: dict[str, list[dict | None]] = {}
        self.active_set_idx: int | None = None
        self.set_buttons: list[QPushButton] = []
        self._loading_set = False
        self._tank_metadata: dict[str, dict] = {}
        self._tank_cache: dict[str, list[dict]] = {}
        self._tank_cache_range: tuple[int, int] | None = None
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
        for idx, name in enumerate(self._analyte_names):
            cb = QCheckBox(name)
            if idx == 0:
                cb.setChecked(True)
            cb.toggled.connect(lambda checked, cb=cb: self._on_analyte_toggled(cb, checked))
            self.analyte_checks.append(cb)
        self._reflow_analyte_grid(cols_analyte)
        analyte_container.setLayout(analyte_checks_layout)
        analyte_layout.addWidget(analyte_container)
        self.tanks_status = QLabel("Select a year range and gas to load tanks.")
        self.tanks_status.setWordWrap(True)
        category_bar = QHBoxLayout()
        self.show_grav_cb = QCheckBox("Gravimetric")
        self.show_grav_cb.setChecked(True)
        self.show_grav_cb.setStyleSheet("color: darkred;")
        self.show_grav_cb.toggled.connect(self._on_category_toggle)
        self.show_other_grav_cb = QCheckBox("Other Gravs")
        self.show_other_grav_cb.setChecked(False)
        self.show_other_grav_cb.setStyleSheet("color: darkblue;")
        self.show_other_grav_cb.toggled.connect(self._on_category_toggle)
        self.show_archive_cb = QCheckBox("Archive")
        self.show_archive_cb.setChecked(True)
        self.show_archive_cb.setStyleSheet("color: darkgreen;")
        self.show_archive_cb.toggled.connect(self._on_category_toggle)
        category_bar.addWidget(self.show_grav_cb)
        category_bar.addWidget(self.show_other_grav_cb)
        category_bar.addWidget(self.show_archive_cb)
        category_bar.addStretch()
        self.tank_grid = QGridLayout()
        self.tank_grid.setContentsMargins(0, 0, 0, 0)
        self.tank_grid.setHorizontalSpacing(8)
        self.tank_grid.setVerticalSpacing(4)
        analyte_layout.addWidget(self.tanks_status)
        analyte_layout.addLayout(category_bar)
        analyte_layout.addLayout(self.tank_grid)
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
        self.refresh_tanks(force_reload=True)
        self._reload_dirty = False
        self.reload_btn.setStyleSheet("")

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
            name = cb.text()
            pnum = (self.instrument.analytes or {}).get(name)
            channel = self._analyte_channel(name)
            if pnum is not None:
                selected.append((name, pnum, channel))
        return selected

    def _reflow_analyte_grid(self, cols: int = 5):
        """Re-layout analyte checkboxes based on sort toggle."""
        layout = getattr(self, "_analyte_checks_layout", None)
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                layout.removeWidget(w)
        if self.alpha_sort_cb.isChecked():
            ordered = sorted(self.analyte_checks, key=lambda cb: cb.text().lower())
        else:
            name_to_cb = {cb.text(): cb for cb in self.analyte_checks}
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

        self.tanks_status.setText(f"{len(tanks)} tanks found.")
        cols = 5
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
                elif use_lower.startswith("grav"):
                    cb.setStyleSheet("color: darkred;")
                elif use_lower.startswith("archive"):
                    cb.setStyleSheet("color: darkgreen;")
            self.tank_checks.append(cb)
            row, col = divmod(idx, cols)
            self.tank_grid.addWidget(cb, row, col)

    def selected_tanks(self) -> list[str]:
        """Return checked tank identifiers keyed by fill (fill_key)."""
        selected = []
        for cb in self.tank_checks:
            if not cb.isChecked():
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
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
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
        payload = {"sets_by_analyte": self.saved_sets}
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
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
            name = cb.text()
            pnum = (self.instrument.analytes or {}).get(name) if self.instrument else None
            channel = self._analyte_channel(name)
            print(pnum, channel)
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
        show_archive = self.show_archive_cb.isChecked() if hasattr(self, "show_archive_cb") else True
        for tank in tanks:
            use_lower = str(tank.get("use_short") or "").lower()
            is_other = use_lower.startswith("other grav")
            is_grav = use_lower.startswith("grav") and not is_other
            is_archive = use_lower.startswith("archive")
            if is_other and not show_other:
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

    def _fetch_calibration_df(
        self,
        serial: str,
        parameter_num: int,
        inst_id: str,
    ) -> pd.DataFrame:
        """Query calibration mole fractions for a tank/parameter."""
        serial_safe = str(serial).replace("'", "''")
        inst_safe = str(inst_id).replace("'", "''")
        sql = f"""
            SELECT c.date, c.time, c.mixratio, c.stddev, c.num, c.run_number
            FROM hats.calibrations c
            WHERE c.serial_number = '{serial_safe}'
            AND c.inst = '{inst_safe}'
              AND c.parameter_num = {int(parameter_num)}
              AND c.mixratio is not NULL
            ORDER BY c.date, c.time;
        """
        try:
            df = pd.DataFrame(self.instrument.db.doquery(sql))
            return df
        except Exception as exc:
            self._toast(f"DB error for tank {serial}: {exc}")
            return pd.DataFrame()

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

        btn = getattr(self, "plot_tanks_btn", None)
        if btn:
            btn.setText("Loading...")
            btn.setStyleSheet("background-color: gold;")
            btn.setEnabled(False)
        # Force a repaint so the user sees the busy state before the DB call.
        try:
            QApplication.processEvents()
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=(9, 5))
        any_data = False
        pick_map: dict[mlines.Line2D, dict] = {}

        try:
            for fill_key in tank_keys:
                meta = self._tank_metadata.get(str(fill_key), {})
                serial = (
                    meta.get("serial_number")
                    or meta.get("tank_serial_num")
                    or str(fill_key).split("::")[0]
                )
                df = self._fetch_calibration_df(serial, parameter_num, inst_id)
                fill_date = meta.get("fill_date") or meta.get("date")
                fill_code = meta.get("code") or meta.get("fill_code")
                next_fill_date = meta.get("next_fill_date")
                if df is None or df.empty:
                    continue
                df = df.copy()
                date_part = pd.to_datetime(df["date"], errors="coerce")
                time_part = pd.to_timedelta(df["time"].astype(str).str.strip(), errors="coerce")
                if time_part.isna().all():
                    time_part = pd.to_timedelta(0)
                df["datetime"] = date_part + time_part
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

                err_container = ax.errorbar(
                    df["datetime"],
                    df["mixratio"],
                    yerr=df["stddev"] if "stddev" in df.columns else None,
                    fmt="o-",
                    markersize=4,
                    linewidth=1,
                    capsize=3,
                    label=f"{serial} ({fill_code})" if fill_code else str(serial),
                )
                line = err_container.lines[0] if err_container.lines else None
                if line is not None:
                    line.set_picker(5)
                    pick_map[line] = {
                        "df": df,
                        "serial": serial,
                        "fill_code": fill_code,
                    }
                any_data = True

            if not any_data:
                plt.close(fig)
                self._toast("No calibration data found for the selected tanks/parameter.")
                return

            species = None
            try:
                pnum_int = int(parameter_num)
                species = getattr(self.instrument, "analytes_inv", {}).get(pnum_int)
            except Exception:
                species = None
            label_species = species or parameter_name or "Species"
            ax.set_title(f"Calibrations for {parameter_name} ({inst_id.upper()})")
            ax.set_xlabel("Datetime")
            ax.set_ylabel(f"Mole Fraction ({label_species})")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()

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
                    return f"{value:.{digits}g}"
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
                if event.button != 1:
                    return
                best_line = None
                best_idx = None
                best_dist = None
                for line in pick_map:
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
                if best_line is None or best_idx is None:
                    QToolTip.hideText()
                    return
                data = pick_map.get(best_line)
                if not data:
                    QToolTip.hideText()
                    return
                df_line = data["df"]
                if best_idx < 0 or best_idx >= len(df_line):
                    QToolTip.hideText()
                    return
                row = df_line.iloc[best_idx]
                serial_val = data.get("serial")
                run_dt = _format_run_dt(row)
                mixratio = _format_value(row.get("mixratio"))
                stddev = _format_value(row.get("stddev"))
                num_samples = _format_value(row.get("num"))
                text = (
                    f"<b>Serial number:</b> {serial_val}<br>"
                    f"<b>Run date/time:</b> {run_dt}<br>"
                    f"<b>Mixing ratio:</b> {mixratio}<br>"
                    f"<b>Standard deviation:</b> {stddev}<br>"
                    f"<b>Number of samples:</b> {num_samples}"
                )
                QToolTip.showText(QCursor.pos(), text)

            def _on_release(event):
                if event.button == 1:
                    QToolTip.hideText()

            app = QApplication.instance()
            if app is not None:
                app.setStyleSheet("QToolTip { background-color: #fff59d; }")
            fig.canvas.mpl_connect("button_press_event", _on_press)
            fig.canvas.mpl_connect("button_release_event", _on_release)
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
        from logos_instruments import M4_Instrument, FE3_Instrument, BLD1_Instrument
    except Exception as exc:  # pragma: no cover - import-time failure path
        print(f"Could not import instrument classes: {exc}", file=sys.stderr)
        M4_Instrument = FE3_Instrument = BLD1_Instrument = None  # type: ignore

    def build_instrument(inst_key: str):
        """Factory to build an instrument by key, with fallback."""
        inst_key = inst_key.lower()
        class_map = {
            "m4": M4_Instrument,
            "fe3": FE3_Instrument,
            "bld1": BLD1_Instrument,
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

    parser = argparse.ArgumentParser(description="TanksWidget test harness")
    parser.add_argument(
        "-i",
        "--instrument",
        choices=["m4", "fe3", "bld1"],
        default="m4",
        help="Instrument to load (default: m4)",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    instrument = build_instrument(args.instrument)
    if instrument is None:
        print("Could not initialize instrument; exiting.", file=sys.stderr)
        sys.exit(1)
    widget = TanksWidget(instrument=instrument)
    widget.setWindowTitle(f"TanksWidget Test Harness ({args.instrument.upper()})")
    widget.resize(320, 420)
    widget.show()
    sys.exit(app.exec_())
    
