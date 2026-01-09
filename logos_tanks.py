#! /usr/bin/env python

import json
import os
import math

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip, QApplication
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

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".logos-tanks.conf")


class TanksPlotter:
    
    
    def __init__(self, db, inst_num):
        self.db = db
        self.inst_num = inst_num
        
        
    def return_active_tanks(self, start_year, end_year, parameter_num=None, channel=None):
        """
        Return a list of tank records (serial + latest fill metadata) active in the window.
        """
        start_ts = f"{start_year}-01-01"
        end_ts = f"{end_year + 1}-01-01"  # half-open interval on year boundary

        # Query 1: find the latest fill_idx per tank observed in the window for this instrument.
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
        if df.empty:
            return []

        # Only keep Grav rows matching the selected parameter; keep Archive/other uses regardless.
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
        self.saved_sets: dict[str, list[dict | None]] = {}
        self.active_set_idx: int | None = None
        self.set_buttons: list[QPushButton] = []
        self._loading_set = False
        self._tank_metadata: dict[str, dict] = {}

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

        controls.addStretch()

        plot_bar = QHBoxLayout()
        plot_bar.addStretch()
        self.plot_tanks_btn = QPushButton("Plot Tanks")
        self.plot_tanks_btn.clicked.connect(self._on_plot_tanks)
        plot_bar.addWidget(self.plot_tanks_btn)
        controls.addLayout(plot_bar)
        self.setLayout(controls)

        # Wire date change after widgets exist
        self.start_year.valueChanged.connect(self._mark_reload_needed)
        self.end_year.valueChanged.connect(self._mark_reload_needed)

        # Populate tanks initially if any analyte starts checked
        self._ready = True
        self._load_saved_sets()
        self._refresh_set_buttons()
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
        self._tank_metadata = {}
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
                    if serial:
                        self._tank_metadata[str(serial)] = entry
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
            f"<span style='color: darkred;'>Gravimetric</span> "
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
            cb.toggled.connect(self._on_tank_toggled)
            cb.setContextMenuPolicy(Qt.CustomContextMenu)
            cb.customContextMenuRequested.connect(
                lambda pos, cb=cb, serial=label: self._on_tank_context(cb, pos, serial)
            )
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
                    sets_by_analyte[key] = [
                        entry if isinstance(entry, dict) else None
                        for entry in (lst if isinstance(lst, list) else [])
                    ][:3] + [None] * max(0, 3 - len(lst))
            elif isinstance(data, dict) and isinstance(data.get("sets"), list):
                # Legacy format: assign sets to their inferred keys.
                for entry in data["sets"]:
                    if not isinstance(entry, dict):
                        continue
                    key = self._key_for_saved_entry(entry)
                    if not key:
                        continue
                    lst = sets_by_analyte.setdefault(key, [None, None, None])
                    for idx in range(3):
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
            btn = QPushButton(f"Tanks {idx + 1}")
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
        entry = {
            "instrument": getattr(self.instrument, "inst_num", None),
            "parameter_name": name,
            "parameter_num": pnum,
            "channel": channel,
            "tanks": tanks,
        }
        sets = self._sets_for_key(key, ensure=True)
        if self.active_set_idx is not None:
            target_idx = self.active_set_idx
        else:
            empty_idx = next((i for i, val in enumerate(sets) if val is None), None)
            target_idx = empty_idx if empty_idx is not None else 0
        sets[target_idx] = entry
        self.saved_sets[key] = sets
        self.active_set_idx = target_idx
        self._persist_sets()
        self._refresh_set_buttons()
        self._toast(f"Saved tank set to Tanks {target_idx + 1}.")

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
            channel = None
            if "(" in name and ")" in name:
                _, ch = name.split("(", 1)
                channel = ch.strip(") ").strip()
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
            cb.setChecked(cb.text() in desired)
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
        self._toast(f"Deleted Tanks {deleted_idx + 1}.")

    def _on_tank_toggled(self, _checked: bool):
        """Tank clicks clear the active-set highlight."""
        if self._loading_set:
            return
        self._clear_active_set_selection()

    def _clear_active_set_selection(self):
        """Reset active set highlight and delete enablement."""
        self.active_set_idx = None
        self._update_set_button_styles()

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
        tooltip = self._build_tank_tooltip(meta, serial)
        if tooltip:
            global_pos = cb.mapToGlobal(pos)
            QToolTip.showText(global_pos, tooltip, cb)

    def _build_tank_tooltip(self, meta: dict, serial_fallback: str) -> str:
        """Build an HTML tooltip with tank metadata."""
        parts = []
        serial_val = meta.get("serial_number") or serial_fallback
        parts.append(f"<b>Serial Number:</b> {serial_val}")
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
            self.saved_sets[key] = [None, None, None]
        return self.saved_sets.get(key, [None, None, None])

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

    def _fetch_calibration_df(self, serial: str, parameter_num: int, inst_id: str) -> pd.DataFrame:
        """Query calibration mole fractions for a tank/parameter."""
        serial_safe = str(serial).replace("'", "''")
        inst_safe = str(inst_id).replace("'", "''")
        sql = f"""
            SELECT c.date, c.time, c.mixratio, c.stddev, c.num
            FROM hats.calibrations c
            WHERE c.serial_number = '{serial_safe}'
            AND c.inst = '{inst_safe}'
              AND c.parameter_num = {int(parameter_num)}
            ORDER BY c.date, c.time;
        """
        try:
            return pd.DataFrame(self.instrument.db.doquery(sql))
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
        parameter_name, parameter_num, _channel = selections[0]
        if parameter_num is None:
            self._toast("Invalid parameter number for selection.")
            return
        tanks = self.selected_tanks()
        if not tanks:
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

        try:
            for serial in tanks:
                df = self._fetch_calibration_df(serial, parameter_num, inst_id)
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
                df = df.dropna(subset=["datetime", "mixratio"]).sort_values("datetime")
                if df.empty:
                    continue

                ax.errorbar(
                    df["datetime"],
                    df["mixratio"],
                    yerr=df["stddev"] if "stddev" in df.columns else None,
                    fmt="o-",
                    markersize=4,
                    linewidth=1,
                    capsize=3,
                    label=str(serial),
                )
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
    
