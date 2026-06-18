#!/usr/bin/env python3
import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
del _here, _sys, _os
import sys
import os
import html
from datetime import datetime, timedelta, timezone
import numpy as np
import math
import re
from functools import lru_cache
import warnings
import pandas as pd
import argparse

from PyQt5 import QtCore
from PyQt5.QtGui import (QBrush, QColor, QCursor, QPainter, QPalette, QPen, QStandardItemModel,
    QStandardItem, QKeySequence, QTextCursor, QFont
)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QToolTip, QFileDialog, QDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTabWidget, QStyle,
    QLabel, QComboBox, QPushButton, QRadioButton, QAction, QPlainTextEdit,
    QButtonGroup, QMessageBox, QSizePolicy, QSpacerItem, QCheckBox, QFrame, QShortcut,
    QLineEdit, QTextEdit, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QStyledItemDelegate, QStyleOptionViewItem,
)
from PyQt5.QtCore import Qt, QTimer, QUrl

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.text as mtext
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import RectangleSelector


import logos_instruments as li
from logos_agent_tools import LOGOSDataAgentTools
from logos_ai_agent import LOGOSChatAgent, api_key_available
from logos_timeseries import TimeseriesWidget
from logos_tanks import TanksWidget

import configparser as _configparser

try:
    import sys as _sys, os as _os
    _itxbin = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _itxbin not in _sys.path:
        _sys.path.insert(0, _itxbin)
    from ie3_cal_test import (
        cal_tank_coefs as _ie3_cal_tank_coefs,
        cal_tank_serials as _ie3_cal_tank_serials,
        weekly_aggregate as _ie3_weekly_aggregate,
        weekly_cal_fits as _ie3_weekly_cal_fits,
        _fill_value_for_date as _ie3_fill_value_for_date,
    )
    _IE3_CAL_AVAILABLE = True
    del _sys, _os, _itxbin
except ImportError:
    _IE3_CAL_AVAILABLE = False


def _load_app_config() -> _configparser.ConfigParser:
    """Load the author-managed logos_data.conf from the same directory as this file."""
    cfg = _configparser.ConfigParser()
    conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logos_data.conf')
    cfg.read(conf_path)
    return cfg


_APP_CONFIG = _load_app_config()


def _inst_cfg(inst_id: str) -> _configparser.SectionProxy | dict:
    """Return the config section for an instrument, or empty dict if absent."""
    section = f'instrument.{inst_id}'
    return _APP_CONFIG[section] if _APP_CONFIG.has_section(section) else {}


def _is_blank(value):
    """Return True for None, NaN, empty, or placeholder strings."""
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "none", "nan", "na", "null"}:
            return True
    return False


def _to_int_or_none(value):
    """Try to convert to int; return None on failure."""
    if _is_blank(value):
        return None
    try:
        i = int(str(value).strip())
        return i
    except Exception:
        return None


def _ie3_chunk_period(inst_cfg) -> str:
    """Return 'month' or 'halfmonth' for IE3 processing-tab chunking."""
    val = inst_cfg.get('chunk_period', 'month') if inst_cfg else 'month'
    val = (val or 'month').strip().lower()
    return val if val in {'month', 'halfmonth'} else 'month'


def _ie3_iter_chunks(t0, t1, period: str):
    """Yield (label, start_ts, end_exclusive_ts) covering [t0, t1].
    Calendar-aligned so labels stay stable across queries."""
    t0 = pd.Timestamp(t0)
    t1 = pd.Timestamp(t1)
    cur = pd.Timestamp(year=t0.year, month=t0.month, day=1)
    while cur <= t1:
        next_month = cur + pd.offsets.MonthBegin(1)
        if period == 'halfmonth':
            mid = cur + pd.Timedelta(days=15)  # the 16th, 00:00
            yield (f"{cur:%Y-%m} a", cur, mid)
            yield (f"{cur:%Y-%m} b", mid, next_month)
        else:
            yield (f"{cur:%Y-%m}", cur, next_month)
        cur = next_month


def _ie3_chunk_for_timestamp(ts, period: str):
    """Return (label, start, end_exclusive) of the chunk containing ts."""
    ts = pd.Timestamp(ts)
    if getattr(ts, 'tzinfo', None) is not None:
        ts = ts.tz_localize(None) if ts.tz is None else ts.tz_convert(None).tz_localize(None)
    month_start = pd.Timestamp(year=ts.year, month=ts.month, day=1)
    next_month = month_start + pd.offsets.MonthBegin(1)
    if period == 'halfmonth':
        mid = month_start + pd.Timedelta(days=15)
        if ts < mid:
            return (f"{month_start:%Y-%m} a", month_start, mid)
        return (f"{month_start:%Y-%m} b", mid, next_month)
    return (f"{month_start:%Y-%m}", month_start, next_month)


def _ie3_parse_chunk_label(label: str):
    """Return (start_ts, end_exclusive_ts) for a label like '2026-04' or '2026-04 a'."""
    base = (label or '').strip()
    if not base:
        return (None, None)
    suffix = None
    if ' ' in base:
        base, suffix = base.split(' ', 1)
        suffix = suffix.strip().lower()
    month_start = pd.Timestamp(base + '-01')
    next_month = month_start + pd.offsets.MonthBegin(1)
    mid = month_start + pd.Timedelta(days=15)
    if suffix == 'a':
        return (month_start, mid)
    if suffix == 'b':
        return (mid, next_month)
    return (month_start, next_month)


def _ie3_chunk_sql_range(label: str):
    """Return (start_str, end_str) suitable for IE3 load_data; end is inclusive."""
    start, end_excl = _ie3_parse_chunk_label(label)
    if start is None:
        return (None, None)
    end_inclusive = end_excl - pd.Timedelta(seconds=1)
    return (start.strftime('%Y-%m-%d %H:%M:%S'),
            end_inclusive.strftime('%Y-%m-%d %H:%M:%S'))


class RubberBandOverlay(QWidget):
    def __init__(self, parent, pen):
        super().__init__(parent)
        self.pen = pen
        self.rect = QtCore.QRectF()
        # Transparent background, don’t steal mouse events
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.hide()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        painter.setPen(self.pen)
        painter.drawRect(self.rect)
        painter.end()

class FastNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, main_window, on_flag_toggle=None):
        super().__init__(canvas, main_window)
        self.main_window = main_window
        self.rubberband = None

        pen = QPen(self.palette().color(QPalette.Highlight))
        pen.setStyle(QtCore.Qt.DashLine)
        self._overlay = RubberBandOverlay(canvas, pen)

        canvas.mpl_connect(
            "resize_event",
            lambda evt: self._overlay.setGeometry(0, 0, canvas.width(), canvas.height()),
        )

        self.on_flag_toggle = on_flag_toggle

    def home(self, *args, **kwargs):
        """Override home to always look at the data's ranges."""
        if hasattr(self, "main_window") and self.main_window:
            self.main_window._pending_xlim = None
            self.main_window._pending_ylim = None
            self.main_window.on_plot_type_changed(self.main_window.current_plot_type)
            self._nav_stack.clear()
            if hasattr(self, 'push_current'):
                self.push_current()
            self.set_history_buttons()
        else:
            super().home(*args, **kwargs)
                
    def zoom(self, *args, **kwargs):
        super().zoom(*args, **kwargs)
        if getattr(self, "mode", None):
            try:
                self.main_window._tagging_btn.setChecked(False)
            except Exception:
                pass

    def pan(self, *args, **kwargs):
        super().pan(*args, **kwargs)
        if getattr(self, "mode", None):
            try:
                self.main_window._tagging_btn.setChecked(False)
            except Exception:
                pass
            
    def draw_rubberband(self, event, x0, y0, x1, y1):
        w, h = self.canvas.width(), self.canvas.height()
        self._overlay.setGeometry(0, 0, w, h)

        y0i = h - y0
        y1i = h - y1

        self._overlay.rect = QtCore.QRectF(
            QtCore.QPointF(x0, y0i), QtCore.QPointF(x1, y1i)
        ).normalized()

        self._overlay.show()
        self._overlay.update()

    def _save_y_limits_if_locked(self, context):
        if self.main_window.lock_y_axis_cb.isChecked():
            ax = self.canvas.figure.gca()
            self.main_window.y_axis_limits = ax.get_ylim()
            #print(f"Y-Axis limits saved after {context}:", self.main_window.y_axis_limits)

    def press_zoom(self, event):
        self._old_aa = rcParams["lines.antialiased"]
        rcParams["lines.antialiased"] = False
        super().press_zoom(event)

    def release_zoom(self, event):
        super().release_zoom(event)
        rcParams["lines.antialiased"] = self._old_aa
        self._overlay.hide()
        self._save_y_limits_if_locked("zoom")

    def press_pan(self, event):
        self._old_aa = rcParams["lines.antialiased"]
        rcParams["lines.antialiased"] = False
        super().press_pan(event)

    def release_pan(self, event):
        super().release_pan(event)
        rcParams["lines.antialiased"] = self._old_aa
        self._save_y_limits_if_locked("pan")
        
_TAG_LAYOUT = [
    ("Sampling/Collection Issues", [
        ("B", "Leaky flask valve",                                   168, 223),
        ("F", "Insufficient flushing of sample collection system",     6, 222),
        ("K", "Contamination (Room air)",                              8,   9),
        ("L", "Leak in sample collection system",                     10,  11),
        ("P", "Bad flask pair agreement",                             14,  15),
        ("T", "Test sample collected not used in data analysis",      16, 189),
        ("U", "Unknown sample collection problem",                    17,  18),
        ("V", "Insufficient sample pressure",                         19, 136),
    ]),
    ("Measurement Issues", [
        ("A", "Known measurement problem",                           282, 326),
        ("A", "Valco Valve Problem",                                 143, 164),
        ("C", "Mole fraction falls outside of calibration",          107, 327),
        ("G", "Chromatography issue",                                290, 291),
        ("M", "Agilent (MS or GC) device issue",                     132, 133),
        ("O", "Measurement lab operator error",                       43, 121),
        ("U", "Unknown measurement problem",                         141, 142),
    ]),
    ("Automatic Tags", [
        ("C", "Mole fraction falls outside of calibration",          286, 287),
        ("G", "Chromatography issue",                                316,   0),
        ("P", "Bad flask pair agreement",                            167,   0),
        ("V", "Out of range sample pressure",                         32,   0),
        ("W", "Rejected in GCwerks integration",                     324,   0),
    ]),
]

_INFO_TAG_NUMS = frozenset(
    i_tag
    for _, entries in _TAG_LAYOUT
    for _, _, _r, i_tag in entries
    if i_tag
)

_INFO_TAG_DESCRIPTIONS = {
    i_tag: desc
    for _, entries in _TAG_LAYOUT
    for _, desc, _r, i_tag in entries
    if i_tag
}

# Auto tags the user may manually remove (but not add).
# 316: first-reference-run flag set by m4_gcwerks2db; qc_status is moved to
# 'F' at the same time, so removing the tag won't cause it to be reapplied.
_USER_REMOVABLE_AUTO_TAGS = frozenset({316})


class _TagComboDelegate(QStyledItemDelegate):
    """Item delegate for the single-tag combobox: shows light yellow on hover/selection."""
    _HOVER_BG = QColor("#fef9c3")

    def paint(self, painter, option, index):
        is_active = bool(option.state & (QStyle.State_MouseOver | QStyle.State_Selected))
        bg = self._HOVER_BG if is_active else index.data(Qt.BackgroundRole)
        painter.save()
        if isinstance(bg, QColor):
            painter.fillRect(option.rect, bg)
        text = index.data(Qt.DisplayRole) or ""
        painter.setPen(QColor("#222222"))
        painter.drawText(option.rect.adjusted(6, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, text)
        painter.restore()


class MultiTagPanel(QWidget):
    """Floating panel: R/I tag checkboxes grouped by Sampling, Measurement, and Auto categories."""

    def __init__(self, main_window):
        super().__init__(None, Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.mw = main_window
        self.setWindowTitle("Multi-Tag")
        self._mf_nums: list[int] = []
        self._row_idxs: list = []
        self._total_points: int = 0
        self._updating = False
        self._rows: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._select_btn = QPushButton("Select Region")
        self._select_btn.setCheckable(True)
        self._select_btn.setToolTip("Drag a rectangle on the plot to select multiple points")
        self._select_btn.setStyleSheet("""
            QPushButton {
                padding: 2px 8px;
                border: 1px solid #aaa;
                border-radius: 6px;
                background-color: #f0f0f0;
            }
            QPushButton:hover:!checked { background-color: #e0e0e0; }
            QPushButton:checked {
                background-color: #2e7d32;
                color: white;
                font-weight: 600;
                border: 1px solid #1b5e20;
            }
        """)
        self._select_btn.toggled.connect(self._on_select_btn_toggled)
        layout.addWidget(self._select_btn)

        self._info_label = QLabel("No point selected")
        self._info_label.setStyleSheet("color: gray; font-style: italic; font-size: 11px;")
        layout.addWidget(self._info_label)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["R", "I", "N", "Description"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Fixed)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setColumnWidth(0, 30)
        self._table.setColumnWidth(1, 30)
        self._table.setColumnWidth(2, 22)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QTableWidget.NoSelection)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self._table)

        # Comment section
        _comment_yellow = "#fffde7"
        comment_header = QWidget()
        comment_header.setStyleSheet(f"background-color: {_comment_yellow};")
        ch_layout = QHBoxLayout(comment_header)
        ch_layout.setContentsMargins(6, 3, 4, 3)
        ch_label = QLabel("Tag Comment")
        ch_label.setStyleSheet(f"background-color: {_comment_yellow}; font-weight: bold; font-size: 11px;")
        self._save_comment_btn = QPushButton("Save/Update Comment")
        self._save_comment_btn.setEnabled(False)
        self._save_comment_btn.clicked.connect(self._save_comment)
        ch_layout.addWidget(ch_label)
        ch_layout.addStretch()
        ch_layout.addWidget(self._save_comment_btn)
        layout.addWidget(comment_header)

        self._comment_edit = QTextEdit()
        self._comment_edit.setPlaceholderText("Enter comment…")
        self._comment_edit.setStyleSheet(f"background-color: {_comment_yellow};")
        self._comment_edit.setEnabled(False)
        self._comment_edit.setFixedHeight(62)
        layout.addWidget(self._comment_edit)

        copy_row = QWidget()
        copy_layout = QHBoxLayout(copy_row)
        copy_layout.setContentsMargins(4, 4, 4, 2)
        copy_layout.addStretch()
        self._copy_tags_btn = QPushButton("Copy Tags to all Analytes")
        self._copy_tags_btn.setToolTip(
            "Propagate pending reject tags to all analytes for the same injection,\n"
            "then recalculate and save mole fractions for all analytes."
        )
        self._copy_tags_btn.clicked.connect(self._copy_tags_to_all)
        copy_layout.addWidget(self._copy_tags_btn)
        layout.addWidget(copy_row)

        self.setMinimumWidth(420)
        self._build_layout()

    def _build_layout(self):
        self._rows = []
        row_count = sum(1 + len(entries) for _, entries in _TAG_LAYOUT)
        self._table.setRowCount(row_count)

        section_bg = QColor("#c8d4e8")

        table_row = 0
        for section_name, entries in _TAG_LAYOUT:
            is_auto_section = (section_name == "Automatic Tags")

            hdr = QTableWidgetItem(f"  {section_name}")
            hdr.setBackground(QBrush(section_bg))
            hdr.setForeground(QBrush(QColor("#1a2a4a")))
            f = hdr.font()
            f.setBold(True)
            hdr.setFont(f)
            hdr.setFlags(Qt.ItemIsEnabled)
            self._table.setItem(table_row, 0, hdr)
            self._table.setSpan(table_row, 0, 1, 4)
            self._table.setRowHeight(table_row, 22)
            table_row += 1

            for letter, desc, r_tag, i_tag in entries:
                r_cb = QCheckBox()
                r_cb.setTristate(True)
                r_cb.setEnabled(False)
                r_wrap = QWidget()
                r_box = QHBoxLayout(r_wrap)
                r_box.addWidget(r_cb)
                r_box.setAlignment(Qt.AlignCenter)
                r_box.setContentsMargins(0, 0, 0, 0)
                self._table.setCellWidget(table_row, 0, r_wrap)
                is_removable = is_auto_section and r_tag in _USER_REMOVABLE_AUTO_TAGS
                if not is_auto_section or is_removable:
                    r_cb.clicked.connect(
                        lambda _c, tnum=r_tag, cb=r_cb: self._on_clicked(cb, tnum, is_reject=True)
                    )

                i_cb = None
                if i_tag:
                    i_cb = QCheckBox()
                    i_cb.setTristate(True)
                    i_cb.setEnabled(False)
                    i_wrap = QWidget()
                    i_box = QHBoxLayout(i_wrap)
                    i_box.addWidget(i_cb)
                    i_box.setAlignment(Qt.AlignCenter)
                    i_box.setContentsMargins(0, 0, 0, 0)
                    self._table.setCellWidget(table_row, 1, i_wrap)
                    if not is_auto_section:
                        i_cb.clicked.connect(
                            lambda _c, tnum=i_tag, cb=i_cb: self._on_clicked(cb, tnum, is_reject=False)
                        )

                n_item = QTableWidgetItem(letter)
                n_item.setTextAlignment(Qt.AlignCenter)
                n_item.setFlags(Qt.ItemIsEnabled)
                fn = n_item.font()
                fn.setBold(True)
                n_item.setFont(fn)
                self._table.setItem(table_row, 2, n_item)

                desc_item = QTableWidgetItem(desc)
                desc_item.setFlags(Qt.ItemIsEnabled)
                self._table.setItem(table_row, 3, desc_item)

                self._rows.append({
                    "table_row": table_row,
                    "r_tag": r_tag,
                    "i_tag": i_tag,
                    "is_auto": is_auto_section,
                    "is_removable": is_removable,
                    "r_cb": r_cb,
                    "i_cb": i_cb,
                })
                table_row += 1

        self._table.resizeRowsToContents()
        self._fit_table_height()

    def _fit_table_height(self):
        header_h = self._table.horizontalHeader().height()
        rows_h = sum(self._table.rowHeight(r) for r in range(self._table.rowCount()))
        self._table.setFixedHeight(header_h + rows_h + 4)
        self.adjustSize()

    def _on_select_btn_toggled(self, checked: bool):
        self.mw._set_multi_tag_select(checked)

    def _save_comment(self):
        if not self._mf_nums:
            return
        try:
            self.mw._save_comment_for_mf_nums(self._mf_nums, self._comment_edit.toPlainText().strip() or None)
        except Exception as exc:
            print(f"MultiTagPanel save comment error: {exc}")

    def _copy_tags_to_all(self):
        try:
            self.mw.update_all_analytes()
        except Exception as exc:
            print(f"Copy tags error: {exc}")

    def populate_tags(self, all_tags_ordered: list):
        pass  # layout is static; kept for API compatibility

    def update_for_point(self, row_idxs: list, mf_nums: list[int],
                         tag_counts: dict[int, int], total: int = 1):
        self._row_idxs = row_idxs
        self._mf_nums = mf_nums
        self._total_points = total
        self._updating = True
        try:
            if len(row_idxs) == 1:
                try:
                    row = self.mw.run.loc[row_idxs[0]]
                    rt = row.get("run_time") or row.get("datetime") or ""
                    self._info_label.setText(str(rt)[:19] if rt else f"row {row_idxs[0]}")
                except Exception:
                    self._info_label.setText(f"row {row_idxs[0]}")
            else:
                self._info_label.setText(f"{len(row_idxs)} points selected")

            for entry in self._rows:
                r_tag = entry["r_tag"]
                i_tag = entry["i_tag"]
                is_auto = entry["is_auto"]
                r_cb: QCheckBox = entry["r_cb"]
                i_cb: QCheckBox | None = entry["i_cb"]

                r_count = tag_counts.get(r_tag, 0)
                is_removable = entry.get("is_removable", False)
                if is_removable:
                    # Enabled only when all selected points carry the tag — remove only, no re-add.
                    r_cb.setEnabled(bool(mf_nums) and r_count >= total)
                else:
                    r_cb.setEnabled(bool(mf_nums) and not is_auto)
                if r_count == 0:
                    r_cb.setCheckState(Qt.Unchecked)
                elif r_count >= total:
                    r_cb.setCheckState(Qt.Checked)
                else:
                    r_cb.setCheckState(Qt.PartiallyChecked)

                if i_cb is not None:
                    i_count = tag_counts.get(i_tag, 0)
                    i_cb.setEnabled(bool(mf_nums) and not is_auto)
                    if i_count == 0:
                        i_cb.setCheckState(Qt.Unchecked)
                    elif i_count >= total:
                        i_cb.setCheckState(Qt.Checked)
                    else:
                        i_cb.setCheckState(Qt.PartiallyChecked)

            self._comment_edit.setEnabled(bool(mf_nums))
            self._save_comment_btn.setEnabled(bool(mf_nums) and bool(tag_counts))
            comment = self.mw._fetch_first_comment_for_mf_nums(mf_nums) if mf_nums else ""
            self._comment_edit.setPlainText(comment)

        finally:
            self._updating = False

    def _on_clicked(self, cb: QCheckBox, tag_num: int, is_reject: bool):
        try:
            self._on_clicked_inner(cb, tag_num, is_reject)
        except Exception as exc:
            print(f"MultiTagPanel click error: {exc}")

    def _on_clicked_inner(self, cb: QCheckBox, tag_num: int, is_reject: bool):
        if self._updating or not self._mf_nums:
            return
        new_state = cb.checkState()

        if new_state == Qt.PartiallyChecked:
            self._updating = True
            cb.setCheckState(Qt.Checked)
            self._updating = False
            apply = True
        elif new_state == Qt.Checked:
            apply = True
        else:
            apply = False

        if apply:
            self.mw._insert_tag_for_mf_nums(self._mf_nums, tag_num)
            if is_reject:
                for idx in self._row_idxs:
                    self.mw.run.at[idx, "_pending_tag_num"] = tag_num
                    self.mw.run.at[idx, "rejected"] = 1
            elif tag_num in _INFO_TAG_NUMS:
                for idx in self._row_idxs:
                    self.mw.run.at[idx, 'has_info_tag'] = True
        else:
            self.mw._delete_tag_for_mf_nums(self._mf_nums, tag_num)
            self.mw._sync_rejected_state(self._row_idxs)

        self.mw.madechanges = True
        self.mw._style_gc_buttons()
        ax = self.mw.figure.axes[0] if self.mw.figure.axes else None
        if ax:
            self.mw._pending_xlim = ax.get_xlim()
            self.mw._pending_ylim = ax.get_ylim()
        self.mw.run = self.mw.instrument.norm.merge_smoothed_data(self.mw.run)
        self.mw.run = self.mw.instrument.calc_mole_fraction(self.mw.run)
        self.mw.gc_plot(self.mw._current_yparam)

    def clear_selection(self):
        self._mf_nums = []
        self._row_idxs = []
        self._total_points = 0
        self._info_label.setText("No point selected")
        self._comment_edit.setPlainText("")
        self._comment_edit.setEnabled(False)
        self._save_comment_btn.setEnabled(False)
        self._updating = True
        try:
            for entry in self._rows:
                entry["r_cb"].setEnabled(False)
                entry["r_cb"].setCheckState(Qt.Unchecked)
                if entry["i_cb"] is not None:
                    entry["i_cb"].setEnabled(False)
                    entry["i_cb"].setCheckState(Qt.Unchecked)
        finally:
            self._updating = False


class MainWindow(QMainWindow):

    AUTO_TAG_NUMS = {316, 26, 25, 2, 32, 324}

    def __init__(self, instrument):
        # Notice: we call super().__init__(instrument=instrument_id) inside HATS_DB_Functions
        super().__init__()
        self.instrument = instrument   # e.g. an M4_Instrument("m4") instance
        self._inst_cfg = _inst_cfg(self.instrument.inst_id)

        self.setWindowTitle(f"{self.instrument.inst_id.upper()} Data Processing Application")

        # Keep track of analytes and run_times
        self.analytes     = self.instrument.analytes
        self.current_pnum = None
        self.current_channel = None  # channel is optional, e.g. for FE3
        self.current_run_times = []   # will be a sorted list of QDateTime strings
        self.current_run_time = None  # currently selected run_time (QDateTime string)
        self.run = pd.DataFrame()  # Placeholder for loaded data
        self.y_axis_limits = None  # Store y-axis limits when locked
        self.calibration_rb = QRadioButton()
        self.resp_rb = QRadioButton()
        self.fit_method_cb = QComboBox()
        self.draw2zero_cb = QCheckBox()
        self.oldcurves_cb = QCheckBox()
        self.calcurve_label = QLabel()
        self.calcurve_combo = QComboBox()
        self.calcurves = []
        self.selected_calc_curve = None  # currently selected calibration curve date
        self.smoothing_cb = QComboBox()
        self.autoscale_samples_rb = QRadioButton()
        self.autoscale_standard_rb = QRadioButton()
        self.autoscale_fullscale_rb = QRadioButton()
        self.autoscale_group = QButtonGroup()
        self.lock_y_axis_cb = QCheckBox()

        self.tagging_enabled = False
        self.tag_options = []
        self._all_tag_names = {}
        self._all_tags_ordered = []
        self.selected_tag = None
        self._last_selected_tag_num = 141
        self._highlighted_point = {}
        self.tag_select_cb = QComboBox()
        self._multi_tag_panel: MultiTagPanel | None = None
        self._rect_selector = None
        self._multi_tag_rect_selector = None
        self._pick_cid = None
        self._zoom_run_time = None  # pd.Timestamp (UTC, tz-naive) when zoomed to one GC run, else None
        self._ie3_exact_run_time = None  # pd.Timestamp for full IE3 carry-in run loads
        self._all_run_times = []  # sorted run_times in the loaded chunk, for nav arrows
        self._scatter_main = []
        self._current_yparam = None
        self._current_yvar = None
        self._pick_refresh = False
        self._pending_xlim = None
        self._pending_ylim = None
        self.madechanges = False
        self.smoothing_changed = False
        self.tabs = None
        
        self._save_payload = None       # data for the Save Cal2DB button
        self.run_type_num = None
        self._fit_method_manual = False
        self._fit_method_last_context = None
        self._fit_method_updating = False
        self.logos_ai_dialog = None

        self.init_ui()

    def _latest_data_month(self):
        """Return (end_year, end_month, start_year, start_month) based on the most
        recent run_time in the DB for this instrument. End = latest data month,
        start = 2 months earlier. Falls back to current date if the query fails."""
        inst_num = getattr(self.instrument, 'inst_num', None)
        run_time = None
        try:
            if inst_num in range(236, 245):  # IE3/CATS
                rows = self.instrument.db.doquery(
                    f"SELECT MAX(analysis_time) AS rt FROM hats.ng_insitu_analysis "
                    f"WHERE inst_num = {inst_num};"
                )
            else:
                rows = self.instrument.db.doquery(
                    f"SELECT MAX(analysis_datetime) AS rt FROM hats.ng_data_processing_view "
                    f"WHERE inst_num = {inst_num};"
                )
            if rows and rows[0]['rt'] is not None:
                run_time = pd.Timestamp(rows[0]['rt'])
        except Exception:
            pass
        if run_time is None:
            run_time = pd.Timestamp.now()
        end = run_time.replace(day=1)
        start = end - pd.DateOffset(months=2)
        return end.year, end.month, start.year, start.month

    def init_ui(self):
        # Central widget + top‐level layout
        central = QWidget()
        self.setCentralWidget(central)
        h_main = QHBoxLayout()
        central.setLayout(h_main)
        self.h_main = h_main

        # Create a matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.current_plot_type = 0

        # ── PROCESSING PANE ──
        processing_pane = QWidget()
        self.processing_pane = processing_pane
        processing_layout = QVBoxLayout()
        processing_layout.setContentsMargins(4, 4, 4, 4)
        processing_layout.setSpacing(6)
        processing_pane.setLayout(processing_layout)

        # ── DATE RANGE SELECTION ──
        date_gb = QGroupBox("DATE RANGE (BY MONTH)")
        date_layout = QVBoxLayout()
        date_layout.setContentsMargins(2, 2, 2, 2)
        date_layout.setSpacing(2)
        date_gb.setLayout(date_layout)

        # Start: Year / Month
        self.start_year_cb = QComboBox()
        self.start_month_cb = QComboBox()
        end_year, end_month, start_year, start_month = self._latest_data_month()
        max_year = max(end_year, datetime.now().year)
        for y in range(int(self.instrument.start_date[0:4]), max_year + 1):
            self.start_year_cb.addItem(str(y))
        for m in range(1, 13):
            self.start_month_cb.addItem(datetime(2000, m, 1).strftime("%b"))

        # End: Year / Month
        self.end_year_cb = QComboBox()
        self.end_month_cb = QComboBox()
        for y in range(int(self.instrument.start_date[0:4]), max_year + 1):
            self.end_year_cb.addItem(str(y))
        for m in range(1, 13):
            self.end_month_cb.addItem(datetime(2000, m, 1).strftime("%b"))

        self.end_year_cb.setCurrentText(str(end_year))
        self.end_month_cb.setCurrentIndex(end_month - 1)
        self.start_year_cb.setCurrentText(str(start_year))
        self.start_month_cb.setCurrentIndex(start_month - 1)

        # Apply button
        self.apply_date_btn = QPushButton("Apply ▶")
        self.apply_date_btn.clicked.connect(self.apply_dates)

        # Row 1: From / To selectors + Apply
        date_row1 = QHBoxLayout()
        date_row1.setSpacing(4)
        date_row1.addWidget(QLabel("From:"))
        date_row1.addWidget(self.start_year_cb)
        date_row1.addWidget(self.start_month_cb)
        date_row1.addWidget(QLabel("To:"))
        date_row1.addWidget(self.end_year_cb)
        date_row1.addWidget(self.end_month_cb)
        date_row1.addWidget(self.apply_date_btn)
        date_layout.addLayout(date_row1)

        # Row 2: month-step buttons
        _step_style = "font-size: 10px; padding: 1px 5px;"
        date_row2 = QHBoxLayout()
        date_row2.setSpacing(2)
        for months in (1, 2, 6):
            btn = QPushButton(f"+{months}M")
            btn.setToolTip(f"Set To date to From date + {months} months and apply")
            btn.setStyleSheet(_step_style)
            btn.clicked.connect(lambda _, m=months: self._set_end_from_start(m))
            date_row2.addWidget(btn)
        date_row2.addStretch()
        for months in (1, 2, 6):
            btn = QPushButton(f"-{months}M")
            btn.setToolTip(f"Set From date to To date - {months} months and apply")
            btn.setStyleSheet(_step_style)
            btn.clicked.connect(lambda _, m=months: self._set_start_from_end(m))
            date_row2.addWidget(btn)
        date_layout.addLayout(date_row2)

        processing_layout.addWidget(date_gb)

        # Save the "last applied" values
        self.last_applied = self.get_selected_dates()

        # Watch for changes
        self.start_year_cb.currentIndexChanged.connect(self.check_dirty)
        self.start_month_cb.currentIndexChanged.connect(self.check_dirty)
        self.end_year_cb.currentIndexChanged.connect(self.check_dirty)
        self.end_month_cb.currentIndexChanged.connect(self.check_dirty)

        # Run Type Selection ComboBox  
        self.runTypeCombo = QComboBox(self)
        self.runTypeCombo.addItems(list(self.instrument.RUN_TYPE_MAP.keys()))
        self.runTypeCombo.setCurrentText("All")
        self.runTypeCombo.currentTextChanged.connect(self.on_run_type_changed)
        
        # If you have a grid/box layout:
        runtype_row = QHBoxLayout()
        runtype_row.addWidget(QLabel("Run Type:"))
        runtype_row.addWidget(self.runTypeCombo)
        
        # Run Selection GroupBox
        run_gb = QGroupBox("RUN SELECTION")
        run_layout = QVBoxLayout()
        run_layout.setSpacing(6)
        run_gb.setLayout(run_layout)
        run_layout.addWidget(date_gb)
        
        # Change Run Type
        change_runtype = QGroupBox("CHANGE RUN TYPE")
        change_runtype_layout = QHBoxLayout()  # horizontal so all on one line
        change_runtype.setLayout(change_runtype_layout)

        # Label
        change_runtype_label = QLabel("New Type:")

        # Combo box (exclude 'All')
        # Enable only for certain instruments to change the set run_type_num on the loaded run
        self.change_runtype_enabled = self._inst_cfg.getboolean('change_run_type', fallback=self.instrument.inst_id in {'fe3', 'bld1'})
        self.change_runtype_cb = QComboBox()
        runtype_items = [k for k in self.instrument.RUN_TYPE_MAP.keys() if k != "All"]
        self.change_runtype_cb.addItems(runtype_items)
        self.change_runtype_cb.setCurrentIndex(-1)  # start blank
        self.change_runtype_cb.currentTextChanged.connect(self.on_change_run_type_db)

        # Save button
        self.save_runtype_btn = QPushButton("Save")
        self.save_runtype_btn.clicked.connect(self.on_save_run_type)
        self.save_runtype_btn.setEnabled(False)  # optional, enable when selection changes

        # Add widgets to layout (all in one row)
        change_runtype_layout.addWidget(change_runtype_label)
        change_runtype_layout.addWidget(self.change_runtype_cb)
        change_runtype_layout.addWidget(self.save_runtype_btn)
        change_runtype_layout.addStretch()  # push everything left, optional

        # Actual run_time selector + Prev/Next buttons
        runsel_hbox = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.on_prev_run)
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.on_next_run)
        self.run_cb = QComboBox()
        self.run_cb.setMinimumWidth(200)
        self.run_cb.currentIndexChanged.connect(self.on_run_changed)
        self._copy_run_time_btn = QPushButton("⎘")
        self._copy_run_time_btn.setToolTip("Copy run_time to clipboard")
        self._copy_run_time_btn.setFixedWidth(28)
        self._copy_run_time_btn.clicked.connect(self._copy_run_time)

        runsel_hbox.addWidget(self.prev_btn)
        runsel_hbox.addWidget(self.run_cb, stretch=1)
        runsel_hbox.addWidget(self.next_btn)
        runsel_hbox.addWidget(self._copy_run_time_btn)
        run_layout.addLayout(runsel_hbox)
        self._setup_run_shortcuts()
        run_layout.addLayout(runtype_row)

        # Add group box to main layout
        if self.change_runtype_enabled:
            run_layout.addWidget(change_runtype)

        # Edit Run Notes button
        self.edit_notes_btn = QPushButton("Edit/View Run Notes (n)")
        self.edit_notes_btn.setToolTip("Add or edit notes for this run. (n)")
        self.edit_notes_btn.clicked.connect(self.on_edit_run_notes)
        self.notes_shortcut = QShortcut(QKeySequence("n"), self)
        self.notes_shortcut.activated.connect(self.on_edit_run_notes)
        run_layout.addWidget(self.edit_notes_btn)

        processing_layout.addWidget(run_gb)

        # Molecule/Analyte Selection GroupBox
        analyte_gb = QGroupBox("ANALYTE SELECTION")
        analyte_layout = QVBoxLayout()
        analyte_layout.setSpacing(6)
        analyte_gb.setLayout(analyte_layout)

        self.analyte_widget = QWidget()
        self.analyte_layout = QGridLayout()
        self.analyte_layout.setSpacing(4)
        self.analyte_widget.setLayout(self.analyte_layout)
        analyte_layout.addWidget(self.analyte_widget)

        processing_layout.addWidget(analyte_gb)

        # Plot Type Selection GroupBox
        plot_gb = QGroupBox("PLOT TYPE SELECTION")
        self.plot_layout = QVBoxLayout()
        self.plot_layout.setSpacing(6)
        plot_gb.setLayout(self.plot_layout)

        self.plot_radio_group = QButtonGroup(self)
        self.resp_rb = QRadioButton("Response (r)")
        self.resp_rb.setToolTip("Switch to Response plot (shortcut: r)")
        self.ratio_rb = QRadioButton("Ratio (t)")
        self.ratio_rb.setToolTip("Switch to Ratio plot (shortcut: t)")
        self.mole_fraction_rb = QRadioButton("Mole Fraction (m)")
        self.mole_fraction_rb.setToolTip("Switch to Mole Fraction plot (shortcut: m)")
        self.calibration_rb = QRadioButton("Calibration")
        self.calibration_rb.setEnabled(False)
        self.draw2zero_cb = QCheckBox("Zero")
        self.oldcurves_cb = QCheckBox("Other Curves")

        self.plot_layout.addWidget(self.resp_rb)
        self.plot_layout.addWidget(self.ratio_rb)
        self.plot_layout.addWidget(self.mole_fraction_rb)
        
        self.calcurve_label = QLabel("Cal Date")
        self.calcurve_combo = QComboBox()
        self.calcurve_label.setVisible(False)   # hidden initially
        self.calcurve_combo.setVisible(False)  
        self.plot_layout.addWidget(self.calcurve_label)
        self.plot_layout.addWidget(self.calcurve_combo)
        self.calcurve_combo.currentIndexChanged.connect(self.on_calcurve_selected)

        # Fit method combo for Calibration row ---
        # IE3 (in-situ) offers the cal-fit methods cal12/cal1/cal2 (recorded as
        # mf_method_num); other instruments offer polynomial fit degrees.
        self.fit_method_cb = QComboBox()
        if self.instrument.inst_id == 'ie3':
            self.fit_method_cb.addItem("ref", self.instrument.MF_METHOD_REF)
            self.fit_method_cb.addItem("cal12", self.instrument.MF_METHOD_CAL12)
            self.fit_method_cb.addItem("cal1", self.instrument.MF_METHOD_CAL1)
            self.fit_method_cb.addItem("cal2", self.instrument.MF_METHOD_CAL2)
            self.fit_method_cb.setCurrentText("cal12")
            self._ie3_cal_method = self.instrument.MF_METHOD_CAL12
        else:
            self.fit_method_cb.addItem("Linear", 1)
            self.fit_method_cb.addItem("Quadratic", 2)
            self.fit_method_cb.addItem("Cubic", 3)
            self.fit_method_cb.setCurrentText("Quadratic")
        self.fit_method_cb.setEnabled(False)  # follows calibration availability

        self.current_fit_degree = 2
        self.fit_method_cb.currentIndexChanged.connect(self.on_fit_method_changed)

        # ── CALIBRATION ROW ──
        cal_row = QHBoxLayout()
        cal_row.addWidget(self.calibration_rb)
        cal_row.addSpacing(6)
        cal_row.addWidget(QLabel("Fit:"))
        cal_row.addWidget(self.fit_method_cb, 1)  # stretch so it hugs the right
        self.plot_layout.addLayout(cal_row)

        # ── EXTRA OPTIONS ROW ──
        self.draw2zero_cb.setEnabled(False)    # start disabled
        self.oldcurves_cb.setEnabled(False)    # start disabled

        cal_row2 = QHBoxLayout()
        cal_row2.addSpacing(24)  # indent to align with calibration radio
        cal_row2.addWidget(self.draw2zero_cb)
        cal_row2.addWidget(self.oldcurves_cb)
        cal_row2.addStretch(1)   # push them left
        self.plot_layout.addLayout(cal_row2)
        self.draw2zero_cb.clicked.connect(self.calibration_plot)
        self.oldcurves_cb.clicked.connect(self.calibration_plot)
        # -----------------------------------------------

        self.plot_radio_group.addButton(self.resp_rb, id=0)
        self.plot_radio_group.addButton(self.ratio_rb, id=1)
        self.plot_radio_group.addButton(self.mole_fraction_rb, id=2)
        self.plot_radio_group.addButton(self.calibration_rb, id=3)
        self.resp_rb.setChecked(True)
        self.plot_radio_group.idClicked[int].connect(self.on_plot_type_changed)
        self._setup_plot_shortcuts()

        # Options GroupBox
        options_gb = QGroupBox("OPTIONS")
        options_layout = QVBoxLayout()
        options_layout.setSpacing(6)
        options_gb.setLayout(options_layout)

        # --- Response smoothing combobox ---
        self.smoothing_label = QLabel("Response smoothing:")
        self.smoothing_cb = QComboBox()
        self.smoothing_cb.addItems([
            "Point to Point (p)",       # 1
            "2-point moving average",   # 3
            "Lowess 5 point (l)",        # 2
            "3-point boxcar",           # 4
            "5-point boxcar",           # 6
            "Lowess ~10 points",        # 5
        ])
        self.smoothing_cb.setCurrentIndex(3)  # show "Lowess 5 points" by default

        options_layout.addWidget(self.smoothing_label)
        options_layout.addWidget(self.smoothing_cb)
        self.smoothing_cb.currentIndexChanged.connect(self.on_smoothing_changed)
        self._setup_smoothing_shortcuts()
        self._setup_save_shortcuts()

        # --- Horizontal separator ---
        options_layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))  # space above
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        options_layout.addWidget(line)
        options_layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))  # space below

        self.autoscale_label = QLabel("Autoscale (a):")
        options_layout.addWidget(self.autoscale_label)

        self.autoscale_group = QButtonGroup(self)
        self.autoscale_samples_rb = QRadioButton("Autoscale Samples")
        self.autoscale_standard_rb = QRadioButton("Autoscale Standard")
        self.autoscale_fullscale_rb = QRadioButton("Fullscale")

        self.autoscale_group.addButton(self.autoscale_samples_rb, id=0)
        self.autoscale_group.addButton(self.autoscale_standard_rb, id=1)
        self.autoscale_group.addButton(self.autoscale_fullscale_rb, id=2)
        self.autoscale_samples_rb.setChecked(True)

        for rb in (self.autoscale_samples_rb, self.autoscale_standard_rb, self.autoscale_fullscale_rb):
            rb.toggled.connect(self.on_autoscale_mode_changed)
            options_layout.addWidget(rb)
        self._setup_autoscale_shortcuts()

        self.lock_y_axis_cb = QCheckBox("Lock Y-Axis Scale")
        self.lock_y_axis_cb.setChecked(False)
        self.lock_y_axis_cb.stateChanged.connect(self.on_lock_y_axis_toggled)
        options_layout.addWidget(self.lock_y_axis_cb)

        # Combine plot_gb and options_gb into a single group box
        combined_gb = QGroupBox("PLOT AND OPTIONS")
        combined_layout = QHBoxLayout()
        combined_layout.setSpacing(12)
        combined_gb.setLayout(combined_layout)

        combined_layout.addWidget(plot_gb, stretch=1)
        combined_layout.addWidget(options_gb, stretch=1)

        processing_layout.addWidget(combined_gb)

        # Tagging controls. The toolbar button still controls tag mode; this
        # selector controls which tag is applied to picked points.
        tag_gb = QGroupBox("TAGGING")
        tag_layout = QVBoxLayout()
        tag_layout.setSpacing(6)
        tag_gb.setLayout(tag_layout)
        tag_layout.addWidget(QLabel("Selected Tag:"))
        self.tag_select_cb.setMinimumWidth(260)
        self.tag_select_cb.currentIndexChanged.connect(self.on_tag_selection_changed)
        self.tag_select_cb.activated.connect(self._apply_tag_combobox_color)
        self.tag_select_cb.setItemDelegate(_TagComboDelegate(self.tag_select_cb))
        tag_layout.addWidget(self.tag_select_cb)

        _btn_style = """
            QPushButton {
                padding: 2px 8px;
                border: 1px solid #aaa;
                border-radius: 6px;
                background-color: #f0f0f0;
            }
            QPushButton:hover:!checked {
                background-color: #e0e0e0;
            }
            QPushButton:checked {
                background-color: #2e7d32;
                color: white;
                font-weight: 600;
                border: 1px solid #1b5e20;
            }
        """
        self._tagging_btn = QPushButton("Tagging (g)")
        self._tagging_btn.setCheckable(True)
        self._tagging_btn.setToolTip("Toggle tagging mode — drag a box or click a point to tag (g)")
        self._tagging_btn.setStyleSheet(_btn_style)
        self._tagging_btn.toggled.connect(self.on_flag_mode_toggled)
        tagging_sc = QShortcut(QKeySequence("G"), self)
        tagging_sc.activated.connect(self._cycle_g_mode)

        self._multi_tag_btn = QPushButton("Multi-Tag")
        self._multi_tag_btn.setCheckable(True)
        self._multi_tag_btn.setToolTip("Open floating panel to view/edit all tags for a selected point")
        self._multi_tag_btn.setStyleSheet(_btn_style)
        self._multi_tag_btn.toggled.connect(self._on_multi_tag_btn_toggled)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addWidget(self._tagging_btn)
        btn_row.addWidget(self._multi_tag_btn)
        tag_layout.addLayout(btn_row)
        self.hide_flagged_cb = QCheckBox("Hide Rejected Data")
        self.hide_flagged_cb.setChecked(False)
        self.hide_flagged_cb.stateChanged.connect(lambda: self.on_plot_type_changed(self.current_plot_type))
        tag_layout.addWidget(self.hide_flagged_cb)
        self.show_info_cb = QCheckBox("Show Info Tags")
        self.show_info_cb.setChecked(False)
        self.show_info_cb.stateChanged.connect(self._on_show_info_toggled)
        tag_layout.addWidget(self.show_info_cb)
        processing_layout.addWidget(tag_gb)
        self.populate_tag_selector()

        # --- Save Run Button ---
        if self._inst_cfg.getboolean('csv_export', fallback=self.instrument.inst_id in {'bld1'}):
            self.save_csv_btn = QPushButton("Save run to .csv file")
            self.save_csv_btn.setToolTip("Export the selected run to a CSV file")
            self.save_csv_btn.clicked.connect(lambda: self.export_csv(self.save_csv_btn))

            # Add it below the existing Plot and Options section
            processing_layout.addWidget(self.save_csv_btn) 

        # Stretch to push everything to the top
        processing_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # ── TABS ──
        _visible_tabs_raw = self._inst_cfg.get(
            'tabs',
            fallback='processing, timeseries, tanks, ai'
        )
        _visible_tabs = {t.strip().lower() for t in _visible_tabs_raw.split(',')}
        if not api_key_available():
            _visible_tabs.discard('ai')

        tabs = QTabWidget()
        tab_font = QFont()
        tab_font.setPointSize(11)
        tab_font.setWeight(QFont.Normal)
        tabs.tabBar().setFont(tab_font)
        tabs.tabBar().setStyleSheet("""
            QTabBar::tab {
                color: darkblue;
                padding: 8px 16px;
                border: 1px solid darkblue;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                font-weight: bold;
                background: white;
            }
            QTabBar::tab:!selected {
                color: #7090b0;
                border-color: #7090b0;
                margin-top: 2px;
            }
        """)
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid darkblue;
                margin-top: -1px;
            }
        """)
        tabs.addTab(processing_pane, "PROCESSING")

        if 'timeseries' in _visible_tabs:
            self.timeseries_tab = TimeseriesWidget(instrument=self.instrument, parent=self)
            tabs.addTab(self.timeseries_tab, "TIMESERIES")
        else:
            self.timeseries_tab = None

        if 'tanks' in _visible_tabs:
            self.tanks_tab = TanksWidget(instrument=self.instrument, parent=self)
            tabs.addTab(self.tanks_tab, "TANKS")
        else:
            self.tanks_tab = None

        if 'ai' in _visible_tabs:
            self.logos_ai_tab = LOGOSAITab(instrument=self.instrument, main_window=self, parent=self)
            tabs.addTab(self.logos_ai_tab, "LOGOS AI")
        else:
            self.logos_ai_tab = None

        self.tabs = tabs
        tabs.currentChanged.connect(self._on_tab_changed)

        left_container = QWidget()
        left_container.setMinimumWidth(420)
        left_container.setMaximumWidth(520)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(tabs)
        left_container.setLayout(left_layout)
        self.left_container = left_container

        # Right pane: matplotlib figure for plotting
        right_placeholder = QGroupBox("PLOT AREA")
        right_layout = QVBoxLayout()
        right_placeholder.setLayout(right_layout)
        right_layout.addWidget(self.canvas)

        # Add a NavigationToolbar for the figure
        self.toolbar = FastNavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        self.right_placeholder = right_placeholder

        right_spacer = QWidget()
        right_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_spacer.setVisible(False)
        self.right_spacer = right_spacer

        # Add both panes to the main hbox
        h_main.addWidget(left_container, stretch=0)  # Fixed width for left pane
        h_main.addWidget(right_placeholder, stretch=1)  # Flexible width for right pane
        h_main.addWidget(right_spacer, stretch=1)

        self.populate_analyte_controls()
        
        # Kick off by selecting the default analyte from config
        _default_analyte = self._inst_cfg.get('default_analyte', fallback=None)
        if not _default_analyte:
            # Instrument-class override (e.g. IE3 sets DEFAULT_ANALYTE_NAME)
            _default_analyte = getattr(self.instrument, 'DEFAULT_ANALYTE_NAME', None)

        if _default_analyte and _default_analyte in self.analytes:
            if '(' in _default_analyte and ')' in _default_analyte:
                self.current_channel = _default_analyte.split('(')[1].split(')')[0].strip()
            else:
                self.current_channel = None
            self.current_pnum = int(self.analytes[_default_analyte])
            self._select_analyte_control(_default_analyte)
            if hasattr(self, "timeseries_tab") and self.timeseries_tab:
                self.timeseries_tab.set_current_analyte(_default_analyte)
            self.apply_dates()
        elif self.instrument.inst_id == 'm4':
            # Fallback for m4 if conf is missing
            self.current_pnum = 20
            self._select_analyte_control('HFC134a')
            self.apply_dates()

        self.figure.tight_layout(rect=[0, 0, 1.05, 1])
        self.canvas.draw_idle()
        self.gc_plot(self._current_yparam)
        self._on_tab_changed(self.tabs.currentIndex())

    def _on_tab_changed(self, _idx: int):
        current = self.tabs.currentWidget() if self.tabs else None
        if not current:
            return
        if current is self.logos_ai_tab:
            self.left_container.setMinimumWidth(0)
            self.left_container.setMaximumWidth(16777215)
            self.left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.right_placeholder.setVisible(False)
            self.right_spacer.setVisible(False)
            self.h_main.setStretch(0, 1)
            self.h_main.setStretch(1, 0)
            self.h_main.setStretch(2, 0)
            return
        if current is self.timeseries_tab:
            width = max(
                self.processing_pane.sizeHint().width(),
                self.processing_pane.minimumSizeHint().width(),
                self.processing_pane.minimumWidth(),
                self.processing_pane.width(),
            )
        else:
            width = max(current.sizeHint().width(), current.minimumSizeHint().width())
            if width <= 0:
                width = current.minimumWidth() or current.width() or 420
        self.left_container.setFixedWidth(width)
        self.left_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        show_plot = current is self.processing_pane
        self.right_placeholder.setVisible(show_plot)
        self.right_spacer.setVisible(not show_plot)
        self.h_main.setStretch(0, 0)
        self.h_main.setStretch(1, 1)
        self.h_main.setStretch(2, 1)

    def undock_logos_ai_tab(self):
        if self.logos_ai_tab is None or self.logos_ai_dialog is not None or self.tabs is None:
            return
        idx = self.tabs.indexOf(self.logos_ai_tab)
        if idx >= 0:
            self.tabs.removeTab(idx)
        self.logos_ai_tab.setParent(None)
        self.logos_ai_dialog = LOGOSAIDialog(self, self.logos_ai_tab)
        self.logos_ai_tab.set_popout_state(True)
        self.logos_ai_tab._render_transcript()
        self.logos_ai_tab.show()
        self.logos_ai_dialog.show()
        self.logos_ai_dialog.raise_()
        self.logos_ai_dialog.activateWindow()
        if self.tabs.count():
            self.tabs.setCurrentIndex(0)
            self._on_tab_changed(self.tabs.currentIndex())

    def redock_logos_ai_tab(self):
        if self.logos_ai_tab is None or self.tabs is None:
            return
        self.logos_ai_tab.setParent(None)
        if self.tabs.indexOf(self.logos_ai_tab) < 0:
            self.tabs.addTab(self.logos_ai_tab, "LOGOS AI")
        self.logos_ai_tab.set_popout_state(False)
        self.tabs.setCurrentWidget(self.logos_ai_tab)
        self.logos_ai_tab._render_transcript()
        self.logos_ai_tab.show()
        dialog = self.logos_ai_dialog
        self.logos_ai_dialog = None
        if dialog is not None:
            dialog.hide()
            dialog.deleteLater()
        self._on_tab_changed(self.tabs.currentIndex())

    def _cycle_g_mode(self):
        """Cycle G key: off → tagging → multi-tag → off → ..."""
        if self._tagging_btn.isChecked():
            # tagging on → switch to multi-tag
            self._tagging_btn.setChecked(False)
            self._multi_tag_btn.setChecked(True)
        elif self._multi_tag_panel is not None and self._multi_tag_panel.isVisible():
            # multi-tag open → close it (back to off)
            self._multi_tag_btn.setChecked(False)
        else:
            # both off → start tagging
            self._tagging_btn.setChecked(True)

    def on_flag_mode_toggled(self, checked: bool):
        self.tagging_enabled = checked
        self.canvas.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)

        tb = getattr(self.canvas, "toolbar", None)
        if tb is not None:
            # Always turn off zoom/pan modes first
            if tb.mode == "zoom rect":
                tb.zoom()
            elif tb.mode == "pan/zoom":
                tb.pan()
            tb.mode = ""

            # Disable/enable the buttons themselves
            if "pan" in tb._actions and "zoom" in tb._actions:
                tb._actions["pan"].setEnabled(not checked)
                tb._actions["zoom"].setEnabled(not checked)

        if checked:
            #print("Enabling rectangle selector")
            if self._rect_selector is None:
                self._rect_selector = RectangleSelector(
                    ax=self.figure.axes[0],
                    onselect=self._on_box_select,
                    useblit=True,
                    button=[1],   # left mouse button
                    minspanx=5, minspany=5,
                    spancoords="pixels",
                    drag_from_anywhere=True,
                    ignore_event_outside=False
                )
            self._rect_selector.set_active(True)
        else:
            if self._rect_selector is not None:
                self._rect_selector.set_active(False)

    def populate_tag_selector(self):
        """Load HATS NG tag options used by the point tagging tool."""
        self.tag_select_cb.blockSignals(True)
        self.tag_select_cb.clear()
        self.tag_options = []
        # Single query: manual tags sorted by flag, auto-tags appended at end.
        try:
            all_rows = self.instrument.db.doquery("""
                SELECT tag_num, internal_flag, display_name, reject
                FROM ccgg.tag_view
                WHERE hats_ng = 1
                ORDER BY
                    CASE WHEN tag_num IN (316, 26, 25, 2, 32, 324) THEN 1 ELSE 0 END,
                    flag;
            """)
        except Exception as e:
            print(f"Could not load tag options: {e}")
            self.tag_select_cb.addItem("Tag options unavailable", None)
            self.tag_select_cb.setEnabled(False)
            self.tag_select_cb.blockSignals(False)
            return

        # _all_tags_ordered: full list in hats_sort order, used by MultiTagPanel.
        # tag_options: Sampling/Measurement R-tags only, built from _TAG_LAYOUT.
        self._all_tag_names = {}
        self._all_tags_ordered = []   # list of (tag_dict, is_auto)
        for row in (all_rows or []):
            tnum = int(row["tag_num"])
            dname = row.get("display_name") or row.get("name") or ""
            self._all_tag_names[tnum] = dname
            tag = {
                "tag_num": tnum,
                "internal_flag": row.get("internal_flag") or "",
                "display_name": dname,
                "reject": int(row.get("reject") or 0),
            }
            is_auto = tnum in self.AUTO_TAG_NUMS
            self._all_tags_ordered.append((tag, is_auto))

        # Populate combobox from _TAG_LAYOUT: Sampling and Measurement sections,
        # R tags only — matches the Multi-Tag panel descriptions, no Auto tags.
        _section_colors = {
            "Sampling/Collection Issues": QColor("#dce8f5"),
            "Measurement Issues":         QColor("#fde8d0"),
        }
        self.tag_options = []
        for section_name, entries in _TAG_LAYOUT:
            if section_name == "Automatic Tags":
                continue
            row_color = _section_colors.get(section_name)
            for letter, desc, r_tag, i_tag in entries:
                if not r_tag:
                    continue
                tag = {
                    "tag_num": r_tag,
                    "internal_flag": letter,
                    "display_name": desc,
                    "reject": 1,
                }
                self.tag_options.append(tag)
                label = f"{letter}: {desc}"
                if len(label) > 90:
                    label = label[:87] + "..."
                self.tag_select_cb.addItem(label, tag)
                if row_color:
                    idx = self.tag_select_cb.count() - 1
                    self.tag_select_cb.setItemData(idx, row_color, Qt.BackgroundRole)

        default_tag_num = self._last_selected_tag_num or 141
        default_idx = next(
            (i for i, tag in enumerate(self.tag_options) if tag["tag_num"] == default_tag_num),
            0,
        )
        if self.tag_options:
            self.tag_select_cb.setCurrentIndex(default_idx)
            self.selected_tag = self.tag_options[default_idx]
        self.tag_select_cb.blockSignals(False)
        self._apply_tag_combobox_color(default_idx)

        if self._multi_tag_panel is not None:
            self._multi_tag_panel.populate_tags(self._all_tags_ordered)

    def on_tag_selection_changed(self, index):
        self.selected_tag = self.tag_select_cb.itemData(index) if index >= 0 else None
        if self.selected_tag:
            self._last_selected_tag_num = int(self.selected_tag["tag_num"])

    def _apply_tag_combobox_color(self, index):
        bg = self.tag_select_cb.itemData(index, Qt.BackgroundRole)
        if isinstance(bg, QColor):
            self.tag_select_cb.setStyleSheet(
                f"QComboBox {{ background-color: {bg.name()}; color: #222; }}"
            )
        else:
            self.tag_select_cb.setStyleSheet("")

    def _tag_table_info(self):
        if self.instrument.inst_id in ('ie3', 'cats'):
            return (
                'hats.ng_insitu_mole_fraction_tags',
                'ng_insitu_mole_fraction_num',
                'hats.ng_insitu_mole_fractions',
                'mf_num',
            )
        return (
            'hats.ng_mole_fraction_tags',
            'ng_mole_fraction_num',
            'hats.ng_mole_fractions',
            'ng_mole_fraction_num',
        )

    def _mole_fraction_nums_for_indices(self, idxs):
        """Return mole-fraction primary keys for DataFrame rows being tagged."""
        if idxs is None or len(idxs) == 0:
            return []
        _tag_table, _tag_key, mf_table, id_col = self._tag_table_info()
        subset = self.run.loc[list(idxs)]
        if id_col in subset.columns:
            vals = pd.to_numeric(subset[id_col], errors='coerce').dropna().astype(int)
            return sorted(vals.unique().tolist())

        if not {'analysis_num', 'parameter_num', 'channel'}.issubset(subset.columns):
            return []

        keys = (
            subset[['analysis_num', 'parameter_num', 'channel']]
            .dropna()
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        keys = list(keys)
        if not keys:
            return []

        out = []
        for i in range(0, len(keys), 400):
            chunk = keys[i:i + 400]
            terms = []
            params = []
            for analysis_num, parameter_num, channel in chunk:
                terms.append("(analysis_num = %s AND parameter_num = %s AND channel = %s)")
                params.extend([int(analysis_num), int(parameter_num), str(channel)])
            rows = self.instrument.db.doquery(
                f"SELECT num FROM {mf_table} WHERE {' OR '.join(terms)}",
                params,
            )
            out.extend(int(r["num"]) for r in rows)
        return sorted(set(out))

    def _insert_tag_for_indices(self, idxs, tag_num):
        mf_nums = self._mole_fraction_nums_for_indices(idxs)
        if not mf_nums:
            return 0
        tag_table, tag_key, _mf_table, _id_col = self._tag_table_info()
        sql = f"""
            INSERT IGNORE INTO {tag_table} ({tag_key}, tag_num)
            VALUES (%s, %s);
        """
        params = [(num, int(tag_num)) for num in mf_nums]
        self.instrument.db.doMultiInsert(sql, params, all=True)
        return len(mf_nums)

    def _insert_tag_for_mf_nums(self, mf_nums: list[int], tag_num: int):
        """Insert a tag directly given mole-fraction primary keys."""
        if not mf_nums:
            return
        tag_table, tag_key, _, _ = self._tag_table_info()
        sql = f"INSERT IGNORE INTO {tag_table} ({tag_key}, tag_num) VALUES (%s, %s);"
        self.instrument.db.doMultiInsert(sql, [(n, tag_num) for n in mf_nums], all=True)

    def _delete_tag_for_mf_nums(self, mf_nums: list[int], tag_num: int):
        """Remove a specific tag from the given mole-fraction primary keys."""
        if not mf_nums:
            return
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        self.instrument.db.doquery(
            f"DELETE FROM {tag_table} WHERE {tag_key} IN ({placeholders}) AND tag_num = %s;",
            list(mf_nums) + [tag_num],
        )

    def _fetch_tag_nums_for_mf_nums(self, mf_nums: list[int]) -> set[int]:
        """Return the set of tag_nums currently applied to the given mf primary keys."""
        if not mf_nums:
            return set()
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        rows = self.instrument.db.doquery(
            f"SELECT DISTINCT tag_num FROM {tag_table} WHERE {tag_key} IN ({placeholders});",
            list(mf_nums),
        )
        return {int(r["tag_num"]) for r in (rows or [])}

    def _fetch_tag_counts_for_mf_nums(self, mf_nums: list[int]) -> dict[int, int]:
        """Return {tag_num: count} for how many of the given mf_nums carry each tag."""
        if not mf_nums:
            return {}
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        rows = self.instrument.db.doquery(
            f"SELECT tag_num, COUNT(*) AS cnt FROM {tag_table} "
            f"WHERE {tag_key} IN ({placeholders}) GROUP BY tag_num;",
            list(mf_nums),
        )
        return {int(r["tag_num"]): int(r["cnt"]) for r in (rows or [])}

    def _fetch_first_comment_for_mf_nums(self, mf_nums: list[int]) -> str:
        if not mf_nums:
            return ""
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        rows = self.instrument.db.doquery(
            f"SELECT comment FROM {tag_table} WHERE {tag_key} IN ({placeholders}) "
            f"AND comment IS NOT NULL AND comment != '' LIMIT 1;",
            list(mf_nums),
        )
        return (rows[0].get("comment") or "") if rows else ""

    def _save_comment_for_mf_nums(self, mf_nums: list[int], comment: str | None):
        if not mf_nums:
            return
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        self.instrument.db.doquery(
            f"UPDATE {tag_table} SET comment = %s WHERE {tag_key} IN ({placeholders});",
            [comment] + list(mf_nums),
        )

    def _on_multi_tag_btn_toggled(self, checked: bool):
        if checked:
            # If tagging (g) mode is active, turn it off before opening multi-tag.
            if self.tagging_enabled:
                self._tagging_btn.setChecked(False)
            if self._multi_tag_panel is None:
                panel = MultiTagPanel(self)
                panel.populate_tags(self._all_tags_ordered)
                # Keep button and combobox in sync when panel closed via title-bar X.
                # Must be a def, not a tuple-returning lambda — SIP requires None.
                def _panel_close(e):
                    e.accept()
                    self._set_multi_tag_select(False)
                    self._clear_highlight()
                    self._multi_tag_btn.setChecked(False)
                panel.closeEvent = _panel_close
                self._multi_tag_panel = panel
            self._multi_tag_panel.clear_selection()
            self._multi_tag_panel.show()
            self._multi_tag_panel.raise_()
            self.tag_select_cb.setEnabled(False)
            self._tagging_btn.setEnabled(False)
        else:
            self._set_multi_tag_select(False)
            if self._multi_tag_panel is not None:
                self._multi_tag_panel._select_btn.blockSignals(True)
                self._multi_tag_panel._select_btn.setChecked(False)
                self._multi_tag_panel._select_btn.blockSignals(False)
                self._multi_tag_panel.hide()
            self._clear_highlight()
            self.tag_select_cb.setEnabled(True)
            self._tagging_btn.setEnabled(True)

    def _set_multi_tag_select(self, enabled: bool):
        """Activate or deactivate the multi-tag window-select RectangleSelector."""
        if self._multi_tag_rect_selector is not None:
            self._multi_tag_rect_selector.set_active(False)
            self._multi_tag_rect_selector.disconnect_events()
            self._multi_tag_rect_selector = None
        if enabled and self.figure.axes:
            self._multi_tag_rect_selector = RectangleSelector(
                ax=self.figure.axes[0],
                onselect=self._on_multi_tag_box_select,
                useblit=True,
                button=[1],
                minspanx=5, minspany=5,
                spancoords="pixels",
                drag_from_anywhere=True,
                ignore_event_outside=False,
            )
            self._multi_tag_rect_selector.set_active(True)
            self.canvas.setCursor(Qt.CrossCursor)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)

    def _on_multi_tag_box_select(self, eclick, erelease):
        """Handle rectangle select in multi-tag mode: select all points in the box."""
        if self._multi_tag_panel is None or not self._multi_tag_panel.isVisible():
            return
        if not self._current_yvar or self.run.empty or self._current_yvar not in self.run.columns:
            return
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        if None in (x0, y0, x1, y1):
            return

        y_all = self.run[self._current_yvar].to_numpy()
        mask = (self._x_num >= x0) & (self._x_num <= x1) & (y_all >= y0) & (y_all <= y1)
        row_idxs = list(self.run.index[mask])
        if not row_idxs:
            return

        mf_nums = self._mole_fraction_nums_for_indices(row_idxs)
        tag_counts = self._fetch_tag_counts_for_mf_nums(mf_nums)
        self._multi_tag_panel.update_for_point(row_idxs, mf_nums, tag_counts, total=len(mf_nums))
        self._highlight_region(row_idxs, self._x_num[mask])

    def set_calibration_enabled(self, enabled: bool):
        # enable or disable the other checkboxes associated with calibration_rb
        self.fit_method_cb.setEnabled(enabled)
        self.draw2zero_cb.setEnabled(enabled)
        self.oldcurves_cb.setEnabled(enabled)

    def get_selected_dates(self):
        return {
            "start_year": self.start_year_cb.currentText(),
            "start_month": self.start_month_cb.currentText(),
            "end_year": self.end_year_cb.currentText(),
            "end_month": self.end_month_cb.currentText(),
        }

    def check_dirty(self):
        # Compare current selection to last applied
        if self.get_selected_dates() != self.last_applied:
            self.apply_date_btn.setStyleSheet("background-color: lightgreen;")
        else:
            self.apply_date_btn.setStyleSheet("")

    def _set_end_from_start(self, months: int):
        """Set the To date to From date + months and apply."""
        year = int(self.start_year_cb.currentText())
        month = self.start_month_cb.currentIndex() + 1
        total = year * 12 + (month - 1) + months
        new_year, new_month_0 = divmod(total, 12)
        new_month = new_month_0 + 1
        max_year = int(self.end_year_cb.itemText(self.end_year_cb.count() - 1))
        new_year = min(max_year, new_year)
        self.end_year_cb.setCurrentText(str(new_year))
        self.end_month_cb.setCurrentIndex(new_month - 1)
        self.apply_dates()

    def _set_start_from_end(self, months: int):
        """Set the From date to To date - months and apply."""
        year = int(self.end_year_cb.currentText())
        month = self.end_month_cb.currentIndex() + 1
        total = year * 12 + (month - 1) - months
        new_year, new_month_0 = divmod(total, 12)
        new_month = new_month_0 + 1
        min_year = int(self.start_year_cb.itemText(0))
        new_year = max(min_year, new_year)
        self.start_year_cb.setCurrentText(str(new_year))
        self.start_month_cb.setCurrentIndex(new_month - 1)
        self.apply_dates()

    def apply_dates(self):
        self.apply_date_btn.setStyleSheet("")
        self.resp_rb.setChecked(True)
        self.on_run_type_changed()

    def on_fit_method_changed(self, _idx: int):
        data = self.fit_method_cb.currentData()
        if data is None:
            return

        # IE3: the combo selects the cal-fit method (cal12/cal1/cal2/ref), not a
        # polynomial degree. Mark the run dirty so the change can be saved, and
        # re-render the cal-curve view live so the fit line reflects the choice.
        if self.instrument.inst_id == 'ie3':
            self._ie3_cal_method = int(data)
            if (self.current_run_time and '(Cal)' in self.current_run_time):
                self.madechanges = True
            if (self.plot_radio_group.checkedId() == 3
                    and self.current_run_time
                    and '(Cal)' in self.current_run_time):
                self._ie3_cal_plot()
            return

        self.current_fit_degree = int(data)  # 1/2/3
        if not self._fit_method_updating:
            self._fit_method_manual = True

        # If you're on the Calibration view, you can re-render immediately:
        if self.plot_radio_group.checkedId() == 3:
            self.on_plot_type_changed(3)
            
    def on_smoothing_changed(self, idx: int):
        #print(f"Smoothing changed to index: {idx} get_selected_detrend_method = {self.get_selected_detrend_method()}", )
        selected_method = self.get_selected_detrend_method()
        if selected_method is not None:
            if self._zoom_run_time is not None and 'run_time' in self.run.columns:
                # Zoom mode: only update detrend method for the zoomed run_time.
                rt_col = pd.to_datetime(self.run['run_time'], utc=True, errors='coerce')
                zr = pd.Timestamp(self._zoom_run_time)
                if zr.tzinfo is None:
                    zr = zr.tz_localize('UTC')
                mask = rt_col == zr
                self.run.loc[mask, 'detrend_method_num'] = selected_method
            else:
                self.run['detrend_method_num'] = selected_method

        # Re-smooth: merge_smoothed_data groups by run_time and reads each
        # group's detrend_method_num, so per-run choices are preserved.
        self.run = self.instrument.norm.merge_smoothed_data(self.run)
        self.run = self.instrument.calc_mole_fraction(self.run)
        self.madechanges = True
        self.smoothing_changed = True
        self._style_gc_buttons()

        # Redraw
        self.gc_plot(self._current_yparam, sub_info="Smoothing changed")
        
    def on_plot_type_changed(self, id: int):
        self.current_plot_type = id
        self.set_calibration_enabled(False)
        self.calcurve_label.setVisible(False)
        self.calcurve_combo.setVisible(False)
        
        if id == 0:
            self.gc_plot('resp')
        elif id == 1:
            self.gc_plot('ratio')
        elif id == 2:
            self.gc_plot('mole_fraction')
        else:
            self.set_calibration_enabled(True)
            self.calibration_plot()
        
        if id != self.current_plot_type:
            self.current_plot_type = id
            self.lock_y_axis_cb.setChecked(False)

        self.set_runtype_combo()

    def _fmt_gc_plot(self, x, y):
        return f"x={mdates.num2date(x).strftime('%Y-%m-%d %H:%M')}  y={y:0.3g}"

    def _fmt_cal_plot(self, x, y):
        return f"x={x:0.3g}  y={y:0.3g}"

    def gc_plot(self, yparam='resp', sub_info=''):
        """
        Plot data with the legend sorted by analysis_datetime.
        While `_zoom_run_time` is set, render only the rows belonging to that
        GC run; `self.run` keeps the full chunk so the rest of the program
        (smoothing scope aside) keeps operating on the full DataFrame.
        """
        if self.run.empty:
            return

        # Store ordered run_times from the full chunk for zoom navigation arrows.
        if 'run_time' in self.run.columns:
            self._all_run_times = list(
                pd.to_datetime(self.run['run_time'], utc=True, errors='coerce')
                .dropna()
                .dt.tz_convert(None)
                .sort_values()
                .unique()
            )
        else:
            self._all_run_times = []

        # Filter to the zoomed run_time, if any. Done as a temporary swap so
        # the existing body uses self.run normally; restored in `finally`.
        _full_run = None
        if self._zoom_run_time is not None and 'run_time' in self.run.columns:
            rt_col = pd.to_datetime(self.run['run_time'], utc=True, errors='coerce')
            zr = pd.Timestamp(self._zoom_run_time)
            if zr.tzinfo is None:
                zr = zr.tz_localize('UTC')
            zoom_mask = rt_col == zr
            if zoom_mask.any():
                _full_run = self.run
                self.run = self.run.loc[zoom_mask].copy()
            else:
                # Zoom target no longer present (e.g. after reload); clear it.
                self._zoom_run_time = None
        try:
            self._gc_plot_impl(yparam, sub_info)
        finally:
            if _full_run is not None:
                self.run = _full_run
                # _x_num was set from the zoomed subset inside _gc_plot_impl;
                # recompute against the full DataFrame so that _on_box_select's
                # mask broadcasts (off-screen rows naturally fall outside the
                # zoomed axes' x-bounds, so box selection still scopes correctly).
                x_dt_full = (
                    pd.to_datetime(self.run['analysis_datetime'], utc=True, errors='coerce')
                    .dt.tz_localize(None)
                )
                self._x_num = mdates.date2num(x_dt_full.to_numpy())

    def _gc_plot_impl(self, yparam='resp', sub_info=''):
        current_curve_date = ''
        if yparam == 'resp':
            yvar = self.instrument.response_type
            tlabel = 'Response'
            if self.instrument.inst_id == 'm4':
                units = f'({yvar} per psi)'
            else:
                units = f'({yvar})'
        elif yparam == 'ratio':
            yvar = 'normalized_resp'
            tlabel = 'Ratio (Normalized Response)'
            units = ''
        elif yparam == 'mole_fraction':
            yvar = 'mole_fraction'
            tlabel = 'Mole Fraction'
            units = '(ppb)' if self.current_pnum == 5 else '(ppt)'  # ppb for N2O, ppt for others
    
            # potentially compute missing mole_fraction values for fe3
            if (self.instrument.inst_id == 'fe3') or (self.instrument.inst_id == 'bld1'):
                current_curve_date = self.run['cal_date'].iat[0]
                # if mole_fraction is missing, compute it for fe3, except for port 9 (Push Port)
                mf_mask = self.run['normalized_resp'].gt(0.1) & self.run['mole_fraction'].isna() & self.run['port'].ne(9)
                if mf_mask.any():
                    self.run.loc[mf_mask, 'mole_fraction'] = self.instrument.calc_mole_fraction(self.run.loc[mf_mask])
                    sub_info = f"Mole Fraction computed"
                    self.madechanges = True
                    self._style_gc_buttons()
                if current_curve_date == None:
                    sub_info = "No calibration curve available"
        else:
            print(f"Unknown yparam: {yparam}")
            return

        self._current_yparam = yparam
        self._current_yvar = yvar
        
        x_dt = (
            pd.to_datetime(self.run['analysis_datetime'], utc=True, errors='coerce')
            .dt.tz_localize(None)   # drop timezone cleanly
        )
        self._x_num = mdates.date2num(x_dt.to_numpy())

        # Calculate mean and std for each port
        flags = (
            self.run['rejected'] if 'rejected' in self.run.columns
            else pd.Series(0, index=self.run.index)
        )
        flags = flags.fillna(0).astype(int).astype(bool)        
        good = self.run.loc[~flags, ['port_idx', yvar]]
        stats_map = (
            good.groupby("port_idx")[yvar]
            .agg(["mean", "std", "count"])
            .to_dict("index")  # -> {port: {"mean": ..., "std": ...}}
        )

        legend_handles = []
        ports_in_run = sorted(self.run['port_idx'].dropna().unique())

        for port in ports_in_run:
            color = self.run.loc[self.run['port_idx'] == port, 'port_color'].iloc[0]
            marker = self.run.loc[self.run['port_idx'] == port, 'port_marker'].iloc[0]
            base_label = self.run.loc[self.run['port_idx'] == port, 'port_label'].iloc[0]
            if base_label == 'Push port':
                continue
            subset_port = self.run.loc[self.run['port_idx'] == port]
            skip_stats = False
            if yparam in ('ratio', 'mole_fraction') and 'run_type_num' in subset_port.columns:
                try:
                    skip_stats = (subset_port['run_type_num'].astype(int) == 5).any()
                except Exception:
                    skip_stats = False

            stats = stats_map.get(port)
            if stats is not None and not skip_stats:
                if yparam == 'resp':
                    label = f"{base_label}"
                elif yparam == 'ratio':
                    label = f"{base_label}\n{stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']})"
                else:
                    label = f"{base_label}\n{stats['mean']:.2f} ± {stats['std']:.2f} ({stats['count']})"
            else:
                label = base_label

            legend_handles.append(Line2D(
                [], [],
                color=color,
                marker=marker,
                linestyle='None',
                markersize=8,
                label=label
            ))
            
        # Sort legend handles alphabetically by their label
        legend_handles = sorted(legend_handles, key=lambda h: h.get_label())

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        hide_flagged = self.hide_flagged_cb.isChecked()
        rejected = self.run['rejected'].fillna(0).astype(int)
        plot_df = self.run.loc[rejected == 0] if hide_flagged else self.run
        for port, subset in plot_df.groupby('port_idx'):
            marker = subset['port_marker'].iloc[0]
            color = subset['port_color']
            scatter = ax.scatter(
                subset['analysis_datetime'],
                subset[yvar],
                marker=marker,
                c=color,
                s=self.instrument.BASE_MARKER_SIZE,
                edgecolors='none',
                zorder=1,
                picker=True,
                pickradius=7
            )
            # Map pick indices back to the DataFrame rows for this scatter
            scatter._df_index = subset.index.to_numpy()
        
            scatter._meta = {
                "site": subset["site"].astype(str).tolist() if "site" in subset else [""] * len(subset),

                # format analysis_time to "YYYY-MM-DD HH:MM:SS"
                "analysis_time": (
                    pd.to_datetime(subset["analysis_datetime"], errors="coerce")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .tolist()
                    if "analysis_datetime" in subset
                    else [""] * len(subset)
                ),
                "sample_datetime": (
                    pd.to_datetime(subset["sample_datetime"], errors="coerce")
                    .dt.strftime("%Y-%m-%d %H:%M:%S")
                    .tolist()
                    if "sample_datetime" in subset
                    else [""] * len(subset)
                ),
                "sample_id": subset["sample_id"].astype(str).tolist() if "sample_id" in subset else [""] * len(subset),
                "pair_id": subset["pair_id_num"].astype(str).tolist() if "pair_id_num" in subset else [""] * len(subset),
                "run_type_num": subset["run_type_num"].tolist() if "run_type_num" in subset else [None] * len(subset),
                "port_info": subset["port_info"].astype(str).tolist() if "port_info" in subset else [""] * len(subset),
                "tank_serial": subset["tank_serial_num"].astype(str).tolist() if "tank_serial_num" in subset else [""] * len(subset),
                "net_pressure": (
                    subset["net_pressure"].map(lambda x: f"{x:.3f}" if pd.notnull(x) else "").tolist()
                    if "net_pressure" in subset
                    else [""] * len(subset)
                ),
                "mole_fraction": subset["mole_fraction"].round(3).astype(str).tolist() if "mole_fraction" in subset else [""] * len(subset),
                "response": subset[f"{self.instrument.response_type}"].round(5).astype(str).tolist() if f"{self.instrument.response_type}" in subset else [""] * len(subset),
                "ratio": subset["normalized_resp"].round(5).astype(str).tolist() if "normalized_resp" in subset else [""] * len(subset),
                "status_comments": subset["status_comments"].astype(str).tolist() if "status_comments" in subset else [""] * len(subset),
                "sample_loop_temp": (
                    subset["sample_loop_temp"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "").tolist()
                    if "sample_loop_temp" in subset
                    else [""] * len(subset)
                ),
                "sample_loop_pressure": (
                    subset["sample_loop_pressure"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "").tolist()
                    if "sample_loop_pressure" in subset
                    else [""] * len(subset)
                ),
                "sample_loop_flow": (
                    subset["sample_loop_flow"].map(lambda x: f"{x:.1f}" if pd.notnull(x) else "").tolist()
                    if "sample_loop_flow" in subset
                    else [""] * len(subset)
                ),

            }
            self._scatter_main.append(scatter)
        
        # overlay: show rejected points — auto-only with circle+X, manual with open circle
        flags = rejected != 0
        auto_rej = flags & self.run.get("auto_rejected", pd.Series(False, index=self.run.index)).fillna(False)
        manual_rej = flags & ~auto_rej
        marker_size = self.instrument.BASE_MARKER_SIZE * 1.1
        if not hide_flagged:
            for mask, draw_x in ((manual_rej, False), (auto_rej, True)):
                grp = self.run.loc[mask]
                if grp.empty:
                    continue
                ax.scatter(
                    grp['analysis_datetime'],
                    grp[yvar],
                    marker='o',
                    facecolors='whitesmoke',
                    edgecolors=grp['port_color'],
                    linewidths=1.5,
                    s=marker_size,
                    zorder=4,
                    picker=False,
                )
                if draw_x:
                    x_ms = float(np.sqrt(marker_size * 0.6))
                    for clr in grp['port_color'].unique():
                        cg = grp[grp['port_color'] == clr]
                        ax.plot(
                            cg['analysis_datetime'],
                            cg[yvar],
                            linestyle='none',
                            marker='x',
                            color=clr,
                            markeredgewidth=1.2,
                            markersize=x_ms,
                            zorder=5,
                        )

        # overlay: informational-tag marker (steelblue +)
        if self.show_info_cb.isChecked() and 'has_info_tag' in self.run.columns:
            info_mask = self.run['has_info_tag'].fillna(False).astype(bool)
            info_grp = self.run.loc[info_mask]
            if not info_grp.empty:
                ax.scatter(
                    info_grp['analysis_datetime'],
                    info_grp[yvar],
                    marker='D',
                    facecolors='none',
                    edgecolors='mediumpurple',
                    linewidths=1.2,
                    s=marker_size * 0.7,
                    zorder=3,
                    picker=False,
                )

        if yparam == 'resp':
            smooth_df = self.run.dropna(subset=['analysis_datetime'])
            ax.plot(smooth_df['analysis_datetime'], smooth_df['smoothed'], color='black', linewidth=0.5, label='Lowess-Smooth')
            
        if self._zoom_run_time is not None:
            zr_str = pd.Timestamp(self._zoom_run_time).strftime('%Y-%m-%d %H:%M:%S')
            top_line = f"{zr_str}  (zoom in {self.current_run_time})"
        elif self._ie3_exact_run_time is not None:
            rt_str = pd.Timestamp(self._ie3_exact_run_time).strftime('%Y-%m-%d %H:%M:%S')
            top_line = f"{rt_str}  (full run from {self.current_run_time})"
        else:
            top_line = f"{self.current_run_time}"
        main_title = "\n".join([
            top_line,
            f"{tlabel}: {self.instrument.analytes_inv[self.current_pnum]} ({self.current_pnum})",
        ])
        ax.set_title(main_title, pad=16)
        if sub_info:
            ax.text(
                0.5, .965, sub_info,
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, color='white', clip_on=False,
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#8B0000',   # dark red
                    edgecolor='none',      # or '#8B0000' if you want a border
                    alpha=0.9
                )
            )
        ax.set_xlabel("Analysis Datetime")
        ax.xaxis.set_tick_params(rotation=30)
        ax.set_ylabel(tlabel + " " + units)

        # Vertical lines at run_time transitions (visible when the loaded period
        # spans multiple GC runs, e.g. IE3 monthly chunks). When shown, suppress
        # the vertical grid so the two don't compete.
        unique_rts = []
        if 'run_time' in self.run.columns:
            unique_rts = (
                pd.to_datetime(self.run['run_time'], utc=True, errors='coerce')
                .dropna()
                .dt.tz_convert(None)
                .sort_values()
                .unique()
            )

        chunk_start = None
        carry_in_rt = None
        if (self.instrument.inst_id in ('ie3', 'cats')
                and self._ie3_exact_run_time is None
                and self.current_run_time):
            try:
                chunk_start, _chunk_end = _ie3_parse_chunk_label(self.current_run_time)
            except Exception:
                chunk_start = None
            if chunk_start is not None and len(unique_rts) > 0:
                prior_rts = [pd.Timestamp(rt) for rt in unique_rts if pd.Timestamp(rt) < chunk_start]
                if prior_rts:
                    carry_in_rt = max(prior_rts)

        self._run_time_lines = []
        if len(unique_rts) > 1 or carry_in_rt is not None:
            ax.yaxis.grid(True, linewidth=0.5, linestyle='--', alpha=0.8)
            ax.xaxis.grid(False)
            if carry_in_rt is not None:
                line = ax.axvline(
                    x=chunk_start,
                    color='lightblue',
                    linewidth=1.2,
                    linestyle=(0, (8, 4)),
                    alpha=0.95,
                    zorder=0,
                    picker=True,
                )
                line.set_pickradius(7)
                line._rt = carry_in_rt
                line._ie3_exact_load = True
                self._run_time_lines.append(line)
            for rt in unique_rts:
                if carry_in_rt is not None and pd.Timestamp(rt) == carry_in_rt:
                    continue
                line = ax.axvline(
                    x=rt,
                    color='lightblue',
                    linewidth=0.8,
                    linestyle='-',
                    alpha=0.85,
                    zorder=0,
                    picker=True,
                )
                line.set_pickradius(5)
                line._rt = pd.Timestamp(rt)  # tz-naive (already converted above)
                self._run_time_lines.append(line)
        else:
            ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.8)

        # add cal curve date selector
        if ((self.instrument.inst_id == 'fe3') or (self.instrument.inst_id == 'bld1')) and (yparam == 'mole_fraction'):
            self.calcurve_label.setVisible(True)
            self.calcurve_combo.setVisible(True)
            self.populate_calcurve_combo(current_curve_date)

        # Cal curve date and age (if available)
        if current_curve_date:
            cal_delta_time = self.run['analysis_datetime'].min() - pd.to_datetime(current_curve_date, utc=True)
            l = current_curve_date.strftime('\nCal Date:\n%Y-%m-%d %H:%M\n') + f'{cal_delta_time.days} days ago'
            legend_handles.append(Line2D([], [], linestyle='None', label=l))

        # --- Detrend summary box for resp/ratio plots ---
        if yparam in ('ratio', 'resp') and self.instrument.inst_id not in ('ie3', 'cats'):
            try:
                # Order from least to most smoothing
                method_order = [1, 3, 2, 4, 6, 5]
                labels = {
                    1: "Pt-to-pt",
                    3: "2-pt avg",
                    2: "Lowess 5",
                    4: "3-pt box",
                    6: "5-pt box",
                    5: "Lowess 10",
                }
                label_width = max(len(v) for v in labels.values())
                stats_df, best_method, _ = self.instrument.norm.detrend_stats_for_run(
                    self.run,
                    methods=method_order,
                    margin_frac=0.2,
                    drop_outlier=True,
                    verbose=False,
                )
                if not stats_df.empty:
                    stats_df = stats_df.set_index('detrend_method_num')
                    # Pick header based on instrument (e.g., BLD1 uses SD wording)
                    inst_num = None
                    if 'inst_num' in self.run.columns and not self.run['inst_num'].empty:
                        try:
                            inst_num = int(self.run['inst_num'].iloc[0])
                        except Exception:
                            inst_num = None
                    if inst_num is None:
                        try:
                            inst_num = int(getattr(self.instrument, "inst_num", None))
                        except Exception:
                            inst_num = None

                    lines = ['SD of reference'] if inst_num == 220 else ['Sample Pair RMS']
                    for m in method_order:
                        if m not in stats_df.index:
                            continue
                        rms = stats_df.at[m, 'rep_resp_rms']
                        if pd.isna(rms):
                            continue
                        label = labels.get(m, f"Method {m}")
                        marker = "● " if best_method == m else "  "
                        lines.append(f"{marker}{label:>{label_width}}: {rms:<8.4f}")

                    if lines:
                        legend_handles.append(Line2D([], [], linestyle='None', label='\u2009'))
                        legend_handles.append(
                            Line2D(
                                [],
                                [],
                                linestyle='None',
                                label="\n".join(lines)
                            )
                        )
            except Exception as e:
                print(f"Warning: unable to compute detrend summary ({e})")

        # --- Always append Save/Revert legend "buttons" ---
        spacer_handle     = Line2D([], [], linestyle='None', label='\u2009')
        save2db_handle    = Line2D([], [], linestyle='None', label='Save current gas (s)')
        save2dball_handle = Line2D([], [], linestyle='None', label='Save all gases')
        revert_handle     = Line2D([], [], linestyle='None', label='Revert changes')

        legend_handles.extend([
            spacer_handle,
            save2db_handle,
            spacer_handle,
            revert_handle,
            spacer_handle,
            save2dball_handle
        ])

        leg = ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.6),
            fontsize=9,
            frameon=False,
            handlelength=0.5,
            handletextpad=0.4,
            borderaxespad=0.3,
            labelspacing=0.2,
        )
        
        # Style Save/Revert entries like buttons
        self._save2db_text = None
        self._save2dball_text = None
        self._revert_text  = None
        self._spacer2_text = None

        for txt in leg.get_texts():
            t = txt.get_text().strip()
            if "Sample Pair RMS" in t or "SD of reference" in t:
                txt.set_bbox(dict(
                    boxstyle='square,pad=0.3',
                    facecolor='white',
                    edgecolor='black',
                    linewidth=0.6
                ))
            if t == 'Save current gas (s)':
                self._save2db_text = txt
                txt.set_picker(True)
                txt.set_color('white')
                txt.set_bbox(dict(
                    boxstyle='round,pad=0.4',
                    facecolor=('#2e7d32' if self.madechanges else '#9e9e9e'),
                    edgecolor='none', alpha=0.95
                ))
            elif t == 'Save all gases':
                txt.set_color('white')
                txt.set_bbox(dict(
                    boxstyle='round,pad=0.4',
                    facecolor = '#9e9e9e',
                    edgecolor='none', alpha=0.95
                ))
                if self.smoothing_changed:
                    continue
                self._save2dball_text = txt
                txt.set_picker(True)
                txt.set_bbox(dict(
                    boxstyle='round,pad=0.4',
                    #facecolor=('#2e7d32' if self.madechanges else '#9e9e9e'),
                    facecolor=('#43a047' if self.madechanges else '#9e9e9e'),
                    edgecolor='none', alpha=0.95
                ))
            elif t == 'Revert changes':
                self._revert_text = txt
                txt.set_picker(True)
                txt.set_color('white')
                txt.set_bbox(dict(
                    boxstyle='round,pad=0.4',
                    facecolor=('#c62828' if self.madechanges else '#9e9e9e'),
                    edgecolor='none', alpha=0.95
                ))
            elif t == '\u2009':  # spacer
                self._spacer2_text = txt
                txt.set_fontsize(10)
                txt.set_color((0, 0, 0, 0))  # fully transparent
            elif "sample pair rms" in t.lower():
                txt.set_fontfamily('monospace')
                txt.set_bbox(dict(
                    boxstyle='round,pad=0.35',
                    facecolor='white',
                    edgecolor='gray',
                    linewidth=0.8,
                    alpha=0.9,
                ))

        # ---- Run-time navigation arrows ----
        arrow_kw = dict(
            transform=ax.transAxes, va='top', fontsize=12,
            fontweight='bold', color='steelblue',
            picker=True,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='steelblue', alpha=0.85),
            zorder=20,
        )
        if self._zoom_run_time is not None:
            # IE3 zoom mode: step between run_time groups within loaded data
            all_rts = getattr(self, '_all_run_times', [])
            zr = pd.Timestamp(self._zoom_run_time)
            try:
                cur_idx = next(i for i, rt in enumerate(all_rts) if pd.Timestamp(rt) == zr)
            except StopIteration:
                cur_idx = None
            if cur_idx is not None:
                if cur_idx > 0:
                    prev_art = ax.text(0.015, 0.99, '◀', ha='left', **arrow_kw)
                    prev_art._rt_nav = all_rts[cur_idx - 1]
                if cur_idx < len(all_rts) - 1:
                    next_art = ax.text(0.985, 0.99, '▶', ha='right', **arrow_kw)
                    next_art._rt_nav = all_rts[cur_idx + 1]
        else:
            # All instruments: step between loaded run_times via the run combobox
            cur_idx = self.run_cb.currentIndex()
            count = self.run_cb.count()
            if count > 1:
                if cur_idx > 0:
                    prev_art = ax.text(0.015, 0.99, '◀', ha='left', **arrow_kw)
                    prev_art._run_nav = 'prev'
                if cur_idx < count - 1:
                    next_art = ax.text(0.985, 0.99, '▶', ha='right', **arrow_kw)
                    next_art._run_nav = 'next'

        ax.format_coord = self._fmt_gc_plot

        if not hasattr(self, "_legend_pick_cid") or self._legend_pick_cid is None:
            self._legend_pick_cid = self.canvas.mpl_connect(
                "pick_event", self._on_legend_pick
            )

        # ---- Y-Axis Scaling ----
        if self.lock_y_axis_cb.isChecked():
            # use the stored y-axis limits
            if self.y_axis_limits is None:
                # If no limits are set, use the current y-limits
                self.y_axis_limits = ax.get_ylim()
            else:
                ax.set_ylim(self.y_axis_limits)

        else:
            # Only adjust y-limits if not locked
            autoscale_mode = self._get_autoscale_mode()
            if autoscale_mode in {"samples", "standard"}:
                exclude = self.instrument.EXCLUDE
                if self.instrument.inst_id == 'm4':
                    # m4 uses run_type_num to exclude blanks/calibrations
                    exclude_variable = 'run_type_num'
                    standard = self.instrument.STANDARD_RUN_TYPE
                else:
                    # fe3, bld1 use port to exclude push ports
                    exclude_variable = 'port'
                    standard = self.instrument.STANDARD_PORT_NUM

                unflagged = self.run['rejected'].fillna(0).astype(int) == 0
                if autoscale_mode == "standard":
                    std_ports = getattr(self.instrument, 'AUTOSCALE_STANDARD_PORTS', None)
                    if std_ports:
                        scale_df = self.run.loc[self.run[exclude_variable].isin(std_ports) & unflagged, yvar].dropna()
                    else:
                        scale_df = self.run.loc[(self.run[exclude_variable] == standard) & unflagged, yvar].dropna()
                    if scale_df.empty:
                        # Fall back to samples mode if no valid unflagged standard data
                        scale_df = self.run.loc[~self.run[exclude_variable].isin(exclude) & unflagged, yvar].dropna()
                else:
                    scale_df = self.run.loc[~self.run[exclude_variable].isin(exclude) & unflagged, yvar].dropna()
                if not scale_df.empty:
                    ymin, ymax = scale_df.min(), scale_df.max()
                    if ymin == ymax:
                        ymin -= 0.05 * abs(ymin) if ymin != 0 else 0.05
                        ymax += 0.05 * abs(ymax) if ymax != 0 else 0.05
                    if pd.notna(ymin) and pd.notna(ymax) and np.isfinite(ymin) and np.isfinite(ymax):
                        ax.set_ylim(ymin * 0.98, ymax * 1.02)
            else:
                try:
                    ax.set_ylim(
                        self.run[yvar].min() * 0.95,
                        self.run[yvar].max() * 1.05
                    )
                except ValueError:
                    pass  # In case of empty data, do not set limits
        
        # ---- Clamp x-axis to valid data range ----
        # Any plot element containing NaT datetimes would otherwise push autoscale to 1970.
        valid_dt = pd.to_datetime(self.run['analysis_datetime'], errors='coerce').dropna()
        if not valid_dt.empty:
            x_min = mdates.date2num(valid_dt.min().to_pydatetime())
            x_max = mdates.date2num(valid_dt.max().to_pydatetime())
            span = x_max - x_min
            margin = span * 0.03 if span > 0 else 1 / 48  # 3% each side; 30 min for single point
            ax.set_xlim(x_min - margin, x_max + margin)

        # ---- Restore view if requested by a prior click ----
        if getattr(self, "_pending_xlim", None) is not None:
            ax.set_xlim(self._pending_xlim)
        if getattr(self, "_pending_ylim", None) is not None:
            ax.set_ylim(self._pending_ylim)
        self._pending_xlim = None
        self._pending_ylim = None

        # Make sure the rectangle selector attaches to the current axes
        self._reattach_rect_selector()

        self._adjust_layout_for_legend(leg)
        self._restore_highlight()

        if self._pick_cid is None:
            self._pick_cid = self.canvas.mpl_connect('pick_event', self._on_pick_point)

        # Connect tooltip click handler
        if not hasattr(self, "_click_tooltip_cid"):
            self._click_tooltip_cid = self.canvas.mpl_connect("button_press_event", self._on_click_tooltip)
                        
    def _reattach_rect_selector(self):
        """Ensure RectangleSelector follows the current axes after a redraw."""
        if self._rect_selector is not None:
            self._rect_selector.set_active(False)
            self._rect_selector.disconnect_events()
            self._rect_selector = None

        if self.figure.axes and self.tagging_enabled:
            ax = self.figure.axes[0]
            self._rect_selector = RectangleSelector(
                ax=ax,
                onselect=self._on_box_select,
                useblit=True,
                button=[1],
                minspanx=5, minspany=5,
                spancoords="pixels",
                drag_from_anywhere=True,
                ignore_event_outside=False,
            )
            self._rect_selector.set_active(True)

        # Rebuild multi-tag region selector on the new axes if it was active.
        multi_select_active = (
            self._multi_tag_rect_selector is not None
            or (self._multi_tag_panel is not None
                and self._multi_tag_panel.isVisible()
                and self._multi_tag_panel._select_btn.isChecked())
        )
        self._set_multi_tag_select(multi_select_active)
                    
    def _style_gc_buttons(self):
        """
        Placeholder for consistency with calibration_plot.
        In gc_plot, buttons are added/removed dynamically,
        so this just redraws the canvas if needed.
        """
        if getattr(self, "canvas", None):
            self.canvas.draw_idle()

    def _adjust_layout_for_legend(self, leg):
        """
        Ensure the legend is visible without squeezing the plot. This does a
        two-pass draw to measure the legend and sets the right margin based
        on its actual width.
        """
        if leg is None:
            return

        # Exclude legend from layout
        try:
            leg.set_in_layout(False)
        except Exception:
            pass

        # First draw to obtain a renderer and legend size
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        try:
            leg_bb = leg.get_window_extent(renderer=renderer).transformed(self.figure.transFigure.inverted())
            legend_width = leg_bb.width
            pad = 0.03
            right = max(0.65, 1.0 - legend_width - pad)
            #print(f"Adjusting right margin to {right:0.3f} to fit legend width {legend_width:0.3f}")
        except Exception:
            right = 0.8

        # Apply margins and redraw
        self.figure.subplots_adjust(right=right, left=0.08, bottom=0.12, top=0.88)
        self.canvas.draw_idle()

    def clear_plot(self, message="No data available"):
        """Clear the main GC plot and legend, resetting interactive elements."""
        # Clear figure contents
        if hasattr(self, "figure"):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_title(message, pad=12)
            ax.set_xlabel("Analysis Datetime")
            ax.set_ylabel("")
            ax.grid(False)
            ax.set_facecolor("#f7f7f7")

            # Clear legend area (force blank layout)
            ax.legend([], [], frameon=False)
            self.figure.tight_layout(rect=[0, 0, 0.9, 1])

            # Redraw the canvas
            if hasattr(self, "canvas"):
                self.canvas.draw_idle()

        # Reset internal state for legend and buttons
        self._scatter_main = []
        self._save2db_text = None
        self._save2dball_text = None
        self._revert_text = None
        self._spacer2_text = None

        # Disconnect pick and tooltip callbacks (if active)
        for attr in ("_legend_pick_cid", "_pick_cid", "_click_tooltip_cid"):
            cid = getattr(self, attr, None)
            if cid is not None and hasattr(self, "canvas"):
                self.canvas.mpl_disconnect(cid)
            setattr(self, attr, None)

        # Hide calibration curve selector if visible
        if hasattr(self, "calcurve_label"):
            self.calcurve_label.setVisible(False)
        if hasattr(self, "calcurve_combo"):
            self.calcurve_combo.setVisible(False)

    def populate_calcurve_combo(self, current_curve):
        self.calcurve_combo.blockSignals(True)
        self.calcurve_combo.clear()

        # parse the current run time. This handles "(Cal)" in the string
        sel_time = pd.to_datetime(self.current_run_time.split(' (')[0])
        self.calcurves = self.instrument.load_calcurves(
            self.current_pnum,
            self.current_channel,
            sel_time
        )

        if self.calcurves.empty:
            self.calcurve_combo.addItem("No curves found", userData=None)
            self.calcurve_combo.setCurrentIndex(0)
            self.calcurve_combo.blockSignals(False)
            return
        
        # Filter to a window: 60 days before to 7 days after selected run_time
        window = self.calcurves.loc[
            self.calcurves['run_date'].between(
                sel_time - pd.Timedelta(days=60),
                sel_time + pd.Timedelta(days=7)
            )
        ].sort_values('run_date')

        # --- Add "Cal Date" label as first item ---
        model = QStandardItemModel()
        label_item = QStandardItem("Cal Date")
        label_item.setFlags(label_item.flags() & ~Qt.ItemIsEnabled)  # Disable item
        model.appendRow(label_item)

        for _, row in window.iterrows():
            label = row['run_date'].strftime('%Y-%m-%d %H:%M:%S')
            self.calcurve_combo.addItem(label, userData=row)

        # Fix: only format if current_curve is not NaT
        if pd.notna(current_curve):
            #print(f'current curve = {current_curve}')
            current_str = pd.to_datetime(current_curve).strftime('%Y-%m-%d %H:%M:%S')
            idx = self.calcurve_combo.findText(current_str)
            if idx != -1:
                self.calcurve_combo.setCurrentIndex(idx)

        self.calcurve_combo.blockSignals(False)    
    
    def _on_pick_point(self, event):
        try:
            self._on_pick_point_inner(event)
        except Exception as exc:
            print(f"_on_pick_point error: {exc}")

    def _on_pick_point_inner(self, event):
        tb = getattr(self.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", None):
            return

        # Run-time navigation arrow clicked (IE3 zoom mode).
        if not self.tagging_enabled and hasattr(event.artist, '_rt_nav'):
            self._toggle_run_time_zoom(event.artist._rt_nav)
            return

        # Run-time navigation arrow clicked (all instruments, non-zoom mode).
        if not self.tagging_enabled and hasattr(event.artist, '_run_nav'):
            if event.artist._run_nav == 'prev':
                self.on_prev_run()
            else:
                self.on_next_run()
            return

        # run_time transition line clicked: zoom toggle (only when tagging is off,
        # so flag-tagging picks always win when tagging mode is active).
        if (not self.tagging_enabled
                and isinstance(event.artist, Line2D)
                and hasattr(event.artist, '_rt')):
            if getattr(event.artist, '_ie3_exact_load', False):
                self._load_ie3_exact_run_time(event.artist._rt)
                return
            self._toggle_run_time_zoom(event.artist._rt)
            return

        if event.artist not in self._scatter_main:
            return

        scatter = event.artist
        inds = np.asarray(event.ind, dtype=int)
        if inds.size == 0:
            return

        mx, my = event.mouseevent.xdata, event.mouseevent.ydata
        if mx is None or my is None:
            i = inds[0]
        else:
            offsets = np.asarray(scatter.get_offsets())
            if offsets.size == 0:
                return
            dx = offsets[inds, 0] - mx
            dy = offsets[inds, 1] - my
            ok = np.isfinite(dx) & np.isfinite(dy)
            if not ok.any():
                return
            i = inds[ok][np.argmin(dx[ok] ** 2 + dy[ok] ** 2)]

        df_index = getattr(scatter, "_df_index", None)
        row_idx = df_index[i] if df_index is not None else self.run.index[i]

        # Update multi-tag panel for any scatter click (tagging mode not required).
        if (self._multi_tag_panel is not None and self._multi_tag_panel.isVisible()):
            mf_nums = self._mole_fraction_nums_for_indices([row_idx])
            tag_counts = self._fetch_tag_counts_for_mf_nums(mf_nums)
            self._multi_tag_panel.update_for_point(
                [row_idx], mf_nums, tag_counts, total=len(mf_nums) or 1
            )
            self._highlight_point(scatter, i, row_idx)

        if self.tagging_enabled:
            self._toggle_flags([row_idx])

    def _toggle_run_time_zoom(self, rt):
        """Click-to-zoom on a run_time transition line. Clicking the active line
        again clears the zoom (Esc has the same effect)."""
        rt = pd.Timestamp(rt)
        if rt.tzinfo is not None:
            rt = rt.tz_convert(None) if rt.tz is not None else rt.tz_localize(None)
        if (self._zoom_run_time is not None
                and pd.Timestamp(self._zoom_run_time) == rt):
            self._zoom_run_time = None
            sub = 'unzoomed'
        else:
            self._zoom_run_time = rt
            sub = f"zoomed to run_time {rt:%Y-%m-%d %H:%M:%S}"
        self.update_smoothing_combobox()
        self.gc_plot(self._current_yparam or 'resp', sub_info=sub)

    def _load_ie3_exact_run_time(self, rt):
        """Load a complete IE3 run_time group that crosses into the visible chunk."""
        rt = pd.Timestamp(rt)
        if rt.tzinfo is not None:
            rt = rt.tz_convert(None) if rt.tz is not None else rt.tz_localize(None)
        self._ie3_exact_run_time = rt
        self._zoom_run_time = None
        self.load_selected_run()
        self.gc_plot(
            self._current_yparam or 'resp',
            sub_info=f"loaded full run_time {rt:%Y-%m-%d %H:%M:%S}",
        )

    def _on_click_tooltip(self, event):
        """Show tooltip on left-click when tagging and navigation are off."""
        # Left mouse only
        if event.button != 1:
            return

        # Skip if tagging or navigation tools are active
        tb = getattr(self.canvas, "toolbar", None)
        if (tb is not None and getattr(tb, "mode", None)) or self.tagging_enabled:
            return

        # Find which artist was clicked
        for artist in self.figure.axes[0].collections:
            cont, ind = artist.contains(event)
            if not cont:
                continue

            nearest_idx = ind["ind"][0]
            meta = getattr(artist, "_meta", {})

            site = meta.get("site", [None])[nearest_idx]
            analysis_time = meta.get("analysis_time", [None])[nearest_idx]
            sample_time = meta.get("sample_datetime", [None])[nearest_idx]
            resp = meta.get("response", [None])[nearest_idx]
            ratio = meta.get("ratio", [None])[nearest_idx]
            mf = meta.get("mole_fraction", [None])[nearest_idx]
            sample_id = meta.get("sample_id", [None])[nearest_idx]
            pair_id = meta.get("pair_id", [None])[nearest_idx]
            run_type_num = meta.get("run_type_num", [None])[nearest_idx]
            net_pressure = meta.get("net_pressure", [None])[nearest_idx]
            port_info = meta.get("port_info", [''])[nearest_idx]
            tank_serial = meta.get("tank_serial", [''])[nearest_idx]
            comments = meta.get("status_comments", [''])[nearest_idx]
            loop_temp = meta.get("sample_loop_temp", [''])[nearest_idx]
            loop_pressure = meta.get("sample_loop_pressure", [''])[nearest_idx]
            loop_flow = meta.get("sample_loop_flow", [''])[nearest_idx]

            parts = []

            # Site — show if not blank/None
            if site not in (None, "", "nan", "None"):
                parts.append(f"<b>Site:</b> {site}")
            
            if resp is not None:
                parts.append(f"<b>Response:</b> {resp}")
                
            if ratio is not None:
                parts.append(f"<b>Ratio:</b> {ratio}")

            if mf is not None:
                parts.append(f"<b>Mole Fraction:</b> {mf}")

            # Sample ID — show only if not "0" or blank
            if isinstance(sample_id, str):
                sid = sample_id.strip()
                if sid and sid not in {"0", "000", "None", "nan"}:
                    parts.append(f"<b>Sample ID:</b> {sid}")

            # Pair ID — show only if not "0" or blank
            if isinstance(sample_id, str):
                pid = pair_id.strip()
                if pid and pid not in {"0", "000", "None", "nan"}:
                    parts.append(f"<b>Pair ID:</b> {pid}")

            # Flask Type — not applicable for in-situ instruments
            if run_type_num is not None and self.instrument.inst_num not in range(236, 245):
                flask_type = "PFP" if int(run_type_num) == 5 else "Flask"
                parts.append(f"<b>Flask Type:</b> {flask_type}")

            if not _is_blank(port_info):
                parts.append(f"<b>Port Info:</b> {port_info}")

            if not _is_blank(tank_serial):
                parts.append(f"<b>Tank Serial Num:</b> {tank_serial}")

            # Pair ID — show only if not "0" or blank
            if isinstance(net_pressure, str):
                presss = net_pressure.strip()
                if presss and presss not in {"0", "000", "None", "nan"}:
                    parts.append(f"<b>Net Pressure:</b> {presss} psi")

            # Sample time — show only if not blank
            if not _is_blank(sample_time):
                parts.append(f"<b>Sample time:</b> {sample_time}")

            # Analysis time — always shown
            parts.append(f"<b>Analysis time:</b> {analysis_time}")
            
            if not _is_blank(comments):
                parts.append(f"<b>Comments:</b> {comments}")

            if not _is_blank(loop_temp):
                parts.append(f"<b>Loop Temp:</b> {loop_temp} °C")
            if not _is_blank(loop_pressure):
                parts.append(f"<b>Loop Pressure:</b> {loop_pressure} mbar")
            if not _is_blank(loop_flow):
                parts.append(f"<b>Loop Flow:</b> {loop_flow} cc/min")

            # Applied tags — look up via DB
            try:
                df_index = getattr(artist, "_df_index", None)
                row_idx = df_index[nearest_idx] if df_index is not None else self.run.index[nearest_idx]
                mf_nums = self._mole_fraction_nums_for_indices([row_idx])
                applied = self._fetch_tag_nums_for_mf_nums(mf_nums)
                tag_names = getattr(self, "_all_tag_names", {})
                for tnum in sorted(applied):
                    if tnum in _INFO_TAG_DESCRIPTIONS:
                        short = " ".join(_INFO_TAG_DESCRIPTIONS[tnum].split()[:5])
                        parts.append(f"<b>Info Tag:</b> {short}")
                    else:
                        desc = tag_names.get(tnum, f"tag {tnum}")
                        short = " ".join(desc.split()[:5])
                        parts.append(f"<b>Tag:</b> {short}")
            except Exception:
                pass

            # Combine for tooltip
            text = "<br>".join(parts)
            QToolTip.showText(QCursor.pos(), text)
            break
        else:
            # Click landed on empty space — deselect multi-tag selection if open.
            if (self._multi_tag_panel is not None
                    and self._multi_tag_panel.isVisible()):
                self._clear_highlight()
                self._multi_tag_panel.clear_selection()

    def _on_box_select(self, eclick, erelease):
        if not self.tagging_enabled:
            return

        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        if None in (x0, y0, x1, y1):
            return

        y_all = self.run[self._current_yvar].to_numpy()
        mask = (self._x_num >= x0) & (self._x_num <= x1) & (y_all >= y0) & (y_all <= y1)
        idxs = self.run.index[mask]

        self._toggle_flags(idxs)

    def _highlight_region(self, row_idxs: list, x_nums):
        """Overlay hollow black circles on all points in a region selection."""
        self._clear_highlight()
        if not row_idxs or not self.figure.axes:
            return
        if not self._current_yvar or self._current_yvar not in self.run.columns:
            return
        try:
            ax = self.figure.axes[0]
            x_arr = np.asarray(x_nums, dtype=float)
            y_arr = self.run.loc[row_idxs, self._current_yvar].to_numpy(dtype=float)
            valid = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_plot, y_plot = x_arr[valid], y_arr[valid]
            if len(x_plot) == 0:
                return
            ms = 10.0
            (artist,) = ax.plot(
                x_plot, y_plot, 'o',
                markerfacecolor='none',
                markeredgecolor='black',
                markeredgewidth=1.5,
                markersize=ms,
                zorder=10,
                linestyle='none',
                transform=ax.transData,
            )
            self._highlighted_point = {'artist': artist, 'x': x_plot, 'y': y_plot, 'ms': ms}
            self.canvas.draw_idle()
        except Exception:
            pass

    def _highlight_point(self, scatter, scatter_i: int, row_idx=None):
        """Overlay a hollow black circle on the selected scatter point."""
        self._clear_highlight()
        try:
            offsets = np.asarray(scatter.get_offsets())
            x, y = offsets[scatter_i]
            ax = scatter.axes
            marker_size = scatter.get_sizes()
            ms = float(np.sqrt(marker_size[0]) * 1.4) if len(marker_size) else 10.0
            (artist,) = ax.plot(
                x, y, 'o',
                markerfacecolor='none',
                markeredgecolor='black',
                markeredgewidth=1.5,
                markersize=ms,
                zorder=10,
                transform=ax.transData,
            )
            self._highlighted_point = {'artist': artist, 'row_idx': row_idx, 'x': x, 'y': y, 'ms': ms}
            scatter.figure.canvas.draw_idle()
        except Exception:
            pass

    def _clear_highlight(self):
        """Remove the overlay highlight marker."""
        artist = self._highlighted_point.get('artist')
        if artist is not None:
            try:
                artist.remove()
            except Exception:
                pass
        self._highlighted_point = {}
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _restore_highlight(self):
        """Re-apply the highlight after gc_plot rebuilds scatter objects.
        Uses stored x,y coordinates so it works regardless of which scatter
        (main or flagged) the point now lives in."""
        x = self._highlighted_point.get('x')
        y = self._highlighted_point.get('y')
        ms = self._highlighted_point.get('ms', 10.0)
        if x is None or y is None:
            return
        if self._multi_tag_panel is None or not self._multi_tag_panel.isVisible():
            self._highlighted_point = {}
            return
        row_idx = self._highlighted_point.get('row_idx')
        self._highlighted_point = {}  # drop stale artist reference
        axes = self.figure.axes
        if not axes:
            return
        ax = axes[0]
        try:
            (artist,) = ax.plot(
                x, y, 'o',
                markerfacecolor='none',
                markeredgecolor='black',
                markeredgewidth=1.5,
                markersize=ms,
                zorder=10,
                transform=ax.transData,
            )
            self._highlighted_point = {'artist': artist, 'row_idx': row_idx, 'x': x, 'y': y, 'ms': ms}
            self.canvas.draw_idle()
        except Exception:
            pass

    def _sync_rejected_state(self, idxs):
        """Re-query the DB and update run['rejected'] for the given DataFrame indices."""
        if "rejected" not in self.run.columns:
            self.run["rejected"] = 0
        reject_nums = {t["tag_num"] for t in self.tag_options
                       if int(t.get("reject") or 0) == 1}
        for row_idx in idxs:
            mf_nums = self._mole_fraction_nums_for_indices([row_idx])
            applied = self._fetch_tag_nums_for_mf_nums(mf_nums)
            self.run.at[row_idx, "rejected"] = 1 if (applied & reject_nums) else 0
        self._update_auto_rejected()
        self._update_info_tagged()

    def _update_auto_rejected(self):
        """Mark run['auto_rejected']=True for points rejected only by automatic tags.
        Points with any manual rejection tag (not in AUTO_TAG_NUMS) get False."""
        if self.run.empty or "rejected" not in self.run.columns:
            return
        rejected_mask = self.run["rejected"].fillna(0).astype(int) != 0
        self.run["auto_rejected"] = False
        if not rejected_mask.any():
            return
        rejected_idxs = self.run.index[rejected_mask].tolist()
        mf_nums = self._mole_fraction_nums_for_indices(rejected_idxs)
        if not mf_nums:
            return
        tag_table, tag_key, _, _ = self._tag_table_info()
        placeholders = ",".join(["%s"] * len(mf_nums))
        auto_placeholders = ",".join(["%s"] * len(self.AUTO_TAG_NUMS))
        # Find mf_nums that have at least one non-auto rejection tag.
        rows = self.instrument.db.doquery(
            f"""SELECT DISTINCT t.{tag_key} AS mf_num
                FROM {tag_table} t
                JOIN ccgg.tag_view tv ON tv.tag_num = t.tag_num
                WHERE t.{tag_key} IN ({placeholders})
                  AND t.tag_num NOT IN ({auto_placeholders})
                  AND tv.reject = 1""",
            mf_nums + list(self.AUTO_TAG_NUMS),
        )
        has_manual = {int(r["mf_num"]) for r in (rows or [])}
        # Identify the id_col so we can map mf_num back to run rows.
        _, _, _, id_col = self._tag_table_info()
        if id_col in self.run.columns:
            for idx in rejected_idxs:
                mf = self.run.at[idx, id_col]
                try:
                    self.run.at[idx, "auto_rejected"] = int(mf) not in has_manual
                except Exception:
                    pass
        else:
            # Can't determine tag breakdown — treat all as manual (plain open circle).
            pass

    def _update_info_tagged(self):
        """Set run['has_info_tag']=True for rows that carry any informational tag."""
        if self.run.empty or not _INFO_TAG_NUMS:
            return
        self.run['has_info_tag'] = False
        tag_table, tag_key, mf_table, id_col = self._tag_table_info()
        it_ph = ",".join(["%s"] * len(_INFO_TAG_NUMS))

        if id_col in self.run.columns:
            # Direct path: id_col (ng_mole_fraction_num) is in the DataFrame
            all_mf_nums = [int(v) for v in self.run[id_col].dropna().tolist()]
            if not all_mf_nums:
                return
            mf_ph = ",".join(["%s"] * len(all_mf_nums))
            rows = self.instrument.db.doquery(
                f"SELECT DISTINCT {tag_key} AS mf_num FROM {tag_table} "
                f"WHERE {tag_key} IN ({mf_ph}) AND tag_num IN ({it_ph});",
                all_mf_nums + list(_INFO_TAG_NUMS),
            )
            has_info = {int(r["mf_num"]) for r in (rows or [])}
            for idx in self.run.index:
                try:
                    self.run.at[idx, 'has_info_tag'] = int(self.run.at[idx, id_col]) in has_info
                except Exception:
                    pass

        elif {'analysis_num', 'parameter_num', 'channel'}.issubset(self.run.columns):
            # IE3 path: look up via join through the mf table
            all_analysis_nums = [int(v) for v in self.run['analysis_num'].dropna().unique()]
            if not all_analysis_nums:
                return
            an_ph = ",".join(["%s"] * len(all_analysis_nums))
            rows = self.instrument.db.doquery(
                f"""SELECT DISTINCT mf.analysis_num, mf.parameter_num, mf.channel
                    FROM {mf_table} mf
                    JOIN {tag_table} t ON t.{tag_key} = mf.num
                    WHERE mf.analysis_num IN ({an_ph})
                      AND t.tag_num IN ({it_ph})""",
                all_analysis_nums + list(_INFO_TAG_NUMS),
            )
            has_info_keys = {
                (int(r["analysis_num"]), int(r["parameter_num"]), str(r["channel"]))
                for r in (rows or [])
            }
            for idx in self.run.index:
                try:
                    key = (
                        int(self.run.at[idx, 'analysis_num']),
                        int(self.run.at[idx, 'parameter_num']),
                        str(self.run.at[idx, 'channel']),
                    )
                    self.run.at[idx, 'has_info_tag'] = key in has_info_keys
                except Exception:
                    pass

    def _on_show_info_toggled(self, state):
        if state:
            self._update_info_tagged()
        self.on_plot_type_changed(self.current_plot_type)

    def _toggle_flags(self, idxs):
        """Toggle the selected tag on one or more points: apply if absent, remove if present."""
        if idxs is None or len(idxs) == 0:
            return
        if not self.selected_tag:
            QMessageBox.information(self, "No Tag Selected", "Select a tag before tagging points.")
            return

        if "rejected" not in self.run.columns:
            self.run["rejected"] = 0
        if "_pending_tag_num" not in self.run.columns:
            self.run["_pending_tag_num"] = pd.NA

        tag_num = int(self.selected_tag["tag_num"])
        mf_nums = self._mole_fraction_nums_for_indices(idxs)
        if not mf_nums:
            QMessageBox.warning(self, "Tag Not Applied",
                                "Could not resolve mole-fraction row IDs for the selected points.")
            return

        # Toggle: if ALL selected points already carry this tag, remove it; otherwise apply it.
        already = self._fetch_tag_nums_for_mf_nums(mf_nums)
        if tag_num in already:
            self._delete_tag_for_mf_nums(mf_nums, tag_num)
            for row_idx in idxs:
                self.run.at[row_idx, "_pending_tag_num"] = pd.NA
            self._sync_rejected_state(idxs)
        else:
            self._insert_tag_for_mf_nums(mf_nums, tag_num)
            is_reject = int(self.selected_tag.get("reject") or 0) == 1
            for row_idx in idxs:
                self.run.at[row_idx, "_pending_tag_num"] = tag_num
                if is_reject:
                    self.run.at[row_idx, "rejected"] = 1
                elif tag_num in _INFO_TAG_NUMS:
                    self.run.at[row_idx, 'has_info_tag'] = True

        self.madechanges = True
        self._style_gc_buttons()

        # Preserve view
        ax = self.figure.axes[0] if self.figure.axes else None
        if ax is not None:
            self._pending_xlim = ax.get_xlim()
            self._pending_ylim = ax.get_ylim()

        # Recalculate normalized response and mole fractions
        self.run = self.instrument.norm.merge_smoothed_data(self.run)
        self.run = self.instrument.calc_mole_fraction(self.run)

        # Redraw
        self.gc_plot(self._current_yparam)
        
    def on_calcurve_selected(self, index):
        
        row = self.calcurve_combo.itemData(index)
        if row is not None:
            self.selected_calc_curve = row['run_date']
            
            self._save_payload = row.to_dict()
            
            ts_str = self.current_run_time.split(" (")[0]
            sel = pd.to_datetime(ts_str, utc=True)
            mf_mask = self.run['run_time'] == sel

            # Update cal info all at once
            self.run.loc[mf_mask, ['cal_date', 'coef0', 'coef1', 'coef2', 'coef3']] = [
                row['run_date'], row['coef0'], row['coef1'], row['coef2'], row['coef3']
            ]

            # Recalculate mole fractions
            self.run.loc[mf_mask, 'mole_fraction'] = self.instrument.calc_mole_fraction(
                self.run.loc[mf_mask]
            )

            self.madechanges = True
            self._style_gc_buttons()
            self.gc_plot('mole_fraction', sub_info='RE-CALCULATED')

    def compute_ref_estimate(self, new_fit: dict) -> pd.DataFrame:
        """
        Build a DataFrame of ref tank mole fraction estimates for this run,
        using the given calibration fit coefficients.
        """
        # Get unflagged STANDARD_PORT rows
        ref_estimate = self.run.loc[
            (self.run['port'] == self.instrument.STANDARD_PORT_NUM)
            & (self.run['rejected'].fillna(0).astype(int) != 1)
        ].copy()

        # Extract coefficients from new_fit dict
        a0, a1, a2, a3 = (
            new_fit['coef0'],
            new_fit['coef1'],
            new_fit['coef2'],
            new_fit['coef3'],
        )

        # Calculate mole fractions row-by-row
        ref_estimate = ref_estimate.assign(
            mole_fraction=[
                self.instrument.invert_poly_to_mf(y, a0, a1, a2, a3, mf_min=-20.0, mf_max=3000)
                for y in ref_estimate['normalized_resp']
            ]
        )

        return ref_estimate

    def _ie3_cal_method_selected(self, week_start):
        """Return the cal method to render: the user's combo choice if set,
        otherwise the value recorded for this week."""
        method = getattr(self, '_ie3_cal_method', None)
        if method is None:
            method = self.instrument.get_week_mf_method(
                self.current_pnum, self.current_channel, week_start
            )
        return int(method)

    def _ie3_week_fit(self, unflagged, pnum, method, weekly=None, coefs=None):
        """Compute the single-week cal fit for the given method from the
        unflagged cal/ref data. ``weekly`` and ``coefs`` may be passed in to
        avoid recomputing them. Returns (fit_row or None, coefs dict)."""
        if not _IE3_CAL_AVAILABLE or unflagged.empty:
            return None, (coefs or {})
        force_zero, single_port = self.instrument.fit_params_for_method(method)
        if weekly is None:
            weekly = _ie3_weekly_aggregate(unflagged)
        if coefs is None:
            serials = _ie3_cal_tank_serials(self.instrument)
            coefs = _ie3_cal_tank_coefs(self.instrument, pnum, serials) if serials else {}
        if not coefs:
            return None, coefs
        if force_zero:
            if single_port not in coefs:
                return None, coefs
            fits = _ie3_weekly_cal_fits(
                weekly, {single_port: coefs[single_port]}, force_zero=True
            )
        else:
            if len(coefs) < 2:
                return None, coefs
            fits = _ie3_weekly_cal_fits(weekly, coefs)
        if fits.empty:
            return None, coefs
        return fits.iloc[-1], coefs

    def _ie3_tank_assigned(self, port, pnum, coefs, week_start):
        """Assigned mole fraction (coef0) for a standard port in this week.
        Ref tank (port 5) uses ref_tank_coef0; cal tanks use the fill active on
        week_start. Returns None if unavailable."""
        if port == self.instrument.STANDARD_PORT_NUM:
            return self.instrument.ref_tank_coef0(pnum)
        fills = (coefs or {}).get(port)
        if not fills:
            return None
        return _ie3_fill_value_for_date(fills, pd.Timestamp(week_start), 'coef0')

    def _ie3_cal_plot(self):
        """Calibration-curve view for IE3 weekly cal runs.

        Plots assigned mole fraction vs normalized_resp (weekly mean per tank).
        The cal2 / ref / cal1 tank means ± std are always shown (diagnostic),
        regardless of the selected method, overlaid with the fit line for the
        current method. The "Zero" checkbox extends the fit line to the origin.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(1, 1, 1)

        if self.run.empty:
            ax.text(0.5, 0.5, 'No data loaded', transform=ax.transAxes,
                    ha='center', va='center')
            self.canvas.draw()
            return

        week_start = self.current_run_time.split(' (')[0].strip()
        gas_name = self.instrument.analytes_inv.get(self.current_pnum, str(self.current_pnum))
        pnum = self.current_pnum
        method = self._ie3_cal_method_selected(week_start)
        method_label = self.instrument.MF_METHOD_LABELS.get(method, str(method))
        zero = self.draw2zero_cb.isChecked()

        unflagged = self.run[self.run['rejected'].fillna(0).astype(int) == 0]

        # Resolve tank coef0 timelines once (reused for points and the fit).
        coefs = {}
        weekly = None
        if _IE3_CAL_AVAILABLE and not unflagged.empty:
            weekly = _ie3_weekly_aggregate(unflagged)
            serials = _ie3_cal_tank_serials(self.instrument)
            coefs = _ie3_cal_tank_coefs(self.instrument, pnum, serials) if serials else {}

        # Always plot the cal2 / ref / cal1 tank means ± std (diagnostic).
        port_color = {
            self.instrument.CAL2_PORT: 'tab:orange',
            self.instrument.STANDARD_PORT_NUM: 'tab:purple',
            self.instrument.CAL1_PORT: 'tab:gray',
        }
        stats_lines = []
        xvals = []
        for port in (self.instrument.CAL2_PORT,
                     self.instrument.STANDARD_PORT_NUM,
                     self.instrument.CAL1_PORT):
            grp = unflagged[unflagged['port'] == port]
            if grp.empty:
                continue
            label = grp['port_label'].iat[0] if 'port_label' in grp.columns else f'Port {port}'
            mean = grp['normalized_resp'].mean()
            std = grp['normalized_resp'].std()
            stats_lines.append(f"{label}: {mean:.5g} ± {std:.2g} (n={len(grp)})")
            assigned = self._ie3_tank_assigned(port, pnum, coefs, week_start)
            if assigned is not None and pd.notna(mean):
                ax.errorbar([mean], [assigned],
                            xerr=[std if pd.notna(std) else 0.0],
                            fmt='o', color=port_color.get(port, 'black'),
                            ms=7, capsize=3, zorder=3, label=label)
                xvals.append(mean)

        # Determine the fit line (slope/intercept) for the selected method.
        slope = intercept = None
        if not self.instrument.uses_ng_response_fit(method):
            # method 1 (ref): mf = coef0(ref tank) * normalized_resp.
            coef0 = self.instrument.ref_tank_coef0(pnum)
            if coef0 is not None:
                slope, intercept = coef0, 0.0
                fit_text = (f"method = {method_label}\n"
                            f"slope (ref coef0) = {coef0:.5g}\nintercept = 0")
            else:
                fit_text = f"method = {method_label}\n(no ref-tank coef0 available)"
        else:
            fit_row, _c = self._ie3_week_fit(unflagged, pnum, method,
                                             weekly=weekly, coefs=coefs)
            if fit_row is not None:
                slope = float(fit_row['slope'])
                intercept = float(fit_row['intercept'])
                fit_text = (f"method = {method_label}\n"
                            f"slope = {slope:.5g}\nintercept = {intercept:.5g}")
            else:
                fit_text = f"method = {method_label}\n(no fit: insufficient cal data)"

        # Draw the fit line; extend to the origin when "Zero" is checked.
        if slope is not None:
            xmax = (max(xvals) if xvals else 1.0) * 1.05
            xmin = 0.0 if zero else (min(xvals) * 0.97 if xvals else 0.9)
            x_line = np.array([xmin, xmax])
            ax.plot(x_line, slope * x_line + intercept, '-',
                    color='steelblue', lw=1.3, zorder=2, label='fit')
            if zero:
                ax.axhline(0.0, lw=0.8, ls='--', color='0.6', zorder=1)
                ax.axvline(0.0, lw=0.8, ls='--', color='0.6', zorder=1)
            # Ref-tank predicted point for cal-fit methods.
            ref_grp = unflagged[unflagged['port'] == self.instrument.STANDARD_PORT_NUM]
            if self.instrument.uses_ng_response_fit(method) and not ref_grp.empty:
                ref_x = ref_grp['normalized_resp'].mean()
                ref_y = slope * ref_x + intercept
                ax.plot([ref_x], [ref_y], 'D', color='crimson', ms=8, zorder=4,
                        label=f'ref predicted = {ref_y:.5g}')

        if stats_lines:
            fit_text += "\n\n" + "\n".join(stats_lines)
        ax.text(0.02, 0.98, fit_text, transform=ax.transAxes,
                ha='left', va='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9))

        ax.set_title(
            f"IE3 @ {self.instrument.site.upper()} — {gas_name} "
            f"weekly cal run {week_start}  [{method_label}]"
        )
        ax.set_xlabel("normalized_resp (weekly mean per tank)")
        ax.set_ylabel("assigned mole fraction")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def _ie3_cal_save(self):
        """Recompute the weekly cal fit from self.run and upsert to ng_response.

        Called from _perform_save_current_gas when the selected run is an IE3
        cal week entry.  Saves rejection flag changes via upsert_mole_fractions,
        then recomputes the fit from the updated unflagged data.
        """
        if not _IE3_CAL_AVAILABLE:
            print("ie3_cal_test not available; cannot recompute fit.")
            return

        # 1. Save flag changes for cal/ref port rows
        self.instrument.upsert_mole_fractions(self.run)

        # 2. Determine and record the per-week calibration method.
        pnum = self.current_pnum
        week_start = self.current_run_time.split(' (')[0].strip()
        method = self._ie3_cal_method_selected(week_start)
        method_label = self.instrument.MF_METHOD_LABELS.get(method, str(method))
        self.instrument.set_week_mf_method(
            pnum, self.current_channel, week_start, method
        )

        # method 1 (ref) stores no ng_response fit; mole fractions come from the
        # reference-tank coef0 directly in update_runs.
        if not self.instrument.uses_ng_response_fit(method):
            week_end = (pd.Timestamp(week_start) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            ch_arg = f" -c {self.current_channel}" if self.current_channel else ""
            print(f"Recorded IE3 cal method ({method_label}) for week={week_start}; "
                  "no ng_response fit (ref-tank scaling).")
            print(f"Run 'ie3_batch.py -p {pnum}{ch_arg} --site {self.instrument.site} "
                  f"-s {week_start} -e {week_end} -i' to recalculate this week's "
                  "air-port mole fractions.")
            return

        # 3. Recompute the week's fit using the selected method.
        unflagged = self.run[self.run['rejected'].fillna(0).astype(int) == 0].copy()
        if unflagged.empty:
            print("No unflagged cal data; skipping fit recompute.")
            return

        row, coefs = self._ie3_week_fit(unflagged, pnum, method)
        if row is None:
            if not coefs:
                print("No scale_assignments for cal tanks; skipping fit recompute.")
            else:
                print(f"Could not compute {method_label} fit for this week; "
                      "check cal tank coverage.")
            return
        ref_raw = unflagged[unflagged['port'] == self.instrument.STANDARD_PORT_NUM]
        sigma_ref = float(ref_raw['normalized_resp'].std()) if len(ref_raw) > 1 else 0.0

        # 4. Upsert the fit
        scale_rows = self.instrument.db.doquery(
            f"SELECT idx FROM reftank.scales "
            f"WHERE parameter_num={pnum} AND current=1"
        )
        if not scale_rows:
            print(f"No current scale for pnum={pnum}; cannot save fit.")
            return
        scale_num = int(scale_rows[0]['idx'])

        pc = self.instrument.port_config
        ref_serial = ''
        if pc is not None:
            mask = ((pc['site_num'] == self.instrument.site_num)
                    & (pc['port_num'] == self.instrument.STANDARD_PORT_NUM))
            rows = pc.loc[mask, 'label']
            if not rows.empty:
                ref_serial = rows.iat[0]

        row_id = self.instrument.upsert_ng_response(
            inst_num=self.instrument.inst_num,
            site=self.instrument.site,
            run_date=week_start,
            channel=self.current_channel or '',
            scale_num=scale_num,
            coef0=float(row['intercept']),
            coef1=float(row['slope']),
            unc_fit=float(row['unc_ref_pred']) if pd.notna(row.get('unc_ref_pred')) else 0.0,
            sigma_ref=sigma_ref,
            serial_number=ref_serial,
        )
        print(f"Saved IE3 cal fit ({method_label}): ng_response id={row_id} "
              f"week={week_start} slope={row['slope']:.4g} "
              f"intercept={row['intercept']:.4g}")
        week_end = (pd.Timestamp(week_start) + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        ch_arg = f" -c {self.current_channel}" if self.current_channel else ""
        print(f"Run 'ie3_batch.py -p {pnum}{ch_arg} --site {self.instrument.site} "
              f"-s {week_start} -e {week_end} -i' to recalculate this week's "
              "air-port mole fractions using the updated fit.")

    def calibration_plot(self):
        """
        Plot data with the legend sorted by analysis_datetime.
        Adds a residuals (diff_y) panel above the main plot that shares the x-axis.
        """
        if self.run.empty:
            self.clear_plot()
            return

        # IE3 cal run: show weekly cal fit plot instead of polynomial cal plot
        if (self.instrument.inst_id == 'ie3'
                and self.current_run_time
                and '(Cal)' in self.current_run_time):
            self._ie3_cal_plot()
            return

        # filter for run_time selected in run_cb
        ts_str = self.current_run_time.split(" (")[0]

        mask = self.run['port'].eq(self.instrument.STANDARD_PORT_NUM)
        if not mask.any():  # no rows match
            print(f"No STANDARD_PORT rows for run_time: {self.current_run_time}")
            self.gc_plot('resp')
            return
        
        sel_rt = self.run['run_time'].iat[0]  # current selected run_time (UTC-aware)

        # ── Load & normalize curves ───────────────────────────────────────────────────
        curves = self.instrument.load_calcurves(self.current_pnum, self.current_channel, sel_rt)
        REQ_COLS = ["id", "run_date","serial_number","coef3","coef2","coef1","coef0","function","flag"]

        if curves is None or curves.empty:
            # Start a fresh frame with the expected schema
            curves = pd.DataFrame(columns=REQ_COLS)
        else:
            curves = curves.copy()
            # Ensure all required columns exist
            for c in REQ_COLS:
                if c not in curves.columns:
                    curves[c] = pd.NA

        curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True, errors='coerce')

        # Reset manual override when run_time or analyte/channel changes
        current_context = (sel_rt, self.current_pnum, self.current_channel)
        if self._fit_method_last_context != current_context:
            self._fit_method_manual = False
            self._fit_method_last_context = current_context

        # Determine which fit degree to use (DB value on first render, user override afterwards)
        db_fit_degree = None
        match_row = curves.loc[curves['run_time'] == sel_rt]
        if not match_row.empty:
            row = match_row.iloc[0]
            # Treat NaN/None as 0 for degree detection
            c3 = np.nan_to_num(pd.to_numeric(row.get('coef3'), errors='coerce'))
            c2 = np.nan_to_num(pd.to_numeric(row.get('coef2'), errors='coerce'))
            c1 = np.nan_to_num(pd.to_numeric(row.get('coef1'), errors='coerce'))

            if abs(c3) > 1e-12:
                db_fit_degree = 3
            elif abs(c2) > 1e-12:
                db_fit_degree = 2
            elif abs(c1) > 1e-12:
                db_fit_degree = 1
                
        if not self._fit_method_manual:
            # use DB value (fallback to quadratic) and sync the combobox without flagging a user change
            fit_degree_from_db = db_fit_degree or 2
            idx = self.fit_method_cb.findData(fit_degree_from_db)
            if idx != -1:
                self._fit_method_updating = True
                self.fit_method_cb.setCurrentIndex(idx)
                self._fit_method_updating = False

        # Always use the value from the combobox for the calculation
        current_data = self.fit_method_cb.currentData()
        self.current_fit_degree = int(current_data) if current_data is not None else 2

        # file in scale_assignment values for calibration tanks in self.run
        self.populate_cal_mf()
        new_fit = self.instrument._fit_row_for_current_run(self.run, order=self.current_fit_degree)
        # save new fit info for Save Cal2DB button
        if new_fit is None:
            self.clear_plot(
                "Could not generate calibration fit.\n"
                "Not enough valid calibration points."
            )
            return

        self._save_payload = new_fit

        # ── Create/append as needed ───────────────────────────────────────────────────
        has_current = curves['run_time'].eq(sel_rt).any()
        calcurve_exists = True
        current_cal_flag = 0
        if curves.empty:
            # No curves at all → fit and create DF with one row
            try:
                curves = pd.DataFrame([new_fit], columns=REQ_COLS)
            except ValueError:
                print(f'Error in calcurve {new_fit}')
                self.figure.clear()
                self.canvas.draw()
                return
            curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True, errors='coerce')
            calcurve_exists = False
        elif not has_current:
            # Existing curves, but not for this run_time → append one row
            curves = pd.concat([curves, pd.DataFrame([new_fit])], ignore_index=True)
            curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True, errors='coerce')
            calcurve_exists = False
        else:
            match = curves.loc[curves['run_time'] == sel_rt]
            current_cal_flag = match['flag'].iat[0]
            if 'id' in match.columns:
                val = match['id'].iat[0]
                if pd.notna(val):
                    self._save_payload['id'] = int(val)
            
        self._save_payload['flag'] = current_cal_flag

        # pd dataframe of ref tank mole fraction estimates for this run
        ref_estimate = self.compute_ref_estimate(new_fit)
        
        #colors = self.run['port_colors']
        ports_in_run = sorted(self.run['port_idx'].dropna().unique())

        legend_handles = []
        for port in ports_in_run:
            color = self.run.loc[self.run['port_idx'] == port, 'port_color'].iloc[0]
            marker = self.run.loc[self.run['port_idx'] == port, 'port_marker'].iloc[0]
            label=self.run.loc[self.run['port_idx'] == port, 'port_label'].iloc[0]
            if port == 9:   # skip port 9 (Push Gas Port)
                continue

            legend_handles.append(Line2D(
                [], [],
                color=color,
                marker=marker,
                linestyle='None',
                markersize=8,
                label=label
            ))

        # ref tank mean and std
        ref_mf_mean = ref_estimate['mole_fraction'].mean()
        ref_mf_sd   = ref_estimate['mole_fraction'].std()
        ref_resp_mean = ref_estimate['normalized_resp'].mean()
        ref_resp_sd   = ref_estimate['normalized_resp'].std()

        yvar = 'normalized_resp'  # Use normalized_resp for calibration plots
        try:
            tlabel = f'Calibration Scale {int(np.nanmin(self.run["cal_scale_num"]))}'
        except TypeError:
            tlabel = 'Calibration Scale UNDEFINED'
            self.figure.clear()
            self.canvas.draw()
            return
            
        mn_cal = float(np.nanmin(self.run['cal_mf']))
        mx_cal = float(np.nanmax(self.run['cal_mf']))
        try:
            #x_one2one = np.linspace(mn_cal * .95, mx_cal * 1.05, 200)
            x_one2one = np.linspace(-ref_mf_mean*0.2, mx_cal * 1.05, 200)
            if np.isfinite(ref_mf_mean) and ref_mf_mean != 0:
                y_one2one = x_one2one / ref_mf_mean
            else:
                x_one2one, y_one2one = [], []
        except ValueError:
            x_one2one, y_one2one = [], []
    
        # Build figure with residuals panel on top sharing x-axis
        self.figure.clear()
        gs = self.figure.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 3], hspace=0.05)
        ax_resid = self.figure.add_subplot(gs[0, 0])                    # top (smaller)
        ax       = self.figure.add_subplot(gs[1, 0], sharex=ax_resid)   # bottom (main)

        # Fill any missing ref cal_mf with ref_mf_mean (fix former mf_mean reference)
        self.run.loc[self.run['port'] == self.instrument.STANDARD_PORT_NUM, 'cal_mf'] = (
            self.run.loc[self.run['port'] == self.instrument.STANDARD_PORT_NUM, 'cal_mf'].fillna(ref_mf_mean)
        )

        # Mask for valid points
        mask_main = self.run[['cal_mf', yvar]].notna().all(axis=1)
        hide_flagged = self.hide_flagged_cb.isChecked()
        
        if hide_flagged:
            plot_df = self.run.loc[mask_main & (self.run['rejected'].fillna(0).astype(int) == 0)]
        else:
            plot_df = self.run.loc[mask_main]

        # Main scatter (all unflagged + flagged, same colors)
        for port, subset in plot_df.groupby('port_idx'):
            marker = subset['port_marker'].iloc[0]
            color = subset['port_color']

            ax.scatter(
                subset['cal_mf'],
                subset[yvar],
                marker=marker,
                c=color,
                s=10,
                zorder=1,
            )

        # Mask for flagged subset
        flags = (
            self.run['rejected'].fillna(0).astype(int) != 0
        ) & mask_main
        
        if not hide_flagged:
            # Overlay flagged points
            ax.scatter(
                self.run.loc[flags, 'cal_mf'],
                self.run.loc[flags, yvar],
                marker='o',
                c='white',
                edgecolors=self.run.loc[flags, 'port_color'],
                s=self.instrument.BASE_MARKER_SIZE * 1.1,  # a bit larger than unflagged for visibility
                linewidths=2,
                zorder=4
            )
        # One-to-one line
        ax.plot(x_one2one, y_one2one, c='grey', ls='--', label='one-to-one')

        # Ref point with error bars
        ax.errorbar(
            ref_mf_mean, ref_resp_mean,
            xerr=ref_mf_sd, yerr=ref_resp_sd,
            fmt='o', color='black', ecolor='black',
            elinewidth=1.2, capsize=3, markersize=10, zorder=10,
        )
        if np.isfinite(ref_mf_mean) and np.isfinite(ref_resp_mean):
            legend_handles.append(Line2D([], [], marker='o', linestyle='None', color='black', 
                                         label="Ref Mean $\\pm 1\\sigma$"))
    
        # Plot stored cal curves (bottom axis).
        stored_curve = curves['run_time'].eq(sel_rt)
        if stored_curve is None or stored_curve.empty:
            print("No stored cal curves available.")
            return
        row = curves.loc[stored_curve].iloc[0]
        stored_coefs = [row['coef3'], row['coef2'], row['coef1'], row['coef0']]

        # fit labels are from stored curve
        fitlabel = f'\nfit =\n'
        for n, coef in enumerate(stored_coefs[::-1]):
            if coef != 0.0:
                fitlabel += f'{coef:0.6f} ($x^{n}$) \n'
        legend_handles.append(Line2D([], [], linestyle='None', label=fitlabel))
            
        # Predicted response at all valid x_all positions
        new_fit_coefs = [new_fit["coef3"], new_fit["coef2"], new_fit["coef1"], new_fit["coef0"]]
        x_all = self.run.loc[mask_main, 'cal_mf'].astype(float)
        y_all = self.run.loc[mask_main, yvar].astype(float)

        y_pred = np.polyval(new_fit_coefs, x_all.to_numpy())
        diff_y = y_all.to_numpy() - y_pred

        # Store residuals on self.run (align by index)
        self.run.loc[mask_main, 'diff_y'] = diff_y

        # --- Residuals ---
        # Unflagged residuals
        ax_resid.scatter(
            self.run.loc[mask_main & ~flags, 'cal_mf'],
            self.run.loc[mask_main & ~flags, 'diff_y'],
            s=15, c=self.run.loc[mask_main & ~flags, 'port_color'],
            alpha=0.8
        )

        if not hide_flagged:
            # Flagged residuals (white or light-gray face, colored edge)
            ax_resid.scatter(
                self.run.loc[mask_main & flags, 'cal_mf'],
                self.run.loc[mask_main & flags, 'diff_y'],
                facecolors='whitesmoke',  # or '#f0f0f0' for slightly darker
                edgecolors=self.run.loc[mask_main & flags, 'port_color'],
                s=self.instrument.BASE_MARKER_SIZE * 1.1,  # a bit larger than unflagged for visibility
                linewidths=1.2,
                zorder=4,
            )
        # Horizontal zero line
        ax_resid.axhline(0.0, lw=1, ls='--', color='0.4')

        # set symmetric y-limits for residuals
        diff_y_vis = self.run.loc[plot_df.index, 'diff_y'].dropna() if not plot_df.empty else pd.Series(dtype=float)
        if not diff_y_vis.empty and np.isfinite(diff_y_vis).any():
            maxabs = np.nanmax(np.abs(diff_y_vis))
            if np.isfinite(maxabs) and maxabs > 0:
                ax_resid.set_ylim(-1.1 * maxabs, 1.1 * maxabs)

        # Extend to zero if checked
        if self.draw2zero_cb.isChecked():
            xgrid = np.linspace(-ref_mf_mean*0.2, mx_cal * 1.05, 300)
        else:
            xgrid = np.linspace(mn_cal * .95, mx_cal * 1.05, 300)
            
        # Original curve from DB (black, thinner)
        if calcurve_exists:
            ygrid = np.polyval(stored_coefs, xgrid)
            ax.plot(xgrid, ygrid, linewidth=2, color='black', alpha=0.7,
                            label=row['run_date'].strftime('%Y-%m-%d'))
            legend_handles.append(Line2D([], [], color='black', linewidth=3, label=f"Fit {row['run_date'].strftime('%Y-%m-%d')}"))
            
        if self.oldcurves_cb.isChecked():
            t0, t1 = self.get_load_range()
            t0 = pd.to_datetime(t0)
            t1 = pd.to_datetime(t1)
            
            curves['run_date'] = pd.to_datetime(curves['run_date'])

            # Filter curves to within the selected range
            mask = (curves['run_date'] >= t0) & (curves['run_date'] <= t1)
            subset = curves.loc[mask].sort_values('run_date', ascending=False)

            # Limit to a manageable number, e.g., the 6 most recent within range
            max_curves = 6
            subset = subset.head(max_curves)

            # plot stored cal curves
            for row in subset.itertuples():
                coefs = [row.coef3, row.coef2, row.coef1, row.coef0]
                flagged = int(row.flag)
                ygrid = np.polyval(coefs, xgrid)
                if flagged == 1:
                    ax.plot(xgrid, ygrid, linewidth=1, color='red', linestyle=':', alpha=0.7)
                else:
                    ax.plot(xgrid, ygrid, linewidth=1, color='red', linestyle='-', alpha=0.7)
            legend_handles.append(Line2D([], [], color='red', linewidth=3, label=f"Other fits"))
        
        # Show the new fit line (green) only if there was no existing curve for this run,
        # or if the user has manually changed the fit degree.
        if not calcurve_exists or self._fit_method_manual:
            ygrid_new = np.polyval(new_fit_coefs, xgrid)
            ax.plot(xgrid, ygrid_new, linewidth=3, color='green', alpha=0.7)
            legend_handles.append(Line2D([], [], color='green', linewidth=3, label="New Fit"))
        
        # Warning box if new curve
        if calcurve_exists == False:
            ax.text(
                0.5, .98, "New Calibration Curve - NOT SAVED",
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, color='white', clip_on=False,
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    facecolor='#8B0000',   # dark red
                    edgecolor='none',      # or '#8B0000' if you want a border
                    alpha=0.9
                )
            )

        # Titles/labels
        title = (f"{self.current_run_time} - {tlabel}: "
                f"{self.instrument.analytes_inv[int(self.current_pnum)]} ({self.current_pnum})")
        self.figure.suptitle(title, y=0.98)  # sits above both axes
        ax_resid.set_title("Residuals")
        units = '(ppb)' if int(self.current_pnum) == 5 else '(ppt)'  # ppb for N2O
        ax.set_xlabel(f"Mole Fraction {units}")
        ax.xaxis.set_tick_params(rotation=30)
        ax.set_ylabel('Normalized Response')
        ax_resid.set_ylabel('obs − curve')
        ax_resid.tick_params(labelbottom=False)  # hide top x tick labels (shared x)
        ax.format_coord = self._fmt_cal_plot
                
        save_handle = Line2D([], [], linestyle='None', label='Save Cal2DB')
        current = int(self._save_payload.get('flag', 0))
        flag_label  = 'Unflag Cal2DB' if current else 'Flag Cal2DB'

        # spacer creates a small gap between the two buttons
        spacer_handle = Line2D([], [], linestyle='None', label='\u2009')  # hair space

        flag_handle = Line2D([], [], linestyle='None', label=flag_label)
        legend_handles.extend([save_handle, spacer_handle, flag_handle])
        
        # Put legend outside; adjust right margin instead of manually resizing axes
        self.figure.subplots_adjust(right=0.82)
        # legend call (add a couple of tweaks so the text-only entry aligns nicely)
        leg = ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=False,
            handlelength=0.5,      # no space for the text-only handle
            handletextpad=0.4,
            borderaxespad=0.3,
            labelspacing=0.6,
        )
        # Style the Save Cal2DB label like a button and make it clickable
        self._save_text = None
        self._flag_text = None
        self._spacer_text = None

        for txt in leg.get_texts():
            t = txt.get_text().strip()
            if t == 'Save Cal2DB':
                self._save_text = txt
                txt.set_color('white')
                txt.set_bbox(dict(boxstyle='round,pad=0.4', facecolor='#2e7d32', edgecolor='none', alpha=0.95))
                txt.set_picker(True)

            elif t in ('Flag Cal2DB', 'Unflag Cal2DB'):
                self._flag_text = txt
                txt.set_picker(True)

            elif txt.get_text() == '\u2009':  # our spacer
                self._spacer_text = txt
                txt.set_fontsize(3)            # tiny line height -> small gap
                txt.set_color((0, 0, 0, 0))    # fully transparent

        # Apply initial style for the flag button based on state
        self._style_flag_button()

        # Connect the pick handler once
        if not hasattr(self, '_legend_pick_cid') or self._legend_pick_cid is None:
            self._legend_pick_cid = self.canvas.mpl_connect('pick_event', self._on_legend_pick)
    
        # left-align multi-line labels (works on modern Matplotlib)
        try:
            for t in leg.get_texts():
                t.set_ha('left')
            leg._legend_box.align = "left"   # noqa: access to private attr; widely used workaround
        except Exception:
            pass
    
        # Ensure plot limits respect visible data only
        try:
            if not plot_df.empty:
                vis_x_min, vis_x_max = plot_df['cal_mf'].min(), plot_df['cal_mf'].max()
                vis_y_min, vis_y_max = plot_df[yvar].min(), plot_df[yvar].max()
                
                # Autoscale X-axis (always)
                if self.draw2zero_cb.isChecked():
                    ax.set_xlim(-ref_mf_mean * 0.2 if np.isfinite(ref_mf_mean) else -0.2, vis_x_max * 1.05)
                else:
                    ax.set_xlim(vis_x_min * 0.95, vis_x_max * 1.05)

                # Determine ideal Y-axis limits
                ideal_y_lim = (-0.2, 1.2) if self.draw2zero_cb.isChecked() else (vis_y_min * 0.95, vis_y_max * 1.05)

                # Apply Y-axis limits
                if self.lock_y_axis_cb.isChecked():
                    if self.y_axis_limits is None:
                        self.y_axis_limits = ideal_y_lim
                    ax.set_ylim(self.y_axis_limits)
                else:
                    ax.set_ylim(ideal_y_lim)
        except ValueError:
            pass
        ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.8)
        
        self.canvas.draw()

    def populate_cal_mf(self) -> None:
        """
        Fill self.run['cal_mf'] from instrument.scale_assignments(tank, pnum).
        Expects scale_assignments to return a dict/Series with keys 'coef0' and 'coef1'
        (or None if not found).
        coef1 is not completely coded yet. No drift allowed.
        """
        if 'port_info' not in self.run:
            raise KeyError("self.run must contain a 'port_info' column")

        # Normalize tank IDs to strings for consistent lookup
        tank_series = (
            self.run['port_info']
            .astype('string')        # keeps NaN semantics
            .str.strip()
        )

        unique_tanks = tank_series.dropna().unique().tolist()
        if not unique_tanks:
            self.run['cal_mf'] = np.nan
            return

        # Date of this run, so refilled tanks resolve to the fill that was in
        # use rather than an arbitrary record.
        run_date = None
        if 'analysis_datetime' in self.run and self.run['analysis_datetime'].notna().any():
            run_date = pd.to_datetime(self.run['analysis_datetime']).max()

        @lru_cache(maxsize=None)
        def get_scale(tank_id: str):
            rec = self.instrument.scale_assignments(tank_id, self.current_pnum, run_date=run_date)
            if rec is None:
                return None
            if isinstance(rec, pd.Series):
                rec = rec.to_dict()
            return rec

        tank_to_coef0 = {}
        drift_tanks = []
        for t in unique_tanks:
            rec = get_scale(t)
            if not rec:
                # no scale assignment for this tank
                continue
            coef1 = float(rec.get('coef1') or 0.0)
            if coef1 != 0.0:
                drift_tanks.append((t, coef1))
            coef0 = rec.get('coef0')
            tank_to_coef0[t] = float(coef0) if coef0 is not None else np.nan

        # Map back to the dataframe (vectorized) and ensure numeric dtype
        self.run['cal_mf'] = tank_series.map(tank_to_coef0).astype('float64')

        # Warn (or error) about drift, depending on your preference
        if drift_tanks:
            msg = "scale_assignments has non-zero coef1 (drift) for: " + \
                ", ".join(f"{t} (coef1={c:g})" for t, c in drift_tanks)
            # Option A (recommended): warn but continue
            warnings.warn(msg, RuntimeWarning)
            # Option B (strict): stop execution
            # raise RuntimeError(msg)

        # Optional: warn if some tanks had no assignment at all
        #missing = sorted(set(tank_series.dropna()) - set(tank_to_coef0))
        #if missing:
        #    warnings.warn("No scale assignment found for: " + ", ".join(missing), UserWarning)

    def _style_flag_button(self):
        """Refresh the legend 'flag' button look based on _save_payload['flag']"""
        if getattr(self, '_flag_text', None) is None:
            return

        # default to unflagged if payload missing
        flagged = False
        if getattr(self, "_save_payload", None) is not None:
            flagged = bool(int(self._save_payload.get('flag', 0)))

        if flagged:
            self._flag_text.set_text('Unflag Cal2DB')
            self._flag_text.set_color('white')
            self._flag_text.set_bbox(dict(
                boxstyle='round,pad=0.4',
                facecolor='darkred',
                edgecolor='none',
                alpha=0.95
            ))
        else:
            self._flag_text.set_text('Flag Cal2DB')
            self._flag_text.set_color('white')
            self._flag_text.set_bbox(dict(
                boxstyle='round,pad=0.4',
                facecolor='#616161',  # neutral gray
                edgecolor='none',
                alpha=0.9
            ))

    def recalculate_dependent_data(self, response_id):
        if response_id is None:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Find all run_times that use this response_id
            sql = f"""
                SELECT DISTINCT a.run_time
                FROM hats.ng_mole_fractions mf
                JOIN hats.ng_analysis a ON mf.analysis_num = a.num
                WHERE mf.ng_response_id = {response_id}
            """
            rows = self.instrument.db.doquery(sql)
            if not rows:
                return
            
            run_times = [r['run_time'] for r in rows]
            print(f"Recalculating {len(run_times)} runs for response_id {response_id}...")
            
            current_run_updated = False
            current_rt_ts = None
            if self.current_run_time:
                ts_str = self.current_run_time.split(" (")[0]
                current_rt_ts = pd.to_datetime(ts_str, utc=True)

            for rt in run_times:
                rt_ts = pd.to_datetime(rt, utc=True)
                # Use a small window to avoid precision issues with exact string matching
                t0 = (rt_ts - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                t1 = (rt_ts + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                
                df = self.instrument.load_data(
                    pnum=self.current_pnum,
                    channel=self.current_channel,
                    start_date=t0,
                    end_date=t1,
                    verbose=False
                )
                if df.empty:
                    continue
                
                df = self.instrument.calc_mole_fraction(df)
                
                if 'ng_response_id' in df.columns:
                    df_sub = df[df['ng_response_id'].fillna(-1).astype(int) == int(response_id)]
                    if not df_sub.empty:
                        self.instrument.upsert_mole_fractions(df_sub, response_id=response_id)
                        self.instrument.upsert_calibrations(df_sub, self.current_pnum)

                if current_rt_ts is not None and rt_ts == current_rt_ts:
                    current_run_updated = True

            if current_run_updated:
                self.load_selected_run()
                
        finally:
            QApplication.restoreOverrideCursor()

    def save_current_curve(self, flag_value=None):
        payload = getattr(self, "_save_payload", None)
        if payload is None:
            print("No calibration payload available.")
            return

        print("Saving calibration curve to database:", payload['flag'])
        if flag_value is not None:
            payload['flag'] = flag_value

        fields = ['scale_num','inst_num','site','run_date','channel',
                'coef0','coef1','coef2','coef3','flag','function','serial_number']

        sql = """
            INSERT INTO ng_response
                (scale_num, inst_num, site, run_date, channel,
                coef0, coef1, coef2, coef3, flag, function, serial_number)
            VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                channel = VALUES(channel),
                site = VALUES(site),
                scale_num = VALUES(scale_num),
                coef0 = VALUES(coef0),
                coef1 = VALUES(coef1),
                coef2 = VALUES(coef2),
                coef3 = VALUES(coef3),
                flag  = VALUES(flag),
                function = VALUES(function),
                serial_number = VALUES(serial_number)
            """
        params = [payload.get(c) for c in fields]
        #print(sql); print(params)
        self.instrument.db.doquery(sql, params)
        self.calibration_plot()
        
        if self._save_payload and 'id' in self._save_payload:
            self.recalculate_dependent_data(self._save_payload['id'])
          
    def _on_legend_pick(self, event):
        art = event.artist
        if not isinstance(art, mtext.Text):
            return

        # Identify by object identity (robust even if label text changes)
        if art is getattr(self, '_save_text', None):
            self.save_current_curve()
        elif art is getattr(self, '_flag_text', None):
            self.toggle_flag_current_curve()
        elif art is self._save2db_text:
            self._save_current_gas_action()
        elif art is self._save2dball_text:
            if not self.madechanges:
                return
            # the button is disabled for all gases while smoothing is changing. Use only one gas at a time.
            if self.smoothing_changed:
                return
            #print(f"Save flags to all gases clicked")
            
            # set button to yellow while running
            bbox = self._save2dball_text.get_bbox_patch()
            if bbox:
                bbox.update(dict(facecolor="yellow", edgecolor="none", alpha=0.95))
            self._save2dball_text.set_color("black")
            self.canvas.draw_idle()   # refresh display immediately
            self.canvas.flush_events()
            QApplication.processEvents()
            QTimer.singleShot(0, self.update_all_analytes)

        elif art is self._revert_text:
            if not self.madechanges:
                return
            self.madechanges = False
            self.smoothing_changed = False
            self.load_selected_run()
            self.gc_plot(self._current_yparam, sub_info='REVERTED')

    def update_all_analytes(self):
            pnum = self.current_pnum
            ch = self.current_channel
            self.instrument.update_flags_all_analytes(self.run)

            start, end = self._current_load_dates()

            # reload all gases for this run_time which will recalculate mole fractions
            for key, param_num in self.analytes.items():
                if "(" in key and ")" in key:
                    analyte, channel = key.split("(", 1)
                    analyte = analyte.strip()
                    channel = channel.strip(") ")
                else:
                    analyte = key.strip()
                    channel = None

                run = self.instrument.load_data(
                    pnum=int(param_num),
                    channel=channel,
                    #run_type_num=self.run_type_num,
                    start_date=start,
                    end_date=end,
                    verbose=False
                )
                run = self.instrument.calc_mole_fraction(run)
                self.instrument.upsert_mole_fractions(run)
                self.instrument.upsert_calibrations(run, int(param_num))

            # reload current gas to refresh display
            self.run = self.instrument.load_data(
                    pnum=pnum,
                    channel=ch,
                    start_date=start,
                    end_date=end,
                    verbose=False
            )
            self.update_smoothing_combobox()

            self.madechanges = False
            self.gc_plot(self._current_yparam, sub_info='SAVED ALL FLAGS')
            
            self._save2dball_text.set_bbox(dict(
                boxstyle="round,pad=0.4",
                facecolor="#9e9e9e",
                edgecolor="none", alpha=0.95
            ))
            self._save2dball_text.set_color("white")
            self.canvas.draw_idle()
            
    def toggle_flag_current_curve(self):
        """Toggle the flag value for the current calibration curve."""
        if getattr(self, "_save_payload", None) is None:
            print("No calibration payload to toggle.")
            return

        # Flip the flag in payload
        current_flag = int(self._save_payload.get("flag", 0))
        new_flag = 0 if current_flag == 1 else 1
        self._save_payload["flag"] = new_flag

        # Save to DB
        self.save_current_curve(flag_value=new_flag)

        # Refresh the button style
        self._style_flag_button()
        if getattr(self, "canvas", None):
            self.canvas.draw_idle()
        
    def get_load_range(self):
        """
        Return (start_date, end_date) strings for SQL filtering based on GUI selections.
        Ensures valid end-of-month dates, handles reversed order by swapping both
        internally and in the UI combo boxes.
        """
        # --- Read selections ---
        sy = int(self.start_year_cb.currentText())
        sm = self.start_month_cb.currentIndex() + 1
        ey = int(self.end_year_cb.currentText())
        em = self.end_month_cb.currentIndex() + 1

        # --- Convert to pandas timestamps ---
        start = pd.Timestamp(f"{sy}-{sm:02d}-01 00:00:00")
        end = pd.Timestamp(f"{ey}-{em:02d}-01 00:00:00") + pd.offsets.MonthEnd(1)

        # Include the very end of the selected end month (23:59:59)
        end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # --- Handle reversed selection ---
        if start > end:
            # Swap dates
            start, end = end, start

            # Update the UI combo boxes to match the corrected order
            self.start_year_cb.setCurrentText(str(start.year))
            self.start_month_cb.setCurrentIndex(start.month - 1)
            self.end_year_cb.setCurrentText(str(end.year))
            self.end_month_cb.setCurrentIndex(end.month - 1)
            self.apply_dates()

        # --- Return valid date strings ---
        return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")
    
    def set_runlist(self, initial_date=None):
        self._ie3_exact_run_time = None
        t0, t1 = self.get_load_range()

        # If runTypeCombo is set, filter the data by run_type_num
        run_type = self.runTypeCombo.currentText()
        self.run_type_num = self.instrument.RUN_TYPE_MAP.get(run_type, None)
        self._update_calibration_button_state()

        if self.instrument.inst_id in ('ie3', 'cats'):
            period = _ie3_chunk_period(self._inst_cfg)
            chunk_labels = [
                label for label, _s, _e in _ie3_iter_chunks(t0, t1, period)
            ]
            if self.instrument.inst_id == 'ie3' and self.current_pnum is not None:
                cal_dates = self.instrument.query_cal_run_dates(
                    self.current_pnum, self.current_channel, t0, t1
                )
                cal_entries = [
                    d.split(' ')[0] + ' (Cal)' for d in cal_dates
                ]
                if run_type == 'Calibrations':
                    self.current_run_times = cal_entries
                elif run_type == 'Air Samples':
                    self.current_run_times = chunk_labels
                else:
                    def _cal_sort_key(s):
                        base = s.split(' (')[0].strip()
                        return base + '-01' if len(base) == 7 else base
                    self.current_run_times = sorted(
                        chunk_labels + cal_entries, key=_cal_sort_key
                    )
            else:
                self.current_run_times = chunk_labels
        else:
            # make run_time lists
            cal_idx = self.instrument.RUN_TYPE_MAP.get('Calibrations')
            pfp_idx = self.instrument.RUN_TYPE_MAP.get('PFPs')  # may be None if this instrument has no PFPs

            runlist = self.instrument.query_return_run_list(runtype=self.run_type_num, start_date=t0, end_date=t1)
            runlist_cals = []
            if cal_idx is not None:
                runlist_cals = self.instrument.query_return_run_list(runtype=cal_idx, start_date=t0, end_date=t1)
            runlist_pfps = []
            if pfp_idx is not None:
                runlist_pfps = self.instrument.query_return_run_list(runtype=pfp_idx, start_date=t0, end_date=t1)

            self.current_run_times = [
                r + " (Cal)" if r in runlist_cals
                else r + " (PFP)" if r in runlist_pfps
                else r
                for r in runlist
            ]

        def normalize_timestamp(value):
            """Return pandas.Timestamp without tz info for reliable comparisons."""
            if value is None:
                return None
            try:
                ts = pd.to_datetime(value)
            except Exception:
                return None
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_localize(None)
            return ts

        def find_run_index(target):
            """Find run index even if target lacks suffix like ' (PFP)'."""
            if target is None:
                return None
            target_str = str(target)
            if self.instrument.inst_id in ('ie3', 'cats'):
                # Direct label match first; otherwise treat target as a timestamp
                # and find the chunk containing it.
                for idx, label in enumerate(self.current_run_times):
                    if label == target_str:
                        return idx
                target_ts = normalize_timestamp(target_str)
                if target_ts is None:
                    return None
                period = _ie3_chunk_period(self._inst_cfg)
                chunk_label, _s, _e = _ie3_chunk_for_timestamp(target_ts, period)
                for idx, label in enumerate(self.current_run_times):
                    if label == chunk_label:
                        return idx
                return None
            target_ts = normalize_timestamp(target_str)
            for idx, label in enumerate(self.current_run_times):
                base = label.split(" (")[0]
                if label == target_str or base == target_str:
                    return idx
                if target_ts is not None:
                    label_ts = normalize_timestamp(base)
                    if label_ts is not None and label_ts == target_ts:
                        return idx
            return None
        
        # Fill the run_cb combo with these run_time strings
        self.run_cb.blockSignals(True)
        self.run_cb.clear()
        for s in self.current_run_times:
            self.run_cb.addItem(s)

        # Preserve the current_run_time if it exists in the new analyte's run_times
        preserved_idx = find_run_index(self.current_run_time)
        if preserved_idx is not None:
            self.run_cb.setCurrentIndex(preserved_idx)
            self.current_run_time = self.current_run_times[preserved_idx]
        elif self.current_run_times:
            # Default to the last run_time if the current_run_time is not found
            last_idx = len(self.current_run_times) - 1
            self.run_cb.setCurrentIndex(last_idx)

        initial_idx = find_run_index(initial_date)
        if initial_idx is not None:
            self.run_cb.setCurrentIndex(initial_idx)
            self.current_run_time = self.current_run_times[initial_idx]
        else:
            # Default to the last run_time if no initial_date provided
            try:
                self.current_run_time = self.current_run_times[-1]
            except IndexError:
                self.current_run_time = None

        self.run_cb.blockSignals(False)
        
        self.load_selected_run()
        self._update_notes_button_style()
        self.gc_plot('resp')
        self._update_calibration_button_state()
            
    def populate_analyte_controls(self):
        """
        If there are ≤ 12 analytes → show radio buttons in two columns,
        first 6 in the left, next 6 in the right.
        If > 12 analytes → show a QComboBox instead.
        """
        # Clear any existing widgets in analyte_layout
        for i in reversed(range(self.analyte_layout.count())):
            w = self.analyte_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        # Reset analyte selectors
        self.analyte_combo = None
        self.radio_group = None

        names = list(self.analytes.keys())
        if len(names) <= 12:
            # Use radio buttons in two columns: first 6 left, rest right
            self.radio_group = QButtonGroup(self)
            left = names[:6]
            right = names[6:]

            # Left column (column 0)
            for row, name in enumerate(left):
                rb = QRadioButton(name)
                self.analyte_layout.addWidget(rb, row, 0)
                self.radio_group.addButton(rb)
                rb.toggled.connect(self.on_analyte_radio_toggled)

            # Right column (column 1)
            for row, name in enumerate(right):
                rb = QRadioButton(name)
                self.analyte_layout.addWidget(rb, row, 1)
                self.radio_group.addButton(rb)
                rb.toggled.connect(self.on_analyte_radio_toggled)

            # Ensure the layout stretches properly to accommodate options_gb
            self.analyte_layout.addWidget(QWidget(), len(left), 0, 1, 2)

        else:
            # Use a QComboBox
            self.analyte_combo = QComboBox()
            for name in names:
                self.analyte_combo.addItem(name)
            self.analyte_combo.currentTextChanged.connect(self.on_analyte_combo_changed)

            self.analyte_prev_btn = QPushButton("◀")
            self.analyte_next_btn = QPushButton("▶")
            self.analyte_prev_btn.setToolTip("Previous analyte")
            self.analyte_next_btn.setToolTip("Next analyte")
            self.analyte_prev_btn.clicked.connect(self.on_prev_analyte)
            self.analyte_next_btn.clicked.connect(self.on_next_analyte)

            combo_row = QHBoxLayout()
            combo_row.setContentsMargins(0, 0, 0, 0)
            combo_row.setSpacing(4)
            combo_row.addWidget(self.analyte_combo, 1)
            combo_row.addWidget(self.analyte_prev_btn)
            combo_row.addWidget(self.analyte_next_btn)

            combo_container = QWidget()
            combo_container.setLayout(combo_row)
            self.analyte_layout.addWidget(combo_container, 0, 0, 1, 2)

        self._setup_analyte_shortcuts()

    def on_analyte_radio_toggled(self):
        """
        Called whenever one of the ≤12 radio buttons toggles to “checked.”
        We only react when it becomes checked.
        """
        rb = self.sender()
        if rb.isChecked():
            name = rb.text()
            # Extract channel if present in analyte_name
            if '(' in name and ')' in name:
                self.current_channel = name.split('(')[1].split(')')[0].strip()
            else:
                self.current_channel = None
            pnum = self.analytes[name]
            self.current_pnum = int(pnum)
            self.set_current_analyte(name)

    def _select_analyte_control(self, name):
        """Select the UI control that matches the current analyte without reloading data."""
        if getattr(self, 'analyte_combo', None) is not None:
            previous = self.analyte_combo.blockSignals(True)
            try:
                self.analyte_combo.setCurrentText(name)
            finally:
                self.analyte_combo.blockSignals(previous)

        if getattr(self, 'radio_group', None) is not None:
            buttons = self.radio_group.buttons()
            previous_states = [(button, button.blockSignals(True)) for button in buttons]
            try:
                for button in buttons:
                    if button.text() == name:
                        button.setChecked(True)
                        break
            finally:
                for button, previous in previous_states:
                    button.blockSignals(previous)

    def on_analyte_combo_changed(self, name):
        """
        Called whenever the QComboBox selection changes (for >12 analytes).
        """
        # Extract channel if present in analyte_name
        if '(' in name and ')' in name:
            self.current_channel = name.split('(')[1].split(')')[0].strip()
        else:
            self.current_channel = None
        pnum = self.analytes[name]
        self.current_pnum = int(pnum)
        self.set_current_analyte(name)

    def on_prev_analyte(self):
        """
        Move the analyte selection one index backward, if possible.
        """
        # Combobox mode (>12 analytes)
        if self.analyte_combo:
            combo = self.analyte_combo
            if combo.count() == 0:
                return

            combo.blockSignals(True)
            idx = combo.currentIndex()
            if idx > 0:
                combo.setCurrentIndex(idx - 1)
                self.on_analyte_combo_changed(combo.currentText())
            combo.blockSignals(False)
            return

        # Radio button mode (<=12 analytes)
        if self.radio_group:
            buttons = self.radio_group.buttons()
            if not buttons:
                return
            current_idx = next((i for i, b in enumerate(buttons) if b.isChecked()), 0)
            new_idx = max(0, current_idx - 1)
            buttons[new_idx].setChecked(True)

    def on_next_analyte(self):
        """
        Move the analyte selection one index forward, if possible.
        """
        # Combobox mode (>12 analytes)
        if self.analyte_combo:
            combo = self.analyte_combo
            if combo.count() == 0:
                return

            combo.blockSignals(True)
            idx = combo.currentIndex()
            if idx < (combo.count() - 1):
                combo.setCurrentIndex(idx + 1)
                self.on_analyte_combo_changed(combo.currentText())
            combo.blockSignals(False)
            return

        # Radio button mode (<=12 analytes)
        if self.radio_group:
            buttons = self.radio_group.buttons()
            if not buttons:
                return
            current_idx = next((i for i, b in enumerate(buttons) if b.isChecked()), 0)
            new_idx = min(len(buttons) - 1, current_idx + 1)
            buttons[new_idx].setChecked(True)

    def _activate_plot_radio(self, button: QRadioButton) -> None:
        """Activate a plot radio button via keyboard shortcut."""
        if button is None:
            return
        button.click()

    def _setup_plot_shortcuts(self):
        """Assign single-key shortcuts for plot type selection."""
        for sc in getattr(self, "plot_shortcuts", []):
            sc.setParent(None)
        self.plot_shortcuts = []

        shortcuts = [
            ("R", self.resp_rb),
            ("T", self.ratio_rb),
            ("M", self.mole_fraction_rb),
        ]

        for key, button in shortcuts:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda b=button: self._activate_plot_radio(b))
            self.plot_shortcuts.append(sc)

    def _set_smoothing_index_by_label(self, label: str) -> None:
        """Update smoothing combobox selection by label text."""
        idx = self.smoothing_cb.findText(label)
        if idx >= 0:
            self.smoothing_cb.setCurrentIndex(idx)

    def _setup_smoothing_shortcuts(self):
        """Assign single-key shortcuts for smoothing selection."""
        for sc in getattr(self, "smoothing_shortcuts", []):
            sc.setParent(None)
        self.smoothing_shortcuts = []

        shortcuts = [
            ("P", "Point to Point (p)"),
            ("L", "Lowess 5 point (l)"),
        ]

        for key, label in shortcuts:
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(lambda lbl=label: self._set_smoothing_index_by_label(lbl))
            self.smoothing_shortcuts.append(sc)

    def _cycle_autoscale_mode(self) -> None:
        modes = [
            self.autoscale_samples_rb,
            self.autoscale_standard_rb,
            self.autoscale_fullscale_rb,
        ]
        current_idx = next((i for i, rb in enumerate(modes) if rb.isChecked()), 0)
        next_idx = (current_idx + 1) % len(modes)
        modes[next_idx].setChecked(True)

    def _setup_autoscale_shortcuts(self):
        """Assign single-key shortcut for autoscale selection."""
        for sc in getattr(self, "autoscale_shortcuts", []):
            sc.setParent(None)
        self.autoscale_shortcuts = []

        sc = QShortcut(QKeySequence("A"), self)
        sc.activated.connect(self._cycle_autoscale_mode)
        self.autoscale_shortcuts.append(sc)

    def _save_current_gas_action(self) -> None:
        """Save the current gas only when the legend button is active.
        Flashes the legend button yellow during the upsert so the user has
        visible feedback when the chunk is large and the save takes a moment.
        """
        if not self.madechanges:
            return

        # IE3 cal view has no save-legend button to flash; save directly.
        if (self.instrument.inst_id == 'ie3'
                and self.current_run_time
                and '(Cal)' in self.current_run_time):
            self._perform_save_current_gas()
            return

        if getattr(self, "_save2db_text", None) is None:
            return

        # Flash the legend button yellow immediately, then defer the actual
        # work to the next event-loop tick so the repaint lands first.
        bbox = self._save2db_text.get_bbox_patch()
        if bbox:
            bbox.update(dict(facecolor="yellow", edgecolor="none", alpha=0.95))
        self._save2db_text.set_color("black")
        self.canvas.draw_idle()
        self.canvas.flush_events()
        QApplication.processEvents()
        QTimer.singleShot(0, self._perform_save_current_gas)

    def _perform_save_current_gas(self) -> None:
        """Actual save work; called via QTimer so the yellow flash is visible."""

        # IE3 cal run: save flag changes + recompute and store the weekly fit
        if (self.instrument.inst_id == 'ie3'
                and self.current_run_time
                and '(Cal)' in self.current_run_time):
            self._ie3_cal_save()
            self.madechanges = False
            self.smoothing_changed = False
            self._ie3_cal_plot()
            return

        response_id = None

        # 1. Try from selected_calc_curve (if set via combo box)
        if self.selected_calc_curve is not None and isinstance(self.calcurves, pd.DataFrame) and not self.calcurves.empty:
            match = self.calcurves.loc[self.calcurves['run_date'] == self.selected_calc_curve]
            if not match.empty:
                response_id = match['id'].iat[0]

        # 2. If not found, try from _save_payload (if set via selection or other means)
        if response_id is None and self._save_payload is not None:
            response_id = self._save_payload.get('id')

        # 3. If still not found, try to use the existing ng_response_id from the data
        if response_id is None and 'ng_response_id' in self.run.columns:
            val = self.run['ng_response_id'].iat[0]
            if pd.notna(val) and val != 0:
                response_id = int(val)

        if response_id is not None:
            print(f"Saving with response_id: {response_id}")
            self.instrument.upsert_mole_fractions(self.run, response_id=response_id)
        else:
            if self.instrument.inst_id not in ('ie3', 'cats'):
                print("Warning: No calibration ID found. Saving with NULL.")
            self.instrument.upsert_mole_fractions(self.run, response_id=None)

        self.instrument.upsert_calibrations(self.run, self.current_pnum)

        self.madechanges = False
        self.smoothing_changed = False
        # gc_plot rebuilds the legend, so the yellow styling is replaced with
        # the standard idle/changed colors as a side effect.
        self.gc_plot(self._current_yparam, sub_info='SAVED')

    def _setup_save_shortcuts(self):
        """Assign single-key shortcuts for save actions."""
        for sc in getattr(self, "save_shortcuts", []):
            sc.setParent(None)
        self.save_shortcuts = []

        sc = QShortcut(QKeySequence("S"), self)
        sc.activated.connect(self._save_current_gas_action)
        self.save_shortcuts.append(sc)

        # Esc returns from narrowed IE3 views to the selected chunk.
        esc = QShortcut(QKeySequence("Esc"), self)
        esc.activated.connect(self._clear_run_time_zoom)
        self.save_shortcuts.append(esc)

    def _clear_run_time_zoom(self):
        had_zoom = self._zoom_run_time is not None
        had_exact_ie3_run = self._ie3_exact_run_time is not None
        if not had_zoom and not had_exact_ie3_run:
            return
        self._zoom_run_time = None
        self._ie3_exact_run_time = None
        if had_exact_ie3_run:
            self.load_selected_run()
        if not self.run.empty:
            self.update_smoothing_combobox()
            sub = 'returned to chunk' if had_exact_ie3_run else 'unzoomed'
            self.gc_plot(self._current_yparam or 'resp', sub_info=sub)

    def _setup_analyte_shortcuts(self):
        """
        Assign keyboard shortcuts to cycle analytes for both combobox and radio layouts.
        """
        # Clear any existing shortcuts to avoid duplicates
        for sc in getattr(self, "analyte_shortcuts", []):
            sc.setParent(None)
        self.analyte_shortcuts = []

        # Use Ctrl+Shift+Up/Down for reliable cross-platform behavior over SSH
        shortcuts = [
            (["Ctrl+Shift+Up"], self.on_prev_analyte),
            (["Ctrl+Shift+Down"], self.on_next_analyte),
        ]

        for seq_list, handler in shortcuts:
            for seq in seq_list:
                sc = QShortcut(QKeySequence(seq), self)
                sc.activated.connect(handler)
                self.analyte_shortcuts.append(sc)

    def _setup_run_shortcuts(self):
        """
        Assign keyboard shortcuts to cycle run dates via the run combobox.
        """
        for sc in getattr(self, "run_shortcuts", []):
            sc.setParent(None)
        self.run_shortcuts = []

        shortcuts = [
            (["Ctrl+Shift+Left"], self.on_prev_run),
            (["Ctrl+Shift+Right"], self.on_next_run),
        ]

        for seq_list, handler in shortcuts:
            for seq in seq_list:
                sc = QShortcut(QKeySequence(seq), self)
                sc.activated.connect(handler)
                self.run_shortcuts.append(sc)
        
    def load_selected_run(self):
        self._clear_highlight()
        # IE3 cal run: load cal+ref port data for the selected week
        if (self.instrument.inst_id == 'ie3'
                and self.current_run_time
                and '(Cal)' in self.current_run_time):
            week_start = self.current_run_time.split(' (')[0].strip()
            self.run = self.instrument.load_cal_week_data(
                self.current_pnum, self.current_channel, week_start
            )
            # Preselect the fit-method combo to this week's recorded method.
            method = self.instrument.get_week_mf_method(
                self.current_pnum, self.current_channel, week_start
            )
            self._ie3_cal_method = method
            idx = self.fit_method_cb.findData(method)
            if idx >= 0:
                self.fit_method_cb.blockSignals(True)
                self.fit_method_cb.setCurrentIndex(idx)
                self.fit_method_cb.blockSignals(False)
        else:
            # call sql load function from instrument class
            start, end = self._current_load_dates()
            self.run = self.instrument.load_data(
                pnum=self.current_pnum,
                channel=self.current_channel,
                run_type_num=self.run_type_num,
                start_date=start,
                end_date=end
            )

        # Keep any active run_time zoom if the target run is still present in
        # the freshly loaded data (true for analyte switches within the same
        # chunk). gc_plot clears the zoom defensively if the target is gone.

        self.update_smoothing_combobox()
        self.set_runtype_combo()
        self._update_auto_rejected()
        self._update_info_tagged()

        self.madechanges = False

    def _current_load_dates(self):
        """Return (start_date, end_date) to pass to instrument.load_data().
        For IE3 the current_run_time is a chunk label spanning a date range;
        for other instruments it's a single GC run timestamp used for both ends.
        When zoomed into a specific IE3 run_time group, returns that run_time
        so that Save all gases operates only on the visible data.
        """
        if self.instrument.inst_id in ('ie3', 'cats') and self._ie3_exact_run_time is not None:
            rt = pd.Timestamp(self._ie3_exact_run_time).strftime('%Y-%m-%d %H:%M:%S')
            return (rt, rt)
        if self.instrument.inst_id in ('ie3', 'cats') and self._zoom_run_time is not None:
            rt = pd.Timestamp(self._zoom_run_time).strftime('%Y-%m-%d %H:%M:%S')
            return (rt, rt)
        if self.instrument.inst_id in ('ie3', 'cats') and self.current_run_time:
            start, end = _ie3_chunk_sql_range(self.current_run_time)
            if start is not None:
                return (start, end)
        return (self.current_run_time, self.current_run_time)

    def update_smoothing_combobox(self):
        """
        Set the smoothing combobox index based on self.run['detrend_method_num'].
        If not found or invalid, defaults to 'Lowess 5 point (l)'.
        """
        index_to_detrend_method = {
            0: 1,  # Point to Point (p)
            1: 3,  # 2-point moving average
            2: 2,  # Lowess 5 point (l)
            3: 4,  # 3-point boxcar
            4: 6,  # 5-point boxcar
            5: 5,  # Lowess ~10 points
        }
        detrend_to_index = {v: k for k, v in index_to_detrend_method.items() if v is not None}

        self.smoothing_cb.blockSignals(True)
        try:
            if self._zoom_run_time is not None and 'run_time' in self.run.columns:
                rt_col = pd.to_datetime(self.run['run_time'], utc=True, errors='coerce')
                zr = pd.Timestamp(self._zoom_run_time)
                if zr.tzinfo is None:
                    zr = zr.tz_localize('UTC')
                vals = self.run.loc[rt_col == zr, 'detrend_method_num']
                dm = int(vals.iat[0]) if len(vals) else int(self.run["detrend_method_num"].iat[0])
            else:
                dm = int(self.run["detrend_method_num"].iat[0])
            idx = detrend_to_index.get(dm, 2) # default to Lowess ~5 points
            self.smoothing_cb.setCurrentIndex(idx)
        except Exception as e:
            #print(f"Warning: could not set smoothing combobox ({e})")
            self.smoothing_cb.setCurrentIndex(2)
        self.smoothing_cb.blockSignals(False)

    def get_selected_detrend_method(self):
        """
        Return the detrend_method_num corresponding to the current combobox selection.
        """
        index_to_detrend_method = {
            0: 1,  # Point to Point (p)
            1: 3,  # 2-point moving average
            2: 2,  # Lowess 5 point (l)
            3: 4,  # 3-point boxcar
            4: 6,  # 5-point boxcar
            5: 5,  # Lowess ~10 points
        }
        #print(f"Selected detrend method index: {self.smoothing_cb.currentIndex()}, {index_to_detrend_method.get(self.smoothing_cb.currentIndex(), 2)}")
        return index_to_detrend_method.get(self.smoothing_cb.currentIndex(), 2)  # default to Lowess ~5 points

    def set_current_analyte(self, name):
        """
        Check to see if channel is in analyte_name.
        Preserve the current_run_time when switching analytes.
        """
        if hasattr(self, "timeseries_tab") and self.timeseries_tab:
            self.timeseries_tab.set_current_analyte(name)

        self._clear_highlight()
        if self._multi_tag_panel is not None and self._multi_tag_panel.isVisible():
            self._multi_tag_panel.clear_selection()

        if self.current_run_time is None:
            self.set_runlist()
        else:
            self.load_selected_run()
        self.on_plot_type_changed(self.current_plot_type)

    def current_run_type_filter(self):
        """
        Return the run_type_num (int) or None if “All” is selected.
        """
        idx = self.run_type_cb.currentIndex()
        return self.run_type_cb.itemData(idx)

    def current_date_filter(self):
        """
        Return the string label for date filter.
        (In your real code, you’d translate “last-two-weeks” → a concrete date range.)
        """
        return self.date_filter_cb.currentText()
        
    def on_run_type_changed(self):
        self.current_plot_type = 0
        self.set_runlist()
        self.set_calibration_enabled(False)
        self.draw2zero_cb.setChecked(False)
        self.oldcurves_cb.setChecked(False)
        
    def on_run_changed(self, index):
        """
        Called whenever the user picks a different run_time in run_cb. 
        """
        if index < 0 or index >= len(self.current_run_times):
            return
        self.current_run_time = self.current_run_times[index]
        self._ie3_exact_run_time = None
        self._zoom_run_time = None

        self._clear_highlight()
        if self._multi_tag_panel is not None and self._multi_tag_panel.isVisible():
            self._multi_tag_panel.clear_selection()

        # If Calibration plot is active but the new run is not a calibration run,
        # switch to the Response plot to prevent a crash.
        is_cal_plot_selected = self.plot_radio_group.checkedId() == 3
        is_cal_run = "(Cal)" in self.current_run_time
        if is_cal_plot_selected and not is_cal_run:
            self.resp_rb.setChecked(True)
            self.current_plot_type = 0  # set to resp plot type

        self.load_selected_run()
        self._update_calibration_button_state()
        self._update_notes_button_style()
        self.on_plot_type_changed(self.current_plot_type)

    def on_edit_run_notes(self):
        """Handle the 'Edit Run Notes' button click."""
        if not self.current_run_time:
            QMessageBox.warning(self, "No Run Selected", "Please select a run first.")
            return

        # IE3/CATS chunk mode: pick a specific GC run_time inside the loaded chunk first
        if self.instrument.inst_id in ('ie3', 'cats'):
            run_time_str = self._pick_ie3_run_time_for_notes()
            if not run_time_str:
                return
        else:
            # Clean up run_time string (e.g., remove ' (Cal)')
            run_time_str = self.current_run_time.split(" (")[0]

        # 1. Fetch existing notes
        query = (
            "SELECT notes FROM hats.ng_run_notes "
            f"WHERE inst_num = {self.instrument.inst_num} "
            f"AND run_time = '{run_time_str}';"
        )
        result = self.instrument.db.doquery(query)
        current_notes = result[0]['notes'] if result and result[0]['notes'] else ""

        dialog = RunNotesDialog(run_time_str, current_notes, self)

        if dialog.exec_() == QDialog.Accepted:
            new_notes = dialog.get_notes()
        else:
            return # User cancelled

        # 3. Save the new notes
        # Use INSERT ... ON DUPLICATE KEY UPDATE to handle both new and existing notes
        save_sql = """
            INSERT INTO hats.ng_run_notes (inst_num, run_time, notes)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE notes = VALUES(notes);
        """
        try:
            self.instrument.db.doquery(save_sql, [self.instrument.inst_num, run_time_str, new_notes.strip()])
            QToolTip.showText(
                self.mapToGlobal(self.edit_notes_btn.pos()),
                "Notes saved successfully!",
                self.edit_notes_btn
            )
            self._update_notes_button_style()
        except Exception as e:
            QMessageBox.critical(self, "Database Error", f"Failed to save notes: {e}")
            print(f"Error saving run notes: {e}")

    def _pick_ie3_run_time_for_notes(self):
        """Show a sub-picker of GC run_times in the loaded chunk; return the chosen
        timestamp string (YYYY-MM-DD HH:MM:SS) or None if cancelled."""
        if self.run is None or self.run.empty:
            QMessageBox.information(
                self, "No Runs Loaded",
                "No GC runs are loaded for this period. Adjust the date range and try again."
            )
            return None

        run_times = (
            pd.to_datetime(self.run['run_time'], utc=True, errors='coerce')
            .dropna()
            .dt.tz_convert(None)
            .sort_values()
            .unique()
        )
        if len(run_times) == 0:
            QMessageBox.information(self, "No Runs", "No run_times in the current chunk.")
            return None

        run_time_strs = [pd.Timestamp(rt).strftime('%Y-%m-%d %H:%M:%S') for rt in run_times]

        placeholders = ','.join(['%s'] * len(run_time_strs))
        notes_sql = f"""
            SELECT run_time, notes FROM hats.ng_run_notes
            WHERE inst_num = %s AND run_time IN ({placeholders});
        """
        notes_rows = self.instrument.db.doquery(
            notes_sql, [self.instrument.inst_num, *run_time_strs]
        ) or []
        notes_map = {
            pd.Timestamp(r['run_time']).strftime('%Y-%m-%d %H:%M:%S'):
                (r['notes'] or '').strip()
            for r in notes_rows
        }

        dialog = IE3RunPickerDialog(run_time_strs, notes_map, self)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_run_time()
        return None

    def set_runtype_combo(self):
        """Set the combo box to the current run's run type."""
        
        if self.change_runtype_enabled is False:
            return

        if self.run is None or self.run.empty:
            self.clear_plot()
            return
                
        # Get the current run's run_type_num
        current_runtype_num = int(self.run['run_type_num'].iloc[0])

        # Build reverse map {num: name} from the instrument
        num_to_name = {v: k for k, v in self.instrument.run_type_num().items()}

        # Lookup the lowercase name for the current type
        current_runtype_name = num_to_name.get(current_runtype_num, "").lower()
        current_runtype_name = re.sub(r's$', '', current_runtype_name)  # remove trailing 's'
        current_runtype_name = 'calibration' if current_runtype_name == 'cal' else current_runtype_name

        # Find the matching index ignoring case and plural
        index = -1
        for i in range(self.change_runtype_cb.count()):
            item = self.change_runtype_cb.itemText(i).lower()
            item_norm = re.sub(r's$', '', item)  # normalize plural form
            if item_norm == current_runtype_name:
                index = i
                break

        # Set the index if found
        self.change_runtype_cb.setCurrentIndex(index)
        
    def on_change_run_type_db(self):
        """Enable save button only if a new run type is selected."""
        if not self.change_runtype_enabled:
            return

        # Get the selected combo text directly
        text = self.change_runtype_cb.currentText().strip()
        if not text:
            self.save_runtype_btn.setEnabled(False)
            return

        # Get the current run's numeric type
        current_runtype_num = int(self.run['run_type_num'].iloc[0])

        # Reverse map from num → name
        num_to_name = {v: k for k, v in self.instrument.run_type_num().items()}

        # Normalize current run type name
        current_name = num_to_name.get(current_runtype_num, "").lower()
        current_name = re.sub(r's$', '', current_name)
        current_name = 'calibration' if current_name == 'cal' else current_name

        # Normalize selected combo text
        selected_name = text.lower()
        selected_name = re.sub(r's$', '', selected_name)
        selected_name = 'calibration' if selected_name == 'cal' else selected_name

        # Compare normalized names
        enabled = selected_name != current_name
        self.save_runtype_btn.setEnabled(enabled)
        self._style_save_runtype_btn(enabled)

    def _style_save_runtype_btn(self, enabled):
        """Update the Save button color based on its enabled state."""
        if enabled:
            # Light green when enabled
            self.save_runtype_btn.setStyleSheet(
                "QPushButton { background-color: lightgreen; color: black; }"
            )
        else:
            # Default / greyed-out look when disabled
            self.save_runtype_btn.setStyleSheet(
                "QPushButton { background-color: #d3d3d3; color: #666; }"
            )

    def on_save_run_type(self):
        """Update the database with the new run type for the current run."""
        runtime = self.run['run_time'].iloc[0]
        runtime_str = runtime.strftime("%Y-%m-%d %H:%M:%S")

        # Get selected run type text from combo box
        text = self.change_runtype_cb.currentText()
        if not text:
            print("No run type selected.")
            return

        run_type_num = self.instrument.RUN_TYPE_MAP.get(text)
        if run_type_num is None:
            print(f"Unknown run type '{text}', skipping update.")
            return

        sql = f"""
            UPDATE hats.ng_analysis
            SET run_type_num = {run_type_num}
            WHERE inst_num = {self.instrument.inst_num}
            AND run_time = '{runtime_str}';
        """

        self.instrument.db.doquery(sql)
        self.save_runtype_btn.setEnabled(False)
        print(f"Run type updated in database for run_time {runtime_str}'.")
        
        self.set_runlist()
        self.load_selected_run()
        self.on_plot_type_changed(self.current_plot_type)
        self._update_notes_button_style()

    def _update_notes_button_style(self):
        """Update the run notes button text and color based on note existence."""
        if not self.current_run_time:
            self.edit_notes_btn.setText("Add Run Notes")
            self.edit_notes_btn.setStyleSheet("background-color: #d3d3d3;") # lightgrey
            return

        if self.instrument.inst_id in ('ie3', 'cats'):
            # Chunk mode: green if any GC run in the chunk has notes
            if self.run is None or self.run.empty:
                has_note = False
            else:
                rts = (
                    pd.to_datetime(self.run['run_time'], utc=True, errors='coerce')
                    .dropna()
                    .dt.tz_convert(None)
                    .unique()
                )
                if len(rts) == 0:
                    has_note = False
                else:
                    rt_strs = [pd.Timestamp(rt).strftime('%Y-%m-%d %H:%M:%S') for rt in rts]
                    placeholders = ','.join(['%s'] * len(rt_strs))
                    sql = (
                        "SELECT 1 FROM hats.ng_run_notes "
                        f"WHERE inst_num = %s AND run_time IN ({placeholders}) "
                        "AND notes IS NOT NULL AND TRIM(notes) <> '' LIMIT 1;"
                    )
                    rows = self.instrument.db.doquery(
                        sql, [self.instrument.inst_num, *rt_strs]
                    ) or []
                    has_note = bool(rows)
            self.edit_notes_btn.setText(
                "View/Edit Run Notes (n)" if has_note else "Add Run Notes (n)"
            )
            color = "#c3fada" if has_note else "#d3d3d3"
            self.edit_notes_btn.setStyleSheet(f"background-color: {color};")
            return

        run_time_str = self.current_run_time.split(" (")[0]
        query = (
            "SELECT notes FROM hats.ng_run_notes "
            f"WHERE inst_num = {self.instrument.inst_num} AND run_time = '{run_time_str}';"
        )
        result = self.instrument.db.doquery(query)
        has_note = result and result[0]['notes'] and result[0]['notes'].strip()

        self.edit_notes_btn.setText("Edit/View Run Notes (n)" if has_note else "Add Run Notes (n)")
        color = "#c3fada" if has_note else "#d3d3d3" # lightgrey
        self.edit_notes_btn.setStyleSheet(f"background-color: {color};")

    def _update_calibration_button_state(self):
        """Enable the calibration radio button if the current run is a calibration run."""
        cal_num = (getattr(self.instrument, "RUN_TYPE_MAP", {}) or {}).get("Calibrations")
        if cal_num is None:
            self.calibration_rb.setEnabled(False)
            return

        is_cal_type_selected = self.run_type_num == cal_num
        is_cal_run_selected = self.current_run_time and "(Cal)" in self.current_run_time

        if is_cal_type_selected or is_cal_run_selected:
            self.calibration_rb.setEnabled(True)
        else:
            self.calibration_rb.setEnabled(False)
                
    def on_prev_run(self):
        """
        Move the run_cb selection one index backward, if possible.
        """
        self.run_cb.blockSignals(True)
        idx = self.run_cb.currentIndex()
        if idx > 0:
            idx -= 1
            self.run_cb.setCurrentIndex(idx)
            self.on_run_changed(idx)
        self.run_cb.blockSignals(False)
        
    def on_next_run(self):
        """
        Move the run_cb selection one index forward, if possible.
        """
        self.run_cb.blockSignals(True)
        idx = self.run_cb.currentIndex()
        if idx < (self.run_cb.count() - 1):
            idx += 1
            self.run_cb.setCurrentIndex(idx)
            self.on_run_changed(idx)
        self.run_cb.blockSignals(False)
        
    def _copy_run_time(self):
        """Copy the current run_time (without Cal/PFP suffix) to the clipboard."""
        text = self.run_cb.currentText().split(" (")[0].strip()
        if not text:
            return
        QApplication.clipboard().setText(text)
        self._copy_run_time_btn.setText("✓")
        QTimer.singleShot(1200, lambda: self._copy_run_time_btn.setText("⎘"))

    def on_lock_y_axis_toggled(self, state):
        """
        Called when the lock_y_axis_cb checkbox is toggled.
        """
        ax = self.canvas.figure.gca()  # Get the current axis from the canvas
        if state == Qt.Checked:
            self.y_axis_limits = ax.get_ylim()  # Save the current y-axis limits
            #print("Y-Axis scale locked:", self.y_axis_limits)
        else:
            self.y_axis_limits = None  # Clear the saved limits
            #print("Y-Axis scale unlocked.")
        self.on_plot_type_changed(self.current_plot_type)

    def _get_autoscale_mode(self) -> str:
        if self.autoscale_samples_rb.isChecked():
            return "samples"
        if self.autoscale_standard_rb.isChecked():
            return "standard"
        return "fullscale"

    def on_autoscale_mode_changed(self, state):
        """
        Called when the autoscale mode is toggled.
        """
        self.on_plot_type_changed(self.current_plot_type)

    def made_changes(self, event):
        if self.madechanges:
            choice = QMessageBox.question(
                self,  
                "Quit", 
                "Some of the data was modified. Do you want to save changes to fe3_db.csv file?", 
                QMessageBox.Yes | QMessageBox.No)

            if choice == QMessageBox.Yes:
                print("Saving changes here...")
            else:
                pass

    def export_csv(self, button):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        run_selected = pd.to_datetime(self.run['run_time'].iloc[0]).strftime('%Y%m%d-%H%M%S')
        runtype = self.run['run_type_num'].iloc[0]
        default_file = f'{run_selected}_data.csv'

        """
        if self.runTypeCombo.currentText() == 'aircore':
            default_file = f'{run_selected}_data.csv'
        else:    
            default_file = f'{run_selected}_summary.csv'
        """
            
        file, _ = QFileDialog.getSaveFileName(
            self,
            'Save run as CSV file',
            default_file,
            'CSV Files (*.csv);;All Files (*)',
            options=options
        )

        if file:
            self.instrument.export_run_alldata(self.run, file)
        
class RunNotesDialog(QDialog):
    def __init__(self, run_time_str, current_notes="", parent=None):
        super().__init__(parent)

        self.setWindowTitle("Edit Run Notes")
        self.resize(500, 300)

        layout = QVBoxLayout(self)

        # Label
        label = QLabel(f"Notes for run: {run_time_str}")
        layout.addWidget(label)

        # Text editor
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlainText(current_notes)

        # Wrap at widget width
        self.text_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)

        # Prevent tabs from inserting tab characters
        self.text_edit.setTabChangesFocus(True)

        # Move cursor to end (no selection)
        self.text_edit.moveCursor(QTextCursor.End)

        layout.addWidget(self.text_edit)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")

        # Add native icons
        self.save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.cancel_button.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))

        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Connections
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_notes(self):
        return self.text_edit.toPlainText()


class IE3RunPickerDialog(QDialog):
    """Pick a single GC run_time from the loaded chunk to view/edit its notes.
    Run_times that already have notes are marked with a pencil glyph."""

    def __init__(self, run_time_strs, notes_map, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select GC Run for Notes")
        self.resize(380, 420)

        self._run_time_strs = list(run_time_strs)
        self._selected = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select a GC run to view/edit notes (✎ = has notes):"))

        self.list_widget = QListWidget()
        for rt in self._run_time_strs:
            has_note = bool(notes_map.get(rt, "").strip())
            label = f"{rt}    {'✎' if has_note else ''}".rstrip()
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, rt)
            self.list_widget.addItem(item)
        if self.list_widget.count():
            self.list_widget.setCurrentRow(self.list_widget.count() - 1)
        self.list_widget.itemDoubleClicked.connect(lambda _i: self.accept())
        layout.addWidget(self.list_widget)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogOkButton))
        cancel_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def selected_run_time(self):
        item = self.list_widget.currentItem()
        return item.data(Qt.UserRole) if item is not None else None


class LOGOSAIWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, agent: LOGOSChatAgent, prompt: str, context: dict[str, object]):
        super().__init__()
        self.agent = agent
        self.prompt = prompt
        self.context = context

    @QtCore.pyqtSlot()
    def run(self):
        try:
            reply = self.agent.ask(self.prompt, context=self.context)
            self.finished.emit(reply)
        except Exception as exc:
            self.failed.emit(str(exc))


class LOGOSAIDialog(QDialog):
    def __init__(self, main_window, ai_tab):
        super().__init__(main_window)
        self.main_window = main_window
        self.ai_tab = ai_tab
        self.setWindowTitle(f"{main_window.instrument.inst_id.upper()} LOGOS AI")
        self.resize(900, 700)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        ai_tab.setParent(self)
        layout.addWidget(ai_tab)
        ai_tab._render_transcript()
        ai_tab.show()

    def closeEvent(self, event):
        self.main_window.redock_logos_ai_tab()
        event.accept()


class LOGOSAITab(QWidget):
    """Free-form chat UI backed by LOGOSChatAgent."""

    def __init__(self, instrument, main_window=None, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.main_window = main_window
        self.tools = LOGOSDataAgentTools(inst_id=instrument.inst_id)
        self.agent = LOGOSChatAgent(self.tools)
        self._worker_thread = None
        self._worker = None
        self._message_history: list[tuple[str, str]] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setAcceptRichText(True)
        self.transcript.setStyleSheet("""
            QTextEdit {
                background: #fbfbf8;
                border: 1px solid #c7c7bf;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.transcript, stretch=1)

        quick_row = QHBoxLayout()
        for label, prompt in [
            ("Site Info", "Give me a site description for SMO."),
            ("Recent Pairs", "Return the 10 most recent flask pairs filled at SMO."),
            ("Pairs + Met", "What flask type and winds did the last 5 SMO pairs have?"),
            ("Compare Year", "What is the mean CFC-11 value of 2025 at SMO and compare to the most recent flasks?"),
        ]:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _checked=False, text=prompt: self._set_prompt(text))
            quick_row.addWidget(btn)
        quick_row.addStretch()
        self.popout_btn = QPushButton("Pop Out")
        self.popout_btn.clicked.connect(self.toggle_popout)
        quick_row.addWidget(self.popout_btn)
        layout.addLayout(quick_row)

        layout.addSpacing(14)

        prompt_label = QLabel("Ask LOGOS AI")
        prompt_label.setStyleSheet("font-weight: 700; color: #222;")
        layout.addWidget(prompt_label)

        input_row = QHBoxLayout()
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Ask a LOGOS data question...")
        self.prompt_edit.setStyleSheet("""
            QLineEdit {
                border: 2px solid #6a6a62;
                border-radius: 8px;
                padding: 8px 10px;
                background: #fffef8;
            }
            QLineEdit:focus {
                border: 2px solid #1f4f8a;
            }
        """)
        self.prompt_edit.returnPressed.connect(self.submit_prompt)
        input_row.addWidget(self.prompt_edit, stretch=1)

        ask_btn = QPushButton("Ask")
        ask_btn.clicked.connect(self.submit_prompt)
        self.ask_btn = ask_btn
        input_row.addWidget(ask_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_chat)
        input_row.addWidget(clear_btn)
        layout.addLayout(input_row)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

    def _set_prompt(self, text: str):
        self.prompt_edit.setText(text)
        self.prompt_edit.setFocus()

    def _append_message(self, role: str, content: str):
        self._message_history.append((role, content))
        self._render_transcript()

    def _message_html(self, role: str, content: str) -> str:
        if role == "assistant":
            body = self._style_ai_reply(content)
            prefix_html = '<span style="font-weight:700;color:#1f4f8a;">LOGOS AI:</span>'
        else:
            body = html.escape(content).replace("\n", "<br>")
            prefix_html = '<span style="font-weight:700;color:#333;">You:</span>'
        return (
            '<div style="margin: 0 0 12px 0; line-height: 1.45;">'
            f'{prefix_html} '
            f'<span style="color:#222;">{body}</span>'
            '</div>'
        )

    def _render_transcript(self):
        html_blocks = [self._message_html(role, content) for role, content in self._message_history]
        self.transcript.setHtml("".join(html_blocks))
        cursor = self.transcript.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.transcript.setTextCursor(cursor)
        self.transcript.ensureCursorVisible()

    def _style_ai_reply(self, content: str) -> str:
        image_urls: list[str] = []

        def _capture_image(match):
            raw_path = match.group(1).strip()
            image_urls.append(QUrl.fromLocalFile(raw_path).toString())
            return ""

        content = re.sub(r"\[\[image:(.+?)\]\]", _capture_image, content)
        text = html.escape(content).replace("\n", "<br>")

        # Highlight pair IDs in dark blue.
        text = re.sub(
            r"(\bpair(?:\s+id|_id(?:_num)?)?\b\s*[=:]?\s*)(\d+)",
            r'\1<b><font color="#123f73">\2</font></b>',
            text,
            flags=re.IGNORECASE,
        )

        # Highlight sample_ids / flask IDs in blue.
        text = re.sub(
            r"((?:sample_ids?|sample id(?:s)?|flask ids?|flask_id|flask id(?:s)?)\s*[=:]?\s*)(\[[^\]]+\]|[\d,\s]+)",
            lambda m: (
                f'{m.group(1)}'
                f'<b><font color="#2563a6">{m.group(2).strip()}</font></b>'
            ),
            text,
            flags=re.IGNORECASE,
        )

        # Slight emphasis for bullet markers.
        text = text.replace("- ", '<b><font color="#555555">- </font></b>', 1000)
        for image_url in image_urls:
            text += (
                '<div style="margin-top:10px;">'
                f'<img src="{image_url}" width="900" style="border:1px solid #c7c7bf; '
                'border-radius:6px; background:#ffffff;" />'
                '</div>'
            )
        return text

    def _current_context(self) -> dict[str, object]:
        parent = self.main_window or self.parent()
        current_analyte = None
        if hasattr(parent, "analyte_combo") and parent.analyte_combo is not None:
            current_analyte = parent.analyte_combo.currentText()
        return {
            "current_analyte": current_analyte,
            "current_run_time": getattr(parent, "current_run_time", None),
            "current_channel": getattr(parent, "current_channel", None),
            "current_pnum": getattr(parent, "current_pnum", None),
        }

    def _set_busy(self, busy: bool):
        self.prompt_edit.setEnabled(not busy)
        self.ask_btn.setEnabled(not busy)
        self.status_label.setText("Thinking..." if busy else "")
        if not busy:
            QtCore.QTimer.singleShot(0, self.prompt_edit.setFocus)

    def set_popout_state(self, is_detached: bool):
        self.popout_btn.setText("Dock Back" if is_detached else "Pop Out")

    def toggle_popout(self):
        if self.main_window is None:
            return
        if getattr(self.main_window, "logos_ai_dialog", None) is not None:
            self.main_window.redock_logos_ai_tab()
        else:
            self.main_window.undock_logos_ai_tab()

    def clear_chat(self):
        self._message_history.clear()
        self.transcript.clear()
        self.agent.reset_conversation()
        self.status_label.setText("Conversation cleared.")
        self.prompt_edit.setFocus()

    def submit_prompt(self):
        prompt = self.prompt_edit.text().strip()
        if not prompt:
            self.prompt_edit.setFocus()
            return
        if self._worker_thread is not None:
            return
        self._append_message("user", prompt)
        self.prompt_edit.clear()
        self._set_busy(True)

        self._worker_thread = QtCore.QThread(self)
        self._worker = LOGOSAIWorker(self.agent, prompt, self._current_context())
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_agent_reply)
        self._worker.failed.connect(self._on_agent_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _on_agent_reply(self, reply: str):
        self._append_message("assistant", reply)
        self._set_busy(False)

    def _on_agent_error(self, error_text: str):
        self._append_message("assistant", f"Error: {error_text}")
        self._set_busy(False)

    def _cleanup_worker(self):
        if self._worker is not None:
            self._worker.deleteLater()
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
        self._worker = None
        self._worker_thread = None
        self.prompt_edit.setFocus()
    
def get_instrument_for(instrument_id: str, site: str | None = None):
    """
    Look up the Instrument class, instantiate.
    """
    inst = instrument_id.upper()
    
    try:
        instrument_cls = getattr(li, f"{inst}_Instrument")
    except AttributeError:
        raise ValueError(
            f"Could not find class '{inst}_Instrument' in logos_instruments.py"
        )

    if inst in ("IE3", "CATS") and site:
        instrument = instrument_cls(site=site)
    else:
        instrument = instrument_cls()

    return instrument

def main():
    logos_instance = li.LOGOS_Instruments()
    insts = list(logos_instance.INSTRUMENTS.keys())  # valid LOGOS instruments

    parser = argparse.ArgumentParser(description="Data Processing Application")
    parser.add_argument(
        "instrument",
        type=str,
        choices=insts,  # Use the instruments list
        help=f"Specify which instrument {insts} to use (default: m4)"
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Site code (e.g. brw, spo, smo). Used with ie3 and cats.",
    )
    
    args = parser.parse_args()
    
    try:
        instrument = get_instrument_for(args.instrument, site=args.site)
    except ValueError as e:
        print(e)
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #d1d5db;
            border-radius: 10px;
            margin-top: 14px;
            padding: 10px 12px 12px 12px;
            font-weight: 600;
            font-size: 12px;
            color: #2255aa;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 4px;
        }
        QToolTip { background-color: #fff59d; }
    """)
    w = MainWindow(instrument)
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()
