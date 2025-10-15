#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import math
from functools import lru_cache
import warnings
import argparse

from PyQt5 import QtCore
from PyQt5.QtGui import QCursor, QPainter, QPalette, QPen, QStandardItemModel, QStandardItem

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QToolTip,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QTabWidget,
    QLabel, QComboBox, QPushButton, QRadioButton, QAction,
    QButtonGroup, QMessageBox, QSizePolicy, QSpacerItem, QCheckBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer

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
from logos_timeseries import TimeseriesWidget


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
        self.addSeparator()
        self.flag_action = QAction("Tagging", self)
        self.flag_action.setCheckable(True)
        self.flag_action.setShortcut("T")
        self.flag_action.setToolTip("Toggle Tagging mode (T)")
        self.flag_action.toggled.connect(self._toggle_flag)
        self.addAction(self.flag_action)

        acts = self.actions()
        anchor_idx = None
        for i, a in enumerate(acts):
            tip = (a.toolTip() or "").lower()
            text = (a.text() or "").lower()
            icon = (a.iconText() or "").lower()
            if "save" in tip or text == "save" or "save" in icon:
                anchor_idx = i
                break

        if anchor_idx is not None:
            # place *after* Zoom
            self.removeAction(self.flag_action)
            if anchor_idx + 1 < len(acts):
                self.insertAction(acts[anchor_idx + 1], self.flag_action)
                # optional: a tiny separator right after your button
                self.insertSeparator(acts[anchor_idx + 1])
            else:
                self.addAction(self.flag_action)
        else:
            # fallback: put it before Save
            for i, a in enumerate(self.actions()):
                if "save" in (a.toolTip() or "").lower():
                    self.removeAction(self.flag_action)
                    self.insertAction(a, self.flag_action)   # before Save
                    break
                
        self._style_flag_button()

    def _style_flag_button(self):
        # Grab the *current* QToolButton for the action (it changes if you reinsert)
        btn = self.widgetForAction(self.flag_action)
        if btn is None:
            # Defer until the toolbar finishes creating the widget
            QTimer.singleShot(0, self._style_flag_button)
            return

        self.flag_button = btn
        # Ensure it paints a background (autoRaise=True makes it flat/grey on some styles)
        self.flag_button.setAutoRaise(False)
        self.flag_button.setCheckable(True)   # mirrors the action's checkable

        # Style normal vs checked states
        self.flag_button.setStyleSheet("""
            QToolButton {
                background: none;
                padding: 2px 8px;
                border-radius: 6px;
            }
            QToolButton:checked {
                background-color: #2e7d32;  /* green */
                color: white;
                font-weight: 600;
            }
        """)
                
    def zoom(self, *args, **kwargs):
        super().zoom(*args, **kwargs)
        # If we just entered a tool mode, turn Tagging OFF
        if getattr(self, "mode", None):  # non-empty when active
            try:
                self.flag_action.setChecked(False)
            except Exception:
                pass

    def pan(self, *args, **kwargs):
        super().pan(*args, **kwargs)
        if getattr(self, "mode", None):
            try:
                self.flag_action.setChecked(False)
            except Exception:
                pass

    def _toggle_flag(self, checked):
        if self.on_flag_toggle:
            self.on_flag_toggle(checked)
            
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
        
class MainWindow(QMainWindow):
    
    def __init__(self, instrument):
        # Notice: we call super().__init__(instrument=instrument_id) inside HATS_DB_Functions
        super().__init__()
        self.instrument = instrument   # e.g. an M4_Instrument("m4") instance

        self.setWindowTitle(f"{self.instrument.inst_id.upper()} Data Processing Application")

        # Keep track of analytes and run_times
        self.analytes     = self.instrument.analytes
        self.current_pnum = None
        self.current_channel = None  # channel is optional, e.g. for FE3
        self.current_run_times = []   # will be a sorted list of QDateTime strings
        self.current_run_time = None  # currently selected run_time (QDateTime string)
        self.run = pd.DataFrame()  # Placeholder for loaded data
        self.y_axis_limits = None  # Store y-axis limits when locked
        self.toggle_grid_cb = None  # Initialize toggle_grid_cb to avoid AttributeError
        self.lock_y_axis_cb = None  # Initialize lock_y_axis_cb to avoid AttributeError
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

        self.tagging_enabled = False
        self._rect_selector = None
        self._pick_cid = None
        self._scatter_main = []
        self._current_yparam = None
        self._current_yvar = None
        self._pick_refresh = False
        self._pending_xlim = None
        self._pending_ylim = None
        self.madechanges = False
        self.tabs = None
        
        self._save_payload = None       # data for the Save Cal2DB button

        # Set up the UI
        self.toggle_grid_cb = QCheckBox("Toggle Grid")  # Initialize toggle_grid_cb
        self.toggle_grid_cb.setChecked(True)  # Default to showing grid
        self.toggle_grid_cb.stateChanged.connect(self.on_toggle_grid_toggled)

        self.lock_y_axis_cb = QCheckBox("Lock Y-Axis Scale")  # Initialize lock_y_axis_cb
        self.lock_y_axis_cb.setChecked(False)  # Default to unlocked
        self.lock_y_axis_cb.stateChanged.connect(self.on_lock_y_axis_toggled)
        self.run_type_num = None  # Will hold the current run_type_num for filtering

        self.init_ui()

    def init_ui(self):
        # Central widget + top‐level layout
        central = QWidget()
        self.setCentralWidget(central)
        h_main = QHBoxLayout()
        central.setLayout(h_main)

        # Create a matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.current_plot_type = 0

        # ── PROCESSING PANE ──
        processing_pane = QWidget()
        processing_pane.setMinimumWidth(420)
        processing_pane.setMaximumWidth(420)
        processing_layout = QVBoxLayout()
        processing_layout.setContentsMargins(4, 4, 4, 4)
        processing_layout.setSpacing(6)
        processing_pane.setLayout(processing_layout)

        # ── DATE RANGE SELECTION ──
        date_gb = QGroupBox("Date Range (by Month)")
        date_layout = QHBoxLayout()
        date_layout.setContentsMargins(2, 2, 2, 2)
        date_layout.setSpacing(4)
        date_gb.setLayout(date_layout)

        # Start: Year / Month
        self.start_year_cb = QComboBox()
        self.start_month_cb = QComboBox()
        current_year = datetime.now().year
        current_month = datetime.now().month
        start_year = (datetime.now() - timedelta(days=30)).year
        start_month = (datetime.now() - timedelta(days=30)).month
        for y in range(int(self.instrument.start_date[0:4]), current_year + 1):
            self.start_year_cb.addItem(str(y))
        for m in range(1, 13):
            self.start_month_cb.addItem(datetime(2000, m, 1).strftime("%b"))

        # End: Year / Month
        self.end_year_cb = QComboBox()
        self.end_month_cb = QComboBox()
        for y in range(int(self.instrument.start_date[0:4]), current_year + 1):
            self.end_year_cb.addItem(str(y))
        for m in range(1, 13):
            self.end_month_cb.addItem(datetime(2000, m, 1).strftime("%b"))

        self.end_year_cb.setCurrentText(str(current_year))
        self.end_month_cb.setCurrentIndex(current_month - 1)
        self.start_year_cb.setCurrentText(str(start_year))
        self.start_month_cb.setCurrentIndex(int(start_month) - 1)

        # Apply button
        self.apply_date_btn = QPushButton("Apply ▶")
        self.apply_date_btn.clicked.connect(self.apply_dates)

        # Add to layout
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_year_cb)
        date_layout.addWidget(self.start_month_cb)
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_year_cb)
        date_layout.addWidget(self.end_month_cb)
        date_layout.addWidget(self.apply_date_btn)

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
        run_gb = QGroupBox("Run Selection")
        run_layout = QVBoxLayout()
        run_layout.setSpacing(6)
        run_gb.setLayout(run_layout)
        run_layout.addWidget(date_gb)

        # Actual run_time selector + Prev/Next buttons
        runsel_hbox = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.clicked.connect(self.on_prev_run)
        self.next_btn = QPushButton("▶")
        self.next_btn.clicked.connect(self.on_next_run)
        self.run_cb = QComboBox()
        self.run_cb.setMinimumWidth(200)
        self.run_cb.currentIndexChanged.connect(self.on_run_changed)

        runsel_hbox.addWidget(self.prev_btn)
        runsel_hbox.addWidget(self.run_cb, stretch=1)
        runsel_hbox.addWidget(self.next_btn)
        run_layout.addLayout(runsel_hbox)
        run_layout.addLayout(runtype_row)

        processing_layout.addWidget(run_gb)

        # Molecule/Analyte Selection GroupBox
        analyte_gb = QGroupBox("Analyte Selection")
        analyte_layout = QVBoxLayout()
        analyte_layout.setSpacing(6)
        analyte_gb.setLayout(analyte_layout)

        self.analyte_widget = QWidget()
        self.analyte_layout = QGridLayout()
        self.analyte_layout.setSpacing(4)
        self.analyte_widget.setLayout(self.analyte_layout)

        # If there are more than 12 analytes, we’ll switch to a QComboBox below.
        self.populate_analyte_controls()
        analyte_layout.addWidget(self.analyte_widget)

        processing_layout.addWidget(analyte_gb)

        # Plot Type Selection GroupBox
        plot_gb = QGroupBox("Plot Type Selection")
        self.plot_layout = QVBoxLayout()
        self.plot_layout.setSpacing(6)
        plot_gb.setLayout(self.plot_layout)

        self.plot_radio_group = QButtonGroup(self)
        self.resp_rb = QRadioButton("Response")
        ratio_rb = QRadioButton("Ratio")
        mole_fraction_rb = QRadioButton("Mole Fraction")
        self.calibration_rb = QRadioButton("Calibration")
        self.calibration_rb.setEnabled(False)
        self.draw2zero_cb = QCheckBox("Zero")
        self.oldcurves_cb = QCheckBox("Other Curves")

        self.plot_layout.addWidget(self.resp_rb)
        self.plot_layout.addWidget(ratio_rb)
        self.plot_layout.addWidget(mole_fraction_rb)
        
        self.calcurve_label = QLabel("Cal Date")
        self.calcurve_combo = QComboBox()
        self.calcurve_label.setVisible(False)   # hidden initially
        self.calcurve_combo.setVisible(False)  
        self.plot_layout.addWidget(self.calcurve_label)
        self.plot_layout.addWidget(self.calcurve_combo)
        self.calcurve_combo.currentIndexChanged.connect(self.on_calcurve_selected)

        # Fit method combo for Calibration row ---
        self.fit_method_cb = QComboBox()
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
        self.plot_radio_group.addButton(ratio_rb, id=1)
        self.plot_radio_group.addButton(mole_fraction_rb, id=2)
        self.plot_radio_group.addButton(self.calibration_rb, id=3)
        self.resp_rb.setChecked(True)
        self.plot_radio_group.idClicked[int].connect(self.on_plot_type_changed)

        # Options GroupBox
        options_gb = QGroupBox("Options")
        options_layout = QVBoxLayout()
        options_layout.setSpacing(6)
        options_gb.setLayout(options_layout)

        # --- Response smoothing combobox ---
        self.smoothing_label = QLabel("Response smoothing:")
        self.smoothing_cb = QComboBox()
        self.smoothing_cb.addItems([
            "Point to Point",          # maps to 1
            "Lowess %10",              # maps to 4
            "Lowess %20",              # maps to 5
            "Lowess %30 (default)",    # maps to 2
            "Lowess %40",              # maps to 7
            "Lowess %50",              # maps to 8
        ])
        self.smoothing_cb.setCurrentIndex(3)  # show "Lowess %30 (default)"

        options_layout.addWidget(self.smoothing_label)
        options_layout.addWidget(self.smoothing_cb)
        self.smoothing_cb.currentIndexChanged.connect(self.on_smoothing_changed)

        # --- Horizontal separator ---
        options_layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))  # space above
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        options_layout.addWidget(line)
        options_layout.addSpacerItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))  # space below

        self.lock_y_axis_cb = QCheckBox("Lock Y-Axis Scale")
        self.lock_y_axis_cb.setChecked(False)
        self.lock_y_axis_cb.stateChanged.connect(self.on_lock_y_axis_toggled)
        options_layout.addWidget(self.lock_y_axis_cb)

        self.toggle_grid_cb = QCheckBox("Toggle Grid")  # Properly initialize toggle_grid_cb
        self.toggle_grid_cb.setChecked(True)  # Default to showing grid
        self.toggle_grid_cb.stateChanged.connect(self.on_toggle_grid_toggled)
        options_layout.addWidget(self.toggle_grid_cb)

        # Combine plot_gb and options_gb into a single group box
        combined_gb = QGroupBox("Plot and Options")
        combined_layout = QHBoxLayout()
        combined_layout.setSpacing(12)
        combined_gb.setLayout(combined_layout)

        combined_layout.addWidget(plot_gb, stretch=1)
        combined_layout.addWidget(options_gb, stretch=1)

        processing_layout.addWidget(combined_gb)

        # Stretch to push everything to the top
        processing_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # ── TABS ──
        tabs = QTabWidget()
        tabs.addTab(processing_pane, "Processing")

        self.timeseries_tab = TimeseriesWidget(instrument=self.instrument, parent=self)
        tabs.addTab(self.timeseries_tab, "Timeseries")
        self.tabs = tabs
    
        # Right pane: matplotlib figure for plotting
        right_placeholder = QGroupBox("Plot Area")
        right_layout = QVBoxLayout()
        right_placeholder.setLayout(right_layout)
        right_layout.addWidget(self.canvas)

        # Add a NavigationToolbar for the figure
        self.toolbar = FastNavigationToolbar(self.canvas, self, on_flag_toggle=self.on_flag_mode_toggled)  # Pass self explicitly
        right_layout.addWidget(self.toolbar)

        # Add both panes to the main hbox
        h_main.addWidget(tabs, stretch=0)  # Fixed width for left pane
        h_main.addWidget(right_placeholder, stretch=1)  # Flexible width for right pane

        # Kick off by selecting the first analyte by default
        if self.instrument.inst_id == 'm4':
            self.current_pnum = 20
            self.set_current_analyte('HFC134a')

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

    def apply_dates(self):
        self.apply_date_btn.setStyleSheet("")
        self.resp_rb.setChecked(True)
        self.on_run_type_changed()

    def on_fit_method_changed(self, _idx: int):
        self.current_fit_degree = int(self.fit_method_cb.currentData())  # 1/2/3
        # If you're on the Calibration view, you can re-render immediately:
        if self.plot_radio_group.checkedId() == 3:
            self.on_plot_type_changed(3)
            
    def on_smoothing_changed(self, idx: int):
        #print(f"Smoothing changed to index: {idx} get_selected_detrend_method = {self.get_selected_detrend_method()}", )
        self.run['detrend_method_num'] = self.get_selected_detrend_method()
        self.run = self.instrument.norm.merge_smoothed_data(self.run)
        self.run = self.instrument.calc_mole_fraction(self.run)
        self.madechanges = True
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

    def _fmt_gc_plot(self, x, y):
        return f"x={mdates.num2date(x).strftime('%Y-%m-%d %H:%M')}  y={y:0.3g}"

    def _fmt_cal_plot(self, x, y):
        return f"x={x:0.3g}  y={y:0.3g}"

    def gc_plot(self, yparam='resp', sub_info=''):
        """
        Plot data with the legend sorted by analysis_datetime.
        """
        if self.run.empty:
            print("No data available for plotting.")
            return

        if self.run.empty:
            print(f"No data for run_time: {self.current_run_time}")
            return

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
            if self.instrument.inst_id == 'fe3':
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
            self.run['data_flag_int'] if 'data_flag_int' in self.run.columns
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

            stats = stats_map.get(port)
            if stats is not None:
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
        for port, subset in self.run.groupby('port_idx'):
            marker = subset['port_marker'].iloc[0]
            color = subset['port_color']
            scatter = ax.scatter(
                subset['analysis_datetime'],
                subset[yvar],
                marker=marker,
                c=color,
                s=68,
                edgecolors='none',
                zorder=1,
                picker=True,
                pickradius=7
            )
        
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

                "sample_id": subset["sample_id"].astype(str).tolist() if "sample_id" in subset else [""] * len(subset),
                "pair_id": subset["pair_id_num"].astype(str).tolist() if "pair_id_num" in subset else [""] * len(subset),
                "port_info": subset["port_info"].astype(str).tolist() if "port_info" in subset else [""] * len(subset),
            }
            self._scatter_main.append(scatter)
        
        # overlay: show data_flag characters on top of flagged points
        flags = self.run['data_flag_int'] != 0   # adjust if you use a different flag condition
        for x, y, flag in zip(
                self.run.loc[flags, 'analysis_datetime'],
                self.run.loc[flags, yvar],
                self.run.loc[flags, 'data_flag']  # assumes this column has the characters
            ):
            ax.text(
                x, y,
                str('X'),
                color='white', fontsize=9, fontweight='bold',
                ha='center', va='center',
                zorder=4, picker=False
            )
    
        if yparam == 'resp':
            ax.plot(self.run['analysis_datetime'], self.run['smoothed'], color='black', linewidth=0.5, label='Loess-Smooth')
            
        main = f"{self.current_run_time} - {tlabel}: {self.instrument.analytes_inv[self.current_pnum]} ({self.current_pnum})"
        ax.set_title(main, pad=12)
        if sub_info:
            ax.text(
                0.5, .98, sub_info,
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

        if self.toggle_grid_cb.isChecked():
            ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.8)
        else:
            ax.grid(False)

        # add cal curve date selector
        if (self.instrument.inst_id == 'fe3') & (yparam == 'mole_fraction'):
            self.calcurve_label.setVisible(True)
            self.calcurve_combo.setVisible(True)
            self.populate_calcurve_combo(current_curve_date)

        # Cal curve date and age (if available)
        if current_curve_date:
            cal_delta_time = self.run['analysis_datetime'].min() - pd.to_datetime(current_curve_date, utc=True)
            l = current_curve_date.strftime('\nCal Date:\n%Y-%m-%d %H:%M\n') + f'{cal_delta_time.days} days ago'
            legend_handles.append(Line2D([], [], linestyle='None', label=l))

        # --- Always append Save/Revert legend "buttons" ---
        spacer_handle     = Line2D([], [], linestyle='None', label='\u2009')
        save2db_handle    = Line2D([], [], linestyle='None', label='Save current gas')
        save2dball_handle = Line2D([], [], linestyle='None', label='Save all gases')
        revert_handle     = Line2D([], [], linestyle='None', label='Revert changes')

        legend_handles.extend([
            spacer_handle,
            save2db_handle,
            spacer_handle,
            save2dball_handle,
            spacer_handle,
            revert_handle
        ])

        # Put legend outside and create it ONCE
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])

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
            if t == 'Save current gas':
                self._save2db_text = txt
                txt.set_picker(True)
                txt.set_color('white')
                txt.set_bbox(dict(
                    boxstyle='round,pad=0.4',
                    facecolor=('#2e7d32' if self.madechanges else '#9e9e9e'),
                    edgecolor='none', alpha=0.95
                ))
            elif t == 'Save all gases':
                self._save2dball_text = txt
                txt.set_picker(True)
                txt.set_color('white')
                txt.set_bbox(dict(
                    boxstyle='round,pad=0.4',
                    facecolor=('#2e7d32' if self.madechanges else '#9e9e9e'),
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

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.format_coord = self._fmt_gc_plot
 
        if not hasattr(self, "_legend_pick_cid") or self._legend_pick_cid is None:
            self._legend_pick_cid = self.canvas.mpl_connect(
                "pick_event", self._on_legend_pick
            )

        if self.lock_y_axis_cb.isChecked():
            # use the stored y-axis limits
            if self.y_axis_limits is None:
                # If no limits are set, use the current y-limits
                self.y_axis_limits = ax.get_ylim()
            else:
                ax.set_ylim(self.y_axis_limits)
        else:
            try:
                ax.set_ylim(
                    self.run[yvar].min() * 0.95,
                    self.run[yvar].max() * 1.05
                )
            except ValueError:
                pass  # In case of empty data, do not set limits
        
        # ---- Restore view if requested by a prior click ----
        if getattr(self, "_pending_xlim", None) is not None:
            ax.set_xlim(self._pending_xlim)
        if getattr(self, "_pending_ylim", None) is not None:
            ax.set_ylim(self._pending_ylim)
        self._pending_xlim = None
        self._pending_ylim = None

        # Make sure the rectangle selector attaches to the current axes
        self._reattach_rect_selector()

        self.canvas.draw_idle()

        if self._pick_cid is None:
            self._pick_cid = self.canvas.mpl_connect('pick_event', self._on_pick_point)

        # Connect tooltip click handler
        if not hasattr(self, "_click_tooltip_cid"):
            self._click_tooltip_cid = self.canvas.mpl_connect("button_press_event", self._on_click_tooltip)
                
    def _reattach_rect_selector(self):
        """Ensure RectangleSelector follows the current axes after a redraw."""
        if not hasattr(self, "_rect_selector"):
            return

        if self._rect_selector is not None:
            # Kill the old selector
            self._rect_selector.set_active(False)
            self._rect_selector.disconnect_events()
            self._rect_selector = None

        # Recreate on the fresh axes
        if self.figure.axes and self.tagging_enabled:
            ax = self.figure.axes[0]
            from matplotlib.widgets import RectangleSelector
            self._rect_selector = RectangleSelector(
                ax=ax,
                onselect=self._on_box_select,
                useblit=True,
                button=[1],             # left mouse
                minspanx=5, minspany=5,
                spancoords="pixels",
                drag_from_anywhere=True,
                ignore_event_outside=False

            )
            self._rect_selector.set_active(True)
                    
    def _style_gc_buttons(self):
        """
        Placeholder for consistency with calibration_plot.
        In gc_plot, buttons are added/removed dynamically,
        so this just redraws the canvas if needed.
        """
        if getattr(self, "canvas", None):
            self.canvas.draw_idle()

    def populate_calcurve_combo(self, current_curve):
        self.calcurve_combo.blockSignals(True)
        self.calcurve_combo.clear()

        # parse the current run time. This handles "(Cal)" in the string
        sel_time = pd.to_datetime(self.current_run_time.split(' (')[0])
        self.calcurves = self.instrument.load_calcurves(
            self.current_pnum,
            self.current_channel,
            sel_time - pd.DateOffset(months=3)
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
            current_str = pd.to_datetime(current_curve).strftime('%Y-%m-%d %H:%M:%S')
            idx = self.calcurve_combo.findText(current_str)
            if idx != -1:
                self.calcurve_combo.setCurrentIndex(idx)

        self.calcurve_combo.blockSignals(False)    
    
    def _on_pick_point(self, event):
        tb = getattr(self.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", None):
            return
        if not self.tagging_enabled or event.artist is not self._scatter_main:
            return

        inds = np.asarray(event.ind, dtype=int)
        if inds.size == 0:
            return

        mx, my = event.mouseevent.xdata, event.mouseevent.ydata
        if mx is None or my is None:
            i = inds[0]
        else:
            x_all = self._x_num
            y_all = self.run[self._current_yvar].to_numpy()
            dx = x_all[inds] - mx
            dy = y_all[inds] - my
            dx = pd.to_numeric(dx, errors="coerce")
            dy = pd.to_numeric(dy, errors="coerce")
            ok = np.isfinite(dx) & np.isfinite(dy)
            if not ok.any():
                return
            i = inds[ok][np.argmin(dx[ok] ** 2 + dy[ok] ** 2)]

        row_idx = self.run.index[i]
        self._toggle_flags([row_idx])

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
            sample_id = meta.get("sample_id", [None])[nearest_idx]
            pair_id = meta.get("pair_id", [None])[nearest_idx]

            parts = []

            # Site — show if not blank/None
            if site not in (None, "", "nan", "None"):
                parts.append(f"<b>Site:</b> {site}")

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

            parts.append(f"<b>Port Info:</b> {meta.get('port_info', [''])[nearest_idx]}")
            # Analysis time — always shown
            parts.append(f"<b>Analysis time:</b> {analysis_time}")

            # Combine for tooltip
            text = "<br>".join(parts)
            QToolTip.showText(QCursor.pos(), text)
            break

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

    def _toggle_flags(self, idxs):
        """Toggle flags for one or more points given by DataFrame indices."""
        if idxs is None or len(idxs) == 0:
            return

        # Ensure flag columns exist
        if "data_flag_int" not in self.run.columns:
            self.run["data_flag_int"] = 0
        if "data_flag" not in self.run.columns:
            self.run["data_flag"] = ""

        for row_idx in idxs:
            cur = int(self.run.at[row_idx, "data_flag_int"]) if not pd.isna(self.run.at[row_idx, "data_flag_int"]) else 0
            new_val = 0 if cur else 1
            self.run.at[row_idx, "data_flag_int"] = new_val
            self.run.at[row_idx, "data_flag"] = 'M..' if new_val else '...'

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
            print("Selected calibration curve:", row['run_date'])
            self.selected_calc_curve = row['run_date']
            print(f"{self.calcurves.loc[self.calcurves['run_date'] == row['run_date']]}")

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
            & (self.run['data_flag_int'] != 1)
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
                self.instrument.invert_poly_to_mf(y, a0, a1, a2, a3, mf_min=0.0, mf_max=3000)
                for y in ref_estimate['normalized_resp']
            ]
        )

        return ref_estimate
                        
    def calibration_plot(self):
        """
        Plot data with the legend sorted by analysis_datetime.
        Adds a residuals (diff_y) panel above the main plot that shares the x-axis.
        """
        if self.run.empty:
            print("No data available for plotting.")
            return

        # filter for run_time selected in run_cb
        ts_str = self.current_run_time.split(" (")[0]
        sel = pd.to_datetime(ts_str, utc=True)
        if self.run.empty:
            print(f"No data for run_time: {self.current_run_time}")
            return

        mask = self.run['port'].eq(self.instrument.STANDARD_PORT_NUM)
        if not mask.any():  # no rows match
            print(f"No STANDARD_PORT rows for run_time: {self.current_run_time}")
            self.gc_plot('resp')
            return
        
        sel_rt = self.run['run_time'].iat[0]  # current selected run_time (UTC-aware)

        # ── Load & normalize curves ───────────────────────────────────────────────────
        earliest_time = self.run['run_time'].min() - pd.DateOffset(months=6)
        curves = self.instrument.load_calcurves(self.current_pnum, self.current_channel, earliest_time)
        REQ_COLS = ["run_date","serial_number","coef3","coef2","coef1","coef0","function","flag"]

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

        # file in scale_assignment values for calibration tanks in self.run
        self.populate_cal_mf()
        new_fit = self.instrument._fit_row_for_current_run(self.run, order=self.current_fit_degree)
        # save new fit info for Save Cal2DB button
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
            calcurve_exists = False
        elif not has_current:
            # Existing curves, but not for this run_time → append one row
            curves = pd.concat([curves, pd.DataFrame([new_fit])], ignore_index=True)
            curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True, errors='coerce')
            calcurve_exists = False
        else:
            current_cal_flag = curves.loc[curves['run_time'] == sel_rt, 'flag'].iat[0]
            
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
            x_one2one = np.linspace(-.02, mx_cal * 1.05, 200)
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

        # Main scatter (all unflagged + flagged, same colors)
        for port, subset in self.run.groupby('port_idx'):
            marker = subset['port_marker'].iloc[0]
            color = subset['port_color']

            ax.scatter(
                subset['cal_mf'],
                subset[yvar],
                marker=marker,
                c=color,
                zorder=1,
            )

        # Mask for flagged subset
        flags = (
            self.run['data_flag_int'].fillna(0).astype(int) != 0
        ) & mask_main
        
        # Overlay flagged points with white "X"
        ax.scatter(
            self.run.loc[flags, 'cal_mf'],
            self.run.loc[flags, yvar],
            marker='x',
            c='white',
            s=60,  # size of the marker
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

        # Flagged residuals (white X overlay)
        ax_resid.scatter(
            self.run.loc[mask_main & flags, 'cal_mf'],
            self.run.loc[mask_main & flags, 'diff_y'],
            marker='x', c='white',
            s=60, linewidths=2,
            zorder=4
        )
        # Horizontal zero line
        ax_resid.axhline(0.0, lw=1, ls='--', color='0.4')

        # set symmetric y-limits for residuals
        if np.isfinite(diff_y).any():
            maxabs = np.nanmax(np.abs(diff_y))
            if np.isfinite(maxabs) and maxabs > 0:
                ax_resid.set_ylim(-1.1 * maxabs, 1.1 * maxabs)

        # Extend to zero if checked
        if self.draw2zero_cb.isChecked():
            xgrid = np.linspace(-0.02, mx_cal * 1.05, 300)
        else:
            xgrid = np.linspace(mn_cal * .95, mx_cal * 1.05, 300)
            
        # Original curve from DB (black, thinner)
        if calcurve_exists:
            ygrid = np.polyval(stored_coefs, xgrid)
            ax.plot(xgrid, ygrid, linewidth=2, color='black', alpha=0.7,
                            label=row['run_date'].strftime('%Y-%m-%d'))
            legend_handles.append(Line2D([], [], color='black', linewidth=3, label=f"Fit {row['run_date'].strftime('%Y-%m-%d')}"))
            
        if self.oldcurves_cb.isChecked():
            #print(curves)
            # plot stored cal curves
            for row in curves[0:6].itertuples():
                coefs = [row.coef3, row.coef2, row.coef1, row.coef0]
                flagged = int(row.flag)
                ygrid = np.polyval(coefs, xgrid)
                if flagged == 1:
                    ax.plot(xgrid, ygrid, linewidth=1, color='red', linestyle=':', alpha=0.7)
                else:
                    ax.plot(xgrid, ygrid, linewidth=1, color='red', linestyle='-', alpha=0.7)
            legend_handles.append(Line2D([], [], color='red', linewidth=3, label=f"Other fits"))
        
        # potentially a new fit (green, thicker)
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
        
        # Grid toggle applies to both axes
        if self.toggle_grid_cb.isChecked():
            for _ax in (ax_resid, ax):
                _ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.8)
        else:
            for _ax in (ax_resid, ax):
                _ax.grid(False)

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
    
        # Optional y-axis locking for the main axis only
        if self.lock_y_axis_cb.isChecked():
            if self.y_axis_limits is None:
                self.y_axis_limits = ax.get_ylim()
            else:
                ax.set_ylim(self.y_axis_limits)
        else:
            try:
                if self.draw2zero_cb.isChecked():
                    ax.set_ylim(0, 1.2)
                    ax.set_xlim(0, x_all.max() * 1.05)
                else:
                    ax.set_ylim(y_all.min() * 0.95, y_all.max() * 1.05)
                    ax.set_xlim(x_all.min() * 0.95, x_all.max() * 1.05)
            except ValueError:
                pass

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

        @lru_cache(maxsize=None)
        def get_scale(tank_id: str):
            rec = self.instrument.scale_assignments(tank_id, self.current_pnum)
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
                coef0 = VALUES(coef0),
                coef1 = VALUES(coef1),
                coef2 = VALUES(coef2),
                coef3 = VALUES(coef3),
                flag  = VALUES(flag),
                function = VALUES(function)
            """
        params = [payload.get(c) for c in fields]
        #print(sql); print(params)
        self.instrument.db.doquery(sql, params)
        self.calibration_plot()
          
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
            if not self.madechanges:
                return
            #print(f"Save 2DB clicked")
            if self.selected_calc_curve is None:
                self.instrument.upsert_mole_fractions(self.run)
            else:    
                id = self.calcurves.loc[self.calcurves['run_date'] == self.selected_calc_curve]['id'].iat[0]
                self.instrument.upsert_mole_fractions(self.run, response_id=id)
            self.madechanges = False
            self.gc_plot(self._current_yparam, sub_info='SAVED')
        elif art is self._save2dball_text:
            if not self.madechanges:
                return
            print(f"Save flags to all gases clicked")
            
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
            print("Revert clicked")
            self.madechanges = False
            self.load_selected_run()
            self.gc_plot(self._current_yparam, sub_info='REVERTED')

    def update_all_analytes(self):
            pnum = self.current_pnum
            ch = self.current_channel
            self.instrument.update_flags_all_analytes(self.run)

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
                    start_date=self.current_run_time,
                    end_date=self.current_run_time,
                    verbose=False
                )
                run = self.instrument.calc_mole_fraction(run)
                self.instrument.upsert_mole_fractions(run)
            
            # reload current gas to refresh display
            self.run = self.instrument.load_data(
                    pnum=pnum,
                    channel=ch,
                    start_date=self.current_run_time,
                    end_date=self.current_run_time,
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
        # Read selection from the four combo boxes
        sy = self.start_year_cb.currentText()
        sm = self.start_month_cb.currentIndex() + 1   # Jan→1, Feb→2, etc.
        ey = self.end_year_cb.currentText()
        em = self.end_month_cb.currentIndex() + 1

        start_sql = f"{sy}-{sm:02d}-01"
        end_sql = f"{ey}-{em:02d}-31"
        return start_sql, end_sql
    
    def set_runlist(self, initial_date=None):
        t0, t1 = self.get_load_range()

        # If runTypeCombo is set, filter the data by run_type_num
        run_type = self.runTypeCombo.currentText()
        self.run_type_num = self.instrument.RUN_TYPE_MAP.get(run_type, None)
        
        cal_num = (getattr(self.instrument, "RUN_TYPE_MAP", {}) or {}).get("Calibrations")
        if self.run_type_num == cal_num:
            self.calibration_rb.setEnabled(True)
        else:
            self.calibration_rb.setEnabled(False)
        if cal_num is None:
            self.calibration_rb.setEnabled(False)
        
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
        
        # Fill the run_cb combo with these run_time strings
        self.run_cb.blockSignals(True)
        self.run_cb.clear()
        for s in self.current_run_times:
            self.run_cb.addItem(s)

        # Preserve the current_run_time if it exists in the new analyte's run_times
        if self.current_run_time in self.current_run_times:
            idx = self.current_run_times.index(self.current_run_time)
            self.run_cb.setCurrentIndex(idx)
        elif self.current_run_times:
            # Default to the last run_time if the current_run_time is not found
            last_idx = len(self.current_run_times) - 1
            self.run_cb.setCurrentIndex(last_idx)

        if str(initial_date) in self.current_run_times:
            idx = self.current_run_times.index(str(initial_date))
            self.run_cb.setCurrentIndex(idx)
            self.current_run_time = self.current_run_times[idx] 
        else:
            # Default to the last run_time if no initial_date provided
            self.current_run_time = self.current_run_times[-1]

        self.run_cb.blockSignals(False)
        
        self.load_selected_run()
        self.gc_plot('resp')
            
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

            # Select the first radio button by default
            buttons = self.radio_group.buttons()
            if buttons:
                buttons[10].setChecked(True)

        else:
            # Use a QComboBox
            self.analyte_combo = QComboBox()
            for name in names:
                self.analyte_combo.addItem(name)
            self.analyte_combo.currentTextChanged.connect(self.on_analyte_combo_changed)
            self.analyte_layout.addWidget(self.analyte_combo, 0, 0)

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
        
    def load_selected_run(self):
        # call sql load function from instrument class
        # all of the input parameters are set with UI controls.
        self.run = self.instrument.load_data(
            pnum=self.current_pnum,
            channel=self.current_channel,
            run_type_num=self.run_type_num,
            start_date=self.current_run_time,
            end_date=self.current_run_time
        )

        self.update_smoothing_combobox()

        self.madechanges = False

    def update_smoothing_combobox(self):
        """
        Set the smoothing combobox index based on self.run['detrend_method_num'].
        If not found or invalid, defaults to 'Lowess %30 (default)'.
        """
        index_to_detrend_method = {
            0: 1,  # Point to Point
            1: 4,  # Lowess 0.1
            2: 5,  # Lowess 0.2
            3: 2,  # Lowess 0.3 (default)
            4: 7,  # Lowess 0.4
            5: 8,  # Lowess 0.5
        }
        detrend_to_index = {v: k for k, v in index_to_detrend_method.items()}

        self.smoothing_cb.blockSignals(True)
        try:
            dm = int(self.run["detrend_method_num"].iat[0])
            idx = detrend_to_index.get(dm, 3)
            self.smoothing_cb.setCurrentIndex(idx)
        except Exception as e:
            print(f"Warning: could not set smoothing combobox ({e})")
            self.smoothing_cb.setCurrentIndex(3)
        self.smoothing_cb.blockSignals(False)

    def get_selected_detrend_method(self):
        """
        Return the detrend_method_num corresponding to the current combobox selection.
        """
        index_to_detrend_method = {
            0: 1,  # Point to Point
            1: 4,  # Lowess 0.1
            2: 5,  # Lowess 0.2
            3: 2,  # Lowess 0.3 (default)
            4: 7,  # Lowess 0.4
            5: 8,  # Lowess 0.5
        }
        return index_to_detrend_method.get(self.smoothing_cb.currentIndex(), 2)

    def set_current_analyte(self, name):
        """
        Check to see if channel is in analyte_name.
        Preserve the current_run_time when switching analytes.
        """
        if hasattr(self, "timeseries_tab") and self.timeseries_tab:
            self.timeseries_tab.set_current_analyte(name)

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

        self.load_selected_run()
        self.on_plot_type_changed(self.current_plot_type)
        
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

    def on_toggle_grid_toggled(self, state):
        """
        Called when the toggle_grid_cb checkbox is toggled.
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
                #self.fe3db.save_db_file(self.fe3db.db)
                #event.accept()
            else:
                pass
        

def get_instrument_for(instrument_id: str):
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
    
    args = parser.parse_args()
    
    try:
        instrument = get_instrument_for(args.instrument)
    except ValueError as e:
        print(e)
        sys.exit(1)

    app = QApplication(sys.argv)
    w = MainWindow(instrument)
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()