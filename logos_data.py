#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from functools import lru_cache
import warnings
import argparse

from PyQt5 import QtCore, QtGui

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QRadioButton, QAction,
    QButtonGroup, QScrollArea, QSizePolicy, QSpacerItem, QCheckBox
)
from PyQt5.QtCore import Qt, QDateTime

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


import logos_instruments as li

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
        painter = QtGui.QPainter(self)
        painter.setCompositionMode(QtGui.QPainter.RasterOp_SourceXorDestination)
        painter.setPen(self.pen)
        painter.drawRect(self.rect)
        painter.end()

class FastNavigationToolbar(NavigationToolbar):
    def __init__(self, canvas, main_window, on_flag_toggle=None):
        super().__init__(canvas, main_window)
        self.main_window = main_window
        self.rubberband = None

        pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Highlight))
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
        self.data = pd.DataFrame()  # Placeholder for loaded data
        self.y_axis_limits = None  # Store y-axis limits when locked
        self.toggle_grid_cb = None  # Initialize toggle_grid_cb to avoid AttributeError
        self.lock_y_axis_cb = None  # Initialize lock_y_axis_cb to avoid AttributeError
        self.calibration_rb = QRadioButton()
        self.resp_rb = QRadioButton()
        self.fit_method_cb = QComboBox()
        self.draw2zero_cb = QCheckBox()
        self.oldcurves_cb = QCheckBox()

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

        # Left pane: all controls (run selection, analyte selection)
        left_pane = QWidget()
        left_pane.setMinimumWidth(420)  # Set fixed width
        left_pane.setMaximumWidth(420)  # Set fixed width
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(4, 4, 4, 4)  # Reduce margins
        left_layout.setSpacing(6)  # Reduce spacing between widgets
        left_pane.setLayout(left_layout)

        # ── DATA RANGE SELECTION ──
        date_gb = QGroupBox("Date Range (by Month)")
        date_layout = QHBoxLayout()
        date_layout.setContentsMargins(2, 2, 2, 2)  # Reduce margins inside the group box
        date_layout.setSpacing(4)  # Reduce spacing inside the group box
        date_gb.setLayout(date_layout)

        # Start: Year / Month
        self.start_year_cb = QComboBox()
        self.start_month_cb = QComboBox()
        current_year = datetime.now().year
        current_month = datetime.now().month
        start_year = (datetime.now() - timedelta(days=30)).year  # 1 month ago
        start_month = (datetime.now() - timedelta(days=30)).month  # 1 month ago
        # Fill years and months
        for y in range(int(self.instrument.start_date[0:4]), int(current_year) + 1):
            self.start_year_cb.addItem(str(y))
        for m in range(1, 13):
            self.start_month_cb.addItem(datetime(2000, m, 1).strftime("%b"))

        # End: Year / Month
        self.end_year_cb = QComboBox()
        self.end_month_cb = QComboBox()
        for y in range(int(self.instrument.start_date[0:4]), int(current_year) + 1):
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

        left_layout.addWidget(run_gb)

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

        left_layout.addWidget(analyte_gb)

        # Plot Type Selection GroupBox
        plot_gb = QGroupBox("Plot Type Selection")
        plot_layout = QVBoxLayout()
        plot_layout.setSpacing(6)
        plot_gb.setLayout(plot_layout)

        self.plot_radio_group = QButtonGroup(self)
        self.resp_rb = QRadioButton("Response")
        ratio_rb = QRadioButton("Ratio")
        mole_fraction_rb = QRadioButton("Mole Fraction")
        self.calibration_rb = QRadioButton("Calibration")
        self.calibration_rb.setEnabled(False)
        self.draw2zero_cb = QCheckBox("Zero")
        self.oldcurves_cb = QCheckBox("Other Curves")

        plot_layout.addWidget(self.resp_rb)
        plot_layout.addWidget(ratio_rb)
        plot_layout.addWidget(mole_fraction_rb)

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
        plot_layout.addLayout(cal_row)

        # ── EXTRA OPTIONS ROW ──
        self.draw2zero_cb.setEnabled(False)    # start disabled
        self.oldcurves_cb.setEnabled(False)    # start disabled

        cal_row2 = QHBoxLayout()
        cal_row2.addSpacing(24)  # indent to align with calibration radio
        cal_row2.addWidget(self.draw2zero_cb)
        cal_row2.addWidget(self.oldcurves_cb)
        cal_row2.addStretch(1)   # push them left
        plot_layout.addLayout(cal_row2)
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

        left_layout.addWidget(combined_gb)

        # Stretch to push everything to the top
        left_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Right pane: matplotlib figure for plotting
        right_placeholder = QGroupBox("Plot Area")
        right_layout = QVBoxLayout()
        right_placeholder.setLayout(right_layout)
        right_layout.addWidget(self.canvas)

        # Add a NavigationToolbar for the figure
        self.toolbar = FastNavigationToolbar(self.canvas, self, on_flag_toggle=self.on_flag_mode_toggled)  # Pass self explicitly
        right_layout.addWidget(self.toolbar)

        # Add both panes to the main hbox
        h_main.addWidget(left_pane, stretch=0)  # Fixed width for left pane
        h_main.addWidget(right_placeholder, stretch=1)  # Flexible width for right pane

        # Kick off by selecting the first analyte by default
        # (This will load data and populate run_times)
        self.set_runlist()
        self.on_plot_type_changed(0)
        if self.instrument.inst_id == 'm4':
            self.current_pnum = 20
            self.set_current_analyte()

    def on_flag_mode_toggled(self, enabled: bool):
        self.flag_mode = enabled
        self.canvas.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        if enabled:
            self._cid = self.canvas.mpl_connect("button_press_event", self.on_flag_click)
        else:
            if hasattr(self, "_cid"):
                self.canvas.mpl_disconnect(self._cid)
                self._cid = None
    
    def on_flag_click(self, index):
        print(f'flag clicked {index}')

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
        
    def on_plot_type_changed(self, id: int):
        self.current_plot_type = id
        self.set_calibration_enabled(False)
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
            
    def _fmt(self, x, y):
        return f"x={mdates.num2date(x).strftime('%Y-%m-%d %H:%M')}  y={y:0.3g}"
            
    def gc_plot(self, yparam='resp'):
        """
        Plot data with the legend sorted by analysis_datetime.
        """
        if self.data.empty:
            print("No data available for plotting.")
            return

        ts_str = self.current_run_time.split(" (")[0]
        sel = pd.to_datetime(ts_str, utc=True)
        self.run = self.data.loc[self.data['run_time'] == sel]
        if self.run.empty:
            print(f"No data for run_time: {self.current_run_time}")
            return

        current_curve_date = ''
        sub_info = ''
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
                # if mole_fraction is missing, compute it for fe3
                mf_mask = self.run['normalized_resp'].gt(0.1) & self.run['mole_fraction'].isna()
                if mf_mask.any():
                    self.run.loc[mf_mask, 'mole_fraction'] = self.instrument.calc_mole_fraction(self.run.loc[mf_mask])
                    sub_info = f"Mole Fraction computed"
                if current_curve_date == None:
                    sub_info = "No calibration curve available"
        else:
            print(f"Unknown yparam: {yparam}")
            return

        colors = self.run['port_idx'].map(self.instrument.COLOR_MAP).fillna('gray')
        ports_in_run = sorted(self.run['port_idx'].dropna().unique())

        port_label_map = (
            self.run
            .loc[self.run['port_idx'].notna(), ['analysis_datetime', 'port_idx', 'port_label']]
            .drop_duplicates()
            .sort_values('analysis_datetime')  # Sort by analysis_datetime
            .set_index('port_idx')['port_label']
            .to_dict()
        )

        # Calculate mean and std for each port
        flags = self.run['data_flag_int'].astype(bool)
        good = self.run.loc[~flags, ['port_idx', yvar]]
        stats_map = (
            good.groupby("port_idx")[yvar]
            .agg(["mean", "std", "count"])
            .to_dict("index")  # -> {port: {"mean": ..., "std": ...}}
        )

        legend_handles = []
        for port in ports_in_run:
            col = self.instrument.COLOR_MAP.get(port, "gray")
            base_label = port_label_map.get(port, str(port))
            if base_label == 'Push port':
                continue

            stats = stats_map.get(port)
            if stats is not None:
                # Two-line label with mean ± std
                if yparam == 'resp':
                    label = f"{base_label}"
                elif yparam == 'ratio':
                    label = f"{base_label}\n{stats['mean']:.3f} ± {stats['std']:.3f} ({stats['count']})"
                else:
                    label = f"{base_label}\n{stats['mean']:.2f} ± {stats['std']:.2f} ({stats['count']})"
            else:
                label = base_label

            legend_handles.append(
                mpatches.Patch(color=col, label=label)
            )
    
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(self.run['analysis_datetime'], self.run[yvar], marker='o', c=colors, s=32, edgecolors='none', zorder=1)
        # overlay: red ring around flagged points
        ax.scatter(self.run.loc[flags, 'analysis_datetime'],
                self.run.loc[flags, yvar],
                facecolors='none', edgecolors='red', linewidths=1.5,
                marker='o', s=36, zorder=3)

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

        # Cal curve date and age (if available)
        if current_curve_date:
            cal_delta_time = self.run['analysis_datetime'].min() - pd.to_datetime(current_curve_date, utc=True)
            l = current_curve_date.strftime('\nCal Date:\n%Y-%m-%d %H:%M\n') + f'{cal_delta_time.days} days ago'
            legend_handles.append(Line2D([], [], linestyle='None', label=l))
            
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.6),
            fontsize=9,
            frameon=False
        )
        ax.format_coord = self._fmt

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
        
        self.canvas.draw()
        
    def calibration_plot(self):
        """
        Plot data with the legend sorted by analysis_datetime.
        Adds a residuals (diff_y) panel above the main plot that shares the x-axis.
        """
        if self.data.empty:
            print("No data available for plotting.")
            return

        # filter for run_time selected in run_cb
        ts_str = self.current_run_time.split(" (")[0]
        sel = pd.to_datetime(ts_str, utc=True)
        self.run = self.data.loc[self.data['run_time'] == sel].copy()
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
        earliest_time = self.data['run_time'].min() - pd.DateOffset(months=6)
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

        # Normalize time
        curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True, errors='coerce')

        # ── Create/append as needed ───────────────────────────────────────────────────
        has_current = curves['run_time'].eq(sel_rt).any()
        
        calcurve_exists = True
        # file in scale_assignment values for calibration tanks in self.run
        self.populate_cal_mf()
        new_fit = self.instrument._fit_row_for_current_run(self.run, order=self.current_fit_degree)
        # save new fit info for Save Cal2DB button
        self._save_payload = new_fit
        
        if curves.empty:
            # No curves at all → fit and create DF with one row
            try:
                curves = pd.DataFrame([new_fit], columns=REQ_COLS)
            except ValueError:
                print(f'Error in calcurve {new_fit}')
                return
            curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True, errors='coerce')
            calcurve_exists = False
        elif not has_current:
            # Existing curves, but not for this run_time → append one row
            curves = pd.concat([curves, pd.DataFrame([new_fit])], ignore_index=True)
            calcurve_exists = False

        curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True, errors='coerce')
        
        # pd dataframe of ref tank mole fraction estimates for this run
        ### TODO: mask out flagged data
        ref_estimate = self.run.loc[self.run['port'] == self.instrument.STANDARD_PORT_NUM].copy()
        ref_estimate['mole_fraction'] = self.instrument.calc_mole_fraction(
            self.run.loc[self.run['port'] == self.instrument.STANDARD_PORT_NUM],
        )
        
        colors = self.run['port_idx'].map(self.instrument.COLOR_MAP).fillna('gray')
        ports_in_run = sorted(self.run['port_idx'].dropna().unique())

        port_label_map = (
            self.run
            .loc[self.run['port_idx'].notna(), ['analysis_datetime', 'port_idx', 'port_label']]
            .drop_duplicates()
            .sort_values('analysis_datetime')
            .set_index('port_idx')['port_label']
            .to_dict()
        )

        legend_handles = []
        for port in ports_in_run:
            col = self.instrument.COLOR_MAP.get(port, 'gray')
            label = port_label_map.get(port, str(port))
            if port != 9:   # skip port 9 (Push Gas Port)
                legend_handles.append(mpatches.Patch(color=col, label=label))

        # ref tank mean and std
        ref_mf_mean = ref_estimate['mole_fraction'].mean()
        ref_mf_sd   = ref_estimate['mole_fraction'].std()
        ref_resp_mean = ref_estimate['normalized_resp'].mean()
        ref_resp_sd   = ref_estimate['normalized_resp'].std()

        yvar = 'normalized_resp'  # Use normalized_resp for calibration plots
        tlabel = f'Calibration Scale {int(np.nanmin(self.run["cal_scale_num"]))}'
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
        run_x = self.run.loc[mask_main, 'cal_mf'].astype(float)
        run_y = self.run.loc[mask_main, yvar].astype(float)
        
        # Main scatter
        ax.scatter(run_x, run_y, marker='o', c=colors.loc[mask_main], alpha=0.7)

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
        sel = self.run['run_time'].iat[0]  # the timestamp you want
        stored_curve = curves['run_time'].eq(sel)

        if curves is None or curves.empty:
            print("No stored cal curves available.")
            return
        row = curves.loc[stored_curve].iloc[0]
        stored_coefs = [row['coef3'], row['coef2'], row['coef1'], row['coef0']]

        # flag the curve?
        is_flagged = False
        if calcurve_exists and 'flag' in row and pd.notna(row['flag']):
            try:
                is_flagged = bool(int(row['flag']))
            except Exception:
                is_flagged = False
        self._flag_state = is_flagged        

        # fit labels are from stored curve
        fitlabel = f'\nfit =\n'
        for n, coef in enumerate(stored_coefs[::-1]):
            if coef != 0.0:
                fitlabel += f'{coef:0.6f} ($x^{n}$) \n'
        legend_handles.append(Line2D([], [], linestyle='None', label=fitlabel))
            
        # Predicted response at the actual run_x positions
        new_fit_coefs = [new_fit["coef3"], new_fit["coef2"], new_fit["coef1"], new_fit["coef0"]]
        y_pred = np.polyval(new_fit_coefs, run_x.to_numpy())
        diff_y = run_y.to_numpy() - y_pred

        # store residuals on self.run (align by index)
        self.run.loc[mask_main, 'diff_y'] = diff_y

        # Top residuals panel
        ax_resid.scatter(run_x, diff_y, s=15, c=colors.loc[mask_main], alpha=0.8)
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
                
        save_handle = Line2D([], [], linestyle='None', label='Save Cal2DB')
        flag_label  = 'Unflag Cal2DB' if self._flag_state else 'Flag Cal2DB'

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
                    ax.set_xlim(0, run_x.max() * 1.05)
                else:
                    ax.set_ylim(run_y.min() * 0.95, run_y.max() * 1.05)
                    ax.set_xlim(run_x.min() * 0.95, run_x.max() * 1.05)
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
        """Refresh the legend 'flag' button look from self._flag_state."""
        if getattr(self, '_flag_text', None) is None:
            return
        if getattr(self, '_flag_state', False):
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
                facecolor='#616161',  # neutral gray when not flagged
                edgecolor='none',
                alpha=0.9
            ))

    def save_current_curve(self, flag_value=None):
        payload = getattr(self, "_save_payload", None)
        if payload is None:
            print("No calibration payload available.")
            return

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
        import matplotlib.text as mtext
        art = event.artist
        if not isinstance(art, mtext.Text):
            return

        # Identify by object identity (robust even if label text changes)
        if art is getattr(self, '_save_text', None):
            self.save_current_curve()
        elif art is getattr(self, '_flag_text', None):
            self.toggle_flag_current_curve()

    def toggle_flag_current_curve(self):
        # flip state
        self._flag_state = not getattr(self, '_flag_state', False)
        self._style_flag_button()
        self.canvas.draw_idle()

        # make sure we have a payload – if the user hasn’t pressed “Save” yet,
        # fall back to the stored curve info on-screen
        if getattr(self, "_save_payload", None) is None:
            # grab the row corresponding to this run_time from `curves`
            sel_rt = self.run['run_time'].iat[0]
            cur_row = self.curves.loc[self.curves['run_time'] == sel_rt].iloc[0]
            self._save_payload = cur_row.to_dict()

        # send the upsert with the new flag
        self.save_current_curve(flag_value=(1 if self._flag_state else 0))
        
    def get_load_range(self):
        # Read selection from the four combo boxes
        sy = self.start_year_cb.currentText()
        sm = self.start_month_cb.currentIndex() + 1   # Jan→1, Feb→2, etc.
        ey = self.end_year_cb.currentText()
        em = self.end_month_cb.currentIndex() + 1

        start_sql = f"{sy}-{sm:02d}-01"
        end_sql = f"{ey}-{em:02d}-31"
        return start_sql, end_sql
    
    def set_runlist(self):
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
        
        self.current_run_time = self.current_run_times[-1]
        
        # Fill the run_cb combo with these run_time strings
        self.run_cb.blockSignals(True)
        self.run_cb.clear()
        for s in self.current_run_times:
            self.run_cb.addItem(s)
        #self.run_cb.blockSignals(False)

        # Preserve the current_run_time if it exists in the new analyte's run_times
        if self.current_run_time in self.current_run_times:
            idx = self.current_run_times.index(self.current_run_time)
            self.run_cb.setCurrentIndex(idx)
        elif self.current_run_times:
            # Default to the last run_time if the current_run_time is not found
            last_idx = len(self.current_run_times) - 1
            self.run_cb.setCurrentIndex(last_idx)
        self.run_cb.blockSignals(False)
            
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
            self.set_current_analyte()

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
        self.set_current_analyte()
        
    def load_selected_run(self):
        # call sql load function from instrument class
        # all of the input parameters are set with UI controls.
        self.data = self.instrument.load_data(
            pnum=self.current_pnum,
            channel=self.current_channel,
            run_type_num=self.run_type_num,
            start_date=self.current_run_time,
            end_date=self.current_run_time
        )

    def set_current_analyte(self):
        """
        Check to see if channel is in analyte_name.
        Preserve the current_run_time when switching analytes.
        """
        if self.current_run_time is None:
            self.set_runlist()

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
        self.set_current_analyte()  
        
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
        #self.gc_plot()  # Update the plot to reflect grid state


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