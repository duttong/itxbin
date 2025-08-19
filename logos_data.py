#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
import argparse

from PyQt5 import QtCore, QtGui

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QRadioButton,
    QButtonGroup, QScrollArea, QSizePolicy, QSpacerItem, QCheckBox
)
from PyQt5.QtCore import Qt, QDateTime

import logos_instruments as li
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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
    def __init__(self, canvas, main_window):
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

        # Set up the UI
        self.toggle_grid_cb = QCheckBox("Toggle Grid")  # Initialize toggle_grid_cb
        self.toggle_grid_cb.setChecked(True)  # Default to showing grid
        self.toggle_grid_cb.stateChanged.connect(self.on_toggle_grid_toggled)

        self.lock_y_axis_cb = QCheckBox("Lock Y-Axis Scale")  # Initialize lock_y_axis_cb
        self.lock_y_axis_cb.setChecked(False)  # Default to unlocked
        self.lock_y_axis_cb.stateChanged.connect(self.on_lock_y_axis_toggled)

        self.RUN_TYPE_MAP = {
            "All": None,        # no filter
            "Flasks": 1,        # run_type_num
            "Calibrations": 2,
            "PFPs": 5,
        }
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
        start_year = (datetime.now() - timedelta(days=60)).year  # 2 months ago
        start_month = (datetime.now() - timedelta(days=60)).month  # 2 months ago
        # Fill years (e.g. 2020..2025) and months (Jan..Dec)
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
        # Set default start date to the instrument's start date
        self.start_year_cb.setCurrentText(str(start_year))
        self.start_month_cb.setCurrentIndex(int(start_month) - 1)

        # “Apply” button
        self.apply_date_btn = QPushButton("Apply ▶")
        self.apply_date_btn.clicked.connect(self.on_apply_month_range)

        # Add to date_layout:
        date_layout.addWidget(QLabel("From:"))
        date_layout.addWidget(self.start_year_cb)
        date_layout.addWidget(self.start_month_cb)
        date_layout.addSpacing(12)
        date_layout.addWidget(QLabel("To:"))
        date_layout.addWidget(self.end_year_cb)
        date_layout.addWidget(self.end_month_cb)
        date_layout.addSpacing(12)
        date_layout.addWidget(self.apply_date_btn)

        # Run Type Selection ComboBox  
        self.runTypeCombo = QComboBox(self)
        self.runTypeCombo.addItems(list(self.RUN_TYPE_MAP.keys()))
        self.runTypeCombo.setCurrentText("All")
        self.runTypeCombo.currentTextChanged.connect(self.on_apply_month_range)
        
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
        resp_rb = QRadioButton("Response")
        ratio_rb = QRadioButton("Ratio")
        mole_fraction_rb = QRadioButton("Mole Fraction")
        self.calibration_rb = QRadioButton("Calibration")
        self.calibration_rb.setEnabled(False)  # Disable calibration until selected

        plot_layout.addWidget(resp_rb)
        plot_layout.addWidget(ratio_rb)
        plot_layout.addWidget(mole_fraction_rb)
        plot_layout.addWidget(self.calibration_rb)

        self.plot_radio_group.addButton(resp_rb, id=0)
        self.plot_radio_group.addButton(ratio_rb, id=1)
        self.plot_radio_group.addButton(mole_fraction_rb, id=2)
        self.plot_radio_group.addButton(self.calibration_rb, id=3)
        resp_rb.setChecked(True)
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
        self.toolbar = FastNavigationToolbar(self.canvas, self)  # Pass self explicitly
        right_layout.addWidget(self.toolbar)

        # Add both panes to the main hbox
        h_main.addWidget(left_pane, stretch=0)  # Fixed width for left pane
        h_main.addWidget(right_placeholder, stretch=1)  # Flexible width for right pane

        # Kick off by selecting the first analyte by default
        # (This will load data and populate run_times)
        if self.analytes:
            first_name = list(self.analytes.keys())[0]
            if self.instrument.inst_id == 'm4':
                self.set_current_analyte(first_name)

    def on_plot_type_changed(self, id: int):
        if id == 0:
            self.gc_plot('resp')
        elif id == 1:
            self.gc_plot('ratio')
        elif id == 2:
            self.gc_plot('mole_fraction')
        else:
            self.calibration_plot()
        
        if id != self.current_plot_type:
            self.current_plot_type = id
            self.lock_y_axis_cb.setChecked(False)
    
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
            
            if self.instrument.inst_id == 'fe3':
                # if mole_fraction is missing, compute it for fe3
                mf_mask = self.run['normalized_resp'].gt(0.1) & self.run['mole_fraction'].isna()
                if mf_mask.any():
                    curves = self.instrument.param_calcurves(self.run)
                    if curves.empty:
                        print("No calibration curves available for mole fraction calculation.")
                    else:
                        if self.instrument.inst_id == 'fe3':
                            target_serial = self.run.loc[self.run['port'] == self.instrument.STANDARD_PORT_NUM, 'port_info'].unique()
                        elif self.instrument.inst_id == 'm4':
                            target_serial = self.run.loc[self.run['run_type_num'] == self.instrument.STANDARD_RUN_TYPE, 'port_info'].unique()
                        if len(target_serial) != 1:
                            print(f"Expected one target serial number, found: {target_serial}")
                            return
                        target_serial = target_serial[0]
                        curves = curves.loc[curves['serial_number'] == target_serial]
                        self.run = self.instrument.select_cal_and_compute_mf(self.run, curves, by=None)
                        sub_info = f"Mole Fraction computed"
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

        legend_handles = []
        for port in ports_in_run:
            col = self.instrument.COLOR_MAP.get(port, 'gray')
            label = port_label_map.get(port, str(port))
            legend_handles.append(
                mpatches.Patch(color=col, label=label)
            )

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(self.run['analysis_datetime'], self.run[yvar], marker='o', c=colors)
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

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.8),
            fontsize=9,
            frameon=False
        )

        if self.lock_y_axis_cb.isChecked():
            # use the stored y-axis limits
            if self.y_axis_limits is None:
                # If no limits are set, use the current y-limits
                self.y_axis_limits = ax.get_ylim()
            else:
                ax.set_ylim(self.y_axis_limits)
            #print('Y-AXIS LIMITS LOCKED:', self.y_axis_limits)
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

        curves = self.instrument.load_calcurves(self.data)
        if curves.empty:
            print("No calibration curves available for plotting.")
            return
        curves['run_time'] = pd.to_datetime(curves['run_date'], utc=True)

        # Ref tank mole fraction estimate
        ref_estimate = self.instrument.select_cal_and_compute_mf(
            self.run.loc[self.run['port'] == self.instrument.STANDARD_PORT_NUM],
            curves
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
            legend_handles.append(mpatches.Patch(color=col, label=label))

        # ref tank mean and std
        ref_mf_mean = ref_estimate['mole_fraction'].mean()
        ref_mf_sd   = ref_estimate['mole_fraction'].std()
        ref_resp_mean = ref_estimate['normalized_resp'].mean()
        ref_resp_sd   = ref_estimate['normalized_resp'].std()

        yvar = 'normalized_resp'  # Use normalized_resp for calibration plots
        tlabel = f'Calibration Scale {int(np.nanmin(self.run["scale_num"]))}'
        mn_cal = float(np.nanmin(self.run['cal_mf']))
        mx_cal = float(np.nanmax(self.run['cal_mf']))
        try:
            x_one2one = np.linspace(mn_cal * .95, mx_cal * 1.05, 200)
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
            legend_handles.append(mpatches.Patch(color='black', label="ref mean $\\pm 1\\sigma$"))
    
        # Plot stored cal curves (bottom axis). Compute residuals against the newest curve (row 0).
        if not curves.empty:
            # Use the newest curve as the model for residuals
            row0 = curves.iloc[0]
            coefs0 = [row0['coef3'], row0['coef2'], row0['coef1'], row0['coef0']]
            # Predicted response at the actual run_x positions
            y_pred = np.polyval(coefs0, run_x.to_numpy())
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

            # Plot the cal curves (lines) on the main axis
            xgrid = np.linspace(mn_cal * .95, mx_cal * 1.05, 300)
            calcurve_exists = False
            for i, row in curves.iloc[0:5].iterrows():
                if self.run['run_time'].iloc[0] == row['run_time']:
                    ygrid = np.polyval([row['coef3'], row['coef2'], row['coef1'], row['coef0']], xgrid)
                    ax.plot(xgrid, ygrid, linewidth=2, color='black', alpha=0.7,
                            label=row['run_date'].strftime('%Y-%m-%d'))
                    calcurve_exists = True
            if calcurve_exists == False:
                ax.text(
                    0.5, .98, "missing calculation curve",
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

        # Grid toggle applies to both axes
        if self.toggle_grid_cb.isChecked():
            for _ax in (ax_resid, ax):
                _ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.8)
        else:
            for _ax in (ax_resid, ax):
                _ax.grid(False)

        # Put legend outside; adjust right margin instead of manually resizing axes
        self.figure.subplots_adjust(right=0.82)
        ax.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=False
        )

        # Optional y-axis locking for the main axis only
        if self.lock_y_axis_cb.isChecked():
            if self.y_axis_limits is None:
                self.y_axis_limits = ax.get_ylim()
            else:
                ax.set_ylim(self.y_axis_limits)
        else:
            try:
                ax.set_ylim(run_y.min() * 0.95, run_y.max() * 1.05)
            except ValueError:
                pass

        self.canvas.draw()

    def get_load_range(self):
        # Read selection from the four combo boxes
        sy = self.start_year_cb.currentText()
        sm = self.start_month_cb.currentIndex() + 1   # Jan→1, Feb→2, etc.
        ey = self.end_year_cb.currentText()
        em = self.end_month_cb.currentIndex() + 1

        # Build start/end strings of the form YYMM
        start_sql = f"{sy[2:4]}{sm:02d}"
        end_sql   = f"{ey[2:4]}{em:02d}"
        return start_sql, end_sql
    
    def on_apply_month_range(self):
        start_sql, end_sql = self.get_load_range()
        # Reload data for the current analyte with that range
        self.data = self.instrument.load_data(
            pnum=self.current_pnum,
            channel=self.current_channel,
            start_date=start_sql,
            end_date=end_sql
        )

        # If runTypeCombo is set, filter the data by run_type_num
        run_type = self.runTypeCombo.currentText()
        self.run_type_num = self.RUN_TYPE_MAP.get(run_type, None)
        if self.run_type_num is not None:
            # Filter the DataFrame for the selected run_type_num
            times = set(self.data.loc[self.data['run_type_num'] == self.run_type_num, 'run_time'])
            self.data = self.data[self.data['run_time'].isin(times)]
        
        if self.run_type_num == self.RUN_TYPE_MAP['Calibrations']:
            self.calibration_rb.setEnabled(True)
        else:
            self.calibration_rb.setEnabled(False)
        
        # Extract unique run_time values (as Python datetime)
        if self.data is not None and not self.data.empty:
            # 1) get all unique times
            times = sorted(self.data["run_time"].unique())

            # 2) build sets of times that need a suffix
            cal_times = set(self.data.loc[self.data['run_type_num'] == 2, 'run_time'])
            pfp_times = set(self.data.loc[self.data['run_type_num'] == 5, 'run_time'])

            # 3) build your display strings, appending (Cal) and/or (PFP)
            self.current_run_times = [
                QDateTime.fromSecsSinceEpoch(
                    int(t.replace(tzinfo=timezone.utc).timestamp()),
                    Qt.UTC
                ).toString("yyyy-MM-dd HH:mm:ss")
                + (" (Cal)" if t in cal_times else "")
                + (" (PFP)" if t in pfp_times else "")
                for t in times
            ]
        else:
            self.current_run_times = []
    
        self.run_cb.blockSignals(True)
        self.run_cb.clear()
        for s in self.current_run_times:
            self.run_cb.addItem(s)
        self.run_cb.blockSignals(False)

        if self.current_run_times:
            last_idx = len(self.current_run_times) - 1
            self.run_cb.setCurrentIndex(last_idx)
            self.on_run_changed(last_idx)
            
        self.on_plot_type_changed(self.current_plot_type)

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
                buttons[0].setChecked(True)

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
            self.set_current_analyte(name)

    def on_analyte_combo_changed(self, name):
        """
        Called whenever the QComboBox selection changes (for >12 analytes).
        """
        self.set_current_analyte(name)

    def set_current_analyte(self, analyte_name):
        """
        Check to see if channel is in analyte_name.
        Preserve the current_run_time when switching analytes.
        """
        # Extract channel if present in analyte_name
        if '(' in analyte_name and ')' in analyte_name:
            self.current_channel = analyte_name.split('(')[1].split(')')[0].strip()
        else:
            self.current_channel = None

        pnum = self.analytes[analyte_name]
        self.current_pnum = int(pnum)
        print(f">>> Setting current analyte: {analyte_name} (pnum={pnum}, channel={self.current_channel})")

        start_sql, end_sql = self.get_load_range()
        # (Re)load data for this analyte and the load range
        self.data = self.instrument.load_data(
            pnum=pnum,
            channel=self.current_channel,
            start_date=start_sql,
            end_date=end_sql
        )
       
        # If runTypeCombo is set, filter the data by run_type_num
        run_type = self.runTypeCombo.currentText()
        self.run_type_num = self.RUN_TYPE_MAP.get(run_type, None)
        if self.run_type_num is not None:
            # Filter the DataFrame for the selected run_type_num
            times = set(self.data.loc[self.data['run_type_num'] == self.run_type_num, 'run_time'])
            self.data = self.data[self.data['run_time'].isin(times)]
            #self.data = self.data[self.data['run_type_num'] == run_type_num]

        # Extract unique run_time values (as Python datetime)
        if self.data is not None and not self.data.empty:
            # 1) get all unique times
            times = sorted(self.data["run_time"].unique())

            # 2) build sets of times that need a suffix
            cal_times = set(self.data.loc[self.data['run_type_num'] == 2, 'run_time'])
            pfp_times = set(self.data.loc[self.data['run_type_num'] == 5, 'run_time'])

            # 3) build your display strings, appending (Cal) and/or (PFP)
            self.current_run_times = [
                QDateTime.fromSecsSinceEpoch(
                    int(t.replace(tzinfo=timezone.utc).timestamp()),
                    Qt.UTC
                ).toString("yyyy-MM-dd HH:mm:ss")
                + (" (Cal)" if t in cal_times else "")
                + (" (PFP)" if t in pfp_times else "")
                for t in times
            ]
        else:
            self.current_run_times = []
            
        # Fill the run_cb combo with these run_time strings
        self.run_cb.blockSignals(True)
        self.run_cb.clear()
        for s in self.current_run_times:
            self.run_cb.addItem(s)
        self.run_cb.blockSignals(False)

        # Preserve the current_run_time if it exists in the new analyte's run_times
        if self.current_run_time in self.current_run_times:
            idx = self.current_run_times.index(self.current_run_time)
            self.run_cb.setCurrentIndex(idx)
            self.on_run_changed(idx)
        elif self.current_run_times:
            # Default to the last run_time if the current_run_time is not found
            last_idx = len(self.current_run_times) - 1
            self.run_cb.setCurrentIndex(last_idx)
            self.on_run_changed(last_idx)

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
        
    def on_run_changed(self, index):
        """
        Called whenever the user picks a different run_time in run_cb. 
        """
        if index < 0 or index >= len(self.current_run_times):
            return
        self.current_run_time = self.current_run_times[index]
        self.on_plot_type_changed(self.current_plot_type)

    def on_prev_run(self):
        """
        Move the run_cb selection one index backward, if possible.
        """
        idx = self.run_cb.currentIndex()
        if idx > 0:
            self.run_cb.setCurrentIndex(idx - 1)

    def on_next_run(self):
        """
        Move the run_cb selection one index forward, if possible.
        """
        idx = self.run_cb.currentIndex()
        if idx < (self.run_cb.count() - 1):
            self.run_cb.setCurrentIndex(idx + 1)

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