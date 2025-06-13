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
    QButtonGroup, QScrollArea, QSizePolicy, QSpacerItem
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
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        # disable the built-in rubberband (a QWidget we don’t use)
        self.rubberband = None

        # build the dashed-pen
        pen = QtGui.QPen(self.palette().color(QtGui.QPalette.Highlight))
        pen.setStyle(QtCore.Qt.DashLine)

        # make our overlay on top of the canvas
        self._overlay = RubberBandOverlay(canvas, pen)
        
        canvas.mpl_connect(
            'resize_event',
            lambda evt: self._overlay.setGeometry(0, 0, canvas.width(), canvas.height())
        )

    def draw_rubberband(self, event, x0, y0, x1, y1):
            # 1) make the overlay match the canvas size
            w, h = self.canvas.width(), self.canvas.height()
            self._overlay.setGeometry(0, 0, w, h)

            # 2) flip the y coordinates so (0,0) is top-left
            y0i = h - y0
            y1i = h - y1

            # 3) build & normalize the rect in widget coords
            self._overlay.rect = QtCore.QRectF(
                QtCore.QPointF(x0, y0i),
                QtCore.QPointF(x1, y1i)
            ).normalized()

            # 4) show & repaint
            self._overlay.show()
            self._overlay.update()

    def press_zoom(self, event):
        self._old_aa = rcParams['lines.antialiased']
        rcParams['lines.antialiased'] = False
        super().press_zoom(event)

    def release_zoom(self, event):
        # first let the base class do the zoom…
        super().release_zoom(event)
        rcParams['lines.antialiased'] = self._old_aa
        # …then hide our overlay
        self._overlay.hide()
        
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

        # Set up the UI
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

        plot_layout.addWidget(resp_rb)
        plot_layout.addWidget(ratio_rb)
        plot_layout.addWidget(mole_fraction_rb)

        self.plot_radio_group.addButton(resp_rb, id=0)
        self.plot_radio_group.addButton(ratio_rb, id=1)
        self.plot_radio_group.addButton(mole_fraction_rb, id=2)
        # Set "Resp" as the default selected option
        resp_rb.setChecked(True)
        self.plot_radio_group.idClicked[int].connect(self.on_plot_type_changed)

        left_layout.addWidget(plot_gb)

        # Stretch to push everything to the top
        left_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Right pane: matplotlib figure for plotting
        right_placeholder = QGroupBox("Plot Area")
        right_layout = QVBoxLayout()
        right_placeholder.setLayout(right_layout)
        right_layout.addWidget(self.canvas)

        # Add a NavigationToolbar for the figure
        self.toolbar = FastNavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        # Add both panes to the main hbox
        h_main.addWidget(left_pane, stretch=0)  # Fixed width for left pane
        h_main.addWidget(right_placeholder, stretch=1)  # Flexible width for right pane

        # Kick off by selecting the first analyte by default
        # (This will load data and populate run_times)
        if self.analytes:
            first_name = list(self.analytes.keys())[0]
            self.set_current_analyte(first_name)

        # Default plot for "Response"
        #self.gc_plot()

    def on_plot_type_changed(self, id: int):
        if id == 0:
            self.gc_plot('resp')
        elif id == 1:
            self.gc_plot('ratio')
        else:
            self.gc_plot('mole_fraction')
        self.current_plot_type = id
    
    def gc_plot(self, yparam='resp'):
        """
        Plot 'Response' (self.data.area vs self.data.analysis_datetime) with the legend outside the plotting area.
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

        if yparam == 'resp':
            yvar = self.instrument.response_type
            tlabel = 'Response'
        elif yparam == 'ratio':
            yvar = self.instrument.ratio_type
            tlabel = 'Ratio'
        elif yparam == 'mole_fraction':
            yvar = 'mole_fraction'
            tlabel = 'Mole Fraction'
        else:
            print(f"Unknown yparam: {yparam}")
            return
                
        colors = self.run['port_idx'].map(self.instrument.COLOR_MAP).fillna('gray')
        ports_in_run = sorted(self.run['port_idx'].dropna().unique())          

        port_label_map = (
            self.run
            .loc[self.run['port_idx'].notna(), ['port_idx','port_label']]
            .drop_duplicates()
            .set_index('port_idx')['port_label']
            .to_dict()
        )
            
        legend_handles = []
        for port in ports_in_run:
            # lookup color (default to gray if missing)
            col = self.instrument.COLOR_MAP.get(port, 'gray')
            label = port_label_map.get(port, str(port))
            legend_handles.append(
                mpatches.Patch(color=col, label=label)
            )
    
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(self.run['analysis_datetime'], self.run[yvar], marker='o', c=colors)
        ax.set_title(f"{self.current_run_time} - {tlabel}: {self.instrument.analytes_inv[int(self.current_pnum)]} ({self.current_pnum})")
        ax.set_xlabel("Analysis Datetime")
        ax.xaxis.set_tick_params(rotation=30)
        ax.set_ylabel(tlabel)

        box = ax.get_position()  
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(
            handles=legend_handles,
            loc='center left',            # legend’s “anchor point”
            bbox_to_anchor=(1.02, 0.8),    # (x, y) in axis-fraction coordinates
            fontsize=9,
            frameon=False
        )
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
            # Use radio buttons in two columns: first 5 left, rest right
            self.radio_group = QButtonGroup(self)
            left  = names[:6]
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
        self.current_pnum = pnum
        print(f">>> Setting current analyte: {analyte_name} (pnum={pnum}, channel={self.current_channel})")

        start_sql, end_sql = self.get_load_range()
        # (Re)load data for this analyte and the load range
        self.data = self.instrument.load_data(
            pnum=pnum,
            channel=self.current_channel,
            start_date=start_sql,
            end_date=end_sql
        )

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
        Right now, we just print it. Later, you’d update the plot on the right side.
        """
        if index < 0 or index >= len(self.current_run_times):
            return
        self.current_run_time = self.current_run_times[index]

        #print(f">>> Selected run_time: {self.current_run_time}")
        self.on_plot_type_changed(self.current_plot_type)
        #self.gc_plot()

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