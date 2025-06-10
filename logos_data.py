#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QRadioButton,
    QButtonGroup, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QDateTime

import logos_instruments as li


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

        # Set up the UI
        self.init_ui()

    def init_ui(self):
        # Central widget + top‐level layout
        central = QWidget()
        self.setCentralWidget(central)
        h_main = QHBoxLayout()
        central.setLayout(h_main)

        # Left pane: all controls (run selection, analyte selection)
        left_pane = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(12)
        left_pane.setLayout(left_layout)

        # ── DATA RANGE SELECTION ──
        date_gb = QGroupBox("Date Range (by Month)")
        date_layout = QHBoxLayout()
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

        # Stretch to push everything to the top
        left_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Right pane: placeholder for a plot (e.g. a matplotlib canvas)
        right_placeholder = QGroupBox("Plot Area (placeholder)")
        right_layout = QVBoxLayout()
        right_placeholder.setLayout(right_layout)
        placeholder_label = QLabel("(plots go here)")
        placeholder_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(placeholder_label)

        # Add both panes to the main hbox
        h_main.addWidget(left_pane, stretch=0)
        h_main.addWidget(right_placeholder, stretch=1)

        # Kick off by selecting the first analyte by default
        # (This will load data and populate run_times)
        if self.analytes:
            first_name = list(self.analytes.keys())[0]
            self.set_current_analyte(first_name)

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
        df = self.instrument.load_data(
            pnum=self.current_pnum,
            channel=self.current_channel,
            start_date=start_sql,
            end_date=end_sql
        )

        # Populate run_cb just like set_current_analyte() does
        if df is not None and not df.empty:
            times = sorted(df["run_time"].unique())
            # may need to limit to the last 50 runs:
            if len(times) > 50:
                print(f">>> Warning: more than 50 runs found, limiting to the last 50.")
                times = times[-50:]
            self.current_run_times = [
                QDateTime.fromSecsSinceEpoch(int(t.timestamp())).toString("yyyy/MM/dd HH:mm:ss")
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
            self.run_cb.setCurrentIndex(0)
            self.on_run_changed(0)

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
        df = self.instrument.load_data(
            pnum=pnum,
            channel=self.current_channel,
            start_date=start_sql,
            end_date=end_sql
        )

        # Extract unique run_time values (as Python datetime)
        if df is not None and not df.empty:
            times = sorted(df["run_time"].unique())
            # Convert to QDateTime strings for display:
            self.current_run_times = [
                QDateTime.fromSecsSinceEpoch(int(t.timestamp())).toString("yyyy/MM/dd HH:mm:ss")
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
            # Default to the first run_time if the current_run_time is not found
            self.run_cb.setCurrentIndex(0)
            self.on_run_changed(0)

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

    def on_filter_changed(self, _=None):
        """
        Called whenever run_type_cb or date_filter_cb changes.
        We simply reload data for the current analyte and re-populate run_times.
        """
        # If no analyte has been chosen yet, do nothing
        if self.current_pnum is None:
            return

        df = self.m4.load_data(
            pnum=self.current_pnum
        )
        if df is not None and not df.empty:
            times = sorted(df["run_time"].unique())
            self.current_run_times = [
                QDateTime.fromSecsSinceEpoch(int(t.timestamp())).toString("yyyy/MM/dd HH:mm:ss")
                for t in times
            ]
        else:
            self.current_run_times = []

        # Update combo box
        self.run_cb.blockSignals(True)
        self.run_cb.clear()
        for s in self.current_run_times:
            self.run_cb.addItem(s)
        self.run_cb.blockSignals(False)

        # Select the first run if possible
        if self.current_run_times:
            self.run_cb.setCurrentIndex(0)
            self.on_run_changed(0)

    def on_run_changed(self, index):
        """
        Called whenever the user picks a different run_time in run_cb. 
        Right now, we just print it. Later, you’d update the plot on the right side.
        """
        if index < 0 or index >= len(self.current_run_times):
            return
        self.current_run_time = self.current_run_times[index]

        print(f">>> Selected run_time: {self.current_run_time}")
        # TODO: Convert run_str back to a datetime, then filter self.m4.data to that run_time,
        # and redraw the matplotlib canvas here.

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
        "-i", "--instrument",
        type=str,
        choices=insts,  # Use the instruments list
        default="m4",
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
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()