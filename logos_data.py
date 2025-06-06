#!/usr/bin/env python3
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QComboBox, QPushButton, QRadioButton,
    QButtonGroup, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, QDateTime

import logos_instruments as li


class M4_Processing(li.M4_Instrument):
    """
    (For the purpose of this example, we’re only using the .m4_analytes() and
     .run_type_num() methods from M4_base.  We’ll override load_data(...) so
     that it returns a dummy DataFrame with a 'run_time' column that we can
     illustrate with.  In your real code, you already have load_data(...) as
     shown in your post.)
    """
    def __init__(self):
        super().__init__()

    def run_type_num(self):
        """
        Return a dict: { run_type_name: run_type_num, … }.  This is exactly
        what M4_base.run_type_num() would do in your real code.
        """
        return {
            "Flask": 1,
            "Other": 4,
            "PFP": 5,
            "Zero": 6,
            "Tank": 7,
            "Standard": 8
        }

    def load_data(self, pnum, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, "%y%m")

        if start_date is None:
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime.strptime(start_date, "%y%m")

        start_date_str = start_date.strftime("%Y-%m-01")
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Loading data from {start_date_str} to {end_date_str} for parameter {pnum}")
        # todo: use flags - using low_flow flag
        query = f"""
            SELECT analysis_datetime, run_time, run_type_num, port_info, detrend_method_num, 
                area, mole_fraction, net_pressure, flag, sample_id, pair_id_num
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                AND area != 0
                AND detrend_method_num != 3
                AND low_flow != 1
                AND run_time BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            print(f"No data found for parameter {pnum} in the specified date range.")
            self.data = pd.DataFrame()
            return
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])
        df['run_time']          = pd.to_datetime(df['run_time'])
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['area']              = df['area'].astype(float)
        df['net_pressure']      = df['net_pressure'].astype(float)
        df['area']              = df['area']/df['net_pressure']
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df['parameter_num']     = pnum
        self.data = df.sort_values('analysis_datetime')
        return self.data

class FE3_Processing(li.FE3_Instrument):
    """
    Placeholder for FE3 processing logic.
    In your real code, you would implement the necessary methods here.
    """
    def __init__(self):
        super().__init__()
        # Initialize any specific attributes or methods for FE3 processing

    def load_data(self, pnum, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, "%y%m")

        if start_date is None:
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime.strptime(start_date, "%y%m")

        start_date_str = start_date.strftime("%Y-%m-01")
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Loading data from {start_date_str} to {end_date_str} for parameter {pnum}")
        # todo: use flags - using low_flow flag
        query = f"""
            SELECT analysis_datetime, run_time, run_type_num, port_info, detrend_method_num, 
                height, mole_fraction, flag, sample_id, pair_id_num
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                AND height != 0
                AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            print(f"No data found for parameter {pnum} in the specified date range.")
            self.data = pd.DataFrame()
            return
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])
        df['run_time']          = pd.to_datetime(df['run_time'])
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['height']            = df['height'].astype(float)
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df['parameter_num']     = pnum
        self.data = df.sort_values('analysis_datetime')
        return self.data
        
class BLD1_Processing(li.BLD1_Instrument):
    """
    Placeholder for BLD1 processing logic.
    In your real code, you would implement the necessary methods here.
    """
    def __init__(self):
        super().__init__()
        # Initialize any specific attributes or methods for BLD1 processing

class MainWindow(QMainWindow, li.HATS_DB_Functions):
    def __init__(self, processor, instrument, instrument_id):
        # Notice: we call super().__init__(instrument=instrument_id) inside HATS_DB_Functions
        super().__init__(instrument=instrument_id)
        self.processor = processor     # e.g. an M4_Processing() instance
        self.instrument = instrument   # e.g. an M4_Instrument("m4") instance
        self.instrument_id = instrument_id

        self.setWindowTitle(f"{self.instrument_id.upper()} Data Processing (Demo)")

        # Keep track of analytes and run_times
        self.analytes     = self.instrument.analytes
        self.current_pnum = None
        self.current_run_times = []   # will be a sorted list of QDateTime strings

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

        # ── 1) DATA RANGE SELECTION ──
        date_gb = QGroupBox("Date Range (by Month)")
        date_layout = QHBoxLayout()
        date_gb.setLayout(date_layout)

        # Start: Year / Month
        self.start_year_cb = QComboBox()
        self.start_month_cb = QComboBox()
        # Fill years (e.g. 2020..2025) and months (Jan..Dec)
        for y in range(2020, datetime.now().year + 1):
            self.start_year_cb.addItem(str(y))
        for m in range(1, 13):
            self.start_month_cb.addItem(datetime(2000, m, 1).strftime("%b"))

        # End: Year / Month
        self.end_year_cb = QComboBox()
        self.end_month_cb = QComboBox()
        for y in range(2020, datetime.now().year + 1):
            self.end_year_cb.addItem(str(y))
        for m in range(1, 13):
            self.end_month_cb.addItem(datetime(2000, m, 1).strftime("%b"))

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

        # Place the “date_gb” above your run‐type/run‐date filters:
        #left_layout.addWidget(date_gb)
    
        # 1) Run Selection GroupBox
        run_gb = QGroupBox("Run Selection")
        run_layout = QVBoxLayout()
        run_layout.setSpacing(6)
        run_gb.setLayout(run_layout)

        # 1.a) Filter by run type dropdown
        run_type_label = QLabel("Filter by run type:")
        self.run_type_cb = QComboBox()
        # Insert a “All” option at top
        self.run_type_cb.addItem("All", userData=None)
        #for name, rnum in sorted(self.run_type_map.items()):
        #    self.run_type_cb.addItem(name, userData=rnum)
        #self.run_type_cb.setCurrentIndex(0)
        #self.run_type_cb.currentIndexChanged.connect(self.on_filter_changed)

        #run_layout.addWidget(run_type_label)
        #run_layout.addWidget(self.run_type_cb)

        # 1.b) Filter by run date dropdown
        #date_filter_label = QLabel("Filter by run date:")
        #self.date_filter_cb = QComboBox()
        #for label in ("last-two-weeks", "last-month", "all"):
        #    self.date_filter_cb.addItem(label)
        #self.date_filter_cb.setCurrentIndex(0)
        #self.date_filter_cb.currentIndexChanged.connect(self.on_filter_changed)

        #run_layout.addWidget(date_filter_label)
        #run_layout.addWidget(self.date_filter_cb)
        run_layout.addWidget(date_gb)

        # 1.c) Actual run_time selector + Prev/Next buttons
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

        # 2) Molecule/Analyte Selection GroupBox
        analyte_gb = QGroupBox("Gases / Molecules")
        analyte_layout = QVBoxLayout()
        analyte_layout.setSpacing(6)
        analyte_gb.setLayout(analyte_layout)

        self.analyte_widget = QWidget()
        self.analyte_layout = QVBoxLayout()
        self.analyte_layout.setSpacing(4)
        self.analyte_widget.setLayout(self.analyte_layout)

        # If there are more than 10 analytes, we’ll switch to a QComboBox below.
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

    def on_apply_month_range(self):
        # 1) Read selection from the four combo boxes
        sy = int(self.start_year_cb.currentText())
        sm = self.start_month_cb.currentIndex() + 1   # Jan→1, Feb→2, etc.
        ey = int(self.end_year_cb.currentText())
        em = self.end_month_cb.currentIndex() + 1

        # 2) Build start/end strings
        from calendar import monthrange
        last_day = monthrange(ey, em)[1]  # e.g. 28, 29, 30, or 31
        start_sql = f"{sy:04d}-{sm:02d}-01"
        end_sql   = f"{ey:04d}-{em:02d}-{last_day:02d}"

        # 3) Reload data for the current analyte with that range
        df = self.m4.load_data(
            pnum=self.current_pnum,
            start_date=start_sql,
            end_date=end_sql
        )

        # 4) Populate run_cb just like set_current_analyte() does
        if df is not None and not df.empty:
            times = sorted(df["run_time"].unique())
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
        If there are ≤ 10 analytes → show radio buttons.
        If > 10 analytes → show a QComboBox instead.
        """
        # Clear any existing widgets in analyte_layout
        for i in reversed(range(self.analyte_layout.count())):
            w = self.analyte_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        names = list(self.analytes.keys())
        if len(names) <= 10:
            # Use radio buttons
            self.radio_group = QButtonGroup(self)
            for name in names:
                rb = QRadioButton(name)
                self.analyte_layout.addWidget(rb)
                self.radio_group.addButton(rb)
                rb.toggled.connect(self.on_analyte_radio_toggled)
            # Select the first radio button by default
            first_rb = self.radio_group.buttons()[0]
            first_rb.setChecked(True)
        else:
            # Use a QComboBox
            self.analyte_combo = QComboBox()
            for name in names:
                self.analyte_combo.addItem(name)
            self.analyte_combo.currentTextChanged.connect(self.on_analyte_combo_changed)
            self.analyte_layout.addWidget(self.analyte_combo)

    def on_analyte_radio_toggled(self):
        """
        Called whenever one of the ≤10 radio buttons toggles to “checked.”
        We only react when it becomes checked.
        """
        rb = self.sender()
        if rb.isChecked():
            name = rb.text()
            self.set_current_analyte(name)

    def on_analyte_combo_changed(self, name):
        """
        Called whenever the QComboBox selection changes (for >10 analytes).
        """
        self.set_current_analyte(name)

    def set_current_analyte(self, analyte_name):
        """
        1) Remember the current pnum
        2) Call load_data(...) with that pnum
        3) Populate run_times in the .run_cb
        4) Optionally trigger the first run to be displayed.
        """
        pnum = self.analytes[analyte_name]
        self.current_pnum = pnum

        # (Re)load data for this analyte
        df = self.processor.load_data(
            pnum=pnum
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

        # If there is at least one run_time, select the first by default
        if self.current_run_times:
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
        run_str = self.current_run_times[index]
        print(f">>> Selected run_time: {run_str}")
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


_processor_map = {
    "m4": M4_Processing,
    "fe3": FE3_Processing,
    "bld1": BLD1_Processing
}

def get_processor_for(instrument_id: str):
    """
    Look up the Processor class and the matching Instrument class, instantiate both,
    and return (processor_instance, instrument_instance).
    """
    inst = instrument_id.lower()
    if inst not in _processor_map:
        raise ValueError(
            f"Invalid instrument '{instrument_id}'. "
            f"Valid choices: {list(_processor_map.keys())}"
        )

    # 1) Instantiate the Processor:
    processor_cls = _processor_map[inst]
    processor = processor_cls()  # e.g. M4_Processing()

    # 2) Instantiate the Instrument class from logos_instruments:
    try:
        instrument_cls = getattr(li, f"{inst.upper()}_Instrument")
    except AttributeError:
        raise ValueError(
            f"Could not find class '{inst.upper()}_Instrument' in logos_instruments.py"
        )

    # Pass in instrument=instrument_id so that HATS_DB_Functions sets inst_num correctly:
    instrument = instrument_cls(instrument=instrument_id)

    return processor, instrument

def main():
    parser = argparse.ArgumentParser(description="Data Processing Application")
    parser.add_argument(
        "--instrument",
        type=str,
        choices=list(_processor_map.keys()),  # ["m4", "fe3", "bld1"]
        default="m4",
        help="Specify which instrument (m4, fe3, or bld1)"
    )
    args = parser.parse_args()

    try:
        processor, instrument = get_processor_for(args.instrument)
    except ValueError as e:
        print(e)
        sys.exit(1)

    app = QApplication(sys.argv)
    w = MainWindow(processor, instrument, args.instrument)
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()