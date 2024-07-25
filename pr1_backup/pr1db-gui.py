#! /usr/bin/env python

import sys
import time
from datetime import date
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QComboBox, QCheckBox, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QListWidget, 
                             QMessageBox, QTextEdit, QProgressBar)
from PyQt5.QtGui import QTextCursor, QFont, QKeySequence

from pr1_export import PR1_GCwerks_Export
from pr1_gcwerks2db import PR1_db


class PR1_DBGUI():
    def __init__(self):
        #super().__init__()
        self.pr1db = PR1_db()

        app = QApplication(sys.argv)

        # Set the look and feel of the app
        app.setStyleSheet("""
            QWidget {
                background-color: mistyrose;
            }
            QLabel {
                font-family: Helvetica;
                font-size: 16px;
            }
            QPushButton {
                font-family: Helvetica;
                font-size: 16px;
                background-color: lightgrey;
                border: 2px solid black;
            }
            QProgressBar {
                font-family: Helvetica;
                font-size: 16px;
                background-color: lightcoral;
            }
            QCheckBox {
                font-family: Helvetica;
                font-size: 16px;
            }
            QListWidget {
                font-family: Helvetica;
                font-size: 12px;
                background-color: seashell;
            }
            QListWidget::item:selected {
                background-color: lightcoral;
                color: white;
            }
            QComboBox {
                font-family: Helvetica;
                font-size: 16px;
            }
            QComboBox::item:selected {
                background-color: lightcoral;
                color: white;
            }
        """)

        window = QWidget()

        # Set the title and initial size of the main window
        window.setWindowTitle('Persus (PR1) HATS DB Update')
        window.setGeometry(100, 100, 600, 400)

        # Create a layout
        layout = QVBoxLayout()

        # Label and list widget for gas selection
        gas_label = QLabel('Select Analytes (ctrl-a for all)')
        layout.addWidget(gas_label)
        self.gas_list = MyListWidget()
        self.gas_list.setSelectionMode(QListWidget.MultiSelection)
        gases = sorted(self.pr1db.analytes)
        try:
            gases.remove('1,2-DCE')
        except ValueError:
            pass
        self.gas_list.addItems(gases)
        layout.addWidget(self.gas_list)

        # Connect the doubleClicked signal to the slot
        self.gas_list.doubleClicked.connect(self.clear_selection)

        # ComboBox for date range selection
        date_label = QLabel('Select Date Range:')
        layout.addWidget(date_label)
        self.date_combo = QComboBox()
        self.date_combo.addItems(['Last Month', 'Last Year', 'Last Two Years', 'Last Three Years', 'All Data'])
        layout.addWidget(self.date_combo)

        # Checkbox for "extract gcwerks first"
        self.extract_checkbox = QCheckBox('Re-extract from GCwerks First')
        layout.addWidget(self.extract_checkbox)

        # Execute button
        self.execute_button = QPushButton('Execute DB Update')
        self.execute_button.clicked.connect(self.execute_process)
        layout.addWidget(self.execute_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0,100)
        layout.addWidget(self.progress_bar)
        
        # TextEdit for stdout display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        layout.addWidget(self.output_display)

        # Set the layout to the main window
        window.setLayout(layout)
        window.show()
        sys.exit(app.exec_())

    def clear_selection(self):
        self.gas_list.clearSelection()

    def execute_process(self):
        # Redirect stdout
        sys.stdout = self.OutputWrapper(self.output_display)

        # Get selected gases
        selected_gases = [item.text() for item in self.gas_list.selectedItems()]

        # Get selected date range
        selected_date_range = self.date_combo.currentText()
        today = date.today()
        end_dt = pd.to_datetime(today)
        
        if selected_date_range == 'Last Month':
            start_dt = end_dt - pd.DateOffset(months=1)
        elif selected_date_range == 'Last Year':
            start_dt = end_dt - pd.DateOffset(years=1)
        elif selected_date_range == 'Last Two Years':
            start_dt = end_dt - pd.DateOffset(years=2)
        elif selected_date_range == 'Last Three Years':
            start_dt = end_dt - pd.DateOffset(years=3)
        elif selected_date_range == 'All Data':
            start_dt = pd.to_datetime(self.pr1db.pr1_start_date)

        def progress_callback(progress):
            self.progress_bar.setValue(progress)

        # Check if "extract gcwerks first" is checked
        extract_first = self.extract_checkbox.isChecked()
        if extract_first:
            PR1_GCwerks_Export().export_gc_data(start_dt, selected_gases, progress=progress_callback)

        for n, gas in enumerate(selected_gases):
            progress = int(n/len(selected_gases)*100)
            progress_callback(progress)
            print(f'Loading {gas}')
            df = self.pr1db.load_gcwerks(gas, start_dt)
            self.pr1db.tmptbl_fill(df)             # create and fill in temp data table with GCwerks results
            self.pr1db.tmptbl_update_analysis()    # insert and update any rows in hats.analysis with new data
            self.pr1db.tmptbl_update_raw_data()    # update the hats.raw_data table with area, ht, w, rt

        # Clear the selection in the gas list
        self.gas_list.clearSelection()
        print('DONE\n')
        progress_callback(100)
        time.sleep(2)
        progress_callback(0)


    class OutputWrapper:
        def __init__(self, text_edit):
            self.text_edit = text_edit

        def write(self, message):
            # Append message without adding extra line feed
            self.text_edit.moveCursor(QTextCursor.End)
            self.text_edit.insertPlainText(message)
            self.text_edit.ensureCursorVisible()

        def flush(self):
            pass


class MyListWidget(QListWidget):
    def keyPressEvent(self, event):
        if event.matches(QKeySequence.SelectAll):
            self.selectAll()
        else:
            super().keyPressEvent(event)


def get_default_date():
    return (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')


def convert_date_format(date_str):
    """
    Converts a date string from 'YYYYMMDD' or 'YYYYMMDD.HHMM' format to 'YYMM' format.
    If the date is already in 'YYMM' format, it returns the date as is.

    Parameters:
    date_str (str): The date string in 'YYYYMMDD', 'YYYYMMDD.HHMM', or 'YYMM' format.

    Returns:
    str: The date string in 'YYMM' format.
    """
    # Check if the length of the date string is 4 and assume it is in 'YYMM' format
    if len(date_str) == 4:
        return date_str
    
    # Extract the first 6 characters which correspond to 'YYYYMM'
    yyyymm = date_str[:6]
    
    # Convert to 'YYMM'
    yymm = yyyymm[2:]
    
    return yymm


def parse_molecules(molecules):
    if molecules:
        try:
            molecules = molecules.replace('1,2-DCE', '12-DCE')
            molecules = molecules.replace(' ','')   # remove spaces
            return molecules.split(',')
        except AttributeError:      # already a list. just return
            return molecules
    return []


def main():
    pr1 = PR1_db()

    parser = argparse.ArgumentParser(description='Insert Perseus GCwerks data into HATS db for selected date range. If no start_date \
                                     is specifide then work on the last 30 days of data.')
    parser.add_argument('date', nargs='?', default=get_default_date(), help='Date in the format YYYYMMDD or YYYYMMDD.HHMM')
    parser.add_argument('-m', '--molecules', type=str, default='All',
                        help='Comma-separated list of molecules. Add quotes around the list if spaces are used. Default all molecules.')
    parser.add_argument('-x', '--extract', action='store_true', help='Re-extract data from GCwerks first.')
    parser.add_argument('--list', action='store_true', help='List all available molecule names.')

    args = parser.parse_args()
    # Check if the program was called with no arguments
    if args.date == get_default_date() and not any([args.molecules != 'All', args.extract, args.list]):
        # start GUI
        gui = PR1_DBGUI()

    if args.list:
        molecules_c = [m.replace(',', '') for m in pr1.molecules]       # remove commas from mol names
        print(f"Valid molecule names: {', '.join(molecules_c)}")
        quit()
    
    start_date = args.date
    if args.molecules == 'All':
        molecules = pr1.molecules
    else:
        molecules = parse_molecules(args.molecules)     # returns a list of molecules

    print(f"Start date: {start_date}")
    print("Processing the following molecules: ", molecules)

    if args.extract:
        yymm = convert_date_format(start_date)
        PR1_GCwerks_Export().export_gc_data(yymm, molecules)
    
    quit()
    for gas in molecules:
        df = pr1.load_gcwerks(gas, start_date)
        #print(df.loc[df['ht'].astype(float) == 64508.90])
        pr1.tmptbl_fill(df)             # create and fill in temp data table with GCwerks results
        #tmp = pr1.tmptbl_output()
        #print(tmp.loc[tmp.analysis_num == 314206])
        pr1.tmptbl_update_analysis()    # insert and update any rows in hats.analysis with new data
        pr1.tmptbl_update_raw_data()    # update the hats.raw_data table with area, ht, w, rt


if __name__ == '__main__':
    main()