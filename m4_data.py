#! /usr/bin/env python

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import to_hex
import matplotlib.dates as mdates
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QMessageBox, QMainWindow
)

import m4_export


class DataLoadPanel(QWidget, m4_export.M4_base):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Define a constant for the standards or reference run
        self.STANDARD_RUN_TYPE_NUM = 8

        # Create UI elements.
        self.duration_combo = QComboBox()
        self.parameter_combo = QComboBox()
        self.plot_button = QPushButton("Area Response")
        self.status_label = QLabel("Select options to load data.")

        # Set up the duration list.
        self.duration_options = ["1 week", "1 month", "3 months", "1 year", "2 years"]
        self.duration_combo.addItems(self.duration_options)

        # Populate the parameter combo from the database.
        self.populate_parameters()

        # Connect signals.
        self.duration_combo.currentIndexChanged.connect(self.load_data)
        self.plot_button.clicked.connect(self.plot_data)

        # Layout the widgets.
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select Duration:"))
        layout.addWidget(self.duration_combo)
        layout.addWidget(QLabel("Select Parameter:"))
        layout.addWidget(self.parameter_combo)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        self.setWindowTitle("Data Loader Panel")
        self.setFixedSize(200, 200)  # or use resize(200, 200)

        # Automatically load data for the default duration ("1 month")
        self.duration_combo.setCurrentText("1 month")
        #self.load_data()

    def populate_parameters(self):
        """Fetch parameters from the database and populate the combo box."""
        try:
            mols = self.m4_analytes()
            self.parameter_combo.clear()
            for display_name, param_num in mols.items():
                # Add display name to combo and store param_num as user data.
                self.parameter_combo.addItem(display_name, param_num)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading parameters: {e}")

    def load_data(self):
        """Load data based on the current duration and store it in self.data."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Retrieve the selected duration and parameter info.
            duration = self.duration_combo.currentText()
            param_index = self.parameter_combo.currentIndex()
            param_name = self.parameter_combo.currentText()
            param_num = self.parameter_combo.itemData(param_index)

            self.status_label.setText(
                f"Loading data for duration: {duration}, parameter: {param_name} (ID: {param_num})"
            )

            # Map duration strings to valid SQL intervals.
            duration_mapping = {
                "1 week": "1 WEEK",
                "1 month": "1 MONTH",
                "3 months": "3 MONTH",
                "1 year": "1 YEAR",
                "2 years": "2 YEAR"
            }
            sql_duration = duration_mapping.get(duration, "1 MONTH")

            sql_query = f"""
                SELECT analysis_num, analysis_datetime, sample_id, site_num, site, sample_type, port, standards_num, run_type_num,
                       port_info, flask_port, pair_id_num, ccgg_event_num, sample_datetime, parameter, parameter_num, 
                       height, area, retention_time, mole_fraction, unc, qc_status, flag
                FROM hats.ng_data_view
                WHERE inst_num = {self.inst_num}
                  AND analysis_datetime BETWEEN DATE_SUB(NOW(), INTERVAL {sql_duration}) AND NOW()
                ORDER BY analysis_datetime DESC;"""
                
            # Execute the SQL query and store the result in self.data.
            self.data = pd.DataFrame(self.db.doquery(sql_query))
            
            # Remove rows where height, area, and retention_time are all 0
            self.data = self.data.loc[~((self.data['height'] == 0) & (self.data['area'] == 0) & (self.data['retention_time'] == 0))]
            
            self.status_label.setText("Data loaded successfully.")
            print("Loaded data:")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def calculate_lowess_smoothed(self, data, gap_hours=3, min_points=8):
        """
        Calculate a smoothed LOWESS function through the data for a specific run_type_num.

        Parameters:
            data (pd.DataFrame): The input DataFrame containing the data.
            run_type_num (int): The run_type_num to filter (default is 8 for standards run).
            gap_hours (int): The threshold for gaps in hours to reset smoothing.
            min_points (int): Minimum number of points required for smoothing.

        Returns:
            pd.DataFrame: A DataFrame with the original and smoothed data.
        """
        # Filter data for the specified run_type_num
        resp = data.loc[data['run_type_num'] == self.STANDARD_RUN_TYPE_NUM][['analysis_datetime', 'area']].copy()
        resp.set_index('analysis_datetime', inplace=True)
        resp.loc[resp['area'] < 0.1, 'area'] = np.nan  # Set area < 0.1 to NaN
        resp = resp.dropna()  # Drop NaN values
        
        # Ensure unique indices by resetting the index and reassigning it
        resp = resp.reset_index().drop_duplicates(subset='analysis_datetime').set_index('analysis_datetime')

        # Ensure the index is datetime and sorted
        resp = resp.sort_index()
        resp.index = pd.to_datetime(resp.index)

        # Compute time difference between measurements
        resp['time_diff'] = resp.index.to_series().diff()

        # Define threshold for gap
        gap_threshold = pd.Timedelta(hours=gap_hours)

        # Create a segment ID that increments after a gap
        resp['segment'] = (resp['time_diff'] > gap_threshold).cumsum()

        # Container for smoothed values
        resp['smoothed'] = np.nan
        
        # Apply LOWESS within each segment
        for seg_id, seg_data in resp.groupby('segment'):
            n = len(seg_data)
            if n < min_points:
                continue
            # Use either 30% or enough to hit min_points
            frac = max(min_points / n, 0.3)
            frac = min(frac, 1.0)  # frac must be <= 1.0

            # Ensure the index is explicitly converted to a DatetimeIndex
            seg_data.index = pd.to_datetime(seg_data.index)

            # Convert index to numeric timestamps for LOWESS calculation
            x = seg_data.index.view('int64') // int(1e9)  # Convert to seconds since epoch
            y = seg_data['area'].values
            smoothed = lowess(y, x, frac=frac, return_sorted=False)
            
            # Debugging: Print lengths of seg_data.index and smoothed
            #print(f"Segment {seg_id}: len(seg_data.index) = {len(seg_data.index)}, len(smoothed) = {len(smoothed)}")

            # Ensure lengths of seg_data.index and smoothed match before assignment
            if len(seg_data.index) == len(smoothed):
                # Align indices explicitly before assignment to avoid mismatches
                smoothed_series = pd.Series(smoothed, index=seg_data.index)
                resp.loc[seg_data.index, 'smoothed'] = smoothed_series
            else:
                print(f"Warning: Length mismatch for segment {seg_id}. Skipping smoothing for this segment.")

        return resp

    def plot_data(self):
        """Plot area response for the selected parameter using parameter_num filtering and auto-activate zoom."""
        # Ensure data is loaded.
        if not hasattr(self, "data") or self.data.empty:
            QMessageBox.warning(self, "No Data", "Data is not loaded yet.")
            return

        # Retrieve the selected parameter number from the combo box.
        selected_param_name = self.parameter_combo.currentText()
        selected_param_num = self.parameter_combo.itemData(self.parameter_combo.currentIndex())

        # Filter the loaded data using parameter_num.
        df = self.data.loc[self.data['parameter_num'] == int(selected_param_num)].copy()
        if df.empty:
            QMessageBox.information(self, "No Data", f"No data found for parameter number: {selected_param_num}")
            return

        # smooth the data using LOWESS for parameter_num
        smoothed_data = self.calculate_lowess_smoothed(df, gap_hours=3, min_points=8)
        df.set_index('analysis_datetime', inplace=True, drop=False)
        df = pd.merge(df, smoothed_data[['smoothed', 'segment']], left_index=True, right_index=True, how='left')
        df['smoothed'] = df['smoothed'].interpolate(method='linear', limit_direction='both')
        df['segment'] = df['segment'].interpolate(method='linear', limit_direction='both')

        # create a gap between segments
        is_transition = df['segment'].ne(df['segment'].shift())
        is_transition &= df['segment'].shift().notna()
        df.loc[is_transition, 'smoothed'] = np.nan
        
        df['ratio_a'] = df['area'] / df['smoothed']
        
        # Swap keys and values in run_type_map
        run_type_map = {v: k for k, v in self.run_type_num().items()}  # {"Standard": 1, "Sample": 2, ...}

        # Hardcode colors for specific run_type_num values
        color_map = {
            1: "#1f77b4",  # Blue for Flask
            4: "#ff7f0e",  # Orange for Other
            5: "#2ca02c",  # Green for PFP
            6: "#dd89f9",  # Purple for Zero
            7: "#c7811b",  # Orange for Tank
            8: "#505c5c"   # Dark brown for Standard
        }

        # Filter out rnums not present in the dataframe before plotting
        filtered_run_type_map = {rnum: rlabel for rnum, rlabel in run_type_map.items() if rnum in df['run_type_num'].unique()}

        # Create a single scatter plot with different colors for each run_type_num.
        # Set the default figure size to be rectangular and adjust layout to fit the xlabel
        fig, ax = plt.subplots(figsize=(10, 6))  # Set a rectangular figure size
        for rnum, rlabel in filtered_run_type_map.items():
            sub = df.loc[df['run_type_num'] == rnum]
            ax.scatter(sub['analysis_datetime'], sub['area'], label=rlabel, color=color_map[rnum])
        
        # Plot the smoothed data
        ax.plot(df.index, df['smoothed'], color='#e04c19', linewidth=1)

        # Set the major and minor locators for better scaling
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show only time in the x-axis labels

        # Add redundant date info to the x-axis label
        date_range = f"{df['analysis_datetime'].min().strftime('%Y-%m-%d')} to {df['analysis_datetime'].max().strftime('%Y-%m-%d')}"
        ax.set_xlabel(f"Analysis Datetime ({date_range})")

        ax.set_ylabel("Area")
        ax.set_title(f"{selected_param_name}:{selected_param_num} Area")
        ax.legend(title="Run Type")

        # Rotate the x-axis date labels to 45 degrees
        plt.xticks(rotation=45)

        # Update x-axis dynamically when the user releases the mouse button
        def update_xlabel_on_release(event):
            if event.button in [1, 3]:  # Left or right mouse button
                x_min, x_max = ax.get_xlim()
                x_min_date = mdates.num2date(x_min).strftime('%Y-%m-%d %H:%M')
                x_max_date = mdates.num2date(x_max).strftime('%Y-%m-%d %H:%M')
                ax.set_xlabel(f"Analysis Datetime ({x_min_date} to {x_max_date})")
                fig.canvas.draw_idle()

        # Connect the 'button_release_event' to the update function
        fig.canvas.mpl_connect('button_release_event', update_xlabel_on_release)

        # Update the toolbar to show full date and time when hovering over a point
        def format_coord(x, y):
            try:
                date_str = mdates.num2date(x).strftime('%Y-%m-%d %H:%M:%S')
                return f"Datetime: {date_str}, Area: {y:.2f}"
            except Exception:
                return f"x: {x:.2f}, y: {y:.2f}"

        ax.format_coord = format_coord

        # Automatically activate the zoom tool when the figure is created
        def activate_zoom():
            try:
                toolbar = fig.canvas.toolbar
                if toolbar is not None:
                    toolbar.zoom()  # This toggles the zoom mode on the toolbar.
            except Exception as e:
                print("Unable to activate zoom mode:", e)

        # Use a QTimer to delay the activation slightly to ensure the toolbar is ready
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(100, activate_zoom)

        # Revert back to using plt.show() to display the plot
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = DataLoadPanel()
    panel.show()
    sys.exit(app.exec_())