#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import RadioButtons
from statsmodels.nonparametric.smoothers_lowess import lowess

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QComboBox,
    QPushButton, QLabel, QMessageBox
)

from logos_instruments import M4_Instrument


class DataLoadPanel(QWidget, M4_Instrument):
    """Panel for loading data and plotting area response with LOWESS smoothing."""

    # Constants
    STANDARD_RUN_TYPE = 8
    COLOR_MAP = {
        1: "#1f77b4",  # Flask
        4: "#ff7f0e",  # Other
        5: "#2ca02c",  # PFP
        6: "#dd89f9",  # Zero
        7: "#c7811b",  # Tank
        8: "#505c5c",  # Standard
        "Response": "#e04c19",  # Response
        "Ratio": "#1f77b4",  # Ratio
        "Mole Fraction": "#2ca02c",  # Mole Fraction
    }
    RADIO_OPTIONS = ("Response", "Ratio", "Mole Fraction")

    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("Data Loader Panel")
        self.setFixedSize(200, 200)

        # Internal state
        self.data = pd.DataFrame()
        self.current_df = None
        self.fig = None
        self.ax = None

        # Build UI
        self._setup_ui()
        self._setup_signals()

        # Load default
        self.duration_combo.setCurrentText("1 month")
        self.load_data()

    def _setup_ui(self):
        # Duration selector
        self.duration_combo = QComboBox()
        self.duration_combo.addItems([
            "1 week", "1 month", "3 months", "1 year", "2 years"
        ])

        # Parameter selector
        self.parameter_combo = QComboBox()
        self._populate_parameters()

        # Plot button and status
        self.plot_button = QPushButton("Area Response")
        self.status_label = QLabel("Select options to load data.")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select Duration:"))
        layout.addWidget(self.duration_combo)
        layout.addWidget(QLabel("Select Parameter:"))
        layout.addWidget(self.parameter_combo)
        layout.addWidget(self.plot_button)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def _setup_signals(self):
        self.duration_combo.currentIndexChanged.connect(self.load_data)
        self.plot_button.clicked.connect(self.plot_data)

    def _populate_parameters(self):
        try:
            params = self.analytes
            self.parameter_combo.clear()
            for name, num in params.items():
                self.parameter_combo.addItem(name, num)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading parameters: {e}")

    def load_data(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            duration = self.duration_combo.currentText()
            name = self.parameter_combo.currentText()
            num = self.parameter_combo.currentData()

            self.status_label.setText(
                f"Loading data for {duration}, parameter: {name} (ID: {num})"
            )

            interval = {
                "1 week": "1 WEEK",
                "1 month": "1 MONTH",
                "3 months": "3 MONTH",
                "1 year": "1 YEAR",
                "2 years": "2 YEAR",
            }.get(duration, "1 MONTH")

            query = f"""
                SELECT analysis_datetime, run_time, area, run_type_num, parameter_num, mole_fraction
                FROM hats.ng_data_view
                WHERE inst_num = {self.inst_num}
                    AND analysis_datetime BETWEEN DATE_SUB(NOW(), INTERVAL {interval}) AND NOW()
                ORDER BY analysis_datetime DESC;
            """
            df = pd.DataFrame(self.db.doquery(query))
            self.data = df[df['area'] != 0].copy()
            self.status_label.setText("Data loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def calculate_lowess_smoothed(self, df, gap_hours=3, min_pts=8):
        std = df[df['run_type_num'] == self.STANDARD_RUN_TYPE][['analysis_datetime','area']].copy()
        std['analysis_datetime'] = pd.to_datetime(std['analysis_datetime'])
        std.set_index('analysis_datetime', inplace=True)
        #std.loc[std['area'] < 0.1, 'area'] = np.nan
        std.dropna(inplace=True)
        #std = std.loc[~std.index.duplicated()]
        std.sort_index(inplace=True)
        std['time_diff'] = std.index.to_series().diff()
        std['segment'] = (std['time_diff'] > pd.Timedelta(hours=gap_hours)).cumsum()
        std['smoothed'] = np.nan
        for seg, seg_df in std.groupby('segment'):
            n = len(seg_df)
            if n < min_pts:
                continue
            frac = min(max(min_pts/n, 0.3), 1.0)
            x = seg_df.index.view('int64') // 10**9
            y = seg_df['area'].values
            sm = lowess(y, x, frac=frac, return_sorted=False)
            std.loc[seg_df.index, 'smoothed'] = sm
        return std

    def plot_data(self):
        df, name, num = self._prepare_dataframe()
        if df is None:
            return

        smooth_df = self.calculate_lowess_smoothed(df)
        df = df.merge(
            smooth_df[['smoothed','segment']],
            left_index=True, right_index=True, how='left'
        )
        df['smoothed'] = df['smoothed'].interpolate(limit_direction='both')
        df['segment'] = df['segment'].interpolate(limit_direction='both')
        df['ratio'] = df['area'] / df['smoothed']
        change = df['segment'].ne(df['segment'].shift()) & df['segment'].shift().notna()
        df.loc[change, 'smoothed'] = np.nan
        self.current_df = df

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10,6))
            # leave extra room on right for widgets
            self.fig.subplots_adjust(right=0.75)
            self._add_radio_buttons(self.fig)
        else:
            self.ax.clear()

        self._draw_plot(self.ax, df, self.RADIO_OPTIONS[0])
        self._format_axes(self.ax, df, name, num)
        
        plt.show()

    def _prepare_dataframe(self):
        if self.data.empty:
            QMessageBox.warning(self, "No Data", "Data is not loaded yet.")
            return None, None, None
        name = self.parameter_combo.currentText()
        num = int(self.parameter_combo.currentData())
        df = self.data[self.data['parameter_num'] == num].copy()
        if df.empty:
            QMessageBox.information(self, "No Data", f"No data for parameter {num}")
            return None, None, None
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])
        df.set_index('analysis_datetime', inplace=True, drop=False)
        return df, name, num

    def _add_radio_buttons(self, fig):
        # dynamic width based on label length
        max_len = max(len(s) for s in self.RADIO_OPTIONS)
        width = min(0.3, 0.01 * max_len + 0.05)
        left = 1 - width - 0.02
        ax = fig.add_axes([left, 0.4, width, 0.15], facecolor='lightgoldenrodyellow')
        ax.set_navigate(False)
        self.radio = RadioButtons(ax, self.RADIO_OPTIONS)
        self.radio.on_clicked(self.update_plot)

    def _draw_plot(self, ax, df, label):
        run_map = {v:k for k,v in self.run_type_num().items()}
        for rnum, label_str in run_map.items():
            if rnum not in df['run_type_num'].unique():
                continue
            sub = df[df['run_type_num'] == rnum]
            if label == "Response":
                y = sub['area']
            elif label == "Ratio":
                y = sub['ratio']
            else:
                y = sub['mole_fraction']
            ax.scatter(sub.index, y, label=label_str,
                        color=self.COLOR_MAP.get(rnum, 'gray'))
        if label == "Response":
            ax.plot(df.index, df['smoothed'], color=self.COLOR_MAP.get('Response'), linewidth=1)
            
    def _format_axes(self, ax, df, name, num):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        lo = df.index.min().strftime('%Y-%m-%d %H:%M')
        hi = df.index.max().strftime('%Y-%m-%d %H:%M')
        ax.set_xlabel(f"Analysis Datetime ({lo} to {hi})")
        ax.set_ylabel("Value")
        ax.set_title(f"{name}:{num}")
        ax.legend()
        plt.xticks(rotation=45)

        def coord(x, y):
            try:
                ts = mdates.num2date(x).strftime('%Y-%m-%d %H:%M:%S')
                return f"{ts}, {y:.2f}"
            except:
                return f"x:{x:.2f}, y:{y:.2f}"
        ax.format_coord = coord

    def update_plot(self, label):
        if self.current_df is None or self.ax is None:
            return
        x_min, x_max = self.ax.get_xlim()
        self.ax.clear()
        df = self.current_df
        name = self.parameter_combo.currentText()
        num = self.parameter_combo.currentData()
        self._draw_plot(self.ax, df, label)
        self._format_axes(self.ax, df, name, num)
        
        self.ax.relim()
        self.ax.autoscale_view(scalex=False, scaley=True)
        #self.ax.margins(y=0.05)
        self.ax.set_xlim(x_min, x_max)
        
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    
    app = QApplication(sys.argv)
    panel = DataLoadPanel()
    panel.show()
    sys.exit(app.exec_())
