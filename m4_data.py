#! /usr/bin/env python

import sys
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QMessageBox
)

import m4_export


class DataLoadPanel(QWidget, m4_export.M4_base):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create UI elements.
        self.duration_combo = QComboBox()
        self.parameter_combo = QComboBox()
        self.plot_button = QPushButton("Area Response")
        self.status_label = QLabel("Select options to load data.")

        # Set up the duration list.
        self.duration_options = ["1 week", "1 month", "3 month", "1 year", "2 year"]
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

        # Automatically load data for the default duration ("1 week")
        self.load_data()

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

            sql_query = f"""
                SELECT analysis_num, analysis_datetime, sample_id, site_num, site, sample_type, port, standards_num, run_type_num,
                       port_info, flask_port, pair_id_num, ccgg_event_num, sample_datetime, parameter, parameter_num, 
                       height, area, retention_time, mole_fraction, unc, qc_status, flag
                FROM hats.ng_data_view
                WHERE inst_num = {self.inst_num}
                  AND analysis_datetime BETWEEN DATE_SUB(NOW(), INTERVAL {duration}) AND NOW()
                ORDER BY analysis_datetime DESC;"""
                
            # Execute the SQL query and store the result in self.data.
            self.data = pd.DataFrame(self.db.doquery(sql_query))
            self.status_label.setText("Data loaded successfully.")
            print("Loaded data:")
            print(self.data.tail())
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def plot_data(self):
        """Plot area response for the selected parameter using parameter_num filtering and auto-activate zoom."""
        # Ensure data is loaded.
        if not hasattr(self, "data") or self.data.empty:
            QMessageBox.warning(self, "No Data", "Data is not loaded yet.")
            return

        # Retrieve the selected parameter number from the combo box.
        selected_param_num = self.parameter_combo.itemData(self.parameter_combo.currentIndex())

        # Filter the loaded data using parameter_num.
        df = self.data.loc[self.data['parameter_num'] == int(selected_param_num)].copy()
        if df.empty:
            QMessageBox.information(self, "No Data", f"No data found for parameter number: {selected_param_num}")
            return

        # Convert analysis_datetime to datetime objects.
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'])

        # Create a scatter plot.
        ax = df.plot(x='analysis_datetime', y='area', c='run_type_num',
                    cmap='rainbow', kind='scatter')
        ax.set_xlabel("Analysis Datetime")
        ax.set_ylabel("Area")
        ax.set_title(f"Area vs. Analysis Datetime for Parameter ID {selected_param_num}")

        # Get the figure object.
        fig = ax.get_figure()

        # Define a helper function to activate the zoom tool.
        from PyQt5.QtCore import QTimer

        def activate_zoom():
            try:
                toolbar = fig.canvas.toolbar
                if toolbar is not None:
                    toolbar.zoom()  # This toggles the zoom mode on the toolbar.
            except Exception as e:
                print("Unable to activate zoom mode:", e)

        # Use a QTimer to delay the activation slightly to ensure the toolbar is ready.
        QTimer.singleShot(100, activate_zoom)

        # Finally, display the plot.
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    panel = DataLoadPanel()
    panel.resize(600, 400)
    panel.show()
    sys.exit(app.exec_())