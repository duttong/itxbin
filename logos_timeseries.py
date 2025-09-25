from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import pandas as pd


class TimeseriesWidget(QWidget):
    def __init__(self, instrument=None, parent=None):
        """
        analytes: dict mapping analyte name -> parameter_num
        sites: iterable of available site codes
        instrument: object with .doquery(sql) method
        """
        super().__init__(parent)
        self.instrument = instrument
        self.analytes = self.instrument.analytes or {}
        self.current_analyte = None
        sites = ['SUM', 'PSA', 'SPO', 'SMO', 'AMY', 'MKO', 'ALT', 'CGO', 'NWR',
                'LEF', 'BRW', 'RPB', 'KUM', 'MLO', 'WIS', 'THD', 'MHD', 'HFM',
                'BLD', 'MKO']
        self.sites = sorted(sites) if sites else []
       
        # cache for last loaded data
        self._cached_df = None
        self._last_query_params = None  # (start, end, analyte)

        # --- Main layout ---
        controls = QVBoxLayout()

        # Year range selection
        i_start = int(self.instrument.start_date[0:4]) if self.instrument else 2020
        i_end = pd.Timestamp.now().year
        year_group = QGroupBox("Year Range")
        year_layout = QHBoxLayout()
        self.start_year = QSpinBox()
        self.start_year.setRange(i_start, i_end)
        self.start_year.setValue(i_end - 2)
        self.end_year = QSpinBox()
        self.end_year.setRange(i_start, i_end)
        self.end_year.setValue(i_end)
        year_layout.addWidget(QLabel("Start"))
        year_layout.addWidget(self.start_year)
        year_layout.addWidget(QLabel("End"))
        year_layout.addWidget(self.end_year)
        year_group.setLayout(year_layout)
        controls.addWidget(year_group)

        # --- Analyte selector ---
        analyte_group = QGroupBox("Analyte")
        analyte_layout = QVBoxLayout()
        self.analyte_combo = QComboBox()

        # Populate using names only
        for name in self.analytes.keys():
            self.analyte_combo.addItem(name)

        analyte_layout.addWidget(self.analyte_combo)
        analyte_group.setLayout(analyte_layout)
        controls.addWidget(analyte_group)

        # --- Site selection with 3 columns ---
        site_group = QGroupBox("Sites")
        site_layout = QVBoxLayout()
        grid = QGridLayout()
        self.site_checks = []
        
        initial_sites = ['BRW', 'MLO', 'SMO', 'SPO']  # default ON
        cols = 3
        for i, site in enumerate(self.sites):
            cb = QCheckBox(site)
            cb.setChecked(site in initial_sites)   # ✅ only check if in subset
            self.site_checks.append(cb)
            row, col = divmod(i, cols)
            grid.addWidget(cb, row, col)
        site_layout.addLayout(grid)

        btns_layout = QHBoxLayout()
        select_all = QPushButton("Select All")
        select_none = QPushButton("Select None")
        select_all.clicked.connect(self.select_all_sites)
        select_none.clicked.connect(self.select_none_sites)
        btns_layout.addWidget(select_all)
        btns_layout.addWidget(select_none)
        site_layout.addLayout(btns_layout)

        site_group.setLayout(site_layout)
        controls.addWidget(site_group)

        # ----- Set default selection -----
        self.set_current_analyte(self.current_analyte)

        # Plot button
        self.plot_button = QPushButton("Plot it")
        self.plot_button.clicked.connect(self.make_plot)
        controls.addWidget(self.plot_button)

        controls.addStretch()
        self.setLayout(controls)

    # --- Helpers ---
    def set_current_analyte(self, analyte_name: str):
        idx = self.analyte_combo.findText(analyte_name, Qt.MatchExactly)
        if idx >= 0:
            self.analyte_combo.setCurrentIndex(idx)
        self.current_analyte = analyte_name
        
    def select_all_sites(self):
        for cb in self.site_checks:
            cb.setChecked(True)

    def select_none_sites(self):
        for cb in self.site_checks:
            cb.setChecked(False)

    def make_plot(self):
        start = self.start_year.value()
        end = self.end_year.value()
        analyte = self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        sites = [cb.text() for cb in self.site_checks if cb.isChecked()]

        if not sites or pnum is None:
            return

        query_params = (start, end, analyte)

        # reload only if analyte/year range changed
        if query_params != self._last_query_params:
            sql = f"""
            SELECT sample_datetime, analysis_datetime, mole_fraction,
                   data_flag, site, sample_id, pair_id_num
            FROM hats.ng_data_processing_view
            WHERE inst_num = {self.instrument.inst_num}
              AND parameter_num = {pnum}
              AND YEAR(sample_datetime) BETWEEN {start} AND {end}
            ORDER BY sample_datetime;
            """
            df = pd.DataFrame(self.instrument.doquery(sql))
            if df.empty:
                print("No data returned")
                return
            df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])

            self._cached_df = df
            self._last_query_params = query_params
        else:
            df = self._cached_df

        if df is None or df.empty:
            print("No cached data")
            return

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        for site, grp in df.groupby("site"):
            if site not in sites:
                continue
            ax.plot(
                grp["sample_datetime"], grp["mole_fraction"],
                marker="o", linestyle="", label=site, alpha=0.7
            )

        ax.set_xlabel("Sample datetime")
        ax.set_ylabel("Mole fraction")
        ax.set_title(f"Mole fraction vs Sample datetime\nAnalyte: {analyte}")
        ax.legend(title="Site", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
