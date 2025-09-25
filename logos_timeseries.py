from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import colorsys


def build_site_colors(sites):
    """
    Assign each site a consistent base color from a colormap.
    Always returns a mapping for *all* sites in the list.
    """
    # use a perceptually uniform colormap with enough variety
    cmap = plt.cm.get_cmap("tab20b", len(sites))
    site_colors = {}
    for i, site in enumerate(sorted(sites)):  # sort to keep stable order
        site_colors[site] = cmap(i)
    return site_colors

def adjust_brightness(color, factor=1.0):
    """Darken/lighten an RGBA color. factor < 1 darkens, > 1 lightens."""
    r, g, b, a = color
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (r, g, b, a)

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
        self.current_channel = None
        
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
                
        # Groupings
        mean_group = QGroupBox("Averaging Options")
        grps_layout = QHBoxLayout()

        self.cb_all_samples = QCheckBox("All samples")
        self.cb_all_samples.setChecked(True)

        self.cb_flask_mean = QCheckBox("Flask Mean")
        self.cb_flask_mean.setChecked(False)

        self.cb_pair_mean = QCheckBox("Pair Mean")
        self.cb_pair_mean.setChecked(False)

        grps_layout.addWidget(self.cb_all_samples)
        grps_layout.addWidget(self.cb_flask_mean)
        grps_layout.addWidget(self.cb_pair_mean)

        mean_group.setLayout(grps_layout)
        controls.addWidget(mean_group)

        # ----- Set default selection -----
        self.set_current_analyte(self.current_analyte)

        # Plot button
        self.plot_button = QPushButton("Plot it")
        self.plot_button.clicked.connect(self.make_plot)
        controls.addWidget(self.plot_button)

        controls.addStretch()
        self.setLayout(controls)

    # --- Helpers ---
    def set_current_analyte(self, analyte_name: str | None):
        if not analyte_name:  # handles None or ""
            self.current_analyte = None
            self.current_channel = None
            return

        idx = self.analyte_combo.findText(analyte_name, Qt.MatchExactly)
        if idx >= 0:
            self.analyte_combo.setCurrentIndex(idx)

        self.current_channel = None
        if "(" in analyte_name and ")" in analyte_name:
            analyte, channel = analyte_name.split("(", 1)
            self.current_channel = channel.strip(") ")

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
        channel = self.current_channel or None
        sites = [cb.text() for cb in self.site_checks if cb.isChecked()]

        if not sites or pnum is None:
            return

        ch_str = '' if channel is None else f'AND channel = {channel})'
            
        query_params = (start, end, analyte)

        # reload only if analyte/year range changed
        if query_params != self._last_query_params:
            sql = f"""
            SELECT sample_datetime, analysis_datetime, mole_fraction, channel, 
                   data_flag, site, sample_id, pair_id_num
            FROM hats.ng_data_processing_view
            WHERE inst_num = {self.instrument.inst_num}
              AND parameter_num = {pnum}
              {ch_str}
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

        datasets = {}

        if self.cb_all_samples.isChecked():
            datasets["All samples"] = df.copy()

        if self.cb_flask_mean.isChecked():
            datasets["Flask mean"] = (
                df.groupby(["site", "sample_id"])
                .agg({
                    "sample_datetime": "first",
                    "mole_fraction": ["mean", "std"]
                })
                .reset_index()
            )
            datasets["Flask mean"].columns = ["site", "sample_id", "sample_datetime", "mean", "std"]

        if self.cb_pair_mean.isChecked():
            datasets["Pair mean"] = (
                df.groupby(["site", "pair_id_num"])
                .agg({
                    "sample_datetime": "first",
                    "mole_fraction": ["mean", "std"]
                })
                .reset_index()
            )
            datasets["Pair mean"].columns = ["site", "pair_id_num", "sample_datetime", "mean", "std"]

        styles = {
            "All samples": {"marker": "o", "shade": 1.1, "error": False, "size": 4, "alpha": 0.4},
            "Flask mean": {"marker": "^", "shade": 0.8, "error": True,  "size": 6, "alpha": 0.9},
            "Pair mean":  {"marker": "s", "shade": 0.7, "error": True,  "size": 6, "alpha": 0.9},
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        site_colors = build_site_colors(self.sites)

        # Keep track of which sites already added to legend
        legend_handles = {}

        for label, dset in datasets.items():
            for site, grp in dset.groupby("site"):
                if site not in sites:
                    continue

                base = site_colors[site]
                style = styles[label]
                color = adjust_brightness(base, style["shade"])

                if label == "All samples":
                    ax.plot(
                        grp["sample_datetime"], grp["mole_fraction"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], alpha=style["alpha"]
                    )
                else:
                    ax.errorbar(
                        grp["sample_datetime"], grp["mean"], yerr=grp["std"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], capsize=2, alpha=style["alpha"],
                        mfc='none', mec=color
                    )
                # Add a single legend entry per site (on first dataset only)
                if site not in legend_handles:
                    h = ax.plot([], [], color=base, marker="o", linestyle="", label=site)[0]
                    legend_handles[site] = h

        ax.set_xlabel("Sample datetime")
        ax.set_ylabel("Mole fraction")
        ax.set_title(f"Mole fraction vs Sample datetime\nAnalyte: {analyte}")

        # --- Site legend ---
        site_legend = ax.legend(
            handles=legend_handles.values(),
            title="Site",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            columnspacing=1.0, handletextpad=0.2,
            ncol=2
        )
        ax.add_artist(site_legend)  # keep this one

        # --- Marker type key ---
        marker_key = [
            mlines.Line2D([], [], color="black", marker="o", linestyle="",
                        markersize=6, label="Flasks"),
            mlines.Line2D([], [], color="black", marker="^", linestyle="",
                        markersize=6, label="Flask Means", mfc="none", mec="black"),
            mlines.Line2D([], [], color="black", marker="s", linestyle="",
                        markersize=6, label="Pair Means", mfc="none", mec="black"),
        ]

        dataset_legend = ax.legend(
            handles=marker_key,
            #title="Dataset",
            bbox_to_anchor=(1.05, 0),   # bottom anchor
            loc="lower left",
            borderaxespad=0.0
        )
        ax.add_artist(dataset_legend)

        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show()