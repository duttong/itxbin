from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip
)
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import colorsys


LOGOS_sites = ['SUM', 'PSA', 'SPO', 'SMO', 'AMY', 'MKO', 'ALT', 'CGO', 'NWR',
            'LEF', 'BRW', 'RPB', 'KUM', 'MLO', 'WIS', 'THD', 'MHD', 'HFM',
            'BLD', 'MKO']


def build_site_colors(sites):
    """
    Assign each site a consistent base color from a colormap.
    Always returns a mapping for *all* sites in the list.
    """
    # use a perceptually uniform colormap with enough variety
    cmap = plt.cm.get_cmap("jet", len(sites))
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
        super().__init__(parent)
        self.instrument = instrument
        self.main_window = self.parent()

        self.analytes = self.instrument.analytes or {}
        self.current_analyte = None
        self.current_channel = None

        if self.instrument.inst_num == 193:     # FE3
            self.current_analyte = 'CFC11 (c)'
            self.current_channel = 'c'
        self.dataset_handles = {}
        
        self.sites = sorted(LOGOS_sites) if LOGOS_sites else []
        self.sites_df = self.get_site_info() if self.instrument else pd.DataFrame()
        self.sites_by_lat = self.sites_df.sort_values("lat", ascending=False)["code"].tolist()
       
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
        for i, site in enumerate(self.sites_by_lat):
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
        
        # ------ Flagging options ------
        self.hide_flagged = QCheckBox("Hide flagged data")
        self.hide_flagged.setChecked(True)
        controls.addWidget(self.hide_flagged)

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
        
    def get_site_info(self):
        sql = f""" SELECT 
            code, lat, lon, elev from gmd.site
            WHERE code in {tuple(LOGOS_sites)}
            ORDER BY code;
            """
        df = pd.DataFrame(self.instrument.doquery(sql))
        return df        
        
    def select_all_sites(self):
        for cb in self.site_checks:
            cb.setChecked(True)

    def select_none_sites(self):
        for cb in self.site_checks:
            cb.setChecked(False)

    def _is_visible(self, handle):
        return handle.get_visible()

    def _disable_pan_zoom(self, toolbar):
        """Turn off pan/zoom if active (Qt toolbar safe)."""
        # Qt backends store QAction objects in _actions
        actions = getattr(toolbar, "_actions", {})

        # Disable pan if it's checked
        pan_action = actions.get("pan")
        if pan_action is not None and pan_action.isChecked():
            toolbar.pan()  # toggles it OFF

        # Disable zoom if it's checked
        zoom_action = actions.get("zoom")
        if zoom_action is not None and zoom_action.isChecked():
            toolbar.zoom()  # toggles it OFF

    def on_legend_pick(self, event):
        """Handle clicks on dataset legend entries only."""
        legend_line = event.artist

        # Only respond to legend handles we flagged
        if not isinstance(legend_line, mlines.Line2D):
            return
        if not getattr(legend_line, "_is_dataset_legend", False):
            return

        label = legend_line.get_label()
        if label not in self.dataset_handles or not self.dataset_handles[label]:
            return

        visible = not self.dataset_handles[label][0].get_visible()
        for h in self.dataset_handles[label]:     # <-- now indented inside the guard
            h.set_visible(visible)

        legend_line.set_alpha(1.0 if visible else 0.2)
        event.canvas.draw_idle()
                 
    def make_plot(self):
        start = self.start_year.value()
        end = self.end_year.value()
        analyte = self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        self.set_current_analyte(analyte)
        channel = self.current_channel
        sites = [cb.text() for cb in self.site_checks if cb.isChecked()]
        site_colors = build_site_colors(self.sites_by_lat)

        if not sites or pnum is None:
            return

        # channel filter string for sql query
        ch_str = '' if channel is None else f'AND channel = "{channel}"'

        query_params = (start, end, analyte)

        # reload only if analyte/year range changed
        if query_params != self._last_query_params:
            sql = f"""
            SELECT sample_datetime, run_time, analysis_datetime, mole_fraction, channel, 
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

        # --- Build datasets ---
        datasets = {}
        
        # chose to show or hide flagged data in "All samples"
        if self.hide_flagged.isChecked():
            # only unflagged data
            datasets["All samples"] = df[df["data_flag"] == "..."].copy()
        else:
            datasets["All samples"] = df.copy()

        # filter flagged rows (this is always done for means)
        clean = df[df["data_flag"] == "..."]

        datasets["Flask mean"] = (
            clean.groupby(["site", "sample_id", "sample_datetime"])
                .agg({"mole_fraction": ["mean", "std"]})
                .reset_index()
        )
        datasets["Flask mean"].columns = ["site", "sample_id", "sample_datetime", "mean", "std"]

        datasets["Pair mean"] = (
            clean.groupby(["site", "pair_id_num"])
                .agg({"sample_datetime": "first", "mole_fraction": ["mean", "std"]})
                .reset_index()
        )
        datasets["Pair mean"].columns = ["site", "pair_id_num", "sample_datetime", "mean", "std"]

        styles = {
            "All samples": {"marker": "o", "shade": 1.1, "error": False, "size": 3, "alpha": 0.4},
            "Flask mean": {"marker": "^", "shade": 0.8, "error": True,  "size": 5, "alpha": 0.9},
            "Pair mean":  {"marker": "s", "shade": 0.5, "error": True,  "size": 6, "alpha": 0.9},
        }

        fig, ax = plt.subplots(figsize=(12, 6))
        self.dataset_handles = {}  # reset

        # --- Plot datasets ---
        for label, dset in datasets.items():
            for site, grp in dset.groupby("site"):
                if site not in sites:
                    continue

                base = site_colors[site]
                style = styles[label]
                color = adjust_brightness(base, style["shade"])

                if label == "All samples":
                    line, = ax.plot(
                        grp["sample_datetime"], grp["mole_fraction"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], alpha=style["alpha"],
                        label=label,
                        mfc=color, mec='gray',
                        picker=5  # 5 points tolerance (pixels)
                    )
                    # attach metadata
                    line._meta = {
                        "run_time": grp.get("run_time", pd.Series([None]*len(grp))).tolist(),
                        "sample_id": grp.get("sample_id", pd.Series([None]*len(grp))).tolist(),
                        "pair_id_num": grp.get("pair_id_num", pd.Series([None]*len(grp))).tolist(),
                        "site": grp.get("site", pd.Series([None]*len(grp))).tolist(),
                        "analyte": analyte,
                        "channel": channel,
                    }
                    self.dataset_handles.setdefault(label, []).append(line)
                else:
                    container = ax.errorbar(
                        grp["sample_datetime"], grp["mean"], yerr=grp["std"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], capsize=2, alpha=style["alpha"],
                        mfc='none', mec=color, label=label
                    )

                    # Flatten container parts into list of artists
                    parts = []
                    for child in container:
                        if isinstance(child, (list, tuple)):
                            parts.extend(child)
                        else:
                            parts.append(child)
                    # Only keep objects that have set_visible
                    parts = [p for p in parts if hasattr(p, "set_visible")]

                    self.dataset_handles.setdefault(label, []).extend(parts)

        for h in self.dataset_handles.get("Flask mean", []):
            h.set_visible(False)
        for h in self.dataset_handles.get("Pair mean", []):
            h.set_visible(False)
    
        # --- Dataset legend (toggleable) ---
        legend_handles = []
        for label in datasets.keys():
            dummy = mlines.Line2D(
                [], [], color="black",
                marker={"All samples": "o", "Flask mean": "^", "Pair mean": "s"}[label],
                linestyle="", markersize=6, label=label
            )
            dummy._is_dataset_legend = True
            legend_handles.append(dummy)

        dataset_legend = ax.legend(
            handles=legend_handles,
            title="Datasets",
            bbox_to_anchor=(1.05, 0),
            loc="lower left",
            borderaxespad=0.0
        )
        ax.add_artist(dataset_legend)

        # Make dummy legend handles clickable
        for legline in dataset_legend.legendHandles:
            label = legline.get_label()
            if label in ["Flask mean", "Pair mean"]:  # start greyed out
                legline.set_alpha(0.4)
            legline.set_picker(5)
            legline._is_dataset_legend = True

        fig.canvas.mpl_connect("pick_event", self.on_legend_pick)
        fig.canvas.mpl_connect("pick_event", self.on_point_pick)

        # --- Site legend ---
        site_handles = {}
        for site in sites:
            site_handles[site] = ax.plot([], [], color=site_colors[site], marker="o", linestyle="", label=site)[0]

        site_legend = ax.legend(
            handles=site_handles.values(),
            title="Sites",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.2,
            borderaxespad=0.0
        )
        ax.add_artist(site_legend)

        ax.set_xlabel("Sample datetime")
        ax.set_ylabel("Mole fraction")
        ax.set_title(f"Mole fraction vs Sample datetime\nAnalyte: {analyte}")
        
        if self.main_window.toggle_grid_cb.isChecked():
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show(block=False)
        
    def on_point_pick(self, event):
        artist = event.artist
        if not hasattr(artist, "_meta"):
            return
        if not hasattr(event, "ind") or len(event.ind) == 0:
            return

        mx, my = event.mouseevent.xdata, event.mouseevent.ydata
        if mx is None or my is None:
            return

        # Only for distance calculation
        xdata = mdates.date2num(artist.get_xdata())
        ydata = artist.get_ydata()

        candidates = event.ind
        dists = [(i, (xdata[i] - mx)**2 + (ydata[i] - my)**2) for i in candidates]
        nearest_idx = min(dists, key=lambda t: t[1])[0]

        # These values come straight from your DataFrame (no conversion)
        sample_id   = artist._meta.get("sample_id", [None])[nearest_idx]
        pair_id_num = artist._meta.get("pair_id_num", [None])[nearest_idx]
        run_time    = artist._meta.get("run_time", [None])[nearest_idx]
        site        = artist._meta.get("site", [None])[nearest_idx]
        analyte     = artist._meta.get("analyte", "Unknown")
        channel     = artist._meta.get("channel", None)

        # Always show tooltip
        text = (
            f"<b>Site:</b> {site}<br>"
            f"<b>Sample ID:</b> {sample_id}<br>"
            f"<b>Pair ID:</b> {pair_id_num}<br>"
            f"<b>Run time:</b> {run_time}"
        )
        QToolTip.showText(QCursor.pos(), text)

        # Right click adds extra action -- loads the run in main window
        if event.mouseevent.button == 3:  # right click
            self.main_window.current_run_time = str(run_time)
            self.main_window.current_pnum = int(self.analytes.get(analyte))
            self.main_window.current_channel = channel

            # --- Update the analyte selection UI ---
            if hasattr(self.main_window, "radio_group") and self.main_window.radio_group:
                # Handle radio buttons
                for rb in self.main_window.radio_group.buttons():
                    if rb.text() == analyte:
                        rb.setChecked(True)
                        break

            elif hasattr(self.main_window, "analyte_combo"):
                # Handle combo box
                idx = self.main_window.analyte_combo.findText(analyte, Qt.MatchExactly)
                if idx >= 0:
                    self.main_window.analyte_combo.setCurrentIndex(idx)

            # --- Update date range UI ---
            # Ensure run_time is a datetime (not string)
            if not isinstance(run_time, pd.Timestamp):
                run_time = pd.to_datetime(run_time)

            end_year = run_time.year
            end_month = run_time.month

            # One month prior
            start_dt = (run_time - pd.DateOffset(months=1))
            start_year = start_dt.year
            start_month = start_dt.month

            # Update combo boxes
            self.main_window.end_year_cb.setCurrentText(str(end_year))
            self.main_window.end_month_cb.setCurrentIndex(end_month - 1)
            self.main_window.start_year_cb.setCurrentText(str(start_year))
            self.main_window.start_month_cb.setCurrentIndex(start_month - 1)
            
            # --- Set run type to "Flasks" ---
            self.main_window.runTypeCombo.blockSignals(True)
            self.main_window.runTypeCombo.setCurrentText("Flasks")
            self.main_window.runTypeCombo.blockSignals(False)
    
            # --- Continue with loading the run ---
            self.main_window.set_runlist(initial_date=run_time)
            self.main_window.on_plot_type_changed(self.main_window.current_plot_type)
            self.main_window.current_run_time = str(run_time) 
            # no need for the apply button highlight
            self.main_window.apply_date_btn.setStyleSheet("")
