from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip
)
from PyQt5.QtGui import QCursor
from PyQt5.QtCore import Qt

from matplotlib.widgets import Button
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
        
        # remember dataset visibility across refreshes
        self._dataset_visibility = {"All samples": True, "Flask mean": False, "Pair mean": False}
        self._fig = None
        self._ax = None
        self._dataset_legend = None
        self._site_legend = None
        self._reload_button = None        
        
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
            cb.setChecked(site in initial_sites)   # only check if in subset
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
        self.plot_button.clicked.connect(self.timeseries_plot)
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
        legend_line = event.artist
        if not isinstance(legend_line, mlines.Line2D):
            return
        if not getattr(legend_line, "_is_dataset_legend", False):
            return

        label = legend_line.get_label()
        if label not in self.dataset_handles or not self.dataset_handles[label]:
            return

        new_visible = not self._dataset_visibility.get(label, True)
        self._dataset_visibility[label] = new_visible
        for h in self.dataset_handles[label]:
            h.set_visible(new_visible)

        legend_line.set_alpha(1.0 if new_visible else 0.2)
        event.canvas.draw_idle()

    def load_flask_data(self):
        start = self.start_year.value()
        end = self.end_year.value()
        analyte = self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        self.set_current_analyte(analyte)
        channel = self.current_channel
        sites = [cb.text() for cb in self.site_checks if cb.isChecked()]

        if not sites or pnum is None:
            return pd.DataFrame()

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
            
        return df

    def _draw_dataset_artists(self, ax, datasets, analyte):
        """(Re)create dataset artists only; legends/axes stay intact."""
        # remove old dataset artists
        for label, handles in getattr(self, "dataset_handles", {}).items():
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        self.dataset_handles = {}

        sites = self.get_active_sites()
        site_colors = build_site_colors(self.sites_by_lat)
        styles = {
            "All samples": {"marker": "o", "shade": 1.1, "error": False, "size": 3, "alpha": 0.4},
            "Flask mean": {"marker": "^", "shade": 0.8, "error": True,  "size": 5, "alpha": 0.9},
            "Pair mean":  {"marker": "s", "shade": 0.5, "error": True,  "size": 6, "alpha": 0.9},
        }

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
                        label=label, mfc=color, mec='gray', picker=5
                    )
                    # metadata for tooltip/right-click behavior
                    line._meta = {
                        "run_time": grp.get("run_time", pd.Series([None]*len(grp))).tolist(),
                        "sample_id": grp.get("sample_id", pd.Series([None]*len(grp))).tolist(),
                        "pair_id_num": grp.get("pair_id_num", pd.Series([None]*len(grp))).tolist(),
                        "site": grp.get("site", pd.Series([None]*len(grp))).tolist(),
                        "analyte": analyte,
                        "channel": self.current_channel,
                    }
                    self.dataset_handles.setdefault(label, []).append(line)
                else:
                    container = ax.errorbar(
                        grp["sample_datetime"], grp["mean"], yerr=grp["std"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], capsize=2, alpha=style["alpha"],
                        mfc='none', mec=color, label=label
                    )
                    parts = []
                    for child in container:
                        if isinstance(child, (list, tuple)):
                            parts.extend(child)
                        else:
                            parts.append(child)
                    parts = [p for p in parts if hasattr(p, "set_visible")]
                    self.dataset_handles.setdefault(label, []).extend(parts)

        # honor persisted visibility
        for label, handles in self.dataset_handles.items():
            visible = self._dataset_visibility.get(label, True)
            for h in handles:
                h.set_visible(visible)

        ax.figure.canvas.draw_idle()

    def refresh_artists(self, keep_limits: bool = True):
        if self._ax is None:
            return

        ax = self._ax
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        df = self.query_flask_data(force=True)
        if df.empty:
            print("No data to reload")
            return

        datasets = self.build_datasets(df)
        analyte = self.analyte_combo.currentText()
        self._draw_dataset_artists(ax, datasets, analyte)

        if keep_limits:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        ax.figure.canvas.draw_idle()

    def timeseries_plot(self):
        analyte = self.analyte_combo.currentText()

        df = self.query_flask_data(force=False)
        if df is None or df.empty:
            print("No data to plot")
            return

        datasets = self.build_datasets(df)
        sites = self.get_active_sites()
        site_colors = build_site_colors(self.sites_by_lat)

        fig, ax = plt.subplots(figsize=(12, 6))
        self._fig, self._ax = fig, ax
        self.dataset_handles = {}

        # --- Draw datasets (artists only) ---
        self._draw_dataset_artists(ax, datasets, analyte)

        # Make sure layout is computed
        plt.tight_layout(rect=[0, 0, 0.75, 1])   # keep your right panel
        fig.canvas.draw()

        # Compute a "panel left" just to the right of the axes
        axpos = ax.get_position()  # Bbox in FIGURE coords (x0, y0, w, h)
        panel_left = axpos.x0 + axpos.width + 0.01           # small gap from axes
        panel_top  = axpos.y0 + axpos.height

        # --- Dataset legend (toggleable, one-time wiring) ---
        legend_handles = []
        for label in ["All samples", "Flask mean", "Pair mean"]:
            dummy = mlines.Line2D([], [], color="black",
                                  marker={"All samples": "o", "Flask mean": "^", "Pair mean": "s"}[label],
                                  linestyle="", markersize=6, label=label)
            dummy._is_dataset_legend = True
            if not self._dataset_visibility.get(label, True):
                dummy.set_alpha(0.4)
            legend_handles.append(dummy)

        dataset_legend = ax.legend(
            handles=legend_handles,
            title="Datasets",
            bbox_to_anchor=(panel_left, axpos.y0),
            bbox_transform=fig.transFigure,
            loc="lower left",
            borderaxespad=0.0
        )
        ax.add_artist(dataset_legend)
        self._dataset_legend = dataset_legend

        for legline in dataset_legend.legendHandles:
            legline.set_picker(5)
            legline._is_dataset_legend = True

        fig.canvas.mpl_connect("pick_event", self.on_legend_pick)
        fig.canvas.mpl_connect("pick_event", self.on_point_pick)

        # --- Site legend (place in FIGURE coords, aligned to right of axes) ---
        sites = self.get_active_sites()
        site_colors = build_site_colors(self.sites_by_lat)
        site_handles = {s: ax.plot([], [], color=site_colors[s], marker="o", linestyle="", label=s)[0]
                        for s in sites}

        # 3) Create the Sites legend ANCHORED IN FIGURE COORDS (not ax coords)
        site_legend = ax.legend(
            handles=site_handles.values(),
            title="Sites",
            bbox_to_anchor=(panel_left, panel_top),    # figure coords
            bbox_transform=fig.transFigure,            # <— key change
            loc="upper left",
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.2,
            borderaxespad=0.0
        )
        ax.add_artist(site_legend)
        self._site_legend = site_legend

        # 4) Now place the Reload button directly UNDER that legend (also fig coords)
        fig.canvas.draw()  # ensure legend size is finalized
        renderer = fig.canvas.get_renderer()
        leg_bbox_fig = site_legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())

        pad_h = 0.01
        btn_h = 0.035
        btn_w = leg_bbox_fig.width * 0.90     # 90% of legend width (nice look)
        btn_x = leg_bbox_fig.x0 + (leg_bbox_fig.width - btn_w) / 2.0
        btn_y = max(0.02, leg_bbox_fig.y0 - pad_h - btn_h)

        if not hasattr(self, "_reload_button_ax") or self._reload_button_ax.figure != fig:
            self._reload_button_ax = fig.add_axes([btn_x, btn_y, btn_w, btn_h])
            from matplotlib.widgets import Button
            self._reload_button = Button(self._reload_button_ax, "Reload")
            self._reload_button.on_clicked(lambda evt: self.refresh_artists(keep_limits=True))
        else:
            # if figure already has the button, just reposition it
            self._reload_button_ax.set_position([btn_x, btn_y, btn_w, btn_h])
            
        # --- Axes labels, grid, layout ---
        ax.set_xlabel("Sample datetime")
        ax.set_ylabel("Mole fraction")
        ax.set_title(f"Mole fraction vs Sample datetime\nAnalyte: {analyte}")

        if self.main_window.toggle_grid_cb.isChecked():
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show(block=False)
        
        self._resize_cid = fig.canvas.mpl_connect("resize_event", self._reposition_reload_under_legend)

    def _reposition_reload_under_legend(self, event=None):
        if not getattr(self, "_site_legend", None) or not getattr(self, "_reload_button_ax", None):
            return
        fig = self._site_legend.axes.figure
        renderer = fig.canvas.get_renderer()
        leg_bbox_fig = self._site_legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        pad_h = 0.01
        btn_h = 0.035
        btn_w = leg_bbox_fig.width * 0.90
        btn_x = leg_bbox_fig.x0 + (leg_bbox_fig.width - btn_w) / 2.0
        btn_y = max(0.02, leg_bbox_fig.y0 - pad_h - btn_h)
        self._reload_button_ax.set_position([btn_x, btn_y, btn_w, btn_h])
        fig.canvas.draw_idle()

    # ---------- Data plumbing ----------
    def get_active_sites(self):
        return [cb.text() for cb in self.site_checks if cb.isChecked()]

    def query_flask_data(self, force: bool = False) -> pd.DataFrame:
        start = self.start_year.value()
        end   = self.end_year.value()
        analyte = self.analyte_combo.currentText()
        pnum    = self.analytes.get(analyte)
        self.set_current_analyte(analyte)
        channel = self.current_channel

        if pnum is None:
            return pd.DataFrame()

        ch_str = '' if channel is None else f'AND channel = "{channel}"'
        query_params = (start, end, analyte)

        if force or query_params != self._last_query_params:
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
                self._cached_df = pd.DataFrame()
            else:
                df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])
                self._cached_df = df
            self._last_query_params = query_params

        return self._cached_df.copy() if self._cached_df is not None else pd.DataFrame()

    def build_datasets(self, df: pd.DataFrame) -> dict:
        datasets = {}
        if self.hide_flagged.isChecked():
            datasets["All samples"] = df[df["data_flag"] == "..."].copy()
        else:
            datasets["All samples"] = df.copy()

        clean = df[df["data_flag"] == "..."]
        fm = (clean.groupby(["site", "sample_id", "sample_datetime"])
                    .agg({"mole_fraction": ["mean", "std"]}).reset_index())
        fm.columns = ["site", "sample_id", "sample_datetime", "mean", "std"]
        pm = (clean.groupby(["site", "pair_id_num"])
                    .agg({"sample_datetime": "first", "mole_fraction": ["mean", "std"]})
                    .reset_index())
        pm.columns = ["site", "pair_id_num", "sample_datetime", "mean", "std"]

        datasets["Flask mean"] = fm
        datasets["Pair mean"]  = pm
        return datasets
                
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
