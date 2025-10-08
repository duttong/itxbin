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


class TimeseriesFigure:
    """Manages a single interactive matplotlib figure for timeseries data."""

    def __init__(self, parent_widget, df, analyte):
        self.parent = parent_widget
        self.df = df
        self.analyte = analyte
        self.channel = parent_widget.current_channel

        # Per-figure state (fully independent)
        self._fig, self._ax = plt.subplots(figsize=(12, 6))
        self.dataset_handles = {}
        self.dataset_visibility = {"All samples": True, "Flask mean": False, "Pair mean": False}

        self._build_plot()
        self._fig.canvas.mpl_connect("close_event", self._on_close)

    # ────────────────────────────────────────────────────────────
    def _on_close(self, evt):
        # drop strong ref from parent so it can GC cleanly
        try:
            self.parent.open_figures.remove(self)
        except ValueError:
            pass

    # ────────────────────────────────────────────────────────────
    def _build_plot(self):
        datasets = self.parent.build_datasets(self.df)
        sites_by_lat = self.parent.sites_by_lat

        # Draw the datasets (reuses parent logic)
        self.dataset_handles = self.parent._draw_dataset_artists(self._ax, datasets, self.analyte)

        # --- Apply initial visibility defaults ---
        for label, visible in self.dataset_visibility.items():
            if label in self.dataset_handles:
                for h in self.dataset_handles[label]:
                    h.set_visible(visible)

        # Layout adjustments
        #plt.tight_layout(rect=[0, 0, 0.85, 1])
        self._fig.tight_layout(rect=[0, 0, 0.85, 1])
        self._fig.subplots_adjust(top=0.90, bottom=0.12, left=0.05, right=0.85)
        self._fig.canvas.draw()

        # Positioning helpers
        axpos = self._ax.get_position()
        panel_left = axpos.x0 + axpos.width + 0.01
        panel_top = axpos.y0 + axpos.height

        # ─── Dataset Legend ───
        legend_handles = []
        for label in ["All samples", "Flask mean", "Pair mean"]:
            dummy = mlines.Line2D([], [], color="black",
                                  marker={"All samples": "o", "Flask mean": "^", "Pair mean": "s"}[label],
                                  linestyle="", markersize=6, label=label)
            dummy._is_dataset_legend = True
            if not self.dataset_visibility.get(label, True):
                dummy.set_alpha(0.4)
            legend_handles.append(dummy)

        dataset_legend = self._ax.legend(
            handles=legend_handles,
            title="Datasets",
            bbox_to_anchor=(panel_left, axpos.y0),
            bbox_transform=self._fig.transFigure,
            loc="lower left",
            borderaxespad=0.0
        )
        self._ax.add_artist(dataset_legend)
        self._dataset_legend = dataset_legend

        for legline in dataset_legend.legendHandles:
            legline.set_picker(5)
            legline._is_dataset_legend = True

        # ─── Site Legend ───
        site_colors = build_site_colors(sites_by_lat)
        sites = self.parent.get_active_sites()
        site_handles = {s: self._ax.plot([], [], color=site_colors[s], marker="o", linestyle="", label=s)[0]
                        for s in sites}
        site_legend = self._ax.legend(
            handles=site_handles.values(),
            title="Sites",
            bbox_to_anchor=(panel_left, panel_top),
            bbox_transform=self._fig.transFigure,
            loc="upper left",
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.2,
            borderaxespad=0.0
        )
        self._ax.add_artist(site_legend)
        self._site_legend = site_legend

        # ─── Reload Button ───
        self._add_reload_button_below(site_legend)

        # Axes labels
        self._ax.set_xlabel("Sample datetime")
        self._ax.set_ylabel("Mole fraction")
        self._ax.set_title(f"Mole fraction vs Sample datetime\nAnalyte: {self.analyte}")

        if self.parent.main_window.toggle_grid_cb.isChecked():
            self._ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.show(block=False)

        # Event connections
        self._fig.canvas.mpl_connect("pick_event", self._on_pick_event)
        self._fig.canvas.mpl_connect("resize_event", self._reposition_reload_under_legend)

    # ────────────────────────────────────────────────────────────
    def _add_reload_button_below(self, legend):
        self._fig.canvas.draw()
        renderer = self._fig.canvas.get_renderer()
        leg_bbox = legend.get_window_extent(renderer=renderer).transformed(self._fig.transFigure.inverted())

        pad_h = 0.01
        btn_h = 0.035
        btn_w = leg_bbox.width * 0.9
        btn_x = leg_bbox.x0 + (leg_bbox.width - btn_w) / 2
        btn_y = max(0.02, leg_bbox.y0 - pad_h - btn_h)

        self._reload_ax = self._fig.add_axes([btn_x, btn_y, btn_w, btn_h])
        self._reload_btn = Button(self._reload_ax, "Reload")
        self._reload_btn.on_clicked(lambda evt: self._on_reload_clicked())

    # ────────────────────────────────────────────────────────────
    def _on_reload_clicked(self):
        """Reload data from parent instrument and preserve per-dataset visibility."""
        # Flash the button yellow while reloading
        self._reload_btn.label.set_text("Reloading...")
        self._reload_ax.set_facecolor("#fff8cc")  # soft yellow
        self._fig.canvas.draw_idle()

        plt.pause(0.05)  # brief visual feedback (keeps GUI responsive)

        # Query using the analyte/channel that this figure was created with
        df = self.parent.query_flask_data(force=True, analyte=self.analyte, channel=self.channel)
        if df.empty:
            print("No data to reload")
            self._reload_btn.label.set_text("Reload")
            self._reload_ax.set_facecolor("lightgray")
            self._fig.canvas.draw_idle()
            return

        # Remove old dataset artists before redrawing
        for handles in self.dataset_handles.values():
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
        self.dataset_handles.clear()

        # Redraw datasets
        datasets = self.parent.build_datasets(df)
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        self.dataset_handles = self.parent._draw_dataset_artists(self._ax, datasets, self.analyte)

        # Re-apply per-dataset visibility preferences
        for label, visible in self.dataset_visibility.items():
            if label in self.dataset_handles:
                for h in self.dataset_handles[label]:
                    h.set_visible(visible)

        # Restore limits
        self._ax.set_xlim(xlim)
        self._ax.set_ylim(ylim)

        # Restore button to normal
        self._reload_btn.label.set_text("Reload")
        self._reload_ax.set_facecolor("lightgray")
        self._fig.canvas.draw_idle()

    def _reposition_reload_under_legend(self, event=None):
        if not getattr(self, "_site_legend", None) or not getattr(self, "_reload_ax", None):
            return
        fig = self._site_legend.axes.figure
        renderer = fig.canvas.get_renderer()
        leg_bbox = self._site_legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
        pad_h = 0.01
        btn_h = 0.035
        btn_w = leg_bbox.width * 0.9
        btn_x = leg_bbox.x0 + (leg_bbox.width - btn_w) / 2
        btn_y = max(0.02, leg_bbox.y0 - pad_h - btn_h)
        self._reload_ax.set_position([btn_x, btn_y, btn_w, btn_h])
        fig.canvas.draw_idle()

    # ────────────────────────────────────────────────────────────
    def _on_pick_event(self, event):
        """Handle both dataset legend toggles and point tooltips."""
        artist = event.artist

        # Dataset legend click
        if isinstance(artist, mlines.Line2D) and getattr(artist, "_is_dataset_legend", False):
            label = artist.get_label()
            if label not in self.dataset_visibility:
                return

            # Toggle visibility state
            new_visible = not self.dataset_visibility[label]
            self.dataset_visibility[label] = new_visible

            # Update corresponding dataset artists
            if label in self.dataset_handles:
                for h in self.dataset_handles[label]:
                    h.set_visible(new_visible)

            # Update legend alpha feedback
            artist.set_alpha(1.0 if new_visible else 0.3)
            event.canvas.draw_idle()
            return

        # Otherwise, delegate to parent for tooltip/right-click actions
        self.parent.on_point_pick(event)

class TimeseriesWidget(QWidget):
    def __init__(self, instrument=None, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.main_window = self.parent()
        self.open_figures = []

        self.analytes = self.instrument.analytes or {}
        self.current_analyte = None
        self.current_channel = None

        if self.instrument.inst_num == 193:     # FE3
            self.current_analyte = 'CFC11 (c)'
            self.current_channel = 'c'
        
        self.sites = sorted(LOGOS_sites) if LOGOS_sites else []
        self.sites_df = self.get_site_info() if self.instrument else pd.DataFrame()
        self.sites_by_lat = self.sites_df.sort_values("lat", ascending=False)["code"].tolist()
       
        # cache for last loaded data
        self._cached_df = None
        self._last_query_params = None  # (start, end, analyte)
        
        # remember dataset visibility across refreshes
        self._dataset_visibility = {"All samples": True, "Flask mean": False, "Pair mean": False}   
        
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

    def _draw_dataset_artists(self, ax, datasets, analyte):
        """Draw datasets on a specific Axes, return dataset_handles dict."""
        dataset_handles = {}

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
                    line._meta = {
                        "run_time": grp.get("run_time", pd.Series([None]*len(grp))).tolist(),
                        "sample_id": grp.get("sample_id", pd.Series([None]*len(grp))).tolist(),
                        "pair_id_num": grp.get("pair_id_num", pd.Series([None]*len(grp))).tolist(),
                        "site": grp.get("site", pd.Series([None]*len(grp))).tolist(),
                        "analyte": analyte,
                        "channel": self.current_channel,
                    }
                    dataset_handles.setdefault(label, []).append(line)
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
                    dataset_handles.setdefault(label, []).extend(parts)

        ax.figure.canvas.draw_idle()
        return dataset_handles

    def timeseries_plot(self):
        df = self.query_flask_data(force=False)
        if df.empty:
            print("No data to plot")
            return
        analyte = self.analyte_combo.currentText()
        fig = TimeseriesFigure(self, df, analyte)
        self.open_figures.append(fig)

    # ---------- Data plumbing ----------
    def get_active_sites(self):
        return [cb.text() for cb in self.site_checks if cb.isChecked()]

    def query_flask_data(self, force: bool = False, analyte: str | None = None, channel: str | None = None) -> pd.DataFrame:
        start = self.start_year.value()
        end   = self.end_year.value()

        analyte = analyte or self.analyte_combo.currentText()
        pnum    = self.analytes.get(analyte)
        self.set_current_analyte(analyte)
        channel = channel or self.current_channel

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

            if hasattr(self.main_window, "tabs"):
                self.main_window.tabs.setCurrentIndex(0)  # switch to first tab ("Processing")

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
