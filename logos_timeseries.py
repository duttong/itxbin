from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip, QSizePolicy, QApplication, QShortcut
)
from PyQt5.QtGui import QCursor, QKeySequence
from PyQt5.QtCore import Qt

from matplotlib.widgets import Button, RadioButtons
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import PathCollection
import pandas as pd
import numpy as np
import colorsys
import time


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
    for i, site in enumerate(sites):
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
        self.parent_widget = parent_widget
        self.df = df
        self.analyte = analyte
        self.channel = parent_widget.current_channel

        # Per-figure state (fully independent)
        self._fig, self._ax = plt.subplots(figsize=(12, 6))
        self.dataset_handles = {}
        self.dataset_visibility = {"All samples": True, "Flask mean": False, "Pair mean": False}

        self._setup_toolbar_widgets()
        self._setup_shortcuts()
        self._build_plot()
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        self._fig.canvas.mpl_connect("pick_event", self._on_pick_event)

    # ────────────────────────────────────────────────────────────
    def _on_close(self, evt):
        # drop strong ref from parent so it can GC cleanly
        try:
            self.parent_widget.open_figures.remove(self)
        except ValueError:
            pass

    def _setup_toolbar_widgets(self):
        analyte_names = list((self.parent_widget.instrument.analytes or {}).keys())
        if not analyte_names:
            analyte_names = [self.analyte]
        
        self.analyte_combo = QComboBox()
        self.analyte_combo.addItems(analyte_names)
        idx = self.analyte_combo.findText(self.analyte, Qt.MatchExactly)
        if idx >= 0:
            self.analyte_combo.setCurrentIndex(idx)

        self.reload_btn = QPushButton("Reload")

        self.analyte_combo.currentTextChanged.connect(self._on_analyte_changed)
        self.reload_btn.clicked.connect(self._on_reload_clicked)

        combo_container = QWidget()
        combo_layout = QHBoxLayout()
        combo_layout.setContentsMargins(0, 0, 0, 0)
        combo_layout.setSpacing(4)
        combo_layout.addWidget(self.analyte_combo)
        combo_layout.addWidget(self.reload_btn)
        combo_container.setLayout(combo_layout)

        toolbar = getattr(getattr(self._fig.canvas, "manager", None), "toolbar", None)
        if toolbar is not None and hasattr(toolbar, "addWidget"):
            spacer = QWidget()
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            toolbar.addWidget(spacer)
            toolbar.addWidget(combo_container)
        else:
            combo_container.setParent(self._fig.canvas)
            combo_container.show()

    def _setup_shortcuts(self):
        self.prev_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Up"), self._fig.canvas)
        self.prev_shortcut.activated.connect(self._prev_analyte)
        self.next_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Down"), self._fig.canvas)
        self.next_shortcut.activated.connect(self._next_analyte)

    def _prev_analyte(self):
        idx = self.analyte_combo.currentIndex()
        if idx > 0:
            self.analyte_combo.setCurrentIndex(idx - 1)

    def _next_analyte(self):
        idx = self.analyte_combo.currentIndex()
        if idx < self.analyte_combo.count() - 1:
            self.analyte_combo.setCurrentIndex(idx + 1)

    # ────────────────────────────────────────────────────────────
    def _build_plot(self):
        """Build full figure layout and legends."""
        datasets = self.parent_widget.build_datasets(self.df)
        sites_by_lat = self.parent_widget.sites_by_lat

        # --- Draw datasets ---
        self.dataset_handles = self.parent_widget._draw_dataset_artists(self._ax, datasets, self.analyte)
        
        self._rebuild_data_artists()

        # Tag each artist with its site (safety)
        for label, artists in self.dataset_handles.items():
            for artist in artists:
                artist._site = getattr(artist, "_site", None)

        # Layout adjustments
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
        sites = self.parent_widget.get_active_sites()

        # Create dummy handles for legend
        site_handles = {
            s: self._ax.plot([], [], color=site_colors[s], marker="o", linestyle="", label=s)[0]
            for s in sites
        }

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

        for legline in site_legend.legendHandles:
            legline.set_picker(5)
            legline._site_legend = True

        # ─── Build mapping of site → all artists ───
        self._site_artists = {}
        for artist in self._ax.get_lines() + self._ax.collections:
            site = getattr(artist, "_site", None)
            if site:
                self._site_artists.setdefault(site, []).append(artist)

        # ─── Initialize site visibility (all ON) ───
        self._site_visibility = {s: True for s in self._site_artists.keys()}

        # ─── Apply dataset + site visibility rules ───
        self._apply_visibility()

        # Axes labels and cosmetics
        self._ax.set_xlabel("Sample datetime")
        self._ax.set_ylabel("Mole fraction")
        self._ax.set_title(f"Mole fraction vs Sample datetime\nAnalyte: {self.analyte}")
        self._ax.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.show(block=False)


    def _apply_visibility(self):
        """Enforce visibility = site_visibility AND dataset_visibility for all artists."""
        for site, artists in getattr(self, "_site_artists", {}).items():
            svis = self._site_visibility.get(site, True)
            for a in artists:
                # Try to get dataset label (either via .get_label() or custom tag)
                dlabel = getattr(a, "_dataset_label", None)
                if dlabel is None:
                    dlabel = getattr(a, "get_label", lambda: None)()
                if not dlabel or dlabel not in self.dataset_visibility:
                    continue

                dvis = self.dataset_visibility[dlabel]
                a.set_visible(bool(svis and dvis))

                vis = bool(svis and dvis)
                a.set_visible(vis)
                if hasattr(a, "set_picker") and hasattr(a, "_meta"):
                    a.set_picker(5 if vis else False)   # False disables picking
        self._fig.canvas.draw_idle()

    def _on_analyte_changed(self, text):
        self.analyte = text
        if "(" in text and ")" in text:
            _, channel = text.split("(", 1)
            self.channel = channel.strip(") ")
        else:
            self.channel = None
        self._on_reload_clicked()

    # ────────────────────────────────────────────────────────────
    def _on_reload_clicked(self):
        """Reload data from parent instrument and preserve per-dataset visibility."""
        self.reload_btn.setText("Reloading...")
        self.reload_btn.setEnabled(False)
        QApplication.processEvents()

        # Query using the analyte/channel that this figure was created with
        df = self.parent_widget.query_flask_data(force=True, analyte=self.analyte, channel=self.channel)
        if df.empty:
            print("No data to reload")
            self.reload_btn.setText("Reload")
            self.reload_btn.setEnabled(True)
            return

        self.df = df
        self._ax.clear()
        self._build_plot()

        self.reload_btn.setText("Reload")
        self.reload_btn.setEnabled(True)

    def _rebuild_data_artists(self):
        """Cache only artists that represent data points we can tooltip (have _meta)."""
        self._data_artists = []
        for artists in self.dataset_handles.values():
            for a in artists:
                if hasattr(a, "_meta") and hasattr(a, "contains"):
                    self._data_artists.append(a)

    def _pick_best_visible(self, mouseevent, max_px=15):
        """
        Return (artist, idx) for nearest visible data point under cursor.
        Uses .contains(mouseevent) and resolves among candidates by pixel distance.
        """
        if mouseevent is None or mouseevent.inaxes != self._ax:
            return (None, None)

        mx, my = mouseevent.x, mouseevent.y  # display pixels
        best_artist, best_idx, best_d2 = None, None, np.inf

        for a in getattr(self, "_data_artists", []):
            # only consider visible artists
            if hasattr(a, "get_visible") and not a.get_visible():
                continue

            hit, info = a.contains(mouseevent)
            if not hit:
                continue

            inds = info.get("ind", [])
            if len(inds) == 0:
                continue

            # Get display coords for this artist's points
            if isinstance(a, mlines.Line2D):
                x = a.get_xdata(orig=False)
                y = a.get_ydata(orig=False)
                xy = np.column_stack([mdates.date2num(x), y]) if np.issubdtype(np.array(x).dtype, np.datetime64) else np.column_stack([x, y])
                disp = self._ax.transData.transform(xy)
            elif isinstance(a, PathCollection):
                # scatter-like
                disp = a.get_offsets()
                disp = a.get_transform().transform(disp)
            else:
                # fallback for other artist types
                try:
                    x = a.get_xdata(orig=False)
                    y = a.get_ydata(orig=False)
                    xy = np.column_stack([x, y])
                    disp = self._ax.transData.transform(xy)
                except Exception:
                    continue

            for i in inds:
                dx = disp[i, 0] - mx
                dy = disp[i, 1] - my
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_artist, best_idx, best_d2 = a, int(i), d2

        if best_artist is None:
            return (None, None)

        if best_d2 > (max_px * max_px):
            return (None, None)

        return (best_artist, best_idx)

    # ────────────────────────────────────────────────────────────
    def _on_pick_event(self, event):
        """Handle dataset legend toggles, site legend toggles, and point tooltips."""

        # --- Prevent double-trigger ---
        tnow = time.time()
        if hasattr(self, "_last_pick_time") and (tnow - self._last_pick_time) < 0.15:
            return
        self._last_pick_time = tnow
        
        artist = event.artist

        # ─── Dataset Legend ───
        if isinstance(artist, mlines.Line2D) and getattr(artist, "_is_dataset_legend", False):
            label = artist.get_label()
            if label not in self.dataset_visibility:
                return

            # Flip dataset state
            self.dataset_visibility[label] = not self.dataset_visibility[label]

            # Visual feedback for the legend icon
            artist.set_alpha(1.0 if self.dataset_visibility[label] else 0.3)

            # Enforce across ALL sites via single pass
            self._apply_visibility()
            return

        # ─── Site Legend ───
        if hasattr(self, "_site_legend"):
            leg = self._site_legend
            label = None

            # Identify clicked site legend entry
            for text in leg.get_texts():
                if event.artist == text:
                    label = text.get_text()
                    break
            for handle in leg.legendHandles:
                if event.artist == handle:
                    label = handle.get_label()
                    break

            if label and label in getattr(self, "_site_visibility", {}):
                # Flip site state
                self._site_visibility[label] = not self._site_visibility[label]

                # Dim legend entry when hidden
                alpha = 1.0 if self._site_visibility[label] else 0.2
                for text in leg.get_texts():
                    if text.get_text() == label:
                        text.set_alpha(alpha)
                for handle in leg.legendHandles:
                    if handle.get_label() == label:
                        handle.set_alpha(alpha)

                # Enforce site x dataset visibility
                self._apply_visibility()
                return

        # ─── Data point pick (always choose nearest VISIBLE) ───
        picked_artist, picked_idx = self._pick_best_visible(event.mouseevent)
        if picked_artist is None:
            return

        self.parent_widget.on_point_pick(event, artist=picked_artist, idx=picked_idx)

class RelStdDevFigure:
    """Manages an interactive matplotlib figure for relative standard deviation."""

    def __init__(self, parent_widget, df, analyte):
        self.parent_widget = parent_widget
        self.df = df
        self.analyte = analyte
        self.channel = parent_widget.current_channel

        # Per-figure state
        self._fig, (self._ax_top, self._ax_bottom) = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        self.site_artists = {}
        self.site_visibility = {}
        
        self.x_var = "run_time"
        self._radio_ax = None
        self._radio = None

        self._setup_toolbar_widgets()
        self._setup_shortcuts()
        self._build_plot()
        self._fig.canvas.mpl_connect("close_event", self._on_close)
        self._fig.canvas.mpl_connect("pick_event", self._on_pick_event)

    def _on_close(self, evt):
        try:
            self.parent_widget.open_figures.remove(self)
        except (ValueError, AttributeError):
            pass

    def _setup_toolbar_widgets(self):
        analyte_names = list((self.parent_widget.instrument.analytes or {}).keys())
        if not analyte_names:
            analyte_names = [self.analyte]
        
        self.analyte_combo = QComboBox()
        self.analyte_combo.addItems(analyte_names)
        idx = self.analyte_combo.findText(self.analyte, Qt.MatchExactly)
        if idx >= 0:
            self.analyte_combo.setCurrentIndex(idx)

        self.reload_btn = QPushButton("Reload")

        self.analyte_combo.currentTextChanged.connect(self._on_analyte_changed)
        self.reload_btn.clicked.connect(self._on_reload_clicked)

        combo_container = QWidget()
        combo_layout = QHBoxLayout()
        combo_layout.setContentsMargins(0, 0, 0, 0)
        combo_layout.setSpacing(4)
        combo_layout.addWidget(self.analyte_combo)
        combo_layout.addWidget(self.reload_btn)
        combo_container.setLayout(combo_layout)

        toolbar = getattr(getattr(self._fig.canvas, "manager", None), "toolbar", None)
        if toolbar is not None and hasattr(toolbar, "addWidget"):
            spacer = QWidget()
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            toolbar.addWidget(spacer)
            toolbar.addWidget(combo_container)
        else:
            combo_container.setParent(self._fig.canvas)
            combo_container.show()

    def _setup_shortcuts(self):
        self.prev_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Up"), self._fig.canvas)
        self.prev_shortcut.activated.connect(self._prev_analyte)
        self.next_shortcut = QShortcut(QKeySequence("Ctrl+Shift+Down"), self._fig.canvas)
        self.next_shortcut.activated.connect(self._next_analyte)

    def _prev_analyte(self):
        idx = self.analyte_combo.currentIndex()
        if idx > 0:
            self.analyte_combo.setCurrentIndex(idx - 1)

    def _next_analyte(self):
        idx = self.analyte_combo.currentIndex()
        if idx < self.analyte_combo.count() - 1:
            self.analyte_combo.setCurrentIndex(idx + 1)

    def _on_analyte_changed(self, text):
        self.analyte = text
        self._on_reload_clicked()

    def _on_reload_clicked(self):
        self.reload_btn.setText("Reloading...")
        self.reload_btn.setEnabled(False)
        QApplication.processEvents()

        df = self.parent_widget.query_rel_stddev_data(analyte=self.analyte)
        if not df.empty:
            self.df = df
            self._build_plot()
        
        self.reload_btn.setText("Reload")
        self.reload_btn.setEnabled(True)

    def _build_plot(self):
        if self._radio_ax is not None:
            self._fig.delaxes(self._radio_ax)
            self._radio_ax = None
            self._radio = None

        self._ax_top.clear()
        self._ax_bottom.clear()
        self.site_artists = {}

        sites_by_lat = self.parent_widget.sites_by_lat
        site_colors = build_site_colors(sites_by_lat)

        # --- Draw data ---
        active_sites = self.df['site'].unique()
        for site in active_sites:
            group = self.df[self.df['site'] == site]
            color = site_colors.get(site, "gray")

            # Top plot: mixratio vs run_time
            top_artist_container = self._ax_top.errorbar(
                group[self.x_var],
                group["mixratio"],
                yerr=group["stddev"],
                fmt="o",
                linestyle="none",
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2,
                markersize=4,
                alpha=0.9,
                label=site
            )
            top_artists = [top_artist_container[0]] + list(top_artist_container[1]) + list(top_artist_container[2])

            # Bottom plot: relstd vs run_time
            bottom_artist_container = self._ax_bottom.vlines(
                group[self.x_var],
                ymin=0,
                ymax=group["relstd"],
                colors=color,
                linewidth=2.2,
                alpha=0.8,
                label=site
            )
            
            visible = self.site_visibility.get(site, True)
            for artist in top_artists:
                artist.set_visible(visible)
            bottom_artist_container.set_visible(visible)

            # Store artists for toggling
            self.site_artists.setdefault(site, []).extend(top_artists)
            self.site_artists[site].append(bottom_artist_container)
            self.site_visibility.setdefault(site, True)

        # --- Figure cosmetics ---
        inst_num = self.parent_widget.instrument.inst_num
        pnum = self.parent_widget.analytes.get(self.analyte)
        self._ax_top.set_title(f"Flask mean mixing ratio and relative stddev vs run time\nAnalyte {self.analyte} (Param {pnum})")
        self._ax_top.set_ylabel("mixratio")
        self._ax_top.grid(alpha=0.3)

        self._ax_bottom.set_ylabel("relstd (%)")
        self._ax_bottom.set_xlabel("Run Time (UTC)" if self.x_var == "run_time" else "Sample datetime (UTC)")
        self._ax_bottom.grid(alpha=0.3)

        self._fig.tight_layout(rect=[0, 0, 0.85, 1])

        # Positioning helpers
        axpos = self._ax_top.get_position()
        panel_left = axpos.x0 + axpos.width + 0.01
        panel_top = axpos.y0 + axpos.height

        # --- Site Legend ---
        # Sort sites by latitude order
        present_sites = set(self.site_artists.keys())
        sorted_sites = [s for s in sites_by_lat if s in present_sites]
        for s in sorted(present_sites):
            if s not in sorted_sites:
                sorted_sites.append(s)

        site_handles = [
            mlines.Line2D([], [], color=site_colors.get(s, "gray"), marker="o", linestyle="", label=s)
            for s in sorted_sites
        ]
        self.site_legend = self._ax_top.legend(
            handles=site_handles,
            title="Sites",
            bbox_to_anchor=(panel_left, panel_top),
            bbox_transform=self._fig.transFigure,
            loc="upper left",
            ncol=2,
            columnspacing=1.0,
            handletextpad=0.2,
            borderaxespad=0.0
        )

        for legline in self.site_legend.legendHandles:
            legline.set_picker(5)
        for text in self.site_legend.get_texts():
            text.set_picker(5)

        self._add_x_axis_radio_buttons(self.site_legend)

        self._fig.canvas.draw_idle()
        plt.show(block=False)

    def _add_x_axis_radio_buttons(self, legend):
        self._fig.canvas.draw()
        renderer = self._fig.canvas.get_renderer()
        leg_bbox = legend.get_window_extent(renderer=renderer).transformed(self._fig.transFigure.inverted())

        pad_h = 0.01
        radio_h = 0.1
        radio_w = leg_bbox.width
        radio_x = leg_bbox.x0
        radio_y = max(0.02, leg_bbox.y0 - pad_h - radio_h)

        self._radio_ax = self._fig.add_axes([radio_x, radio_y, radio_w, radio_h])
        self._radio_ax.set_frame_on(False)
        active_idx = 0 if self.x_var == 'run_time' else 1
        self._radio = RadioButtons(self._radio_ax, ('Run Time', 'Sample Time'), active=active_idx)
        self._radio.on_clicked(self._on_xaxis_change)
        
        for label in self._radio.labels:
            label.set_fontsize(9)

    def _on_xaxis_change(self, label):
        new_var = 'run_time' if label == 'Run Time' else 'sample_datetime'
        if new_var != self.x_var:
            self.x_var = new_var
            self._build_plot()

    def _on_pick_event(self, event):
        leg = self.site_legend
        label_to_handle = {h.get_label(): h for h in leg.legendHandles}
        clicked_label = getattr(event.artist, 'get_label', lambda: None)() or getattr(event.artist, 'get_text', lambda: None)()

        if clicked_label and clicked_label in self.site_visibility:
            site = clicked_label
            self.site_visibility[site] = not self.site_visibility[site]
            visible = self.site_visibility[site]
            for artist in self.site_artists.get(site, []):
                artist.set_visible(visible)
            handle = label_to_handle.get(site)
            if handle:
                handle.set_alpha(1.0 if visible else 0.2)
            for text in leg.get_texts():
                if text.get_text() == site:
                    text.set_alpha(1.0 if visible else 0.2)
            self._fig.canvas.draw_idle()

class TimeseriesWidget(QWidget):
    def __init__(self, instrument=None, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.main_window = parent
        self.open_figures = []
        self._cached_df = None
        self._last_query_params = None
        self._dataset_visibility = {}

        self.analytes = self.instrument.analytes if self.instrument else {}
        self.current_analyte = list(self.analytes.keys())[0] if self.analytes else None
        self.current_channel = None

        controls = QVBoxLayout()

        # Analyte selection
        analyte_group = QGroupBox("Analyte")
        analyte_layout = QHBoxLayout()
        self.analyte_combo = QComboBox()
        self.analyte_combo.addItems(list(self.analytes.keys()))
        self.analyte_combo.currentTextChanged.connect(self.set_current_analyte)
        analyte_layout.addWidget(self.analyte_combo)
        analyte_group.setLayout(analyte_layout)
        controls.addWidget(analyte_group)

        # Date range
        date_group = QGroupBox("Year Range")
        date_layout = QHBoxLayout()
        self.start_year = QSpinBox()
        self.start_year.setRange(1990, 2030)
        self.start_year.setValue(2020)
        self.end_year = QSpinBox()
        self.end_year.setRange(1990, 2030)
        self.end_year.setValue(pd.Timestamp.now().year)
        date_layout.addWidget(QLabel("Start:"))
        date_layout.addWidget(self.start_year)
        date_layout.addWidget(QLabel("End:"))
        date_layout.addWidget(self.end_year)
        date_group.setLayout(date_layout)
        controls.addWidget(date_group)

        # Site selection
        site_group = QGroupBox("Sites")
        site_layout = QGridLayout()
        self.site_checks = []
        
        if self.instrument:
            try:
                site_info = self.get_site_info()
                site_info = site_info.sort_values("lat", ascending=False)
                self.sites_by_lat = site_info["code"].tolist()
            except Exception:
                self.sites_by_lat = sorted(LOGOS_sites)
        else:
            self.sites_by_lat = sorted(LOGOS_sites)

        cols = 4
        row = 0
        for i, site in enumerate(self.sites_by_lat):
            cb = QCheckBox(site)
            if site in ['BRW', 'MLO', 'SMO', 'SPO']:
                cb.setChecked(True)
            else:
                cb.setChecked(False)
            self.site_checks.append(cb)
            row, col = divmod(i, cols)
            site_layout.addWidget(cb, row, col)
        
        btn_layout = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(self.select_all_sites)
        none_btn = QPushButton("None")
        none_btn.clicked.connect(self.select_none_sites)
        btn_layout.addWidget(all_btn)
        btn_layout.addWidget(none_btn)
        site_layout.addLayout(btn_layout, row + 1, 0, 1, cols)

        site_group.setLayout(site_layout)
        controls.addWidget(site_group)
                
        # ----- Set default selection -----
        self.set_current_analyte(self.current_analyte)
        
        # ------ Flagging options ------
        self.hide_flagged = QCheckBox("Hide flagged data")
        self.hide_flagged.setChecked(True)
        controls.addWidget(self.hide_flagged)

        # Plot button
        self.plot_button = QPushButton("Mole Fractions Figure")
        self.plot_button.clicked.connect(self.timeseries_plot)
        controls.addWidget(self.plot_button)
        self.rel_plot_button = QPushButton("Relative Stddev Figure")
        self.rel_plot_button.clicked.connect(self.rel_stddev_plot)
        controls.addWidget(self.rel_plot_button)

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

                # determine visibility for this dataset
                visible = self._dataset_visibility.get(label, True)

                if label == "All samples":
                    line, = ax.plot(
                        grp["sample_datetime"], grp["mole_fraction"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], alpha=style["alpha"],
                        label=label, mfc=color, mec='gray', picker=5
                    )
                    line._site = site
                    line._dataset_label = label
                    line._meta = {
                        "run_time": grp.get("run_time", pd.Series([None]*len(grp))).tolist(),
                        "sample_datetime": grp.get("sample_datetime", pd.Series([None]*len(grp))).tolist(),
                        "sample_id": grp.get("sample_id", pd.Series([None]*len(grp))).tolist(),
                        "pair_id_num": grp.get("pair_id_num", pd.Series([None]*len(grp))).tolist(),
                        "site": grp.get("site", pd.Series([None]*len(grp))).tolist(),
                        "analyte": analyte,
                        "channel": self.current_channel,
                    }
                    line.set_visible(visible)
                    dataset_handles.setdefault(label, []).append(line)

                else:
                    # draw errorbar and tag all its parts
                    container = ax.errorbar(
                        grp["sample_datetime"], grp["mean"], yerr=grp["std"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], capsize=2, alpha=style["alpha"],
                        mfc='none', mec=color, label=label
                    )

                    # Flatten all visible artist components
                    parts = []
                    for child in container:
                        if isinstance(child, (list, tuple)):
                            parts.extend(child)
                        else:
                            parts.append(child)

                    # Apply site label + visibility to every piece
                    for p in parts:
                        if hasattr(p, "set_visible"):
                            p._site = site
                            p._dataset_label = label
                            p.set_visible(visible)

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

    def rel_stddev_plot(self):
        """Handle the 'Relative Stddev Plot' button click."""
        df = self.query_rel_stddev_data()
        if df.empty:
            print("No data available for the relative standard deviation plot.")
            return
        analyte = self.analyte_combo.currentText()
        fig = RelStdDevFigure(self, df, analyte)
        self.open_figures.append(fig)

    def query_rel_stddev_data(self, analyte=None):
        """Query data for the relative standard deviation plot."""
        start_year = self.start_year.value()
        end_year = self.end_year.value()
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()

        inst_num = self.instrument.inst_num
        sites = self.get_active_sites()
        if not sites:
            return pd.DataFrame()

        query = f"""
        SELECT
            run_time, sample_id, pair_id_num, site,
            MIN(sample_datetime) AS sample_datetime,
            AVG(mole_fraction) AS mixratio,
            STDDEV_SAMP(mole_fraction) AS stddev,
            100 * STDDEV_SAMP(mole_fraction) / NULLIF(AVG(mole_fraction), 0) AS relstd,
            COUNT(*) AS n
        FROM hats.ng_data_processing_view
        WHERE inst_num = %s
          AND parameter_num = %s
          AND run_type_num = 1 AND data_flag = '...'
          AND sample_datetime BETWEEN %s AND %s
          AND site IN ({",".join(["%s"]*len(sites))})
        GROUP BY run_time, sample_id, pair_id_num, site
        ORDER BY run_time;
        """
        params = [inst_num, pnum, start_date, end_date] + sites
        df = pd.DataFrame(self.instrument.doquery(query, params))
        if not df.empty:
            df["run_time"] = pd.to_datetime(df["run_time"])
            df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])
            df = df.sort_values(["site", "run_time"])
        return df

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
                   data_flag, site, sample_id, pair_id_num, run_type_num
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

        clean = df[df["data_flag"] == "..."].copy()
        flask_runs = clean[(clean["run_type_num"] == 1) & (clean["pair_id_num"] > 0)].copy()
        pfp_runs = clean[clean["run_type_num"] == 5].copy()

        fm_cols = ["site", "sample_id", "sample_datetime", "mean", "std"]
        if flask_runs.empty:
            fm = pd.DataFrame(columns=fm_cols)
        else:
            fm = (flask_runs.groupby(["site", "sample_id", "sample_datetime"])
                            .agg({"mole_fraction": ["mean", "std"]})
                            .reset_index())
            fm.columns = fm_cols

        pm_frames = []
        if not flask_runs.empty:
            pm_flask = (flask_runs.groupby(["site", "pair_id_num"])
                                 .agg({"sample_datetime": "first", "mole_fraction": ["mean", "std"]})
                                 .reset_index())
            pm_flask.columns = ["site", "pair_id_num", "sample_datetime", "mean", "std"]
            pm_frames.append(pm_flask)

        if not pfp_runs.empty:
            pm_pfp = (pfp_runs.groupby(["site", "sample_datetime"])
                               .agg({"sample_id": "first", "mole_fraction": ["mean", "std"]})
                               .reset_index())
            pm_pfp.columns = ["site", "sample_datetime", "pair_id_num", "mean", "std"]
            pm_pfp = pm_pfp[["site", "pair_id_num", "sample_datetime", "mean", "std"]]
            pm_frames.append(pm_pfp)

        if pm_frames:
            pm = pd.concat(pm_frames, ignore_index=True)
        else:
            pm = pd.DataFrame(columns=["site", "pair_id_num", "sample_datetime", "mean", "std"])

        datasets["Flask mean"] = fm
        datasets["Pair mean"]  = pm
        return datasets
                
    def on_point_pick(self, event, artist=None, idx=None):
        artist = artist or event.artist

        # ignore hidden artists
        if hasattr(artist, "get_visible") and not artist.get_visible():
            return

        if not hasattr(artist, "_meta"):
            return

        # idx is now the authoritative point index
        if idx is None:
            if not hasattr(event, "ind") or len(event.ind) == 0:
                return
            # fallback: keep your old behavior
            idx = int(event.ind[0])

        nearest_idx = idx

        # These values come straight from your DataFrame (no conversion)
        sample_id   = artist._meta.get("sample_id", [None])[nearest_idx]
        pair_id_num = artist._meta.get("pair_id_num", [None])[nearest_idx]
        run_time    = artist._meta.get("run_time", [None])[nearest_idx]
        sample_time = artist._meta.get("sample_datetime", [None])[nearest_idx]
        site        = artist._meta.get("site", [None])[nearest_idx]
        analyte     = artist._meta.get("analyte", "Unknown")
        channel     = artist._meta.get("channel", None)

        # Always show tooltip
        text = (
            f"<b>Site:</b> {site}<br>"
            f"<b>Sample ID:</b> {sample_id}<br>"
            f"<b>Pair ID:</b> {pair_id_num}<br>"
            f"<b>Sample time:</b> {sample_time}<br>"
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
            
            # --- Ensure run type is "All" so PFPs show up too ---
            self.main_window.runTypeCombo.blockSignals(True)
            self.main_window.runTypeCombo.setCurrentText("All")
            self.main_window.runTypeCombo.blockSignals(False)
    
            # --- Continue with loading the run ---
            self.main_window.set_runlist(initial_date=run_time)
            self.main_window.on_plot_type_changed(self.main_window.current_plot_type)
            self.main_window.current_run_time = str(run_time)
            # no need for the apply button highlight
            self.main_window.apply_date_btn.setStyleSheet("")
        # ----- Set default selection -----
        self.set_current_analyte(self.current_analyte)
        
        # ------ Flagging options ------
        self.hide_flagged = QCheckBox("Hide flagged data")
        self.hide_flagged.setChecked(True)
        controls.addWidget(self.hide_flagged)

        # Plot button
        self.plot_button = QPushButton("Mole Fractions Plot")
        self.plot_button.clicked.connect(self.timeseries_plot)
        controls.addWidget(self.plot_button)

        self.rel_plot_button = QPushButton("Relative Std Plot")
        self.rel_plot_button.clicked.connect(self.rel_stddev_plot)
        controls.addWidget(self.rel_plot_button)

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

                # determine visibility for this dataset
                visible = self._dataset_visibility.get(label, True)

                if label == "All samples":
                    line, = ax.plot(
                        grp["sample_datetime"], grp["mole_fraction"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], alpha=style["alpha"],
                        label=label, mfc=color, mec='gray', picker=5
                    )
                    line._site = site
                    line._dataset_label = label
                    line._meta = {
                        "run_time": grp.get("run_time", pd.Series([None]*len(grp))).tolist(),
                        "sample_datetime": grp.get("sample_datetime", pd.Series([None]*len(grp))).tolist(),
                        "sample_id": grp.get("sample_id", pd.Series([None]*len(grp))).tolist(),
                        "pair_id_num": grp.get("pair_id_num", pd.Series([None]*len(grp))).tolist(),
                        "site": grp.get("site", pd.Series([None]*len(grp))).tolist(),
                        "analyte": analyte,
                        "channel": self.current_channel,
                    }
                    line.set_visible(visible)
                    dataset_handles.setdefault(label, []).append(line)

                else:
                    # draw errorbar and tag all its parts
                    container = ax.errorbar(
                        grp["sample_datetime"], grp["mean"], yerr=grp["std"],
                        marker=style["marker"], linestyle="",
                        color=color, markersize=style["size"], capsize=2, alpha=style["alpha"],
                        mfc='none', mec=color, label=label
                    )

                    # Flatten all visible artist components
                    parts = []
                    for child in container:
                        if isinstance(child, (list, tuple)):
                            parts.extend(child)
                        else:
                            parts.append(child)

                    # Apply site label + visibility to every piece
                    for p in parts:
                        if hasattr(p, "set_visible"):
                            p._site = site
                            p._dataset_label = label
                            p.set_visible(visible)

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

    def rel_stddev_plot(self):
        """Handle the 'Relative Stddev Plot' button click."""
        df = self.query_rel_stddev_data()
        if df.empty:
            print("No data available for the relative standard deviation plot.")
            return
        analyte = self.analyte_combo.currentText()
        fig = RelStdDevFigure(self, df, analyte)
        self.open_figures.append(fig)

    def query_rel_stddev_data(self, analyte=None):
        """Query data for the relative standard deviation plot."""
        start_year = self.start_year.value()
        end_year = self.end_year.value()
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()

        inst_num = self.instrument.inst_num
        sites = self.get_active_sites()
        if not sites:
            return pd.DataFrame()

        query = f"""
        SELECT
            run_time, sample_id, pair_id_num, site,
            MIN(sample_datetime) AS sample_datetime,
            AVG(mole_fraction) AS mixratio,
            STDDEV_SAMP(mole_fraction) AS stddev,
            100 * STDDEV_SAMP(mole_fraction) / NULLIF(AVG(mole_fraction), 0) AS relstd,
            COUNT(*) AS n
        FROM hats.ng_data_processing_view
        WHERE inst_num = %s
          AND parameter_num = %s
          AND run_type_num = 1  # only flask runs for the stddev plot
          AND data_flag = '...'
          AND sample_datetime BETWEEN %s AND %s
          AND site IN ({",".join(["%s"]*len(sites))})
        GROUP BY run_time, sample_id, pair_id_num, site
        ORDER BY run_time;
        """
        params = [inst_num, pnum, start_date, end_date] + sites
        df = pd.DataFrame(self.instrument.doquery(query, params))
        if not df.empty:
            df["run_time"] = pd.to_datetime(df["run_time"])
            df = df.sort_values(["site", "run_time"])
        return df

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
                   data_flag, site, sample_id, pair_id_num, run_type_num
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

        clean = df[df["data_flag"] == "..."].copy()
        flask_runs = clean[(clean["run_type_num"] == 1) & (clean["pair_id_num"] > 0)].copy()
        pfp_runs = clean[clean["run_type_num"] == 5].copy()

        fm_cols = ["site", "sample_id", "sample_datetime", "mean", "std"]
        if flask_runs.empty:
            fm = pd.DataFrame(columns=fm_cols)
        else:
            fm = (flask_runs.groupby(["site", "sample_id", "sample_datetime"])
                            .agg({"mole_fraction": ["mean", "std"]})
                            .reset_index())
            fm.columns = fm_cols

        pm_frames = []
        if not flask_runs.empty:
            pm_flask = (flask_runs.groupby(["site", "pair_id_num"])
                                 .agg({"sample_datetime": "first", "mole_fraction": ["mean", "std"]})
                                 .reset_index())
            pm_flask.columns = ["site", "pair_id_num", "sample_datetime", "mean", "std"]
            pm_frames.append(pm_flask)

        if not pfp_runs.empty:
            pm_pfp = (pfp_runs.groupby(["site", "sample_datetime"])
                               .agg({"sample_id": "first", "mole_fraction": ["mean", "std"]})
                               .reset_index())
            pm_pfp.columns = ["site", "sample_datetime", "pair_id_num", "mean", "std"]
            pm_pfp = pm_pfp[["site", "pair_id_num", "sample_datetime", "mean", "std"]]
            pm_frames.append(pm_pfp)

        if pm_frames:
            pm = pd.concat(pm_frames, ignore_index=True)
        else:
            pm = pd.DataFrame(columns=["site", "pair_id_num", "sample_datetime", "mean", "std"])

        datasets["Flask mean"] = fm
        datasets["Pair mean"]  = pm
        return datasets
                
    def on_point_pick(self, event, artist=None, idx=None):
        artist = artist or event.artist

        # ignore hidden artists
        if hasattr(artist, "get_visible") and not artist.get_visible():
            return

        if not hasattr(artist, "_meta"):
            return

        # idx is now the authoritative point index
        if idx is None:
            if not hasattr(event, "ind") or len(event.ind) == 0:
                return
            # fallback: keep your old behavior
            idx = int(event.ind[0])

        nearest_idx = idx

        # These values come straight from your DataFrame (no conversion)
        sample_id   = artist._meta.get("sample_id", [None])[nearest_idx]
        pair_id_num = artist._meta.get("pair_id_num", [None])[nearest_idx]
        run_time    = artist._meta.get("run_time", [None])[nearest_idx]
        sample_time = artist._meta.get("sample_datetime", [None])[nearest_idx]
        site        = artist._meta.get("site", [None])[nearest_idx]
        analyte     = artist._meta.get("analyte", "Unknown")
        channel     = artist._meta.get("channel", None)

        # Always show tooltip
        text = (
            f"<b>Site:</b> {site}<br>"
            f"<b>Sample ID:</b> {sample_id}<br>"
            f"<b>Pair ID:</b> {pair_id_num}<br>"
            f"<b>Sample time:</b> {sample_time}<br>"
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
            
            # --- Ensure run type is "All" so PFPs show up too ---
            self.main_window.runTypeCombo.blockSignals(True)
            self.main_window.runTypeCombo.setCurrentText("All")
            self.main_window.runTypeCombo.blockSignals(False)
    
            # --- Continue with loading the run ---
            self.main_window.set_runlist(initial_date=run_time)
            self.main_window.on_plot_type_changed(self.main_window.current_plot_type)
            self.main_window.current_run_time = str(run_time)
            # no need for the apply button highlight
            self.main_window.apply_date_btn.setStyleSheet("")
