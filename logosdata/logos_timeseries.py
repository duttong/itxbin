from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QComboBox, QGroupBox, QSpinBox, QGridLayout,
    QToolTip, QSizePolicy, QApplication, QShortcut, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QCursor, QKeySequence, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer

from matplotlib.widgets import Button, RadioButtons
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.collections import PathCollection
import pandas as pd
import numpy as np
import colorsys
import time

from pathlib import Path

from data_export import MstarDataExporter, FecdDataExporter

import configparser

_USER_CONF = Path.home() / '.logos_data_user.conf'


def _load_timeseries_years() -> tuple[int, int]:
    """Return (start_year, end_year) from user config, or sensible defaults."""
    cfg = configparser.ConfigParser()
    cfg.read(str(_USER_CONF))
    current_year = pd.Timestamp.now().year
    start = cfg.getint('timeseries', 'start_year', fallback=2020)
    end   = cfg.getint('timeseries', 'end_year',   fallback=current_year)
    return start, end


def _save_timeseries_years(start_year: int, end_year: int) -> None:
    """Persist start/end year to the user config file."""
    cfg = configparser.ConfigParser()
    cfg.read(str(_USER_CONF))
    if not cfg.has_section('timeseries'):
        cfg.add_section('timeseries')
    cfg.set('timeseries', 'start_year', str(start_year))
    cfg.set('timeseries', 'end_year',   str(end_year))
    with open(_USER_CONF, 'w') as fh:
        cfg.write(fh)


LOGOS_sites = ['SUM', 'PSA', 'SPO', 'SMO', 'AMY', 'ALT', 'CGO', 'NWR',
            'LEF', 'BRW', 'RPB', 'KUM', 'MLO', 'WIS', 'THD', 'MHD', 'HFM',
            'BLD', 'MLO_PFP', 'MKO_PFP']

# Pseudo-site names that represent PFP-only subsets of the named base site.
# These do not exist in gmd.site; they are handled by filtering run_type_num=5
# (ng_data_processing_view) or sample_type='PFP' (ng_pair_avg_view).
PFP_SITES = {'MLO_PFP': 'MLO', 'MKO_PFP': 'MKO'}

# Sites only shown when FE3 is active (OTTO predecessor data only).
FE3_EXTRA_SITES = ['ITN', 'USH']

# Sites excluded from the "Export M* Data -- All Sites" action.
# MKO_PFP: only PFP samples exist at MKO, no standard flask M* record.
# BLD: M4-only site, no M1/M3 data.
MSTAR_EXPORT_EXCLUDE = {'MKO_PFP', 'BLD'}


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

    def __init__(self, parent_widget, df, analyte, insitu_df=None):
        self.parent_widget = parent_widget
        self.df = df
        self.analyte = analyte
        self.insitu_df = insitu_df if insitu_df is not None else pd.DataFrame()
        self.channel = parent_widget.current_channel

        # Per-figure state (fully independent)
        self._fig, self._ax = plt.subplots(figsize=(12, 6))
        self.dataset_handles = {}
        # When FE3 has no data (e.g. OTTO-only sites like ITN/USH), show OTTO pair by default.
        fe3_empty = df.empty and parent_widget.instrument.inst_num == 193
        self.dataset_visibility = {"All samples": True, "Flask mean": False, "Pair mean": False, "Air1": True, "Air2": True, "10-day mean": False, "Monthly mean": False, "Mstar pair mean": False, "Mstar 10-day mean": False, "Mstar monthly mean": False, "Otto pair mean": fe3_empty, "Otto 10-day mean": False, "Otto monthly mean": False}
        self.legend_label_map = {
            "Mstar pair mean": "M* pair",
            "Mstar 10-day mean": "M* 10-day",
            "Mstar monthly mean": "M* monthly",
            "Otto pair mean": "OTTO pair",
            "Otto 10-day mean": "OTTO 10-day",
            "Otto monthly mean": "OTTO monthly",
        }
        self.legend_label_lookup = {v: k for k, v in self.legend_label_map.items()}

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

        site_colors = build_site_colors(sites_by_lat)

        # --- Draw datasets ---
        self.dataset_handles = self.parent_widget._draw_dataset_artists(self._ax, datasets, self.analyte)

        if not self.insitu_df.empty:
            insitu_handles = self._draw_insitu_artists(self._ax, site_colors)
            for lbl, artists in insitu_handles.items():
                self.dataset_handles.setdefault(lbl, []).extend(artists)

        tenday_handles = self._draw_10day_mean_artists(self._ax, site_colors)
        for lbl, artists in tenday_handles.items():
            self.dataset_handles.setdefault(lbl, []).extend(artists)

        monthly_handles = self._draw_monthly_mean_artists(self._ax, site_colors)
        for lbl, artists in monthly_handles.items():
            self.dataset_handles.setdefault(lbl, []).extend(artists)

        if self.parent_widget.instrument.inst_num == 192:
            mstar_handles = self._draw_mstar_artists(self._ax, site_colors)
            for lbl, artists in mstar_handles.items():
                self.dataset_handles.setdefault(lbl, []).extend(artists)

        if self.parent_widget.instrument.inst_num == 193:
            otto_handles = self._draw_otto_artists(self._ax, site_colors)
            for lbl, artists in otto_handles.items():
                self.dataset_handles.setdefault(lbl, []).extend(artists)

        self._rebuild_data_artists()

        # Tag each artist with its site (safety)
        for label, artists in self.dataset_handles.items():
            for artist in artists:
                artist._site = getattr(artist, "_site", None)

        # Layout adjustments
        self._fig.tight_layout(rect=[0, 0, 0.85, 1])
        self._fig.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.85)
        self._fig.canvas.draw()

        # Positioning helpers
        axpos = self._ax.get_position()
        panel_left = axpos.x0 + axpos.width + 0.01
        panel_top = axpos.y0 + axpos.height

        # ─── Dataset Legend ───
        # Use a representative site color for Air1/Air2 dummies so they match the plot
        _active = self.parent_widget.get_active_sites()
        _air_base = site_colors.get(_active[0], "gray") if _active else "gray"
        _legend_markers = {"All samples": "o", "Flask mean": "^", "Pair mean": "s", "Air1": "o", "Air2": "o", "10-day mean": "v", "Monthly mean": "D"}
        _legend_colors  = {"All samples": "black", "Flask mean": "black", "Pair mean": "black",
                           "Air1": adjust_brightness(_air_base, 1.0),
                           "Air2": adjust_brightness(_air_base, 0.65),
                           "10-day mean": "black", "Monthly mean": "black"}

        is_m4  = self.parent_widget.instrument.inst_num == 192
        is_fe3 = self.parent_widget.instrument.inst_num == 193
        has_data = not self.df.empty or not self.insitu_df.empty
        flask_entries  = ["All samples", "Flask mean", "Pair mean"] if not self.df.empty else []
        insitu_entries = ["Air1", "Air2"] if not self.insitu_df.empty else []
        tenday_entries  = ["10-day mean"] if has_data else []
        monthly_entries = ["Monthly mean"] if has_data else []
        mstar_entries   = ["Mstar pair mean", "Mstar 10-day mean", "Mstar monthly mean"] if (is_m4 and has_data) else []
        otto_entries    = ["Otto pair mean", "Otto 10-day mean", "Otto monthly mean"] if is_fe3 else []

        _mstar_markers = {"Mstar pair mean": "P", "Mstar 10-day mean": "<", "Mstar monthly mean": "h"}
        _mstar_color = "dimgray"
        _otto_markers = {"Otto pair mean": "P", "Otto 10-day mean": "<", "Otto monthly mean": "h"}
        _otto_color = "dimgray"

        legend_handles = []
        for label in flask_entries + insitu_entries + tenday_entries + monthly_entries:
            dummy = mlines.Line2D([], [], color=_legend_colors[label],
                                marker=_legend_markers[label],
                                linestyle="", markersize=6,
                                label=self.legend_label_map.get(label, label))
            dummy._is_dataset_legend = True
            dummy._dataset_key = label
            if not self.dataset_visibility.get(label, True):
                dummy.set_alpha(0.4)
            legend_handles.append(dummy)

        if mstar_entries:
            divider = mlines.Line2D([], [], color="lightgray", linestyle="-",
                                    linewidth=1.5, markersize=0, label="──────────")
            divider._is_dataset_legend = True
            legend_handles.append(divider)
            for label in mstar_entries:
                dummy = mlines.Line2D([], [], color=_mstar_color,
                                      marker=_mstar_markers[label],
                                      linestyle="", markersize=6,
                                      label=self.legend_label_map.get(label, label))
                dummy._is_dataset_legend = True
                dummy._dataset_key = label
                if not self.dataset_visibility.get(label, True):
                    dummy.set_alpha(0.4)
                legend_handles.append(dummy)

        if otto_entries:
            divider = mlines.Line2D([], [], color="lightgray", linestyle="-",
                                    linewidth=1.5, markersize=0, label="──────────")
            divider._is_dataset_legend = True
            legend_handles.append(divider)
            for label in otto_entries:
                dummy = mlines.Line2D([], [], color=_otto_color,
                                      marker=_otto_markers[label],
                                      linestyle="", markersize=6,
                                      label=self.legend_label_map.get(label, label))
                dummy._is_dataset_legend = True
                dummy._dataset_key = label
                if not self.dataset_visibility.get(label, True):
                    dummy.set_alpha(0.4)
                legend_handles.append(dummy)

        dataset_legend = self._ax.legend(
            handles=legend_handles,
            title="Datasets",
            bbox_to_anchor=(panel_left, axpos.y0),
            bbox_transform=self._fig.transFigure,
            loc="lower left",
            borderaxespad=0.0,
            fontsize=9,
            title_fontsize=10
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
            borderaxespad=0.0,
            fontsize=9,
            title_fontsize=10
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

    def _draw_insitu_artists(self, ax, site_colors):
        """Draw IE3 in-situ air ports grouped by port then site; return dataset_handles dict."""
        PORT_LABELS = {3: "Air1", 7: "Air2"}
        PORT_SHADE  = {"Air1": 1.0, "Air2": 0.65}
        dataset_handles = {}
        sites = self.parent_widget.get_active_sites()
        filtered = self.insitu_df[self.insitu_df["site"].isin(sites)]

        for (port, site), grp in filtered.groupby(["port", "site"]):
            label = PORT_LABELS.get(port, f"Air(port {port})")
            color = adjust_brightness(site_colors.get(site, "gray"), PORT_SHADE[label])
            visible = self.dataset_visibility.get(label, True)
            line, = ax.plot(
                grp["analysis_time"], grp["mole_fraction"],
                marker="o", linestyle="", color=color,
                markersize=4, alpha=0.6, label=label, picker=5
            )
            line._site = site
            line._dataset_label = label
            line._meta = {
                "run_time":        grp["run_time"].tolist(),
                "sample_datetime": grp["analysis_time"].tolist(),
                "site":            grp["site"].tolist(),
                "port":            grp["port"].tolist(),
                "analyte":         self.analyte,
                "channel":         self.channel,
            }
            line.set_visible(visible)
            dataset_handles.setdefault(label, []).append(line)

        return dataset_handles

    def _draw_10day_mean_artists(self, ax, site_colors):
        """Draw 10-day mean ± std for flask and insitu data; return dataset_handles entries.
        Bins: days 1-10 → 1st, days 11-20 → 11th, days 21+ → 21st of each month."""
        handles = {}
        visible = self.dataset_visibility.get("10-day mean", False)

        # ── Flask 10-day means (via ng_pair_avg_view, DB-aggregated) ───
        tenday_df = self.parent_widget.query_10day_mean_data(self.analyte)
        if not tenday_df.empty:
            for _, row in tenday_df.iterrows():
                site = row["site"]
                mean = row["period_avg"]
                std  = row["period_std"]
                if not np.isfinite(mean) or not np.isfinite(std):
                    continue
                mid = row["period_start"]
                color = adjust_brightness(site_colors.get(site, "gray"), 0.9)
                container = ax.errorbar(
                    [mid], [mean], yerr=[std],
                    marker="v", linestyle="",
                    color=color, markersize=6, capsize=3, alpha=0.9,
                    mfc="none", mec=color, label="10-day mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "10-day mean"
                        p.set_visible(visible)
                handles.setdefault("10-day mean", []).extend(parts)

        # ── Insitu 10-day means ─────────────────────────────────────
        if not self.insitu_df.empty:
            PORT_LABELS = {3: "Air1", 7: "Air2"}
            PORT_SHADE  = {"Air1": 1.0, "Air2": 0.65}
            sites = self.parent_widget.get_active_sites()
            insitu = self.insitu_df[self.insitu_df["site"].isin(sites)].copy()
            month_start = insitu["analysis_time"].dt.to_period("M").dt.to_timestamp()
            day = insitu["analysis_time"].dt.day
            offset = pd.to_timedelta(
                np.where(day <= 10, 0, np.where(day <= 20, 10, 20)), unit="D"
            )
            insitu["_period"] = month_start + offset
            for (port, site, period), grp in insitu.groupby(["port", "site", "_period"]):
                vals = grp["mole_fraction"].dropna()
                if len(vals) < 2:
                    continue
                port_label = PORT_LABELS.get(port, f"Air(port {port})")
                color = adjust_brightness(site_colors.get(site, "gray"), PORT_SHADE.get(port_label, 1.0))
                mean = vals.mean()
                std  = vals.std()
                if not np.isfinite(std):
                    continue
                container = ax.errorbar(
                    [period], [mean], yerr=[std],
                    marker="v", linestyle="",
                    color=color, markersize=6, capsize=3, alpha=0.9,
                    mfc="none", mec=color, label="10-day mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "10-day mean"
                        p.set_visible(visible)
                handles.setdefault("10-day mean", []).extend(parts)

        return handles

    def _draw_monthly_mean_artists(self, ax, site_colors):
        """Draw monthly mean ± std for flask and insitu data; return dataset_handles entries."""
        handles = {}
        visible = self.dataset_visibility.get("Monthly mean", False)

        # ── Flask monthly means (via ng_pair_avg_view, DB-aggregated) ───
        pair_df = self.parent_widget.query_monthly_mean_data(self.analyte)
        if not pair_df.empty:
            for _, row in pair_df.iterrows():
                site = row["site"]
                mean = row["monthly_avg"]
                std  = row["monthly_std"]
                if not np.isfinite(mean) or not np.isfinite(std):
                    continue
                mid = row["month_start"]
                color = adjust_brightness(site_colors.get(site, "gray"), 0.9)
                container = ax.errorbar(
                    [mid], [mean], yerr=[std],
                    marker="D", linestyle="",
                    color=color, markersize=7, capsize=4, alpha=0.9,
                    mfc="none", mec=color, label="Monthly mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "Monthly mean"
                        p.set_visible(visible)
                handles.setdefault("Monthly mean", []).extend(parts)

        # ── Insitu monthly means ─────────────────────────────────
        if not self.insitu_df.empty:
            PORT_LABELS = {3: "Air1", 7: "Air2"}
            PORT_SHADE  = {"Air1": 1.0, "Air2": 0.65}
            sites = self.parent_widget.get_active_sites()
            insitu = self.insitu_df[self.insitu_df["site"].isin(sites)].copy()
            insitu["_month"] = insitu["analysis_time"].dt.to_period("M").dt.to_timestamp()
            for (port, site, month), grp in insitu.groupby(["port", "site", "_month"]):
                vals = grp["mole_fraction"].dropna()
                if len(vals) < 2:
                    continue
                port_label = PORT_LABELS.get(port, f"Air(port {port})")
                color = adjust_brightness(site_colors.get(site, "gray"), PORT_SHADE.get(port_label, 1.0))
                mid  = month
                mean = vals.mean()
                std  = vals.std()
                if not np.isfinite(std):
                    continue
                container = ax.errorbar(
                    [mid], [mean], yerr=[std],
                    marker="D", linestyle="",
                    color=color, markersize=7, capsize=4, alpha=0.9,
                    mfc="none", mec=color, label="Monthly mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "Monthly mean"
                        p.set_visible(visible)
                handles.setdefault("Monthly mean", []).extend(parts)

        return handles

    def _draw_mstar_artists(self, ax, site_colors):
        """Draw M1+M3 pair, 10-day, and monthly means (M4 only); return dataset_handles entries."""
        handles = {}

        # ── Mstar pair mean ─────────────────────────────────────────
        pair_df = self.parent_widget.query_mstar_pair_data(self.analyte)
        visible_pair = self.dataset_visibility.get("Mstar pair mean", False)
        if not pair_df.empty:
            for site, grp in pair_df.groupby("site"):
                color = adjust_brightness(site_colors.get(site, "gray"), 0.75)
                line, = ax.plot(
                    grp["sample_datetime"], grp["pair_avg"],
                    marker="P", linestyle="",
                    color=color, markersize=5, alpha=0.6,
                    mfc=color, mec=color, label="Mstar pair mean"
                )
                line._site = site
                line._dataset_label = "Mstar pair mean"
                line.set_visible(visible_pair)
                handles.setdefault("Mstar pair mean", []).append(line)

        # ── Mstar 10-day mean ───────────────────────────────────────
        tenday_df = self.parent_widget.query_mstar_10day_mean_data(self.analyte)
        visible_10d = self.dataset_visibility.get("Mstar 10-day mean", False)
        if not tenday_df.empty:
            for _, row in tenday_df.iterrows():
                site = row["site"]
                mean = row["period_avg"]
                std  = row["period_std"]
                if not np.isfinite(mean) or not np.isfinite(std):
                    continue
                color = adjust_brightness(site_colors.get(site, "gray"), 0.75)
                container = ax.errorbar(
                    [row["period_start"]], [mean], yerr=[std],
                    marker="<", linestyle="",
                    color=color, markersize=6, capsize=3, alpha=0.9,
                    mfc="none", mec=color, label="Mstar 10-day mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "Mstar 10-day mean"
                        p.set_visible(visible_10d)
                handles.setdefault("Mstar 10-day mean", []).extend(parts)

        # ── Mstar monthly mean ──────────────────────────────────────
        monthly_df = self.parent_widget.query_mstar_monthly_mean_data(self.analyte)
        visible_mo = self.dataset_visibility.get("Mstar monthly mean", False)
        if not monthly_df.empty:
            for _, row in monthly_df.iterrows():
                site = row["site"]
                mean = row["monthly_avg"]
                std  = row["monthly_std"]
                if not np.isfinite(mean) or not np.isfinite(std):
                    continue
                color = adjust_brightness(site_colors.get(site, "gray"), 0.75)
                container = ax.errorbar(
                    [row["month_start"]], [mean], yerr=[std],
                    marker="h", linestyle="",
                    color=color, markersize=7, capsize=4, alpha=0.9,
                    mfc="none", mec=color, label="Mstar monthly mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "Mstar monthly mean"
                        p.set_visible(visible_mo)
                handles.setdefault("Mstar monthly mean", []).extend(parts)

        return handles

    def _draw_otto_artists(self, ax, site_colors):
        """Draw OTTO pair, 10-day, and monthly means (FE3 only); return dataset_handles entries."""
        handles = {}

        # ── Otto pair mean ──────────────────────────────────────────
        pair_df = self.parent_widget.query_otto_pair_data(self.analyte)
        visible_pair = self.dataset_visibility.get("Otto pair mean", False)
        if not pair_df.empty:
            for site, grp in pair_df.groupby("site"):
                color = adjust_brightness(site_colors.get(site, "gray"), 0.75)
                line, = ax.plot(
                    grp["sample_datetime"], grp["pair_avg"],
                    marker="P", linestyle="",
                    color=color, markersize=5, alpha=0.6,
                    mfc=color, mec=color, label="Otto pair mean"
                )
                line._site = site
                line._dataset_label = "Otto pair mean"
                line.set_visible(visible_pair)
                handles.setdefault("Otto pair mean", []).append(line)

        # ── Otto 10-day mean ────────────────────────────────────────
        tenday_df = self.parent_widget.query_otto_10day_mean_data(self.analyte)
        visible_10d = self.dataset_visibility.get("Otto 10-day mean", False)
        if not tenday_df.empty:
            for _, row in tenday_df.iterrows():
                site = row["site"]
                mean = row["period_avg"]
                std  = row["period_std"]
                if not np.isfinite(mean) or not np.isfinite(std):
                    continue
                color = adjust_brightness(site_colors.get(site, "gray"), 0.75)
                container = ax.errorbar(
                    [row["period_start"]], [mean], yerr=[std],
                    marker="<", linestyle="",
                    color=color, markersize=6, capsize=3, alpha=0.9,
                    mfc="none", mec=color, label="Otto 10-day mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "Otto 10-day mean"
                        p.set_visible(visible_10d)
                handles.setdefault("Otto 10-day mean", []).extend(parts)

        # ── Otto monthly mean ───────────────────────────────────────
        monthly_df = self.parent_widget.query_otto_monthly_mean_data(self.analyte)
        visible_mo = self.dataset_visibility.get("Otto monthly mean", False)
        if not monthly_df.empty:
            for _, row in monthly_df.iterrows():
                site = row["site"]
                mean = row["monthly_avg"]
                std  = row["monthly_std"]
                if not np.isfinite(mean) or not np.isfinite(std):
                    continue
                color = adjust_brightness(site_colors.get(site, "gray"), 0.75)
                container = ax.errorbar(
                    [row["month_start"]], [mean], yerr=[std],
                    marker="h", linestyle="",
                    color=color, markersize=7, capsize=4, alpha=0.9,
                    mfc="none", mec=color, label="Otto monthly mean"
                )
                parts = []
                for child in container:
                    if isinstance(child, (list, tuple)):
                        parts.extend(child)
                    else:
                        parts.append(child)
                for p in parts:
                    if hasattr(p, "set_visible"):
                        p._site = site
                        p._dataset_label = "Otto monthly mean"
                        p.set_visible(visible_mo)
                handles.setdefault("Otto monthly mean", []).extend(parts)

        return handles

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
        default_text = "Reload"
        self.parent_widget._set_button_loading_state(self.reload_btn, True, default_text)
        try:
            df        = self.parent_widget.query_flask_data(force=True, analyte=self.analyte, channel=self.channel)
            insitu_df = self.parent_widget.query_insitu_data(analyte=self.analyte, force=True)
            # For FE3, allow rebuild even when flask data is empty — OTTO data may exist.
            if df.empty and insitu_df.empty and self.parent_widget.instrument.inst_num != 193:
                print("No data to reload")
                return

            self.df = df
            self.insitu_df = insitu_df
            self._ax.clear()
            self._build_plot()
        finally:
            self.parent_widget._set_button_loading_state(self.reload_btn, False, default_text)

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
            label = getattr(artist, "_dataset_key", artist.get_label())
            label = self.legend_label_lookup.get(label, label)
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
        self._cached_insitu_df = None
        self._last_insitu_params = None
        self._dataset_visibility = {}

        self.analytes = self.instrument.analytes if self.instrument else {}
        self.current_analyte = list(self.analytes.keys())[0] if self.analytes else None
        self.current_channel = None

        controls = QVBoxLayout()

        # Analyte selection
        analyte_group = QGroupBox("ANALYTE")
        analyte_layout = QHBoxLayout()
        self.analyte_combo = QComboBox()
        self.analyte_combo.addItems(list(self.analytes.keys()))
        self.analyte_combo.currentTextChanged.connect(self.set_current_analyte)
        analyte_layout.addWidget(self.analyte_combo)
        analyte_group.setLayout(analyte_layout)
        controls.addWidget(analyte_group)

        # Date range
        date_group = QGroupBox("YEAR RANGE")
        date_layout = QHBoxLayout()
        self.start_year = QSpinBox()
        self.start_year.setRange(1990, 2030)
        self.end_year = QSpinBox()
        self.end_year.setRange(1990, 2030)
        _saved_start, _saved_end = _load_timeseries_years()
        self.start_year.setValue(_saved_start)
        self.end_year.setValue(_saved_end)
        self.start_year.valueChanged.connect(self._save_year_range)
        self.end_year.valueChanged.connect(self._save_year_range)
        date_layout.addWidget(QLabel("Start:"))
        date_layout.addWidget(self.start_year)
        date_layout.addWidget(QLabel("End:"))
        date_layout.addWidget(self.end_year)
        date_group.setLayout(date_layout)
        controls.addWidget(date_group)

        # Site selection
        site_group = QGroupBox("SITES")
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
        default_site = getattr(self.instrument, "site", None)
        default_sites = {default_site.upper()} if default_site else {"BRW", "MLO", "SMO", "SPO"}
        for i, site in enumerate(self.sites_by_lat):
            cb = QCheckBox(site)
            cb.setChecked(site in default_sites)
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

        # Save group — instrument-specific export buttons
        if self.instrument and self.instrument.inst_num == 192:
            save_group = QGroupBox("SAVE")
            save_layout = QVBoxLayout()

            _tip_all = (
                "<b>Export M* Data — All Sites &amp; Time</b><br><br>"
                "Writes a GML-format text file of M-system (M1/M3/M4) flask pair "
                "means and 1-σ standard deviations for <b>all LOGOS network sites</b> "
                "across <b>all available years</b> (ignores the year range spinboxes)."
            )
            _tip_sel = (
                "<b>Export M* Data — Selected Sites &amp; Time</b><br><br>"
                "Writes a GML-format text file of M-system (M1/M3/M4) flask pair "
                "means and 1-σ standard deviations for the <b>sites checked above</b>.<br><br>"
                "<b>Year range:</b> set by the Start / End spinboxes above."
            )

            self.export_mstar_all_btn = QPushButton("Export M* Data -- All Sites and Time")
            self.export_mstar_all_btn.clicked.connect(self._export_mstar_data_all_sites)
            self.export_mstar_sel_btn = QPushButton("Export M* Data -- Selected Sites and Time")
            self.export_mstar_sel_btn.clicked.connect(self._export_mstar_data_selected_sites)

            save_layout.addLayout(self._export_row(self.export_mstar_all_btn, _tip_all))
            save_layout.addLayout(self._export_row(self.export_mstar_sel_btn, _tip_sel))
            save_group.setLayout(save_layout)
            controls.addWidget(save_group)

        if self.instrument and self.instrument.inst_num == 193:
            save_group = QGroupBox("SAVE")
            save_layout = QVBoxLayout()

            _tip_fecd_all = (
                "<b>Export fECD Data — All Sites and Time</b><br><br>"
                "Writes one GML-format text file per site ({site}_{analyte}_All.txt) "
                "containing OTTO and FE3 flask pair averages for <b>all LOGOS network "
                "sites</b> across <b>all available years</b>.<br><br>"
                "You will be prompted to choose an output directory."
            )
            _tip_fecd_sel = (
                "<b>Export fECD Data — Selected Sites and Time</b><br><br>"
                "Writes one GML-format text file per site ({site}_{analyte}_All.txt) "
                "containing OTTO and FE3 flask pair averages for the "
                "<b>sites checked above</b>.<br><br>"
                "<b>Year range:</b> set by the Start / End spinboxes above.<br><br>"
                "You will be prompted to choose an output directory."
            )

            self.export_fecd_all_btn = QPushButton("Export fECD Data -- All Sites and Time")
            self.export_fecd_all_btn.clicked.connect(self._export_fecd_data_all_sites)
            self.export_fecd_sel_btn = QPushButton("Export fECD Data -- Selected Sites and Time")
            self.export_fecd_sel_btn.clicked.connect(self._export_fecd_data_selected_sites)

            save_layout.addLayout(self._export_row(self.export_fecd_all_btn, _tip_fecd_all))
            save_layout.addLayout(self._export_row(self.export_fecd_sel_btn, _tip_fecd_sel))
            save_group.setLayout(save_layout)
            controls.addWidget(save_group)

        controls.addStretch()
        self.setLayout(controls)

    # --- Helpers ---
    def _export_row(self, btn: QPushButton, tooltip_html: str):
        """Return a QHBoxLayout with *btn* and a press-and-hold info icon."""
        icon_path = Path(__file__).parent / 'assets' / 'icons8-info-30.png'
        info_btn = QPushButton()
        info_btn.setFixedSize(22, 22)
        info_btn.setFlat(True)
        info_btn.setCursor(QCursor(Qt.WhatsThisCursor))
        if icon_path.exists():
            pix = QPixmap(str(icon_path)).scaled(18, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            info_btn.setIcon(QIcon(pix))
        else:
            info_btn.setText("?")

        # Press-and-hold: show tooltip after 400 ms, hide on release
        _timer = QTimer(info_btn)
        _timer.setSingleShot(True)
        _timer.setInterval(400)
        _timer.timeout.connect(lambda: QToolTip.showText(
            info_btn.mapToGlobal(info_btn.rect().bottomLeft()), tooltip_html, info_btn
        ))
        info_btn.pressed.connect(_timer.start)
        info_btn.released.connect(lambda: (_timer.stop(), QToolTip.hideText()))

        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(btn, stretch=1)
        row.addWidget(info_btn, stretch=0)
        return row

    def _save_year_range(self):
        _save_timeseries_years(self.start_year.value(), self.end_year.value())

    def _set_button_loading_state(self, button, loading: bool, default_text: str):
        if loading:
            button.setText("Loading data...")
            button.setStyleSheet(
                "QPushButton {"
                "background-color: #f6e7a1;"
                "border: 1px solid #c5ae45;"
                "color: #3f3200;"
                "padding: 3px 6px;"
                "}"
            )
            button.setEnabled(False)
        else:
            button.setText(default_text)
            button.setStyleSheet("")
            button.setEnabled(True)

        QApplication.processEvents()

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
        extra = FE3_EXTRA_SITES if getattr(self.instrument, 'inst_num', None) == 193 else []
        real_sites = [s for s in LOGOS_sites if s not in PFP_SITES] + extra
        # Also fetch base sites for PFP pseudo-sites even if they are not in
        # LOGOS_sites (e.g. MKO was removed as a standalone site but MKO_PFP
        # still needs MKO's lat/lon to place itself in the sorted list).
        pfp_base_sites = [s for s in PFP_SITES.values() if s not in real_sites]
        all_query_sites = real_sites + pfp_base_sites
        sql = f""" SELECT
            code, lat, lon, elev from gmd.site
            WHERE code in {tuple(all_query_sites)}
            ORDER BY code;
            """
        df = pd.DataFrame(self.instrument.doquery(sql))
        # Add virtual rows for PFP pseudo-sites using the base site's lat/lon
        pfp_rows = []
        for pfp_site, base_site in PFP_SITES.items():
            base_row = df[df['code'] == base_site]
            if not base_row.empty:
                row = base_row.iloc[0].copy()
                row['code'] = pfp_site
                pfp_rows.append(row)
        if pfp_rows:
            df = pd.concat([df, pd.DataFrame(pfp_rows)], ignore_index=True)
        # Drop any base sites that were only fetched for lat/lon lookup purposes
        df = df[df['code'].isin(real_sites + list(PFP_SITES.keys()))].copy()
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
                        "run_type_num": grp.get("run_type_num", pd.Series([None]*len(grp))).tolist(),
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
        default_text = "Mole Fractions Figure"
        self._set_button_loading_state(self.plot_button, True, default_text)
        try:
            df = self.query_flask_data(force=False)
            insitu_df = self.query_insitu_data()
            # For FE3, allow plotting even when flask data is empty — OTTO predecessor
            # data may still be available for legacy-only sites like ITN and USH.
            if df.empty and insitu_df.empty and self.instrument.inst_num != 193:
                print("No data to plot")
                return
            analyte = self.analyte_combo.currentText()
            fig = TimeseriesFigure(self, df, analyte, insitu_df=insitu_df)
            self.open_figures.append(fig)
        finally:
            self._set_button_loading_state(self.plot_button, False, default_text)

    def rel_stddev_plot(self):
        """Handle the 'Relative Stddev Plot' button click."""
        df = self.query_rel_stddev_data()
        if df.empty:
            print("No data available for the relative standard deviation plot.")
            return
        analyte = self.analyte_combo.currentText()
        fig = RelStdDevFigure(self, df, analyte)
        self.open_figures.append(fig)

    def _export_mstar_data_all_sites(self):
        """Export M* data for all sites and all time, minus those in MSTAR_EXPORT_EXCLUDE."""
        sites = [s for s in self.sites_by_lat if s not in MSTAR_EXPORT_EXCLUDE]
        self._run_mstar_export(sites=sites, all_time=True)

    def _export_mstar_data_selected_sites(self):
        """Export M* data for the currently checked sites and the selected year range."""
        self._run_mstar_export(sites=self.get_active_sites(), all_time=False)

    def _run_mstar_export(self, sites: list[str], all_time: bool = False):
        """Shared logic: build exporter, prompt for path, write file."""
        exporter = MstarDataExporter.from_timeseries_widget(self, sites=sites, all_time=all_time)
        default_name = exporter.default_filename()
        path, _ = QFileDialog.getSaveFileName(
            self, "Export M* Data", default_name, "Text files (*.txt);;All files (*)"
        )
        if not path:
            return
        n = exporter.export(path)
        if n == 0:
            QMessageBox.warning(self, "Export M* Data", "No data found for the current selection.")
        else:
            QMessageBox.information(self, "Export M* Data", f"Wrote {n} records to {path}")

    def _export_fecd_data_all_sites(self):
        """Export fECD data for all sites and all time to a user-chosen directory."""
        sites = [s for s in self.sites_by_lat if s not in PFP_SITES]
        self._run_fecd_export(sites=sites, all_time=True)

    def _export_fecd_data_selected_sites(self):
        """Export fECD data for the currently checked sites and the selected year range."""
        sites = [s for s in self.get_active_sites() if s not in PFP_SITES]
        self._run_fecd_export(sites=sites, all_time=False)

    def _run_fecd_export(self, sites: list[str], all_time: bool = False):
        """Shared logic: build fECD exporter, prompt for output dir, write per-site files."""
        exporter = FecdDataExporter.from_timeseries_widget(self, sites=sites, all_time=all_time)
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for fECD Data"
        )
        if not output_dir:
            return
        results = exporter.export_all(output_dir)
        if not results:
            QMessageBox.warning(self, "Export fECD Data",
                                "No data found for the current selection.")
        else:
            total = sum(results.values())
            files = len(results)
            QMessageBox.information(
                self, "Export fECD Data",
                f"Wrote {total} records across {files} file(s) in:\n{output_dir}"
            )

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

    def _uses_forced_preferred_channel(self) -> bool:
        """True when a caller wants ng_preferred_channel to choose duplicate channels."""
        return bool(getattr(self, "force_preferred_channel", False)) and hasattr(
            self.instrument, "return_preferred_channel"
        )

    def _preferred_channel_filter_sql(
        self,
        channel_expr: str,
        parameter_expr: str,
        date_expr: str,
    ) -> str:
        """Return a SQL filter matching the preferred channel for each row date."""
        if not self._uses_forced_preferred_channel():
            return ""

        inst_num = int(self.instrument.inst_num)
        return f"""
              AND {channel_expr} = COALESCE(
                  (
                      SELECT pc.channel
                      FROM hats.ng_preferred_channel pc
                      WHERE pc.inst_num = {inst_num}
                        AND pc.parameter_num = {parameter_expr}
                        AND pc.start_date <= {date_expr}
                      ORDER BY pc.start_date DESC
                      LIMIT 1
                  ),
                  (
                      SELECT pc.channel
                      FROM hats.ng_preferred_channel pc
                      WHERE pc.inst_num = {inst_num}
                        AND pc.parameter_num = {parameter_expr}
                      ORDER BY pc.start_date ASC
                      LIMIT 1
                  ),
                  {channel_expr}
              )
        """

    def query_flask_data(self, force: bool = False, analyte: str | None = None, channel: str | None = None) -> pd.DataFrame:
        start = self.start_year.value()
        end   = self.end_year.value()

        analyte = analyte or self.analyte_combo.currentText()
        pnum    = self.analytes.get(analyte)
        self.set_current_analyte(analyte)
        use_preferred_channel = self._uses_forced_preferred_channel()
        channel = None if use_preferred_channel else channel or self.current_channel

        if pnum is None:
            return pd.DataFrame()

        ch_str = (
            self._preferred_channel_filter_sql("channel", "parameter_num", "sample_datetime")
            if use_preferred_channel
            else ('' if channel is None else f'AND channel = "{channel}"')
        )
        query_params = (start, end, analyte, channel, use_preferred_channel)

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
                # Split PFP runs out of base sites → separate MLO_PFP / MKO_PFP series.
                # Base site (MLO, MKO) keeps only non-PFP runs; PFP pseudo-site gets
                # run_type_num=5 rows relabelled with the pseudo-site name.
                pfp_frames = []
                for pfp_site, base_site in PFP_SITES.items():
                    pfp_rows = df[(df['site'] == base_site) & (df['run_type_num'] == 5)].copy()
                    if not pfp_rows.empty:
                        pfp_rows['site'] = pfp_site
                        pfp_frames.append(pfp_rows)
                # Remove PFP rows from base sites so they appear only under the pseudo-site
                pfp_base_sites = set(PFP_SITES.values())
                df = df[~((df['site'].isin(pfp_base_sites)) & (df['run_type_num'] == 5))].copy()
                if pfp_frames:
                    df = pd.concat([df] + pfp_frames, ignore_index=True).sort_values('sample_datetime')
                self._cached_df = df
            self._last_query_params = query_params

        return self._cached_df.copy() if self._cached_df is not None else pd.DataFrame()


    def query_insitu_data(self, analyte: str | None = None, force: bool = False) -> pd.DataFrame:
        """Query unflagged IE3 in-situ air port data for the selected analyte and date range."""
        if self.instrument.inst_num != 236:
            return pd.DataFrame()

        start = self.start_year.value()
        end = self.end_year.value()

        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()

        # Extract channel from analyte name, e.g. "CFC12 (b)" -> "b"
        channel = None
        if "(" in analyte and ")" in analyte:
            channel = analyte.split("(", 1)[1].strip(") ")
        use_preferred_channel = self._uses_forced_preferred_channel()
        if use_preferred_channel:
            channel = None

        query_params = (start, end, analyte, use_preferred_channel)
        if not force and query_params == self._last_insitu_params and self._cached_insitu_df is not None:
            return self._cached_insitu_df.copy()

        ch_filter = (
            self._preferred_channel_filter_sql("mf.channel", "mf.parameter_num", "a.analysis_time")
            if use_preferred_channel
            else (f"AND mf.channel = '{channel}'" if channel else "")
        )
        sql = f"""
        SELECT a.run_time, a.analysis_time, s.code AS site, mf.mole_fraction, a.port, mf.channel
        FROM hats.ng_insitu_analysis a
        JOIN hats.ng_insitu_mole_fractions mf ON a.num = mf.analysis_num
        JOIN gmd.site s ON a.site_num = s.num
        WHERE a.inst_num = 236
          AND a.port IN (3, 7)
          AND mf.flag = '...'
          AND mf.parameter_num = {pnum}
          {ch_filter}
          AND YEAR(a.analysis_time) BETWEEN {start} AND {end}
        ORDER BY a.analysis_time;
        """
        df = pd.DataFrame(self.instrument.doquery(sql))
        if not df.empty:
            df["run_time"]      = pd.to_datetime(df["run_time"])
            df["analysis_time"] = pd.to_datetime(df["analysis_time"])
        self._cached_insitu_df = df
        self._last_insitu_params = query_params
        return df.copy() if not df.empty else df

    def query_10day_mean_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query ng_pair_avg_view for 10-day means (M4 and FE3 only; IE3 handled via insitu).
        Bins: days 1-10 → 1st, days 11-20 → 11th, days 21+ → 21st of each month.
        PFP pseudo-sites (MLO_PFP, MKO_PFP) use sample_type='PFP' filter."""
        if self.instrument.inst_num == 236:
            return pd.DataFrame()

        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()

        channel = None
        if "(" in analyte and ")" in analyte:
            channel = analyte.split("(", 1)[1].strip(") ")
        use_preferred_channel = self._uses_forced_preferred_channel()
        if use_preferred_channel:
            channel = None
        ch_filter = (
            self._preferred_channel_filter_sql("v.channel", "v.parameter_num", "v.sample_datetime")
            if use_preferred_channel
            else (f"AND v.channel = '{channel}'" if channel else "")
        )

        start = self.start_year.value()
        end   = self.end_year.value()
        sites = self.get_active_sites()
        if not sites:
            return pd.DataFrame()

        _period_expr = """CASE
                WHEN DAY(sample_datetime) <= 10 THEN DATE_FORMAT(sample_datetime, '%%Y-%%m-01')
                WHEN DAY(sample_datetime) <= 20 THEN DATE_FORMAT(sample_datetime, '%%Y-%%m-11')
                ELSE DATE_FORMAT(sample_datetime, '%%Y-%%m-21')
            END"""

        frames = []
        regular_sites = [s for s in sites if s not in PFP_SITES]
        pfp_pseudo_sites = [s for s in sites if s in PFP_SITES]

        # For sites that have a PFP pseudo-site counterpart, exclude PFP rows so
        # the means here cover flask-only data (PFP data belongs to the pseudo-site).
        pfp_base_sites = set(PFP_SITES.values())
        has_pfp_base = any(s in pfp_base_sites for s in regular_sites)
        pfp_exclusion = "AND v.sample_type IN ('S', 'G')" if has_pfp_base else ""

        if regular_sites:
            sql = f"""
            SELECT v.site, {_period_expr} AS period_start,
                AVG(v.pair_avg) AS period_avg, STDDEV(v.pair_avg) AS period_std
            FROM hats.ng_pair_avg_view v
            WHERE v.inst_num = %s AND v.parameter_num = %s {ch_filter}
              AND v.site IN ({",".join(["%s"] * len(regular_sites))})
              {pfp_exclusion}
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
            GROUP BY v.site, period_start ORDER BY v.site, period_start;
            """
            params = [self.instrument.inst_num, pnum] + regular_sites + [start, end]
            frames.append(pd.DataFrame(self.instrument.doquery(sql, params)))

        for pfp_site in pfp_pseudo_sites:
            base_site = PFP_SITES[pfp_site]
            sql = f"""
            SELECT %s AS site, {_period_expr} AS period_start,
                AVG(v.pair_avg) AS period_avg, STDDEV(v.pair_avg) AS period_std
            FROM hats.ng_pair_avg_view v
            WHERE v.inst_num = %s AND v.parameter_num = %s {ch_filter}
              AND v.sample_type = 'PFP' AND v.site = %s
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
            GROUP BY period_start ORDER BY period_start;
            """
            params = [pfp_site, self.instrument.inst_num, pnum, base_site, start, end]
            frames.append(pd.DataFrame(self.instrument.doquery(sql, params)))

        non_empty = [f for f in frames if not f.empty]
        df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        if not df.empty:
            df["period_start"] = pd.to_datetime(df["period_start"])
        return df

    def query_monthly_mean_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query ng_pair_avg_view for flask monthly means (M4 and FE3 only; IE3 handled via insitu).
        PFP pseudo-sites (MLO_PFP, MKO_PFP) use sample_type='PFP' filter."""
        if self.instrument.inst_num == 236:
            return pd.DataFrame()

        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()

        channel = None
        if "(" in analyte and ")" in analyte:
            channel = analyte.split("(", 1)[1].strip(") ")
        use_preferred_channel = self._uses_forced_preferred_channel()
        if use_preferred_channel:
            channel = None
        ch_filter = (
            self._preferred_channel_filter_sql("v.channel", "v.parameter_num", "v.sample_datetime")
            if use_preferred_channel
            else (f"AND v.channel = '{channel}'" if channel else "")
        )

        start = self.start_year.value()
        end   = self.end_year.value()
        sites = self.get_active_sites()
        if not sites:
            return pd.DataFrame()

        frames = []
        regular_sites = [s for s in sites if s not in PFP_SITES]
        pfp_pseudo_sites = [s for s in sites if s in PFP_SITES]

        # Exclude PFP rows from base sites that have a PFP pseudo-site counterpart
        pfp_base_sites = set(PFP_SITES.values())
        has_pfp_base = any(s in pfp_base_sites for s in regular_sites)
        pfp_exclusion = "AND v.sample_type IN ('S', 'G')" if has_pfp_base else ""

        if regular_sites:
            sql = f"""
            SELECT v.site, DATE_FORMAT(v.sample_datetime, '%%Y-%%m-01') AS month_start,
                AVG(v.pair_avg) AS monthly_avg, STDDEV(v.pair_avg) AS monthly_std
            FROM hats.ng_pair_avg_view v
            WHERE v.inst_num = %s AND v.parameter_num = %s {ch_filter}
              AND v.site IN ({",".join(["%s"] * len(regular_sites))})
              {pfp_exclusion}
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
            GROUP BY v.site, month_start ORDER BY v.site, month_start;
            """
            params = [self.instrument.inst_num, pnum] + regular_sites + [start, end]
            frames.append(pd.DataFrame(self.instrument.doquery(sql, params)))

        for pfp_site in pfp_pseudo_sites:
            base_site = PFP_SITES[pfp_site]
            sql = f"""
            SELECT %s AS site, DATE_FORMAT(v.sample_datetime, '%%Y-%%m-01') AS month_start,
                AVG(v.pair_avg) AS monthly_avg, STDDEV(v.pair_avg) AS monthly_std
            FROM hats.ng_pair_avg_view v
            WHERE v.inst_num = %s AND v.parameter_num = %s {ch_filter}
              AND v.sample_type = 'PFP' AND v.site = %s
              AND YEAR(v.sample_datetime) BETWEEN %s AND %s
            GROUP BY month_start ORDER BY month_start;
            """
            params = [pfp_site, self.instrument.inst_num, pnum, base_site, start, end]
            frames.append(pd.DataFrame(self.instrument.doquery(sql, params)))

        non_empty = [f for f in frames if not f.empty]
        df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        if not df.empty:
            df["month_start"] = pd.to_datetime(df["month_start"])
        return df

    def query_pr1_monthly_mean_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query PR1 monthly means from legacy HATS tables.

        Regular LOGOS sites use PR1 sample_type='HATS'.  PFP pseudo-sites
        (MLO_PFP, MKO_PFP) use sample_type='PFP' on the matching base site.
        For HATS rows, analysis.event_num stores hats.Status_MetData.PairID;
        the sample datetime comes from Status_MetData.sample_datetime_utc.
        For PFP rows, analysis.event_num stores ccgg.flask_event.num.
        """
        if self.instrument.inst_num != 58:
            return pd.DataFrame()

        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()

        start = self.start_year.value()
        end = self.end_year.value()
        sites = self.get_active_sites()
        if not sites:
            return pd.DataFrame()

        frames = []
        regular_sites = [s for s in sites if s not in PFP_SITES]
        pfp_pseudo_sites = [s for s in sites if s in PFP_SITES]

        if regular_sites:
            sql = f"""
            SELECT
                event_data.site,
                DATE_FORMAT(event_data.sample_datetime, '%%Y-%%m-01') AS month_start,
                AVG(event_data.event_avg) AS monthly_avg,
                STDDEV(event_data.event_avg) AS monthly_std,
                COUNT(*) AS n
            FROM (
                SELECT
                    UPPER(sm.Station) AS site,
                    sm.sample_datetime_utc AS sample_datetime,
                    a.event_num,
                    AVG(mf.C_reported) AS event_avg
                FROM hats.analysis a
                JOIN hats.mole_fractions mf
                  ON mf.analysis_num = a.num
                JOIN hats.Status_MetData sm
                  ON sm.PairID = a.event_num
                WHERE a.inst_num = 58
                  AND mf.parameter_num = %s
                  AND a.event_num > 0
                  AND a.site_num > 0
                  AND a.sample_type = 'HATS'
                  AND sm.sample_datetime_utc IS NOT NULL
                  AND mf.C_reported IS NOT NULL
                  AND mf.C_reported > -900
                  AND UPPER(sm.Station) IN ({",".join(["%s"] * len(regular_sites))})
                  AND YEAR(sm.sample_datetime_utc) BETWEEN %s AND %s
                  AND NOT EXISTS (
                      SELECT 1
                      FROM hats.flags_internal f
                      WHERE f.analysis_num = a.num
                        AND f.parameter_num = mf.parameter_num
                        AND COALESCE(f.iflag, '') <> ''
                  )
                GROUP BY site, sample_datetime, a.event_num
            ) AS event_data
            GROUP BY event_data.site, month_start
            ORDER BY site, month_start;
            """
            params = [pnum] + regular_sites + [start, end]
            frames.append(pd.DataFrame(self.instrument.doquery(sql, params)))

        for pfp_site in pfp_pseudo_sites:
            base_site = PFP_SITES[pfp_site]
            sql = """
            SELECT
                event_data.site,
                DATE_FORMAT(event_data.sample_datetime, '%%Y-%%m-01') AS month_start,
                AVG(event_data.event_avg) AS monthly_avg,
                STDDEV(event_data.event_avg) AS monthly_std,
                COUNT(*) AS n
            FROM (
                SELECT
                    %s AS site,
                    TIMESTAMP(fe.date, fe.time) AS sample_datetime,
                    a.event_num,
                    AVG(mf.C_reported) AS event_avg
                FROM hats.analysis a
                JOIN hats.mole_fractions mf
                  ON mf.analysis_num = a.num
                JOIN ccgg.flask_event fe
                  ON fe.num = a.event_num
                JOIN gmd.site s
                  ON s.num = a.site_num
                WHERE a.inst_num = 58
                  AND mf.parameter_num = %s
                  AND a.event_num > 0
                  AND a.site_num > 0
                  AND a.sample_type = 'PFP'
                  AND mf.C_reported IS NOT NULL
                  AND mf.C_reported > -900
                  AND UPPER(s.code) = %s
                  AND YEAR(fe.date) BETWEEN %s AND %s
                  AND NOT EXISTS (
                      SELECT 1
                      FROM hats.flags_internal f
                      WHERE f.analysis_num = a.num
                        AND f.parameter_num = mf.parameter_num
                        AND COALESCE(f.iflag, '') <> ''
                  )
                GROUP BY site, sample_datetime, a.event_num
            ) AS event_data
            GROUP BY event_data.site, month_start
            ORDER BY month_start;
            """
            params = [pfp_site, pnum, base_site, start, end]
            frames.append(pd.DataFrame(self.instrument.doquery(sql, params)))

        non_empty = [f for f in frames if not f.empty]
        df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
        if not df.empty:
            df["month_start"] = pd.to_datetime(df["month_start"])
        return df

    def query_mstar_pair_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query M1+M3 individual pair rows from ng_pair_avg_view (M4 only)."""
        if self.instrument.inst_num != 192:
            return pd.DataFrame()
        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()
        start = self.start_year.value()
        end   = self.end_year.value()
        # M* data is M1/M3 flask-only; PFP pseudo-sites don't apply
        sites = [s for s in self.get_active_sites() if s not in PFP_SITES]
        if not sites:
            return pd.DataFrame()
        sql = f"""
        SELECT UPPER(site) AS site, sample_datetime, pair_avg, pair_stdv
        FROM hats.ng_pair_avg_view
        WHERE inst_id IN ('M1', 'M3')
          AND parameter_num = %s
          AND sample_type IN ('S', 'G')
          AND UPPER(site) IN ({",".join(["%s"] * len(sites))})
          AND YEAR(sample_datetime) BETWEEN %s AND %s
        ORDER BY site, sample_datetime
        """
        params = [pnum] + sites + [start, end]
        df = pd.DataFrame(self.instrument.doquery(sql, params))
        if not df.empty:
            df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])
        return df

    def query_mstar_10day_mean_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query M1+M3 10-day binned means from ng_pair_avg_view (M4 only)."""
        if self.instrument.inst_num != 192:
            return pd.DataFrame()
        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()
        start = self.start_year.value()
        end   = self.end_year.value()
        # M* data is M1/M3 flask-only; PFP pseudo-sites don't apply
        sites = [s for s in self.get_active_sites() if s not in PFP_SITES]
        if not sites:
            return pd.DataFrame()
        sql = f"""
        SELECT
            UPPER(site) AS site,
            CASE
                WHEN DAY(sample_datetime) <= 10 THEN DATE_FORMAT(sample_datetime, '%%Y-%%m-01')
                WHEN DAY(sample_datetime) <= 20 THEN DATE_FORMAT(sample_datetime, '%%Y-%%m-11')
                ELSE DATE_FORMAT(sample_datetime, '%%Y-%%m-21')
            END AS period_start,
            AVG(pair_avg)    AS period_avg,
            STDDEV(pair_avg) AS period_std
        FROM hats.ng_pair_avg_view
        WHERE inst_id IN ('M1', 'M3')
          AND parameter_num = %s
          AND sample_type IN ('S', 'G')
          AND UPPER(site) IN ({",".join(["%s"] * len(sites))})
          AND YEAR(sample_datetime) BETWEEN %s AND %s
        GROUP BY site, period_start
        ORDER BY site, period_start
        """
        params = [pnum] + sites + [start, end]
        df = pd.DataFrame(self.instrument.doquery(sql, params))
        if not df.empty:
            df["period_start"] = pd.to_datetime(df["period_start"])
        return df

    def query_mstar_monthly_mean_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query M1+M3 monthly means from ng_pair_avg_view (M4 only)."""
        if self.instrument.inst_num != 192:
            return pd.DataFrame()
        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()
        start = self.start_year.value()
        end   = self.end_year.value()
        # M* data is M1/M3 flask-only; PFP pseudo-sites don't apply
        sites = [s for s in self.get_active_sites() if s not in PFP_SITES]
        if not sites:
            return pd.DataFrame()
        sql = f"""
        SELECT
            UPPER(site) AS site,
            DATE_FORMAT(sample_datetime, '%%Y-%%m-01') AS month_start,
            AVG(pair_avg)    AS monthly_avg,
            STDDEV(pair_avg) AS monthly_std
        FROM hats.ng_pair_avg_view
        WHERE inst_id IN ('M1', 'M3')
          AND parameter_num = %s
          AND sample_type IN ('S', 'G')
          AND UPPER(site) IN ({",".join(["%s"] * len(sites))})
          AND YEAR(sample_datetime) BETWEEN %s AND %s
        GROUP BY site, month_start
        ORDER BY site, month_start
        """
        params = [pnum] + sites + [start, end]
        df = pd.DataFrame(self.instrument.doquery(sql, params))
        if not df.empty:
            df["month_start"] = pd.to_datetime(df["month_start"])
        return df

    def query_otto_pair_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query OTTO individual pair rows from ng_pair_avg_view (FE3 only)."""
        if self.instrument.inst_num != 193:
            return pd.DataFrame()
        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()
        start = self.start_year.value()
        end   = self.end_year.value()
        sites = [s for s in self.get_active_sites() if s not in PFP_SITES]
        if not sites:
            return pd.DataFrame()
        sql = f"""
        SELECT UPPER(site) AS site, sample_datetime, pair_avg, pair_stdv
        FROM hats.ng_pair_avg_view
        WHERE inst_id = 'OTTO'
          AND parameter_num = %s
          AND sample_type IN ('S', 'G')
          AND UPPER(site) IN ({",".join(["%s"] * len(sites))})
          AND YEAR(sample_datetime) BETWEEN %s AND %s
        ORDER BY site, sample_datetime
        """
        params = [pnum] + sites + [start, end]
        df = pd.DataFrame(self.instrument.doquery(sql, params))
        if not df.empty:
            df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])
        return df

    def query_otto_10day_mean_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query OTTO 10-day binned means from ng_pair_avg_view (FE3 only)."""
        if self.instrument.inst_num != 193:
            return pd.DataFrame()
        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()
        start = self.start_year.value()
        end   = self.end_year.value()
        sites = [s for s in self.get_active_sites() if s not in PFP_SITES]
        if not sites:
            return pd.DataFrame()
        sql = f"""
        SELECT
            UPPER(site) AS site,
            CASE
                WHEN DAY(sample_datetime) <= 10 THEN DATE_FORMAT(sample_datetime, '%%Y-%%m-01')
                WHEN DAY(sample_datetime) <= 20 THEN DATE_FORMAT(sample_datetime, '%%Y-%%m-11')
                ELSE DATE_FORMAT(sample_datetime, '%%Y-%%m-21')
            END AS period_start,
            AVG(pair_avg)    AS period_avg,
            STDDEV(pair_avg) AS period_std
        FROM hats.ng_pair_avg_view
        WHERE inst_id = 'OTTO'
          AND parameter_num = %s
          AND sample_type IN ('S', 'G')
          AND UPPER(site) IN ({",".join(["%s"] * len(sites))})
          AND YEAR(sample_datetime) BETWEEN %s AND %s
        GROUP BY site, period_start
        ORDER BY site, period_start
        """
        params = [pnum] + sites + [start, end]
        df = pd.DataFrame(self.instrument.doquery(sql, params))
        if not df.empty:
            df["period_start"] = pd.to_datetime(df["period_start"])
        return df

    def query_otto_monthly_mean_data(self, analyte: str | None = None) -> pd.DataFrame:
        """Query OTTO monthly means from ng_pair_avg_view (FE3 only)."""
        if self.instrument.inst_num != 193:
            return pd.DataFrame()
        analyte = analyte or self.analyte_combo.currentText()
        pnum = self.analytes.get(analyte)
        if pnum is None:
            return pd.DataFrame()
        start = self.start_year.value()
        end   = self.end_year.value()
        sites = [s for s in self.get_active_sites() if s not in PFP_SITES]
        if not sites:
            return pd.DataFrame()
        sql = f"""
        SELECT
            UPPER(site) AS site,
            DATE_FORMAT(sample_datetime, '%%Y-%%m-01') AS month_start,
            AVG(pair_avg)    AS monthly_avg,
            STDDEV(pair_avg) AS monthly_std
        FROM hats.ng_pair_avg_view
        WHERE inst_id = 'OTTO'
          AND parameter_num = %s
          AND sample_type IN ('S', 'G')
          AND UPPER(site) IN ({",".join(["%s"] * len(sites))})
          AND YEAR(sample_datetime) BETWEEN %s AND %s
        GROUP BY site, month_start
        ORDER BY site, month_start
        """
        params = [pnum] + sites + [start, end]
        df = pd.DataFrame(self.instrument.doquery(sql, params))
        if not df.empty:
            df["month_start"] = pd.to_datetime(df["month_start"])
        return df

    def build_datasets(self, df: pd.DataFrame) -> dict:
        if df.empty:
            all_s = pd.DataFrame(columns=["sample_datetime", "run_time", "analysis_datetime",
                                          "mole_fraction", "channel", "data_flag", "site",
                                          "sample_id", "pair_id_num", "run_type_num"])
            fm = pd.DataFrame(columns=["site", "sample_id", "sample_datetime", "mean", "std"])
            pm = pd.DataFrame(columns=["site", "pair_id_num", "sample_datetime", "mean", "std"])
            return {"All samples": all_s, "Flask mean": fm, "Pair mean": pm}

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

        def _mval(key):
            vals = artist._meta.get(key)
            if not vals or nearest_idx >= len(vals):
                return None
            return vals[nearest_idx]

        sample_id   = _mval("sample_id")
        pair_id_num = _mval("pair_id_num")
        run_type_num = _mval("run_type_num")
        run_time    = _mval("run_time")
        sample_time = _mval("sample_datetime")
        site        = _mval("site")
        port        = _mval("port")
        air_label   = getattr(artist, "_dataset_label", None)
        analyte     = artist._meta.get("analyte", "Unknown")
        channel     = artist._meta.get("channel", None)

        # For in-situ points run_time is absent; fall back to analysis_time
        effective_run_time = run_time if run_time is not None else sample_time

        # Build tooltip, skipping fields with no data
        lines = [f"<b>Site:</b> {site}"]
        if air_label  is not None and port is not None:
            lines.append(f"<b>Port:</b> {air_label} (port {port})")
        if sample_id   is not None: lines.append(f"<b>Sample ID:</b> {sample_id}")
        if pair_id_num is not None: lines.append(f"<b>Pair ID:</b> {pair_id_num}")
        if run_type_num is not None:
            lines.append(f"<b>Flask Type:</b> {'PFP' if int(run_type_num) == 5 else 'Flask'}")
        if sample_time is not None: lines.append(f"<b>Time:</b> {sample_time}")
        if run_time    is not None: lines.append(f"<b>Run time:</b> {run_time}")
        QToolTip.showText(QCursor.pos(), "<br>".join(lines))

        # Right click adds extra action -- loads the run in main window
        if event.mouseevent.button == 3:  # right click
            self.main_window.current_run_time = str(effective_run_time)
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
            t = effective_run_time
            if not isinstance(t, pd.Timestamp):
                t = pd.to_datetime(t)

            end_year   = t.year
            end_month  = t.month
            start_dt   = t - pd.DateOffset(months=1)
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
            self.main_window.set_runlist(initial_date=t)
            self.main_window.on_plot_type_changed(self.main_window.current_plot_type)
            self.main_window.current_run_time = str(t)
            # no need for the apply button highlight
            self.main_window.apply_date_btn.setStyleSheet("")
