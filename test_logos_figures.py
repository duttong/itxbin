import configparser
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QTabWidget,
)
from matplotlib.figure import Figure
from logosdata import logos_figures
from logosdata.logos_data import MainWindow


class FiguresTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_all_instrument_configs_hide_ai_and_enable_figures(self):
        config = configparser.ConfigParser()
        config.read(Path(logos_figures.__file__).with_name("logos_data.conf"))
        for section in config.sections():
            with self.subTest(section=section):
                tabs = [item.strip() for item in config[section]["tabs"].split(",")]
                self.assertNotIn("ai", tabs)
                self.assertIn("figures", tabs)

    def make_panel(self, inst_id="m4"):
        run_cb = QComboBox()
        run_cb.addItem("2026-10-02 11:30:00 (Cal)")
        panel = logos_figures.FiguresWidget(
            SimpleNamespace(inst_id=inst_id, gc_dir=Path("/example/m4")),
            SimpleNamespace(run_cb=run_cb),
        )
        self.addCleanup(panel.deleteLater)
        self.addCleanup(panel.plot_widget.deleteLater)
        return panel

    def test_button_renders_selected_run_and_replaces_previous_plot(self):
        panel = self.make_panel()
        bd = Path("/example/bd100226.txt")
        with patch.object(logos_figures, "bd_file_for_run", return_value=bd) as locate, patch.object(logos_figures, "build_figure", side_effect=[(Figure(), 1), (Figure(), 2)]) as build:
            panel.thermocouples_btn.click()
            first = panel.canvas
            panel.thermocouples_btn.click()
        locate.assert_called_with(
            Path("/example/m4/MassHunter/GCMS/M4 GSPC Files"),
            "2026-10-02 11:30:00 (Cal)",
        )
        build.assert_called_with(bd)
        self.assertIsNot(panel.canvas, first)
        self.assertEqual(panel.plot_layout.count(), 3)  # message, canvas, toolbar
        self.assertTrue(panel.thermocouples_btn.isEnabled())

    def test_missing_data_reports_error_and_restores_button_and_cursor(self):
        panel = self.make_panel()
        with patch.object(logos_figures, "bd_file_for_run", side_effect=FileNotFoundError("missing log")), patch.object(logos_figures.QMessageBox, "warning") as warning:
            panel.thermocouples_btn.click()
        self.assertEqual(warning.call_args.args[2], "missing log")
        self.assertTrue(panel.thermocouples_btn.isEnabled())
        self.assertIsNone(QApplication.overrideCursor())

    def test_no_run_prompts_for_selection(self):
        panel = self.make_panel()
        panel.main_window.run_cb.clear()
        with patch.object(logos_figures.QMessageBox, "information") as info, patch.object(logos_figures, "build_figure") as build:
            panel.thermocouples_btn.click()
        info.assert_called_once()
        build.assert_not_called()

    def test_other_instruments_have_no_thermocouples_button(self):
        panel = self.make_panel("fe3")
        self.assertFalse(hasattr(panel, "thermocouples_btn"))

    def test_auto_refresh_starts_after_render_and_skips_unchanged_or_empty_runs(self):
        panel = self.make_panel()
        with patch.object(panel, "show_thermocouples") as render:
            panel.refresh_for_run()
            render.assert_not_called()
        with patch.object(logos_figures, "bd_file_for_run", return_value=Path("bd100226.txt")), patch.object(logos_figures, "build_figure", return_value=(Figure(), 1)):
            panel.show_thermocouples()
        with patch.object(panel, "show_thermocouples") as render:
            panel.refresh_for_run()
            render.assert_not_called()
            panel.main_window.run_cb.addItem("2026-10-01 09:00:00")
            panel.main_window.run_cb.setCurrentIndex(1)
            panel.refresh_for_run()
            render.assert_called_once()
            render.reset_mock()
            panel.main_window.run_cb.clear()
            panel.refresh_for_run()
            render.assert_not_called()

    def test_run_navigation_and_date_range_reload_refresh_rendered_figure(self):
        panel = self.make_panel()
        window = panel.main_window
        window.current_run_times = ["2026-10-01 09:00:00", "2026-10-02 11:30:00"]
        window.run_cb.clear()
        window.run_cb.addItems(window.current_run_times)
        window.current_run_time = window.current_run_times[0]
        window.current_plot_type = 0
        window._multi_tag_panel = None
        window.plot_radio_group = Mock()
        window.plot_radio_group.checkedId.return_value = 0
        for method in ("_refresh_preferred_channel_markers", "_clear_highlight",
                       "load_selected_run", "_update_calibration_button_state",
                       "_update_notes_button_style", "on_plot_type_changed"):
            setattr(window, method, Mock())
        window.figures_tab = panel
        window.on_run_changed = lambda index: MainWindow.on_run_changed(window, index)
        with patch.object(logos_figures, "bd_file_for_run", return_value=Path("bd100126.txt")), patch.object(logos_figures, "build_figure", return_value=(Figure(), 1)):
            panel.show_thermocouples()
        with patch.object(panel, "show_thermocouples") as render:
            MainWindow.on_next_run(window)
            render.assert_called_once()
        window.instrument = SimpleNamespace(
            inst_id="m4", RUN_TYPE_MAP={"All": None},
            query_return_run_list=Mock(return_value=["2026-10-03 10:00:00"]),
        )
        window.runTypeCombo = QComboBox()
        window.runTypeCombo.addItem("All")
        window.get_load_range = lambda: ("2026-10-01", "2026-10-31")
        with patch.object(panel, "show_thermocouples") as render:
            MainWindow.set_runlist(window)
            render.assert_called_once()
        self.assertEqual(window.run_cb.currentText(), "2026-10-03 10:00:00")

    def test_switching_tabs_moves_shared_date_and_run_controls_and_restores_them(self):
        window = MainWindow.__new__(MainWindow)
        QMainWindow.__init__(window)
        self.addCleanup(window.deleteLater)
        central = QWidget(window)
        window.setCentralWidget(central)
        window.h_main = QHBoxLayout(central)
        window.left_container = QWidget()
        left = QVBoxLayout(window.left_container)
        window.tabs = QTabWidget()
        left.addWidget(window.tabs)
        window.processing_pane = QWidget()
        window.run_selection_layout = QVBoxLayout(window.processing_pane)
        window.run_date_group = QWidget()
        window.run_selector_widget = QWidget()
        window.run_cb = QComboBox(window.run_selector_widget)
        window.run_cb.addItem("2026-10-02 11:30:00")
        for widget in (window.run_date_group, window.run_selector_widget):
            window.run_selection_layout.addWidget(widget)
        window.figures_tab = logos_figures.FiguresWidget(
            SimpleNamespace(inst_id="m4"), window, parent=window,
        )
        window.timeseries_tab = window.tanks_tab = window.logos_ai_tab = None
        window.tabs.addTab(window.processing_pane, "Processing")
        window.tabs.addTab(window.figures_tab, "Figures")
        window.right_placeholder, window.right_spacer = QWidget(), QWidget()
        for widget in (window.left_container, window.right_placeholder, window.right_spacer, window.figures_tab.plot_widget):
            window.h_main.addWidget(widget)
        window.tabs.currentChanged.connect(window._on_tab_changed)
        window.tabs.setCurrentIndex(1)
        self.assertEqual(window.figures_tab.controls.indexOf(window.run_date_group), 0)
        self.assertEqual(window.figures_tab.controls.indexOf(window.run_selector_widget), 1)
        self.assertFalse(window.figures_tab.plot_widget.isHidden())
        self.assertTrue(window.right_placeholder.isHidden())
        window.tabs.setCurrentIndex(0)
        self.assertEqual(window.run_selection_layout.indexOf(window.run_date_group), 0)
        self.assertEqual(window.run_selection_layout.indexOf(window.run_selector_widget), 1)
        self.assertEqual(window.run_cb.currentText(), "2026-10-02 11:30:00")
        self.assertTrue(window.figures_tab.plot_widget.isHidden())
        self.assertFalse(window.right_placeholder.isHidden())


if __name__ == "__main__":
    unittest.main()
