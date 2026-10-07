"""Instrument-specific figures shown beside the shared run selection controls."""
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox,
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)

if __package__:
    from .m4_thermocouples import bd_file_for_run, build_figure
else:
    from m4_thermocouples import bd_file_for_run, build_figure


class FiguresWidget(QWidget):
    def __init__(self, instrument, main_window, parent=None):
        super().__init__(parent)
        self.instrument = instrument
        self.main_window = main_window
        self.controls = QVBoxLayout(self)
        self.controls.setContentsMargins(4, 4, 4, 4)
        self.controls.setSpacing(6)
        self.plot_widget = QWidget(parent)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.message = QLabel("Select a run and a figure from the left panel.")
        self.message.setAlignment(Qt.AlignCenter)
        self.plot_layout.addWidget(self.message)
        self.canvas = None
        self.toolbar = None
        self._rendered_run_time = None
        if instrument.inst_id == "m4":
            self.thermocouples_btn = QPushButton("Thermocouples")
            self.thermocouples_btn.clicked.connect(self.show_thermocouples)
            self.controls.addWidget(self.thermocouples_btn)
        else:
            self.controls.addWidget(QLabel("No custom figures available yet."))
        self.controls.addStretch()

    def refresh_for_run(self):
        """Follow run changes once the user has chosen the thermocouple figure."""
        run_time = self.main_window.run_cb.currentText()
        if self.canvas is not None and run_time and run_time != self._rendered_run_time:
            self.show_thermocouples()

    def show_thermocouples(self):
        run_time = self.main_window.run_cb.currentText()
        if not run_time:
            QMessageBox.information(self, "Thermocouples", "Select a run first.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.thermocouples_btn.setEnabled(False)
        try:
            directory = Path(self.instrument.gc_dir) / "MassHunter/GCMS/M4 GSPC Files"
            bd = bd_file_for_run(directory, run_time)
            figure, _ = build_figure(bd, directory)
            canvas = FigureCanvasQTAgg(figure)
            toolbar = NavigationToolbar2QT(canvas, self.plot_widget)
            for old in (self.canvas, self.toolbar):
                if old is not None:
                    self.plot_layout.removeWidget(old)
                    old.deleteLater()
            self.message.hide()
            self.canvas, self.toolbar = canvas, toolbar
            self.plot_layout.addWidget(canvas, 1)
            self.plot_layout.addWidget(toolbar)
            canvas.draw_idle()
            self._rendered_run_time = run_time
        except (Exception, SystemExit) as exc:
            QMessageBox.warning(self, "Thermocouples", str(exc))
        finally:
            self.thermocouples_btn.setEnabled(True)
            QApplication.restoreOverrideCursor()
