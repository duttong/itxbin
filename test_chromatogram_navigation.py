import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-logos-navigation-tests")

LOGOSDATA_DIR = Path(__file__).resolve().parent / "logosdata"
if str(LOGOSDATA_DIR) not in sys.path:
    sys.path.insert(0, str(LOGOSDATA_DIR))

from PyQt5.QtWidgets import QApplication  # noqa: E402
from gcwerks_chromatogram import (  # noqa: E402
    GCWerksChromatogram,
    GCWerksMSChromatogram,
    GCWerksMSTrace,
    GCWerksPeakWindow,
)
from logos_data import ChromatogramWindow, MSChromatogramWindow  # noqa: E402


class ChromatogramNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def _navigator(payloads):
        def navigate(row_idx, direction):
            position = next(
                index
                for index, payload in enumerate(payloads)
                if payload["row_idx"] == row_idx
            )
            position += -1 if direction < 0 else 1
            return payloads[position] if 0 <= position < len(payloads) else None

        return navigate

    @staticmethod
    def _peak_window():
        return GCWerksPeakWindow("test", 2.0, 1.0, 83.0, Path("peakid"))

    def test_scalar_arrows_replace_or_accumulate_traces(self):
        payloads = []
        for index in range(3):
            chrom = GCWerksChromatogram(
                path=Path(f"chrom-{index}"),
                version=1,
                start_time=datetime(2026, 8, 1, index, tzinfo=timezone.utc),
                sample_rate=1.0,
                inject_time_offset=0.0,
                signal=np.asarray([0, index + 1, 10 + index, 1, 0]),
                elapsed_seconds=np.arange(5, dtype=float),
            )
            payloads.append({
                "chromatogram": chrom,
                "site": "spo",
                "channel_number": 0,
                "point_info_html": f"<b>Sample ID:</b> S{index}",
                "peak_window": self._peak_window(),
                "row_idx": index,
            })

        first = payloads[0]
        window = ChromatogramWindow(
            first["chromatogram"],
            first["site"],
            first["channel_number"],
            point_info_html=first["point_info_html"],
            peak_window=first["peak_window"],
            row_idx=0,
            navigator=self._navigator(payloads),
        )
        self.addCleanup(window.close)

        custom_x = (0.025, 0.055)
        custom_y = (-2.0, 8.0)
        window.axes.set_xlim(*custom_x)
        window.axes.set_ylim(*custom_y)
        window._navigate(1)
        self.assertEqual(len(window.axes.lines), 1)
        self.assertIn("chrom-1", window.windowTitle())
        self.assertTrue(any("S1" in text.get_text() for text in window.axes.texts))
        self.assertEqual(window.axes.get_xlim(), custom_x)
        self.assertEqual(window.axes.get_ylim(), custom_y)

        window._navigate(-1)
        window.overlay_cb.setChecked(True)
        window._navigate(1)
        window._navigate(1)
        self.assertEqual(len(window.axes.lines), 3)
        self.assertFalse(window.axes.texts)
        self.assertEqual(
            [text.get_text() for text in window.axes.get_legend().get_texts()],
            ["chrom-0", "chrom-1", "chrom-2"],
        )
        self.assertEqual(window.axes.get_xlim(), custom_x)
        self.assertEqual(window.axes.get_ylim(), custom_y)

        window.overlay_cb.setChecked(False)
        self.assertEqual(len(window.axes.lines), 1)
        self.assertTrue(any("S2" in text.get_text() for text in window.axes.texts))

    def test_ms_overlay_keeps_selected_ion_and_uses_filename_legend(self):
        payloads = []
        for index in range(2):
            elapsed = np.arange(5, dtype=float)
            chrom = GCWerksMSChromatogram(
                path=Path(f"ms-{index}"),
                version=1,
                start_time=datetime(2026, 8, 1, index, tzinfo=timezone.utc),
                traces=(
                    GCWerksMSTrace(0.0, np.asarray([1, 2, 3, 2, 1]), elapsed),
                    GCWerksMSTrace(
                        83.0,
                        np.asarray([0, index + 1, 10 + index, 1, 0]),
                        elapsed,
                    ),
                ),
            )
            payloads.append({
                "chromatogram": chrom,
                "site": "m4",
                "channel_number": 0,
                "point_info_html": f"<b>Sample ID:</b> M{index}",
                "peak_window": self._peak_window(),
                "row_idx": index,
            })

        first = payloads[0]
        window = MSChromatogramWindow(
            first["chromatogram"],
            first["site"],
            first["channel_number"],
            default_mass=83.0,
            point_info_html=first["point_info_html"],
            peak_window=first["peak_window"],
            row_idx=0,
            navigator=self._navigator(payloads),
        )
        self.addCleanup(window.close)

        custom_x = (0.02, 0.06)
        custom_y = (-5.0, 15.0)
        window.axes.set_xlim(*custom_x)
        window.axes.set_ylim(*custom_y)
        window.overlay_cb.setChecked(True)
        window._navigate(1)
        self.assertEqual(window.mass_combo.currentText(), "m/z 83")
        self.assertEqual(len(window.axes.lines), 2)
        self.assertFalse(window.axes.texts)
        self.assertEqual(
            [text.get_text() for text in window.axes.get_legend().get_texts()],
            ["ms-0", "ms-1"],
        )
        self.assertEqual(window.axes.get_xlim(), custom_x)
        self.assertEqual(window.axes.get_ylim(), custom_y)

        window.mass_combo.setCurrentIndex(0)
        self.assertEqual(window.axes.get_xlim(), custom_x)
        self.assertEqual(window.axes.get_ylim(), custom_y)


if __name__ == "__main__":
    unittest.main()
