import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from logosdata.gcwerks_chromatogram import (
    GCWerksPeakIntegration,
    GCWerksPeakWindow,
    gcwerks_focus_limits,
    gcwerks_peak_integration,
)


class GCWerksPeakIntegrationTests(unittest.TestCase):
    def test_focus_uses_integration_boundaries_with_ten_percent_padding(self):
        integration = GCWerksPeakIntegration(
            "SF6_q", 100.0, 10.0, 200.0, 11.0, Path("peaks.2608")
        )
        peak_window = GCWerksPeakWindow(
            "SF6_q", 500.0, 50.0, None, Path("peakid")
        )
        elapsed = np.arange(0.0, 601.0)
        signal = elapsed * 2.0

        x_limits, y_limits = gcwerks_focus_limits(
            elapsed, signal, peak_window, integration
        )

        self.assertEqual(x_limits, (90.0, 210.0))
        self.assertEqual(y_limits, (156.0, 444.0))

    def test_focus_falls_back_to_peakid_window_without_integration(self):
        peak_window = GCWerksPeakWindow(
            "SF6_q", 500.0, 50.0, None, Path("peakid")
        )
        elapsed = np.arange(0.0, 601.0)
        signal = elapsed * 2.0

        x_limits, _y_limits = gcwerks_focus_limits(
            elapsed, signal, peak_window, None
        )

        self.assertEqual(x_limits, (450.0, 550.0))

    def test_reads_selected_channel_integration_from_native_peak_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc_dir = Path(tmpdir)
            result_dir = gc_dir / "results" / "peaks" / "26"
            result_dir.mkdir(parents=True)
            (result_dir / ".peaks").write_text(
                "time unixdate unixdate3\n"
                "extension string60 string30\n"
                "SF6_q_start_time float float5.2\n"
                "SF6_q_start_level float float5.2\n"
                "SF6_q_end_time float float5.2\n"
                "SF6_q_end_level float float5.2\n"
                "SF6_a_start_time float float5.2\n"
                "SF6_a_start_level float float5.2\n"
                "SF6_a_end_time float float5.2\n"
                "SF6_a_end_level float float5.2\n",
                encoding="ascii",
            )
            analysis_time = datetime(2026, 8, 1, 4, 33, tzinfo=timezone.utc)
            extension = b"2" + (b"\0" * 59)
            row = struct.pack(
                "<i60s8f",
                int(analysis_time.timestamp()),
                extension,
                100.0,
                1000.0,
                110.0,
                1001.0,
                200.0,
                2000.0,
                220.0,
                2002.0,
            )
            (result_dir / "peaks.2608").write_bytes(row)

            integration = gcwerks_peak_integration(
                gc_dir,
                Path("260801.0433.2"),
                analysis_time,
                "SF6 (a)",
            )

            self.assertIsNotNone(integration)
            self.assertEqual(integration.analyte, "SF6_a")
            self.assertEqual(integration.start_time, 200.0)
            self.assertEqual(integration.start_level, 2000.0)
            self.assertEqual(integration.end_time, 220.0)
            self.assertEqual(integration.end_level, 2002.0)

    def test_matches_fe3_attached_channel_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc_dir = Path(tmpdir)
            result_dir = gc_dir / "results" / "peaks" / "26"
            result_dir.mkdir(parents=True)
            (result_dir / ".peaks").write_text(
                "time unixdate unixdate3\n"
                "extension string60 string30\n"
                "CFC11a_start_time float float5.2\n"
                "CFC11a_start_level float float5.2\n"
                "CFC11a_end_time float float5.2\n"
                "CFC11a_end_level float float5.2\n"
                "CFC11c_start_time float float5.2\n"
                "CFC11c_start_level float float5.2\n"
                "CFC11c_end_time float float5.2\n"
                "CFC11c_end_level float float5.2\n",
                encoding="ascii",
            )
            analysis_time = datetime(2026, 8, 4, 12, 14, tzinfo=timezone.utc)
            extension = b"1" + (b"\0" * 59)
            (result_dir / "peaks.2608").write_bytes(
                struct.pack(
                    "<i60s8f",
                    int(analysis_time.timestamp()),
                    extension,
                    60.0,
                    1000.0,
                    80.0,
                    1001.0,
                    320.0,
                    2000.0,
                    370.0,
                    2002.0,
                )
            )

            integration = gcwerks_peak_integration(
                gc_dir,
                Path("260804.1214.1"),
                analysis_time,
                "CFC11 (c)",
            )

            self.assertIsNotNone(integration)
            self.assertEqual(integration.analyte, "CFC11c")
            self.assertEqual(integration.start_time, 320.0)
            self.assertEqual(integration.end_time, 370.0)

    def test_returns_none_when_chromatogram_row_is_not_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gc_dir = Path(tmpdir)
            result_dir = gc_dir / "results" / "peaks" / "26"
            result_dir.mkdir(parents=True)
            (result_dir / ".peaks").write_text(
                "time unixdate unixdate3\n"
                "extension string60 string30\n"
                "SF6_q_start_time float float5.2\n"
                "SF6_q_start_level float float5.2\n"
                "SF6_q_end_time float float5.2\n"
                "SF6_q_end_level float float5.2\n",
                encoding="ascii",
            )
            (result_dir / "peaks.2608").write_bytes(b"")

            integration = gcwerks_peak_integration(
                gc_dir,
                Path("260801.0433.2"),
                datetime(2026, 8, 1, 4, 33, tzinfo=timezone.utc),
                "SF6 (q)",
            )

            self.assertIsNone(integration)


if __name__ == "__main__":
    unittest.main()
