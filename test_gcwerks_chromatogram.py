import struct
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


LOGOSDATA_DIR = Path(__file__).resolve().parent / "logosdata"
if str(LOGOSDATA_DIR) not in sys.path:
    sys.path.insert(0, str(LOGOSDATA_DIR))

from gcwerks_chromatogram import (  # noqa: E402
    export_gcwerks_chromatogram,
    find_gcwerks_chromatogram,
    gcwerks_channel_number,
    gcwerks_ms_quantitation_mass,
    read_gcwerks_chromatogram,
    read_gcwerks_ms_chromatogram,
)


def _write_chromatogram(path, timestamp, payload, sample_rate=4.0):
    header = struct.pack("<IIff4I", 1, timestamp, sample_rate, 0.0, 0, 0, 0, 0)
    path.write_bytes(header + payload)


def _compressed_vector(values):
    payload = [struct.pack("<i", values[0])]
    payload.extend(
        struct.pack("<i", current - previous)
        for previous, current in zip(values, values[1:])
    )
    payload.append(struct.pack("<i", 2_000_000_300))
    return b"".join(payload)


def _write_ms_chromatogram(path, timestamp, traces):
    master_count = len(traces[0][1])
    header = struct.pack(
        "<12I", 1, timestamp, 0, 1, 4, 0x447A0000, 0, 0, 0, 0,
        len(traces), master_count,
    )
    payload = []
    for mass, times_or_indexes, signals in traces:
        payload.append(struct.pack("<fI", mass, len(signals)))
        payload.append(_compressed_vector(times_or_indexes))
        payload.append(_compressed_vector(signals))
    path.write_bytes(header + b"".join(payload))


class GCWerksChromatogramTests(unittest.TestCase):
    def test_decodes_all_delta_storage_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "260417.1237.2"
            payload = b"".join(
                [
                    struct.pack("<i", 1000),
                    struct.pack("<i", 2_000_000_100),
                    struct.pack("<b", 1),
                    struct.pack("<b", -2),
                    struct.pack("<b", 126),
                    struct.pack("<h", 200),
                    struct.pack("<h", 32_400),
                    struct.pack("<i", -40_000),
                ]
            )
            _write_chromatogram(path, 1_776_429_420, payload)

            chrom = read_gcwerks_chromatogram(path)

            self.assertEqual(chrom.sample_rate, 4.0)
            self.assertEqual(chrom.start_time, datetime(2026, 4, 17, 12, 37, tzinfo=timezone.utc))
            self.assertEqual(chrom.signal.tolist(), [1000, 1001, 999, 1199, -38801])

    def test_decodes_current_repeat_and_end_markers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "current"
            payload = b"".join(
                [
                    struct.pack("<i", 1000),
                    struct.pack("<i", 2_000_000_100),
                    struct.pack("<b", 1),
                    struct.pack("<b", -127),
                    struct.pack("<b", 3),
                    struct.pack("<b", 126),
                    struct.pack("<h", 200),
                    struct.pack("<h", 32_450),
                    struct.pack("<b", 2),
                    struct.pack("<h", 32_400),
                    struct.pack("<i", 40_000),
                    struct.pack("<i", 2_000_000_250),
                    struct.pack("<b", 4),
                    struct.pack("<i", 2_000_000_300),
                ]
            )
            _write_chromatogram(path, 1_776_429_420, payload)

            chrom = read_gcwerks_chromatogram(path)

            self.assertEqual(
                chrom.signal.tolist(),
                [
                    1000,
                    1001, 1001, 1001, 1001,
                    1201, 1201, 1201,
                    41201, 41201, 41201, 41201, 41201,
                ],
            )
            self.assertEqual(chrom.elapsed_seconds[-1], 3.0)

    def test_accepts_current_end_marker_in_each_storage_state(self):
        payloads = {
            "long": struct.pack("<ii", 1000, 2_000_000_300),
            "short": b"".join(
                [
                    struct.pack("<i", 1000),
                    struct.pack("<i", 2_000_000_200),
                    struct.pack("<h", 32_500),
                ]
            ),
            "byte": b"".join(
                [
                    struct.pack("<i", 1000),
                    struct.pack("<i", 2_000_000_100),
                    struct.pack("<b", -126),
                ]
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for state, payload in payloads.items():
                with self.subTest(state=state):
                    path = Path(tmpdir) / state
                    _write_chromatogram(path, 1_776_429_420, payload)
                    chrom = read_gcwerks_chromatogram(path)
                    self.assertEqual(chrom.signal.tolist(), [1000])

    def test_rejects_non_gcwerks_sample_rate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad"
            _write_chromatogram(path, 1_776_429_420, struct.pack("<i", 1), sample_rate=0.0)
            with self.assertRaisesRegex(ValueError, "sample rate"):
                read_gcwerks_chromatogram(path)

    def test_official_exporter_supplies_corrected_display_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "260417.1237.2"
            _write_chromatogram(path, 1_776_429_420, struct.pack("<i", 1))
            executable = Path(tmpdir) / "chromatogram_export"
            executable.touch()
            completed = type("Completed", (), {
                "returncode": 0,
                "stdout": "0.00  1000\n0.25  1002\n0.50  1001\n",
                "stderr": "GCWerks diagnostic output\n",
            })()

            with patch("gcwerks_chromatogram.subprocess.run", return_value=completed) as run:
                chrom = export_gcwerks_chromatogram(
                    tmpdir, 2, path, executable=executable
                )

            self.assertEqual(chrom.signal.tolist(), [1000, 1002, 1001])
            self.assertEqual(chrom.elapsed_seconds.tolist(), [0.0, 0.25, 0.5])
            run.assert_called_once_with(
                [str(executable), tmpdir, "-nchannel", "2", path.name],
                capture_output=True,
                text=True,
                check=False,
            )

    def test_channel_mappings(self):
        self.assertEqual(gcwerks_channel_number("cats", "f", "spo"), 2)
        self.assertEqual(gcwerks_channel_number("cats", "q", "spo"), 0)
        self.assertEqual(gcwerks_channel_number("cats", "a", "smo"), 0)
        self.assertEqual(gcwerks_channel_number("ie3", "c", "smo"), 2)
        self.assertEqual(gcwerks_channel_number("fe3", "b"), 1)
        self.assertEqual(gcwerks_channel_number("m4", None), 0)

    def test_finder_uses_header_time_to_break_minute_tie(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir) / "26" / "chromatograms" / "channel2"
            directory.mkdir(parents=True)
            early = directory / "260417.1237.1"
            late = directory / "260417.1237.2"
            payload = struct.pack("<i", 10)
            _write_chromatogram(early, 1_776_429_420, payload)
            _write_chromatogram(late, 1_776_429_450, payload)

            found = find_gcwerks_chromatogram(
                tmpdir,
                datetime(2026, 4, 17, 12, 37, 28, tzinfo=timezone.utc),
                2,
            )

            self.assertEqual(found, late)

    def test_decodes_ms_tic_and_indexed_ion_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "260804.0413.zero_air"
            _write_ms_chromatogram(
                path,
                1_775_449_580,
                [
                    (0.0, [245_250, 245_500, 245_750], [10, 12, 11]),
                    (127.0, [0, 2], [1000, 1200]),
                ],
            )

            chrom = read_gcwerks_ms_chromatogram(path)

            self.assertEqual(chrom.masses, (0.0, 127.0))
            self.assertEqual(chrom.traces[0].elapsed_seconds.tolist(), [245.25, 245.5, 245.75])
            self.assertEqual(chrom.traces[0].signal.tolist(), [10, 12, 11])
            ion = chrom.trace_for_mass(127)
            self.assertEqual(ion.elapsed_seconds.tolist(), [245.25, 245.75])
            self.assertEqual(ion.signal.tolist(), [1000, 1200])

    def test_rejects_ms_trace_with_invalid_master_time_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad-ms"
            _write_ms_chromatogram(
                path,
                1_775_449_580,
                [
                    (0.0, [1000, 1250], [10, 11]),
                    (83.0, [0, 2], [20, 21]),
                ],
            )

            with self.assertRaisesRegex(ValueError, "invalid time indexes"):
                read_gcwerks_ms_chromatogram(path)

    def test_ms_scan_combines_fractional_bins_into_nominal_ion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scan"
            _write_ms_chromatogram(
                path,
                1_775_449_580,
                [
                    (0.0, [1000, 1250, 1500], [100, 110, 120]),
                    (31.9, [0, 2], [10, 20]),
                    (32.0, [0, 1], [100, 200]),
                    (32.1, [1, 2], [1000, 2000]),
                ],
            )

            chrom = read_gcwerks_ms_chromatogram(path)
            ion = chrom.trace_for_mass(32)

            self.assertTrue(chrom.is_profile_scan)
            self.assertEqual(chrom.display_masses, (0.0, 32.0))
            self.assertEqual(ion.elapsed_seconds.tolist(), [1.0, 1.25, 1.5])
            self.assertEqual(ion.signal.tolist(), [110, 1200, 2020])

    def test_reads_ms_quantitation_mass_with_name_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            peakid = (
                Path(tmpdir) / "integrator" / "channel0" / "peakid" / "initial"
            )
            peakid.parent.mkdir(parents=True)
            peakid.write_text(
                "HFC-134a 256.8 10.0 83.00000 0.0 0.0\n"
                "CFC-11b 599.6 6.0 103.00000 0.0 0.0\n"
            )

            self.assertEqual(gcwerks_ms_quantitation_mass(tmpdir, "HFC134a"), 83.0)
            self.assertEqual(gcwerks_ms_quantitation_mass(tmpdir, "CFC11b"), 103.0)
            self.assertIsNone(gcwerks_ms_quantitation_mass(tmpdir, "unknown"))


if __name__ == "__main__":
    unittest.main()
