"""Read and locate GCWerks compressed chromatogram files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import struct
import subprocess

import numpy as np


_HEADER = struct.Struct("<IIff4I")
_MS_HEADER = struct.Struct("<12I")
_MS_TRACE_HEADER = struct.Struct("<fI")
_VALUE_FORMATS = {
    "long": struct.Struct("<i"),
    "short": struct.Struct("<h"),
    "byte": struct.Struct("<b"),
}
_STATE_FLAGS = {
    "long": {2_000_000_100: "byte", 2_000_000_200: "short"},
    "short": {32_100: "byte", 32_400: "long"},
    "byte": {126: "short", 127: "long"},
}
# Current GCWerks compression collapses a run of zero deltas into a
# state-specific marker plus a signed one-byte repeat count.  It also writes an
# explicit end marker for the active storage state.  Older files simply end at
# EOF, which remains supported below.
_REPEAT_FLAGS = {
    "long": 2_000_000_250,
    "short": 32_450,
    "byte": -127,
}
_END_FLAGS = {
    "long": 2_000_000_300,
    "short": 32_500,
    "byte": -126,
}


@dataclass(frozen=True)
class GCWerksChromatogram:
    path: Path
    version: int
    start_time: datetime
    sample_rate: float
    inject_time_offset: float
    signal: np.ndarray
    elapsed_seconds: np.ndarray | None = None


@dataclass(frozen=True)
class GCWerksPeakWindow:
    """An analyte identification window from a GCWerks peakid file."""

    analyte: str
    center_seconds: float
    width_seconds: float
    mass: float | None
    path: Path


@dataclass(frozen=True)
class GCWerksPeakIntegration:
    """GCWerks-selected integration baseline for one analyte and run."""

    analyte: str
    start_time: float
    start_level: float
    end_time: float
    end_level: float
    path: Path


@dataclass(frozen=True)
class GCWerksMSTrace:
    """One ion trace from a GCWerks mass-spectrometer chromatogram."""

    mass: float
    signal: np.ndarray
    elapsed_seconds: np.ndarray


@dataclass(frozen=True)
class GCWerksMSChromatogram:
    """Native contents of a GCWerks MSD chromatogram file."""

    path: Path
    version: int
    start_time: datetime
    traces: tuple[GCWerksMSTrace, ...]

    @property
    def masses(self) -> tuple[float, ...]:
        return tuple(trace.mass for trace in self.traces)

    @property
    def is_profile_scan(self) -> bool:
        """Whether the file stores fractional mass bins from a scan run."""
        return any(
            abs(trace.mass - round(trace.mass)) > 0.02
            for trace in self.traces
            if trace.mass > 0
        )

    @property
    def display_masses(self) -> tuple[float, ...]:
        """Mass choices suitable for an ion selector."""
        if not self.is_profile_scan:
            return self.masses
        nominal = sorted(
            {int(np.floor(trace.mass + 0.5)) for trace in self.traces if trace.mass > 0}
        )
        return (0.0, *(float(mass) for mass in nominal))

    def trace_for_mass(self, mass: float) -> GCWerksMSTrace:
        """Return a stored SIM trace or a nominal extracted-ion scan trace."""
        if not self.traces:
            raise ValueError(f"MS chromatogram contains no ion traces: {self.path}")
        mass = float(mass)
        if not self.is_profile_scan or abs(mass) < 1e-6:
            return min(self.traces, key=lambda trace: abs(trace.mass - mass))

        # GCWerks' scan exporter treats an integer -mass request as a nominal
        # ion and sums all stored 0.1-u bins in [mass - 0.5, mass + 0.5).
        selected = [
            trace
            for trace in self.traces
            if mass - 0.5 <= trace.mass < mass + 0.5
        ]
        if not selected:
            return min(self.traces, key=lambda trace: abs(trace.mass - mass))
        by_millisecond = {}
        for trace in selected:
            for elapsed, signal in zip(trace.elapsed_seconds, trace.signal):
                key = int(round(float(elapsed) * 1000.0))
                by_millisecond[key] = by_millisecond.get(key, 0) + int(signal)
        milliseconds = np.asarray(sorted(by_millisecond), dtype=np.int64)
        return GCWerksMSTrace(
            mass=mass,
            signal=np.asarray(
                [by_millisecond[key] for key in milliseconds], dtype=np.int64
            ),
            elapsed_seconds=milliseconds.astype(float) / 1000.0,
        )


def _decode_compressed_values(
    compressed: bytes,
    position: int,
    path: Path,
    *,
    byte_offset: int = 0,
) -> tuple[np.ndarray, int]:
    """Decode one GCWerks delta-compressed integer vector."""
    if position + _VALUE_FORMATS["long"].size > len(compressed):
        raise ValueError(f"GCWerks chromatogram contains no signal data: {path}")

    level = _VALUE_FORMATS["long"].unpack_from(compressed, position)[0]
    position += _VALUE_FORMATS["long"].size
    values = [level]
    state = "long"

    while position < len(compressed):
        value_format = _VALUE_FORMATS[state]
        if position + value_format.size > len(compressed):
            raise ValueError(
                f"Truncated {state} value at byte {byte_offset + position}: {path}"
            )
        delta = value_format.unpack_from(compressed, position)[0]
        position += value_format.size

        if delta == _END_FLAGS[state]:
            break

        if delta == _REPEAT_FLAGS[state]:
            if position >= len(compressed):
                raise ValueError(
                    f"Missing repeat count at byte {byte_offset + position}: {path}"
                )
            repeat_count = _VALUE_FORMATS["byte"].unpack_from(
                compressed, position
            )[0]
            position += _VALUE_FORMATS["byte"].size
            if repeat_count <= 0:
                raise ValueError(
                    f"Invalid repeat count {repeat_count} at byte "
                    f"{byte_offset + position - 1}: {path}"
                )
            values.extend([level] * repeat_count)
            continue

        next_state = _STATE_FLAGS[state].get(delta)
        if next_state is not None:
            state = next_state
            continue

        level += delta
        values.append(level)

    return np.asarray(values, dtype=np.int64), position


def read_gcwerks_header(path: str | Path) -> tuple[int, datetime, float, float]:
    """Return version, UTC start time, sample rate, and injection offset."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(_HEADER.size)
    if len(raw) != _HEADER.size:
        raise ValueError(f"Incomplete GCWerks chromatogram header: {path}")

    version, timestamp, sample_rate, inject_offset, *_reserved = _HEADER.unpack(raw)
    if version != 1:
        raise ValueError(f"Unsupported GCWerks chromatogram version {version}: {path}")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"Invalid GCWerks sample rate {sample_rate:g}: {path}")
    try:
        start_time = datetime.fromtimestamp(timestamp, timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid GCWerks timestamp {timestamp}: {path}") from exc
    return version, start_time, float(sample_rate), float(inject_offset)


def read_gcwerks_chromatogram(path: str | Path) -> GCWerksChromatogram:
    """Decode a version-1 GCWerks delta-compressed chromatogram.

    In addition to the original byte/short/long delta representation, current
    files use a state-specific repeat marker followed by a one-byte count to
    represent runs of unchanged detector levels.
    """
    path = Path(path)
    version, start_time, sample_rate, inject_offset = read_gcwerks_header(path)
    with path.open("rb") as stream:
        stream.seek(_HEADER.size)
        compressed = stream.read()

    values, _position = _decode_compressed_values(
        compressed, 0, path, byte_offset=_HEADER.size
    )

    return GCWerksChromatogram(
        path=path,
        version=version,
        start_time=start_time,
        sample_rate=sample_rate,
        inject_time_offset=inject_offset,
        signal=values,
        elapsed_seconds=np.arange(len(values), dtype=float) / sample_rate,
    )


def read_gcwerks_ms_header(path: str | Path) -> tuple[int, datetime, int, int]:
    """Return version, UTC start time, trace count, and master point count."""
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(_MS_HEADER.size)
    if len(raw) != _MS_HEADER.size:
        raise ValueError(f"Incomplete GCWerks MSD chromatogram header: {path}")

    fields = _MS_HEADER.unpack(raw)
    version, timestamp = fields[:2]
    trace_count, master_point_count = fields[-2:]
    if version != 1:
        raise ValueError(f"Unsupported GCWerks MSD version {version}: {path}")
    if not 1 <= trace_count <= 10_000:
        raise ValueError(f"Invalid GCWerks MSD trace count {trace_count}: {path}")
    if not 1 <= master_point_count <= 10_000_000:
        raise ValueError(
            f"Invalid GCWerks MSD master point count {master_point_count}: {path}"
        )
    try:
        start_time = datetime.fromtimestamp(timestamp, timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid GCWerks MSD timestamp {timestamp}: {path}") from exc
    return version, start_time, trace_count, master_point_count


def read_gcwerks_ms_chromatogram(path: str | Path) -> GCWerksMSChromatogram:
    """Decode a GCWerks mass-spectrometer (MSD) chromatogram natively.

    Each stored ion has two compressed integer vectors.  The first trace is
    mass zero (TIC), whose first vector contains absolute acquisition times in
    milliseconds.  For every other ion the first vector contains indexes into
    that master time vector; the second vector always contains signal counts.
    """
    path = Path(path)
    version, start_time, trace_count, master_point_count = read_gcwerks_ms_header(path)
    data = path.read_bytes()
    position = _MS_HEADER.size
    raw_traces = []

    for trace_index in range(trace_count):
        if position + _MS_TRACE_HEADER.size > len(data):
            raise ValueError(
                f"Truncated GCWerks MSD trace header {trace_index + 1}: {path}"
            )
        mass, point_count = _MS_TRACE_HEADER.unpack_from(data, position)
        position += _MS_TRACE_HEADER.size
        if not np.isfinite(mass) or mass < 0 or point_count > 10_000_000:
            raise ValueError(
                f"Invalid GCWerks MSD trace header at trace {trace_index + 1}: {path}"
            )

        time_values, position = _decode_compressed_values(data, position, path)
        signal, position = _decode_compressed_values(data, position, path)
        if len(time_values) != point_count or len(signal) != point_count:
            raise ValueError(
                f"GCWerks MSD trace {mass:g} declares {point_count} points but "
                f"contains {len(time_values)} times and {len(signal)} signals: {path}"
            )
        raw_traces.append((float(mass), time_values, signal))

    if position != len(data):
        raise ValueError(
            f"GCWerks MSD chromatogram has {len(data) - position} trailing bytes: {path}"
        )
    if not raw_traces or abs(raw_traces[0][0]) > 1e-6:
        raise ValueError(f"GCWerks MSD chromatogram has no mass-zero master trace: {path}")

    master_times = raw_traces[0][1]
    if len(master_times) != master_point_count:
        raise ValueError(
            f"GCWerks MSD header declares {master_point_count} master points but "
            f"contains {len(master_times)}: {path}"
        )

    traces = []
    for trace_index, (mass, time_values, signal) in enumerate(raw_traces):
        if trace_index == 0:
            elapsed_seconds = time_values.astype(float) / 1000.0
        else:
            if np.any(time_values < 0) or np.any(time_values >= len(master_times)):
                raise ValueError(f"GCWerks MSD trace {mass:g} has invalid time indexes: {path}")
            elapsed_seconds = master_times[time_values].astype(float) / 1000.0
        traces.append(
            GCWerksMSTrace(
                mass=mass,
                signal=signal,
                elapsed_seconds=elapsed_seconds,
            )
        )

    return GCWerksMSChromatogram(
        path=path,
        version=version,
        start_time=start_time,
        traces=tuple(traces),
    )


def gcwerks_ms_quantitation_masses(
    gc_dir: str | Path,
    method: str = "initial",
) -> dict[str, float]:
    """Read analyte-to-quantitation-mass assignments from GCWerks peakid."""
    path = Path(gc_dir) / "integrator" / "channel0" / "peakid" / method
    masses = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except OSError:
        return masses
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            masses[fields[0]] = float(fields[3])
        except ValueError:
            continue
    return masses


def gcwerks_ms_quantitation_mass(
    gc_dir: str | Path,
    analyte: str,
    method: str = "initial",
) -> float | None:
    """Return an analyte's configured quantitation mass, tolerating punctuation."""
    normalize = lambda value: "".join(ch for ch in value.lower() if ch.isalnum())
    wanted = normalize(str(analyte))
    for name, mass in gcwerks_ms_quantitation_masses(gc_dir, method).items():
        if normalize(name) == wanted:
            return mass
    return None


def _as_utc_datetime(value: object) -> datetime:
    timestamp = value
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if not isinstance(timestamp, datetime):
        timestamp = datetime.fromisoformat(str(timestamp))
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def find_gcwerks_peakid_file(
    gc_dir: str | Path,
    channel_number: int,
    analysis_time: object,
) -> Path | None:
    """Find the dated peakid file applicable to one chromatogram."""
    directory = (
        Path(gc_dir)
        / "integrator"
        / f"channel{int(channel_number)}"
        / "peakid"
    )
    if not directory.is_dir():
        return None

    target = _as_utc_datetime(analysis_time)
    dated = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = re.fullmatch(r"(\d{6})(?:\.(\d{4}))?", path.name)
        if not match:
            continue
        date_text, time_text = match.groups()
        try:
            effective_time = datetime.strptime(
                date_text + (time_text or "0000"), "%y%m%d%H%M"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        dated.append((effective_time, path))

    applicable = [item for item in dated if item[0] <= target]
    if applicable:
        return max(applicable, key=lambda item: (item[0], item[1].name))[1]

    initial = directory / "initial"
    if initial.is_file():
        return initial
    if dated:
        return min(dated, key=lambda item: (item[0], item[1].name))[1]
    return None


def read_gcwerks_peak_windows(path: str | Path) -> tuple[GCWerksPeakWindow, ...]:
    """Read analyte center, full window width, and optional quantitation mass."""
    path = Path(path)
    windows = []
    for raw_line in path.read_text(encoding="ascii").splitlines():
        fields = raw_line.split("#", 1)[0].split()
        if len(fields) < 3:
            continue
        try:
            center = float(fields[1])
            width = float(fields[2])
            mass = float(fields[3]) if len(fields) >= 4 else None
        except ValueError:
            continue
        if not np.isfinite(center) or not np.isfinite(width) or width <= 0:
            continue
        if mass is not None and not np.isfinite(mass):
            mass = None
        windows.append(
            GCWerksPeakWindow(
                analyte=fields[0],
                center_seconds=center,
                width_seconds=width,
                mass=mass,
                path=path,
            )
        )
    return tuple(windows)


def _analyte_name_parts(value: object) -> tuple[str, str | None]:
    """Return a normalized analyte name and optional GC channel suffix."""
    name = str(value).strip().lower()
    channel = None
    display_match = re.search(r"\s*\((q|a|b|c|d|f|cc)\)\s*$", name)
    if display_match:
        channel = display_match.group(1)
        name = name[:display_match.start()]

    peakid_match = re.search(r"_(q|a|b|c|d|f|cc)$", name)
    if peakid_match:
        channel = peakid_match.group(1)
        name = name[:peakid_match.start()]

    base = "".join(character for character in name if character.isalnum())
    return base, channel


def _analyte_name_aliases(value: object) -> set[str]:
    """Return aliases for GUI and peakid channel naming conventions."""
    base, channel = _analyte_name_parts(value)
    aliases = {base}
    if channel:
        aliases.add(base + channel)
    return aliases


def _peak_result_analyte_prefix(
    field_names: set[str], analyte: str
) -> str | None:
    """Match a display analyte to its GCWerks peak-result field prefix."""
    wanted_base, wanted_channel = _analyte_name_parts(analyte)
    candidates = []
    suffix = "_start_time"
    for field_name in field_names:
        if not field_name.endswith(suffix):
            continue
        prefix = field_name[:-len(suffix)]
        base, channel = _analyte_name_parts(prefix)
        normalized_prefix = "".join(
            character for character in prefix.lower() if character.isalnum()
        )
        attached_channel_match = (
            wanted_channel is not None
            and normalized_prefix == wanted_base + wanted_channel
        )
        if wanted_channel is not None and (
            channel == wanted_channel or attached_channel_match
        ):
            priority = 0
        elif base == wanted_base and channel is None:
            priority = 1
        elif base == wanted_base and wanted_channel is None:
            priority = 2
        else:
            continue
        candidates.append((priority, prefix))
    if not candidates:
        return None
    return min(candidates)[1]


def _gcwerks_peak_result_layout(path: Path):
    """Return physical fields and row length from a GCWerks data key."""
    scalar_widths = {
        "double": 8,
        "float": 4,
        "long": 4,
        "short": 2,
        "char": 1,
        "date": 4,
        "unixdate": 4,
        "lat": 4,
        "lon": 4,
    }
    fields = {}
    offset = 0
    for raw_line in path.read_text(encoding="ascii").splitlines():
        tokens = raw_line.split("#", 1)[0].split()
        if len(tokens) < 2:
            continue
        name, field_type = tokens[:2]
        string_match = re.fullmatch(r"string(\d+)", field_type)
        if string_match:
            width = int(string_match.group(1))
        else:
            try:
                width = scalar_widths[field_type]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported GCWerks peak-result field type {field_type!r}: {path}"
                ) from exc
        fields[name] = (offset, field_type, width)
        offset += width
    if not fields or offset <= 0:
        raise ValueError(f"Empty GCWerks peak-result schema: {path}")
    return fields, offset


def gcwerks_peak_integration(
    gc_dir: str | Path,
    chromatogram_path: str | Path,
    analysis_time: object,
    analyte: str,
) -> GCWerksPeakIntegration | None:
    """Read one analyte's stored GCWerks integration start and end points.

    Peak-result times are seconds from the start of the chromatogram.  Missing
    tables, analytes, rows, and null integration results return ``None``.
    """
    chromatogram_path = Path(chromatogram_path)
    name_parts = chromatogram_path.name.split(".", 2)
    if len(name_parts) != 3:
        return None
    extension = name_parts[2]

    target = _as_utc_datetime(analysis_time)
    directory = Path(gc_dir) / "results" / "peaks" / target.strftime("%y")
    table_path = directory / f"peaks.{target:%y%m}"
    schema_path = directory / ".peaks"
    if not table_path.is_file() or not schema_path.is_file():
        return None

    fields, row_length = _gcwerks_peak_result_layout(schema_path)
    prefix = _peak_result_analyte_prefix(set(fields), analyte)
    if prefix is None:
        return None
    required_names = (
        "time",
        "extension",
        f"{prefix}_start_time",
        f"{prefix}_start_level",
        f"{prefix}_end_time",
        f"{prefix}_end_level",
    )
    if any(name not in fields for name in required_names):
        return None

    data = table_path.read_bytes()
    if len(data) % row_length:
        raise ValueError(
            f"GCWerks peak-result size is not a whole number of rows: {table_path}"
        )

    time_offset, time_type, _time_width = fields["time"]
    extension_offset, extension_type, extension_width = fields["extension"]
    if time_type not in {"long", "unixdate"} or not extension_type.startswith("string"):
        raise ValueError(f"Unexpected GCWerks peak-result key fields: {schema_path}")

    target_minute = int(target.timestamp()) // 60
    for row_offset in range(0, len(data), row_length):
        row_time = struct.unpack_from("<i", data, row_offset + time_offset)[0]
        if row_time // 60 != target_minute:
            continue
        raw_extension = data[
            row_offset + extension_offset:
            row_offset + extension_offset + extension_width
        ]
        stored_extension = raw_extension.split(b"\0", 1)[0].decode(
            "ascii", errors="replace"
        )
        extension_matches = stored_extension == extension
        if (
            not extension_matches
            and len(stored_extension) == extension_width
            and extension.startswith(stored_extension)
        ):
            extension_matches = True
        if not extension_matches:
            continue

        values = []
        for name in required_names[2:]:
            field_offset, field_type, _field_width = fields[name]
            if field_type != "float":
                raise ValueError(
                    f"Unexpected GCWerks integration field type {field_type!r}: "
                    f"{schema_path}"
                )
            values.append(struct.unpack_from("<f", data, row_offset + field_offset)[0])
        if not all(np.isfinite(value) for value in values):
            return None
        return GCWerksPeakIntegration(prefix, *values, table_path)
    return None


def gcwerks_peak_window(
    gc_dir: str | Path,
    channel_number: int,
    analysis_time: object,
    analyte: str,
) -> GCWerksPeakWindow | None:
    """Return the applicable peak identification window for an analyte."""
    path = find_gcwerks_peakid_file(gc_dir, channel_number, analysis_time)
    if path is None:
        return None
    wanted = _analyte_name_aliases(analyte)
    for window in read_gcwerks_peak_windows(path):
        if wanted.intersection(_analyte_name_aliases(window.analyte)):
            return window
    return None


def gcwerks_focus_limits(
    elapsed_seconds: np.ndarray,
    signal: np.ndarray,
    peak_window: GCWerksPeakWindow | None,
    peak_integration: GCWerksPeakIntegration | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Calculate focused x-seconds and padded y-limits for an analyte peak."""
    elapsed = np.asarray(elapsed_seconds, dtype=float)
    levels = np.asarray(signal, dtype=float)
    if elapsed.shape != levels.shape or elapsed.size == 0:
        return None

    if (
        peak_integration is not None
        and np.isfinite(peak_integration.start_time)
        and np.isfinite(peak_integration.end_time)
        and peak_integration.end_time > peak_integration.start_time
    ):
        integration_width = peak_integration.end_time - peak_integration.start_time
        x_padding = 0.1 * integration_width
        x_min = peak_integration.start_time - x_padding
        x_max = peak_integration.end_time + x_padding
    elif peak_window is not None:
        x_min = peak_window.center_seconds - peak_window.width_seconds
        x_max = peak_window.center_seconds + peak_window.width_seconds
    else:
        return None

    visible = (
        np.isfinite(elapsed)
        & np.isfinite(levels)
        & (elapsed >= x_min)
        & (elapsed <= x_max)
    )
    if not visible.any():
        return None

    y_min = float(np.min(levels[visible]))
    y_max = float(np.max(levels[visible]))
    y_range = y_max - y_min
    if y_range > 0:
        padding = 0.1 * y_range
    else:
        padding = max(abs(y_min) * 0.1, 1.0)
    return (x_min, x_max), (y_min - padding, y_max + padding)


def export_gcwerks_chromatogram(
    gc_dir: str | Path,
    channel_number: int,
    path: str | Path,
    executable: str | Path = "/hats/gc/gcwerks-3/bin/chromatogram_export",
) -> GCWerksChromatogram:
    """Load reference display data through GCWerks' official exporter.

    This is retained for format validation and is not used by the GUI viewer.
    """
    gc_dir = Path(gc_dir)
    path = Path(path)
    executable = Path(executable)
    if not executable.is_file():
        raise FileNotFoundError(f"GCWerks chromatogram exporter not found: {executable}")

    command = [
        str(executable),
        str(gc_dir),
        "-nchannel",
        str(int(channel_number)),
        path.name,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise OSError(f"Could not run GCWerks chromatogram exporter: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        detail = detail[-1] if detail else f"exit status {completed.returncode}"
        raise ValueError(f"GCWerks could not export {path.name}: {detail}")

    times = []
    values = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            elapsed = float(fields[0])
            level = int(fields[1])
        except ValueError:
            continue
        times.append(elapsed)
        values.append(level)
    if not values:
        raise ValueError(f"GCWerks returned no chromatogram data for {path.name}")

    elapsed_seconds = np.asarray(times, dtype=float)
    if np.any(np.diff(elapsed_seconds) <= 0):
        raise ValueError(f"GCWerks returned non-increasing times for {path.name}")

    version, start_time, sample_rate, inject_offset = read_gcwerks_header(path)
    return GCWerksChromatogram(
        path=path,
        version=version,
        start_time=start_time,
        sample_rate=sample_rate,
        inject_time_offset=inject_offset,
        signal=np.asarray(values, dtype=np.int64),
        elapsed_seconds=elapsed_seconds,
    )


def gcwerks_channel_number(
    instrument_id: str,
    data_channel: object,
    site: str | None = None,
) -> int:
    """Translate a database channel name to its GCWerks channel directory."""
    instrument_id = str(instrument_id or "").strip().lower()
    channel = "" if data_channel is None else str(data_channel).strip().lower()
    if channel in {"", "none", "nan", "<na>"}:
        if instrument_id == "m4":
            return 0
        raise ValueError("This data point does not identify a chromatogram channel.")

    if channel.isdigit():
        return int(channel)

    if instrument_id == "cats":
        # The current SMO/IE3 installation uses a/b/c for physical channels
        # 0/1/2. Other CATS sites use q/a/f/cc for channels 0/1/2/3.
        if str(site or "").strip().lower() == "smo":
            mapping = {"a": 0, "b": 1, "c": 2, "f": 2, "q": 0, "cc": 3}
        else:
            mapping = {"q": 0, "a": 1, "f": 2, "c": 3, "cc": 3}
    else:
        mapping = {"a": 0, "b": 1, "c": 2, "d": 3}

    try:
        return mapping[channel]
    except KeyError as exc:
        raise ValueError(
            f"No GCWerks channel mapping for {instrument_id or 'instrument'} "
            f"data channel {channel!r}."
        ) from exc


def find_gcwerks_chromatogram(
    gc_dir: str | Path,
    analysis_time: object,
    channel_number: int,
) -> Path:
    """Find the chromatogram whose filename minute matches an analysis time.

    If more than one extension exists in that minute, the timestamp stored in
    each header is used to select the closest match.
    """
    timestamp = analysis_time
    if hasattr(timestamp, "to_pydatetime"):
        timestamp = timestamp.to_pydatetime()
    if not isinstance(timestamp, datetime):
        timestamp = datetime.fromisoformat(str(timestamp))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    directory = (
        Path(gc_dir)
        / timestamp.strftime("%y")
        / "chromatograms"
        / f"channel{int(channel_number)}"
    )
    pattern = timestamp.strftime("%y%m%d.%H%M") + ".*"
    candidates = sorted(directory.glob(pattern)) if directory.is_dir() else []
    if not candidates:
        raise FileNotFoundError(
            f"No chromatogram matching {directory / pattern}"
        )
    if len(candidates) == 1:
        return candidates[0]

    target = timestamp.timestamp()
    timed_candidates = []
    for candidate in candidates:
        try:
            _version, start_time, _rate, _offset = read_gcwerks_header(candidate)
        except (OSError, ValueError):
            try:
                _version, start_time, _traces, _points = read_gcwerks_ms_header(candidate)
            except (OSError, ValueError):
                continue
        timed_candidates.append((abs(start_time.timestamp() - target), candidate))
    if timed_candidates:
        return min(timed_candidates, key=lambda item: (item[0], str(item[1])))[1]
    return candidates[0]
