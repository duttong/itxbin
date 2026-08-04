"""Read and locate GCWerks compressed chromatogram files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
