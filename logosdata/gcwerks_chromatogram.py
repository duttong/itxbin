"""Read and locate GCWerks compressed chromatogram files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import struct
import subprocess

import numpy as np


_HEADER = struct.Struct("<IIff4I")
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

    if len(compressed) < 4:
        raise ValueError(f"GCWerks chromatogram contains no signal data: {path}")

    position = 0
    level = _VALUE_FORMATS["long"].unpack_from(compressed, position)[0]
    position += _VALUE_FORMATS["long"].size
    values = [level]
    state = "long"

    while position < len(compressed):
        value_format = _VALUE_FORMATS[state]
        if position + value_format.size > len(compressed):
            raise ValueError(
                f"Truncated {state} value at byte {_HEADER.size + position}: {path}"
            )
        delta = value_format.unpack_from(compressed, position)[0]
        position += value_format.size

        if delta == _END_FLAGS[state]:
            break

        if delta == _REPEAT_FLAGS[state]:
            if position >= len(compressed):
                raise ValueError(
                    f"Missing repeat count at byte {_HEADER.size + position}: {path}"
                )
            repeat_count = _VALUE_FORMATS["byte"].unpack_from(
                compressed, position
            )[0]
            position += _VALUE_FORMATS["byte"].size
            if repeat_count <= 0:
                raise ValueError(
                    f"Invalid repeat count {repeat_count} at byte "
                    f"{_HEADER.size + position - 1}: {path}"
                )
            values.extend([level] * repeat_count)
            continue

        next_state = _STATE_FLAGS[state].get(delta)
        if next_state is not None:
            state = next_state
            continue

        level += delta
        values.append(level)

    return GCWerksChromatogram(
        path=path,
        version=version,
        start_time=start_time,
        sample_rate=sample_rate,
        inject_time_offset=inject_offset,
        signal=np.asarray(values, dtype=np.int64),
        elapsed_seconds=np.arange(len(values), dtype=float) / sample_rate,
    )


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
            continue
        timed_candidates.append((abs(start_time.timestamp() - target), candidate))
    if timed_candidates:
        return min(timed_candidates, key=lambda item: (item[0], str(item[1])))[1]
    return candidates[0]
