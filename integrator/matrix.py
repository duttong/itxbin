"""
Multi-run chromatogram matrix for cross-injection analysis.

The matrix collects one channel from many injections into a 2D array
[n_injections × n_samples], enabling median/std computation and
cross-run peak alignment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from integrator.run_loader import RunLoader
from integrator.chrom import Chromatogram, Peak


class ChromMatrix:
    """
    2-D matrix of chromatograms: [n_injections × n_samples].

    All rows share the same channel index.  Each row is one injection
    (one port from one run directory).
    """

    def __init__(self, channel: int = 0):
        self.channel = channel
        # Each entry: (run_dir_name, port, Chromatogram)
        self._rows: list[tuple[str, str, Chromatogram]] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        incoming_dir: Path | str,
        channel: int = 0,
        exclude_ports: tuple[str, ...] = ('10',),
        smooth: bool = True,
        limit: int | None = None,
    ) -> ChromMatrix:
        """
        Build a matrix from every run sub-directory in *incoming_dir*.

        Parameters
        ----------
        incoming_dir : parent directory whose children are run dirs (e.g. /hats/gc/fe3/26/incoming)
        channel : which detector channel to collect (0-based)
        exclude_ports : port numbers to skip (FE3 push port = '10')
        smooth : apply SG smoothing when loading
        limit : cap the number of run directories (useful during development)
        """
        mat = cls(channel=channel)
        dirs = sorted(Path(incoming_dir).glob('*-*'))
        if limit is not None:
            dirs = dirs[:limit]

        for run_dir in dirs:
            loader = RunLoader(run_dir)
            injections = loader.load(exclude_ports=exclude_ports, smooth=smooth)
            for port, chroms in sorted(injections.items()):
                if channel < len(chroms):
                    mat._rows.append((run_dir.name, port, chroms[channel]))

        return mat

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        """'run_dir/port' label for every row."""
        return [f'{r}/{p}' for r, p, _ in self._rows]

    def array(self, truncate: bool = True) -> np.ndarray:
        """
        2-D numpy array [n_injections × n_samples].

        When *truncate* is True all rows are clipped to the shortest
        chromatogram length so the array is rectangular.
        """
        signals = [c.signal for _, _, c in self._rows]
        if truncate:
            min_len = min(len(s) for s in signals)
            signals = [s[:min_len] for s in signals]
        return np.array(signals, dtype=float)

    def corrected_array(self, truncate: bool = True) -> np.ndarray:
        """Baseline-subtracted 2-D array."""
        arrays = [c.corrected for _, _, c in self._rows]
        if truncate:
            min_len = min(len(a) for a in arrays)
            arrays = [a[:min_len] for a in arrays]
        return np.array(arrays, dtype=float)

    def median(self) -> np.ndarray:
        """Element-wise median across all injections."""
        return np.median(self.array(), axis=0)

    def std(self) -> np.ndarray:
        """Element-wise standard deviation across all injections."""
        return np.std(self.array(), axis=0)

    def to_dataframe(self) -> pd.DataFrame:
        """One row per injection; columns are sample indices."""
        return pd.DataFrame(self.array(), index=self.labels)

    # ------------------------------------------------------------------
    # Cross-run peak statistics
    # ------------------------------------------------------------------

    def peak_table(
        self,
        min_width_sec: float = 2.0,
        distance_sec: float = 15.0,
        prominence_factor: float = 5.0,
    ) -> pd.DataFrame:
        """
        Detect peaks in every injection and return a tidy DataFrame.

        Columns: label, port, center_time, height, area.
        """
        records = []
        for run_name, port, chrom in self._rows:
            peaks = chrom.detect_peaks(
                min_width_sec=min_width_sec,
                distance_sec=distance_sec,
                prominence_factor=prominence_factor,
            )
            for pk in peaks:
                records.append({
                    'run': run_name,
                    'port': port,
                    'label': f'{run_name}/{port}',
                    'center_time': pk.center_time,
                    'height': pk.height,
                    'area': pk.area,
                })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return f'ChromMatrix(channel={self.channel}, n_injections={len(self)})'
