"""
Load a GC run directory of ITX files into Chromatogram objects.

A run directory (e.g. 20260105-195518/) holds one .itx.Z file per SSV
port injection, named like:  2026ecd0052006.1.itx.Z
The port number is the second dot-delimited field in the filename.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# itx_import lives one level up
sys.path.insert(0, str(Path(__file__).parent.parent))
import itx_import
from integrator.chrom import Chromatogram


class RunLoader:
    """
    Loads all ITX injections from one run directory.

    Parameters
    ----------
    path : directory produced by the GC acquisition system
    """

    _ITX_PATTERNS = ('*.itx', '*.itx.gz', '*.itx.Z')

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.meta = self._load_meta()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict:
        metas = list(self.path.glob('meta_*.json'))
        if metas:
            with open(metas[0]) as fh:
                return json.load(fh)
        return {}

    def _itx_files(self) -> list[Path]:
        files: list[Path] = []
        for pat in self._ITX_PATTERNS:
            files.extend(self.path.glob(pat))
        return sorted(files)

    @staticmethod
    def port_from_filename(file: Path) -> str:
        """Return SSV port string from filename (second dot-delimited field)."""
        return file.name.split('.')[1]

    def port_name(self, port: str) -> str:
        """
        Look up a human-readable port label from the run metadata JSON.
        Falls back to 'port <N>' when metadata is absent or incomplete.
        """
        try:
            label = self.meta[2].get(port, '')
            return label if label else f'port {port}'
        except (IndexError, TypeError, KeyError):
            return f'port {port}'

    # ------------------------------------------------------------------
    # Main loader
    # ------------------------------------------------------------------

    def load(
        self,
        exclude_ports: tuple[str, ...] = ('10',),
        smooth: bool = True,
        sg_winsize: int = 61,
        sg_order: int = 4,
    ) -> dict[str, list[Chromatogram]]:
        """
        Load all injections, grouped by SSV port.

        Returns
        -------
        dict mapping port string → list of Chromatogram (one per channel).
        Ports in *exclude_ports* are skipped (FE3 push port = '10').
        When *smooth* is True, Savitzky-Golay smoothing is applied before
        returning so peak detection works on clean signals.
        """
        result: dict[str, list[Chromatogram]] = {}

        for file in self._itx_files():
            port = self.port_from_filename(file)
            if port in exclude_ports:
                continue

            itx = itx_import.ITX(file)
            if getattr(itx, 'data', None) is None or not hasattr(itx, 'chroms'):
                continue

            chroms: list[Chromatogram] = []
            for ch in range(itx.chans):
                c = Chromatogram(
                    signal=itx.chroms[ch],
                    hz=itx.datafreq,
                    name=itx.name,
                    channel=ch,
                )
                if smooth:
                    c.smooth_sg(winsize=sg_winsize, order=sg_order)
                chroms.append(c)

            result[port] = chroms

        return result
