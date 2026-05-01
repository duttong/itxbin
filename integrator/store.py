"""
Persistent HDF5-backed store for raw GC chromatogram signals.

All signals are stored raw (unsmoothed) at int32.  Smoothing, baseline
estimation, and peak integration are applied at analysis time so parameters
can be changed without rebuilding the store.

File layout
-----------
{path}.h5
  /signal      int32  [N, n_channels, MAX_SAMPLES]   raw detector counts
  /run_dir     bytes  [N]   incoming sub-directory name  (e.g. "20260105-195518")
  /port        bytes  [N]   SSV port string              (e.g. "1")
  /name        bytes  [N]   chromatogram name            (e.g. "260105.2006.1")
  /tank        bytes  [N]   tank label from meta JSON    (e.g. "CA07099")
  /hz          int32  [N]   sample frequency in Hz
  /ts          int64  [N]   Unix timestamp of injection

Usage
-----
    # First build
    store = ChromStore.build('/hats/gc/fe3/26/fe3_chroms.h5',
                             '/hats/gc/fe3/26/incoming')

    # Append new runs
    n_added = store.update('/hats/gc/fe3/26/incoming')

    # Load all port-1 injections on channel 0
    meta, signals = store.load(port='1', channel=0)
    # meta: pandas DataFrame, signals: np.ndarray [n, MAX_SAMPLES]
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import itx_import
from integrator.run_loader import RunLoader

# All chromatograms are truncated to this length.
# Observed range in FE3 2026 data: 5979–5999 samples.
MAX_SAMPLES = 5979

_STR_DTYPE = h5py.string_dtype()
_CHUNK_ROWS = 64       # injections per HDF5 chunk
_COMPRESS = dict(compression='gzip', compression_opts=4)


class ChromStore:
    """
    HDF5-backed store for raw GC chromatogram signals.

    All public methods open and close the HDF5 file themselves so the
    store object can safely be serialised or held across long sessions.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, path: Path | str, n_channels: int = 3) -> ChromStore:
        """Initialise an empty store file.  Fails if the file already exists."""
        p = Path(path)
        if p.exists():
            raise FileExistsError(f'Store already exists: {p}')
        with h5py.File(p, 'w') as f:
            f.create_dataset(
                'signal',
                shape=(0, n_channels, MAX_SAMPLES),
                maxshape=(None, n_channels, MAX_SAMPLES),
                dtype='int32',
                chunks=(_CHUNK_ROWS, n_channels, MAX_SAMPLES),
                **_COMPRESS,
            )
            for name in ('run_dir', 'port', 'name', 'tank'):
                f.create_dataset(name, shape=(0,), maxshape=(None,),
                                 dtype=_STR_DTYPE)
            for name, dtype in (('hz', 'int32'), ('ts', 'int64')):
                f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)
            f.attrs['n_channels'] = n_channels
            f.attrs['max_samples'] = MAX_SAMPLES
            f.attrs['created'] = datetime.now(timezone.utc).isoformat()
        return cls(p)

    @classmethod
    def build(
        cls,
        path: Path | str,
        incoming_dir: Path | str,
        exclude_ports: tuple[str, ...] = ('10',),
        limit: int | None = None,
        verbose: bool = True,
    ) -> ChromStore:
        """
        Build a new store by scanning *incoming_dir* for run sub-directories.

        Parameters
        ----------
        path : destination .h5 file (must not already exist)
        incoming_dir : directory whose children are run dirs (e.g. 20260105-195518/)
        exclude_ports : SSV port numbers to skip
        limit : cap number of run dirs for testing
        verbose : print progress
        """
        store = cls.create(path)
        n = store.update(incoming_dir, exclude_ports=exclude_ports,
                         limit=limit, verbose=verbose)
        if verbose:
            print(f'Build complete: {n} injections stored → {path}')
        return store

    # ------------------------------------------------------------------
    # Append / update
    # ------------------------------------------------------------------

    def update(
        self,
        incoming_dir: Path | str,
        exclude_ports: tuple[str, ...] = ('10',),
        limit: int | None = None,
        verbose: bool = True,
    ) -> int:
        """
        Append any run directories in *incoming_dir* not yet in the store.

        Returns the number of injections added.
        """
        incoming = Path(incoming_dir)
        existing = self._existing_keys()

        run_dirs = sorted(incoming.glob('*-*'))
        if limit is not None:
            run_dirs = run_dirs[:limit]

        rows: list[dict] = []
        signals: list[np.ndarray] = []

        for run_dir in run_dirs:
            loader = RunLoader(run_dir)
            itx_files = loader._itx_files()

            for file in itx_files:
                port = loader.port_from_filename(file)
                if port in exclude_ports:
                    continue
                key = (run_dir.name, port)
                if key in existing:
                    continue

                itx = itx_import.ITX(file)
                if getattr(itx, 'data', None) is None or not hasattr(itx, 'chroms'):
                    continue

                n_ch, n_samp = itx.chroms.shape
                sig = itx.chroms[:, :MAX_SAMPLES].astype(np.int32)
                if n_samp < MAX_SAMPLES:
                    # pad short injections with last value per channel
                    pad = np.zeros((n_ch, MAX_SAMPLES - n_samp), dtype=np.int32)
                    for ch in range(n_ch):
                        pad[ch] = sig[ch, -1]
                    sig = np.concatenate([sig, pad], axis=1)

                tank = loader.port_name(port)
                ts = _name_to_timestamp(itx.name)

                rows.append({
                    'run_dir': run_dir.name,
                    'port': port,
                    'name': itx.name,
                    'tank': tank,
                    'hz': int(itx.datafreq),
                    'ts': ts,
                })
                signals.append(sig)

        if not rows:
            return 0

        self._append(signals, rows)
        if verbose:
            print(f'Added {len(rows)} injections from {incoming_dir}')
        return len(rows)

    def _append(self, signals: list[np.ndarray], rows: list[dict]) -> None:
        """Write a batch of new injections to the HDF5 file."""
        arr = np.stack(signals, axis=0)   # [batch, n_channels, MAX_SAMPLES]
        batch = len(rows)

        with h5py.File(self.path, 'a') as f:
            n = f['signal'].shape[0]
            for ds in f.values():
                ds.resize(n + batch, axis=0)

            f['signal'][n:] = arr
            f['run_dir'][n:] = [r['run_dir'].encode() for r in rows]
            f['port'][n:] = [r['port'].encode() for r in rows]
            f['name'][n:] = [r['name'].encode() for r in rows]
            f['tank'][n:] = [r['tank'].encode() for r in rows]
            f['hz'][n:] = [r['hz'] for r in rows]
            f['ts'][n:] = [r['ts'] for r in rows]

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def _existing_keys(self) -> set[tuple[str, str]]:
        """Return set of (run_dir, port) already in the store."""
        with h5py.File(self.path, 'r') as f:
            if f['run_dir'].shape[0] == 0:
                return set()
            dirs = f['run_dir'][:].astype(str)
            ports = f['port'][:].astype(str)
        return set(zip(dirs, ports))

    def metadata(self) -> pd.DataFrame:
        """Return all injection metadata as a DataFrame."""
        with h5py.File(self.path, 'r') as f:
            df = pd.DataFrame({
                'run_dir': f['run_dir'][:].astype(str),
                'port':    f['port'][:].astype(str),
                'name':    f['name'][:].astype(str),
                'tank':    f['tank'][:].astype(str),
                'hz':      f['hz'][:],
                'ts':      f['ts'][:],
            })
        df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        return df

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        channel: int | None = None,
        port: str | None = None,
        run_dir: str | None = None,
        tank: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Load raw signal data with optional filtering.

        Parameters
        ----------
        channel : if given, return only that channel → shape [n, MAX_SAMPLES]
                  otherwise return all channels → shape [n, n_channels, MAX_SAMPLES]
        port, run_dir, tank : metadata filters (exact string match)

        Returns
        -------
        (meta_df, signals)
            meta_df : DataFrame with injection metadata, index reset
            signals : numpy array of raw int32 signal data
        """
        meta = self.metadata()
        mask = pd.Series([True] * len(meta))
        if port is not None:
            mask &= meta['port'] == port
        if run_dir is not None:
            mask &= meta['run_dir'] == run_dir
        if tank is not None:
            mask &= meta['tank'] == tank

        indices = np.where(mask.values)[0]
        if len(indices) == 0:
            empty = np.empty((0,) if channel is None else (0, MAX_SAMPLES),
                             dtype=np.int32)
            return meta.iloc[[]], empty

        with h5py.File(self.path, 'r') as f:
            # Full sequential read then numpy filter — much faster than HDF5
            # fancy indexing which decompresses one chunk per scattered index.
            sig = f['signal'][:]                # [N, n_channels, MAX_SAMPLES]

        sig = sig[indices]                      # numpy filter: near-instant
        if channel is not None:
            sig = sig[:, channel, :]            # [n, MAX_SAMPLES]

        return meta[mask].reset_index(drop=True), sig

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> str:
        with h5py.File(self.path, 'r') as f:
            n = f['signal'].shape[0]
            n_ch = f['signal'].shape[1]
            created = f.attrs.get('created', '?')
            size_mb = self.path.stat().st_size / 1e6
        meta = self.metadata()
        n_runs = meta['run_dir'].nunique()
        date_range = (
            f"{meta['datetime'].min().date()} – {meta['datetime'].max().date()}"
            if n > 0 else '—'
        )
        return (
            f'ChromStore: {self.path}\n'
            f'  injections : {n:,}\n'
            f'  runs       : {n_runs:,}\n'
            f'  channels   : {n_ch}\n'
            f'  date range : {date_range}\n'
            f'  file size  : {size_mb:.1f} MB\n'
            f'  created    : {created}'
        )

    def __repr__(self) -> str:
        with h5py.File(self.path, 'r') as f:
            n = f['signal'].shape[0]
        return f'ChromStore({self.path.name}, n={n:,})'


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _name_to_timestamp(name: str) -> int:
    """
    Convert chromatogram name (YYMMDD.HHMM.port) to a Unix timestamp (UTC).
    Example: '260105.2006.1' → 2026-01-05 20:06 UTC
    """
    try:
        dt = datetime.strptime(name[:11], '%y%m%d.%H%M').replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except ValueError:
        return 0
