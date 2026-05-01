"""
Batch integration pipeline: store → smooth → RT-align → windowed integration.

Entry point for processing a full year's worth of injections from the HDF5 store.
Each channel is processed independently: a cross-correlation shift is estimated
for every injection using the marked reference compound, then applied to all
compound windows on that channel.

Memory strategy: channels are processed one at a time.  The float64 Chromatogram
objects for one channel (~200 MB) are released before the next channel is loaded.
The raw int32 store array (~303 MB for 4 k injections) stays resident throughout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import savgol_filter

from integrator.chrom import Chromatogram
from integrator.peak_list import PeakList
from integrator.rt_align import build_shift_table
from integrator.store import ChromStore


def batch_integrate(
    store: ChromStore,
    peak_list: PeakList,
    smooth: bool = True,
    smooth_method: str = 'sg',
    sg_winsize: int = 61,
    sg_order: int = 4,
    box_winsize: int = 25,
    exclude_ports: tuple[str, ...] = ('10',),
    align: bool = True,
    max_shift_sec: float = 5.0,
    shift_lookback: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Integrate all injections in *store* using windowed integration from *peak_list*.

    Parameters
    ----------
    store         : HDF5 chromatogram store
    peak_list     : compounds with nominal RTs, windows, and ref flags
    smooth        : apply smoothing before integration
    smooth_method : 'sg' (Savitzky-Golay) or 'box' (moving average)
    sg_winsize    : SG window size in samples (must be odd); used when smooth_method='sg'
    sg_order      : SG polynomial order; used when smooth_method='sg'
    box_winsize   : box-filter window size in samples; used when smooth_method='box'
    exclude_ports : SSV port numbers to skip (default: port 10 = initial push run)
    align         : apply per-channel RT shift correction using ref compound
    max_shift_sec : shifts beyond this magnitude are treated as alignment failures;
                    the shift is then filled from the nearest preceding valid injection
    shift_lookback: how many prior injections to search for a valid shift when
                    alignment fails; defaults to 3 (falls back to 0.0 if none found)
    verbose       : print progress messages

    Returns
    -------
    DataFrame with columns:
        run_dir, port, name, tank, ts, datetime,
        shift_ch{N} for each channel that has a ref compound,
        one float area column per compound (NaN = compound absent or not detected).
    Sorted by ts (injection timestamp).
    """
    if verbose:
        print(f'Loading store: {store.path} ...', end=' ', flush=True)

    meta, sigs = store.load()   # sigs: [N, n_channels, MAX_SAMPLES] int32

    if verbose:
        print(f'{len(meta):,} injections loaded')

    # Exclude unwanted ports
    mask = ~meta['port'].isin(list(exclude_ports))
    meta = meta[mask].reset_index(drop=True)
    sigs = sigs[mask.values]
    n = len(meta)

    if verbose and mask.sum() < len(mask):
        excluded = (~mask).sum()
        print(f'  excluded {excluded} injections (ports {", ".join(exclude_ports)})')
        print(f'  processing {n:,} injections')

    channels = peak_list.channels()
    n_ch_stored = sigs.shape[1]

    # Per-compound result arrays (indexed by injection order in meta)
    compound_areas: dict[str, np.ndarray] = {
        c.report_name: np.full(n, np.nan) for c in peak_list.compounds
    }
    compound_heights: dict[str, np.ndarray] = {
        c.report_name: np.full(n, np.nan) for c in peak_list.compounds
    }
    shift_arrays: dict[str, np.ndarray] = {}

    for ch in channels:
        if ch >= n_ch_stored:
            if verbose:
                print(f'ch{ch}: not in store (only {n_ch_stored} channels) — skipped')
            continue

        compounds_ch = peak_list.by_channel(ch)
        ref = peak_list.reference_peak(ch)

        if verbose:
            ref_label = f'{ref.report_name} at {ref.nominal_rt}s' if ref else 'none'
            print(f'ch{ch}: {len(compounds_ch)} compounds, ref={ref_label}', flush=True)

        # Batch smooth + baseline for the entire channel at once.
        # This is ~10× faster than calling smooth_sg() / estimate_baseline()
        # on 4k individual Chromatogram objects.
        sigs_ch = sigs[:, ch, :].astype(float)   # [n, MAX_SAMPLES] float64
        if smooth:
            if smooth_method == 'box':
                sigs_ch = uniform_filter1d(sigs_ch, size=box_winsize, axis=1, mode='nearest')
            else:
                sigs_ch = savgol_filter(sigs_ch, sg_winsize, sg_order, axis=1)

        # Batch baseline: rolling minimum → SG smooth (mirrors Chromatogram.estimate_baseline)
        hz_ref = int(meta['hz'].iloc[0])   # all injections share the same Hz
        win = max(int(60.0 * hz_ref), 3)
        if win % 2 == 0:
            win += 1
        raw_min = minimum_filter1d(sigs_ch, size=win, mode='nearest', axis=1)
        sg_win = min(win, sigs_ch.shape[1] if sigs_ch.shape[1] % 2 == 1
                     else sigs_ch.shape[1] - 1)
        if sg_win % 2 == 0:
            sg_win -= 1
        sg_win = max(sg_win, 5)
        baseline_ch = savgol_filter(raw_min, sg_win, 3, axis=1)
        del raw_min

        # Build Chromatogram objects with pre-computed smoothed signal + baseline
        chroms: list[Chromatogram] = []
        hz_vals = meta['hz'].values
        names = meta['name'].values
        for i in range(n):
            c = Chromatogram(sigs_ch[i], int(hz_vals[i]),
                             name=str(names[i]), channel=ch)
            c._baseline = baseline_ch[i]
            chroms.append(c)
        del sigs_ch, baseline_ch

        if verbose:
            action = 'smoothed' if smooth else 'loaded'
            print(f'  {action} {len(chroms):,} chromatograms')

        # RT alignment
        shifts: list[float]
        if align and ref is not None and ref.nominal_rt is not None:
            raw_shifts = build_shift_table(
                chroms,
                ref.nominal_rt,
                search_window_sec=min(ref.rt_window_lo, ref.rt_window_hi),
            )
            arr = np.array(raw_shifts)
            # Replace failed alignments with the nearest preceding valid shift
            # (up to shift_lookback injections back).  Injections without a
            # reference compound (zero-air, push runs) still get a sensible RT
            # correction so that small peaks near zero can be found accurately.
            bad = np.abs(arr) > max_shift_sec
            if bad.any():
                filled = 0
                unfilled = 0
                for i in np.where(bad)[0]:
                    for k in range(1, shift_lookback + 1):
                        j = i - k
                        if j >= 0 and not bad[j]:
                            arr[i] = arr[j]
                            filled += 1
                            break
                    else:
                        arr[i] = 0.0   # no valid predecessor found
                        unfilled += 1
                if verbose:
                    print(f'  {bad.sum()} alignment failures: '
                          f'{filled} filled from prior injection, '
                          f'{unfilled} defaulted to 0'
                          f'  (|shift| > {max_shift_sec}s)')
            shifts = arr.tolist()
            col = f'shift_ch{ch}'
            shift_arrays[col] = arr
            if verbose:
                good = arr[~bad] if bad.any() else arr
                print(
                    f'  RT alignment: shift [{good.min():.2f}, {good.max():.2f}] s'
                    f'  σ={good.std():.3f} s  ({(~bad).sum()} valid)'
                )
        else:
            shifts = [0.0] * n

        # Integration for every compound on this channel (method from config)
        for compound in compounds_ch:
            if compound.nominal_rt is None:
                continue
            areas   = compound_areas[compound.report_name]
            heights = compound_heights[compound.report_name]
            for i, (chrom, shift) in enumerate(zip(chroms, shifts)):
                if compound.meth == 2:
                    pk = chrom.integrate_tangent_skim(
                        center_rt=compound.nominal_rt,
                        win_lo=compound.rt_window_lo,
                        win_hi=compound.rt_window_hi,
                        rt_shift=shift,
                    )
                else:
                    pk = chrom.integrate_window(
                        center_rt=compound.nominal_rt,
                        win_lo=compound.rt_window_lo,
                        win_hi=compound.rt_window_hi,
                        rt_shift=shift,
                    )
                if pk is not None:
                    areas[i]   = pk.area
                    heights[i] = pk.height

        # Release Chromatogram objects before next channel
        del chroms

    # Assemble result DataFrame
    df = meta[['run_dir', 'port', 'name', 'tank', 'ts']].copy()
    df['datetime'] = pd.to_datetime(df['ts'], unit='s', utc=True)

    for col, arr in shift_arrays.items():
        df[col] = arr

    for compound in peak_list.compounds:
        df[compound.report_name] = compound_areas[compound.report_name]
        df[f'{compound.report_name}_ht'] = compound_heights[compound.report_name]

    df = df.sort_values('ts').reset_index(drop=True)

    if verbose:
        compound_cols = [c.report_name for c in peak_list.compounds]
        n_complete = df[compound_cols].notna().all(axis=1).sum()
        n_any = df[compound_cols].notna().any(axis=1).sum()
        print(
            f'Done: {len(df):,} injections  '
            f'{n_complete:,} fully detected  '
            f'{len(df) - n_any:,} all-absent'
        )

    return df
