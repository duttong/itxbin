"""
Retention-time alignment for GC chromatograms.

Strategy
--------
Use a reference peak (the largest stable peak on a given channel) to measure
the RT shift in each injection relative to a nominal/template RT.  Apply that
shift when placing integration windows, so all compounds on the channel are
corrected together.

The reference peak is selected automatically as the detected peak with the
highest median area across a batch of injections, subject to a clip-detection
guard (peaks that saturate the detector are excluded as unreliable references).

Cross-correlation within a search window is used to compute the sub-sample RT
shift, which is then rounded to the nearest 0.1 s.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate

from integrator.chrom import Chromatogram, Peak


# ──────────────────────────────────────────────────────────────────────────────
# Single-injection RT shift estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_rt_shift(
    chrom: Chromatogram,
    ref_rt: float,
    search_window_sec: float = 10.0,
    template: np.ndarray | None = None,
) -> float:
    """
    Estimate the RT shift for one injection relative to *ref_rt*.

    Two modes:
    - If *template* is supplied: cross-correlate the region around *ref_rt* in
      *chrom.corrected* against the template to find the sub-sample shift.
    - Otherwise: find the peak apex within ±*search_window_sec* of *ref_rt*
      and return (apex_time − ref_rt).

    Parameters
    ----------
    chrom : Chromatogram
        Single-channel baseline-corrected chromatogram.
    ref_rt : float
        Nominal retention time of the reference peak (seconds).
    search_window_sec : float
        Half-width of the search window in seconds.
    template : np.ndarray | None
        Optional reference signal snippet to cross-correlate against.

    Returns
    -------
    shift_sec : float   positive → peak arrived late, negative → arrived early
    """
    y = chrom.corrected
    hz = chrom.hz

    lo = max(0, int((ref_rt - search_window_sec) * hz))
    hi = min(len(y), int((ref_rt + search_window_sec) * hz))
    segment = y[lo:hi]

    if len(segment) == 0:
        return 0.0

    if template is not None and len(template) == len(segment):
        # cross-correlation: peak of xcorr gives the lag
        xcorr = correlate(segment, template, mode='full')
        lag_samples = int(np.argmax(xcorr)) - (len(template) - 1)
        return round(lag_samples / hz, 2)
    else:
        # simple apex search
        apex_idx = int(np.argmax(segment)) + lo
        apex_time = apex_idx / hz
        return round(apex_time - ref_rt, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Batch: discover nominal RT and build per-injection shift table
# ──────────────────────────────────────────────────────────────────────────────

def discover_reference_rt(
    chroms: list[Chromatogram],
    candidate_peaks: list[list[Peak]],
    clip_fraction: float = 0.90,
) -> tuple[float, int]:
    """
    Choose the best reference peak for RT alignment from a batch of injections.

    Selects the peak index (within each injection's peak list) that has the
    highest median area and is not clipped.  "Clipped" means the apex height
    is ≥ *clip_fraction* of the signal's max value.

    Returns
    -------
    nominal_rt : float   median RT of the chosen reference peak (seconds)
    peak_index : int     index into each injection's peak list
    """
    if not candidate_peaks or not candidate_peaks[0]:
        raise ValueError('No peaks supplied for reference discovery')

    n_peaks = min(len(p) for p in candidate_peaks if p)
    if n_peaks == 0:
        raise ValueError('All injections have empty peak lists')

    best_idx = 0
    best_area = -np.inf

    for pk_idx in range(n_peaks):
        areas = []
        clipped = 0
        for chrom, peaks in zip(chroms, candidate_peaks):
            if pk_idx >= len(peaks):
                continue
            pk = peaks[pk_idx]
            # clip guard: skip if apex > clip_fraction of signal max
            if pk.height >= clip_fraction * chrom.signal.max():
                clipped += 1
                continue
            areas.append(pk.area)

        if not areas:
            continue
        # penalise if frequently clipped
        if clipped > len(candidate_peaks) // 2:
            continue

        median_area = float(np.median(areas))
        if median_area > best_area:
            best_area = median_area
            best_idx = pk_idx

    rts = [
        peaks[best_idx].center_time
        for peaks in candidate_peaks
        if best_idx < len(peaks)
    ]
    nominal_rt = float(np.median(rts))
    return nominal_rt, best_idx


def build_shift_table(
    chroms: list[Chromatogram],
    ref_rt: float,
    search_window_sec: float = 10.0,
) -> list[float]:
    """
    Return per-injection RT shifts (seconds) relative to *ref_rt*.

    Uses cross-correlation against the median segment as a template so
    that the shift is robust to noise in individual injections.
    """
    hz = chroms[0].hz
    lo = max(0, int((ref_rt - search_window_sec) * hz))
    hi = min(len(chroms[0].corrected), int((ref_rt + search_window_sec) * hz))

    # Build median template from all injections
    segments = []
    for c in chroms:
        seg = c.corrected[lo:hi]
        if len(seg) == hi - lo:
            segments.append(seg)

    if not segments:
        return [0.0] * len(chroms)

    template = np.median(np.array(segments), axis=0)

    shifts = []
    for c in chroms:
        shift = estimate_rt_shift(c, ref_rt, search_window_sec, template=template)
        shifts.append(shift)
    return shifts
