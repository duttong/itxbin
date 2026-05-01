"""
Single-channel chromatogram: baseline estimation, peak detection, integration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve1d, minimum_filter1d
from scipy.signal import find_peaks, savgol_filter


def _find_peak_edges(
    dy: np.ndarray,
    apex_rel: int,
    frac: float = 0.10,
    min_width: int = 5,
) -> tuple[int, int]:
    """
    Find peak start/stop by scanning outward from each derivative inflection point
    until the derivative magnitude falls to *frac* × its peak value.

    On the left side the derivative is positive (rising signal); the inflection
    point is where dy is maximum.  We scan left from there until dy drops below
    frac × max_dy.  Mirror logic on the right (dy negative, falling signal).

    Parameters
    ----------
    dy        : first derivative of the (smoothed, baseline-corrected) signal
    apex_rel  : index of signal apex within dy
    frac      : fraction of peak derivative magnitude that defines the boundary
                (0.10 = 10 % — wider → boundaries further from peak)
    min_width : minimum half-width in samples; returns window edges if narrower

    Returns
    -------
    (left_b, right_b) — sample indices within dy
    """
    n = len(dy)
    left_b, right_b = 0, n - 1

    # Left boundary
    left_dy = dy[:apex_rel]
    if len(left_dy) >= min_width:
        max_dy = float(left_dy.max())
        if max_dy > 0:
            infl_l = int(np.argmax(left_dy))
            for i in range(infl_l, -1, -1):
                if dy[i] < frac * max_dy:
                    left_b = i
                    break

    # Right boundary
    right_dy = dy[apex_rel + 1:]
    if len(right_dy) >= min_width:
        min_dy = float(right_dy.min())
        if min_dy < 0:
            infl_r = apex_rel + 1 + int(np.argmin(right_dy))
            for i in range(infl_r, n):
                if dy[i] > frac * min_dy:   # dy approaching 0 from below
                    right_b = i
                    break

    return left_b, right_b


@dataclass
class Peak:
    """Integrated peak result for one detected peak."""
    center_time: float   # seconds at apex
    center_idx: int      # sample index at apex
    left_idx: int        # left integration boundary (sample index)
    right_idx: int       # right integration boundary (sample index)
    height: float        # baseline-corrected apex height (counts)
    area: float          # trapezoid area above baseline (counts·s)


class Chromatogram:
    """
    One channel of raw GC detector signal.

    Provides Savitzky-Golay / box smoothing, rolling-minimum baseline
    estimation, and scipy-based peak detection with trapezoidal integration.
    """

    def __init__(self, signal: np.ndarray, hz: int, name: str = '', channel: int = 0):
        self.signal = np.asarray(signal, dtype=float)
        self.hz = hz
        self.name = name
        self.channel = channel
        self.n = len(self.signal)
        self.time = np.arange(self.n) / hz      # seconds
        self._baseline: np.ndarray | None = None
        self._noise_std: float | None = None     # cached noise floor for integrate_window

    # ------------------------------------------------------------------
    # Smoothing (in-place, returns self for chaining)
    # ------------------------------------------------------------------

    def smooth_sg(self, winsize: int = 61, order: int = 4) -> Chromatogram:
        """Savitzky-Golay smooth."""
        self.signal = savgol_filter(self.signal, winsize, order)
        self._baseline = None
        self._noise_std = None
        return self

    def smooth_box(self, winsize: int = 25) -> Chromatogram:
        """Box (moving-average) smooth."""
        kernel = np.ones(winsize) / winsize
        self.signal = convolve1d(self.signal, kernel, mode='nearest')
        self._baseline = None
        self._noise_std = None
        return self

    # ------------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------------

    def estimate_baseline(self, window_sec: float = 60.0) -> np.ndarray:
        """
        Rolling minimum over window_sec, then Savitzky-Golay smoothed.

        The rolling minimum tracks the signal floor even with slow detector
        drift, and is robust against narrow peaks because the window spans
        many peak widths.  The subsequent SG smooth prevents the staircase
        artifact that a raw running minimum produces.
        """
        win = max(int(window_sec * self.hz), 3)
        if win % 2 == 0:
            win += 1
        raw_min = minimum_filter1d(self.signal, size=win, mode='nearest')

        # SG window must be odd and ≤ n; use same width capped at signal length
        sg_win = min(win, self.n if self.n % 2 == 1 else self.n - 1)
        if sg_win % 2 == 0:
            sg_win -= 1
        sg_win = max(sg_win, 5)

        self._baseline = savgol_filter(raw_min, sg_win, 3)
        return self._baseline

    @property
    def corrected(self) -> np.ndarray:
        """Baseline-subtracted signal; estimates baseline on first access."""
        if self._baseline is None:
            self.estimate_baseline()
        return self.signal - self._baseline

    # ------------------------------------------------------------------
    # Peak detection + integration
    # ------------------------------------------------------------------

    def detect_peaks(
        self,
        min_height: float | None = None,
        min_width_sec: float = 2.0,
        distance_sec: float = 15.0,
        prominence_factor: float = 5.0,
    ) -> list[Peak]:
        """
        Detect and integrate peaks on the baseline-corrected signal.

        Parameters
        ----------
        min_height:
            Minimum apex height above baseline.  Auto-computed as
            ``prominence_factor × noise_std`` when None.
        min_width_sec:
            Minimum peak width in seconds.
        distance_sec:
            Minimum separation between adjacent peaks in seconds.
        prominence_factor:
            Multiplier on noise std for automatic height / prominence threshold.

        Returns
        -------
        List of Peak objects sorted by retention time.
        """
        y = self.corrected

        # Noise floor: median std across non-overlapping 30-second windows.
        # min() would pick the near-zero tail (after all peaks) and make the
        # threshold too small; median is anchored to typical baseline variation.
        win_pts = max(int(30 * self.hz), 10)
        if self.n > win_pts:
            stds = np.array([
                y[i:i + win_pts].std()
                for i in range(0, self.n - win_pts, win_pts)
            ])
            noise_std = float(np.median(stds))
        else:
            noise_std = float(y.std())
        noise_std = max(noise_std, 1.0)
        threshold = prominence_factor * noise_std

        if min_height is None:
            min_height = max(threshold, 1.0)

        idxs, props = find_peaks(
            y,
            height=min_height,
            width=int(min_width_sec * self.hz),
            distance=int(distance_sec * self.hz),
            prominence=threshold,
        )

        peaks = []
        for i, idx in enumerate(idxs):
            left = int(props['left_bases'][i])
            right = int(props['right_bases'][i])
            area = float(np.trapz(y[left:right + 1], self.time[left:right + 1]))
            peaks.append(Peak(
                center_time=float(self.time[idx]),
                center_idx=int(idx),
                left_idx=left,
                right_idx=right,
                height=float(y[idx]),
                area=area,
            ))
        return peaks

    def _noise_floor(self) -> float:
        """Cached per-chromatogram noise floor from the corrected signal."""
        if self._noise_std is None:
            y = self.corrected
            win_pts = max(int(30 * self.hz), 10)
            if self.n > win_pts:
                stds = np.array([y[i:i + win_pts].std()
                                 for i in range(0, self.n - win_pts, win_pts)])
                self._noise_std = max(float(stds.min()), 1.0)
            else:
                self._noise_std = max(float(y.std()), 1.0)
        return self._noise_std

    def integrate_window(
        self,
        center_rt: float,
        win_lo: float = 15.0,
        win_hi: float | None = None,
        rt_shift: float = 0.0,
    ) -> Peak | None:
        """
        Fixed-window integration (method 1).

        Places integration boundaries at [center_rt - win_lo, center_rt + win_hi],
        draws a local linear baseline between those two endpoints, then measures
        height (apex above baseline) and area (trapezoid above baseline).

        Choose win_lo / win_hi so the boundaries land in flat baseline regions —
        the local linear baseline is only valid there.  Use asymmetric windows to
        avoid adjacent peaks (e.g. a large O₂ peak that starts right of the RT).

        Parameters
        ----------
        center_rt : float   nominal RT of the compound (seconds)
        win_lo    : float   seconds before RT for the left boundary
        win_hi    : float   seconds after  RT for the right boundary (default = win_lo)
        rt_shift  : float   per-injection RT correction (from alignment)
        """
        if win_hi is None:
            win_hi = win_lo

        y = self.corrected
        adjusted_rt = center_rt - rt_shift

        lo = max(0, int((adjusted_rt - win_lo) * self.hz))
        hi = min(self.n - 1, int((adjusted_rt + win_hi) * self.hz))
        if hi <= lo:
            return None

        # Apex is searched within ±half the tighter window so that a larger
        # adjacent compound can't steal the argmax.
        apex_hw = min(win_lo, win_hi) / 2.0
        alo = max(lo, int((adjusted_rt - apex_hw) * self.hz))
        ahi = min(hi, int((adjusted_rt + apex_hw) * self.hz))
        apex_idx = alo + int(np.argmax(y[alo:ahi + 1]))
        apex_rel = apex_idx - lo            # position within full segment

        segment = y[lo:hi + 1]
        if segment[apex_rel] < 3.0 * self._noise_floor():
            return None

        # Height and area above the rolling-minimum baseline.
        height = float(segment[apex_rel])
        area   = float(np.trapz(segment, self.time[lo:hi + 1]))
        if area <= 0:
            return None

        return Peak(
            center_time=float(self.time[apex_idx]),
            center_idx=apex_idx,
            left_idx=lo,
            right_idx=hi,
            height=height,
            area=area,
        )

    def integrate_tangent_skim(
        self,
        center_rt: float,
        win_lo: float = 15.0,
        win_hi: float | None = None,
        rt_shift: float = 0.0,
        n_iter: int = 2,
    ) -> Peak | None:
        """
        Tangent-skim integration (method 2).

        Iteratively finds peak boundaries via first-derivative zero crossings,
        subtracts the linear baseline between them, and repeats.  Each iteration
        drives the baseline flatter, giving more accurate boundary detection.
        Height is the apex of the fully-skimmed residual.

        Parameters
        ----------
        center_rt : float   nominal RT (seconds)
        win_lo    : float   search window before RT (must include the baseline)
        win_hi    : float   search window after  RT (must include the baseline)
        rt_shift  : float   per-injection RT correction
        n_iter    : int     number of skim iterations (1–3; default 2)
        """
        if win_hi is None:
            win_hi = win_lo

        y = self.corrected
        adjusted_rt = center_rt - rt_shift

        lo = max(0, int((adjusted_rt - win_lo) * self.hz))
        hi = min(self.n - 1, int((adjusted_rt + win_hi) * self.hz))
        if hi <= lo:
            return None

        seg = y[lo:hi + 1].copy()
        t   = self.time[lo:hi + 1]
        n   = len(seg)

        if seg.max() < 3.0 * self._noise_floor():
            return None

        left_b, right_b = 0, n - 1
        dt = 1.0 / self.hz

        apex_hw = min(win_lo, win_hi) / 2.0
        apex_alo = max(0, int((adjusted_rt - apex_hw) * self.hz) - lo)
        apex_ahi = min(n - 1, int((adjusted_rt + apex_hw) * self.hz) - lo)

        for _ in range(n_iter):
            apex_rel = apex_alo + int(np.argmax(seg[apex_alo:apex_ahi + 1]))
            dy = np.gradient(seg, dt)
            left_b, right_b = _find_peak_edges(dy, apex_rel)

            if right_b <= left_b + 2:
                break

            # Subtract linear baseline between discovered boundaries
            n_bl = right_b - left_b + 1
            bl   = np.linspace(seg[left_b], seg[right_b], n_bl)
            seg[left_b:right_b + 1] -= bl
            # Zero-clamp outside the peak region
            seg[:left_b]      = np.maximum(seg[:left_b],      0.0)
            seg[right_b + 1:] = np.maximum(seg[right_b + 1:], 0.0)

        # Final apex: search within the restricted apex window, clipped to discovered boundaries
        final_alo = max(left_b, apex_alo)
        final_ahi = min(right_b, apex_ahi)
        if final_ahi > final_alo:
            apex_rel = final_alo + int(np.argmax(seg[final_alo:final_ahi + 1]))
        else:
            apex_rel = apex_alo

        height = float(seg[apex_rel])
        if height < 3.0 * self._noise_floor():
            return None

        area = float(np.trapz(seg[left_b:right_b + 1], t[left_b:right_b + 1]))
        if area <= 0:
            return None

        return Peak(
            center_time=float(t[apex_rel]),
            center_idx=lo + apex_rel,
            left_idx=lo + left_b,
            right_idx=lo + right_b,
            height=height,
            area=area,
        )
