#!/usr/bin/env python3
"""Flag CATS chromatograms whose overall shape deviates from their own
local neighbors' typical shape, and persist results to hats.ng_chromatogram_qc.

Reads raw chromatogram files directly (not DB mole-fraction rows). Each
chromatogram is normalized by dividing by the mean signal in its own
29.0-29.5 min tail window -- a quiet, settled region after all peaks have
eluted, so every run is expressed relative to its own steady-state level.

Design (Geoff, Aug 2026): build a *local reference trace* from the pointwise
MEDIAN of the N nearest chromatograms before and after the target (same
channel), then score how far the target deviates from that reference as the
mean of target/reference over the WHOLE trace, with the analyte peak
region(s) masked out.

Whole trace, not a fixed early/late window: peak retention times drift over
a 25+ year archive and differ by channel, so any hardcoded minute range
(the original design scored only 0-5 min) eventually goes stale or misses
events outside it -- e.g. 260713.0258.8 (SPO channel0) has a normal 0-5 min
baseline but a real post-peak sag around minute 22-23 that a pre-peak-only
window can't see. Masking instead of windowing sidesteps the drift problem
entirely: the mask comes from GCwerks' own dated peakid files
(find_gcwerks_peakid_file / read_gcwerks_peak_windows), which already track
each channel's real analyte retention times across the whole archive, so it
adapts automatically instead of needing per-channel/per-era config.

Ratio, not difference: dividing target by reference (rather than the
original design's target-minus-reference) makes the statistic scale-free
and lets one mean summarize the ENTIRE non-peak trace as a single number,
rather than needing per-region difference thresholds. On a quiet month
(SPO channel0, June 2026, 60 runs) the whole-trace ratio mean sits tight at
0.986-1.007 (std 0.004); the 260713 event scores 0.89-0.90, a clean
separation with no z-score tuning needed.

Masking peaks is required, not optional: an unmasked ratio blows up at any
point where the reference is large and changing fast (peak edges), because
a small height or timing mismatch produces a huge ratio error there --
tested directly, and it let an entirely ordinary tall-peak run (ratio mean
1.6, peak/tail height ratio 7.65x) outrank the real anomaly (ratio mean
0.89). Masking each analyte's peakid window (+0.3 min margin per side)
resolves this while still correctly leaving genuinely oversized peaks
visible in the score, since an outsized peak's base still extends past its
own masked window into the "baseline" being scored.

Median (not mean) is required for the reference: manual review on CATS-SPO
channel0, 2026-07-17 showed a mean-based reference gets measurably dragged
toward a real anomaly when several of the N neighbors are themselves part of
the same multi-run contamination event (the run 260717.2145.2 saturated the
detector; its baseline elevation bled into ~2345.2 UTC that night) -- the
mean reference sat at ~1.2x baseline near the event instead of flat 1.0,
understating the anomaly, while the median stayed correctly flat at ~1.0
until roughly half the window was itself contaminated.

Known limitation: the local median reference degrades if a majority of the
N neighbors on one side are themselves anomalous (a long-duration or
back-to-back event) -- a median only resists a minority of outliers. Keep N
modest (default 10 each side) and review flagged clusters together rather
than trusting any single run's score in isolation.

This does NOT catch pure peak-height dropouts (the peak region is masked
out of the score entirely) -- see cats_peak_qc.py for a peak-height feature.

Scoped by physical GCwerks channel, not by analyte (Geoff, Aug 2026): a
chromatogram file covers every analyte reported on that channel together
(e.g. CATS channel 'f' covers CCl4, CFC11, CFC113, CFC12, CH3CCl3, CHCl3,
H1211, TCE all at once), and decoding+scoring a chromatogram is the
expensive step -- re-running it once per analyte on the same channel would
redo identical work 5-8x for no benefit. This script scans a channel ONCE
and upserts every scored chromatogram (flagged or not) into
hats.ng_chromatogram_qc, keyed on (inst_num, gcwerks_channel_num,
analysis_time, algo_name) so re-running an overlapping range updates in
place rather than duplicating. At the scale this is meant to run at
(25+ years of chromatograms), that means this script is a slow, run-once
(or run-per-new-data) step; cats_tagging.py's "baseline" algorithm then
reads the persisted table (fast, no chromatogram decoding) and tags every
analyte on that channel -- see build_baseline_tag_qc() below.

Usage::

    # Scan one channel and print flagged runs; upsert every scored row
    # (flagged or not) to hats.ng_chromatogram_qc
    python3 cats_baseline_qc.py --site spo --channel f --start 20260101 --end 20260801

    # Preview only -- compute and print, but do not write to the DB
    python3 cats_baseline_qc.py --site spo --channel f --start 20260717 --end 20260718 --dry-run

    # Also write every computed feature to CSV for review
    python3 cats_baseline_qc.py --site spo --channel f --start 20260701 --end 20260731 \\
        --output cats_baseline_features.csv

    # Full workflow: scan once per channel (this script), then tag every
    # analyte on that channel from the persisted results (fast, repeatable)
    python3 cats_baseline_qc.py --site spo --channel f --start 20260101 --end 20260801
    python3 cats_tagging.py --site spo --algo baseline --analyte all --start 20260101 --end 20260801
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "logosdata"))
from gcwerks_chromatogram import (
    find_gcwerks_peakid_file,
    gcwerks_channel_number,
    read_gcwerks_chromatogram,
    read_gcwerks_peak_windows,
)
from cats_cal_step_qc import _group_periods
from cats_batch import CATS_batch


def _parse_yyyymmdd(s: str) -> pd.Timestamp:
    s = s.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt), tz="UTC")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid date {s!r}; expected YYYYMMDD or YYYY-MM-DD.")


def list_chromatogram_files(gc_dir: Path, channel_number: int, start: pd.Timestamp, end: pd.Timestamp):
    """Yield (timestamp, path) for every chromatogram file in [start, end],
    following the {gc_dir}/{YY}/chromatograms/channel{N}/{YYMMDD}.{HHMM}.{ext}
    convention used by find_gcwerks_chromatogram()."""
    for year in range(start.year, end.year + 1):
        directory = gc_dir / f"{year % 100:02d}" / "chromatograms" / f"channel{channel_number}"
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            stem = path.name.split(".")
            if len(stem) < 2:
                continue
            try:
                ts = datetime.strptime(f"{stem[0]}.{stem[1]}", "%y%m%d.%H%M").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            ts = pd.Timestamp(ts)
            if start <= ts <= end:
                yield ts, path


def peak_mask_for_chromatogram(
    gc_dir: Path,
    channel_number: int,
    analysis_time: pd.Timestamp,
    minutes: np.ndarray,
    margin_min: float = 0.3,
    peakid_cache: dict | None = None,
) -> np.ndarray:
    """Boolean mask, True wherever `minutes` falls inside any analyte's peak
    window (+/- margin_min) on this channel, per GCwerks' own dated peakid
    file for analysis_time -- see find_gcwerks_peakid_file(). One
    chromatogram covers every analyte on its channel, so this masks all of
    them in one pass, not just one analyte's window.

    peakid_cache, if given, memoizes the (peakid path -> raw window list)
    lookup so a full scan (which reuses the same handful of dated peakid
    files across potentially hundreds of thousands of chromatograms) doesn't
    re-glob the peakid directory or re-parse the same file per chromatogram.
    """
    path = find_gcwerks_peakid_file(gc_dir, channel_number, analysis_time)
    if path is None:
        return np.zeros(len(minutes), dtype=bool)

    if peakid_cache is not None and path in peakid_cache:
        windows = peakid_cache[path]
    else:
        windows = read_gcwerks_peak_windows(path)
        if peakid_cache is not None:
            peakid_cache[path] = windows

    mask = np.zeros(len(minutes), dtype=bool)
    for w in windows:
        lo = (w.center_seconds - w.width_seconds) / 60.0 - margin_min
        hi = (w.center_seconds + w.width_seconds) / 60.0 + margin_min
        mask |= (minutes >= lo) & (minutes <= hi)
    return mask


def normalized_trace(
    path: Path,
    tail_start_min: float = 29.0,
    tail_end_min: float = 29.5,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp] | None:
    """Read one chromatogram and divide its signal by its own tail-window
    mean, so every trace is expressed relative to its own steady state.
    Returns (minutes, normalized_signal, analysis_datetime) or None."""
    try:
        c = read_gcwerks_chromatogram(path)
    except (OSError, ValueError) as exc:
        print(f"  Skipping {path.name}: {exc}")
        return None
    if len(c.signal) == 0:
        return None

    minutes = c.elapsed_seconds / 60.0
    tail_mask = (minutes >= tail_start_min) & (minutes <= tail_end_min)
    if not tail_mask.any():
        return None
    tail_ref = float(np.mean(c.signal[tail_mask].astype(float)))
    if tail_ref <= 0:
        return None

    ts = (pd.Timestamp(c.start_time, tz="UTC") if c.start_time.tzinfo is None
          else pd.Timestamp(c.start_time).tz_convert("UTC"))
    return minutes, c.signal.astype(float) / tail_ref, ts


def _neighbor_median(sub: np.ndarray) -> np.ndarray:
    """Pointwise median of sub's rows via np.partition, NaN-aware.

    A median only needs the middle one/two order statistics, not a full sort,
    so np.partition beats np.median here (~1.2x) -- worthwhile because this
    runs once per scored chromatogram. Falls back to np.nanmedian when any
    column has masked (NaN) values from peak_mask_for_chromatogram, since
    np.partition doesn't handle NaN placement consistently across columns
    with different numbers of them.
    """
    if np.isnan(sub).any():
        # A column that's a real peak on every row (the normal case: the
        # analyte's own peak window, masked out for every chromatogram)
        # is all-NaN here by design -- nanmedian warns about that on every
        # call, harmlessly; score_matrix already drops non-finite ratio
        # values downstream, so silence the expected warning rather than
        # let it fire once per scored chromatogram.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(sub, axis=0)
    n = len(sub)
    if n % 2:
        return np.partition(sub, n // 2, axis=0)[n // 2]
    part = np.partition(sub, [n // 2 - 1, n // 2], axis=0)
    return 0.5 * (part[n // 2 - 1] + part[n // 2])


def score_matrix(
    matrix: np.ndarray,
    neighbor_window: int,
    min_neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every row of a [n_chromatograms x n_samples] matrix against the
    pointwise median of its time-nearest neighbors, as target/reference
    (ratio, not difference) over the whole trace.

    matrix rows must already be tail-normalized (see build_baseline_qc), with
    each analyte's peak window(s) set to NaN by peak_mask_for_chromatogram --
    masking, not a fixed window, is what lets this score the WHOLE trace: a
    hardcoded early/late minute range would eventually go stale as retention
    times drift over a 25+ year archive, where GCwerks' own dated peakid
    files already track that drift per channel.

    The reference median deliberately EXCLUDES the target row (leave-one-out)
    and uses np.nanmedian per column so a neighbor's own masked samples don't
    corrupt the reference elsewhere. An inclusive median (e.g.
    scipy.ndimage.median_filter) is marginally faster but lets a strongly
    anomalous run drag its own reference toward itself, damping the very
    score meant to catch it -- always in the under-detection direction, and
    worst during the multi-run events this is built for.

    Ratio (not difference) is what lets one mean/std summarize the entire
    non-peak trace as a single scalar: on a quiet month (SPO channel0, June
    2026) the whole-trace ratio mean sits at 0.986-1.007 across 60 ordinary
    runs (std 0.004), against 0.89-0.90 for a validated real event
    (2026-07-13 SPO channel0) -- a clean separation with no per-region
    threshold tuning. An unmasked ratio instead blows up at peak edges (a
    small height/timing mismatch produces a huge ratio error where the
    reference is large and changing fast), which is why peaks must be masked
    rather than merely reduced in weight.

    Returns (ratio_mean, ratio_std, n_neighbors) per row; rows with fewer
    than min_neighbors usable neighbors, or with every sample masked, get NaN
    scores.
    """
    n_rows = len(matrix)
    ratio_mean = np.full(n_rows, np.nan)
    ratio_std = np.full(n_rows, np.nan)
    n_neighbors = np.zeros(n_rows, dtype=int)

    for i in range(n_rows):
        lo = max(0, i - neighbor_window)
        hi = min(n_rows, i + neighbor_window + 1)
        idxs = np.r_[lo:i, i + 1:hi]
        n_neighbors[i] = len(idxs)
        if len(idxs) < min_neighbors:
            continue
        reference = _neighbor_median(matrix[idxs])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = matrix[i] / reference
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            continue
        ratio_mean[i] = ratio.mean()
        ratio_std[i] = ratio.std()

    return ratio_mean, ratio_std, n_neighbors


def build_baseline_qc(
    gc_dir: Path,
    channel_number: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    neighbor_window: int = 10,
    peak_margin_min: float = 0.3,
    ratio_threshold: float = 0.05,
    max_gap_hours: float = 1.0,
    workers: int = 16,
    verbose: bool = False,
) -> pd.DataFrame:
    """Scan chromatograms in [start, end] and flag whole-trace shape
    anomalies against each run's own local (time-nearest) neighbors.

    Loads an extra neighbor_window runs of context on each side of the
    requested range so runs near the start/end of the window still get a
    full neighbor set; those context-only rows are dropped before scoring.

    peak_margin_min pads each analyte's GCwerks peakid window (see
    peak_mask_for_chromatogram) before excluding it from the score.

    ratio_threshold is an absolute cutoff on |ratio_mean - 1|, not a
    z-score -- deliberately, since the whole point of the local-median
    design is to not need a separately-tuned statistical threshold. 0.05
    sits well above the 0.986-1.007 range (std 0.004) seen on 60 quiet SPO
    channel0 runs in June 2026, and below the 0.89-0.90 range seen on the
    validated 2026-07-13 event; adjust after reviewing --output on more
    data or other channels/sites.

    workers threads read+decode chromatograms concurrently. Reads come off
    NFS and are latency-bound, not bandwidth- or CPU-bound (~38 ms/file cold
    serially, so the CPU mostly idles waiting on round-trips) -- overlapping
    them measured ~7.5x (28 -> 209 files/sec) at 16 workers, which dominates
    the total runtime of a full-history scan. workers=1 forces serial reads.
    """
    load_start = start - pd.Timedelta(hours=neighbor_window * 2)
    load_end = end + pd.Timedelta(hours=neighbor_window * 2)

    paths = [path for _ts, path in
             list_chromatogram_files(gc_dir, channel_number, load_start, load_end)]
    if verbose:
        print(f"  {len(paths)} chromatogram files to read ({workers} workers)...")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(normalized_trace, paths))
    else:
        results = [normalized_trace(path) for path in paths]

    traces = [
        (result[0], result[1], result[2], path)
        for result, path in zip(results, paths) if result is not None
    ]
    if len(traces) < neighbor_window * 2 + 1:
        return pd.DataFrame()

    traces.sort(key=lambda t: t[2])
    if verbose:
        print(f"  decoded {len(traces)} chromatograms; masking peaks and scoring...")

    # One [n_chromatograms x n_samples] matrix over the WHOLE trace (not a
    # fixed early/late window -- see module docstring), rows truncated to
    # the shortest trace so the array is rectangular. Each analyte's peak
    # window on this channel is set to NaN per row, using that row's own
    # applicable dated peakid file, so retention-time drift across the
    # archive never needs a hardcoded minute range.
    min_len = min(len(norm) for _m, norm, _t, _p in traces)
    minutes = traces[0][0][:min_len]
    matrix = np.array([norm[:min_len] for _m, norm, _t, _p in traces])

    peakid_cache: dict = {}
    for i, (_m, _n, ts, _p) in enumerate(traces):
        mask = peak_mask_for_chromatogram(
            gc_dir, channel_number, ts, minutes,
            margin_min=peak_margin_min, peakid_cache=peakid_cache,
        )
        matrix[i, mask] = np.nan

    ratio_mean, ratio_std, n_neighbors = score_matrix(matrix, neighbor_window)

    timestamps = np.array([ts for _m, _n, ts, _p in traces])
    in_range = np.array([start <= ts <= end for ts in timestamps])
    scored = in_range & ~np.isnan(ratio_mean)
    if not scored.any():
        return pd.DataFrame()

    df = pd.DataFrame({
        "analysis_datetime": timestamps[scored],
        "path": [str(traces[i][3]) for i in np.flatnonzero(scored)],
        "n_neighbors": n_neighbors[scored],
        "ratio_mean": ratio_mean[scored],
        "ratio_std": ratio_std[scored],
        "n_masked_samples": [int(np.isnan(matrix[i]).sum()) for i in np.flatnonzero(scored)],
    }).sort_values("analysis_datetime").reset_index(drop=True)
    df["baseline_outlier"] = (df["ratio_mean"] - 1.0).abs() > ratio_threshold

    flagged_times = df.loc[df["baseline_outlier"], "analysis_datetime"]
    periods = _group_periods(flagged_times, max_gap_hours)
    df["period_id"] = 0
    for i, (pstart, pend) in enumerate(periods, start=1):
        in_period = df["analysis_datetime"].between(pstart, pend)
        df.loc[in_period, "period_id"] = i

    return df


ALGO_NAME = "baseline"

# /hats/gc/smo now belongs to the new IE3 instrument; the legacy CATS
# instrument's chromatogram archive was moved to /hats/gc/cats_smo instead
# of being removed. Add further site: dirname overrides here if other sites
# are similarly renamed in the future.
GC_DIR_OVERRIDES = {"smo": "cats_smo"}


def upsert_chromatogram_qc(db, inst_num: int, channel_number: int, df: pd.DataFrame, batch_size: int = 500) -> int:
    """Upsert every scored row (flagged or not) from build_baseline_qc()'s
    output into hats.ng_chromatogram_qc, keyed on (inst_num,
    gcwerks_channel_num, analysis_time, algo_name). Re-running an overlapping
    date range updates rows in place rather than duplicating -- the table is
    meant to grow monotonically as more of the 25+ year chromatogram archive
    gets scanned, with each (site, channel) scanned at most once per range.
    """
    if df.empty:
        return 0
    sql = """
        INSERT INTO hats.ng_chromatogram_qc
            (inst_num, gcwerks_channel_num, analysis_time, algo_name,
             ratio_mean, ratio_std, n_masked_samples, outlier, period_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            ratio_mean = VALUES(ratio_mean),
            ratio_std = VALUES(ratio_std),
            n_masked_samples = VALUES(n_masked_samples),
            outlier = VALUES(outlier),
            period_id = VALUES(period_id)
    """
    rows = df.reset_index(drop=True)
    n = 0
    for i in range(0, len(rows), batch_size):
        chunk = rows.iloc[i:i + batch_size]
        params = [
            (
                inst_num, channel_number,
                row.analysis_datetime.strftime("%Y-%m-%d %H:%M:%S"), ALGO_NAME,
                float(row.ratio_mean), float(row.ratio_std), int(row.n_masked_samples),
                int(bool(row.baseline_outlier)), int(row.period_id),
            )
            for row in chunk.itertuples(index=False)
        ]
        db.doMultiInsert(sql, params, all=True)
        n += len(params)
    return n


def read_channel_outliers(db, inst_num: int, channel_number: int, start: str, end: str) -> pd.DataFrame:
    """Read persisted outlier analysis_time values for one (inst_num,
    gcwerks_channel_num) from hats.ng_chromatogram_qc -- the fast path used
    by cats_tagging.py's "baseline" algorithm, no chromatogram decoding.
    """
    rows = db.doquery(
        """
        SELECT analysis_time, period_id
        FROM hats.ng_chromatogram_qc
        WHERE inst_num = %s AND gcwerks_channel_num = %s AND algo_name = %s
          AND outlier = 1 AND analysis_time BETWEEN %s AND %s
        """,
        [inst_num, channel_number, ALGO_NAME, start, end],
    ) or []
    return pd.DataFrame(rows)


def build_baseline_tag_qc(
    batch,
    pnum: int,
    channel: str,
    start: str,
    end: str,
    **_ignored,
) -> pd.DataFrame:
    """cats_tagging.py-compatible wrapper: (batch, pnum, channel, start, end,
    **kwargs) -> DataFrame with an mf_num column, for ALGORITHMS registration.

    Reads already-persisted outlier timestamps from hats.ng_chromatogram_qc
    (written by a prior `cats_baseline_qc.py --site ... --channel ...` scan)
    -- no chromatogram decoding here, since that expensive work is meant to
    happen once per channel via this module's main(), not once per analyte
    every time cats_tagging.py runs. All scan-tuning parameters (neighbor
    window, thresholds, etc.) belong to that scan step, not this read step;
    **_ignored just future-proofs the signature against unrelated global
    cats_tagging.py flags being passed through generically. Every analyte on
    this channel shares the same flagged timestamps, so this wrapper is
    identical for any pnum on the channel -- cats_tagging.py's --analyte all
    is the normal way to cover them all.
    """
    channel_number = gcwerks_channel_number("cats", channel, site=batch.site)
    outliers = read_channel_outliers(batch.db, batch.inst_num, channel_number, start, end)
    if outliers.empty:
        return pd.DataFrame()
    flagged_times = pd.to_datetime(outliers["analysis_time"], utc=True)

    df = batch.load_data(pnum, channel=channel, start_date=start, end_date=end, verbose=False)
    if df.empty or "mf_num" not in df:
        return pd.DataFrame()
    df = df.copy()
    df["analysis_datetime"] = pd.to_datetime(df["analysis_datetime"], utc=True)

    matched = df.loc[df["analysis_datetime"].isin(set(flagged_times))]
    return matched


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--site", required=True, help="CATS site code (e.g. spo, brw).")
    p.add_argument("--channel", required=True,
                    help="DB channel letter (q, a, f, c/cc) or the physical GCwerks "
                         "channel number directly (0-3) -- one chromatogram scan "
                         "covers every analyte reported on this channel.")
    p.add_argument("--start", type=_parse_yyyymmdd, default=None,
                    help="Start date, YYYYMMDD (default: Jan 1 of the current year).")
    p.add_argument("--end", type=_parse_yyyymmdd, default=None,
                    help="End date, YYYYMMDD (default: now).")
    p.add_argument("--neighbor-window", type=int, default=10,
                    help="Number of chromatograms before/after used to build the "
                         "local median reference trace (default: 10).")
    p.add_argument("--peak-margin-min", type=float, default=0.3,
                    help="Minutes of margin added to each side of every analyte's "
                         "GCwerks peakid window before excluding it from the score "
                         "(default: 0.3).")
    p.add_argument("--ratio-threshold", type=float, default=0.05,
                    help="Absolute cutoff on |ratio_mean - 1|, the whole-trace "
                         "target/reference ratio averaged over all non-peak samples "
                         "(default: 0.05; see build_baseline_qc docstring).")
    p.add_argument("--max-gap-hours", type=float, default=1.0,
                    help="Max gap between flagged runs to merge into one period (default: 1.0).")
    p.add_argument("--workers", type=int, default=16,
                    help="Threads used to read+decode chromatograms (default: 16). "
                         "NFS reads are latency-bound, so overlapping them is the "
                         "single biggest speedup; use 1 to force serial reads.")
    p.add_argument("--output", type=Path, default=None,
                    help="Write all computed features (not just flagged rows) to this CSV.")
    p.add_argument("--dry-run", action="store_true",
                    help="Compute and print, but do not upsert to hats.ng_chromatogram_qc.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    start = args.start or pd.Timestamp(f"{datetime.today().year}-01-01", tz="UTC")
    end = args.end or pd.Timestamp(datetime.today(), tz="UTC")

    gc_dir = Path(f"/hats/gc/{GC_DIR_OVERRIDES.get(args.site.lower(), args.site)}")
    channel_number = gcwerks_channel_number("cats", args.channel, site=args.site)

    print(f"CATS-{args.site.upper()} channel {args.channel!r} (gcwerks channel{channel_number}) "
          f"{start.date()} -> {end.date()}")

    df = build_baseline_qc(
        gc_dir, channel_number, start, end,
        neighbor_window=args.neighbor_window,
        peak_margin_min=args.peak_margin_min,
        ratio_threshold=args.ratio_threshold,
        max_gap_hours=args.max_gap_hours,
        workers=args.workers,
        verbose=args.verbose,
    )
    if df.empty:
        print("No chromatograms found/scanned in that window (or not enough "
              "neighbor context available).")
        return 0

    print(f"Scanned {len(df)} chromatograms.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False, float_format="%.6g")
        print(f"Wrote all features to {args.output}")

    flagged = df.loc[df["baseline_outlier"]]
    print(f"Flagged {len(flagged)} run(s) across {flagged['period_id'].nunique()} period(s).")
    if not flagged.empty:
        cols = ["analysis_datetime", "period_id", "ratio_mean", "ratio_std", "path"]
        print(flagged[cols].to_string(index=False))

    if args.dry_run:
        print("--dry-run: not upserted to hats.ng_chromatogram_qc.")
        return 0

    batch = CATS_batch(args.site)
    n = upsert_chromatogram_qc(batch.db, batch.inst_num, channel_number, df)
    print(f"Upserted {n} scored rows into hats.ng_chromatogram_qc "
          f"(inst_num={batch.inst_num}, gcwerks_channel_num={channel_number}, algo={ALGO_NAME!r}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
