#!/usr/bin/env python3
"""Flag CATS chromatograms whose pre-peak shape deviates from their own
local neighbors' typical shape, and persist results to hats.ng_chromatogram_qc.

Reads raw chromatogram files directly (not DB mole-fraction rows). Each
chromatogram is normalized by dividing by the mean signal in its own
29.0-29.5 min tail window -- a quiet, settled region after all peaks have
eluted, so every run is expressed relative to its own steady-state level.

Design (Geoff, Aug 2026): rather than judging a run's pre-peak baseline
(0-5 min, before any analyte peak) against a fixed shape rule, build a
*local reference trace* from the pointwise MEDIAN of the N nearest
chromatograms before and after it (same channel), then measure how far the
target's own normalized trace deviates from that reference, point by point,
within the pre-peak window. This is deliberately scoped to the pre-peak
region: baseline changes there ("heart-cutting"-adjacent behavior) can occur
without touching the downstream analyte peaks at all, so a whole-trace
comparison would conflate two different questions.

Median (not mean) is required for the reference: manual review on CATS-SPO
channel0, 2026-07-17 showed a mean-based reference gets measurably dragged
toward a real anomaly when several of the N neighbors are themselves part of
the same multi-run contamination event (the run 260717.2145.2 saturated the
detector; its baseline elevation bled into ~2345.2 UTC that night) -- the
mean reference sat at ~1.2x baseline near the event instead of flat 1.0,
understating the anomaly, while the median stayed correctly flat at ~1.0
until roughly half the window was itself contaminated.

Validated against the 2026-07-17 SPO channel0 event: the root-cause run and
every affected run out to the recovery tail (~2345 UTC) score mean|diff|
0.2-1.4 in the pre-peak window, against ~0.001-0.06 for genuinely quiet runs
sampled the same day -- a clean order-of-magnitude separation, no z-score
tuning required. This also correctly *cleared* a run (260717.0843.6) that an
earlier slope/range-based prototype flagged as a false positive.

Known limitation: the local median reference degrades if a majority of the
N neighbors on one side are themselves anomalous (a long-duration or
back-to-back event) -- a median only resists a minority of outliers. Keep N
modest (default 10 each side) and review flagged clusters together rather
than trusting any single run's score in isolation.

This does NOT catch pure peak-height dropouts with an otherwise-normal
pre-peak baseline (a different, separately-seen failure mode) or anomalies
confined to the post-peak region (this only scores pre-peak by design, per
the heart-cutting rationale above) -- see cats_peak_qc.py for a peak-height
feature, or widen --pre-peak-min / add a symmetric post-peak scorer if
post-peak shape turns out to matter too.

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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "logosdata"))
from gcwerks_chromatogram import gcwerks_channel_number, read_gcwerks_chromatogram
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
    """Pointwise median of sub's rows via np.partition.

    A median only needs the middle one/two order statistics, not a full sort,
    so np.partition beats np.median here (~1.2x) -- worthwhile because this
    runs once per scored chromatogram.
    """
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
    """Score every row of a [n_chromatograms x n_pre_peak_samples] matrix
    against the pointwise median of its time-nearest neighbors.

    matrix rows must already be tail-normalized and sliced to the pre-peak
    window (see build_baseline_qc) -- slicing *before* the median matters:
    scoring only ever reads the pre-peak region, so medianing the full ~7100
    sample trace throws away ~83% of the work (measured 5.4x on a month of
    SPO channel0).

    The reference median deliberately EXCLUDES the target row (leave-one-out).
    An inclusive median (e.g. scipy.ndimage.median_filter over the same
    matrix) is marginally faster but lets a strongly anomalous run drag its
    own reference toward itself, damping the very score meant to catch it --
    always in the under-detection direction, and worst during the multi-run
    events this is built for.

    Returns (mean_abs_diff, max_abs_diff, n_neighbors) per row; rows with
    fewer than min_neighbors usable neighbors get NaN scores.
    """
    n_rows = len(matrix)
    mean_abs = np.full(n_rows, np.nan)
    max_abs = np.full(n_rows, np.nan)
    n_neighbors = np.zeros(n_rows, dtype=int)

    for i in range(n_rows):
        lo = max(0, i - neighbor_window)
        hi = min(n_rows, i + neighbor_window + 1)
        idxs = np.r_[lo:i, i + 1:hi]
        n_neighbors[i] = len(idxs)
        if len(idxs) < min_neighbors:
            continue
        diff = np.abs(matrix[i] - _neighbor_median(matrix[idxs]))
        mean_abs[i] = diff.mean()
        max_abs[i] = diff.max()

    return mean_abs, max_abs, n_neighbors


def build_baseline_qc(
    gc_dir: Path,
    channel_number: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    neighbor_window: int = 10,
    pre_peak_min: float = 5.0,
    diff_threshold: float = 0.15,
    max_gap_hours: float = 1.0,
    workers: int = 16,
    verbose: bool = False,
) -> pd.DataFrame:
    """Scan chromatograms in [start, end] and flag pre-peak shape anomalies
    against each run's own local (time-nearest) neighbors.

    Loads an extra neighbor_window runs of context on each side of the
    requested range so runs near the start/end of the window still get a
    full neighbor set; those context-only rows are dropped before scoring.

    diff_threshold is an absolute cutoff on pre_mean_abs_diff (fraction of
    the run's own tail-normalized baseline), not a z-score -- deliberately,
    since the whole point of the local-median design is to not need a
    separately-tuned statistical threshold. 0.15 sits roughly 3-10x above
    the ~0.001-0.06 range seen on quiet runs and well below the 0.2-1.4
    range seen on the validated 2026-07-17 event; adjust after reviewing
    --output on more data.

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
        print(f"  decoded {len(traces)} chromatograms; scoring...")

    # One [n_chromatograms x n_pre_peak_samples] matrix, sliced to the
    # pre-peak window up front (the only region scored). Rows are truncated
    # to the shortest trace so the array is rectangular -- run lengths vary
    # by a few samples (7118-7181 seen on SPO), well inside the pre-peak
    # window, so this never actually clips scored data.
    min_len = min(len(norm) for _m, norm, _t, _p in traces)
    minutes = traces[0][0][:min_len]
    pre_n = int((minutes <= pre_peak_min).sum())
    if pre_n < 1:
        return pd.DataFrame()
    matrix = np.array([norm[:pre_n] for _m, norm, _t, _p in traces])

    mean_abs, max_abs, n_neighbors = score_matrix(matrix, neighbor_window)

    timestamps = np.array([ts for _m, _n, ts, _p in traces])
    in_range = np.array([start <= ts <= end for ts in timestamps])
    scored = in_range & ~np.isnan(mean_abs)
    if not scored.any():
        return pd.DataFrame()

    df = pd.DataFrame({
        "analysis_datetime": timestamps[scored],
        "path": [str(traces[i][3]) for i in np.flatnonzero(scored)],
        "n_neighbors": n_neighbors[scored],
        "pre_mean_abs_diff": mean_abs[scored],
        "pre_max_abs_diff": max_abs[scored],
    }).sort_values("analysis_datetime").reset_index(drop=True)
    df["baseline_outlier"] = df["pre_mean_abs_diff"] > diff_threshold

    flagged_times = df.loc[df["baseline_outlier"], "analysis_datetime"]
    periods = _group_periods(flagged_times, max_gap_hours)
    df["period_id"] = 0
    for i, (pstart, pend) in enumerate(periods, start=1):
        in_period = df["analysis_datetime"].between(pstart, pend)
        df.loc[in_period, "period_id"] = i

    return df


ALGO_NAME = "baseline"


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
             pre_mean_abs_diff, pre_max_abs_diff, outlier, period_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            pre_mean_abs_diff = VALUES(pre_mean_abs_diff),
            pre_max_abs_diff = VALUES(pre_max_abs_diff),
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
                float(row.pre_mean_abs_diff), float(row.pre_max_abs_diff),
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
                    help="DB channel letter (q, a, f, c/cc) -- one chromatogram scan "
                         "covers every analyte reported on this channel.")
    p.add_argument("--start", type=_parse_yyyymmdd, default=None,
                    help="Start date, YYYYMMDD (default: Jan 1 of the current year).")
    p.add_argument("--end", type=_parse_yyyymmdd, default=None,
                    help="End date, YYYYMMDD (default: now).")
    p.add_argument("--neighbor-window", type=int, default=10,
                    help="Number of chromatograms before/after used to build the "
                         "local median reference trace (default: 10).")
    p.add_argument("--pre-peak-min", type=float, default=5.0,
                    help="Pre-peak window scored against the reference: first N "
                         "minutes of the run (default: 5).")
    p.add_argument("--diff-threshold", type=float, default=0.15,
                    help="Absolute cutoff on mean|deviation| from the local median "
                         "reference, as a fraction of tail-normalized baseline "
                         "(default: 0.15; see build_baseline_qc docstring).")
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

    gc_dir = Path(f"/hats/gc/{args.site}")
    channel_number = gcwerks_channel_number("cats", args.channel, site=args.site)

    print(f"CATS-{args.site.upper()} channel {args.channel!r} (gcwerks channel{channel_number}) "
          f"{start.date()} -> {end.date()}")

    df = build_baseline_qc(
        gc_dir, channel_number, start, end,
        neighbor_window=args.neighbor_window,
        pre_peak_min=args.pre_peak_min,
        diff_threshold=args.diff_threshold,
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
        cols = ["analysis_datetime", "period_id", "pre_mean_abs_diff", "pre_max_abs_diff", "path"]
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
