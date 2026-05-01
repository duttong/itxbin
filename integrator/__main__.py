"""
CLI entry point:  python -m integrator <command> [options]

Commands
--------
run       Free peak detection on one run directory
matrix    Free peak detection summary across many runs
discover  Discover nominal RTs from a batch of runs
integrate Windowed integration using nominal RTs from a peak config file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from integrator.run_loader import RunLoader
from integrator.matrix import ChromMatrix
from integrator.peak_list import PeakList
from integrator.pipeline import batch_integrate
from integrator.rt_align import discover_reference_rt, build_shift_table
from integrator.store import ChromStore


# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """Integrate every injection in one run directory."""
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        sys.exit(f'Not a directory: {run_dir}')

    exclude = tuple(args.exclude_ports.split(','))
    loader = RunLoader(run_dir)
    injections = loader.load(
        exclude_ports=exclude,
        smooth=args.smooth,
        sg_winsize=args.sg_winsize,
        sg_order=args.sg_order,
    )

    if not injections:
        print('No injections found (check --exclude-ports).')
        return

    for port in sorted(injections, key=lambda p: int(p)):
        chroms = injections[port]
        label = loader.port_name(port)
        print(f'\nPort {port}  ({label})  — {chroms[0].name}')
        for ch_idx, chrom in enumerate(chroms):
            peaks = chrom.detect_peaks(
                min_width_sec=args.min_width,
                distance_sec=args.distance,
                prominence_factor=args.prominence,
            )
            print(f'  ch{ch_idx}: {len(peaks)} peak(s)')
            for pk in peaks:
                print(
                    f'    t={pk.center_time:7.2f}s  '
                    f'height={pk.height:>14,.0f}  '
                    f'area={pk.area:>14,.1f}'
                )


def cmd_matrix(args: argparse.Namespace) -> None:
    """Load many runs and print a peak-area summary table."""
    incoming = Path(args.incoming)
    if not incoming.is_dir():
        sys.exit(f'Not a directory: {incoming}')

    exclude = tuple(args.exclude_ports.split(','))
    mat = ChromMatrix.from_directory(
        incoming,
        channel=args.channel,
        exclude_ports=exclude,
        smooth=args.smooth,
        limit=args.limit,
    )
    print(f'Loaded {len(mat)} injections  (channel {args.channel})\n')

    df = mat.peak_table(
        min_width_sec=args.min_width,
        distance_sec=args.distance,
        prominence_factor=args.prominence,
    )

    if df.empty:
        print('No peaks detected.')
        return

    # Group by approximate retention time bin (round to nearest 5 s)
    df['rt_bin'] = (df['center_time'] / 5).round() * 5

    print(df[['label', 'center_time', 'height', 'area']].to_string(index=False))
    print()
    print('Peak summary by retention-time bin:')
    summary = (
        df.groupby('rt_bin')[['center_time', 'height', 'area']]
        .agg(['mean', 'std', 'count'])
        .round(2)
    )
    print(summary.to_string())


# ------------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------------

def cmd_discover(args: argparse.Namespace) -> None:
    """
    Discover nominal retention times for each compound by running auto
    peak detection across a batch of injections, then match detected peaks
    to the peak.list compounds by channel and elution order.
    """
    incoming = Path(args.incoming)
    if not incoming.is_dir():
        sys.exit(f'Not a directory: {incoming}')

    pl = PeakList.from_file(Path(args.peak_list))
    exclude = tuple(args.exclude_ports.split(','))

    print(f'Peak list: {args.peak_list}')
    print(f'Channels: {pl.channels()}')
    print()

    for ch in pl.channels():
        compounds = pl.by_channel(ch)
        mat = ChromMatrix.from_directory(
            incoming,
            channel=ch,
            exclude_ports=exclude,
            smooth=True,
            limit=args.limit,
        )
        if len(mat) == 0:
            print(f'ch{ch}: no injections found')
            continue

        # Collect per-injection peak lists, filtering noise by area fraction
        all_peaks = []
        chroms = [c for _, _, c in mat._rows]
        for chrom in chroms:
            raw = chrom.detect_peaks(
                min_width_sec=args.min_width,
                distance_sec=args.distance,
                prominence_factor=args.prominence,
            )
            if raw:
                max_area = max(p.area for p in raw)
                cutoff = args.min_area_pct / 100.0 * max_area
                raw = [p for p in raw if p.area >= cutoff]
            all_peaks.append(raw)

        # Filter to injections with at least as many peaks as compounds
        n_expected = len(compounds)
        good = [(c, p) for c, p in zip(chroms, all_peaks) if len(p) >= n_expected]

        print(f'ch{ch}: {len(mat)} injections, {len(good)} with ≥{n_expected} peaks '
              f'(area ≥{args.min_area_pct}% of max)')

        if not good:
            print(f'  (no injections qualified — try --min-area-pct lower or --prominence lower)')
            continue

        # Match by elution order: take the top-N peaks by area from each injection
        # (avoids mismatches from small interstitial noise peaks), then sort by time.
        # peak.list order = GC elution order.
        per_compound: list[list[tuple[float, float]]] = [[] for _ in compounds]  # [(rt, area)]
        for chrom, peaks in good:
            ranked = sorted(peaks, key=lambda p: p.area, reverse=True)[:n_expected]
            ranked.sort(key=lambda p: p.center_time)   # restore time order
            for comp_idx, pk in enumerate(ranked):
                if comp_idx < len(compounds):
                    per_compound[comp_idx].append((pk.center_time, pk.area))

        for comp_idx, compound in enumerate(compounds):
            entries = per_compound[comp_idx]
            if entries:
                rts = [e[0] for e in entries]
                areas = [e[1] for e in entries]
                median_rt = float(np.median(rts))
                std_rt = float(np.std(rts))
                median_area = float(np.median(areas))
                compound.nominal_rt = median_rt
                flag = '  ⚠ high σ' if std_rt > 5.0 else ''
                print(
                    f'  {compound.report_name:<12}  RT={median_rt:6.1f}s  '
                    f'σ={std_rt:.2f}s  area={median_area:,.0f}{flag}'
                )
            else:
                print(f'  {compound.report_name:<12}  — not detected')

        # Show RT shift stats using reference peak (largest by area)
        if good:
            ref_peaks_per_inj = [p for _, p in good]
            ref_chroms = [c for c, _ in good]
            try:
                ref_rt, ref_idx = discover_reference_rt(ref_chroms, ref_peaks_per_inj)
                ref_name = compounds[ref_idx].report_name if ref_idx < len(compounds) else '?'
                shifts = build_shift_table(ref_chroms, ref_rt)
                print(
                    f'  RT ref: {ref_name} at {ref_rt:.1f}s  '
                    f'shift range [{min(shifts):.2f}, {max(shifts):.2f}]s'
                )
            except (ValueError, IndexError):
                pass
        print()


def cmd_integrate(args: argparse.Namespace) -> None:
    """
    Windowed integration using nominal RTs from a peak config file.

    For each injection in the run directory, integrates only the compounds
    listed in the config file by searching within ±window of each compound's
    nominal RT.  No free peak detection — only peak-list compounds are reported.
    """
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        sys.exit(f'Not a directory: {run_dir}')

    pl = PeakList.from_file(Path(args.peak_config))

    # Verify all compounds have nominal RTs
    missing = [c.report_name for c in pl.compounds if c.nominal_rt is None]
    if missing:
        sys.exit(f'Missing nominal RT for: {", ".join(missing)}\n'
                 f'Use a 5-column config file (see integrator/fe3_peaks.conf) or run discover first.')

    exclude = tuple(args.exclude_ports.split(','))
    loader = RunLoader(run_dir)
    injections = loader.load(
        exclude_ports=exclude,
        smooth=args.smooth,
        sg_winsize=args.sg_winsize,
        sg_order=args.sg_order,
    )

    if not injections:
        print('No injections found (check --exclude-ports).')
        return

    # Header
    compound_names = [c.report_name for c in pl.compounds]
    print(f'{"Port":<6}  {"Tank":<14}  {"Name":<22}', end='')
    for name in compound_names:
        print(f'  {name:>12}', end='')
    print()
    print('-' * (44 + 14 * len(compound_names)))

    for port in sorted(injections, key=lambda p: int(p)):
        chroms = injections[port]
        tank = loader.port_name(port)
        inj_name = chroms[0].name

        areas: list[str] = []
        for compound in pl.compounds:
            ch = compound.channel
            if ch >= len(chroms):
                areas.append('         —')
                continue
            chrom = chroms[ch]
            if compound.meth == 2:
                pk = chrom.integrate_tangent_skim(
                    center_rt=compound.nominal_rt,
                    win_lo=compound.rt_window_lo,
                    win_hi=compound.rt_window_hi,
                )
            else:
                pk = chrom.integrate_window(
                    center_rt=compound.nominal_rt,
                    win_lo=compound.rt_window_lo,
                    win_hi=compound.rt_window_hi,
                )
            if pk is None:
                areas.append('         —')
            else:
                areas.append(f'{pk.area:>12,.0f}')

        print(f'{port:<6}  {tank:<14}  {inj_name:<22}', end='')
        for a in areas:
            print(f'  {a}', end='')
        print()


def cmd_batch(args: argparse.Namespace) -> None:
    """
    Batch integration: load all injections from the HDF5 store, apply
    RT alignment per channel, integrate each compound, and write a CSV.
    """
    store = ChromStore(args.store)
    pl = PeakList.from_file(Path(args.peak_config))

    missing = [c.report_name for c in pl.compounds if c.nominal_rt is None]
    if missing:
        sys.exit(f'Missing nominal RT for: {", ".join(missing)}\n'
                 f'Use a 5- or 6-column config file (see integrator/fe3_peaks.conf).')

    exclude = tuple(args.exclude_ports.split(','))

    df = batch_integrate(
        store=store,
        peak_list=pl,
        smooth=args.smooth,
        smooth_method=args.smooth_method,
        sg_winsize=args.sg_winsize,
        sg_order=args.sg_order,
        box_winsize=args.box_winsize,
        exclude_ports=exclude,
        align=not args.no_align,
        verbose=True,
    )

    if args.output:
        df.to_csv(args.output, index=False)
        print(f'Results written to {args.output}')
    else:
        # Print a compact summary table to stdout
        compound_cols = [c.report_name for c in pl.compounds]
        print()
        print(df[['name', 'port', 'tank'] + compound_cols].to_string(index=False))


def cmd_store(args: argparse.Namespace) -> None:
    """Dispatch store sub-commands: build, update, info."""
    if args.store_cmd == 'build':
        exclude = tuple(args.exclude_ports.split(','))
        ChromStore.build(
            path=args.output,
            incoming_dir=args.incoming,
            exclude_ports=exclude,
            limit=args.limit,
            verbose=True,
        )

    elif args.store_cmd == 'update':
        exclude = tuple(args.exclude_ports.split(','))
        store = ChromStore(args.store)
        n = store.update(args.incoming, exclude_ports=exclude, verbose=True)
        print(f'{n} injections added.')

    elif args.store_cmd == 'info':
        store = ChromStore(args.store)
        print(store.info())


def _shared_peak_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--min-width', type=float, default=2.0, metavar='SEC',
                   help='Minimum peak width in seconds (default 2.0)')
    p.add_argument('--distance', type=float, default=15.0, metavar='SEC',
                   help='Minimum peak separation in seconds (default 15.0)')
    p.add_argument('--prominence', type=float, default=5.0,
                   help='Prominence factor × noise std (default 5.0)')


def _shared_smooth_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--no-smooth', dest='smooth', action='store_false', default=True,
                   help='Disable Savitzky-Golay smoothing')
    p.add_argument('--sg-winsize', type=int, default=61,
                   help='SG smoothing window in points (default 61)')
    p.add_argument('--sg-order', type=int, default=4,
                   help='SG polynomial order (default 4)')


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m integrator',
        description='GC chromatogram integrator (replaces GCwerks integration)',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # -- run --
    p_run = sub.add_parser('run', help='Integrate a single run directory')
    p_run.add_argument('run', help='Path to run directory')
    p_run.add_argument('--exclude-ports', default='10', metavar='PORTS',
                       help='Comma-separated port numbers to exclude (default 10)')
    _shared_smooth_args(p_run)
    _shared_peak_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # -- matrix --
    p_mat = sub.add_parser('matrix', help='Summarise peaks across many runs')
    p_mat.add_argument('incoming', help='Path to incoming/ directory (contains run sub-dirs)')
    p_mat.add_argument('--channel', type=int, default=0,
                       help='Detector channel to analyse (default 0)')
    p_mat.add_argument('--exclude-ports', default='10', metavar='PORTS',
                       help='Comma-separated port numbers to exclude (default 10)')
    p_mat.add_argument('--limit', type=int, default=None,
                       help='Cap number of run directories loaded (for testing)')
    _shared_smooth_args(p_mat)
    _shared_peak_args(p_mat)
    p_mat.set_defaults(func=cmd_matrix)

    # -- discover --
    p_dis = sub.add_parser('discover', help='Discover nominal RTs from a batch of runs')
    p_dis.add_argument('incoming', help='Path to incoming/ directory')
    p_dis.add_argument('--peak-list', default='/hats/gc/fe3/config/peak.list',
                       help='Path to peak.list (default: FE3)')
    p_dis.add_argument('--exclude-ports', default='10', metavar='PORTS',
                       help='Comma-separated port numbers to exclude (default 10)')
    p_dis.add_argument('--limit', type=int, default=20,
                       help='Max run directories to load (default 20)')
    p_dis.add_argument('--min-area-pct', type=float, default=0.1,
                       help='Minimum peak area as %% of largest peak (filters noise, default 0.1)')
    _shared_smooth_args(p_dis)
    _shared_peak_args(p_dis)
    p_dis.set_defaults(func=cmd_discover)

    # -- integrate --
    _default_conf = str(Path(__file__).parent / 'fe3_peaks.conf')
    p_int = sub.add_parser('integrate', help='Windowed integration using nominal RTs from config')
    p_int.add_argument('run', help='Path to run directory')
    p_int.add_argument('--peak-config', default=_default_conf,
                       help=f'5-column peak config file (default: fe3_peaks.conf)')
    p_int.add_argument('--exclude-ports', default='10', metavar='PORTS',
                       help='Comma-separated port numbers to exclude (default 10)')
    _shared_smooth_args(p_int)
    p_int.set_defaults(func=cmd_integrate)

    # -- batch --
    _default_conf = str(Path(__file__).parent / 'fe3_peaks.conf')
    p_bat = sub.add_parser(
        'batch',
        help='Batch integrate all injections from a store with RT alignment',
    )
    p_bat.add_argument('store', help='Path to HDF5 store file (.h5)')
    p_bat.add_argument('--peak-config', default=_default_conf,
                       help='6-column peak config with ref flags (default: fe3_peaks.conf)')
    p_bat.add_argument('--exclude-ports', default='10', metavar='PORTS',
                       help='Comma-separated port numbers to exclude (default 10)')
    p_bat.add_argument('--no-align', action='store_true',
                       help='Disable RT alignment (use nominal RTs directly)')
    p_bat.add_argument('--output', metavar='FILE',
                       help='Write results to CSV (default: print summary to stdout)')
    p_bat.add_argument('--smooth-method', choices=['sg', 'box'], default='sg',
                       help='Smoothing method: sg (Savitzky-Golay, default) or box (moving average)')
    p_bat.add_argument('--box-winsize', type=int, default=25,
                       help='Box-filter window in points (default 25, used when --smooth-method=box)')
    _shared_smooth_args(p_bat)
    p_bat.set_defaults(func=cmd_batch)

    # -- store --
    p_st = sub.add_parser('store', help='Build or update the persistent HDF5 chromatogram store')
    st_sub = p_st.add_subparsers(dest='store_cmd', required=True)

    p_st_build = st_sub.add_parser('build', help='Build a new store from an incoming/ directory')
    p_st_build.add_argument('incoming', help='Path to incoming/ directory')
    p_st_build.add_argument('output', help='Destination .h5 file path')
    p_st_build.add_argument('--exclude-ports', default='10', metavar='PORTS',
                            help='Comma-separated port numbers to exclude (default 10)')
    p_st_build.add_argument('--limit', type=int, default=None,
                            help='Cap number of run directories (for testing)')

    p_st_update = st_sub.add_parser('update', help='Append new runs to an existing store')
    p_st_update.add_argument('store', help='Path to existing .h5 store file')
    p_st_update.add_argument('incoming', help='Path to incoming/ directory')
    p_st_update.add_argument('--exclude-ports', default='10', metavar='PORTS',
                             help='Comma-separated port numbers to exclude (default 10)')

    p_st_info = st_sub.add_parser('info', help='Print store statistics')
    p_st_info.add_argument('store', help='Path to .h5 store file')

    p_st.set_defaults(func=cmd_store)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
