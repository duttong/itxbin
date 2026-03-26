#! /usr/bin/env python

import argparse
import datetime as dt
import logging
import os
import shutil
import tarfile
from datetime import date
from pathlib import Path, PurePosixPath

from gcwerks_import import GCwerks_Import


class IE3_import(GCwerks_Import):

    def __init__(self, site, args, incoming_dir='incoming'):
        super().__init__(site, args, incoming_dir)
        self.source = Path(self.options.get('source', '/nfs/isftp/sftp/data/logos/smo/incoming'))
        self.dest_root = Path(self.options.get('dest_root', '/hats/gc/smo'))
        self.past_days = int(self.options.get('past_days', 2))
        self.target_date = self.options.get('target_date')

    @staticmethod
    def _parse_yyyymmdd(value):
        if value in (None, ''):
            return None
        try:
            return dt.datetime.strptime(str(value), '%Y%m%d').date()
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid date '{value}'. Expected YYYYMMDD."
            ) from exc

    @staticmethod
    def _is_within(base: Path, target: Path) -> bool:
        return os.path.commonpath([str(base), str(target)]) == str(base)

    @staticmethod
    def _remove_path(path: Path):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()

    @staticmethod
    def _safe_member_parts(member_name: str):
        p = PurePosixPath(member_name)
        if p.is_absolute():
            return None
        parts = tuple(part for part in p.parts if part not in ('', '.'))
        if not parts:
            return None
        if any(part == '..' for part in parts):
            return None
        return parts

    @staticmethod
    def _selected_dates(target_date, past_days):
        if target_date is not None:
            return [target_date]
        today = dt.date.today()
        return [today - dt.timedelta(days=offset) for offset in range(past_days)]

    def _extract_replace(self, tgz_path: Path, dest_incoming: Path):
        base = dest_incoming.resolve()
        with tarfile.open(tgz_path, 'r:gz') as tf:
            members = []
            top_level = set()
            for m in tf.getmembers():
                parts = self._safe_member_parts(m.name)
                if parts is None:
                    logging.warning('Skipping unsafe tar member: %s', m.name)
                    continue
                if m.issym() or m.islnk():
                    logging.warning('Skipping link member: %s', m.name)
                    continue

                target = (base / Path(*parts)).resolve()
                if not self._is_within(base, target):
                    logging.warning('Skipping path traversal member: %s', m.name)
                    continue

                top_level.add(parts[0])
                members.append(m)

            for name in sorted(top_level):
                existing = base / name
                if existing.exists() or existing.is_symlink():
                    logging.info('Removing existing extracted path: %s', existing)
                    self._remove_path(existing)

            logging.info('Extracting %d members from %s', len(members), tgz_path)
            tf.extractall(path=dest_incoming, members=members)

    def _remove_last_itx(self, extracted_dir: Path):
        """Remove the most recent *.itx* file from one extracted YYYYMMDD directory."""
        if not extracted_dir.is_dir():
            logging.info('No extracted directory found at %s', extracted_dir)
            return
        try:
            last_file = max(extracted_dir.rglob('*.itx*'), key=lambda p: p.stat().st_mtime)
        except ValueError:
            logging.info('No *.itx* files found in %s', extracted_dir)
            return
        logging.info('Removing partial itx file: %s', last_file)
        self._remove_path(last_file)

    def sync_incoming(self):
        if self.past_days < 1:
            raise ValueError('--past-days must be >= 1')
        if not self.source.is_dir():
            logging.warning('Source directory does not exist: %s', self.source)
            return

        days = sorted(set(self._selected_dates(self.target_date, self.past_days)))
        logging.info('SMO incoming sync for %d day(s): %s',
                     len(days), ', '.join(f'{d:%Y-%m-%d}' for d in days))

        for day in days:
            fname = f'{day:%Y%m%d}.tgz'
            src = self.source / fname
            if not src.exists():
                logging.info('Missing source tarball for %s: %s', day, src)
                continue

            yy = day.strftime('%y')
            incoming = self.dest_root / yy / 'incoming'
            incoming.mkdir(parents=True, exist_ok=True)

            copied_tgz = incoming / fname
            logging.info('Copying %s -> %s', src, copied_tgz)
            shutil.copy2(src, copied_tgz)
            self._extract_replace(copied_tgz, incoming)
            logging.info('Removing copied tarball: %s', copied_tgz)
            copied_tgz.unlink(missing_ok=True)
            self._remove_last_itx(incoming / f'{day:%Y%m%d}')

    def main(self, import_method, *args, **kwargs):
        self.sync_incoming()
        super().main(import_method, *args, **kwargs)

    def should_skip_itx(self, itx):
        return itx.has_note_flag('SKIP')


if __name__ == '__main__':

    yyyy = date.today().year
    SGwin, SGorder = 81, 4      # Savitzky Golay default variables
    WSTART = -1
    BOXWIDTH = 11  # Default box width for smoothing
    O2_LOCK_CHANS = [0,1,2]
    O2_LOCK_TIME = [60, 70, 60] # Retention times for O2 leading edge in seconds
    site = 'smo'

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the FE3 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default is False)')
    parser.add_argument('-W', action="store", dest='ws_start', default=WSTART,
        help='Apply wide spike filter (default off)')
    parser.add_argument('-b', action='store', dest='boxwidth', metavar='Win', type=int, default=BOXWIDTH,
        help=f'Apply a Box smooth with window width (default = {BOXWIDTH})')
    parser.add_argument('-lock', action='store_true', default=True,
        help='Lock the oxygen peak to a specific retention time (default is False)')
    parser.add_argument('-g', action='store_true', default=False,
        help='Apply Savitzky Golay smoothing (default is False)')
    parser.add_argument('-gw', action='store', dest='SGwin', metavar='Win',
        default=SGwin, type=int,
        help=f'Sets Savitzky Golay smoothing window (default = {SGwin} points)')
    parser.add_argument('-go', action='store', dest='SGorder', metavar='Order',
        default=SGorder, type=int,
        help=f'Sets Savitzky Golay order of fit (default = {SGorder})')
    parser.add_argument('-year', action='store', default=yyyy,
        help=f'Which year? (default is {yyyy})')
    parser.add_argument('-reimport', action='store_true', default=False,
        help='Reimport all itx files including .Z archived.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--past-days', type=int, default=2, dest='past_days',
        help='Process today and previous N-1 days from SMO incoming .tgz files (default: 2)')
    group.add_argument('--date', action='store', type=IE3_import._parse_yyyymmdd,
        dest='target_date', default=None,
        help='Process one specific SMO incoming tarball date YYYYMMDD')
    parser.add_argument('--source', action='store', default='/nfs/isftp/sftp/data/logos/smo/incoming',
        help='Source directory containing YYYYMMDD.tgz files')
    parser.add_argument('--dest-root', action='store', default='/hats/gc/smo',
        help='Destination root containing YY/incoming directories')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='Enable verbose logging.')
    parser.add_argument('site', nargs='?', default=site,
        help=f'Valid station code (default is {site})')

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
    )
    args.O2_LOCK_CHANS = O2_LOCK_CHANS
    args.O2_LOCK_TIMES = O2_LOCK_TIME

    ie3 = IE3_import(args.site, args)
    if args.reimport:
        types = ('*.itx', '*.itx.gz', '*.itx.Z')
        ie3.main(import_method=ie3.import_recursive_itx, types=types)
    else:
        ie3.main(import_method=ie3.import_recursive_itx)
