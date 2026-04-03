#!/usr/bin/env python3
"""
Extract sample loop temperature, pressure, and flow from IE3 engineering data
and upsert them into hats.ng_insitu_mole_fractions.

For each analysis in ng_insitu_analysis:
  - flow:     mean flow_samp over [-20, -15] s before analysis_time (SSV still on odd port)
  - temp:     temp_s1/s2/s3 (channel a/b/c) at 1 s before analysis_time
  - pressure: press_samp at 1 s before analysis_time

Engineering files from 2026-04-01 onward have an embedded CSV header row.
Earlier files use the legacy fallback header (eng_header_smo_early.txt).

Usage:
    python3 ie3_eng2db.py [--site smo|mlo|spo|brw] [--days 3] [--year 2026] [--all]
"""

import gzip
import io
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import typer

SITE_ROOT = Path('/hats/gc')
GSV_COLS = ('GSV1', 'GSV2', 'GSV3')
CHANNEL_TEMP = {'a': 'temp_s1', 'b': 'temp_s2', 'c': 'temp_s3'}
INST_NUM_BY_SITE = {'smo': 236}
VALID_SITES = {'smo', 'mlo', 'spo', 'brw'}
LEGACY_HEADER_NAME = 'eng_header_smo_early.txt'

app = typer.Typer(add_completion=False)


# ─── engineering data loading ─────────────────────────────────────────────────

def load_legacy_header(site: str) -> list[str] | None:
    """Load column names from the legacy header file (pre-embedded-header era)."""
    path = SITE_ROOT / site / LEGACY_HEADER_NAME
    if not path.exists():
        return None
    cols = [c.strip() for c in path.read_text().replace('\n', ',').split(',') if c.strip()]
    return cols or None


def _read_eng_file(path: Path, fallback_cols: list[str] | None) -> pd.DataFrame | None:
    """Read one eng CSV file.

    If the first line starts with 'ie3_time' it is treated as an embedded header
    row and used directly.  Otherwise fallback_cols is applied as the column list.
    """
    try:
        opener = gzip.open(path, 'rt') if path.name.endswith('.gz') else open(path, 'r')
        with opener as fh:
            raw = fh.read()
        first_line = raw[:raw.index('\n')] if '\n' in raw else raw
        if first_line.startswith('ie3_time'):
            return pd.read_csv(io.StringIO(raw), header=0, low_memory=False)
        elif fallback_cols:
            return pd.read_csv(io.StringIO(raw), header=None, names=fallback_cols,
                               low_memory=False)
        else:
            print(f'Warning: no header in {path} and no fallback cols available',
                  file=sys.stderr)
            return None
    except Exception as e:
        print(f'Warning: could not read {path}: {e}', file=sys.stderr)
        return None


def load_eng_data(site: str, dates: list[date]) -> pd.DataFrame | None:
    """Load engineering data for the given list of dates into a time-indexed DataFrame."""
    fallback_cols = load_legacy_header(site)
    frames = []
    for d in dates:
        day_dir = SITE_ROOT / site / d.strftime('%y') / 'incoming' / d.strftime('%Y%m%d')
        if day_dir.is_dir():
            for f in sorted(day_dir.glob('eng*.csv*')):
                df = _read_eng_file(f, fallback_cols)
                if df is not None:
                    frames.append(df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df['ie3_time'] = (pd.to_datetime(df['ie3_time'], format='mixed', utc=True, errors='coerce')
                       .dt.tz_convert(None))
    df = (df.dropna(subset=['ie3_time'])
            .drop_duplicates('ie3_time')
            .sort_values('ie3_time')
            .set_index('ie3_time'))

    for col in df.select_dtypes(include='object').columns:
        if col not in GSV_COLS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ─── measurement extraction ───────────────────────────────────────────────────

def _find_nearest_row(eng: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
    """Return the eng row closest to timestamp t."""
    if eng.empty:
        return None
    idx = eng.index.searchsorted(t)
    idx = min(idx, len(eng) - 1)
    return eng.iloc[idx]


def extract_measurements(eng: pd.DataFrame, analyses: pd.DataFrame) -> pd.DataFrame:
    """
    For each analysis in analyses (columns: analysis_num, analysis_time, port),
    extract flow/temp/pressure from eng data relative to the inject time.

    analysis_time is when all GSVs switch A→B (inj_all_channels / start of chromatogram).
    Since GCwerks rounds it to the minute, we find the precise A→B transition within ±2 min.

    Sequence relative to inject (t=0):
      t = -10 s : SSV switches to even (samp_flush_stop = chromduration - 10)
      t =  -1 s : sample temp/pressure measured (SSV sealed, loop pressurized)
      t =   0   : all GSVs A→B — start of chromatogram = analysis_time
      t = -20 to -15 s : flow_samp sampled (SSV still on odd port, ~50 cc/min)

    Measurements:
      - temp/pressure : nearest eng row at t_inject − 1 s
      - flow          : mean flow_samp over [t_inject − 20 s, t_inject − 15 s]

    Returns DataFrame with columns:
        analysis_num, channel, sample_loop_flow, sample_loop_temp, sample_loop_pressure
    """
    records = []

    for row in analyses.itertuples(index=False):
        analysis_num = row.analysis_num
        analysis_time = pd.Timestamp(row.analysis_time)

        # ── Find precise inject time: GSV A→B (all channels simultaneous) ────
        win = eng.loc[
            analysis_time - pd.Timedelta(minutes=2) :
            analysis_time + pd.Timedelta(minutes=2)
        ]

        t_inject = analysis_time  # fallback: use the (rounded) DB time
        inject_candidates = []
        for gcol in GSV_COLS:
            if gcol not in win.columns:
                continue
            gsv = win[gcol]
            a_to_b = win.index[(gsv == 'B') & (gsv.shift(1) == 'A')]
            if not a_to_b.empty:
                # Pick the candidate closest to analysis_time
                diffs = abs(a_to_b - analysis_time)
                inject_candidates.append(a_to_b[diffs.argmin()])

        if inject_candidates:
            # All GSVs switch simultaneously; use median to be robust to noise
            times_sec = [(t - analysis_time).total_seconds() for t in inject_candidates]
            t_inject = inject_candidates[int(len(inject_candidates) / 2)]

        # ── Temp and pressure: nearest row at t_inject − 1 s ─────────────────
        t_sample = t_inject - pd.Timedelta(seconds=1)
        ref_row = _find_nearest_row(eng.loc[:t_sample], t_sample)
        if ref_row is None:
            continue

        press_val = ref_row.get('press_samp')
        if pd.isna(press_val):
            press_val = None

        # ── Flow: mean flow_samp over [t_inject − 20 s, t_inject − 15 s] ─────
        flow_val = None
        if 'flow_samp' in eng.columns:
            flow_window = eng.loc[
                t_inject - pd.Timedelta(seconds=20) :
                t_inject - pd.Timedelta(seconds=15),
                'flow_samp'
            ]
            flow_series = pd.to_numeric(flow_window, errors='coerce').dropna()
            if not flow_series.empty:
                flow_val = float(flow_series.mean())

        # ── Emit one record per channel ───────────────────────────────────────
        for channel, temp_col in CHANNEL_TEMP.items():
            temp_val = ref_row.get(temp_col)
            records.append({
                'analysis_num': analysis_num,
                'channel': channel,
                'sample_loop_flow': None if (flow_val is None or flow_val != flow_val) else flow_val,
                'sample_loop_temp': None if (temp_val is None or pd.isna(temp_val)) else float(temp_val),
                'sample_loop_pressure': None if (press_val is None or pd.isna(press_val)) else float(press_val),
            })

    return pd.DataFrame(records)


# ─── database class ───────────────────────────────────────────────────────────

class IE3_Eng2DB:
    def __init__(self, site: str):
        site = site.lower()
        if site not in VALID_SITES:
            raise ValueError(f'Invalid site {site!r}. Valid: {sorted(VALID_SITES)}')
        self.site = site
        try:
            self.inst_num = INST_NUM_BY_SITE[site]
        except KeyError as e:
            raise ValueError(
                f'Missing inst_num for site {site!r}. Add to INST_NUM_BY_SITE.'
            ) from e

        sys.path.append('/ccg/src/db/')
        import db_utils.db_conn as db_conn  # type: ignore
        self.db = db_conn.HATS_ng()
        self.site_num = self._site_num()

    def _site_num(self) -> int:
        rows = self.db.doquery('SELECT num, code FROM gmd.site')
        df = pd.DataFrame(rows)
        return int(dict(zip(df['code'].str.lower(), df['num']))[self.site])

    def query_analyses(self, start: date, end: date) -> pd.DataFrame:
        """Return analysis_num, analysis_time, port for the given date range."""
        sql = """
            SELECT num AS analysis_num, analysis_time, port
            FROM hats.ng_insitu_analysis
            WHERE inst_num = %s AND site_num = %s
              AND analysis_time >= %s AND analysis_time < %s
            ORDER BY analysis_time
        """
        rows = self.db.doquery(sql, [
            self.inst_num, self.site_num,
            start.strftime('%Y-%m-%d'), (end + timedelta(days=1)).strftime('%Y-%m-%d'),
        ])
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['analysis_num', 'analysis_time', 'port']
        )

    def upsert(self, measurements: pd.DataFrame, batch_size: int = 500):
        """UPDATE ng_insitu_mole_fractions rows with the extracted measurements."""
        if measurements.empty:
            return

        sql = """
            UPDATE hats.ng_insitu_mole_fractions
            SET sample_loop_temp     = %s,
                sample_loop_pressure = %s,
                sample_loop_flow     = %s
            WHERE analysis_num = %s AND channel = %s
        """

        def _db_val(v):
            """Convert NaN/NA to None for the MySQL driver."""
            try:
                return None if pd.isna(v) else v
            except (TypeError, ValueError):
                return v

        params = [
            (_db_val(r.sample_loop_temp), _db_val(r.sample_loop_pressure),
             _db_val(r.sample_loop_flow), int(r.analysis_num), r.channel)
            for r in measurements.itertuples(index=False)
        ]
        for i in range(0, len(params), batch_size):
            self.db.doMultiInsert(sql, params[i:i + batch_size], all=True)

    def run(self, start: date, end: date):
        """Process analyses in [start, end] month by month."""
        # Build list of months in range
        months = []
        cur = date(start.year, start.month, 1)
        while cur <= end:
            months.append(cur)
            # advance one month
            if cur.month == 12:
                cur = date(cur.year + 1, 1, 1)
            else:
                cur = date(cur.year, cur.month + 1, 1)

        total_updated = 0
        for month_start in months:
            if month_start.month == 12:
                month_end = date(month_start.year + 1, 1, 1) - timedelta(days=1)
            else:
                month_end = date(month_start.year, month_start.month + 1, 1) - timedelta(days=1)

            period_start = max(start, month_start)
            period_end = min(end, month_end)

            analyses = self.query_analyses(period_start, period_end)
            if analyses.empty:
                continue

            print(f'  {period_start} – {period_end}: {len(analyses)} analyses')

            # Load eng data for the period + 1-day buffer on each side
            eng_dates = [
                period_start - timedelta(days=1) + timedelta(days=i)
                for i in range((period_end - period_start).days + 3)
            ]
            eng = load_eng_data(self.site, eng_dates)
            if eng is None:
                print(f'  No engineering data found for {period_start} – {period_end}', file=sys.stderr)
                continue

            measurements = extract_measurements(eng, analyses)
            if measurements.empty:
                print(f'  No measurements extracted', file=sys.stderr)
                continue

            self.upsert(measurements)
            total_updated += len(measurements)

        print(f'Done. Updated {total_updated} rows in ng_insitu_mole_fractions.')


# ─── CLI ──────────────────────────────────────────────────────────────────────

@app.command()
def main(
    site: str = typer.Option('smo', '--site', help='Station code (smo|mlo|spo|brw)'),
    days: int = typer.Option(3, '--days', help='Process the past N days (default: 3)'),
    year: int | None = typer.Option(None, '--year', help='Process a full year (YYYY)'),
    all_data: bool = typer.Option(False, '--all', help='Process all years'),
):
    """Upsert sample loop temp/pressure/flow into ng_insitu_mole_fractions from eng data."""
    eng2db = IE3_Eng2DB(site)
    today = date.today()

    if all_data:
        # Find the earliest analysis in the DB
        rows = eng2db.db.doquery(
            'SELECT MIN(analysis_time) AS min_t FROM hats.ng_insitu_analysis '
            'WHERE inst_num = %s AND site_num = %s',
            [eng2db.inst_num, eng2db.site_num],
        )
        min_t = rows[0]['min_t'] if rows else None
        if min_t is None:
            print('No analyses found in database.', file=sys.stderr)
            raise typer.Exit(1)
        start = min_t.date() if hasattr(min_t, 'date') else min_t
        end = today
        print(f'Processing all data: {start} – {end}')
    elif year is not None:
        start = date(year, 1, 1)
        end = date(year, 12, 31)
        print(f'Processing year {year}: {start} – {end}')
    else:
        end = today
        start = today - timedelta(days=days - 1)
        print(f'Processing past {days} days: {start} – {end}')

    eng2db.run(start, end)


if __name__ == '__main__':
    app()
