#!/usr/bin/env python3
"""Audit and optionally backfill missing M4 PFP CCGG event links.

The default mode is read-only.  Pass --apply to update only rows whose current
ccgg_event_num is NULL or zero and for which the resolver found a unique match.
"""

import argparse
from pathlib import Path

import pandas as pd

from m4_samplogs import M4_SampleLogs, parse_run_index_sample_info


def find_missing_pfp_events(instrument):
    sql = f"""
        SELECT num, analysis_time, run_time, port_info, flask_port
        FROM hats.ng_analysis
        WHERE inst_num = {instrument.inst_num}
          AND run_type_num = 5
          AND (ccgg_event_num IS NULL OR ccgg_event_num = 0)
        ORDER BY analysis_time;
    """
    return pd.DataFrame(instrument.db.doquery(sql))


def run_index_info_by_time(instrument):
    runs = instrument.load_runindex()
    return runs['info'].groupby(level=0).last().to_dict()


def audit_missing_pfp_events(instrument, max_lag_days=120):
    missing = find_missing_pfp_events(instrument)
    if missing.empty:
        return missing

    info_by_time = run_index_info_by_time(instrument)
    audit_rows = []
    for row in missing.to_dict('records'):
        analysis_time = pd.Timestamp(row['analysis_time'])
        source = parse_run_index_sample_info(info_by_time.get(analysis_time)) or {}
        lookup_row = {
            'samptype': 'pfp',
            'tank': row['port_info'],
            'site': source.get('site'),
            'sample_time': source.get('sample_time'),
            'dt_run': analysis_time,
        }
        match = instrument.resolve_pfp_event(
            lookup_row, max_lag_days=max_lag_days
        )
        audit_rows.append(
            {
                **row,
                'source_site': source.get('site'),
                'source_sample_date': source.get('sample_time'),
                'event_num': match['event_num'] if match else pd.NA,
                'event_id': match['event_id'] if match else pd.NA,
                'event_date': match['sample_date'] if match else pd.NaT,
                'event_site': match['site'] if match else pd.NA,
                'match_method': match['method'] if match else 'unresolved',
            }
        )
    return pd.DataFrame(audit_rows)


def apply_backfill(instrument, audit):
    matched = audit.loc[audit['event_num'].notna(), ['num', 'event_num']].copy()
    if matched.empty:
        return 0

    matched['num'] = matched['num'].astype(int)
    matched['event_num'] = matched['event_num'].astype(int)
    cases = ' '.join(
        f"WHEN {row.num} THEN {row.event_num}" for row in matched.itertuples()
    )
    analysis_nums = ', '.join(str(value) for value in matched['num'])
    sql = f"""
        UPDATE hats.ng_analysis
        SET ccgg_event_num = CASE num {cases} END
        WHERE inst_num = {instrument.inst_num}
          AND (ccgg_event_num IS NULL OR ccgg_event_num = 0)
          AND num IN ({analysis_nums});
    """
    instrument.db.doquery(sql)

    verify_sql = f"""
        SELECT COUNT(*) AS updated
        FROM hats.ng_analysis
        WHERE inst_num = {instrument.inst_num}
          AND num IN ({analysis_nums})
          AND ccgg_event_num IS NOT NULL
          AND ccgg_event_num <> 0;
    """
    return int(instrument.db.doquery(verify_sql)[0]['updated'])


def print_summary(audit):
    if audit.empty:
        print('No M4 PFP analyses have missing event links.')
        return

    print(audit['match_method'].value_counts(dropna=False).to_string())
    columns = [
        'analysis_time', 'port_info', 'source_site', 'source_sample_date',
        'event_id', 'event_site', 'event_date', 'match_method',
    ]
    print()
    print(audit[columns].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply', action='store_true',
        help='Write unambiguous matches to hats.ng_analysis (default: dry-run).',
    )
    parser.add_argument(
        '--max-lag-days', type=int, default=120,
        help='Maximum sample-to-analysis lag for package fallback (default: 120).',
    )
    parser.add_argument(
        '--output', type=Path,
        help='Optional CSV path for the complete audit.',
    )
    args = parser.parse_args()

    instrument = M4_SampleLogs()
    audit = audit_missing_pfp_events(instrument, args.max_lag_days)
    print_summary(audit)
    if args.output:
        audit.to_csv(args.output, index=False)
        print(f"Wrote audit to {args.output}")

    matched = int(audit['event_num'].notna().sum()) if not audit.empty else 0
    if args.apply:
        updated = apply_backfill(instrument, audit)
        print(f"Updated {updated} of {matched} matched rows.")
    else:
        print(f"Dry run: {matched} rows can be backfilled; no database changes made.")


if __name__ == '__main__':
    main()
