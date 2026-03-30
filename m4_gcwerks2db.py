#!/usr/bin/env python

import pandas as pd
import argparse
from datetime import datetime, timedelta

from logos_instruments import M4_Instrument

class M4_GCwerks(M4_Instrument):

    def __init__(self, flagged=False):
        super().__init__()
        self.flagged = flagged
        self.gcwerks_results = self.export_dir  # Path type
        self.sites = self.gml_sites()           # site codes and numbers

    def load_gcwerks(self, gas, t_start, t_stop='end'): 
        """ Loads GCwerks data for a given gas and returns it as a DataFrame.
            The data files should be stored at self.gcwerks_results path.
            Times should be in YYMM format.
        """
        start_dt = pd.to_datetime(t_start, format='%y%m')
        if t_stop == 'end':
            stop_dt = pd.to_datetime(datetime.today())
        else:
            stop_dt = pd.to_datetime(t_stop, format='%y%m') + pd.offsets.MonthEnd(0)

        if start_dt < pd.to_datetime(self.start_date):
            #print(f'Start date "{t_start}" selected was too early, using "{self.start_date}"')
            start_dt = pd.to_datetime(self.start_date)
        
        if gas not in self.molecules:
            print(f'Incorrect gas {gas} name.')
            return

        suffix = '_flagged' if self.flagged else ''
        file = self.gcwerks_results / f'data_{gas}{suffix}.csv'
        if file.exists():
            if self.flagged:
                df = pd.read_csv(file, na_values=[' nan'], dtype=str)
            else:
                df = pd.read_csv(file, na_values=[' nan'])
            df.columns = [col.replace(f'{gas}_', '').strip() for col in df.columns]
        else:
            print(f'Missing GCwerks output file: {file}')
            return

        if self.flagged:
            df['gcwerks_flag'] = False
            for col in ('area', 'ht', 'rt'):
                if col not in df.columns:
                    continue
                series = df[col].fillna('').astype(str).str.strip()
                flagged = series.str.endswith(('F', '*'))
                cleaned = series.str.replace(r'[F*]$', '', regex=True)
                df[col] = pd.to_numeric(cleaned, errors='coerce')
                df['gcwerks_flag'] = df['gcwerks_flag'] | flagged

        # Make a datetime column and set it as the index
        df = df.assign(dt_run=pd.to_datetime(df.time))
        df.set_index('dt_run', inplace=True, drop=False)

        # trim the dataframe to use start_date (start_dt is the datetime value)
        if t_stop == 'end':
            df = df.loc[start_dt:]
        else:
            df = df.loc[start_dt:stop_dt]

        # trim the last two rows if area and height are nan (this happens when the chrom is not finished)
        if pd.isna(df.iloc[-2]['area']) & pd.isna(df.iloc[-2]['ht']):
            df = df[:-2]
        elif pd.isna(df.iloc[-1]['area']) & pd.isna(df.iloc[-1]['ht']):
            df = df[:-1]

        # insure there are no duplicate index rows
        df = df.reset_index(drop=True)

        # parameter and event numbers
        df['pnum'] = self.analytes[gas].strip()
        
        df = self.return_analysis_nums(df)       # add analysis_num to dataframe
        
        return df
    
    def insert_mole_fractions(self, df):
        """
        Inserts or updates rows in hats.ng_mole_fractions using a batch upsert.
        This function uses db.doMultiInsert to perform the insertions.
        NOTE: detrend_method_num is hardcoded to 2 (Lowess) and qc_status is set to "P" (Preliminary).
        """
        sql_insert = """
            INSERT INTO hats.ng_mole_fractions (
                analysis_num,
                parameter_num,
                area,
                height,
                retention_time,
                detrend_method_num,
                qc_status,
                flag
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                flag           = IF(VALUES(flag) = 'W..', 'W..', flag),
                area           = VALUES(area),
                height         = VALUES(height),
                retention_time = VALUES(retention_time)
            """

        df = df.fillna('')
        params = []

        for _, row in df.iterrows():
            p = (
                row.analysis_num,    # analysis_num
                row.pnum,            # parameter_num
                row.area,            # area
                row.ht,              # height
                row.rt,              # retention time
                2,                   # detrend method number (default to 2 = Lowess)
                "P",                 # Preliminary QC status flag
                "W.." if bool(getattr(row, 'gcwerks_flag', False)) else "..."
            )
            params.append(p)

            if self.db.doMultiInsert(sql_insert, params): 
                params=[]

        # Process any remaining rows in the final batch:
        self.db.doMultiInsert(sql_insert, params, all=True)

    def flag_first_reference_run(self, start_time, end_time):
        """Flag the first reference run in each run_time group after mole fractions exist."""
        start_str = pd.to_datetime(start_time).strftime('%Y-%m-%d %H:%M:%S')
        end_str = pd.to_datetime(end_time).strftime('%Y-%m-%d %H:%M:%S')

        sql = f"""
            UPDATE hats.ng_mole_fractions mf
            JOIN hats.ng_analysis a
            ON mf.analysis_num = a.num
            JOIN (
                SELECT
                    run_time,
                    MIN(analysis_time) AS first_analysis_time
                FROM hats.ng_analysis
                WHERE inst_num = {self.inst_num}
                AND run_type_num = 8
                AND analysis_time BETWEEN '{start_str}' AND '{end_str}'
                GROUP BY run_time
            ) firsts
            ON a.run_time = firsts.run_time
            AND a.analysis_time = firsts.first_analysis_time
            SET mf.flag = 'X..',
                mf.qc_status = 'F'
            WHERE a.inst_num = {self.inst_num}
            AND a.run_type_num = 8
            AND a.analysis_time BETWEEN '{start_str}' AND '{end_str}'
            AND mf.qc_status = 'P';
        """
        self.db.doquery(sql)


def get_default_date():
    return (datetime.now() - timedelta(days=30)).strftime('%y%m')


def parse_molecules(molecules):
    if molecules:
        try:
            molecules = molecules.replace(' ','')   # remove spaces
            return molecules.split(',')
        except AttributeError:      # already a list. just return
            return molecules
    return []


def main():
    m4 = M4_Instrument()

    parser = argparse.ArgumentParser(description='Insert M4 GCwerks data into HATS db for selected date range. If no start_date \
                                    is specified then work on the last 30 days of data.')
    parser.add_argument('date', nargs='?', default=get_default_date(), help='Date in the format YYMM')
    parser.add_argument('-m', '--molecules', type=str, default=m4.molecules,
                        help='Comma-separated list of molecules. Add quotes around the list if spaces are used. Default all molecules.')
    parser.add_argument('-x', '--extract', action='store_true', help='Re-extract data from GCwerks first.')
    parser.add_argument('--flagged', action='store_true', help='Use flagged GCwerks exports and set flag=W.. when indicated.')
    parser.add_argument('--list', action='store_true', help='List all available molecule names.')

    args = parser.parse_args()

    if args.list:
        molecules_c = [m.replace(',', '') for m in m4.molecules]       # remove commas from mol names
        print(f"Valid molecule names: {', '.join(molecules_c)}")
        quit()
    
    start_date = args.date
    molecules = parse_molecules(args.molecules)     # returns a list of molecules

    print(f"Start date: {start_date}")
    print("Processing the following molecules: ", molecules)

    if args.extract:
        from m4_export import M4_GCwerks_Export
        exp = M4_GCwerks_Export()
        exp.export_gc_data(start_date, molecules, flagged=args.flagged)
    
    m4 = M4_GCwerks(flagged=args.flagged)
    loaded_ranges = []
    for n, gas in enumerate(molecules):
        df = m4.load_gcwerks(gas, start_date)
        m4.insert_mole_fractions(df)
        if df is not None and not df.empty:
            loaded_ranges.append((df['dt_run'].min(), df['dt_run'].max()))
        print(f'{df.shape[0]} rows inserted/updated for parameter {gas} into hats.ng_mole_fractions')

    if loaded_ranges:
        start_time = min(start for start, _ in loaded_ranges)
        end_time = max(end for _, end in loaded_ranges)
        m4.flag_first_reference_run(start_time, end_time)


if __name__ == '__main__':
    main()
