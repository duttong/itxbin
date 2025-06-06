#!/usr/bin/env python

import pandas as pd
import argparse
from datetime import datetime, timedelta

from logos_instruments import M4_Instrument

class M4_GCwerks(M4_Instrument):

    def __init__(self):
        super().__init__()
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
            print(f'Start date "{t_start}" selected was too early, using "{self.start_date}"')
            start_dt = pd.to_datetime(self.start_date)
        
        if gas not in self.molecules:
            print(f'Incorrect gas {gas} name.')
            return

        file = self.gcwerks_results / f'data_{gas}.csv'
        if file.exists():
            df = pd.read_csv(file, na_values=[' nan'])
            df.columns = [col.replace(f'{gas}_', '').strip() for col in df.columns]
        else:
            print(f'Missing GCwerks output file: {file}')
            return

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
                qc_status
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
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
                "P"                  # Preliminary QC status flag
            )
            params.append(p)

            if self.db.doMultiInsert(sql_insert, params): 
                params=[]

        # Process any remaining rows in the final batch:
        self.db.doMultiInsert(sql_insert, params, all=True)


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
        exp.export_gc_data(start_date, molecules)
    
    m4 = M4_GCwerks()
    for n, gas in enumerate(molecules):
        df = m4.load_gcwerks(gas, start_date)
        m4.insert_mole_fractions(df)
        print(f'{df.shape[0]} rows inserted/updated for parameter {gas} into hats.ng_mole_fractions')


if __name__ == '__main__':
    main()
