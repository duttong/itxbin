#!/usr/bin/env python

import pandas as pd
import numpy as np
from pathlib import Path
#from datatable import fread        # fast but unsupported with python 3.11 as of 240703
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# HATS db connection
sys.path.append('/ccg/src/db/')
import db_utils.db_conn as db_conn

class PR1_db:
    def __init__(self):
        self.inst_num = 58  # PR1 instrument number
        self.pr1_start_date = '20150601'       # data before this date is not used.
        self.db = db_conn.HATS_ng()  # Initialize database connection
        self.gcwerks_results = Path('/hats/gc/pr1/results/')
        self.molecules = self.pr1_molecules()  # Molecules PR1 measures
        self.molecules = ['12-DCE' if m == '1,2-DCE' else m for m in self.molecules]
        self.molecules_c = [m.replace(',', '') for m in self.molecules]       # remove commas from mol names
        self.sites = self.gml_sites()   # site codes and numbers
        self.analytes = self.pr1_analytes()     # PR1 analytes
        self.gas = ''
        self.analysis_table = 'analysis_gsd'    # set to a table like analysis_gsd for debugging
        self.raw_data_table = 'raw_data_gsd'    # raw_data_ gsd

    def gml_sites(self):
        """Returns a dictionary of site codes and site numbers from gmd.site."""
        sql = "SELECT num, code FROM gmd.site;"
        df = pd.DataFrame(self.db.doquery(sql))
        site_dict = dict(zip(df['code'], df['num']))
        return site_dict

    def pr1_standards(self):
        """Returns a dictionary of standards files and a key used in the HATS db."""
        sql = "SELECT num, serial_number, std_ID FROM hats.standards"
        df = pd.DataFrame(self.db.doquery(sql))
        standards_dict = df.set_index('std_ID')[['num', 'serial_number']].T.to_dict('list')
        return standards_dict

    def pr1_analytes(self):
        """Returns a dictionary of PR1 analytes and parameter numbers."""
        sql = "SELECT param_num, display_name FROM hats.analyte_list WHERE inst_num = %s"
        df = pd.DataFrame(self.db.doquery(sql, (self.inst_num,)))
        analytes_dict = dict(zip(df['display_name'], df['param_num']))
        analytes_dict['12-DCE'] = analytes_dict['1,2-DCE']
        return analytes_dict

    def pr1_molecules(self):
        """Returns the list of molecules from the HATS db."""
        sql = "SELECT display_name FROM hats.analyte_list WHERE inst_num = %s ORDER BY disp_order"
        r = self.db.doquery(sql, (self.inst_num,))
        display_names = [item['display_name'] for item in r]
        return display_names

    def return_rownum(self, start_year):
        """ All of the PR1 data files start from 2013 except F12 which starts in 2023.
            This is a lookup table for the row number where the year starts in the data files.
            This is not exact just a good guess. Used to speed up the loading with the fread
            call.
        """
        row_num = {2015: 12805, 
                2016: 30984,
                2017: 53682,
                2018: 73983,
                2019: 95566,
                2020: 116898,
                2021: 139087,
                2022: 161675,
                2023: 183528,
                2024: 205233}
        try:
            row = row_num[start_year]
        except KeyError:
            row = 0
        return row

    def load_gcwerks(self, gas, start_date):
        """Loads GCwerks data for a given gas and returns it as a DataFrame.
           The data files should be stored at self.gcwerks_results path.
        """
        pr1_start_dt = pd.to_datetime(self.pr1_start_date)
        start_dt = pd.to_datetime(start_date)
        if start_dt < pr1_start_dt:
            print(f'Start date "{start_date}" selected was too early, using "{self.pr1_start_date}"')
            start_dt = pr1_start_dt

        self.gas = gas
        if gas not in self.molecules:
            print(f'Incorrect gas {gas} name.')
            return

        file = self.gcwerks_results / f'data_{gas}.csv'
        if file.exists():
            df = pd.read_csv(file)
            df.columns = [col.replace(f'{gas}_', '').strip() for col in df.columns]
        else:
            print(f'Missing GCwerks output file: {file}')
            return
        """
        # this code is for using the full gcwerks extracted data file and a lookup table for row number.
        if file.exists():
            with open(file, 'r') as f:
                column_names = [col.strip() for col in f.readline().strip().split(',')]
            # start loading data from this row number onward. Measurements of F12 started later than
            # other gases. Load all of F12 for now.
            if gas == 'F12':
                row = 0
            else:
                row = self.return_rownum(start_dt.year) - 100
            #df = fread(file, skip_to_line=row).to_pandas()  # Read data file into a DataFrame
            df = pd.read_csv(file, skiprows=range(1, row))  # Read data file into a DataFrame
            df.columns = [col.replace(f'{gas}_', '') for col in column_names]
        else:
            print(f'Missing GCwerks output file: {file}')
            return
        """

        # Make a datetime column and set it as the index
        df = df.assign(dt=pd.to_datetime(df.time))
        df.set_index('dt', inplace=True)

        # trim the dataframe to use start_date (start_dt is the datetime value)
        df = df.loc[start_dt:]

        # parameter and event numbers
        df['pnum'] = self.analytes[gas].strip()
        df['event'] = 0
        
        df['type'] = df['type'].str.strip()
        df['sample'] = df['sample'].str.strip()
        df['standard'] = df['standard'].str.strip()

        # This section parses the sample field. This field is not standardized. It can have
        # a variety of info including sample site code, sample ID, and pair_ID for HATS flasks.
        # no dash in sample
        # blank and burn runs
        df.loc[df['sample'].str.count('-') == 0, 'site'] = np.nan
        df.loc[df['sample'].str.count('-') == 0, 'sample_ID'] = df['sample']
        # one dash in sample
        # BLD-badtestB
        pattern = r'^([A-Z]+)-?([A-Za-z0-9]+)?$'
        df.loc[df['sample'].str.count('-') == 1, 'site'] = df['sample'].str.extract(pattern)[0]
        df.loc[df['sample'].str.count('-') == 1, 'sample_ID'] = df['sample'].str.extract(pattern)[1]
        # two dashes in sample
        # SMO-1223-34333 and BLD-badtestA-00
        pattern = r'^([A-Z]{3})-([A-Za-z0-9]+)-?([A-Za-z0-9]+)?$'
        extracted = df['sample'].str.extract(pattern)
        df.loc[(df['sample'].str.count('-') == 2), 'site'] = extracted[0]
        df.loc[(df['sample'].str.count('-') == 2), 'sample_ID'] = extracted[1] + "-" + extracted[2]
        df.loc[(df['sample'].str.count('-') == 2) & (df['type'] == 'HATS'), 'sample_ID'] = extracted[1]
        df.loc[(df['sample'].str.count('-') == 2) & (df['type'] == 'HATS'), 'event'] = extracted[2]
        # this is needed if the previous line if the "pattern" regex doesn't find a extracted group 2
        df.loc[df['event'].isnull(), 'event'] = 0 

        # lookup site number, 0 if not found
        df['site_num'] = df['site'].map(self.sites).fillna(0).astype(int)

        # Data processing steps...
        standards = self.pr1_standards()
        standards_num = {k: v[0] for k, v in standards.items()}
        standards_sn = {k: v[1] for k, v in standards.items()}
        # standardized serial number
        df['serial_num'] = df['standard'].map(standards_sn).fillna('unknown')
        # unique number assigned to a serial number (this column is slatted to go away)
        df['standard_num'] = df['standard'].map(standards_num).fillna(0)
        
        # code lab_num not sure what it is used for?
        lab_num_mapping = {
            'PFP': 1,
            'CCGG': 1,
            'tank': 1,
            'Tank': 2,
            'FLASK': 2,
            'HATS': 2
        }
        # Map the 'type' column to 'lab_num' using the mapping dictionary
        df['lab_num'] = df['type'].str.upper().map(lab_num_mapping).fillna(0).astype(int)

        # Put columns in order
        columns = ['time', 'type', 'sample', 'site', 'site_num', 'sample_ID', 'event', 'standard', 
                   'serial_num', 'standard_num', 'lab_num', 'port', 'psamp0', 'psamp', 'T1', 
                   'pnum', 'area', 'ht', 'rt', 'w']
        print(gas)
        df = df[columns]

        return df
    
    def tmptbl_create(self):
        """ Create a temporary table for manipulation and insertion. 
             Drop the temporary table if it already exists
        """
        drop_table_query = f"DROP TEMPORARY TABLE IF EXISTS t_data;"
        self.db.doquery(drop_table_query)

        # SQL to create a temporary table
        create_table_query = f"""
            CREATE TEMPORARY TABLE t_data AS
            SELECT 
                a.num, 
                a.analysis_datetime, 
                a.inst_num, 
                a.sample_ID, 
                a.site_num, 
                a.sample_type, 
                a.port, 
                a.standards_num, 
                a.std_serial_num, 
                a.event_num, 
                a.lab_num,   
                r.analysis_num, 
                r.parameter_num, 
                r.peak_area, 
                r.peak_height, 
                r.peak_width, 
                r.peak_RT
            FROM 
                analysis a 
                JOIN raw_data r ON r.analysis_num = a.num
            WHERE 
                1 = 0;  # This ensures the table is created empty
        """
        self.db.doquery(create_table_query)

    @staticmethod
    def NULL(val):
        return None if pd.isnull(val) else val

    def tmptbl_fill(self, df):
        """ Create and fill the temporary table with data from the df """

        self.tmptbl_create()    # create temporary table

        sql_insert = """ 
        INSERT INTO t_data (
            analysis_num, analysis_datetime, inst_num, sample_ID, site_num, sample_type, port, 
            standards_num, std_serial_num, event_num, lab_num, parameter_num,
            peak_area, peak_height, peak_width, peak_RT
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """

        params = []
        for _, row in df.iterrows():
            anum = 0
            sid = row.sample_ID if not pd.isnull(row.sample_ID) else ''
            p0 = (
                anum, row.time, self.inst_num, sid, row.site_num, row.type, row.port,
                row.standard_num, row.serial_num, row.event, row.lab_num, int(self.analytes[self.gas]),
                self.NULL(row.area), self.NULL(row.ht), self.NULL(row.w), self.NULL(row.rt)
            )
            if row.type != 'unknown' and row.type != 'TEST':
                params.append(p0)
                #print(p0)
                if self.db.doMultiInsert(sql_insert, params): 
                    params=[]

        self.db.doMultiInsert(sql_insert, params, all=True)

        self.tmptbl_get_eventnum()       # get event_num info from ccgg.flask_event table
        self.tmptbl_get_analnum()        # fill in previous anaysis data

    def tmptbl_get_eventnum(self):
        """
        Update the `event_num` in the temporary table `t_data` based on matching criteria 
        from the `ccgg.flask_event` table.
        """
        sql = """
            UPDATE t_data t
            SET event_num = (
                SELECT num
                FROM ccgg.flask_event
                WHERE id = t.sample_ID 
                AND site_num = t.site_num 
                AND date < t.analysis_datetime
                ORDER BY date DESC
                LIMIT 1
            )
            WHERE t.sample_type != 'HATS';
        """
        self.db.doquery(sql)

    def tmptbl_get_analnum(self):
        """
        Update the `analysis_num` in the temporary table `t_data` based on matching criteria 
        from the `analysis` table.
        """
        sql = f"""
            UPDATE t_data t
            SET analysis_num = (
                SELECT num
                FROM {self.analysis_table}
                WHERE analysis_datetime = t.analysis_datetime 
                AND inst_num = t.inst_num 
                AND event_num = t.event_num
            );
        """
        self.db.doquery(sql)

    def tmptbl_insert_analysis_debug(self):
        # Returns missing rows
        sql = f"""
            SELECT DISTINCT 
                analysis_datetime, inst_num, sample_ID, site_num, sample_type, port, 
                standards_num, std_serial_num, event_num, lab_num
            FROM t_data 
            WHERE analysis_num = 0;
        """
        inserted = self.db.doquery(sql)
        return inserted

    def tmptbl_insert_analysis(self):
        """
        Insert missing analysis rows into the `analysis` table from the temporary table `t_data`
        where `analysis_num` is 0, then update `analysis_num` in `t_data`.
        """
        sql = f"""
            INSERT INTO {self.analysis_table} (
                analysis_datetime, inst_num, sample_ID, site_num, sample_type, port, 
                standards_num, std_serial_num, event_num, lab_num
            )
            SELECT DISTINCT 
                analysis_datetime, inst_num, sample_ID, site_num, sample_type, port, 
                standards_num, std_serial_num, event_num, lab_num
            FROM t_data 
            WHERE analysis_num = 0;
        """
        inserted = self.db.doquery(sql)

        if inserted is not None:
            print(f'Inserted {inserted} new records into hats.{self.analysis_table}.')
            self.tmptbl_get_analnum()

    def tmptbl_update_analysis(self):
        """
        Update analysis rows using data from the temporary table `t_data`.
        """

        self.tmptbl_insert_analysis()    # insert any missing/new analysis rows

        sql = f"""
            UPDATE hats.{self.analysis_table} a, t_data t 
            SET a.standards_num=t.standards_num, a.std_serial_num=t.std_serial_num, a.port=t.port, a.sample_type=t.sample_type
            WHERE a.num=t.analysis_num and t.analysis_num!=0
        """
        updated = self.db.doquery(sql)
        if updated is not None:
            print(f'Updated {updated} records in hats.{self.analysis_table}.')

    def tmptbl_update_raw_data(self):
        # update area, height, w, rt, etc with data from the t_data temporary table
        sql = f"""
            INSERT INTO {self.raw_data_table} (analysis_num, parameter_num, peak_area, peak_height, peak_width, peak_RT)
            SELECT
                t.analysis_num,
                t.parameter_num,
                t.peak_area,
                t.peak_height,
                t.peak_width,
                t.peak_RT
            FROM
                t_data t
            ON DUPLICATE KEY UPDATE
                peak_area = IF(VALUES(peak_area) <> {self.raw_data_table}.peak_area, VALUES(peak_area), {self.raw_data_table}.peak_area),
                peak_height = IF(VALUES(peak_height) <> {self.raw_data_table}.peak_height, VALUES(peak_height), {self.raw_data_table}.peak_height),
                peak_width = IF(VALUES(peak_width) <> {self.raw_data_table}.peak_width, VALUES(peak_width), {self.raw_data_table}.peak_width),
                peak_RT = IF(VALUES(peak_RT) <> {self.raw_data_table}.peak_RT, VALUES(peak_RT), {self.raw_data_table}.peak_RT);
        """
        updated = self.db.doquery(sql)
        if updated is not None:
            print(f'Updated {updated} rows in hats.{self.raw_data_table}')

    def tmptbl_test(self, v):
        sql = f"""
                UPDATE t_data
                SET peak_height = {v}
                WHERE analysis_num = 312997;
            """
        self.db.doquery(sql)
        
    def tmptbl_output(self):
        return pd.DataFrame(self.db.doquery("SELECT * from t_data;"))


def get_default_date():
    return (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')


def parse_molecules(molecules):
    if molecules:
        try:
            molecules = molecules.replace('1,2-DCE', '12-DCE')
            molecules = molecules.replace(' ','')   # remove spaces
            return molecules.split(',')
        except AttributeError:      # already a list. just return
            return molecules
    return []


def main():
    pr1 = PR1_db()

    parser = argparse.ArgumentParser(description='Insert Perseus GCwerks data into HATS db for selected date range. If no start_date \
                                     is specifide then work on the last 30 days of data.')
    parser.add_argument('date', nargs='?', default=get_default_date(), help='Date in the format YYYYMMDD or YYYYMMDD.HHMM')
    parser.add_argument('-m', '--molecules', type=str, default=pr1.molecules,
                        help='Comma-separated list of molecules. Add quotes around the list if spaces are used. Default all molecules.')
    parser.add_argument('--list', action='store_true', help='List all available molecule names.')

    args = parser.parse_args()

    if args.list:
        print(f"Valid molecule names: {', '.join(pr1.molecules_c)}")
        quit()
    
    start_date = args.date
    molecules = parse_molecules(args.molecules)     # returns a list of molecules

    print(f"Start date: {start_date}")
    print("Processing the following molecules: ", molecules)
    
    for gas in molecules:
        df = pr1.load_gcwerks(gas, start_date)
        pr1.tmptbl_fill(df)             # create and fill in temp data table with GCwerks results
        #insert = pr1.tmptbl_insert_analysis_debug()
        #if insert:
        #    pd.DataFrame(insert).to_csv(f'pr1_missing.csv', index=None)
        pr1.tmptbl_update_analysis()    # insert and update any rows in hats.analysis with new data
        pr1.tmptbl_update_raw_data()    # update the hats.raw_data table with area, ht, w, rt
        
        # view temp table's data
        #tmpdf = pr1.tmptbl_output()
        #print(tmpdf.loc[tmpdf.sample_type == 'PFP'][['analysis_datetime', 'sample_ID', 'event_num', 'analysis_num', 'peak_height', 'peak_RT']])


if __name__ == '__main__':
    main()
