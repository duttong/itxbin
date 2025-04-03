#! /usr/bin/env python

import argparse
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path

from m4_export import M4_base

class M4_SampleLogs(M4_base):
    
    TIME_OFFSET = 15.5   # minutes after the run starts and the press data is logged.

    def __init__(self):
        super().__init__()
        # rglob will read all .xl files
        self.incoming_dir = self.gc_dir / 'MassHunter/GCMS/M4 GSPC Files'
        #self.incoming_dir = self.gc_dir / 'chemstation'
        self.xlfiles = sorted(self.incoming_dir.rglob('*.xl'))
        self.merge_delta = '8 minutes'
                        
    def read_custom_xl_file(self, file_path):
        """
        Reads a tab-delimited file and pads rows with missing trailing columns.
        Renames the DataFrame columns to lowercase, removes '#' symbols,
        and renames 'date' and 'time' to 'xl_date' and 'xl_time'.
        """
        columns = [
            "Filename", "Date", "Time", "Sample#", "SSVPos", "SampType", "Net_Pressure", 
            "Init_P", "Final_P", "InitP_RSD", "FinalP_RSD", "Low_Flow", "cryocount", 
            "loflocount", "Last_flow", "Last_vflow", "pfpFlask", "pfpOPEN", "pfpCLOSE", 
            "PRESS_#1", "PRESS_#2", "PRESS_#3"
        ]
        
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("Filename"):
                continue
            
            parts = line.split('\t')
            # If there are fewer than 22 columns, pad with empty strings.
            if len(parts) < len(columns):
                parts.extend([""] * (len(columns) - len(parts)))
            # If there are extra columns, keep only the first 22.
            elif len(parts) > len(columns):
                parts = parts[:len(columns)]
            
            # Ensure the filename ends with '.xl' if non-empty
            if parts[0] and not parts[0].endswith('.xl'):
                parts[0] = parts[0] + '.xl'
                
            data.append(parts)
        
        df = pd.DataFrame(data, columns=columns)
        
        # Convert specified columns to floats
        float_columns = [
            "Net_Pressure", "Init_P", "Final_P", "InitP_RSD", 
            "FinalP_RSD", "Last_flow", "Last_vflow",
            "cryocount", "loflocount", 
            "PRESS_#1", "PRESS_#2", "PRESS_#3"
        ]
        for col in float_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert Low_Flow to int (if non-empty)
        df["Low_Flow"] = df["Low_Flow"].map({"Y": 1, "N": 0})
        
        # Rename columns to lowercase and remove '#' characters.
        df.rename(columns={'PRESS_#1': 'pfp_press1', 'PRESS_#2': 'pfp_press2', 'PRESS_#3': 'pfp_press3'}, inplace=True)
        df.columns = df.columns.str.lower()
        
        # Further rename 'date' and 'time' to 'xl_date' and 'xl_time'
        df.rename(columns={"date": "xl_date", "time": "xl_time"}, inplace=True)
        
        # Retain the valve number that was opened or closed. This number is parsed
        # from the string returned by the pfp. If a valve can't be determined use -1
        mask = df['pfpopen'].str.contains('valve open', na=False)

        # Extract the digits immediately before 'F'
        # The regex (\d+)(?=F) captures one or more digits only if they are immediately followed by 'F'
        extracted = df.loc[mask, 'pfpopen'].str.extract(r'(\d+)(?=F)', expand=False)

        # Convert extracted values to integers; where extraction fails, fill with -1
        df.loc[mask, 'test'] = extracted.fillna(-1).astype(int)

        # For rows without 'valve open', set 'test' to -1
        df.loc[~mask, 'test'] = -1
        df.rename(columns={'pfpopen': 'pfpopen_org', 'test': 'pfpopen'}, inplace=True)

        mask = df['pfpclose'].str.contains('valve close', na=False)

        # Extract the digits immediately before 'F'
        # The regex (\d+)(?=F) captures one or more digits only if they are immediately followed by 'F'
        extracted = df.loc[mask, 'pfpclose'].str.extract(r'(\d+)(?=F)', expand=False)

        # Convert extracted values to integers; where extraction fails, fill with -1
        df.loc[mask, 'test'] = extracted.fillna(-1).astype(int)

        # For rows without 'valve open', set 'test' to -1
        df.loc[~mask, 'test'] = -1
        df.rename(columns={'pfpclose': 'pfpclose_org', 'test': 'pfpclose'}, inplace=True)        

        return df
    
    def load_all_xl_files(self):
        """ Load all .xl pressure files into a single dataframe. Drop duplicate rows. """
        dfs = []
        for file in self.xlfiles:
            #print(file)
            df = self.read_custom_xl_file(file)
            df['dt_xl'] = pd.to_datetime(df['xl_date'].astype(str) + ' ' + df['xl_time'].astype(str))
            dfs.append(df)
            
        df = pd.concat(dfs, axis=0)
        df['dt_sync'] = df['dt_xl'] + pd.Timedelta(minutes=-self.TIME_OFFSET)
        df = df.drop_duplicates(subset='dt_xl', keep='first')
        df = df.set_index('dt_xl')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        return df

    def load_runindex(self):
        """ Load GCwerks .run_index file which contains the names of all chromatograms in GCwersk. """
        
        runindex = self.gc_dir / '.run-index'
        with open(runindex, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Process each line
        parsed_data = []
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace
            line = line.replace(",", "_")  # Replace commas with underscores
            
            parts = line.split(".", maxsplit=2)  # Split into three parts
            if len(parts) == 3:
                datestamp, timestamp, info = parts
            else:
                datestamp, timestamp, info = parts[0], parts[1], ""  # Handle cases with missing fields
            
            parsed_data.append((datestamp, timestamp, info))
            
        df = pd.DataFrame(parsed_data, columns=['date', 'time', 'info'])
        df['dt_run'] = pd.to_datetime(df['date'] + df['time'], format="%y%m%d%H%M")
        df = df.set_index('dt_run')
        return df

    def merged_rundata(self, duration='2ME', save=True):
        """ Merge pressure files with chromatogram file names """
        
        valid_sites = ['alt', 'amy', 'bld', 'brw', 'cgo', 'hfm', 'kum', 'lef', 'mhd', 'mko',
                        'mlo', 'nwr', 'psa', 'rpb', 'smo', 'spo', 'sum', 'thd', 'wis']
        
        if duration.lower() == 'all':
            start_date = pd.to_datetime(self.m4_start_date, format='%Y%m%d')
        else:
            # Convert the string (e.g., "2ME") into a DateOffset
            offset = pd.tseries.frequencies.to_offset(duration)
            # Use today's date as the reference
            start_date = pd.Timestamp.today() - offset
        print(f'Merging run-index and .xl files since {start_date}')
                
        xls = self.load_all_xl_files()
        xls = xls.loc[start_date:]
        xls.reset_index(inplace=True)
        runs = self.load_runindex()
        runs = runs.loc[start_date:]
        runs.reset_index(inplace=True)
        
        # Merge with a tolerance of self.merge_delta
        mm = pd.merge_asof(
            runs,
            xls,
            left_on='dt_run',
            right_on='dt_sync',
            tolerance=pd.Timedelta(self.merge_delta),
            direction='nearest'  # or 'backward'/'forward' depending on your need
        )
        mm = mm.dropna(subset=['dt_sync'])
        
        # Define a regex pattern that only matches strings with all three parts:
        #   - site: one or more alphanumeric characters before the first underscore
        #   - sample_time: a pattern like "06_jan_25" (day, month, year)
        #   - tank: digits and hyphens after a '#'
        pattern = r'^(?P<site>[A-Za-z0-9]{3})_?(?P<sample_time>(?:\d{1,2}_)?[A-Za-z]{3}_[0-9]{2})_?#(?P<tank>[\w-]+)$'

        mask = mm['info'].str.lower().str[0:3].isin(valid_sites)
        extracted = mm.loc[mask, 'info'].str.extract(pattern)

        # The extract will return NaN for groups that don't match the pattern.
        # Now assign the results to new columns in the original dataframe.
        mm.loc[mask, 'site'] = extracted['site']
        mm.loc[mask, 'sample_time'] = extracted['sample_time']
        mm.loc[mask, 'tank'] = extracted['tank']

        # Convert sample_time (currently in "DD_mon_YY" format, e.g., "06_jan_25") to YYMMDD format.
        # To handle the month abbreviation, we first title-case the string.
        mm.loc[mask, 'sample_time'] = (
            pd.to_datetime(mm.loc[mask, 'sample_time'].str.title(), format='%d_%b_%y', errors='coerce')
            .dt.strftime('%y%m%d')
        )

        mm.loc[~mask, 'tank'] = mm.loc[~mask, 'info']
        
        # Label tank type as zero
        mm.loc[mm['info'].str.contains('zero', case=False, na=False), 'tank'] = mm['info']
        mm.loc[mm['info'].str.contains('zero', case=False, na=False), 'samptype'] = 'zero'
        
        # Label tank as std for port 14
        mm.loc[mm['ssvpos'] == '14', 'samptype'] = 'std'
        # Set samptype to pfp if the tank is like nn-xxxx
        mm.loc[mm['tank'].str.match(r'^\d{1,2}-(\d{4}|x{4})$', na=False), 'samptype'] = 'pfp'
        
        mask = mm['samptype'] == 'flask'
        # Create two new columns directly using the string split results for flask runs
        flask_pair_split = mm.loc[mask, 'tank'].str.split('-', expand=True)
        mm.loc[mask, 'flask_id'] = flask_pair_split[0].astype(str)
        mm.loc[mask, 'pair_id'] = flask_pair_split[1].astype(str)
        
        # add run_type_num (used for table insertion)
        mapping = self.run_type_num()
        mm['run_type_num'] = mm['samptype'].str.lower().map(mapping)
        
        # add ccgg_event_num for pfps (used for table insertion)
        mm['ccgg_event_num'] = mm.apply(lambda row: self.fetch_ccgg_event_num(row), axis=1)
        
        if save:
            self.save_samplelogs(mm)
        
        return mm
    
    def fetch_ccgg_event_num(self, row):
        """ Determine and return the ccgg_event number for a flask in a pfp """

        # Only need to work on these sites for M4 (so far)
        site_map = {'mlo': 75, 'mko': 73}
        
        try:
            if row['samptype'] != 'pfp':
                return None
            
            flask, pfp = row['tank'].split('-')
            flask_id = f"{int(flask):02d}"
            site_id = site_map.get(row['site'])

            if site_id is None:
                return None

            if 'x' in pfp:
                id_filter = f"%-{flask_id}"
            else:
                id_filter = f"{pfp}-{flask_id}"

            sql = f"""
                SELECT num FROM ccgg.flask_event 
                WHERE date = '{row['sample_time']}'
                AND site_num = {site_id}
                AND id LIKE '{id_filter}';
            """

            result = self.db.doquery(sql)
            return result[0]['num'] if result else None

        except Exception as e:
            print(f"Error processing row: {e}")
            return None
    
    def save_samplelogs(self, merged_df):
        # Define the output directory (assuming it's a directory)
        log_dir = self.gc_dir / 'logs' / 'sample.log'
        
        # Select the desired columns from your DataFrame.
        df_out = merged_df[['dt_sync', 'date', 'time', 'ssvpos', 'samptype', 'site', 'sample_time', 'tank', 'net_pressure']].copy()

        # Rename the columns to match the desired header.
        df_out.columns = ['dt_sync', 'date', 'time', 'port', 'type', 'site', 'sample_time', 'tank', 'psamp']

        # Replace missing values with '-' in selected string-based columns
        df_out[['port', 'type', 'site', 'sample_time', 'tank', 'psamp']] = (
            df_out[['port', 'type', 'site', 'sample_time', 'tank', 'psamp']].fillna('-')
        )
        # Extract the year into a new column.
        df_out['year'] = df_out['dt_sync'].dt.year
        
        # Process each group by year.
        for year, group in df_out.groupby('year'):
            # Create the file name: take the last two digits of the year and append "01"
            file_name = f"{str(year)[2:]}01"
            file_path = log_dir / file_name
            
            # Prepare the new data: drop columns that aren't needed in the output.
            new_data = group.drop(columns=['dt_sync', 'year'])
            
            # Ensure the new data is all string for consistency when merging.
            new_data = new_data.astype(str)
            
            if file_path.exists():
                # Load the existing data (assuming tab-separated values).
                existing_data = pd.read_csv(file_path, sep='\t', dtype=str)
                # Concatenate the existing data with the new data.
                merged_data = pd.concat([existing_data, new_data], ignore_index=True)
                # Drop duplicate rows and sort the merged data.
                # Here, we sort by date, time, and port columns; adjust as necessary.
                merged_data = merged_data.drop_duplicates(subset=['date', 'time']).sort_values(by=['date', 'time'])
                print(f'Updating existing file: {file_path} total rows = {merged_data.shape[0]}')
                merged_data.to_csv(file_path, sep='\t', index=False)
            else:
                print(f'Creating new file: {file_path}')
                # Save the new data as-is if no file exists.
                new_data.to_csv(file_path, sep='\t', index=False)
                
    def ng_analysis(self, df):
        """ Inserts and updates data in the hats.ng_analysis table. 
            These routines work on the whole df. Make sure df is trimmed to the pertinent data. """
        self.insert_ng_analysis(df)
        self.update_pfp_flask_port(df)
        self.return_analysis_nums(df)

    def insert_ng_analysis(self, df):
        """
        Inserts or updates rows in hats.ng_analysis using a batch upsert.
        This function uses db.doMultiInsert to perform the insertions.
        """
        sql_insert = """
        INSERT INTO hats.ng_analysis (
            analysis_time,
            inst_num,
            run_type_num,
            port,
            port_info,
            pair_id_num,
            flask_id, 
            ccgg_event_num
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            run_type_num = VALUES(run_type_num),
            port         = VALUES(port),
            port_info    = VALUES(port_info),
            pair_id_num  = VALUES(pair_id_num),
            flask_id     = VALUES(flask_id),
            ccgg_event_num = VALUES(ccgg_event_num),
            num          = LAST_INSERT_ID(num)
        """

        df = df.fillna('')
        params = []

        for idx, row in df.iterrows():
            p = (
                row.dt_run,        # analysis_time
                self.inst_num,     # inst_num (M4)
                row.run_type_num,  # run_type_num
                row.ssvpos ,       # port (mapped from df['ssvpos'])
                row.tank,          # tank
                row.pair_id,       # pair_id
                row.flask_id,      # flask_id
                row.ccgg_event_num # ccgg_event_num
            )
            params.append(p)

            if self.db.doMultiInsert(sql_insert, params): 
                params=[]

        # Process any remaining rows in the final batch:
        self.db.doMultiInsert(sql_insert, params, all=True)
        
    def update_pfp_flask_port(self, df):
        """ Specifically looks at PFP runs and uses the port_info column. This column has
            flask port followed by pfp id. 
            Reads the flask port from port_info and updates the flask_port column for PFPs only. 
            run_type_num = 5 are all PFP runs. """
            
        start_time = df['dt_run'].min()
        sql = f"""
            UPDATE hats.ng_analysis
            SET flask_port = CAST(SUBSTRING_INDEX(port_info, '-', 1) AS UNSIGNED)
            WHERE inst_num = {self.inst_num}
                AND run_type_num = 5
                AND analysis_time > '{start_time}';
        """
        self.db.doquery(sql)

    def return_analysis_nums(self, df):
        """
        Loops over each row in the DataFrame and queries the database
        for the primary key (num) based on analysis_time and inst_num.
        Returns the DataFrame with a new 'analysis_num' column.
        """
        analysis_nums = []
        
        for _, row in df.iterrows():
            # Adjust the formatting of analysis_time if necessary.
            query = f"SELECT num FROM hats.ng_analysis WHERE analysis_time = '{row.dt_run}' AND inst_num = {self.inst_num}"
            result = self.db.doquery(query)
            
            if result:
                # Depending on the return type, extract the num value.
                num_value = result[0][0] if isinstance(result[0], (list, tuple)) else result[0]['num']
                analysis_nums.append(num_value)
            else:
                analysis_nums.append(None)
        
        # Add the primary keys as a new column in the DataFrame.
        df['analysis_num'] = analysis_nums
        return df
    
    def insert_ancillary_data(self, df):
        """
        Inserts or updates rows in hats.ng_analysis using a batch upsert.
        This function uses db.doMultiInsert to perform the insertions.
        """
        sql_insert = """
            INSERT INTO hats.ng_ancillary_data (
                analysis_num,
                init_p,
                final_p,
                net_pressure,
                initp_rsd,
                finalp_rsd,
                low_flow,
                cryocount,
                loflocount,
                last_flow,
                last_vflow,
                pfpopen,
                pfpclose,
                pfp_press1,
                pfp_press2,
                pfp_press3
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON DUPLICATE KEY UPDATE
                analysis_num = VALUES(analysis_num),
                init_p       = VALUES(init_p),
                final_p      = VALUES(final_p),
                net_pressure = VALUES(net_pressure),
                initp_rsd    = VALUES(initp_rsd),
                finalp_rsd   = VALUES(finalp_rsd),
                low_flow     = VALUES(low_flow),
                cryocount    = VALUES(cryocount),
                loflocount   = VALUES(loflocount),
                last_flow    = VALUES(last_flow),
                last_vflow   = VALUES(last_vflow),
                pfpopen      = VALUES(pfpopen),
                pfpclose     = VALUES(pfpclose),
                pfp_press1   = VALUES(pfp_press1),
                pfp_press2   = VALUES(pfp_press2),
                pfp_press3   = VALUES(pfp_press3)
            """

        df = df.fillna('')
        params = []

        for idx, row in df.iterrows():
            p = (
                row.analysis_num,    # analysis_num
                row.init_p,          # init_p
                row.final_p,         # final_p
                row.net_pressure,    # net_pressure
                row.initp_rsd,       # initp_rsd
                row.finalp_rsd,      # finalp_rsd
                row.low_flow,        # low_flow
                row.cryocount,       # cryocount
                row.loflocount,      # loflocount
                row.last_flow,       # last_flow
                row.last_vflow,      # last_vflow
                row.pfpopen,         # pfpopen
                row.pfpclose,        # pfpclose
                row.pfp_press1,      # pfp_press1
                row.pfp_press2,      # pfp_press2
                row.pfp_press3       # pfp_press3
            )
            params.append(p)

            if self.db.doMultiInsert(sql_insert, params): 
                params=[]

        # Process any remaining rows in the final batch:
        self.db.doMultiInsert(sql_insert, params, all=True)


if __name__ == '__main__':

    opt = argparse.ArgumentParser(
        description="""Load GSPC pressure files (.xl) from chemstation directory. Merge these data
        with imported chromatogram names (.run_index) and save to sample.log files. """
    )
    opt.add_argument("-d", dest="duration", default='2ME',
                    help="Select the most recent portion of data to process (duration such as 2W, 1ME, 2ME, 1Y, etc., default=2ME)")  
    opt.add_argument("--all", action='store_true',
                    help="Instead of using duration, insert/update all M4 data.")
    opt.add_argument("-i", "--insert", action='store_true',
                    help="Insert data into HATS DB tables (ng_analysis, etc.)")
    
    options = opt.parse_args()
    
    m4 = M4_SampleLogs()
    if options.all:
        df = m4.merged_rundata(duration='all')
    else:
        df = m4.merged_rundata(duration=options.duration)
        
    print(df.shape)
    
    if options.insert:
        pd.set_option('future.no_silent_downcasting', True)
        #df = df.set_index('dt_x', drop=False)
        m4.ng_analysis(df)
        m4.insert_ancillary_data(df)
        print(f'Inserted or updated {df.shape[0]} rows in hats.ng_analysis and ng_ancillary_data.')
    
