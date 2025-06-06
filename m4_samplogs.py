#! /usr/bin/env python

import argparse
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import DateOffset
from pathlib import Path

from m4_export import M4_base

class M4_SampleLogs(M4_base):
    
    TIME_OFFSET = 15.5   # minutes after the run starts and the press data is logged.
    RUN_TIME_GAP = 2.0   # hours. Minimum time between runs to be considered a new run.

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
        """Merge pressure files with chromatogram file names."""
        
        valid_sites = [
            'alt', 'amy', 'bld', 'brw', 'cgo', 'hfm', 'kum', 'lef', 'mhd', 'mko',
            'mlo', 'nwr', 'psa', 'rpb', 'smo', 'spo', 'sum', 'thd', 'wis'
        ]
        
        # Determine the start date based on the duration argument.
        if duration.lower() == 'all':
            start_date = pd.to_datetime(self.m4_start_date, format='%Y%m%d')
        else:
            offset = pd.tseries.frequencies.to_offset(duration)
            start_date = pd.Timestamp.today() - offset
        print(f'Merging run-index and .xl files since {start_date}')
        
        # Load data and filter rows based on the start date.
        xls = self.load_all_xl_files().loc[start_date:].reset_index()
        runs = self.load_runindex().loc[start_date:].reset_index()
        
        # Merge data using an asof merge with the specified tolerance.
        mm = pd.merge_asof(
            runs,
            xls,
            left_on='dt_run',
            right_on='dt_sync',
            tolerance=pd.Timedelta(self.merge_delta),
            direction='nearest'
        ).dropna(subset=['dt_sync'])
        
        # Extract site, sample_time, and tank info from the 'info' column
        pattern = (
            r'^(?P<site>[A-Za-z0-9]{3})_?'
            r'(?P<sample_time>(?:\d{1,2}_)?[A-Za-z]{3}_[0-9]{2})_?'
            r'#(?P<tank>[\w-]+)$'
        )
        mask_valid = mm['info'].str.lower().str[:3].isin(valid_sites)
        extracted = mm.loc[mask_valid, 'info'].str.extract(pattern)
        mm.loc[mask_valid, ['site', 'sample_time', 'tank']] = extracted[['site', 'sample_time', 'tank']]
        
        # Convert sample_time to YYMMDD format (e.g., "06_jan_25" -> "250106")
        mm.loc[mask_valid, 'sample_time'] = (
            pd.to_datetime(mm.loc[mask_valid, 'sample_time'].str.title(),
                        format='%d_%b_%y', errors='coerce')
            .dt.strftime('%y%m%d')
        )
        
        # For rows that don't have a valid site, set 'tank' to the original 'info'
        mm.loc[~mask_valid, 'tank'] = mm.loc[~mask_valid, 'info']
        
        # Label tanks that contain 'zero' in the 'info' column.
        zero_mask = mm['info'].str.contains('zero', case=False, na=False)
        mm.loc[zero_mask, 'tank'] = mm.loc[zero_mask, 'info']
        mm.loc[zero_mask, 'samptype'] = 'zero'
        
        # Label standard tanks where ssvpos is '14'
        mm.loc[mm['ssvpos'] == '14', 'samptype'] = 'std'
        
        # Label tanks matching a specific pattern as 'pfp'
        pfp_mask = mm['tank'].str.match(r'^\d{1,2}-(\d{4}|x{4})$', na=False)
        mm.loc[pfp_mask, 'samptype'] = 'pfp'
        
        # Process flask runs: split the tank string into flask_id and pair_id.
        flask_mask = mm['samptype'] == 'flask'
        if flask_mask.any():
            flask_split = mm.loc[flask_mask, 'tank'].str.split('-', expand=True)
            mm.loc[flask_mask, 'flask_id'] = flask_split[0].astype(str)
            mm.loc[flask_mask, 'pair_id'] = flask_split[1].astype(str)
        
        # Map run type and compute ccgg event number.
        mm['run_type_num'] = mm['samptype'].str.lower().map(self.run_type_num())
        mm['ccgg_event_num'] = mm.apply(lambda row: self.fetch_ccgg_event_num(row), axis=1)
        
        # Estimate and create a run_time column. This is the time the GSPC sequence started.
        mm['time_diff'] = mm['dt_run'].diff()
        mm['segment'] = (mm['time_diff'] > pd.Timedelta(hours=self.RUN_TIME_GAP)).cumsum()
        mm['run_time'] = mm.groupby('segment')['dt_run'].transform('first')
        mm.loc[mm['segment'] == 0, 'run_time'] = pd.NaT  # set the first segment to NaT in case the load only partial data
        
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
        df = self.return_analysis_nums(df)
        return df

    def insert_ng_analysis(self, df):
        """
        Inserts or updates rows in hats.ng_analysis using a batch upsert.
        This function uses db.doMultiInsert to perform the insertions.
        """
        sql_insert = """
        INSERT INTO hats.ng_analysis (
            analysis_time,
            run_time,
            inst_num,
            run_type_num,
            port,
            port_info,
            pair_id_num,
            flask_id, 
            ccgg_event_num
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            run_type_num = VALUES(run_type_num),
            run_time     = VALUES(run_time),
            port         = VALUES(port),
            port_info    = VALUES(port_info),
            pair_id_num  = VALUES(pair_id_num),
            flask_id     = VALUES(flask_id),
            ccgg_event_num = VALUES(ccgg_event_num),
            num          = LAST_INSERT_ID(num)
        """

        df = df.fillna('')
        params = []

        for _, row in df.iterrows():
            # Skip rows where run_time is NaT (this is the first run in the loaded dataframe and maybe a partial run)
            if row.run_time is not pd.NaT:
                p = (
                    row.dt_run,        # analysis_time
                    row.run_time,      # run_time
                    self.inst_num,     # inst_num (M4)
                    row.run_type_num,  # run_type_num
                    row.ssvpos,        # port (mapped from df['ssvpos'])
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
        
    if options.insert:
        pd.set_option('future.no_silent_downcasting', True)
        #df = df.set_index('dt_x', drop=False)
        df = m4.ng_analysis(df)
        m4.insert_ancillary_data(df)
        print(f'Inserted or updated {df.shape[0]} rows in hats.ng_analysis and ng_ancillary_data.')
    
