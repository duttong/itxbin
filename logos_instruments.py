import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timedelta
import calendar


class LOGOS_Instruments:
    INSTRUMENTS = {'m4': 192, 'fe3': 193, 'bld1': 999} 

    def __init__(self):
        # gcwerks-3 path
        self.gcexport_path = "/hats/gc/gcwerks-3/bin/gcexport"
        

class HATS_DB_Functions(LOGOS_Instruments):
    """ Class for accessing HATS database functions related to instruments. 
        Tailored to works on 'next generation' or 'ng_' tables."""
        
    def __init__(self):
        super().__init__()

        # database connection
        sys.path.append('/ccg/src/db/')
        import db_utils.db_conn as db_conn # type: ignore
        self.db = db_conn.HATS_ng()
        self.doquery = self.db.doquery

    def gml_sites(self):
        """ Returns a dictionary of site codes and site numbers from gmd.site."""
        sql = "SELECT num, code FROM gmd.site;"
        df = pd.DataFrame(self.doquery(sql))
        site_dict = dict(zip(df['code'], df['num']))
        return site_dict

    def query_analytes(self):
        """Returns a dictionary of analytes and parameter numbers."""
        sql = f"SELECT param_num, display_name FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
        df = pd.DataFrame(self.doquery(sql))
        analytes_dict = dict(zip(df['display_name'], df['param_num']))
        return analytes_dict
    
    def query_molecules(self):
        """ Returns a list of analytes or molecules (no parameter number) """
        analytes = self.query_analytes()
        return analytes.keys()
    
    def run_type_num(self):
        """ Run types defined in the hats.ng_run_types table """
        sql = "SELECT * FROM hats.ng_run_types;"
        r = self.doquery(sql)
        results = {entry['name'].lower(): entry['num'] for entry in r}
        results['std'] = 8
        return results

    def standards(self):
            """ Returns a dictionary of standards files and a key used in the HATS db."""
            sql = "SELECT num, serial_number, std_ID FROM hats.standards"
            df = pd.DataFrame(self.doquery(sql))
            standards_dict = df.set_index('std_ID')[['num', 'serial_number']].T.to_dict('list')
            return standards_dict
        
    def scale_values(self, tank, pnum):
        """
        Returns a dictionary of scale values for a given tank and parameter number (pnum).
        """
        # Extract only the digits before the first "_" in the tank variable
        match = re.search(r'(\d+)[^\d_]*_', tank)
        tank = match.group(1) if match else ''.join(filter(str.isdigit, tank))
        
        sql = f"""
            SELECT start_date, serial_number, level, coef0, coef1, coef2 FROM hats.scale_assignments 
            where serial_number like '%{tank}%'
            and inst_num = {self.inst_num} 
            and scale_num = (select idx from reftank.scales where parameter_num = {pnum});
        """
        df = pd.DataFrame(self.db.doquery(sql))
        if not df.empty:
            return df.iloc[0].to_dict()
        else:
            Warning(f"Scale values not found for tank {tank} and parameter number {pnum}.")
            return None

    def return_analysis_nums(self, df, time_col='dt_run'):
        """ Inserts the analysis numbers into the dataframe based on the time column.
        This function assumes that the time_col in df is in a datetime format.
        Args:
            df (pd.DataFrame): DataFrame containing the time column.
            time_col (str): Name of the column in df that contains the run times.
        Returns:
            pd.DataFrame: DataFrame with an additional column 'analysis_num' containing the analysis numbers.
        """
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in the DataFrame.")
        if df.empty:
            df['analysis_num'] = None
            return df
        
        # Copy and ensure df[time_col] is datetime64
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        # If df is empty, just add the column and return
        if df.empty:
            df['analysis_num'] = None
            return df

        # Determine the min/max run times so we only pull what's needed
        min_time = df[time_col].min().strftime('%Y-%m-%d %H:%M:%S')
        max_time = df[time_col].max().strftime('%Y-%m-%d %H:%M:%S')
    
        sql = (
            "SELECT analysis_time, num "
            "FROM hats.ng_analysis "
            f"WHERE inst_num = {self.inst_num} "
            f"  AND analysis_time >= '{min_time}' "
            f"  AND analysis_time <= '{max_time}';"
        )

        db_df = pd.DataFrame(self.db.doquery(sql))

        # Now merge back onto the original df:
        out = df.merge(
            db_df,
            how='left',
            left_on=time_col,
            right_on='analysis_time'
        ).rename(columns={'num': 'analysis_num'})

        out.drop(columns=['analysis_time'], inplace=True)
        return out


class M4_Instrument(HATS_DB_Functions):
    """ Class for accessing M4 specific functions in the HATS database. """
    
    STANDARD_RUN_TYPE = 8
    COLOR_MAP_RUN_TYPE = {
        1: "#1f77b4",  # Flask
        4: "#ff7f0e",  # Other
        5: "#2ca02c",  # PFP
        6: "#dd89f9",  # Zero
        7: "#c7811b",  # Tank
        8: "#505c5c",  # Standard
        "Response": "#e04c19",  # Response
        "Ratio": "#1f77b4",  # Ratio
        "Mole Fraction": "#2ca02c",  # Mole Fraction
    }
    
    COLOR_MAP = {
        # SSV ports (0-16)
        0: 'cornflowerblue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'hotpink',
        5: 'purple', 6: 'orange', 7: 'darkgreen', 8: 'darkred', 9: 'lightgreen',
        10: 'cornflowerblue', 11: 'green', 12: 'red', 13: 'cyan', 14: 'pink',
        15: 'teal', 16: 'orange'}
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'm4'
        self.inst_num = 192
        self.start_date = '20231223'         # data before this date is not used.
        self.gc_dir = Path("/hats/gc/m4")
        self.export_dir = self.gc_dir / "results"

        self.molecules = self.query_molecules()
        self.analytes = self.query_analytes()
        self.analytes_inv = {int(v): k for k, v in self.analytes.items()}
        self.response_type = 'area'
                
    def load_data(self, pnum, channel=None, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, "%y%m")
        last_day = calendar.monthrange(end_date.year, end_date.month)[1]
        end_date = end_date.replace(day=last_day)

        if start_date is None:
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime.strptime(start_date, "%y%m")

        start_date_str = start_date.strftime("%Y-%m-01")
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Loading data from {start_date_str} to {str(end_date_str)} for parameter {pnum}")
        # todo: use flags - using low_flow flag
        query = f"""
            SELECT analysis_datetime, run_time, run_type_num, port, port_info, detrend_method_num, 
                area, mole_fraction, net_pressure, flag, sample_id, pair_id_num, site
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                AND area != 0
                AND detrend_method_num != 3
                AND low_flow != 1
                AND run_time BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            print(f"No data found for parameter {pnum} in the specified date range.")
            self.data = pd.DataFrame()
            return
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'], errors='raise', utc=True)
        df['run_time']          = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['port']              = df['port'].astype(int)
        df['area']              = df['area'].astype(float)
        df['net_pressure']      = df['net_pressure'].astype(float)
        df['area']              = df['area']/df['net_pressure']         # response per pressure
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df['parameter_num']     = pnum
        df['port_idx']          = df['port'].astype(int)        # used for plotting
 
        # base port label on port_info and port number
        df['port_label'] = (
            df['port_info'].fillna('').str.strip() + ' (' +
            df['port'].astype(int).astype(str) + ')'
        )

        # flask_port label
        mask = (df['run_type_num'] == 1)
        df.loc[mask, 'port_label'] = (
            df.loc[mask, 'site'] + ' ' +
            df.loc[mask, 'pair_id_num'].astype(int).astype(str) + '-' +
            df.loc[mask, 'sample_id'].astype(int).astype(str) + ' (' +
            df.loc[mask, 'port'].astype(int).astype(str) + ')'
        )

        # clean up any stray spaces
        df['port_label'] = df['port_label'] \
                            .str.replace(r'\s+', ' ', regex=True) \
                            .str.strip()
            
        self.data = df.sort_values('analysis_datetime')
        return self.data


class FE3_Instrument(HATS_DB_Functions):
    """ Class for accessing M4 specific functions in the HATS database. """
    
    STANDARD_RUN_TYPE = 1       # port number the standard is run on.
    WARMUP_RUN_TYPE = 3         # run type num warmup runs are on.
    # color map made for a combination of SSV and Flask ports.
    COLOR_MAP = {
        # SSV ports (0-9)
        0: 'cornflowerblue', 1: 'green', 2: 'red', 3: 'cyan', 4: 'pink',
        5: 'gray', 6: 'orange', 7: 'darkgreen', 8: 'darkred', 9: 'lightgreen',
        # Flask ports (10-19)
        10: 'cornflowerblue', 11: 'green', 12: 'red', 13: 'cyan', 14: 'pink',
        15: 'gray', 16: 'orange', 17: 'darkgreen', 18: 'darkred', 19: 'lightgreen'}
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'fe3'
        self.inst_num = 193
        self.start_date = '20191217'         # data before this date is not used.
        self.gc_dir = Path("/hats/gc/fe3")
        self.export_dir = self.gc_dir / "results"
        
        self.molecules = self.query_molecules()
        self.analytes = self.query_analytes()
        self.analytes_inv = {int(v): k for k, v in self.analytes.items()}
        self.response_type = 'height'

        # code to handle CFC11 and CFC113 on two channels
        new = {}
        for name, num in self.analytes.items():
            if name in ('CFC11', 'CFC113'):
                # rename to (a) then duplicate with (c)
                new[f"{name} (a)"] = num
            else:
                new[name] = num
        self.analytes = new
        self.analytes['CFC11 (c)'] = self.analytes['CFC11 (a)']
        self.analytes['CFC113 (c)'] = self.analytes['CFC113 (a)']

    def load_data(self, pnum, channel=None, start_date=None, end_date=None):
        """Load data from the database with date filtering.
        Args:
            pnum (int): Parameter number to filter data.
            channel (str, optional): Channel to filter data. Defaults to None.
            start_date (str, optional): Start date in YYMM format. Defaults to None.
            end_date (str, optional): End date in YYMM format. Defaults to None.
        """
        
        if end_date is None:
            end_date = datetime.today()
        else:
            end_date = datetime.strptime(end_date, "%y%m")
        last_day = calendar.monthrange(end_date.year, end_date.month)[1]
        end_date = end_date.replace(day=last_day)

        if start_date is None:
            start_date = end_date - timedelta(days=60)
        else:
            start_date = datetime.strptime(start_date, "%y%m")

        start_date_str = start_date.strftime("%Y-%m-01")
        end_date_str = end_date.strftime("%Y-%m-%d")

        print(f"Loading data from {start_date_str} to {end_date_str} for parameter {pnum}")
        # todo: use flags
        channel_str = f"AND channel = '{channel}'" if channel else ""
        query = f"""
            SELECT analysis_datetime, run_time, run_type_num, port, port_info, flask_port, detrend_method_num, 
                height, mole_fraction, channel, flag, sample_id, pair_id_num, site
            FROM hats.ng_data_view
            WHERE inst_num = {self.inst_num}
                AND parameter_num = {pnum}
                {channel_str}
                AND height != 0
                AND run_type_num != {self.WARMUP_RUN_TYPE}
                AND detrend_method_num != 3
                AND run_time BETWEEN '{start_date_str}' AND '{end_date_str}'
            ORDER BY analysis_datetime;
        """
        df = pd.DataFrame(self.db.doquery(query))
        if df.empty:
            print(f"No data found for parameter {pnum} in the specified date range.")
            self.data = pd.DataFrame()
            return
        
        df['analysis_datetime'] = pd.to_datetime(df['analysis_datetime'], errors='raise', utc=True)
        df['run_time']          = pd.to_datetime(df['run_time'], errors='raise', utc=True)
        df['run_type_num']      = df['run_type_num'].astype(int)
        df['detrend_method_num'] = df['detrend_method_num'].astype(int)
        df['height']            = df['height'].astype(float)
        df['mole_fraction']     = df['mole_fraction'].astype(float)
        df['parameter_num']     = pnum
        df['port_idx']          = df['port'].astype(int) + df['flask_port'].fillna(0).astype(int)
        
        # base port label on port_info and port number
        df['port_label'] = (
            df['port_info'].fillna('').str.strip() + ' (' +
            df['port'].astype(int).astype(str) + ')'
        )

        # flask_port label
        mask = df['flask_port'].notna()
        df.loc[mask, 'port_label'] = (
            df.loc[mask, 'site'] + ' ' +
            df.loc[mask, 'pair_id_num'].astype(int).astype(str) + '-' +
            df.loc[mask, 'sample_id'].astype(int).astype(str) + ' (' +
            df.loc[mask, 'flask_port'].astype(int).astype(str) + ')'
        )

        # clean up any stray spaces
        df['port_label'] = df['port_label'] \
                            .str.replace(r'\s+', ' ', regex=True) \
                            .str.strip()
                            
        
        self.data = df.sort_values('analysis_datetime')
        return self.data

class BLD1_Instrument(HATS_DB_Functions):
    """ Class for accessing BLD1 (Stratcore) specific functions in the HATS database. """
    
    def __init__(self):
        super().__init__()
        self.inst_id = 'bld1'
        self.inst_num = 999
        self.start_date = '20191217'         # data before this date is not used.
        self.gc_dir = Path("/hats/gc/bld1")
        self.export_dir = self.gc_dir / "results"
