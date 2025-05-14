#! /usr/bin/env python

import subprocess
import argparse
import sys
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class M4_base:
    inst_num = 192
    m4_start_date = '20231223'         # data before this date is not used.
    gcexport_path = "/hats/gc/gcwerks-3/bin/gcexport"
    gc_dir = Path("/hats/gc/m4")
    export_dir = gc_dir / "results"

    def __init__(self, *args, **kwargs):
        # Call the next initializer in the cooperative chain (if there is one)
        super().__init__(*args, **kwargs)
        sys.path.append('/ccg/src/db/')
        import db_utils.db_conn as db_conn # type: ignore
        self.db = db_conn.HATS_ng()
        self.molecules = self.m4_molecules()
        
    def gml_sites(self):
        """ Returns a dictionary of site codes and site numbers from gmd.site."""
        sql = "SELECT num, code FROM gmd.site;"
        df = pd.DataFrame(self.db.doquery(sql))
        site_dict = dict(zip(df['code'], df['num']))
        return site_dict

    def m4_standards(self):
        """ Returns a dictionary of standards files and a key used in the HATS db."""
        sql = "SELECT num, serial_number, std_ID FROM hats.standards"
        df = pd.DataFrame(self.db.doquery(sql))
        standards_dict = df.set_index('std_ID')[['num', 'serial_number']].T.to_dict('list')
        return standards_dict

    def m4_analytes(self):
        """Returns a dictionary of M4 analytes and parameter numbers."""
        sql = f"SELECT param_num, display_name FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
        df = pd.DataFrame(self.db.doquery(sql))
        analytes_dict = dict(zip(df['display_name'], df['param_num']))
        return analytes_dict
    
    def m4_molecules(self):
        """ Returns a list of analytes or molecules (no parameter number) """
        analytes = self.m4_analytes()
        return analytes.keys()
    
    def m4_scale_values(self, tank, pnum):
        """
        Returns a dictionary of scale values for a given tank and parameter number.
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

    def run_type_num(self):
        """ Run types defined in the hats.ng_run_types table """
        sql = "SELECT * FROM hats.ng_run_types;"
        r = self.db.doquery(sql)
        results = {entry['name'].lower(): entry['num'] for entry in r}
        results['std'] = 8
        return results

    def return_analysis_nums(self, df, time_col='dt_run'):
        """
        Loops over each row in the DataFrame and queries the database
        for the primary key (num) based on analysis_time and inst_num.
        Returns the DataFrame with a new 'analysis_num' column.
        """
        analysis_nums = []
        
        for _, row in df.iterrows():
            # Adjust the formatting of analysis_time if necessary.
            dt = row[time_col]
            query = f"SELECT num FROM hats.ng_analysis WHERE analysis_time = '{dt}' AND inst_num = {self.inst_num}"
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
    
    @staticmethod
    def convert_date_format(date_str):
        """
        Converts a date string from 'YYYYMMDD' or 'YYYYMMDD.HHMM' format to 'YYMM' format.
        If the date is already in 'YYMM' format, it returns the date as is.

        Parameters:
        date_str (str): The date string in 'YYYYMMDD', 'YYYYMMDD.HHMM', or 'YYMM' format.

        Returns:
        str: The date string in 'YYMM' format.
        """
        # Check if the length of the date string is 4 and assume it is in 'YYMM' format
        if len(date_str) == 4:
            return date_str
        
        # Extract the first 6 characters which correspond to 'YYYYMM'
        yyyymm = date_str[:6]
        
        # Convert to 'YYMM'
        yymm = yyyymm[2:]
        
        return yymm

class M4_GCwerks_Export(M4_base):

    def __init__(self):
        super().__init__()
        self.gcdir = '/hats/gc/m4'

    def export_gc_data(self, start_date, molecules, progress=None):
        """
        Calls the gcexport program for each molecule via the subprocess.Popen function. gcexport
        will extract data from gcwerks and send it to a .csv file.

        start_date should be in the form YYMM
        """
        
        processes = []
        for molecule in molecules:
            if molecule in self.molecules:
                filename = f"data_{molecule}.csv"
                params = f"time runtype tank stdtank port psamp {molecule}.area {molecule}.ht {molecule}.rt {molecule}.w"
                # params_extra = f"{params} {molecule}.skew {molecule}.rl.a {molecule}.rl.ht {molecule}.rl.report {molecule}.c.a {molecule}.c.ht {molecule}.c.report"
                command = f"{self.gcexport_path} {self.gcdir} -csv -nonan -mindate {start_date} {params} > {self.export_dir}/{filename}"

                # Execute the command and redirect output to /dev/null
                process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                processes.append((process, molecule))
            else:
                print(f'Wrong molecule name: {molecule}')

        for process, molecule in tqdm(processes):
            if progress:
                progress(10)
            process.wait()

    @staticmethod
    def verify_start_date(date_str):
        """Verify and convert start date to YYMM format."""
        if re.match(r'^\d{8}$', date_str):
            # Convert YYYYMMDD to YYMM
            return date_str[2:6]
        elif re.match(r'^\d{4}$', date_str):
            # Already in YYMM format
            return date_str
        else:
            raise ValueError(f"Invalid date format: {date_str}. Expected format is YYMM or YYYYMMDD.")            

    @staticmethod
    def parse_molecules(molecules):
        if molecules:
            try:
                molecules = molecules.replace(' ', '')   # remove spaces
                return molecules.split(',')
            except AttributeError:      # already a list. just return
                return molecules
        return []

    @staticmethod
    def main():
        m4_export = M4_GCwerks_Export()
        default_molecules = list(m4_export.molecules)
        
        parser = argparse.ArgumentParser(description='Export Perseus data with specified start date.')
        parser.add_argument('start_date', type=str, nargs='?', default='2201',
                            help='Start date in the format YYMM (default: 2201)')
        parser.add_argument('-m', '--molecules', type=str, default=default_molecules,
                            help='Comma-separated list of molecules. Add quotes around the list if spaces are used. Default all molecules.')
        parser.add_argument('--list', action='store_true', help='List all available molecule names.')

        args = parser.parse_args()
        if args.list:
            print(f"Valid molecule names: {', '.join(m4_export.molecules)}")
            quit()

        molecules = m4_export.parse_molecules(args.molecules)     # returns a list of molecules
        start_date = m4_export.verify_start_date(args.start_date)
        m4_export.export_gc_data(start_date, molecules)


if __name__ == '__main__':
    M4_GCwerks_Export.main()
