#! /usr/bin/env python

import subprocess
import argparse
import sys
import re
from pathlib import Path

class PR1_GCwerks_Export:
    inst_num = 58  # PR1
    PR1_mols = [
        "CF4", "NF3", "C2H6", "PFC-116", "C2H4", "SF6", "CFC-13", "HFC-23", "C2H2",
        "COS", "HFC-32", "SO2F2", "H-1301", "PFC-218", "C3H8", "CFC-115", "HFC-125", "HFC-143a",
        "HCFC-22", "C3H6", "CFC-12", "HFC-134a", "HFO-1234yf", "HFC-134", "CH3Cl", "HFC-152a",
        "HFO-1234zeE", "CS2", "iC4H10", "HFC-227ea", "H-1211", "nC4H10", "CH3Br", "HCFC-142b",
        "HCFC-124", "HFC-236fa", "HCFC-21", "CFC-114", "HCFC-133a", "HFC-245fa", "CFC-11", "CH3I",
        "CH2Cl2", "iC5H12", "morpholine", "nC5H12", "HCFC-141b", "HCFC-123", "CFC-113", "PFTEA",
        "H-1011", "H-2402", "HFC-365mfc", "CHCl3", "nC6H14", "CCl4", "TCE", "CH2Br2", "CH3CCl3",
        "1,2-DCE", "C6H6", "CFC-112", "PCE", "PFTPA", "C7H8", "CHBr3"
    ]

    def __init__(self):
        sys.path.append('/ccg/src/db/')
        import db_utils.db_conn as db_conn
        self.db = db_conn.HATS_ng()
        self.gcexport_path = "/hats/gc/gcwerks-3/bin/gcexport"
        self.export_dir = "/hats/gc/pr1/results"

    def PR1_molecules(self):
        """ This method returns the list of molecules from the HATS database
            except the metadata field. """
        sql = f"SELECT display_name FROM hats.analyte_list WHERE inst_num = {self.inst_num} ORDER BY disp_order;"
        r = self.db.doquery(sql)
        display_names = [item['display_name'] for item in r]
        return display_names

    def export_gc_data(self, start_date, molecules):
        """ Calls the gcexport program for each molecule via the subprocess.Popen function. gcexport
            will extract data from gcwerks and send it to a .csv file. """
        processes = []
        for molecule in molecules:
            if molecule in self.PR1_mols:
                filename = f"data_{molecule}.csv"
                params = f"time runtype tank stdtank port psamp0 psamp T1 {molecule}.area {molecule}.ht {molecule}.rt {molecule}.w"
                # params_extra = f"{params} {molecule}.skew {molecule}.rl.a {molecule}.rl.ht {molecule}.rl.report {molecule}.c.a {molecule}.c.ht {molecule}.c.report"
                command = f"{self.gcexport_path} /data/Perseus-1 -csv -mindate {start_date} {params} > {self.export_dir}/{filename}"

                # Execute the command and redirect output to /dev/null
                process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                processes.append((process, molecule))
            else:
                print(f'Wrong molecule name: {molecule}')

        # Wait for all subprocesses to finish
        for process, molecule in processes:
            process.wait()
            print(f"Process for molecule {molecule} has finished.")

        self.fix_12DCE()

    def fix_12DCE(self):
        """
        The molecule name 1,2-DCE is a problem in the csv format due to the comman in the name.
        Remove the comma in the file name and column header.
        """
        old = Path(f"{self.export_dir}/data_1,2-DCE.csv")
        new = Path(f"{self.export_dir}/data_12-DCE.csv")

        if not old.exists():
            return
        with open(old, 'r') as file:
            lines = file.readlines()
    
        columns = lines[0].replace('1,2-DCE', '12-DCE')
    
        # Write to the new file with the updated header
        with open(new, 'w') as file:
            file.write(columns)
            file.writelines(lines[1:])
        
        old.unlink()
        new.chmod(0o664)

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
    def main():
        parser = argparse.ArgumentParser(description='Export Perseus data with specified start date.')
        parser.add_argument('start_date', type=str, nargs='?', default='2201',
                            help='Start date in the format YYMM (default: 2201)')
        parser.add_argument('-m', '--molecules', nargs='*', default=PR1_GCwerks_Export.PR1_mols,
                            help='List of molecules to process. Default is all molecules.')
        parser.add_argument('--list', action='store_true', help='List all available molecule names.')

        args = parser.parse_args()
        if args.list:
            print(f"Valid molecule names: {', '.join(PR1_GCwerks_Export.PR1_mols)}")
            quit()

        start_date = PR1_GCwerks_Export.verify_start_date(args.start_date)
        pr1_export = PR1_GCwerks_Export()
        pr1_export.export_gc_data(start_date, args.molecules)

if __name__ == '__main__':
    PR1_GCwerks_Export.main()
