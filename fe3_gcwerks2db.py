#! /usr/bin/env python

import argparse
import pandas as pd
from pandas.tseries.offsets import DateOffset
import sys
sys.path.append('/ccg/src/db/')

import fe3_merge
import db_utils.db_conn as db_conn
db = db_conn.HATS_ng()

class FE3_instrument:
    """ Instrument specific definitions. """

    def __init__(self):
 
        self.instrument = 193   # FE3 instrument number (changed 240520)
        self.param_num = {}     # db parameter numbers
        self.fe3db_chans = {}   # db channel identifier and gases
        self.mols = []          # list of molecules on FE3

        # query the db for fe3 parameters        
        cmd = f'SELECT display_name, param_num, channel \
            FROM hats.analyte_list where inst_num = {self.instrument}'
        df = pd.DataFrame(db.doquery(cmd))

        # a dict of channels and a list of molecules.
        for ch in df.channel.unique():
            self.fe3db_chans[ch] = list(df.loc[df.channel == ch].display_name)

        # a dict of molecules and parameter number
        for mol in df.display_name.unique():
            self.param_num[mol] = int(list(df.loc[df.display_name == mol].param_num)[0])

        self.mols = list(df.display_name.unique())
        self.mols.sort()
        self.runtypes_df = self.fe3_run_types()
        self.detrend_df = self.fe3_detrend_methods()

    def fe3_param_numbers(self, mols: list) -> dict:
        """ paramater numbers for the gases FE3 measures, retrieved from the DB 
            NOTE: this info is now coming from the hats.analyte_list """
        pnum = {}
        for mol in mols:
            cmd = f"SELECT * FROM gmd.parameter WHERE formula='{mol}';"
            q = db.doquery(cmd)[0]
            pnum[mol] = q['num']
        return pnum

    def return_preferred_channel(self, gas: str) -> str:
        """ Function returns the preferred channel letter code """
        gas = gas.lower()
        if gas == 'cfc11': return 'c'
        if gas == 'cfc113': return 'c'
        for ch, mols in self.fe3db_chans.items():
            for mol in mols:
                if mol.lower() == gas:
                    return ch
        return None
    
    @staticmethod
    def fe3_run_types():
        cmd = "SELECT * FROM hats.ng_run_types;"
        return pd.DataFrame(db.doquery(cmd))
    
    @staticmethod
    def fe3_detrend_methods():
        cmd = "SELECT * FROM hats.ng_detrend_methods;"
        return pd.DataFrame(db.doquery(cmd))


class FE3_GCwerks2db(FE3_instrument):
    """ Class for syncing GCwerks results for FE3 to the HATS DB. The GCwerks data is exported
        with the fe3_export.py program to a raw data file /hats/gc/fe3/results/fe3_gcwerks_all.csv """

    def __init__(self):
        super().__init__()

    def insert_ng_analysis(self, atime: str, port: int, inst_num: int) -> int:
        """ Inserts a new row into hats.ng_analysis table """
        a_sql = """insert hats.ng_analysis (analysis_time, port, inst_num) values (%s,%s,%s) """
        params = (atime, port, inst_num)
        rec = db.doquery(a_sql, params, insert=True)
        return rec

    def insert_ng_mole_fractions(self, analysis_num: int, row: list):
        """ Inserts or updates a record in the ng_mole_fractions table """
        params = []
        
        m_sql = """insert hats.ng_mole_fractions (analysis_num, parameter_num, channel, height, area, retention_time) 
            values (%s,%s,%s,%s,%s,%s) on duplicate key update 
            parameter_num=values(parameter_num), channel=values(channel), height=values(height),
            area=values(area), retention_time=values(retention_time)"""

        for ch, mol_list in self.fe3db_chans.items():
            """ Step through each channel """
            for mol in mol_list:
                parameter_num = self.param_num[mol]
                
                # the pandas dataframe uses the channel in the molecule definition for CFC11 and 113
                mol = 'MC' if mol == 'CH3CCl3' else mol
                mol = f'{mol}{ch}' if mol == 'CFC11' else mol
                mol = f'{mol}{ch}' if mol == 'CFC113' else mol

                #method = row[f'{mol}_methcal']   # need to add to table
                height = row[f'{mol}_ht']
                area = row[f'{mol}_area']
                retention_time = row[f'{mol}_rt']
                params0 = [analysis_num, parameter_num, ch, height, area, retention_time]
                params.append(params0)
                #print('updating ng_mole_faction', mol, params0)
                if db.doMultiInsert(m_sql, params): params=[]
            
        r = db.doMultiInsert(m_sql, params, all=True)

    def gcwerks_2_hatsdb(self, df: pd.DataFrame):
        """ Method to insert new records and update existing records in the HATS DB tables.
            The dataframe is GCwerks data from fe3_gcwerks_all.csv """
        
        for row in df.iterrows():
            """ Step through each row of a DataFrame with GCwerks results. """
            r = row[1]
            a_time = str(r.time)    # analysis time
            port = int(r.port)      # SSV port

            # analysis time uniquely defines a record in hats.ng_analysis table.
            cmd = f"select num from hats.ng_analysis where analysis_time = '{a_time}'"
            n = db.doquery(cmd)

            if n is None:
                # No record found. Create new record.
                analysis_num = self.insert_ng_analysis(a_time, port, self.instrument)
                print(f'new row: {analysis_num} at {a_time}')
            else:
                analysis_num = n[0]['num']

            # With analysis_num found or inserted, insert or update GCwerks results.
            #print(analysis_num)
            self.insert_ng_mole_fractions(analysis_num, r)


if __name__ == '__main__':
    import time

    opt = argparse.ArgumentParser(
        description="""Load GCwerks results for FE3 into the HATS DB.
            The default behaviour is to work on the last two months of the exported data stored in
            /hats/gc/fe3/results/fe3_gcwerks_all.csv file."""
    )
    opt.add_argument("-a", "--all", action="store_true",
                     dest="allyears", help="process all of the data (all years)")
    opt.add_argument("-y", "--year", action="store", 
                     dest="yyyy", help=f"operate on a years worth of GCwerks results.")
    
    options = opt.parse_args()
    
    t0 = time.time()
    fe3 = FE3_GCwerks2db()
    werks = fe3_merge.FE3_GCwerks()

    # full record
    df = werks.gcwerks_df()
    
    if options.allyears:
        pass
    elif options.yyyy:
        df = df.loc[df.index.year == int(options.yyyy)]
    else:
        # Determine the most recent date in the DataFrame index
        last_date = df.index.max()

        # Calculate the date 2 months earlier using DateOffset
        start_date = last_date - DateOffset(months=2)

        # Filter the DataFrame to include rows from the last 2 months
        df = df.loc[start_date:]        
        #df = df.last('2M') # depricated

    fe3.gcwerks_2_hatsdb(df.reset_index())
    print(f'Execution time {time.time()-t0:3.3f} seconds on {df.shape[0]} records.')
    