#! /usr/bin/env python

import argparse
import pandas as pd
from pandas.tseries.offsets import DateOffset
import sys
sys.path.append('/ccg/src/db/')

import fe3_merge
from fe3_gcwerks2db import FE3_instrument
import db_utils.db_conn as db_conn # type: ignore

# HATS db class created by John Mund
db = db_conn.HATS_ng()


class FE3_DataProg2db(FE3_instrument):
    """ Class to load the fe3_db.csv file and upload changes to the HATS Sql DB """

    def __init__(self):
        super().__init__()
    
    @staticmethod
    def detrend_method(method: str) -> int:
        """ Returns the detrend method number. Currently only using two methods, lowess or point-to-point """
        m = 2 if method == 'lowess' else 1
        return m  

    @staticmethod
    def return_run_type_num(t: str, db_types: pd.DataFrame):
        """ method returns the run type number as defined in the hats.ng_run_types table """
        t = 'Calibration' if t == 'cal' else t.capitalize()
        try:
            return db_types.loc[db_types.name == t, 'num'].values[0]
        except IndexError:
            print(f'Bad run type: {t}')
            return None   

    @staticmethod
    def split_pairid_flaskid(id_str: str):
        """ method splits a pairid-flaskid string. 
            If a serial number of a tank is in this field it with leave it as the pairid variable. 
            Added portinfo field 230620 """
        if str(id_str).find('-') > 0:
            pairid, flaskid = id_str.split('-')
            try:
                test = int(pairid)
                return pairid, flaskid, 'flask'
            except ValueError:
                pass
        return None, None, id_str
    
    def _update_ng_analysis(self, analysis_num, df_row):
        r = df_row[1]['dir']
        run_time = f'{r[0:4]}-{r[4:6]}-{r[6:8]} {r[9:11]}:{r[11:13]}:{r[13:15]}'
        run_type = self.return_run_type_num(df_row[1]['type'], self.runtypes_df)
        port = int(df_row[1]['port'])
        pair_id_num, flask_id, port_info = self.split_pairid_flaskid(df_row[1]['port_id'])
        try:
            flask_port = int(df_row[1]['flask_port'])
        except ValueError:
            flask_port = None     # null

        cmd = f"""UPDATE hats.ng_analysis 
          SET run_time = '{run_time}', 
              run_type_num = {run_type}, 
              port = {port}, 
              port_info = '{port_info}',
              pair_id_num = '{pair_id_num}', 
              flask_id = '{flask_id}', 
              flask_port = '{flask_port}'
          WHERE num = {analysis_num} and inst_num = {self.inst_num} """
          
        # use NULL instead of 'None' for mysql
        cmd = cmd.replace("'None'", "NULL")
        
        db.doquery(cmd)

    def fe3data_2_hatsdb(self, df: pd.DataFrame):
        """ Method to add run_time, detrend method, and mole_fractions to the HATS DB tables. 
            First run gcwerks_2_hatsdb to insert new rows in the ng_analysis table then run this function. """

        m_sql = """insert hats.ng_mole_fractions (analysis_num, parameter_num, channel, mole_fraction, flag, detrend_method_num) 
            values (%s,%s,%s,%s,%s,%s) on duplicate key update 
            parameter_num=values(parameter_num), channel=values(channel), mole_fraction=values(mole_fraction),
            flag=values(flag), detrend_method_num=values(detrend_method_num)"""
        
        for row in df.iterrows():
            """ Step through each row of a DataFrame with fe3_data results. """
            r = row[1]
            a_time = str(r.time)[0:19]    # analysis time

            # analysis time uniquely defines a record in hats.ng_analysis table.
            cmd = f"select num from hats.ng_analysis where analysis_time = '{a_time}'"
            n = db.doquery(cmd)

            if n is None:
                # No record found. Create new record.
                print(f'Missing analysis_num {a_time}')
                return
            else:
                analysis_num = n[0]['num']

            # first update the ng_analysis table
            self._update_ng_analysis(analysis_num, row)

            params = []

            # update the ng_mole_fractions table
            for ch, mol_list in self.fe3db_chans.items():
                for mol in mol_list:
                    pn = self.param_num[mol]
                    # the pandas dataframe uses the channel in the molecule definition for CFC11 and 113
                    mol = 'MC' if mol == 'CH3CCl3' else mol
                    mol = f'{mol}{ch}' if mol == 'CFC11' else mol
                    mol = f'{mol}{ch}' if mol == 'CFC113' else mol
                    #method = row[1][f'{mol}_methcal']   # need to add to table
                    try:
                        mole_fraction = row[1][f'{mol}_value']
                        test = int(mole_fraction)   # this triggers an error if mole_fraction is messed up.
                        if abs(mole_fraction) > 1000000: 
                            mole_fraction = None
                    except (ValueError, OverflowError) as e:
                        mole_fraction = None

                    # The flag value in fe3_db.csv is a boolean.
                    flag = row[1][f'{mol}_flag']
                    flag_str = '...'
                    if flag:
                        # retrieve current flag feild from db
                        #cmd = f"select flag from hats.ng_mole_fractions where analysis_num = '{analysis_num}'"
                        #old_flag = db.doquery(cmd)
                        # update flag feild's first character to M for manual flag
                        #new_flag = 'M' + old_flag[1:]
                        flag_str = 'M..'

                    det = self.detrend_method(row[1][f'{mol}_methdet'])
                    
                    params0 = [analysis_num, pn, ch, mole_fraction, flag_str, det]
                    params.append(params0)
                        
                    if db.doMultiInsert(m_sql, params): params=[]

            r = db.doMultiInsert(m_sql, params, all=True)
                

if __name__ == '__main__':
    import time

    opt = argparse.ArgumentParser(
        description="""Syncs mole fraction data calculated by fe3_data.py to the HATS DB. 
            By default, only the last two months of data are synced to the HATS DB. Use
            the -a or -y options to work on larger sets of fe3 data. """
    )
    opt.add_argument("-a", "--all", action="store_true",
                     dest="allyears", help="process all of the data (all years)")
    opt.add_argument("-y", "--year", action="store", default=None,
                     dest="yyyy", help=f"operate on a years worth of fe3_data.py results.")
    
    options = opt.parse_args()

    t0 = time.time()
    data = FE3_DataProg2db()
    fe3 = fe3_merge.FE3_db()

    # full record
    df = fe3.db
    
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
            
    data.fe3data_2_hatsdb(df.reset_index())
    print(f'Execution time {time.time()-t0:3.3f} seconds on {df.shape[0]} records.')
