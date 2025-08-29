#! /usr/bin/env python

import pandas as pd
import argparse
import time


from logos_instruments import FE3_Instrument

class FE3_batch(FE3_Instrument):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def update_cal_curves(self):
        pass
    
    def update_runs(self, pnum, channel=None, start_date=None, end_date=None):
        """ Calculates mole_fraction for a range of dates.
            Updates the ng_mole_fractions table """
        
        EXCLUDE_PORT = 9   # no need to calculate mf for the "Push port"
        
        gas = self.analytes_inv[pnum]
        if channel is None:
            channel = self.return_preferred_channel(gas)
                
        df = self.load_data(
            pnum=pnum,
            channel=channel,
            start_date=start_date,
            end_date=end_date
        )
        if self.data.empty:
            return pd.DataFrame()
        
        df = df.loc[df['port'] != EXCLUDE_PORT].copy()
        df['mole_fraction'] = self.calc_mole_fraction(df)
        df.loc[df['height'] == 0, 'mole_fraction'] = 0     # set mole_fraction to 0 if height = 0
        return df
    
    def update_mole_fraction_table(self, df, batch_size=500):
        """ Update the mole_fraction data in ng_mole_fractions """
        sql = """
            INSERT INTO hats.ng_mole_fractions (
                analysis_num,
                parameter_num,
                channel,
                mole_fraction
            ) VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                mole_fraction = VALUES(mole_fraction);
        """
        params = []
        for r in df.itertuples(index=False):
            analysis_num = r.analysis_num
            parameter_num = r.parameter_num
            channel = r.channel
            mole_fraction = None if pd.isna(r.mole_fraction) else r.mole_fraction
            params.append((analysis_num, parameter_num, channel, mole_fraction))
                
        for i in range(0, len(params), batch_size):
            batch = params[i : i + batch_size]
            self.db.doMultiInsert(sql, batch, all=True)
            
    def main(self):
        parser = argparse.ArgumentParser(
            description="Process FE3 data and optionally plot results"
        )
        parser.add_argument(
            '-p', '--parameter-num',
            type=str,  # Change to str to allow "all"
            required=True,
            help="Parameter number or 'all' to process all analytes"
        )
        parser.add_argument(
            '-c', '--channel',
            type=str,
            required=False,
            help="GC Channel (a, b, c)"
        )
        parser.add_argument(
            '-s', '--start-date',
            type=str,
            help="Start date in YYMM format (e.g. '2503')"
        )
        parser.add_argument(
            '-e', '--end-date',
            type=str,
            help="End date in YYMM format (e.g. '2505')"
        )
        parser.add_argument(
            '-f', '--figures',
            action='store_true',
            help="Show figures if provided, otherwise no figures"
        )
        parser.add_argument(
            '-i', '--insert',
            action='store_true',
            help="Insert mole fractions into the database if provided"
        )
        args = parser.parse_args()

        t0 = time.time()
        
        if args.parameter_num.lower() == "all":
            # Process all analytes
            sql = f"SELECT param_num, channel, display_name FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
            adf = pd.DataFrame(self.doquery(sql))
            for r in adf.itertuples(index=False):
                pnum = int(r.param_num)
                ch = r.channel
                gas = r.display_name
                print(f"Processing analyte: {gas} (Parameter {pnum} channel {ch})")
                
                df = self.update_runs(pnum, channel=ch, start_date=args.start_date, end_date=args.end_date)
                self.update_mole_fraction_table(df)

            print(f"Processing complete for all analytes. Total time: {time.time() - t0:.2f} seconds")
        else:
            # Process a single parameter
            pnum = int(args.parameter_num)
            gas = self.analytes_inv[pnum]
            ch = args.channel
            if ch is None:
                ch = self.return_preferred_channel(gas)
            
            df = self.update_runs(pnum, channel=ch, start_date=args.start_date, end_date=args.end_date)
            self.update_mole_fraction_table(df)
                            
            print(f"Processing complete for {df.shape[0]} rows. Total time: {time.time() - t0:.2f} seconds")  
        
if __name__ == "__main__":
    fe3 = FE3_batch()
    fe3.main()