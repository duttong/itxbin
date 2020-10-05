#! /usr/bin/env python

""" These classes should be db into the fe3_export.py file. GSD """

import os
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import json


class FE3_paths:

    def __init__(self):
        self.basepath = Path('/Users/geoff/programing/agc1')


class FE3_runs(FE3_paths):
    """ Class to handle data specific to each FE3 run. Meta data such as
        which tanks and flasks are on a specific port and the sequence order. """

    current_year = date.today().year

    def __init__(self):
        super().__init__()

    def return_2char_year(self, year):
        """ Method to check if year is 4 or 2 digits and return a
            two digit year """
        if len(str(year)) == 4:
            yy = str(year)[2:]
        elif len(str(year)) == 2:
            yy = year
        else:
            print(f'year variable {year} is incorrect')
            quit()
        return yy

    def runs_list(self, fullpath=False, year=current_year):
        """ Returns a list of FE3 run dates for a given year that is
            defined in self.incoming.
        """
        yy = self.return_2char_year(year)
        incoming = self.basepath / f'{yy}' / 'incoming'
        if fullpath:
            runs = list(incoming.glob('*-*'))
        else:
            runs = [run.name for run in incoming.glob('*-*')]
        return sorted(runs)

    def runs_list_allyears(self, fullpath=False):
        incoming = self.basepath
        if fullpath:
            runs = list(incoming.rglob('incoming/*-*'))
        else:
            runs = [run.name for run in incoming.rglob('incoming/*-*')]
        return sorted(runs)

    @staticmethod
    def run_type(runseq):
        # use upper to find 'f' and 'F'
        type = 'flask' if runseq.upper().find('F') > 0 else 'other'
        return type

    @staticmethod
    def return_datetime(metafile):
        """ Returns a datatime from the metafile name.
            for example: meta_20200207-183526.json
        """
        m = metafile.name
        dt = pd.to_datetime(m[5:20], infer_datetime_format=True)
        return dt

    def runs_df_yy(self, generate=True, year=current_year):
        """ Returns a pandas dataframe with info from meta files for one year.

            Set generate to False to load from previous saved .pkl file, this
            will be much faster when there are a lot of FE3 runs.
        """
        yy = self.return_2char_year(year)
        incoming = self.basepath / f'{yy}' / 'incoming'
        outgoing = self.basepath / f'{yy}' / 'fe3_runs.pkl'

        if not generate:
            """ load from pickle file instead of regenerating """
            if outgoing.exists():
                # need code to see if pickle file is old and needs
                # to be regenerated.
                # pkl_date = os.path.getmtime(outgoing)
                return pd.read_pickle(outgoing)
            else:
                pass

        seq, types, flasks, ports, dirs, dates = [], [], [], [], [], []
        for meta in incoming.rglob('meta_*.json'):
            # load data from meta file
            with open(meta) as f:
                data = json.load(f)

            # process meta file data
            seq.append(data[0])
            types.append(self.run_type(data[0]))
            flasks.append(list(data[1].values()))
            ports.append(list(data[2].values()))
            dirs.append(meta.name[5:20])
            dates.append(self.return_datetime(meta))

        # build up dataframe
        df = pd.DataFrame()
        df['type'] = types
        df['dir'] = dirs
        df['ports'] = ports
        df['flasks'] = flasks
        df['seq'] = seq
        df['time'] = dates
        df = df.set_index('time').sort_index()
        df.to_pickle(outgoing)     # save to pickle file

        return df

    def runs_df(self):
        """ Returns a pandas dataframe with info from meta files for all years.
            The method loads previous years from .pkl files and the current
            year from directories in yy/incoming
            Remove the .pkl files and the method will regenerate them.
        """
        years = sorted([dir.name for dir in self.basepath.glob('??')])
        most_recent_year = years.pop(-1)
        dfs = []
        for year in years:
            df = self.runs_df_yy(generate=False, year=year)
            dfs.append(df)
        # regenerate last years df in case new data
        df = self.runs_df_yy(generate=True, year=most_recent_year)
        dfs.append(df)
        df = pd.concat(dfs, axis=0)
        return df


class FE3_cal_curves(FE3_paths):
    """ Class for handling a database of cal curves. """

    def __init__(self):
        super().__init__()
        self.calcurves_file = self.basepath / 'fe3_calcurves.csv'
        self.calcurves_df = self.load()

    def load(self):
        return pd.read_csv(self.calcurves_file, index_col='dir')

    def save(self):
        self.calcurves_df.to_csv(self.calcurves_file)


class FE3_GCwerks(FE3_paths):
    """ Class and methods for reading GCwerks result file.
        Currently export GCwerks results into a single file for all years,
        may need to break into individual years for loading performance. """

    def __init__(self):
        super().__init__()
        self.gcwerksexport = self.basepath / 'fe3_gcwerks_all.csv'

    def gcwerks_df(self, range='all'):
        # range can be a yyyy-mm-dd string to return a subset of the df
        df = pd.read_csv(self.gcwerksexport,
            index_col=0, skipinitialspace=True, parse_dates=True)
        if range == 'all':
            return df
        else:
            return df[range]


class FE3_cals(FE3_runs):
    """ Class and methods for reading calibration tank DB which is
        currently a .csv file.
        NOTE: the molecule names need to match the names from GCwerks """

    def __init__(self):
        super().__init__()
        self.calibration_values = self.basepath / 'fe3_cals.csv'
        self.cals = self.load_cals()

    def load_cals(self):
        return pd.read_csv(self.calibration_values, index_col=0, skipinitialspace=True, parse_dates=True)


class FE3_db(FE3_runs, FE3_GCwerks):

    def __init__(self):
        FE3_runs.__init__(self)
        FE3_GCwerks.__init__(self)
        self.dbfile = self.basepath / 'fe3_db.csv'
        self.db = self.return_db_file()
        self.mols = [item[:-3] for item in self.db.columns if '_ht' in item]

    @staticmethod
    def _port_id(row):
        """ assign port_id by cal SSV position where row is from a
            gcwerks dataframe """
        ssv = row['port'] % 10
        try:
            pname = row['ports'][ssv]
        except TypeError:
            pname = ''
        return pname

    @staticmethod
    def _seq2list(df):
        """ This method takes the seq string and returns flask ids and port
            position in two lists. """
        seq = df.iloc[0].seq[1:]
        flasks = df.iloc[0].flasks
        ports = df.iloc[0].ports
        des, num = [], []
        inc = 0
        for n, s in enumerate(seq):
            # if the len of the df is shorter than seq, the run was stopped prematurely
            if n == df.shape[0]:
                break
            if s == 'F':    # new flask from flask list in run_df
                fl = flasks[inc]
                des.append(fl)
                num.append(inc)
                inc += 1
            elif s == 'f':  # repeat the same flask
                des.append(fl)
                num.append(inc-1)
            else:
                ssv = df.iloc[n].port % 10
                des.append(ports[ssv])
                num.append('')
        return des, num

    def merge_gcwerks_and_metadata(self):
        """ Load two streams of data. Meta data from FE3 and GCwerks results, then
            merges into a single file (self.dbfile) """

        fe3_runs = self.runs_df()
        gcwerks = self.gcwerks_df()

        # merge the two streams of data
        # if runs are longer than 24 hours, this may not work.
        df = pd.merge_asof(gcwerks, fe3_runs,
            on='time', direction='backward', tolerance=pd.Timedelta('24h'))

        # add port_id and flask_port to dataframe
        df['port_id'] = df.apply(self._port_id, axis=1)
        df['flask_port'] = ''

        # step through all flask runs and update port_id with flask_port
        flask_runs = df.loc[df.type == 'flask'].dir.unique()
        for run in flask_runs:
            port_id, port_num = self._seq2list(df.loc[df.dir == run])
            df.loc[df.dir == run, 'port_id'] = port_id
            df.loc[df.dir == run, 'flask_port'] = port_num

        # don't need these columns anymore
        df.drop(columns=['ports', 'flasks', 'seq'], inplace=True)

        # need to create flag calculation method columns
        for mol in self.mols:
            flag = f'{mol}_flag'
            meth = f'{mol}_meth'
            mf = f'{mol}_value'
            unc = f'{mol}_unc'
            df[flag] = False
            df[meth] = 'lowess;quadratic'
            df[mf] = np.nan
            df[unc] = np.nan

        df = self.cleanup_db_df(df)
        self.db = df

        return df

    def cleanup_db_df(self, df, dropcols=True):
        """ Sorts columns by molecule name but leaves the main columns at the
            front of the df """

        df = df.reset_index()
        if dropcols:
            # drop area and retention time columns to save space.
            droplist = []
            for mol in self.mols:
                droplist.append(f'{mol}_area')
                droplist.append(f'{mol}_rt')
            cols = df.columns.difference(droplist)
            df = df[cols]

        # rearrange DataFrame
        cols = list(df.columns)
        first = ['time', 'port', 'port_id', 'flask_port', 'type', 'dir']
        for item in first:
            cols.remove(item)
        cols = first + sorted(cols, key=str.casefold)
        df = df[cols]

        df = df.loc[~((df.type != 'flask') & (df.type != 'other'))]
        df = df.set_index('time')
        df = df.drop(['index'], axis=1)
        df = df.tz_localize('utc')  # set time zone

        return df

    def update_db(self):

        df = self.merge_gcwerks_and_metadata()
        db = self.db_df()   # original saved db
        db = db.loc[~db.duplicated()]

        attribs = ['_flag', '_meth', '_value', '_unc']
        cols = [f'{mol}{at}' for mol in self.mols for at in attribs]

        # preseve data from original db into extended df
        idx = df.index.intersection(db.index)
        df.loc[idx, cols] = db.loc[idx, cols]

        self.db = df

        return df

    def db_df(self):
        """" Returns the current saved FE3 db """
        return pd.read_csv(self.dbfile, index_col=0, skipinitialspace=True, parse_dates=True)

    def save_db_file(self):
        """ Save to csv file """
        self.db.to_csv(self.dbfile)

    def return_db_file(self):
        """ Three possibilities:
            1) if the db file does not exist, create it from GCwerks data and
               the FE3 meta data.
            2) if the db exists and is the same date or newer than the GCwerks
               integration results, return the saved db file.
            3) if the db exists and is older than the GCwerks file, then presumably
               there is new integrations or more runs added. The db is extended
               while flags, methods, etc are preserved.
            TODO: The method works on all years of FE3 data and will progressively
                  get slower. Maybe only work on the last two years of data?
        """

        # load the fe3_db file if same date or newer than the gcwerk exported data.
        if self.dbfile.exists():
            gcw_date = os.path.getmtime(self.gcwerksexport)
            out_date = os.path.getmtime(self.dbfile)
            if out_date >= gcw_date:
                df = self.db_df()
            else:
                df = self.update_db()
                self.save_db_file()
        else:
            # if db does not exist, create it.
            df = self.merge_gcwerks_and_metadata()
            self.save_db_file()

        return df


if __name__ == '__main__':

    fe3 = FE3_db()
    df = fe3.results()
    print(df)
