#! /usr/bin/env python

import argparse
from datetime import date, timedelta
from pathlib import Path
from subprocess import run
from multiprocessing import Pool, cpu_count
import pandas as pd

import itx_import


class GCwerks_Import:

    def __init__(self, site, options, incoming_dir='chroms_itx'):
        self.options = vars(options)
        self.usesmoothfile = False
        self.site = site
        self.yyyy = self.options['year']
        self.gcdir = f'/hats/gc/{self.site}'
        self.incoming = Path(f'{self.gcdir}/{str(self.yyyy)[2:4]}/{incoming_dir}')
        # GCWerks program to ingest external chromatogram data
        self.chromatogram_import = Path('/hats/gc/gcwerks-3/bin/chromatogram_import')
        
        smoothfile = self.options.get('smoothfile')
        if smoothfile:
            smoothfile_path = Path(smoothfile)
            if smoothfile_path.is_file():
                self.usesmoothfile = True
                sm = itx_import.ITX_smoothfile(smoothfile_path)
                self.params_df = sm.params_df

        # if any command line filtering/smoothing options are set, don't use smoothfile
        if (
            ('s', True) in self.options.items() or
            (self.options.get('boxwidth') is not None and int(self.options['boxwidth']) > -1) or
            ('g', True) in self.options.items() or
            (self.options.get('ws_start') is not None and int(self.options['ws_start']) > -1)
        ):
            self.usesmoothfile = False

    def import_itx(self, itx_file):
        """ Import a single ITX file (all chroms) into GCwerks
            Apply filters and smoothing
        """
        print(itx_file)
        itx = itx_import.ITX(itx_file)   # load itx file
        
        # use either smoothfile or command line options (one or the other).
        if self.usesmoothfile:
            itxtime = pd.to_datetime(itx.name[:11], format='%y%m%d.%H%M')

            # apply different params to each chrom channel.
            for ch in range(itx.chans):
                # find the smoothing parameter date appropriate for the itxfile and channel
                params = self.params_df.loc[(self.params_df.index < itxtime) & (self.params_df.chan == ch)].iloc[-1]
                if params.spike:
                    itx.spike_filter(ch)
                if params.wide_spike:
                    #print(f'   channel {ch} wide spike {params.wide_spike}')
                    itx.wide_spike_filter(ch, start=params.wide_start)
                if params.sg:
                    #print(f'   channel {ch} savitzky golay {params.sg_win}')
                    itx.savitzky_golay(ch, winsize=params.sg_win, order=params.sg_ord)
                elif params.boxwidth:
                    #print(f'   channel {ch} box smooth {params.boxwidth}')
                    itx.box_smooth(ch, winsize=int(params.boxwidth))
        else:
            # apply spike filters before smoothing
            if ('s', True) in self.options.items():
                itx.spike_filter('all')
            if int(self.options['ws_start']) > -1:
                itx.wide_spike_filter('all', start=int(self.options['ws_start']))
            try:
                if self.options['O2_LOCK_CHANS'] and self.options['O2_LOCK_TIMES']:
                    itx.o2_lock(self.options['O2_LOCK_CHANS'], self.options['O2_LOCK_TIMES'])
            except KeyError:
                pass
            # apply Box smoothing
            if self.options['boxwidth'] is not None:
                itx.box_smooth('all', winsize=self.options['boxwidth'])
            # apply Savitzky Golay smoothing
            if ('g', True) in self.options.items():
                _win = self.options['SGwin']
                _ord = self.options['SGorder']
                itx.savitzky_golay('all', winsize=_win, order=_ord)

        # sends the itx data to the chromatogram_import program
        run([self.chromatogram_import, '-gcdir', self.gcdir], input=itx.write(stdout=False), text=True, capture_output=True)

        itx.compress_to_Z(itx_file)

    def import_recursive_itx(self, types=None):
        """Recursively import all .itx and .itx.gz files in self.incoming using multiprocessing."""
        if types is None:
            types = ['*.itx', '*.itx.gz']

        # 1) collect files
        files = [f for pat in types for f in self.incoming.rglob(pat)]
        if not files:
            return False

        # 2) pick a reasonable worker count
        n_workers = min(len(files), cpu_count())

        # 3) process
        with Pool(n_workers) as pool:
            pool.map(self.import_itx, files)

        return True
        
    def load_smoothfile(self, file):
        pass

    def main(self, import_method, *args, **kwargs):
        loaded = import_method(*args, **kwargs)
        if loaded:
            # updates integration and mixing ratios
            run(['/hats/gc/gcwerks-3/bin/run-index', '-gcdir', self.gcdir])
            
            # compute last month’s YYMM
            today = date.today()
            first_of_this_month = today.replace(day=1)
            last_month_date = first_of_this_month - timedelta(days=1)
            yymm = last_month_date.strftime("%y%m")
            
            run([
                '/hats/gc/gcwerks-3/bin/gcupdate',
                '-gcdir', self.gcdir,
                yymm
            ])
            
            run(['/hats/gc/gcwerks-3/bin/gccalc', '-gcdir', self.gcdir])
            

if __name__ == '__main__':

    SGwin, SGorder = 61, 4      # Savitzky Golay default variables
    WSTART = -1                 # Wide spike filter start time
    yyyy = date.today().year

    opt = argparse.ArgumentParser(
        description='Imports ITX chromatogram files into GCwerks integration program.'
    )
    opt.add_argument('-s', action='store_true', default=False,
                     help='Apply 1-point spike filter (default off)')
    opt.add_argument('-W', action='store', dest='ws_start', default=WSTART,
                     help='Apply wide spike filter (default off)')
    opt.add_argument('-b', action='store', dest='boxwidth', metavar='Win', type=int,
                     help=f'Apply a Box smooth with window width in points')
    opt.add_argument('-g', action='store_true', default=False,
                     help='Apply Savitzky Golay smoothing (default off)')
    opt.add_argument('-gw', action='store', dest='SGwin', metavar='Win', default=SGwin, type=int,
                     help=f'Sets Savitzky Golay smoothing window (default: {SGwin} points)')
    opt.add_argument('-go', action='store', dest='SGorder', metavar='Order', default=SGorder, type=int,
                     help=f'Sets Savitzky Golay order of fit (default = {SGorder})')
    opt.add_argument('-file', action='store', dest='smoothfile', type=str,
                     help='Use the smoothing parameters defined in this file.')
    opt.add_argument('--year', action='store', default=yyyy, dest='year',
                     help=f'the year (default is current year: {yyyy})')
    opt.add_argument('-reimport', action='store_true', default=False,
                     help='Reimport all itx files including .Z archived.')
    opt.add_argument('site', help='A station or "all" for all CATS sites.')

    options = opt.parse_args()
        
    if options.site.lower() == 'all':
        for s in ('brw', 'nwr', 'mlo', 'smo', 'spo'):
            werks = GCwerks_Import(s, options)
            print(f'Working on {s}')
            werks.main()
    else:
        werks = GCwerks_Import(options.site, options)
        if options.smoothfile:
            werks.load_smoothfile(options.smoothfile)
            
        if options.reimport:
            types = ('*.itx', '*.itx.gz', '*.itx.Z')
            werks.main(import_method=werks.import_recursive_itx, types=types)
        else:
            werks.main(werks.import_recursive_itx)
