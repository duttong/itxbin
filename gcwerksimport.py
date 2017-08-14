#! /usr/bin/env python
''' Added itx_import filters to optional arguments.  GSD 150227 '''

from glob import glob
import argparse
import os
import gzip
import shutil
import multiprocessing
from datetime import date

from catsbase import CATS_Chromatograph
cats = CATS_Chromatograph()

cats_sites = cats.sites
today = date.today()

class GCwerks_import():

    def __init__(self, site, options):
        self.site = site
        self.options = options
        self.path = '/hats/gc/'
        if 'yyyy' not in self.options:
            self.options['yyyy'] = today.year
        yy = self.options['yyyy'][2:] if len(self.options['yyyy']) == 4 else self.options['yyyy']
        self.chromsdir = self.path + self.site + '/' + yy + '/chroms_itx/'
        self.permission()

    def permission(self):
        """ Used to prevent other hats member from running command on CATS data. """
        if os.environ['LOGNAME'] != 'gdutton' and (self.site in cats_sites or self.site == 'all'):
            print("Sorry, you don't have permission to run this command on site: {}".format(self.site))
            quit()

    def import2gcwerks(self):
        """ Importing routine.
    
            Import *.itx and *.gz (unless options.all is selected).  Once imported
            compress *.itx files and rename with .Z extension.
        
            Excludes cal3 injections.  Will need a different routine for c1_c, c2_c and c3_c 
   
            /hats/gc/itxbin/itx_import.py -g 2012nwr1672338.a1.itx.gz | /hats/gc/gcwerks-3/bin/chromatogram_import -gcdir /hats/gc/nwr
        """
        
        if self.site == 'bld1' or self.site == 'agc1':
            files = glob(self.chromsdir+'*.itx') + glob(self.chromsdir+'*.itx.gz')
            if ('all', True) in self.options.items():
                files += glob(self.chromsdir+'*.itx.Z')
        elif self.site == 'std' or self.site == 'stdhp':
            ''' Standard GCs '''
            files = glob(self.chromsdir+'*.itx') + glob(self.chromsdir+'*.itx.gz')
            if ('all', True) in self.options.items():
                files += glob(self.chromsdir+'*.itx.Z')
        else:
            ''' CATS sites '''
            files = glob(self.chromsdir+'*.??.itx') + glob(self.chromsdir+'*.??.itx.gz')
            if ('all', True) in self.options.items():
                files += glob(self.chromsdir+'*.??.itx.Z')

        cmd = self.path + 'itxbin/itx_import.py '
        werks = '{}gcwerks-3/bin/chromatogram_import -gcdir {}{} > /dev/null 2>&1'.format(self.path, self.path, self.site)
            
        # apply spike filter before SG smoothing
        if ('s', True) in self.options.items():
            cmd += '-s '
            if ('v', True) in self.options.items():
                print('Applying single point spike filter')
        
        # apply wide spike filter?
        ws = self.options.get('ws_start')
        if ws is not None and ws > 0:
            cmd += '-W {} '.format(self.options.ws_start)
            if ('v', True) in self.options.items():
                print('Applying wide spike filter starting at time={}'.format(self.options.ws_start))

        # apply Savitzky Golay smoothing    
        if ('g', True) in self.options.items():
            win = self.options.get('SGwin')
            ord = self.options.get('SGorder')
            cmd += '-g -gw {} -go {} '.format(win, ord)
            if ('v', True) in self.options.items():
                print('Applying Savitzky Golay smoothing: {}, {}'.format(win, ord))

        if len(files) > 0:    
            print('Importing {} files from {}.'.format(len(files), self.site))
            pool = multiprocessing.Pool()
            for file in files:
                pool.apply_async(self._process, args=(file, cmd, werks))
            pool.close()
            pool.join()

    def _process(self, file, cmd, werks):
        os.system(cmd + " " + file + " | " + werks)
        if ('v', True) in self.options.items():
            print('   {}'.format(os.path.basename(file)))
                
    def compress_to_Z(self, file):
        with open(file, 'rb') as f_in, gzip.open(file+'.Z', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            os.remove(file)
        
    def rename(self):
        """ Gzip *.itx files in the chroms_itx directory. 
    
            The files are also give the .Z suffix so that they will
            not be re-imported.  Also rename any .gz files to .Z suffix.
        """
                
        """ compress .itx files to .Z files """
        files = glob(self.chromsdir + '*.itx')
        if len(files) > 0:
            print('Compressing *.itx files.')
            with multiprocessing.Pool() as pool:
                for _ in pool.imap_unordered(self.compress_to_Z, files, chunksize=1):
                    pass
        
        """ rename .gz file to .Z files """
        files = glob(self.chromsdir + '*.gz')
        if len(files) > 0:
            print('Renameing *.gz files')
            for file in files:
                os.rename(file, file[:-2]+'Z')
    
        
    def main(self):
        self.import2gcwerks()
        self.rename()
    
        # recreate index files for GCwerks -- not needed.  GCwerks adds to the index file.
        os.system('/hats/gc/itxbin/quickindex.py ' + self.site)
        #os.system('/hats/gc/gcwerks-3/bin/run-index -gcdir /hats/gc/' + site)

        # updates integration and mixing ratios
        os.system('/hats/gc/gcwerks-3/bin/gcupdate -gcdir /hats/gc/' + self.site)
        os.system('/hats/gc/gcwerks-3/bin/gccalc -gcdir /hats/gc/' + self.site + ' -1')


if __name__ == '__main__':

    SGwin, SGorder = 61, 4      # Savitzky Golay default variables
    WSTART = -1                 # Wide spike filter start time

    opt = argparse.ArgumentParser(
        description="Imports itx files into GCwerks."
    )
    opt.add_argument('--all', action="store_true", default=False,
                     help="re-import all itx files in chroms_itx directory.")
    opt.add_argument('-s', action="store_true", default=False,
                     help='Apply 1-point spike filter (default=False)')
    opt.add_argument('-W', action="store", dest='ws_start', default=WSTART,
                     help='Apply wide spike filter (default off)')
    opt.add_argument('-g', action="store_true", default=False,
                     help='Apply Savitzky Golay smoothing (default off)')
    opt.add_argument('-gw', action="store", dest='SGwin', metavar='Win',
                     default=SGwin, type=int,
                     help='Sets Savitzky Golay smoothing window (default: {} points)'.format(SGwin))
    opt.add_argument('-go', action="store", dest='SGorder', metavar='Order',
                     default=SGorder, type=int,
                     help='Sets Savitzky Golay order of fit (default = '+str(SGorder)+')')
    opt.add_argument('-v', action="store_true", default=False,
                     help='verbose output.')
    opt.add_argument("--year", action="store", default=str(today.year), dest="yyyy", 
                     help="the year (default is current year: {})".format(today.year))
    opt.add_argument("site", help="A station or 'all' for all CATS sites.")
            
    options = opt.parse_args()
    
    if options.site.lower() == 'all':
        for s in cats_sites:
            werks = GCwerks_import(s, vars(options))
            print('Working on {}'.format(s))
            werks.main()
    else:
        werks = GCwerks_import(options.site, vars(options))
        werks.main()
