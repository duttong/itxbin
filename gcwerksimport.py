#! /usr/bin/env python
''' Added itx_import filters to optional arguments.  GSD 150227 '''

from glob import glob
import argparse
import os
from datetime import date

cats_sites = ('brw', 'sum', 'nwr', 'mlo', 'smo', 'spo')
path = '/hats/gc/'


def permission(site):
    """ Used to prevent other hats member from running command on CATS data. """
    if os.environ['LOGNAME'] != 'gdutton' and (site in cats_sites or site == 'all'):
        print("Sorry, you don't have permission to run this command on site: {}".format(site))
        quit()
            

def import2gcwerks(site, chromsdir):
    """ Importing routine.
    
        Import *.itx and *.gz (unless options.all is selected).  Once imported
        compress *.itx files and rename with .Z extension.
        
        Excludes cal3 injections.  Will need a different routine for c1_c, c2_c and c3_c 
   
        /hats/gc/itxbin/itx_import.py -g 2012nwr1672338.a1.itx.gz | /hats/gc/gcwerks-3/bin/chromatogram_import -gcdir /hats/gc/nwr
    """
    if site == 'bld1' or site == 'agc1':
        files = glob(chromsdir+'*.itx') + glob(chromsdir+'*.itx.gz')
        if options.all:
            files += glob(chromsdir+'*.itx.Z')
    elif site == 'std' or site == 'stdhp':
        ''' Standard GCs '''
        files = glob(chromsdir+'*.itx') + glob(chromsdir+'*.itx.gz')
        if options.all:
            files += glob(chromsdir+'*.itx.Z')
    else:
        ''' CATS sites '''
        files = glob(chromsdir+'*.??.itx') + glob(chromsdir+'*.??.itx.gz')
        if options.all:
            files += glob(chromsdir+'*.??.itx.Z')

    cmd = path + 'itxbin/itx_import.py '
    werks = '{}gcwerks-3/bin/chromatogram_import -gcdir {}{} > /dev/null 2>&1'.format(path, path, site)
            
    # apply spike filter before SG smoothing
    if options.s:
        cmd += '-s '
        if options.v:
            print('Applying single point spike filter')
        
    # apply wide spike filter?
    if options.ws_start > 0:
        cmd += '-W {} '.format(options.ws_start)
        if options.v:
            print('Applying wide spike filter starting at time={}'.format(options.ws_start))

    # apply Savitzky Golay smoothing    
    if options.g:
        cmd += '-g -gw {} -go {} '.format(options.SGwin, options.SGorder)
        if options.v:
            print('Applying Savitzky Golay smoothing: {}, {}'.format(options.SGwin, options.SGorder))

    if len(files) > 0:    
        print('Importing {} files from {}.'.format(len(files), site))
        for file in files:
            if options.v:
                print('   {}'.format(os.path.basename(file)))
            os.system(cmd + " " + file + " | " + werks)
        

def rename(chromsdir):
    """ Gzip *.itx files in the chroms_itx directory. 
    
        The files are also give the .Z suffix so that they will
        not be re-imported.  Also rename any .gz files to .Z suffix.
    """
    files = glob(chromsdir + '*.itx')
    if len(files) > 0:
        print('Compressing *.itx files.')
        for file in files:
            os.system('/bin/gzip -fS .Z ' + file)
            
    files = glob(chromsdir + '*.gz')
    if len(files) > 0:
        print('Renameing *.gz files')
        for file in files:
            os.rename(file, file[:-2]+'Z')
    
        
def main(site):
    yy = options.yyyy[2:] if len(options.yyyy) == 4 else options.yyyy

    chromsdir = path + site + '/' + yy + '/chroms_itx/'

    import2gcwerks(site, chromsdir)
    rename(chromsdir)
    
    # recreate index files for GCwerks -- not needed.  GCwerks adds to the index file.
    os.system('/hats/gc/itxbin/quickindex.py ' + site)
    #os.system('/hats/gc/gcwerks-3/bin/run-index -gcdir /hats/gc/' + site)

    # updates integration and mixing ratios
    os.system('/hats/gc/gcwerks-3/bin/gcupdate -gcdir /hats/gc/' + site)
    os.system('/hats/gc/gcwerks-3/bin/gccalc -gcdir /hats/gc/' + site + ' -1')


if __name__ == '__main__':

    today = date.today()
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
    
    permission(options.site)
    
    if options.site.lower() == 'all':
        for s in cats_sites:
            print('Working on {}'.format(s))
            main(s)
    else:
        main(options.site)
