#! /usr/bin/env python
''' The datetime sort used in the main loop is susceptible to problems if the 
    file names don't have hours, minutes that are in range.  Brad has a file with
    a '60' for minutes.  The sort fails since 0..59 are valid minutes.  GSD 150225
'''

import argparse
import glob
import os.path
from datetime import date, datetime

cats_sites = ('brw', 'sum', 'nwr', 'mlo', 'smo', 'spo')

def loadindexfile(site):
    try:
        with open('/hats/gc/' + site + '/.run-index', 'r') as f:
            lines = f.readlines()
            lines = map(lambda s: s.strip(), lines)
            return lines
    except IOError as e:
        print("Loading index file failed: {}".format(e.strerror))
        
def writeindexfile(site, data):
    try:
        with open('/hats/gc/' + site + '/.run-index', 'w') as f:
            for line in data:
                f.write(line+'\n')
    except IOError as e:
        print("Writing index file failed: {}".format(e.strerror))
        
def main(site, year):
    org = loadindexfile(options.site)

    yy = str(year)[2:]
    
    path = '/hats/gc/' + options.site + '/' + yy + '/chromatograms/channel0/'
    files = glob.glob(path + '*')
    new = [os.path.basename(file) for file in files]

    comb = list(set(org) | set(new))
    comb = sorted(comb, key=lambda comb: sortable(comb))
    
    writeindexfile(options.site, comb)
    
def sortable(f):
    yr = int(f[0:2])
    yr += 1900 if yr > 90 else 2000
    hour = int(f[7:9])
    hour = 23 if hour > 23 else hour
    minute = int(f[9:11])
    minute = 59 if minute > 59 else minute
    return datetime(yr, int(f[2:4]), int(f[4:6]), hour, minute)
    
if __name__=='__main__':

    today = date.today()

    opt = argparse.ArgumentParser(
        description = "Indexes the chromatogram files for GCwerks."
    )
    opt.add_argument("--year", default=today.year, 
            dest="yyyy", help="The year (default is current year: {})".format(today.year))
    opt.add_argument("site", help="A station or 'all' CATS sites.")
            
    options = opt.parse_args()
    
    if options.site == 'all':
        for s in cats_sites:
            main(s, options.yyyy)
    else:
        main(options.site, options.yyyy)

