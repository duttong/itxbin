#! /usr/bin/env python
''' The datetime sort used in the main loop is susceptible to problems if the 
    file names don't have hours, minutes that are in range.  Brad has a file with
    a '60' for minutes.  The sort fails since 0..59 are valid minutes.  GSD 150225
    
    Converted to a class.  170814
    Maybe should be combined into a GCwerks file with gcwerksimport.
'''

import argparse
from glob import glob
import os.path
from datetime import date, datetime

class GCwerks_runindex():

    def __init__(self, site):
        self.site = site
        self.org = self.loadindexfile()
        
    def loadindexfile(self):
        """ returns a map object in python3 instead of list """
        try:
            with open('/hats/gc/' + self.site + '/.run-index', 'r') as f:
                lines = f.readlines()
                lines = map(lambda s: s.strip(), lines)
                return lines
        except IOError as e:
            print("Loading index file failed: {}".format(e.strerror))
        
    def writeindexfile(self, data):
        try:
            with open('/hats/gc/' + self.site + '/.run-index', 'w') as f:
                for line in data:
                    f.write(line+'\n')
        except IOError as e:
            print("Writing index file failed: {}".format(e.strerror))
        
    @staticmethod
    def sortable(f):
        yr = int(f[0:2])
        yr += 1900 if yr > 90 else 2000
        hour = int(f[7:9])
        hour = 23 if hour > 23 else hour
        minute = int(f[9:11])
        minute = 59 if minute > 59 else minute
        return datetime(yr, int(f[2:4]), int(f[4:6]), hour, minute)
    
    def updateindex(self, year):
        """ Load original .run-index file add new files from /chromatograms/channel0
            directory to the run-index file.
        """
        yy = str(year)[2:]
        path = '/hats/gc/' + self.site + '/' + yy + '/chromatograms/channel0/'
        new = [os.path.basename(file) for file in glob(path + '*')]

        comb = list(set(self.org) | set(new))
        comb = sorted(comb, key=lambda comb: self.sortable(comb))
    
        self.writeindexfile(comb)
    
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
        from catsbase import CATS_Chromatograph
        cats = CATS_Chromatograph()
        for s in cats.sites:
            idx = GCwerks_runindex(s)
            idx.updateindex(options.yyyy)
    else:
        idx = GCwerks_runindex(options.site)
        idx.updateindex(options.yyyy)

