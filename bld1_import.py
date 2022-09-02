#! /usr/bin/env python

""" 
    211119: Added methods to create a meta data file with run sequence information.
"""

import argparse
from datetime import date
from pathlib import Path
import json

from gcwerks_import import GCwerks_Import


class BLD1_import(GCwerks_Import):
    """ Class hardcoded for stratcore instrument (BLD1) """

    def __init__(self, args):
        self.site = 'bld1'
        incoming_dir = 'incoming'   # was chroms_itx now incoming on 211108
        super().__init__(self.site, args, incoming_dir)

    def create_metafiles(self):
        p = Path(self.incoming)
        for dir_ in p.glob('*-*'):
            self.sequence2metafile(dir_)
            
    def sequence2metafile(self, directory):
        metafile = f'{directory}/meta_{directory.name}.json'
        if Path(metafile).exists():
            return
        
        p = Path(directory)
        ssv = []
        for file in sorted(p.glob('*itx*')):
            ssv.append(self.itxport(file))
        s = ''.join(i for i in ssv)
        
        runtype = 'aircore'

        # HARDCODED. should be fixed in stratcore software!
        ports = [f'port{p}' for p in range(1, 13)]
        stands = {n+1:port for n, port in enumerate(ports)}

        with open(metafile, 'w') as f:
            print(f'creating {metafile}')
            obj = [s, self.runtype(s), stands]
            json.dump(obj, f, indent=2)
            
    @staticmethod
    def port_ascii_encode(v: int):
        ''' Returns 1-9 followed by a-z for integers greater than 9 '''
        c = str(v) if v < 10 else chr(87 + v)
        return c            

    def itxport(self, filename):
        n = filename.name
        try:
            base, port, _ = n.split('.')
        except ValueError:
            base, port, _, _ = n.split('.')
        return f'{self.port_ascii_encode(int(port))}'

    @staticmethod
    def runtype(sequence):
        
        if len(sequence) <= 10:
            return 'warmup'
        
        if len(set(sequence)) == 1:
            return 'warmup'
        
        if (len(set(sequence)) == 2 and ('1' in sequence) and ('b' in sequence)):
            return 'aircore'
        else:
            return 'cal'
            

if __name__ == '__main__':

    yyyy = date.today().year
    SGwin, SGorder = 37, 1      # Savitzky Golay default variables
    WSTART = -1                 # Wide spike filter start time
    smoothfile_default = Path('/hats/gc/itxbin/bld1_smoothing.txt')

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the BLD1 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default off)')
    parser.add_argument('-W', action="store", dest='ws_start', default=WSTART,
        help='Apply wide spike filter (default off)')
    parser.add_argument('-g', action='store_true', default=False,
        help='Apply Savitzky Golay smoothing (default off)')
    parser.add_argument('-gw', action='store', dest='SGwin', metavar='Win', default=SGwin, type=int,
        help=f'Sets Savitzky Golay smoothing window (default = {SGwin} points)')
    parser.add_argument('-go', action='store', dest='SGorder', metavar='Order', default=SGorder, type=int,
        help=f'Sets Savitzky Golay order of fit (default = {SGorder})')
    parser.add_argument('-file', action='store', dest='smoothfile', type=str, default=smoothfile_default,
        help=f'Use the smoothing parameters defined in this file. (default: {smoothfile_default})')
    parser.add_argument('-year', action='store', default=yyyy,
        help=f'Which year? (default is {yyyy})')
    parser.add_argument('-reimport', action='store_true', default=False,
        help='Reimport all itx files including .Z archived.')

    args = parser.parse_args()
    
    inst = BLD1_import(args)
    if args.reimport:
        types = ('*.itx', '*.itx.gz', '*.itx.Z')
        inst.main(import_method=inst.import_recursive_itx, types=types)
    else:
        inst.main(import_method=inst.import_recursive_itx)

    inst.create_metafiles()