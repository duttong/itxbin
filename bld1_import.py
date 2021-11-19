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

    def itxport(self, filename):
        n = filename.name
        try:
            base, port, _ = n.split('.')
        except ValueError:
            base, port, _, _ = n.split('.')
        return f'{int(port):x}'    # returns a hex number

    def sequence2metafile(self, directory):
        metafile = f'{directory}/meta_{directory.name}.json'
        if Path(metafile).exists():
            return
        
        p = Path(directory)
        ssv = []
        for file in sorted(p.glob('*itx*')):
            ssv.append(self.itxport(file))

        s = ''.join(i for i in ssv)
        # HARDCODED. should be fixed in stratcore software!
        stands = {1: 'port1', 2: 'port2', 3: 'port3', 4: 'port4', 5: 'port5', 6: 'port6', 7: 'port7', 8: 'port8'}

        with open(metafile, 'w') as f:
            print(f'creating {metafile}')
            obj = [s, stands]
            json.dump(obj, f, indent=2)

if __name__ == '__main__':

    yyyy = date.today().year
    SGwin, SGorder = 37, 1      # Savitzky Golay default variables
    WSTART = 25                 # Wide spike filter start time

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the BLD1 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default is False)')
    parser.add_argument('-W', action="store", dest='ws_start', default=WSTART,
        help='Apply wide spike filter (default off)')
    parser.add_argument('-g', action='store_true', default=True,
        help='Apply Savitzky Golay smoothing (default is True)')
    parser.add_argument('-gw', action='store', dest='SGwin', metavar='Win',
        default=SGwin, type=int,
        help=f'Sets Savitzky Golay smoothing window (default = {SGwin} points)')
    parser.add_argument('-go', action='store', dest='SGorder', metavar='Order',
        default=SGorder, type=int,
        help=f'Sets Savitzky Golay order of fit (default = {SGorder})')
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