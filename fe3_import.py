#! /usr/bin/env python

import argparse
from datetime import date
import multiprocessing as mp

import itx_import
from gcwerks_import import GCwerks_Import


class FE3_import(GCwerks_Import):

    def __init__(self, site, args, incoming_dir='incoming'):
        super().__init__(site, args, incoming_dir)
        self.exclude_port = '10'     # push port

    def import_exclude(self, exclude=1, types=['*.itx', '*.itx.gz']):
        """ Imports from a directory and excludes the first exclude=N itx files.
        """
        loaded = False
        num_workers = 10    # faster
        pool = mp.Pool(num_workers)
        for dir in self.incoming.glob('*-*'):
            # create a sorted list of files of types in dir
            files = []
            for type in types:
                files.extend(dir.glob(type))
            files.sort()

            if len(files) <= exclude:
                continue

            # if all the ports are self.exclude_port then don't
            # load into GCwerks or the FE3 db.
            portsused = self.itx_portnumbers(files)
            if (len(portsused) == 1) & (portsused[0] == self.exclude_port):
                print(f'Only push port runs. Excluding: {dir}')
                for file in files:
                    itx = itx_import.ITX(file)
                    itx.compress_to_Z(file)

            # import files except the first exclude=N files.
            else:
                loaded = True
                for file in files[0:exclude]:
                    itx = itx_import.ITX(file) 
                    itx.compress_to_Z(file)
                for file in files[exclude:]:
                    pool.apply_async(self.import_itx, args=(file,))
        pool.close()
        pool.join()

        return loaded

    @staticmethod
    def itx_portnumbers(files):
        ports = [file.name.split('.')[1] for file in files]
        return list(set(ports))


if __name__ == '__main__':

    yyyy = date.today().year
    box_win = 25
    SGwin, SGorder = 21, 4      # Savitzky Golay default variables
    WSTART = -1
    site = 'fe3'

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the FE3 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default is False)')
    parser.add_argument('-W', action="store", dest='ws_start', default=WSTART,
        help='Apply wide spike filter (default off)')
    parser.add_argument('-b', action='store', dest='boxwidth', metavar='Win', default=box_win, type=int,
        help=f'Apply a Box smooth with window width in points')
    parser.add_argument('-g', action='store_true', default=False,
        help='Apply Savitzky Golay smoothing (default is False)')
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
    parser.add_argument('site', nargs='?', default=site,
        help=f'Valid station code (default is {site})')

    args = parser.parse_args()
    
    fe3 = FE3_import(args.site, args)
    if args.reimport:
        types = ('*.itx', '*.itx.gz', '*.itx.Z')
        fe3.main(import_method=fe3.import_exclude, types=types)
    else:
        fe3.main(import_method=fe3.import_exclude)
