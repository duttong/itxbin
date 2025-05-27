#! /usr/bin/env python

import argparse
from datetime import date

from gcwerks_import import GCwerks_Import


class IE3_import(GCwerks_Import):

    def __init__(self, site, args, incoming_dir='incoming'):
        super().__init__(site, args, incoming_dir)


if __name__ == '__main__':

    yyyy = date.today().year
    SGwin, SGorder = 81, 4      # Savitzky Golay default variables
    WSTART = -1
    BOXWIDTH = 11  # Default box width for smoothing
    site = 'smo'

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the FE3 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default is False)')
    parser.add_argument('-W', action="store", dest='ws_start', default=WSTART,
        help='Apply wide spike filter (default off)')
    parser.add_argument('-b', action='store', dest='boxwidth', metavar='Win', type=int, default=BOXWIDTH,
        help=f'Apply a Box smooth with window width (default = {BOXWIDTH})')
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

    ie3 = IE3_import(args.site, args)
    if args.reimport:
        types = ('*.itx', '*.itx.gz', '*.itx.Z')
        ie3.main(import_method=ie3.import_recursive_itx, types=types)
    else:
        ie3.main(import_method=ie3.import_recursive_itx)
