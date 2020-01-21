#! /home/hats/gdutton/anaconda3/bin/python

import argparse
from datetime import date

from itx_import import ITX
from gcwerksimport import GCWerks_Import

if __name__ == '__main__':

    yyyy = date.today().year
    SGwin, SGorder = 21, 4      # Savitzky Golay default variables
    site = 'agc1'

    parser = argparse.ArgumentParser(
        description='Import chromatograms in the Igor Text File (.itx) format for the FE3 instrument.')
    parser.add_argument('-s', action='store_true', default=False,
        help='Apply 1-point spike filter (default is False)')
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
    parser.add_argument('site', nargs='?', default=site,
        help=f'Valid station code (default is {site})')

    args = parser.parse_args()

    fe3 = GCWerks_Import(args.site, args, 'incoming')
    fe3.main()
