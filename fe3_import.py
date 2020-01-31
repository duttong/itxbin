#! /home/hats/gdutton/anaconda3/bin/python

import argparse
from datetime import date

from itx_import import ITX
from gcwerksimport import GCwerks_Import


class FE3_import(GCwerks_Import):

    def __init__(self, site, args, incoming_dir='incoming'):
        super().__init__(site, args, incoming_dir)

    def mark_first_itx_bad(self):
        for dir in self.incoming.glob('*-*'):
            itxs = [itx for itx in sorted(list(dir.glob('*itx*')))]
            first_itx = itxs[0].name
            extension = first_itx[21:]
            if extension == '.Z' or extension == '.gz':
                itxs[0].rename(first_itx[:-(len(extension)-1)]+'B')
            elif extension == '':
                itx = ITX(itxs[0])
                itx.compress_to_Z(extension='B')


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

    fe3 = FE3_import(args.site, args)
    fe3.mark_first_itx_bad()
    quit()
    fe3.main()
