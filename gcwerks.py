#! /usr/bin/env python

import argparse
import os

gcwerks = '/hats/gc/gcwerks-3/bin/gcwerks'
cats_sites = ('brw','sum','nwr','mlo','smo','spo')

if __name__ == '__main__':
    goodsites = cats_sites + ('std','stdhp','bld1','fe3')

    opt = argparse.ArgumentParser(
        description="Used for launching GCwerks integration software."
    )
    opt.add_argument("site", default='all', help=f"Open GCwerks for {goodsites}")
    options = opt.parse_args()

    if options.site in goodsites:
        os.system(f'{gcwerks} -gcdir /hats/gc/{options.site}')
    else:
        os.system(f'{gcwerks} -datadir /hats/gc')
