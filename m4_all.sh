#! /bin/bash

/hats/gc/itxbin/m4_samplogs.py --all
/hats/gc/itxbin/m4_gcwerks2db.py -x 2312
/hats/gc/itxbin/m4_batch.py -p all -i -s 2312
