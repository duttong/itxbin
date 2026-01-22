#! /bin/bash

# rebuild .run-index
rm /hats/gc/m4/.run-index
/hats/gc/gcwerks-3/bin/run-index -gcdir /hats/gc/m4

/hats/gc/itxbin/m4_samplogs.py --all -i
/hats/gc/itxbin/m4_gcwerks2db.py -x 2312
/hats/gc/itxbin/m4_batch.py -p all -i -s 2312
