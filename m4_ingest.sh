#! /usr/bin/bash

# directory GCwerks looks at for new chromatograms
INCOMING="/hats/gc/m4/chemstation"
# directory synced from M4 computer with Masshunter running
RAW="/hats/gc/m4/MassHunter/GCMS/1/data"
# GSPC output files with sample pressure, etc.
GSPC="/hats/gc/m4/MassHunter/GCMS/M4 GSPC Files"

# delete data older than two weeks from INCOMING
find "$INCOMING" -maxdepth 1 -type d -mtime +14 -exec rm -r {} \;
find "$INCOMING" -maxdepth 1 -type f -mtime +14 -exec rm {} \;

# copy recent data to INCOMING
find "$RAW" -mindepth 1 -maxdepth 1 -type d -mtime -14 -print -exec cp -r {} "$INCOMING" \;
find "$RAW" -maxdepth 1 -type f -mtime -14 -print -exec cp  {} "$INCOMING" \;
find "$GSPC" -maxdepth 1 -type f -mtime -14 -print -exec cp {} "$INCOMING" \;

# import and integrate with GCwerks
GCDIR=/hats/gc/m4
/hats/gc/gcwerks-3/bin/gcimport -gcdir $GCDIR
/hats/gc/gcwerks-3/bin/run-index -gcdir $GCDIR

# update sample logs
/hats/gc/itxbin/m4_samplogs.py -i

# use sample logs in GCwerks
/hats/gc/gcwerks-3/bin/gcupdate -gcdir $GCDIR
/hats/gc/gcwerks-3/bin/gccalc -gcdir $GCDIR

# export the last two months from GCwerks and insert to db
/hats/gc/itxbin/m4_gcwerks2db.py -x

# calculate mole fractions
/hats/gc/itxbin/m4_batch.py -p all -i
