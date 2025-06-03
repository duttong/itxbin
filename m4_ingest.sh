#!/usr/bin/bash

DAYS=14

INCOMING="/hats/gc/m4/chemstation"
RAW="/hats/gc/m4/MassHunter/GCMS/1/data"
GSPC="/hats/gc/m4/MassHunter/GCMS/M4 GSPC Files"

# 1) Delete anything older than $DAYS days, in the background
(
  find "$INCOMING" -maxdepth 1 -type d -mtime +"$DAYS" -exec rm -r {} \;
  find "$INCOMING" -maxdepth 1 -type f -mtime +"$DAYS" -exec rm {}   \;
) &

# 2) Wait for the deletions to finish before moving on
wait

# 3) Rsync “recent” data from RAW → INCOMING
(
  cd "$RAW" || exit 1
  find . -mindepth 1 -maxdepth 1 -mtime -"$DAYS" -print0 \
    | rsync -arv --progress --files-from=- --from0 ./ "$INCOMING/"
)

# 4) Rsync “recent” data from GSPC → INCOMING
(
  cd "$GSPC" || exit 1
  find . -maxdepth 1 -mtime -"$DAYS" -print0 \
    | rsync -arv --progress --files-from=- --from0 ./ "$INCOMING/"
)

# 5) Import into GCwerks, update logs, etc.
GCDIR=/hats/gc/m4
/hats/gc/gcwerks-3/bin/gcimport   -gcdir "$GCDIR"
/hats/gc/gcwerks-3/bin/run-index  -gcdir "$GCDIR"

/hats/gc/itxbin/m4_samplogs.py -i

/hats/gc/gcwerks-3/bin/gcupdate    -gcdir "$GCDIR"
/hats/gc/gcwerks-3/bin/gccalc      -gcdir "$GCDIR"

/hats/gc/itxbin/m4_gcwerks2db.py -x
/hats/gc/itxbin/m4_batch.py      -p all -i
