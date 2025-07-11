#!/usr/bin/bash

DAYS=14
INCOMING=/hats/gc/m4/chemstation
RAW=/hats/gc/m4/MassHunter/GCMS/1/data
GSPC=/hats/gc/m4/MassHunter/GCMS/M4\ GSPC\ Files

# ——————————————————————————————————————————————————————————————
# CLEANUP: delete old data
# ——————————————————————————————————————————————————————————————
echo "Cleaning up files older than ${DAYS} days in $INCOMING"
find "$INCOMING" -mindepth 1 -mtime +$DAYS -exec rm -rf {} +

# ——————————————————————————————————————————————————————————————
# COPY: recent RAW and GSPC in parallel
# ——————————————————————————————————————————————————————————————
for SRC in "$RAW" "$GSPC"; do
  {
    echo "Syncing recent files from $SRC → $INCOMING"
    cd "$SRC"
    find . -mindepth 1 -maxdepth 1 -mtime -"$DAYS" -print0 \
      | rsync -rlt --whole-file --quite --from0 --files-from=- ./ "$INCOMING/"
    echo "Done syncing $SRC"
  } &
done

wait

# 3) Import into GCwerks, update logs, etc.
GCDIR=/hats/gc/m4
/hats/gc/gcwerks-3/bin/gcimport   -gcdir "$GCDIR"
/hats/gc/gcwerks-3/bin/run-index  -gcdir "$GCDIR"

/hats/gc/itxbin/m4_samplogs.py -i

/hats/gc/gcwerks-3/bin/gcupdate    -gcdir "$GCDIR"
/hats/gc/gcwerks-3/bin/gccalc      -gcdir "$GCDIR"

/hats/gc/itxbin/m4_gcwerks2db.py -x
/hats/gc/itxbin/m4_batch.py      -p all -i
