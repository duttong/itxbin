#!/usr/bin/bash

/hats/gc/itxbin/fe3_import.py > /dev/null 2>&1 
/hats/gc/itxbin/fe3_export.py > /dev/null 2>&1
/hats/gc/itxbin/fe3_merge.py #> /dev/null 2>&1
/hats/gc/itxbin/fe3_gcwerks2db.py
/hats/gc/itxbin/fe3_data2db.py
