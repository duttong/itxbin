"""Which integration feeds an instrument's hats in situ tables: hats.ng_integ_source.

One current row per instrument (DDL: sql/ng_integ_source.sql); no row, or no table, means GCwerks.
Used by cats_ingest.py (which export and loader to run), cats_gcwerks2db.py (never loads logosint
values before logosint_from) and logosint_jobs.py (job ranges and switches).
"""

import sys
from dataclasses import dataclass

import pandas as pd

sys.path[:0] = ["/ccg/src/db", "/ccg/src/db/db_utils"]

# logosint channel number -> CATS channel letter (the export's column suffix), per CATS instrument.
# Channel 3 (cc) is not integrated by logosint: in logosint mode its rows still come from GCwerks.
CATS_CHANNELS = {0: "q", 1: "a", 2: "f"}
GCWERKS_ONLY_CHANNELS = ("cc",)


@dataclass
class Source:
    source: str = "gcwerks"
    logosint_from: pd.Timestamp | None = None
    flags_frozen: pd.Timestamp | None = None

    @property
    def logosint(self) -> bool:
        return self.source == "logosint"


def current(inst_num: int, db=None, strict: bool = False) -> Source:
    """The instrument's current source: GCwerks when it has no row. Without the table, GCwerks too,
    unless strict (cats_ingest.py), which raises rather than guess."""
    if db is None:
        import db_conn
        db = db_conn.HATS_ng()
    have = db.doquery("SELECT COUNT(*) AS n FROM information_schema.tables "
                      "WHERE table_schema = 'hats' AND table_name = 'ng_integ_source'")
    if not have or not int(have[0]["n"]):
        if strict:
            raise RuntimeError("hats.ng_integ_source does not exist (run sql/ng_integ_source.sql)")
        return Source()
    rows = db.doquery(
        "SELECT source, logosint_from, flags_frozen FROM hats.ng_integ_source "
        "WHERE inst_num = %s AND retired_at IS NULL", [int(inst_num)]) or []
    if not rows:
        return Source()
    r = rows[0]
    ts = lambda v: pd.Timestamp(v) if v is not None else None
    return Source(r["source"], ts(r["logosint_from"]), ts(r["flags_frozen"]))
