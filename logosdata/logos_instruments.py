"""Facade — implementation split across logos_instruments_core.py,
logos_instruments_flask.py, and logos_instruments_insitu.py."""
import sys
from pathlib import Path

# The root shim executes this file as a top-level module via
# spec_from_file_location, so the logosdata dir may not be on sys.path;
# the sibling imports below need it.
_PKG_DIR = str(Path(__file__).resolve().parent)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from logos_instruments_core import LOGOS_Instruments, HATS_DB_Functions, Normalizing
from logos_instruments_flask import (
    M4_Instrument,
    FE3_Instrument,
    Perseus_Instrument,
    PRS_Instrument,
)
from logos_instruments_insitu import IE3_Instrument, CATS_Instrument, BLD1_Instrument

__all__ = [
    "LOGOS_Instruments",
    "HATS_DB_Functions",
    "Normalizing",
    "M4_Instrument",
    "FE3_Instrument",
    "Perseus_Instrument",
    "PRS_Instrument",
    "IE3_Instrument",
    "CATS_Instrument",
    "BLD1_Instrument",
]
