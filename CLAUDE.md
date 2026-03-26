# CLAUDE.md — itxbin project context

## Environment

- Production conda env: `prod6`
- Run scripts as: `python3 <script>.py` or via the installed entry points
- Database: HATS next-generation MySQL database, accessed via
  `/ccg/src/db/db_utils/db_conn.py` → `db_conn.HATS_ng()`
- The helper wrapper `hats_db.py` provides `HATSdb` for quick one-off queries
- **Do not use `NUMERIC` for float columns** — the DB driver maps it to
  `decimal(10,0)` (scale=0), silently truncating decimal values. Use
  `DOUBLE PRECISION` or `FLOAT8` instead.

## Key instrument/parameter numbers

| Instrument | inst_num | inst_id |
|---|---|---|
| M4 (mass spec) | 192 | `m4` |
| FE3 (flask ECD) | 193 | `fe3` |
| BLD1 | 220 | `bld1` |
| IE3 | 236 | `ie3` |

| Compound | parameter_num |
|---|---|
| CFC-113 | 32 |
| CFC-113a | 178 |

## Key database tables

- `hats.ng_data_processing_view` — main analysis view (areas, normalized
  responses, mole fractions, run metadata)
- `hats.ng_mole_fractions` — computed mole fraction output table (upserted by
  batch scripts and logos_data)
- `hats.ng_cfc113a` — M4 molar response factors (R1–R4) for CFC-113/113a
  deconvolution, windowed by reference tank and date
- `hats.scale_assignments` — calibration scale coefficients (coef0, coef1)
  used by `calc_mole_fraction_scalevalues` for most M4 analytes

## CFC-113 / CFC-113a coupling (M4)

These two compounds co-elute on M4 and must always be processed as a pair
using the Montzka two-equation deconvolution. Key points:

- `M4_Instrument.calc_mole_fraction()` automatically routes pnum 32 and 178
  through the deconvolution — all logos_data call sites are covered
- `m4_batch.py -p 32` and `-p 178` are equivalent: both recalculate and
  upsert pnum=32 and pnum=178 together
- After any logos_data edits to CFC-113 or CFC-113a, run
  `m4_batch.py -p 32 -i` to write correct deconvolved values back to DB
- Reference data and the Montzka email are in `cfc113a/`

## Class hierarchy

```
LOGOS_Instruments
  └── HATS_DB_Functions        # DB connection, shared methods
        ├── M4_Instrument      # inst_num=192, scale-value mole fractions
        ├── FE3_Instrument     # inst_num=193, polynomial response inversion
        ├── IE3_Instrument     # inst_num=236
        └── BLD1_Instrument    # inst_num=220
```

## logos_data.py (GUI)

- Requires a display (`$DISPLAY` must be set); connect with `ssh -Y`
- Single-parameter context: `self.run` holds data for one pnum at a time
- Save action calls `upsert_mole_fractions(self.run)` — mole fractions must
  be correctly computed before saving
- For M4 CFC-113/113a, the GUI recalc paths load the partner pnum from DB
  internally; authoritative recalc should be done with `m4_batch.py -p 32 -i`
