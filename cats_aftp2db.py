#!/usr/bin/env python3
"""Import CATS published mole fractions from /aftp/hats into ng_insitu_mole_fractions.

The /aftp/hats files contain the QC'd, published hourly mole fractions for
each site and compound.  Each timestamp maps 1-to-1 with an injection row in
ng_insitu_analysis (loaded by cats_gcwerks2db.py), so we can update
mole_fraction and unc on the matching ng_insitu_mole_fractions row.

Run cats_gcwerks2db.py first to populate ng_insitu_analysis and the
height/area/rt columns of ng_insitu_mole_fractions.
"""

from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import DateOffset
import typer

app = typer.Typer(add_completion=False)

AFTP_ROOT = Path("/aftp/hats")

# Each entry: aftp_stem → (relative path template, param_num, channel)
# Path template uses {site} (lowercase, e.g. "brw").
# Channel matches the GCwerks column channel for that compound's primary measurement.
AFTP_MANIFEST: dict[str, tuple[str, int, str]] = {
    "N2O":      ("n2o/insituGCs/CATS/hourly/{site}_N2O_All.dat",          5,   "q"),
    "SF6":      ("sf6/insituGCs/CATS/hourly/{site}_SF6_All.dat",           6,   "q"),
    "F12":      ("cfcs/cfc12/insituGCs/CATS/hourly/{site}_F12_All.dat",    22,  "a"),
    "F11":      ("cfcs/cfc11/insituGCs/CATS/hourly/{site}_F11_All.dat",    114, "f"),
    "F113":     ("cfcs/cfc113/insituGCs/CATS/hourly/{site}_F113_All.dat",  32,  "f"),
    "H1211":    ("halons/insituGCs/CATS/hourly/{site}_H1211_All.dat",      26,  "f"),
    "HCFC22":   ("hcfcs/hcfc22/insituGCs/CATS/hourly/{site}_HCFC22_All.dat",     21,  "cc"),
    "HCFC142b": ("hcfcs/hcfc142b/insituGCs/CATS/hourly/{site}_HCFC142b_All.dat", 25,  "cc"),
    "CH3Cl":    ("methylhalides/ch3cl/insituGCs/CATS/hourly/{site}_CH3Cl_All.dat", 23, "cc"),
    "CCl4":     ("solvents/CCl4/insituGCs/CATS/hourly/{site}_CCl4_All.dat", 37,  "f"),
    "MC":       ("solvents/CH3CCl3/insituGCs/CATS/hourly/{site}_MC_All.dat", 131, "f"),
}


def read_aftp_file(path: Path) -> pd.DataFrame:
    """Parse a CATS /aftp All.dat file into a DataFrame with columns:
    time (datetime), mole_fraction (float), unc (float).

    File format (whitespace-delimited, # comment header):
      yr mon day hour min  mf  sd
    Missing values are the string 'Nan'.
    """
    # Find the first non-comment, non-empty line — that is the column header.
    with open(path) as fh:
        header_line = None
        data_start = 0
        for i, line in enumerate(fh):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                header_line = stripped
                data_start = i + 1
                break
        if header_line is None:
            raise ValueError(f"No data header found in {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=data_start,
        names=["yr", "mon", "day", "hour", "min", "mole_fraction", "unc"],
        na_values=["Nan", "NaN", "nan", "-999", "-999.0"],
        comment="#",
    )

    df["time"] = pd.to_datetime(
        df[["yr", "mon", "day", "hour", "min"]].rename(
            columns={"yr": "year", "mon": "month", "min": "minute"}
        )
    )
    return df[["time", "mole_fraction", "unc"]].dropna(subset=["mole_fraction"])


class CATS_AFTP2DB:
    """Import CATS published mole fractions from /aftp/hats into the DB.

    Requires that ng_insitu_analysis rows already exist for the target site
    (i.e. cats_gcwerks2db.py has been run).

    Usage:
        loader = CATS_AFTP2DB("brw")
        loader.load()                      # recent 2 months
        loader.load(year=2010)             # single year
        loader.load(duration_months=None)  # all data
    """

    VALID_SITES = {"brw", "spo", "nwr", "mlo", "smo", "sum"}
    INST_NUM_BY_SITE: dict[str, int] = {
        "brw": 239,
        "sum": 240,
        "nwr": 241,
        "mlo": 242,
        "smo": 243,
        "spo": 244,
    }

    def __init__(self, site: str):
        site = site.lower()
        if site not in self.VALID_SITES:
            raise ValueError(f"Invalid site {site!r}. Valid sites: {sorted(self.VALID_SITES)}")
        self.site = site

        inst_num = self.INST_NUM_BY_SITE.get(site)
        if inst_num is None:
            raise ValueError(
                f"No inst_num for CATS site {site!r}. Add it to INST_NUM_BY_SITE."
            )
        self.inst_num = inst_num

        import sys
        sys.path.append("/ccg/src/db/")
        import db_utils.db_conn as db_conn  # type: ignore
        self.db = db_conn.HATS_ng()
        self.site_num = self._site_num()

    def _site_num(self) -> int:
        sql = "SELECT num, code FROM gmd.site;"
        df = pd.DataFrame(self.db.doquery(sql))
        site_dict = dict(zip(df["code"].str.lower(), df["num"]))
        return int(site_dict[self.site])

    def _build_analysis_map(self, times: list[str], batch_size: int = 500) -> dict[str, int]:
        """Return analysis_time_str → analysis_num for the given time strings."""
        analysis_map: dict[str, int] = {}
        for i in range(0, len(times), batch_size):
            chunk = times[i:i + batch_size]
            placeholders = ",".join(["%s"] * len(chunk))
            sql = (
                "SELECT analysis_time, num FROM hats.ng_insitu_analysis "
                "WHERE inst_num = %s AND site_num = %s "
                f"AND analysis_time IN ({placeholders})"
            )
            rows = self.db.doquery(sql, [self.inst_num, self.site_num] + chunk) or []
            analysis_map.update(
                {r["analysis_time"].strftime("%Y-%m-%d %H:%M:%S"): r["num"] for r in rows}
            )
        return analysis_map

    def load_compound(
        self,
        compound: str,
        duration_months: int | None = 2,
        year: int | None = None,
        batch_size: int = 1000,
    ):
        """Load one compound from its /aftp All.dat file into ng_insitu_mole_fractions."""
        if compound not in AFTP_MANIFEST:
            raise ValueError(f"Unknown compound {compound!r}. Valid: {sorted(AFTP_MANIFEST)}")

        path_template, param_num, channel = AFTP_MANIFEST[compound]
        path = AFTP_ROOT / path_template.format(site=self.site)
        if not path.exists():
            print(f"  Skipping {compound}: {path} not found.")
            return 0

        df = read_aftp_file(path)
        if df.empty:
            print(f"  Skipping {compound}: no data in {path.name}")
            return 0

        if year is not None:
            df = df.loc[df["time"].dt.year == int(year)]
        elif duration_months is not None:
            last_date = df["time"].max()
            start_date = last_date - DateOffset(months=duration_months)
            df = df.loc[df["time"] >= start_date]

        if df.empty:
            return 0

        df["time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        analysis_map = self._build_analysis_map(df["time_str"].tolist())

        unmatched = df.loc[~df["time_str"].isin(analysis_map)].shape[0]
        if unmatched:
            print(f"  {compound}: {unmatched} timestamps not in ng_insitu_analysis (run cats_gcwerks2db.py first)")

        mole_sql = """
            INSERT INTO hats.ng_insitu_mole_fractions (
                analysis_num, parameter_num, channel, mole_fraction, unc
            ) VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                mole_fraction = VALUES(mole_fraction),
                unc           = VALUES(unc)
        """

        params = []
        loaded = 0
        for r in df.itertuples(index=False):
            analysis_num = analysis_map.get(r.time_str)
            if analysis_num is None:
                continue
            mf = None if pd.isna(r.mole_fraction) else float(r.mole_fraction)
            unc = None if pd.isna(r.unc) else float(r.unc)
            params.append((analysis_num, param_num, channel, mf, unc))
            loaded += 1

            if len(params) >= batch_size:
                self.db.doMultiInsert(mole_sql, params, all=True)
                params = []

        if params:
            self.db.doMultiInsert(mole_sql, params, all=True)

        print(f"  {compound}: upserted {loaded} mole fraction rows.")
        return loaded

    def load(
        self,
        duration_months: int | None = 2,
        year: int | None = None,
        compounds: list[str] | None = None,
    ):
        """Load all (or selected) compounds for this site."""
        targets = compounds if compounds is not None else list(AFTP_MANIFEST)
        print(f"Loading CATS /aftp mole fractions for {self.site.upper()}...")
        total = 0
        for compound in targets:
            total += self.load_compound(compound, duration_months=duration_months, year=year)
        print(f"Total rows upserted: {total}")


@app.command()
def load(
    site: str = typer.Argument(..., help="Site code: brw, spo, nwr, mlo, smo, sum"),
    all_data: bool = typer.Option(False, "--all", help="Process all data"),
    year: int | None = typer.Option(None, "--year", help="Process a single year (YYYY)"),
    compound: list[str] | None = typer.Option(None, "--compound", "-c", help="Specific compound(s)"),
):
    """Import CATS published mole fractions from /aftp/hats into the DB.

    Requires ng_insitu_analysis rows to already exist (run cats_gcwerks2db.py first).
    Valid compounds: N2O, SF6, F12, F11, F113, H1211, HCFC22, HCFC142b,
                     CH3Cl, CCl4, MC
    """
    loader = CATS_AFTP2DB(site)
    if all_data:
        loader.load(duration_months=None, compounds=list(compound) if compound else None)
    else:
        loader.load(year=year, compounds=list(compound) if compound else None)


if __name__ == "__main__":
    app()
