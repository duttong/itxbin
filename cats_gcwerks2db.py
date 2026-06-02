#!/usr/bin/env python3
"""Load CATS GCwerks export data into HATS ng_insitu tables."""

import time
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import DateOffset
import typer

app = typer.Typer(add_completion=False)


class CATS_GCwerks2DB:
    """Load per-molecule GCwerks CSV files for a CATS site into ng_insitu tables.

    CATS exports one CSV per analyte (e.g. brw_N2O.csv, brw_F12.csv) to
    /hats/gc/cats_results/.  This class merges them on (time, port) and
    upserts into hats.ng_insitu_analysis and hats.ng_insitu_mole_fractions.

    Usage:
        cats = CATS_GCwerks2DB("brw", flagged=True)
        cats.load()               # recent 2 months
        cats.load(year=2022)      # single year
        cats.load(duration_months=None)  # all data
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

    # CATS GCwerks column channels beyond the IE3 set {a, b, c}.
    # q  = N2O/SF6 (Shimadzu ECD, column Q)
    # f  = halogenated solvents (column F)
    # cc = HCFCs / CH3Br / CH3Cl (column CC)
    CATS_CHANNELS = {"a", "b", "c", "f", "q", "cc"}

    RESULTS_DIR = Path("/hats/gc/cats_results")

    # Gap (minutes) between consecutive injections that marks a new GC run.
    # CATS cycles through ports every ~30 min; 90 min separates distinct cycles.
    RUN_TIME_GAP_MIN = 90

    # GCwerks mol name → parameter_num.  Used as a fallback until
    # hats.analyte_list is populated for the CATS inst_nums.
    # Mol names come from the GCwerks column prefix after stripping the channel
    # suffix (e.g. "N2O_q_ht" → mol="N2O", channel="q").
    ANALYTE_MAP: dict[str, int] = {
        "N2O":      5,
        "SF6":      6,
        "CFC12":    22,
        "CFC11":    114,
        "CFC113":   32,
        "H1211":    26,
        "CHCl3":    34,
        "CH3CCl3":  131,   # MC; FE3 uses 131, Perseus uses 35 — confirm for CATS
        "CCl4":     37,
        "OCS":      42,    # stored as COS (param 42) in hats.parameters
        "HCFC22":   21,
        "CH3Cl":    23,
        "HCFC142b": 25,
        "CH3Br":    27,
    }

    def __init__(self, site: str, flagged: bool = False):
        site = site.lower()
        if site not in self.VALID_SITES:
            raise ValueError(f"Invalid site {site!r}. Valid sites: {sorted(self.VALID_SITES)}")
        self.site = site
        self.flagged = flagged

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
        self.analytes = self._query_analytes()

    def _site_num(self) -> int:
        sql = "SELECT num, code FROM gmd.site;"
        df = pd.DataFrame(self.db.doquery(sql))
        site_dict = dict(zip(df["code"].str.lower(), df["num"]))
        return int(site_dict[self.site])

    def _query_analytes(self) -> dict[str, int]:
        """Return mol_name → param_num from analyte_list, or fall back to ANALYTE_MAP."""
        sql = (
            "SELECT DISTINCT display_name, param_num "
            f"FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
        )
        rows = self.db.doquery(sql)
        if rows:
            return {r["display_name"]: int(r["param_num"]) for r in rows}
        print(
            f"No analyte_list entries for inst_num {self.inst_num} — "
            "using built-in ANALYTE_MAP."
        )
        return dict(self.ANALYTE_MAP)

    def _gcwerks_files(self) -> list[Path]:
        """All full-dataset (non-2yr) CSVs for this site."""
        return sorted(
            f for f in self.RESULTS_DIR.glob(f"{self.site}_*.csv")
            if "_2yr" not in f.name
        )

    def read_gcwerks(self, start_date=None) -> pd.DataFrame:
        """Merge all per-molecule CSVs into one wide DataFrame keyed on (time, port).

        start_date filters each file immediately after loading so the merge
        works on a small window rather than the full history.
        """
        files = self._gcwerks_files()
        if not files:
            raise FileNotFoundError(
                f"No CATS CSV files found for site {self.site!r} in {self.RESULTS_DIR}"
            )
        print(f"  Reading {len(files)} GCwerks CSV files...")

        frames = []
        for f in files:
            df = pd.read_csv(f, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            df["time"] = pd.to_datetime(df["time"])
            if start_date is not None:
                df = df.loc[df["time"] >= start_date]
            df["port"] = pd.to_numeric(df["port"], errors="coerce")
            for col in df.columns:
                if col not in ("time", "port"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").replace(-999, pd.NA)
            frames.append(df)

        print(f"  Merging...")
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on=["time", "port"], how="outer")

        return merged.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    def read_gcwerks_flagged(self, start_date=None) -> pd.DataFrame:
        """Merge per-molecule CSVs, preserving GCwerks flag characters (* / F).

        start_date filters each file immediately after loading.
        """
        files = self._gcwerks_files()
        if not files:
            raise FileNotFoundError(
                f"No CATS CSV files found for site {self.site!r} in {self.RESULTS_DIR}"
            )
        print(f"  Reading {len(files)} GCwerks CSV files (flagged)...")

        frames = []
        for f in files:
            df = pd.read_csv(f, skipinitialspace=True, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            df["time"] = pd.to_datetime(df["time"])
            if start_date is not None:
                df = df.loc[df["time"] >= start_date]
            df["port"] = pd.to_numeric(df["port"], errors="coerce")

            col_map = self._parse_measurement_columns(df.columns)
            for (mol, channel), cols in col_map.items():
                flag_col = self._flag_column_name(mol, channel)
                df[flag_col] = False
                for metric in ("ht", "area", "rt"):
                    col = cols.get(metric)
                    if not col:
                        continue
                    series = df[col].fillna("").astype(str).str.strip()
                    flagged = series.str.endswith(("F", "*", "B"))
                    cleaned = series.str.replace(r"[FB*]$", "", regex=True)
                    df[col] = pd.to_numeric(cleaned, errors="coerce").replace(-999, pd.NA)
                    df[flag_col] = df[flag_col] | flagged
            frames.append(df)

        print(f"  Merging...")
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on=["time", "port"], how="outer")

        merged = merged.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

        # After outer merge, boolean flag columns have NaN for rows where that
        # analyte file had no entry — treat those as unflagged.
        flag_cols = [c for c in merged.columns if c.endswith("_flag")]
        merged[flag_cols] = merged[flag_cols].fillna(False)
        return merged

    @staticmethod
    def _flag_column_name(mol: str, channel: str | None) -> str:
        return f"{mol}_{channel}_flag" if channel else f"{mol}_flag"

    def _assign_run_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("time").copy()
        gap = df["time"].diff() > pd.Timedelta(minutes=self.RUN_TIME_GAP_MIN)
        segment = gap.cumsum()
        df["run_time"] = df.groupby(segment)["time"].transform("first")
        return df

    @staticmethod
    def _parse_measurement_columns(columns) -> dict:
        """Return mapping (mol, chan) → {metric: column_name}.

        Handles CATS channel suffixes: a, b, c (IE3-style) plus q, f, cc.
        Examples:
          N2O_q_ht    → (mol='N2O',     chan='q')
          CFC12_f_ht  → (mol='CFC12',   chan='f')
          HCFC22_cc_ht → (mol='HCFC22', chan='cc')
          CFC12_a_ht  → (mol='CFC12',   chan='a')
        """
        metrics = {"ht", "area", "rt"}
        cats_channels = {"a", "b", "c", "f", "q", "cc"}
        mapping: dict = {}
        for col in columns:
            parts = col.split("_")
            if len(parts) < 2:
                continue
            metric = parts[-1]
            if metric not in metrics:
                continue
            channel = None
            if len(parts) >= 3 and parts[-2] in cats_channels:
                channel = parts[-2]
                mol = "_".join(parts[:-2])
            else:
                mol = "_".join(parts[:-1])
            mapping.setdefault((mol, channel), {})[metric] = col
        return mapping

    def upsert_analysis(self, df: pd.DataFrame, batch_size: int = 500) -> dict:
        df = df.copy()
        df["analysis_time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["run_time_str"] = df["run_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["port"] = pd.to_numeric(df["port"], errors="coerce").fillna(0).astype(int)

        analysis_sql = """
            INSERT INTO hats.ng_insitu_analysis (
                analysis_time, run_time, site_num, inst_num, port
            ) VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                run_time = IF(run_time = '1900-01-01 00:00:00', VALUES(run_time), run_time),
                port = VALUES(port)
        """
        params = [
            (r.analysis_time_str, r.run_time_str, self.site_num, self.inst_num, int(r.port))
            for r in df.itertuples(index=False)
        ]
        for i in range(0, len(params), batch_size):
            self.db.doMultiInsert(analysis_sql, params[i:i + batch_size], all=True)

        analysis_map = {}
        unique_times = df["analysis_time_str"].unique().tolist()
        for i in range(0, len(unique_times), batch_size):
            chunk = unique_times[i:i + batch_size]
            placeholders = ",".join(["%s"] * len(chunk))
            select_sql = (
                "SELECT analysis_time, num FROM hats.ng_insitu_analysis "
                "WHERE inst_num = %s AND site_num = %s "
                f"AND analysis_time IN ({placeholders})"
            )
            rows = self.db.doquery(select_sql, [self.inst_num, self.site_num] + chunk)
            analysis_map.update(
                {r["analysis_time"].strftime("%Y-%m-%d %H:%M:%S"): r["num"] for r in rows}
            )
        return analysis_map

    def upsert_mole_fractions(self, df: pd.DataFrame, analysis_map: dict, batch_size: int = 1000):
        mole_sql = """
            INSERT INTO hats.ng_insitu_mole_fractions (
                analysis_num, parameter_num, channel, height, area, retention_time
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                height         = VALUES(height),
                area           = VALUES(area),
                retention_time = VALUES(retention_time)
        """
        tag_sql = """
            INSERT IGNORE INTO hats.ng_insitu_mole_fraction_tags (
                ng_insitu_mole_fraction_num, tag_num
            ) VALUES (%s, %s)
        """

        col_map = self._parse_measurement_columns(df.columns)
        missing = sorted({mol for (mol, _ch) in col_map if mol not in self.analytes})
        if missing:
            print(f"Warning: no analyte mapping for GCwerks mols: {missing}")

        params = []
        all_keys: set = set()
        flagged_keys: set = set()

        for r in df.itertuples(index=False):
            analysis_num = analysis_map.get(r.analysis_time_str)
            if analysis_num is None:
                continue
            for (mol, ch), cols in col_map.items():
                param = self.analytes.get(mol)
                if param is None:
                    continue
                channel = ch or "a"
                flag_col = self._flag_column_name(mol, ch)
                ht_col = cols.get("ht")
                area_col = cols.get("area")
                rt_col = cols.get("rt")
                if not (ht_col and area_col and rt_col):
                    continue
                ht = getattr(r, ht_col, None)
                area = getattr(r, area_col, None)
                rt = getattr(r, rt_col, None)
                if pd.isna(ht): ht = None
                if pd.isna(area): area = None
                if pd.isna(rt): rt = None
                flag_val = getattr(r, flag_col, False)
                flagged = flag_val is True
                params.append((analysis_num, param, channel, ht, area, rt))
                all_keys.add((analysis_num, param, channel))
                if flagged:
                    flagged_keys.add((analysis_num, param, channel))

            if len(params) >= batch_size:
                self.db.doMultiInsert(mole_sql, params, all=True)
                params = []

        if params:
            self.db.doMultiInsert(mole_sql, params, all=True)

        # Sync GCwerks flag tags (tag_num=324): delete stale then reinsert current.
        if self.flagged and all_keys:
            keys = sorted(all_keys)
            for i in range(0, len(keys), batch_size):
                chunk = keys[i:i + batch_size]
                where_terms = " OR ".join(
                    ["(m.analysis_num = %s AND m.parameter_num = %s AND m.channel = %s)"] * len(chunk)
                )
                query_params = [v for triple in chunk for v in triple]
                delete_sql = f"""
                    DELETE t FROM hats.ng_insitu_mole_fraction_tags t
                    JOIN hats.ng_insitu_mole_fractions m
                        ON t.ng_insitu_mole_fraction_num = m.num
                    WHERE t.tag_num = 324 AND ({where_terms})
                """
                self.db.doquery(delete_sql, query_params)

        if flagged_keys:
            tag_params = []
            keys = sorted(flagged_keys)
            for i in range(0, len(keys), batch_size):
                chunk = keys[i:i + batch_size]
                where_terms = " OR ".join(
                    ["(analysis_num = %s AND parameter_num = %s AND channel = %s)"] * len(chunk)
                )
                query_params = [v for triple in chunk for v in triple]
                select_sql = (
                    "SELECT num FROM hats.ng_insitu_mole_fractions "
                    f"WHERE {where_terms}"
                )
                rows = self.db.doquery(select_sql, query_params)
                tag_params.extend((r["num"], 324) for r in rows)
                if len(tag_params) >= batch_size:
                    self.db.doMultiInsert(tag_sql, tag_params, all=True)
                    tag_params = []
            if tag_params:
                self.db.doMultiInsert(tag_sql, tag_params, all=True)

    def load(self, duration_months: int | None = 2, year: int | None = None):
        """Read GCwerks CSVs and upsert into ng_insitu tables.

        Run cats_export.py first to refresh the CSV files from GCwerks.
        """
        t0 = time.time()

        # Determine start_date early so read methods can filter each file
        # before merging, avoiding full-history merges for short windows.
        if year is not None:
            start_date = pd.Timestamp(f"{year}-01-01")
        elif duration_months is not None:
            start_date = pd.Timestamp.now() - DateOffset(months=duration_months)
        else:
            start_date = None

        label = f"{year}" if year else (f"last {duration_months}mo" if duration_months else "all data")
        print(f"CATS {self.site.upper()} — {label}")

        if self.flagged:
            df = self.read_gcwerks_flagged(start_date=start_date)
        else:
            df = self.read_gcwerks(start_date=start_date)

        # For year mode trim the upper bound; duration_months used today as anchor
        # so no upper trim needed.
        if year is not None:
            df = df.loc[df["time"].dt.year == int(year)]

        df = self._assign_run_time(df)
        df["analysis_time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        print(f"  {len(df)} rows  ({time.time()-t0:.1f}s)")

        print(f"  Upserting analysis rows...")
        analysis_map = self.upsert_analysis(df)
        print(f"  Upserting mole fractions...")
        self.upsert_mole_fractions(df, analysis_map)
        print(f"Done. ({time.time()-t0:.1f}s total)")


@app.command()
def load(
    site: str = typer.Argument(..., help="Site code: brw, spo"),
    all_data: bool = typer.Option(False, "--all", help="Process all data"),
    year: int | None = typer.Option(None, "--year", help="Process a single year (YYYY)"),
    flagged: bool = typer.Option(False, "--flagged", help="Parse and sync GCwerks flag characters"),
):
    """Load CATS GCwerks export data into HATS ng_insitu tables.

    Run cats_export.py first to refresh the per-molecule CSV files from GCwerks.
    """
    cats = CATS_GCwerks2DB(site, flagged=flagged)
    if all_data:
        cats.load(duration_months=None)
    else:
        cats.load(year=year)


if __name__ == "__main__":
    app()
