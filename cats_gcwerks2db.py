#!/usr/bin/env python3
"""Load CATS GCwerks export data into HATS ng_insitu tables.

--source logosint loads logosint's export (logosint_export.py, /hats/gc/cats_results/logosint/)
instead of GCwerks': same files and columns, but no GCwerks flags are read or synced (tag 324
rows already in the DB are left as they are), and a gas with no value at all in a row (logosint
refused that run's parameter file) is left alone in the DB rather than deleted; explicit zeros
still mean "not measured", but a row with tags (a manual reject, a GCwerks flag) is kept rather
than deleted, so the run stays rejected. A gas logosint excluded (XB: the ECD pegged high; rt -1 in
the export) is deleted with its tags, tagged or not. --dry-run writes nothing and reports what a load would change.

--start / --end (YYYY-MM-DD[ HH:MM] or YYMM; end exclusive) limit a load to rows in that window, and
--channel (CATS channel letters as in the export columns: q, a, f, ...) to the gases on those channels.
Rows are read, upserted and (for "no peak" rows) deleted only inside the window and channels; the
default two-month window does not apply when --start or --end is given.
"""

import bisect
import time
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import DateOffset
import typer

app = typer.Typer(add_completion=False)


# logosint's peak detail columns in its export (<gas>_<suffix>), in ng_insitu_peak_details order:
# width, skew, start_time, end_time, start_level, end_level, codes.
DETAIL_COLS = ("w", "skew", "st", "et", "sl", "el", "codes")


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

    # CATS port layout: 2=cal1 (Std), 4=air1, 6=cal2 (Ref), 8=air2. Odd port
    # numbers show up in the raw GCwerks export during valve malfunctions
    # (e.g. NWR Jun-Jul 2005, Sep 2018-Jul 2019, Jun 2023) -- never a real
    # CATS sample port -- so they're dropped before loading.
    VALID_PORTS = {2, 4, 6, 8}
    AIR_PORTS = (4, 8)

    # How far before a load's first row to look for existing air rows whose cal
    # method new rows take (see _carry_forward_methods()).
    CARRY_FORWARD_LOOKBACK = "1 YEAR"

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

    # Mols to upload to ng_insitu_mole_fractions. All channels for each mol
    # are uploaded; excluded mols (HCFCs, HFCs, CH3Cl, CH3Br, OCS, TCE, etc.)
    # remain in analyte_list for reference but are not loaded into the DB.
    UPLOAD_MOLS: frozenset = frozenset({
        "N2O", "SF6",
        "CFC12", "CFC11", "CFC113",
        "H1211", "CCl4", "CH3CCl3", "CHCl3",
    })

    # (site, mol) -> analysis time from which that analyte is no longer loaded: no new rows,
    # and existing rows are neither updated nor deleted (they carry the reject tag that ended it).
    # BRW CH3CCl3: a contaminant in the cal tanks installed 2023-06-16 co-elutes, so CH3CCl3 can't
    # be measured on cal runs (logosint channel 2 pilot; tag 40 via sql/brw_ch3ccl3_stop_230616.sql).
    STOP_AFTER: dict = {
        ("brw", "CH3CCl3"): "2023-06-16 15:54:00",
    }

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
        "H2":       4,
        "CH4":      2,
        "CO":       3,
        "TCE":      139,
        "H1301":    64,
        "HCFC22":   21,
        "CH3Cl":    23,
        "HCFC142b": 25,
        "CH3Br":    27,
    }

    SOURCES = ("gcwerks", "logosint")

    def __init__(self, site: str, flagged: bool = False, source: str = "gcwerks", dry_run: bool = False):
        site = site.lower()
        if source not in self.SOURCES:
            raise ValueError(f"source must be one of {self.SOURCES}")
        self.source = source
        self.dry_run = dry_run
        self.results_dir = self.RESULTS_DIR / "logosint" if source == "logosint" else self.RESULTS_DIR
        if source == "logosint" and flagged:
            print("  logosint results carry no GCwerks flags: not syncing tag 324 (existing 324 tags kept).")
            flagged = False
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
            f for f in self.results_dir.glob(f"{self.site}_*.csv")
            if "_2yr" not in f.name
        )

    @classmethod
    def _columns_channel(cls, columns) -> set:
        """Channel letters of a CSV's measurement columns (None for a column with no suffix)."""
        return {ch for (_mol, ch) in cls._parse_measurement_columns(columns)}

    @classmethod
    def _limit_channels(cls, df: pd.DataFrame, channels) -> pd.DataFrame:
        """Drop measurement columns of other channels (a file holding several would otherwise leak them)."""
        if not channels:
            return df
        drop = [col for (mol, ch), cols in cls._parse_measurement_columns(df.columns).items()
                if ch not in channels for col in cols.values()]
        return df.drop(columns=drop)

    def _select_files(self, channels=None) -> list[Path]:
        """The site's CSV files, limited to those whose gases are on one of `channels`
        (a file holds one gas, so its header names its channel)."""
        files = self._gcwerks_files()
        if channels:
            keep = []
            for f in files:
                cols = [c.strip() for c in pd.read_csv(f, skipinitialspace=True, nrows=0).columns]
                if self._columns_channel(cols) & set(channels):
                    keep.append(f)
            files = keep
        if not files:
            raise FileNotFoundError(
                f"No CATS CSV files found for site {self.site!r}"
                + (f" channel(s) {sorted(channels)}" if channels else "") + f" in {self.results_dir}"
            )
        return files

    def read_gcwerks(self, start_date=None, end_date=None, channels=None) -> pd.DataFrame:
        """Merge all per-molecule CSVs into one wide DataFrame keyed on (time, port).

        start_date / end_date (exclusive) filter each file immediately after loading so
        the merge works on a small window rather than the full history. channels limits
        the files read to those gases.
        """
        files = self._select_files(channels)
        print(f"  Reading {len(files)} {self.source} CSV files from {self.results_dir}"
              + (f" (channel {','.join(sorted(channels))})" if channels else "") + "...")

        frames = []
        for f in files:
            df = pd.read_csv(f, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            df["time"] = pd.to_datetime(df["time"])
            if start_date is not None:
                df = df.loc[df["time"] >= start_date]
            if end_date is not None:
                df = df.loc[df["time"] < end_date]
            df = self._limit_channels(df, channels)
            df["port"] = pd.to_numeric(df["port"], errors="coerce")
            for col in df.columns:
                if col not in ("time", "port") and not col.endswith("_codes"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").replace(-999, pd.NA)
            frames.append(df)

        print(f"  Merging...")
        merged = frames[0]
        for df in frames[1:]:
            merged = merged.merge(df, on=["time", "port"], how="outer")

        return merged.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    def read_gcwerks_flagged(self, start_date=None, end_date=None, channels=None) -> pd.DataFrame:
        """Merge per-molecule CSVs, preserving GCwerks flag characters (* / F).

        start_date / end_date (exclusive) filter each file immediately after loading;
        channels limits the files read.
        """
        files = self._select_files(channels)
        print(f"  Reading {len(files)} GCwerks CSV files (flagged)...")

        frames = []
        for f in files:
            df = pd.read_csv(f, skipinitialspace=True, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            df["time"] = pd.to_datetime(df["time"])
            if start_date is not None:
                df = df.loc[df["time"] >= start_date]
            if end_date is not None:
                df = df.loc[df["time"] < end_date]
            df = self._limit_channels(df, channels)
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
        metrics = {"ht", "area", "rt"} | set(DETAIL_COLS)   # logosint exports add the peak details
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
        if not self.dry_run:
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

    @staticmethod
    def _empty_measurement(*values) -> bool:
        """True when every chromatogram metric is either NULL-like or zero."""
        return all(value is None or pd.isna(value) or value == 0 for value in values)

    def _delete_mole_fraction_keys(self, keys: set, batch_size: int = 1000) -> None:
        """Delete tags, then mole-fraction rows, for invalid source records."""
        sorted_keys = sorted(keys)
        for i in range(0, len(sorted_keys), batch_size):
            chunk = sorted_keys[i:i + batch_size]
            where_terms = " OR ".join(
                ["(m.analysis_num = %s AND m.parameter_num = %s AND m.channel = %s)"]
                * len(chunk)
            )
            query_params = [value for key in chunk for value in key]
            if self._details_table():
                self.db.doquery(f"""
                DELETE d FROM hats.ng_insitu_peak_details d
                JOIN hats.ng_insitu_mole_fractions m ON d.ng_insitu_mole_fraction_num = m.num
                WHERE {where_terms}
            """, query_params)
            self.db.doquery(f"""
                DELETE t FROM hats.ng_insitu_mole_fraction_tags t
                JOIN hats.ng_insitu_mole_fractions m
                  ON t.ng_insitu_mole_fraction_num = m.num
                WHERE {where_terms}
            """, query_params)
            self.db.doquery(f"""
                DELETE m FROM hats.ng_insitu_mole_fractions m
                WHERE {where_terms}
            """, query_params)

    def _carry_forward_methods(self, t0: str, t1: str) -> dict:
        """(parameter_num, channel) -> (sorted analysis times, methods) of the
        existing air rows from CARRY_FORWARD_LOOKBACK before t0 through t1.

        A new row takes the method of the nearest air row at or before its own
        time (_method_for), so the cal method chosen for a period keeps going
        into newly ingested days, and a row inserted into the past (a reload
        that finds a peak the old integration didn't) gets the method of its
        own period rather than today's. Existing rows are never touched. Rows
        without a mole fraction count: an analyte on ref with no scale
        assignment for its tanks has none, and must stay on ref rather than
        fall through to the cal12 default.
        """
        sql = f"""
            SELECT m.parameter_num, m.channel, a.analysis_time, m.mf_method_num
            FROM hats.ng_insitu_mole_fractions m
            JOIN hats.ng_insitu_analysis a ON a.num = m.analysis_num
            WHERE a.inst_num = {self.inst_num}
              AND a.port IN ({','.join(str(p) for p in self.AIR_PORTS)})
              AND a.analysis_time >= '{t0}' - INTERVAL {self.CARRY_FORWARD_LOOKBACK}
              AND a.analysis_time <= '{t1}'
              AND m.mf_method_num IS NOT NULL
            ORDER BY a.analysis_time
        """
        out: dict = {}
        for r in self.db.doquery(sql) or []:
            key = (int(r["parameter_num"]), r["channel"])
            times, methods = out.setdefault(key, ([], []))
            times.append(pd.Timestamp(r["analysis_time"]).strftime("%Y-%m-%d %H:%M:%S"))
            methods.append(int(r["mf_method_num"]))
        return out

    def _method_for(self, methods: dict, param: int, channel: str, t: str) -> int:
        """Method of the nearest existing air row at or before t (else the
        first one after it, else the analyte's default)."""
        hit = methods.get((param, channel))
        if not hit:
            return self._default_mf_method(param)
        times, meths = hit
        i = bisect.bisect_right(times, t)
        return meths[i - 1] if i > 0 else meths[0]

    def _default_mf_method(self, param: int) -> int:
        # Same as CATS_Instrument.default_mf_method(): cal1 for CCl4, else cal12.
        return 3 if int(param) == 37 else 2

    def upsert_mole_fractions(self, df: pd.DataFrame, analysis_map: dict, batch_size: int = 1000):
        # mf_method_num is set only when a row is first inserted (carried
        # forward from the analyte's latest air row); the UPDATE
        # clause leaves it alone so methods set by hand are never reset.
        mole_sql = """
            INSERT INTO hats.ng_insitu_mole_fractions (
                analysis_num, parameter_num, channel, height, area, retention_time,
                mf_method_num
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
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
        missing = sorted({mol for (mol, _ch) in col_map
                          if mol in self.UPLOAD_MOLS and mol not in self.analytes})
        if missing:
            print(f"Warning: no analyte mapping for GCwerks mols: {missing}")

        methods = self._carry_forward_methods(df["analysis_time_str"].min(), df["analysis_time_str"].max()) \
            if len(df) else {}

        params = []
        all_keys: set = set()
        flagged_keys: set = set()
        invalid_keys: set = set()
        excluded_keys: set = set()   # logosint XB exclusions (rt -1 in the export)
        details: dict = {}   # key -> logosint's peak details, for hats.ng_insitu_peak_details

        for r in df.itertuples(index=False):
            analysis_num = analysis_map.get(r.analysis_time_str)
            if analysis_num is None:
                continue
            for (mol, ch), cols in col_map.items():
                if mol not in self.UPLOAD_MOLS:
                    continue
                stop = self.STOP_AFTER.get((self.site, mol))
                if stop is not None and r.analysis_time_str >= stop:
                    continue
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
                key = (analysis_num, param, channel)
                if self.source == "logosint" and ht is None and area is None and rt is None:
                    continue   # not in logosint's export (refused run): leave the DB's row alone
                if self.source == "logosint" and rt is not None and rt < 0:
                    excluded_keys.add(key)   # XB: the peak is gone, tags and all
                    continue
                if self._empty_measurement(ht, area, rt):
                    invalid_keys.add(key)
                    continue
                method = self._method_for(methods, param, channel, r.analysis_time_str)
                params.append((analysis_num, param, channel, ht, area, rt, method))
                all_keys.add(key)
                if self.source == "logosint":
                    d = self._details_of(r, cols)
                    if d is not None:
                        details[key] = d
                if flagged:
                    flagged_keys.add(key)

            if len(params) >= batch_size and not self.dry_run:
                self.db.doMultiInsert(mole_sql, params, all=True)
                params = []

        if self.source == "logosint" and invalid_keys:
            # Rows someone tagged (a manual reject, a GCwerks flag) keep their value and tags when
            # logosint finds no peak there: the run stays rejected and the tagging work isn't lost.
            kept = self._tagged_keys(invalid_keys)
            if kept:
                print(f"  Keeping {len(kept)} tagged rows where logosint found no peak.")
                invalid_keys = invalid_keys - kept

        if self.dry_run:
            self._dry_run_report(params, invalid_keys, analysis_map, excluded_keys)
            if details:
                print(f"  would write peak details for {len(details)} rows (hats.ng_insitu_peak_details)")
            return

        if params:
            self.db.doMultiInsert(mole_sql, params, all=True)
        if details and self._details_table():
            self._upsert_details(details, batch_size=batch_size)

        # Invalid exports mean that analyte/channel was not measured at this
        # analysis time. Remove any previously imported value so reprocessing
        # cannot leave either a zero row or a stale nonzero row behind. The
        # shared ng_insitu_analysis row remains for other valid analytes.
        if invalid_keys:
            self._delete_mole_fraction_keys(invalid_keys, batch_size=batch_size)
            print(f"  Removed/skipped {len(invalid_keys)} empty measurement rows.")
        if excluded_keys:
            # Unlike a plain missing peak, an XB exclusion goes even where tagged.
            self._delete_mole_fraction_keys(excluded_keys, batch_size=batch_size)
            print(f"  Removed {len(excluded_keys)} rows logosint excluded (XB), with their tags.")

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

    def _details_table(self) -> bool:
        """Whether hats.ng_insitu_peak_details exists (the loader runs without it, as before)."""
        if not hasattr(self, "_has_details"):
            self._has_details = bool(self.db.doquery("SHOW TABLES FROM hats LIKE 'ng_insitu_peak_details'"))
        return self._has_details

    @staticmethod
    def _details_of(r, cols: dict):
        """logosint's peak details for one gas in one row, as DETAIL_COLS order, or None when the
        export has none (a file written before they were added)."""
        vals = []
        for c in DETAIL_COLS:
            col = cols.get(c)
            v = getattr(r, col, None) if col else None
            vals.append(None if v is None or (not isinstance(v, str) and pd.isna(v)) or v == "" else v)
        if all(v is None for v in vals):
            return None
        return tuple(str(v).strip() if c == "codes" and v is not None else (None if v is None else float(v))
                     for c, v in zip(DETAIL_COLS, vals))

    def _upsert_details(self, details: dict, batch_size: int = 1000) -> None:
        """Write logosint's peak details to hats.ng_insitu_peak_details for the rows just upserted."""
        keys = sorted(details)
        sql = """
            INSERT INTO hats.ng_insitu_peak_details (
                ng_insitu_mole_fraction_num, width, skew, start_time, end_time, start_level, end_level, codes, source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'logosint')
            ON DUPLICATE KEY UPDATE width = VALUES(width), skew = VALUES(skew), start_time = VALUES(start_time),
                end_time = VALUES(end_time), start_level = VALUES(start_level), end_level = VALUES(end_level),
                codes = VALUES(codes), source = VALUES(source)
        """
        n = 0
        for i in range(0, len(keys), batch_size):
            chunk = keys[i:i + batch_size]
            where = " OR ".join(["(analysis_num = %s AND parameter_num = %s AND channel = %s)"] * len(chunk))
            rows = self.db.doquery(
                f"SELECT num, analysis_num, parameter_num, channel FROM hats.ng_insitu_mole_fractions WHERE {where}",
                [v for k in chunk for v in k]) or []
            vals = [(r["num"],) + details[(int(r["analysis_num"]), int(r["parameter_num"]), r["channel"])]
                    for r in rows if (int(r["analysis_num"]), int(r["parameter_num"]), r["channel"]) in details]
            if vals:
                self.db.doMultiInsert(sql, vals, all=True)
                n += len(vals)
        print(f"  Wrote peak details for {n} rows.")

    def _tagged_keys(self, keys: set, batch_size: int = 1000) -> set:
        """The (analysis_num, parameter_num, channel) keys among these whose row has any tag."""
        out = set()
        keys = sorted(keys)
        for i in range(0, len(keys), batch_size):
            chunk = keys[i:i + batch_size]
            where = " OR ".join(["(m.analysis_num = %s AND m.parameter_num = %s AND m.channel = %s)"] * len(chunk))
            rows = self.db.doquery(
                "SELECT DISTINCT m.analysis_num, m.parameter_num, m.channel FROM hats.ng_insitu_mole_fractions m "
                f"JOIN hats.ng_insitu_mole_fraction_tags t ON t.ng_insitu_mole_fraction_num = m.num WHERE {where}",
                [v for k in chunk for v in k]) or []
            out |= {(int(r["analysis_num"]), int(r["parameter_num"]), r["channel"]) for r in rows}
        return out

    def _dry_run_report(self, params: list, invalid_keys: set, analysis_map: dict, excluded_keys: set = frozenset(), batch_size: int = 1000) -> None:
        """What a load would change, against the DB's rows: per analyte and channel, new rows,
        height changes (median and share over 0.1%), and rows that would be deleted (with tags)."""
        nums = sorted({p[0] for p in params} | {k[0] for k in invalid_keys} | {k[0] for k in excluded_keys})
        have = {}
        tagged = set()
        tag_nums: dict = {}
        for i in range(0, len(nums), batch_size):
            chunk = nums[i:i + batch_size]
            marks = ",".join(["%s"] * len(chunk))
            for r in self.db.doquery(
                f"SELECT num, analysis_num, parameter_num, channel, height FROM hats.ng_insitu_mole_fractions "
                f"WHERE analysis_num IN ({marks})", chunk) or []:
                have[(int(r["analysis_num"]), int(r["parameter_num"]), r["channel"])] = (r["num"], r["height"])
            for r in self.db.doquery(
                f"SELECT m.analysis_num, m.parameter_num, m.channel, t.tag_num FROM hats.ng_insitu_mole_fraction_tags t "
                f"JOIN hats.ng_insitu_mole_fractions m ON m.num = t.ng_insitu_mole_fraction_num "
                f"WHERE m.analysis_num IN ({marks})", chunk) or []:
                k = (int(r["analysis_num"]), int(r["parameter_num"]), r["channel"])
                tagged.add(k)
                tag_nums.setdefault(k, set()).add(int(r["tag_num"]))
        rows = []
        for a_num, param, ch, ht, area, rt, _m in params:
            old = have.get((a_num, param, ch))
            rows.append({"param": param, "channel": ch, "new": old is None,
                         "dht": None if old is None or not old[1] or ht is None else 100.0 * (float(ht) / float(old[1]) - 1.0)})
        d = pd.DataFrame(rows)
        dels = [k for k in invalid_keys if k in have]
        print(f"  DRY RUN, nothing written. {len(params)} rows to upsert, {len(set(analysis_map.values()))} existing analyses matched.")
        if not d.empty:
            g = d.groupby(["param", "channel"])
            out = pd.DataFrame({
                "rows": g.size(), "new": g["new"].sum(),
                "median dht%": g["dht"].median().round(3),
                "|dht|>0.1%": g["dht"].apply(lambda x: f"{100.0 * (x.abs() > 0.1).mean():.1f}%" if x.notna().any() else ""),
            })
            print(out.to_string())
        counts: dict = {}
        for k in dels:
            for t in tag_nums.get(k, ()):
                counts[t] = counts.get(t, 0) + 1
        print(f"  would delete {len(dels)} existing rows (no peak in this export), "
              f"{sum(1 for k in dels if k in tagged)} of them with tags"
              + (" (tag_num: rows " + ", ".join(f"{t}: {n}" for t, n in sorted(counts.items())) + ")" if counts else ""))
        xdel = [k for k in excluded_keys if k in have]
        if excluded_keys:
            print(f"  would delete {len(xdel)} existing rows logosint excluded (XB), "
                  f"{sum(1 for k in xdel if k in tagged)} of them with tags (deleted with them)")

    @staticmethod
    def parse_when(s) -> "pd.Timestamp | None":
        """'YYMM' -> start of that month; anything else pandas reads (YYYY-MM-DD[ HH:MM])."""
        if s is None or isinstance(s, pd.Timestamp):
            return s
        s = str(s).strip()
        if len(s) == 4 and s.isdigit():
            yy = int(s[:2])
            return pd.Timestamp(1900 + yy if yy >= 70 else 2000 + yy, int(s[2:]), 1)
        return pd.Timestamp(s)

    def load(self, duration_months: int | None = 2, year: int | None = None,
             start=None, end=None, channels=None):
        """Read GCwerks CSVs and upsert into ng_insitu tables.

        start / end (end exclusive) limit the rows loaded to a time window and replace the
        default lookback; channels (letters such as 'q', 'a', 'f') limit the gases loaded.
        Nothing outside the window and channels is read, written or deleted.

        Run cats_export.py first to refresh the CSV files from GCwerks.
        """
        t0 = time.time()
        channels = {c.strip().lower() for c in channels if c.strip()} if channels else None
        bad = (channels or set()) - self.CATS_CHANNELS
        if bad:
            raise ValueError(f"unknown channel(s) {sorted(bad)}; valid: {sorted(self.CATS_CHANNELS)}")
        w_start, w_end = self.parse_when(start), self.parse_when(end)
        if w_start is not None and w_end is not None and w_end <= w_start:
            raise ValueError(f"--end {w_end} is not after --start {w_start}")

        # Determine start_date early so read methods can filter each file
        # before merging, avoiding full-history merges for short windows.
        end_date = None
        if w_start is not None or w_end is not None:
            start_date, end_date = w_start, w_end
        elif year is not None:
            start_date = pd.Timestamp(f"{year}-01-01")
        elif duration_months is not None:
            start_date = pd.Timestamp.now() - DateOffset(months=duration_months)
        else:
            start_date = None

        if w_start is not None or w_end is not None:
            label = f"{w_start if w_start is not None else 'beginning'} to {w_end if w_end is not None else 'end'}"
        else:
            label = f"{year}" if year else (f"last {duration_months}mo" if duration_months else "all data")
        if channels:
            label += f", channel {','.join(sorted(channels))}"
        print(f"CATS {self.site.upper()} — {label}")

        if self.flagged:
            df = self.read_gcwerks_flagged(start_date=start_date, end_date=end_date, channels=channels)
        else:
            df = self.read_gcwerks(start_date=start_date, end_date=end_date, channels=channels)

        # For year mode trim the upper bound; duration_months used today as anchor
        # so no upper trim needed.
        if year is not None:
            df = df.loc[df["time"].dt.year == int(year)]

        bad_port = ~df["port"].isin(self.VALID_PORTS)
        if bad_port.any():
            print(f"  Dropping {int(bad_port.sum())} rows on non-CATS ports "
                  f"(valid: {sorted(self.VALID_PORTS)}).")
            df = df.loc[~bad_port].reset_index(drop=True)

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
    flagged: bool = typer.Option(True, "--flagged/--no-flagged", help="Parse and sync GCwerks flag characters"),
    source: str = typer.Option("gcwerks", "--source", help="gcwerks (cats_results/) or logosint (cats_results/logosint/, no flags)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="write nothing; report what a load would change"),
    start: str | None = typer.Option(None, "--start", help="first time to load: YYYY-MM-DD[ HH:MM] or YYMM (replaces the default 2-month window)"),
    end: str | None = typer.Option(None, "--end", help="end of the window, exclusive: YYYY-MM-DD[ HH:MM] or YYMM"),
    channel: str | None = typer.Option(None, "--channel", help="only gases on these channel letters, comma separated (q, a, f, ...)"),
):
    """Load CATS GCwerks export data into HATS ng_insitu tables.

    Run cats_export.py first to refresh the per-molecule CSV files from GCwerks
    (or logosint_export.py for --source logosint).
    """
    if (start or end) and year is not None:
        typer.secho("--start/--end cannot be combined with --year", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    channels = [c for c in channel.split(",")] if channel else None
    cats = CATS_GCwerks2DB(site, flagged=flagged, source=source, dry_run=dry_run)
    try:
        if all_data:
            cats.load(duration_months=None, start=start, end=end, channels=channels)
        else:
            cats.load(year=year, start=start, end=end, channels=channels)
    except (ValueError, FileNotFoundError) as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
