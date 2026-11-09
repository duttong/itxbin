#!/usr/bin/env python3
"""Load CATS GCwerks export data into HATS ng_insitu tables.

--source logosint loads logosint's export (logosint_export.py, /hats/gc/cats_results/logosint/)
instead of GCwerks': same files and columns, but no GCwerks flags are read or synced (tag 324
rows already in the DB are left as they are), and a gas with no value at all in a row (logosint
refused that run's parameter file) is left alone in the DB rather than deleted; explicit zeros
still mean "not measured", but a row with tags (a manual reject, a GCwerks flag) is kept rather
than deleted, so the run stays rejected. --dry-run writes nothing and reports what a load would change.
"""

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

    # CATS port layout: 2=cal1 (Std), 4=air1, 6=cal2 (Ref), 8=air2. Odd port
    # numbers show up in the raw GCwerks export during valve malfunctions
    # (e.g. NWR Jun-Jul 2005, Sep 2018-Jul 2019, Jun 2023) -- never a real
    # CATS sample port -- so they're dropped before loading.
    VALID_PORTS = {2, 4, 6, 8}
    AIR_PORTS = (4, 8)

    # How far back to look for an analyte's current cal method when new rows
    # are inserted (see _carry_forward_methods()).
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

    def read_gcwerks(self, start_date=None) -> pd.DataFrame:
        """Merge all per-molecule CSVs into one wide DataFrame keyed on (time, port).

        start_date filters each file immediately after loading so the merge
        works on a small window rather than the full history.
        """
        files = self._gcwerks_files()
        if not files:
            raise FileNotFoundError(
                f"No CATS CSV files found for site {self.site!r} in {self.results_dir}"
            )
        print(f"  Reading {len(files)} {self.source} CSV files from {self.results_dir}...")

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
                f"No CATS CSV files found for site {self.site!r} in {self.results_dir}"
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

    def _carry_forward_methods(self) -> dict:
        """(parameter_num, channel) -> mf_method_num of the latest air row,
        looking back CARRY_FORWARD_LOOKBACK.

        New rows inherit it, so the cal method chosen for an analyte keeps
        going into newly ingested days; existing rows are never touched. Rows
        without a mole fraction count: an analyte on ref with no scale
        assignment for its tanks has none, and must stay on ref rather than
        fall through to the cal12 default.
        """
        sql = f"""
            SELECT m.parameter_num, m.channel, m.mf_method_num
            FROM hats.ng_insitu_mole_fractions m
            JOIN hats.ng_insitu_analysis a ON a.num = m.analysis_num
            JOIN (
                SELECT m2.parameter_num, m2.channel, MAX(a2.analysis_time) AS t
                FROM hats.ng_insitu_mole_fractions m2
                JOIN hats.ng_insitu_analysis a2 ON a2.num = m2.analysis_num
                WHERE a2.inst_num = {self.inst_num}
                  AND a2.port IN ({','.join(str(p) for p in self.AIR_PORTS)})
                  AND a2.analysis_time >= NOW() - INTERVAL {self.CARRY_FORWARD_LOOKBACK}
                  AND m2.mf_method_num IS NOT NULL
                GROUP BY m2.parameter_num, m2.channel
            ) last ON last.parameter_num = m.parameter_num
                  AND last.channel = m.channel AND last.t = a.analysis_time
            WHERE a.inst_num = {self.inst_num}
              AND a.port IN ({','.join(str(p) for p in self.AIR_PORTS)})
        """
        rows = self.db.doquery(sql) or []
        return {(int(r["parameter_num"]), r["channel"]): int(r["mf_method_num"])
                for r in rows if r["mf_method_num"] is not None}

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

        methods = self._carry_forward_methods()

        params = []
        all_keys: set = set()
        flagged_keys: set = set()
        invalid_keys: set = set()

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
                if self._empty_measurement(ht, area, rt):
                    invalid_keys.add(key)
                    continue
                method = methods.get((param, channel), self._default_mf_method(param))
                params.append((analysis_num, param, channel, ht, area, rt, method))
                all_keys.add(key)
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
            self._dry_run_report(params, invalid_keys, analysis_map)
            return

        if params:
            self.db.doMultiInsert(mole_sql, params, all=True)

        # Invalid exports mean that analyte/channel was not measured at this
        # analysis time. Remove any previously imported value so reprocessing
        # cannot leave either a zero row or a stale nonzero row behind. The
        # shared ng_insitu_analysis row remains for other valid analytes.
        if invalid_keys:
            self._delete_mole_fraction_keys(invalid_keys, batch_size=batch_size)
            print(f"  Removed/skipped {len(invalid_keys)} empty measurement rows.")

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

    def _dry_run_report(self, params: list, invalid_keys: set, analysis_map: dict, batch_size: int = 1000) -> None:
        """What a load would change, against the DB's rows: per analyte and channel, new rows,
        height changes (median and share over 0.1%), and rows that would be deleted (with tags)."""
        nums = sorted({p[0] for p in params} | {k[0] for k in invalid_keys})
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
):
    """Load CATS GCwerks export data into HATS ng_insitu tables.

    Run cats_export.py first to refresh the per-molecule CSV files from GCwerks
    (or logosint_export.py for --source logosint).
    """
    cats = CATS_GCwerks2DB(site, flagged=flagged, source=source, dry_run=dry_run)
    if all_data:
        cats.load(duration_months=None)
    else:
        cats.load(year=year)


if __name__ == "__main__":
    app()
