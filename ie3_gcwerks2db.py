#! /usr/bin/env python

from pathlib import Path

import pandas as pd
from ie3_export import IE3_Process
from pandas.tseries.offsets import DateOffset
import typer

app = typer.Typer(add_completion=False)


class IE3_GCwerks2DB:
    """Load IE3 GCwerks export data into HATS ng_insitu tables."""

    VALID_SITES = {"smo", "mlo", "spo", "brw"}
    INST_NUM_BY_SITE = {"smo": 236}
    RUN_TIME_GAP_MIN = 15

    def __init__(self, site: str, flagged: bool = False):
        site = site.lower()
        if site not in self.VALID_SITES:
            raise ValueError(f"Invalid site {site!r}. Valid sites: {sorted(self.VALID_SITES)}")

        self.site = site
        self.flagged = flagged
        try:
            self.inst_num = self.INST_NUM_BY_SITE[site]
        except KeyError as e:
            raise ValueError(
                f"Missing inst_num for site {site!r}. "
                f"Set INST_NUM_BY_SITE for this site."
            ) from e
        suffix = "_flagged" if flagged else ""
        self.gcwerks_file = Path(f"/hats/gc/{site}/results/ie3_{site}_gcwerks_all{suffix}.csv")

        # database connection
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

    def _query_analytes(self) -> dict:
        sql = (
            "SELECT param_num, display_name "
            f"FROM hats.analyte_list WHERE inst_num = {self.inst_num};"
        )
        df = pd.DataFrame(self.db.doquery(sql))
        if df.empty:
            raise ValueError(
                f"No analytes found for inst_num {self.inst_num} in hats.analyte_list. "
                "Confirm inst_num is correct and analytes are loaded."
            )
        return dict(zip(df["display_name"], df["param_num"]))

    def read_gcwerks(self) -> pd.DataFrame:
        if not self.gcwerks_file.exists():
            raise FileNotFoundError(self.gcwerks_file)

        df = pd.read_csv(self.gcwerks_file, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df["time"] = pd.to_datetime(df["time"])

        # de-dup by time
        df = df.drop_duplicates("time", keep="last")
        return df

    @staticmethod
    def _flag_column_name(mol: str, channel: str | None) -> str:
        return f"{mol}_{channel}_flag" if channel else f"{mol}_flag"

    def read_gcwerks_flagged(self) -> pd.DataFrame:
        if not self.gcwerks_file.exists():
            raise FileNotFoundError(self.gcwerks_file)

        df = pd.read_csv(self.gcwerks_file, skipinitialspace=True, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        df["time"] = pd.to_datetime(df["time"])

        col_map = self._parse_measurement_columns(df.columns)
        for (mol, channel), cols in col_map.items():
            flag_col = self._flag_column_name(mol, channel)
            df[flag_col] = False
            for metric in ("ht", "area", "rt"):
                col = cols.get(metric)
                if not col:
                    continue
                series = df[col].fillna("").astype(str).str.strip()
                flagged = series.str.endswith(("F", "*"))
                cleaned = series.str.replace(r"[F*]$", "", regex=True)
                df[col] = pd.to_numeric(cleaned, errors="coerce").replace(-999, pd.NA)
                df[flag_col] = df[flag_col] | flagged

        # de-dup by time
        df = df.drop_duplicates("time", keep="last")
        return df

    def _assign_run_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("time").copy()
        gap = df["time"].diff() > pd.Timedelta(minutes=self.RUN_TIME_GAP_MIN)
        segment = gap.cumsum()
        df["run_time"] = df.groupby(segment)["time"].transform("first")
        return df

    @staticmethod
    def _parse_measurement_columns(columns):
        """Return mapping: (mol, chan) -> {metric: column_name}."""
        metrics = {"ht", "area", "rt"}
        mapping = {}
        for col in columns:
            parts = col.split("_")
            if len(parts) < 2:
                continue
            metric = parts[-1]
            if metric not in metrics:
                continue

            channel = None
            if len(parts) >= 3 and parts[-2] in {"a", "b", "c"}:
                channel = parts[-2]
                mol = "_".join(parts[:-2])
            else:
                mol = "_".join(parts[:-1])

            mapping.setdefault((mol, channel), {})[metric] = col
        return mapping

    def _normalize_measurements(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if col.endswith("_ht") or col.endswith("_area") or col.endswith("_rt"):
                df[col] = pd.to_numeric(df[col], errors="coerce").replace(-999, pd.NA)
        return df

    def upsert_analysis(self, df: pd.DataFrame, batch_size: int = 500) -> dict:
        df = df.copy()
        df["analysis_time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["run_time_str"] = df["run_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df["port"] = pd.to_numeric(df["port"], errors="coerce").fillna(0).astype(int)

        analysis_sql = """
            INSERT INTO hats.ng_insitu_analysis (
                analysis_time,
                run_time,
                site_num,
                inst_num,
                port
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
            batch = params[i : i + batch_size]
            self.db.doMultiInsert(analysis_sql, batch, all=True)

        # fetch analysis nums
        analysis_map = {}
        unique_times = df["analysis_time_str"].unique().tolist()
        for i in range(0, len(unique_times), batch_size):
            chunk = unique_times[i : i + batch_size]
            placeholders = ",".join(["%s"] * len(chunk))
            select_sql = (
                "SELECT analysis_time, num "
                "FROM hats.ng_insitu_analysis "
                "WHERE inst_num = %s AND site_num = %s "
                f"AND analysis_time IN ({placeholders})"
            )
            rows = self.db.doquery(select_sql, [self.inst_num, self.site_num] + chunk)
            analysis_map.update(
                {
                    r["analysis_time"].strftime("%Y-%m-%d %H:%M:%S"): r["num"]
                    for r in rows
                }
            )
        return analysis_map

    def upsert_mole_fractions(self, df: pd.DataFrame, analysis_map: dict, batch_size: int = 1000):
        if self.flagged:
            mole_sql = """
                INSERT INTO hats.ng_insitu_mole_fractions (
                    analysis_num,
                    parameter_num,
                    channel,
                    flag,
                    height,
                    area,
                    retention_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    flag           = IF(VALUES(flag) = 'W..', 'W..', flag),
                    height         = VALUES(height),
                    area           = VALUES(area),
                    retention_time = VALUES(retention_time)
            """
        else:
            mole_sql = """
                INSERT INTO hats.ng_insitu_mole_fractions (
                    analysis_num,
                    parameter_num,
                    channel,
                    flag,
                    height,
                    area,
                    retention_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    flag           = VALUES(flag),
                    height         = VALUES(height),
                    area           = VALUES(area),
                    retention_time = VALUES(retention_time)
            """

        col_map = self._parse_measurement_columns(df.columns)
        missing = sorted({mol for (mol, _ch) in col_map if mol not in self.analytes})
        if missing:
            print(f"Warning: missing analyte mapping for {missing}")

        params = []
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
                if not ht_col or not area_col or not rt_col:
                    continue
                ht = getattr(r, ht_col)
                area = getattr(r, area_col)
                rt = getattr(r, rt_col)
                if pd.isna(ht):
                    ht = None
                if pd.isna(area):
                    area = None
                if pd.isna(rt):
                    rt = None
                flagged = bool(getattr(r, flag_col, False))
                flag = "W.." if flagged else "..."
                params.append((analysis_num, param, channel, flag, ht, area, rt))

            if len(params) >= batch_size:
                self.db.doMultiInsert(mole_sql, params, all=True)
                params = []

        if params:
            self.db.doMultiInsert(mole_sql, params, all=True)

    def load(self, duration_months=2, year=None):
        IE3_Process(site=self.site, flagged=self.flagged).export_onefile()
        if self.flagged:
            df = self.read_gcwerks_flagged()
        else:
            df = self.read_gcwerks()
            df = self._normalize_measurements(df)
        df = self._assign_run_time(df)
        df["analysis_time_str"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        if year is not None:
            df = df.loc[df["time"].dt.year == int(year)]
        elif duration_months is not None:
            last_date = df["time"].max()
            start_date = last_date - DateOffset(months=duration_months)
            df = df.loc[df["time"] >= start_date]

        analysis_map = self.upsert_analysis(df)
        self.upsert_mole_fractions(df, analysis_map)


@app.command()
def load(
    site: str = typer.Argument("smo", help="Site code (default: smo)"),
    all_data: bool = typer.Option(False, "--all", help="Process all data"),
    year: int | None = typer.Option(None, "--year", help="Process a single year (YYYY)"),
    flagged: bool = typer.Option(False, "--flagged", help="Load the flagged GCwerks export file"),
):
    """Load IE3 GCwerks export data into HATS ng_insitu tables."""
    ie3 = IE3_GCwerks2DB(site, flagged=flagged)
    if all_data:
        ie3.load(duration_months=None)
    else:
        ie3.load(year=year)


if __name__ == "__main__":
    app()
