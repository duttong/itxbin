#!/usr/bin/env python3
"""
Read-only query helpers for a LOGOS data agent.

These helpers wrap the existing HATS database connection code and expose
small, curated functions that are safer for agent use than raw SQL.
"""

from __future__ import annotations

import sys as _sys, os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
del _here, _sys, _os

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
from matplotlib.figure import Figure

from logos_instruments import HATS_DB_Functions


def _normalize_site_code(site_code: str) -> str:
    return site_code.strip().upper()


def _normalize_site_query(site_query: str) -> str:
    return " ".join(site_query.strip().upper().split())


def _normalize_analyte_name(analyte: str) -> str:
    return "".join(ch for ch in analyte.upper() if ch.isalnum())


def _normalize_country_name(country: str) -> str:
    return " ".join(country.strip().upper().replace(".", "").split())


@dataclass(frozen=True)
class AnalyteMatch:
    display_name: str
    parameter_num: int
    channel: str | None


class LOGOSDataAgentTools:
    """Small read-only tool surface for agent-driven HATS database queries."""

    CONTINENT_ALIASES = {
        "AFRICA": "Africa",
        "ANTARCTICA": "Antarctica",
        "ASIA": "Asia",
        "EUROPE": "Europe",
        "NORTH AMERICA": "North America",
        "SOUTH AMERICA": "South America",
        "OCEANIA": "Oceania",
        "AUSTRALIA": "Oceania",
    }

    COUNTRY_TO_CONTINENT = {
        "AMERICAN SAMOA": "Oceania",
        "ARGENTINA": "South America",
        "AUSTRALIA": "Oceania",
        "AUSTRIA": "Europe",
        "BAHAMAS": "North America",
        "BARBADOS": "North America",
        "BELGIUM": "Europe",
        "BERMUDA": "North America",
        "BOLIVIA": "South America",
        "BOTSWANA": "Africa",
        "BRAZIL": "South America",
        "CANADA": "North America",
        "CAPE VERDE": "Africa",
        "CABO VERDE": "Africa",
        "CHILE": "South America",
        "CHINA": "Asia",
        "COLOMBIA": "South America",
        "COSTA RICA": "North America",
        "CZECH REPUBLIC": "Europe",
        "CZECHIA": "Europe",
        "DENMARK": "Europe",
        "ECUADOR": "South America",
        "FINLAND": "Europe",
        "FRANCE": "Europe",
        "FRENCH POLYNESIA": "Oceania",
        "GERMANY": "Europe",
        "GHANA": "Africa",
        "GREENLAND": "North America",
        "GUAM": "Oceania",
        "HONG KONG": "Asia",
        "ICELAND": "Europe",
        "INDIA": "Asia",
        "IRELAND": "Europe",
        "ISRAEL": "Asia",
        "ITALY": "Europe",
        "JAMAICA": "North America",
        "JAPAN": "Asia",
        "KENYA": "Africa",
        "MADAGASCAR": "Africa",
        "MAURITIUS": "Africa",
        "MEXICO": "North America",
        "NAMIBIA": "Africa",
        "NETHERLANDS": "Europe",
        "NEW ZEALAND": "Oceania",
        "NORWAY": "Europe",
        "PERU": "South America",
        "PORTUGAL": "Europe",
        "PUERTO RICO": "North America",
        "REPUBLIC OF KOREA": "Asia",
        "SOUTH KOREA": "Asia",
        "RUSSIAN FEDERATION": "Europe",
        "RUSSIA": "Europe",
        "SAMOA": "Oceania",
        "SEYCHELLES": "Africa",
        "SOUTH AFRICA": "Africa",
        "SPAIN": "Europe",
        "SWEDEN": "Europe",
        "SWITZERLAND": "Europe",
        "TAIWAN": "Asia",
        "THAILAND": "Asia",
        "TRINIDAD AND TOBAGO": "North America",
        "UNITED KINGDOM": "Europe",
        "UK": "Europe",
        "UNITED STATES": "North America",
        "USA": "North America",
        "URUGUAY": "South America",
        "VENEZUELA": "South America",
        "ANTARCTICA": "Antarctica",
    }

    M4_ANALYTE_ALIASES = {
        "CFC11B": "CFC11",
        "CH3CCL3B": "CH3CCL3",
    }

    PLOT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "logos_ai_plots"

    def __init__(self, inst_id: str = "fe3"):
        self.db = HATS_DB_Functions(inst_id=inst_id)
        self.inst_id = inst_id
        self.inst_num = self.db.inst_num

    def continent_for_country(self, country: str | None) -> str | None:
        if not country:
            return None
        key = _normalize_country_name(country)
        return self.COUNTRY_TO_CONTINENT.get(key)

    def get_site_info(self, site_query: str) -> dict[str, Any]:
        """
        Return site metadata from gmd.site, resolving by code or site name.

        Exact code match is preferred. If there is no code match, fall back to
        exact or partial case-insensitive name matches.
        """
        query_norm = _normalize_site_query(site_query)
        sql = """
            SELECT code, name, country, lat, lon, elev
            FROM gmd.site
            WHERE UPPER(code) = %s
               OR UPPER(name) = %s
               OR UPPER(name) LIKE %s
            ORDER BY
                CASE
                    WHEN UPPER(code) = %s THEN 0
                    WHEN UPPER(name) = %s THEN 1
                    ELSE 2
                END,
                code
            LIMIT 5;
        """
        like_pattern = f"%{query_norm}%"
        rows = self.db.doquery(
            sql,
            [query_norm, query_norm, like_pattern, query_norm, query_norm],
        )
        if not rows:
            raise ValueError(f"Unknown site query: {site_query}")
        if len(rows) > 1:
            return {
                "instrument": self.inst_id,
                "query": site_query,
                "matches": [dict(row) for row in rows],
            }
        site = dict(rows[0])
        site["continent"] = self.continent_for_country(site.get("country"))
        return {
            "instrument": self.inst_id,
            "query": site_query,
            "site": site,
        }

    def list_site_countries(self) -> dict[str, Any]:
        """Return distinct gmd.site.country values and their continent mapping."""
        sql = """
            SELECT DISTINCT country
            FROM gmd.site
            WHERE country IS NOT NULL
              AND TRIM(country) <> ''
            ORDER BY country;
        """
        rows = self.db.doquery(sql)
        countries = []
        unresolved = []
        for row in rows:
            country = row.get("country")
            continent = self.continent_for_country(country)
            countries.append({"country": country, "continent": continent})
            if continent is None:
                unresolved.append(country)
        return {
            "instrument": self.inst_id,
            "countries": countries,
            "unresolved_countries": unresolved,
        }

    def list_sites(
        self,
        min_lat: float | None = None,
        max_lat: float | None = None,
        min_lon: float | None = None,
        max_lon: float | None = None,
        country: str | None = None,
        continent: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Return sites with optional geographic and country/continent filtering."""
        clauses = []
        params: list[Any] = []
        if min_lat is not None:
            clauses.append("lat >= %s")
            params.append(float(min_lat))
        if max_lat is not None:
            clauses.append("lat <= %s")
            params.append(float(max_lat))
        if min_lon is not None:
            clauses.append("lon >= %s")
            params.append(float(min_lon))
        if max_lon is not None:
            clauses.append("lon <= %s")
            params.append(float(max_lon))
        where_sql = ""
        if clauses:
            where_sql = "WHERE " + " AND ".join(clauses)
        sql = f"""
            SELECT code, name, country, lat, lon, elev
            FROM gmd.site
            {where_sql}
            ORDER BY lat DESC, code;
        """
        rows = self.db.doquery(sql, params)
        country_norm = _normalize_country_name(country) if country else None
        continent_norm = self.CONTINENT_ALIASES.get(_normalize_country_name(continent), continent) if continent else None
        sites = []
        for row in rows:
            item = dict(row)
            item["continent"] = self.continent_for_country(item.get("country"))
            if country_norm and _normalize_country_name(str(item.get("country", ""))) != country_norm:
                continue
            if continent_norm and item.get("continent") != continent_norm:
                continue
            sites.append(item)
            if len(sites) >= int(limit):
                break
        return {
            "instrument": self.inst_id,
            "filters": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "country": country,
                "continent": continent_norm or continent,
                "limit": limit,
            },
            "sites": sites,
        }

    def list_supported_analytes(self) -> dict[str, Any]:
        """Return analytes available for the selected instrument."""
        sql = """
            SELECT display_name, param_num, channel
            FROM hats.analyte_list
            WHERE inst_num = %s
            ORDER BY display_name, channel;
        """
        rows = self.db.doquery(sql, [self.inst_num])
        analytes = [
            {
                "display_name": str(row["display_name"]),
                "parameter_num": int(row["param_num"]),
                "channel": None if pd.isna(row["channel"]) else str(row["channel"]),
            }
            for row in rows
        ]
        return {
            "instrument": self.inst_id,
            "analytes": analytes,
        }

    def resolve_analyte(self, analyte: str) -> dict[str, Any]:
        """
        Resolve a user-facing analyte name to a parameter number.

        Normalization removes spaces and hyphens, so 'CFC-11' matches 'CFC11'.
        For duplicated FE3 names like CFC11/CFC113, prefer channel 'c' when
        available because the codebase already treats that as the default.
        """
        sql = """
            SELECT display_name, param_num, channel
            FROM hats.analyte_list
            WHERE inst_num = %s
            ORDER BY display_name, channel;
        """
        df = pd.DataFrame(self.db.doquery(sql, [self.inst_num]))
        if df.empty:
            raise ValueError(f"No analytes found for instrument: {self.inst_id}")

        target = _normalize_analyte_name(analyte)
        if self.inst_num == 192:
            target = self.M4_ANALYTE_ALIASES.get(target, target)
        df["normalized_name"] = df["display_name"].map(_normalize_analyte_name)
        matches = df.loc[df["normalized_name"] == target].copy()
        if matches.empty:
            raise ValueError(f"Could not resolve analyte '{analyte}' for instrument '{self.inst_id}'")

        if target in {"CFC11", "CFC113"}:
            preferred = matches.loc[matches["channel"].fillna("").str.lower() == "c"]
            if not preferred.empty:
                matches = preferred

        row = matches.iloc[0]
        display_name = str(row["display_name"])
        match = AnalyteMatch(
            display_name=display_name,
            parameter_num=int(row["param_num"]),
            channel=None if pd.isna(row["channel"]) else str(row["channel"]),
        )
        return {
            "instrument": self.inst_id,
            "analyte": {
                "display_name": match.display_name,
                "parameter_num": match.parameter_num,
                "channel": match.channel,
            },
        }

    @staticmethod
    def _parse_sample_ids(raw_ids) -> list[Any]:
        sample_ids = []
        for value in str(raw_ids or "").split(","):
            value = value.strip()
            if not value:
                continue
            try:
                sample_ids.append(int(value))
            except ValueError:
                sample_ids.append(value)
        return sample_ids

    def _pair_counts_by_ids(self, site_code_norm: str, pair_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not pair_ids:
            return {}
        placeholders = ",".join(["%s"] * len(pair_ids))
        counts_sql = f"""
            SELECT
                pair_id_num,
                MAX(analysis_datetime) AS run_time,
                MAX(sample_datetime) AS sample_datetime,
                MAX(sample_id) AS sample_ids,
                MAX(n) AS num_analyses,
                COUNT(CASE WHEN pair_avg IS NOT NULL THEN 1 END) AS num_analytes_measured
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND UPPER(site) = %s
              AND pair_id_num IN ({placeholders})
            GROUP BY pair_id_num;
        """
        rows = self.db.doquery(counts_sql, [self.inst_num, site_code_norm] + pair_ids) or []
        return {int(row["pair_id_num"]): dict(row) for row in rows}

    def get_recent_flask_pairs(self, site_code: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Return recent flask-pair runs for a site.

        Define "recent" from hats.Status_MetData.sample_datetime_utc, which is
        the authoritative flask sample timestamp for the pair. Then enrich those
        pairs with analysis/analyte counts from ng_data_processing_view.
        """
        site_code_norm = _normalize_site_code(site_code)

        pair_sql = """
            SELECT
                PairID AS pair_id_num,
                Station AS site,
                sample_datetime_utc AS sample_datetime,
                Flask_1,
                Flask_2
            FROM hats.Status_MetData
            WHERE UPPER(Station) = %s
              AND PairID IS NOT NULL
              AND sample_datetime_utc IS NOT NULL
            ORDER BY sample_datetime_utc DESC, PairID DESC
            LIMIT %s;
        """

        pair_rows = self.db.doquery(pair_sql, [site_code_norm, int(limit)])
        if not pair_rows:
            return {
                "instrument": self.inst_id,
                "site_code": site_code_norm,
                "rows": [],
            }

        pair_ids = [int(row["pair_id_num"]) for row in pair_rows]
        counts_by_pair = self._pair_counts_by_ids(site_code_norm, pair_ids)

        out_rows = []
        for row in pair_rows:
            pair_id_num = int(row["pair_id_num"])
            counts = counts_by_pair.get(pair_id_num, {})
            item = {
                "pair_id_num": pair_id_num,
                "site": row.get("site"),
                "sample_datetime": row.get("sample_datetime"),
                "run_time": counts.get("run_time"),
                "num_analyses": counts.get("num_analyses", 0),
                "num_analytes_measured": counts.get("num_analytes_measured", 0),
            }
            raw_ids = counts.get("sample_ids")
            if not raw_ids:
                raw_ids = ",".join(
                    str(v) for v in (row.get("Flask_1"), row.get("Flask_2")) if v not in (None, "")
                )
            item["sample_ids"] = self._parse_sample_ids(raw_ids)
            out_rows.append(item)
        return {
            "instrument": self.inst_id,
            "site_code": _normalize_site_code(site_code),
            "rows": out_rows,
        }

    def get_recent_processed_flask_pairs(self, site_code: str, limit: int = 10) -> dict[str, Any]:
        """
        Return recent processed/analyzed flask pairs for a site.

        Define "recent" from the latest run_time in ng_data_processing_view, so
        only pairs that have actually been processed are included.
        """
        site_code_norm = _normalize_site_code(site_code)
        sql = """
            SELECT
                pair_id_num,
                MAX(run_time) AS run_time,
                MAX(sample_datetime) AS sample_datetime
            FROM hats.ng_data_processing_view
            WHERE inst_num = %s
              AND run_type_num = 1
              AND pair_id_num > 0
              AND rejected = 0
              AND UPPER(site) = %s
            GROUP BY pair_id_num
            ORDER BY run_time DESC, pair_id_num DESC
            LIMIT %s;
        """
        pair_rows = self.db.doquery(sql, [self.inst_num, site_code_norm, int(limit)]) or []
        pair_ids = [int(row["pair_id_num"]) for row in pair_rows]
        counts_by_pair = self._pair_counts_by_ids(site_code_norm, pair_ids)

        out_rows = []
        for row in pair_rows:
            pair_id_num = int(row["pair_id_num"])
            counts = counts_by_pair.get(pair_id_num, {})
            out_rows.append(
                {
                    "pair_id_num": pair_id_num,
                    "site": site_code_norm,
                    "sample_datetime": counts.get("sample_datetime", row.get("sample_datetime")),
                    "run_time": counts.get("run_time", row.get("run_time")),
                    "num_analyses": counts.get("num_analyses", 0),
                    "num_analytes_measured": counts.get("num_analytes_measured", 0),
                    "sample_ids": self._parse_sample_ids(counts.get("sample_ids")),
                }
            )
        return {
            "instrument": self.inst_id,
            "site_code": site_code_norm,
            "rows": out_rows,
        }

    def get_pair_metadata(self, pair_id_num: int) -> dict[str, Any]:
        """Return flask IDs, flask type, and met data for a flask pair."""
        sql = """
            SELECT
                PairID AS pair_id_num,
                Station AS site_code,
                Flask_1,
                Flask_2,
                Flask_Type,
                Sample_Date,
                sample_datetime_utc,
                Wind_Speed,
                Wind_Direction,
                Air_Temp,
                Dew_Point,
                Precipitation,
                Sky,
                Comments,
                Operator
            FROM hats.Status_MetData
            WHERE PairID = %s
            ORDER BY sample_datetime_utc DESC
            LIMIT 1;
        """
        rows = self.db.doquery(sql, [int(pair_id_num)])
        if not rows:
            raise ValueError(f"No metadata found for pair_id_num={pair_id_num}")

        row = dict(rows[0])
        sample_ids = []
        for key in ("Flask_1", "Flask_2"):
            value = row.get(key)
            if value not in (None, ""):
                try:
                    sample_ids.append(int(value))
                except (TypeError, ValueError):
                    sample_ids.append(value)

        return {
            "instrument": self.inst_id,
            "pair_metadata": {
                "pair_id_num": row.get("pair_id_num"),
                "site_code": row.get("site_code"),
                "sample_ids": sample_ids,
                "flask_type": row.get("Flask_Type"),
                "sample_date": row.get("Sample_Date"),
                "sample_datetime_utc": row.get("sample_datetime_utc"),
                "wind_speed": row.get("Wind_Speed"),
                "wind_direction": row.get("Wind_Direction"),
                "air_temp": row.get("Air_Temp"),
                "dew_point": row.get("Dew_Point"),
                "precipitation": row.get("Precipitation"),
                "sky": row.get("Sky"),
                "operator": row.get("Operator"),
                "comments": row.get("Comments"),
            },
        }

    def get_recent_flask_pairs_with_metadata(self, site_code: str, limit: int = 10) -> dict[str, Any]:
        """Return recent flask pairs and enrich them with Status_MetData metadata."""
        pairs_payload = self.get_recent_flask_pairs(site_code, limit=limit)
        rows: list[dict[str, Any]] = []

        for pair_row in pairs_payload["rows"]:
            pair_id_num = pair_row.get("pair_id_num")
            enriched = dict(pair_row)
            if pair_id_num not in (None, 0):
                try:
                    meta = self.get_pair_metadata(int(pair_id_num))["pair_metadata"]
                except ValueError:
                    meta = None
                enriched["pair_metadata"] = meta
            else:
                enriched["pair_metadata"] = None
            rows.append(enriched)

        return {
            "instrument": self.inst_id,
            "site_code": _normalize_site_code(site_code),
            "rows": rows,
        }

    def get_recent_processed_flask_pairs_with_metadata(self, site_code: str, limit: int = 10) -> dict[str, Any]:
        """Return recent processed/analyzed flask pairs enriched with Status_MetData metadata."""
        pairs_payload = self.get_recent_processed_flask_pairs(site_code, limit=limit)
        rows: list[dict[str, Any]] = []

        for pair_row in pairs_payload["rows"]:
            pair_id_num = pair_row.get("pair_id_num")
            enriched = dict(pair_row)
            if pair_id_num not in (None, 0):
                try:
                    meta = self.get_pair_metadata(int(pair_id_num))["pair_metadata"]
                except ValueError:
                    meta = None
                enriched["pair_metadata"] = meta
            else:
                enriched["pair_metadata"] = None
            rows.append(enriched)

        return {
            "instrument": self.inst_id,
            "site_code": _normalize_site_code(site_code),
            "rows": rows,
        }

    def get_site_flask_mean(
        self,
        site_code: str,
        analyte: str,
        start_date: str,
        end_date: str,
    ) -> dict[str, Any]:
        """Return summary statistics for good flask data in a date window."""
        analyte_info = self.resolve_analyte(analyte)
        sql = """
            SELECT
                COUNT(*) AS num_measurements,
                MIN(sample_datetime) AS first_sample_datetime,
                MAX(sample_datetime) AS last_sample_datetime,
                AVG(pair_avg) AS mean_mole_fraction,
                STDDEV_SAMP(pair_avg) AS stddev_mole_fraction,
                MIN(pair_avg) AS min_mole_fraction,
                MAX(pair_avg) AS max_mole_fraction
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
              AND UPPER(site) = %s
              AND sample_datetime >= %s
              AND sample_datetime < %s
              AND pair_avg IS NOT NULL;
        """
        params = [
            self.inst_num,
            analyte_info["analyte"]["parameter_num"],
            _normalize_site_code(site_code),
            start_date,
            end_date,
        ]
        rows = self.db.doquery(sql, params)
        summary = dict(rows[0]) if rows else {}
        return {
            "site_code": _normalize_site_code(site_code),
            "instrument": self.inst_id,
            "analyte": analyte_info,
            "start_date": start_date,
            "end_date": end_date,
            "summary": summary,
        }

    def get_recent_flask_values(
        self,
        site_code: str,
        analyte: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the most recent good flask values for a site/analyte."""
        analyte_info = self.resolve_analyte(analyte)
        sql = """
            SELECT
                sample_datetime,
                analysis_datetime,
                pair_id_num,
                sample_id,
                pair_avg AS mole_fraction,
                n AS num_analyses,
                pair_stdv AS stddev_mole_fraction,
                sample_type,
                Wind_Speed AS wind_speed,
                Wind_Direction AS wind_direction
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
              AND UPPER(site) = %s
              AND pair_avg IS NOT NULL
            ORDER BY sample_datetime DESC, analysis_datetime DESC
            LIMIT %s;
        """
        params = [
            self.inst_num,
            analyte_info["analyte"]["parameter_num"],
            _normalize_site_code(site_code),
            int(limit),
        ]
        rows = self.db.doquery(sql, params)
        return {
            "instrument": self.inst_id,
            "site_code": _normalize_site_code(site_code),
            "analyte": analyte_info,
            "rows": [dict(row) for row in rows],
        }

    def plot_site_timeseries(
        self,
        site_code: str,
        analyte: str,
        start_date: str,
        end_date: str | None = None,
        aggregation: str = "raw",
        output_format: str = "png",
        overlay_recent_pairs_limit: int | None = None,
    ) -> dict[str, Any]:
        """
        Create a local PNG plot for site/analyte flask data.

        Supported aggregations:
        - raw
        - monthly_mean
        """
        output_format_norm = output_format.strip().lower()
        if output_format_norm != "png":
            raise ValueError("Only PNG output is currently supported.")

        aggregation_norm = aggregation.strip().lower()
        if aggregation_norm not in {"raw", "monthly_mean"}:
            raise ValueError("aggregation must be 'raw' or 'monthly_mean'")

        analyte_info = self.resolve_analyte(analyte)
        site_code_norm = _normalize_site_code(site_code)
        if not end_date:
            end_date = (datetime.utcnow().date() + timedelta(days=1)).isoformat()

        sql = """
            SELECT
                sample_datetime,
                analysis_datetime,
                sample_id,
                pair_id_num,
                pair_avg,
                n,
                pair_stdv,
                sample_type,
                Wind_Speed,
                Wind_Direction
            FROM hats.ng_pair_avg_view
            WHERE inst_num = %s
              AND parameter_num = %s
              AND UPPER(site) = %s
              AND sample_datetime >= %s
              AND sample_datetime < %s
              AND pair_avg IS NOT NULL
            ORDER BY sample_datetime ASC, analysis_datetime ASC;
        """
        params = [
            self.inst_num,
            analyte_info["analyte"]["parameter_num"],
            site_code_norm,
            start_date,
            end_date,
        ]
        rows = self.db.doquery(sql, params) or []
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError(
                f"No flask data found for {site_code_norm} {analyte_info['analyte']['display_name']} "
                f"between {start_date} and {end_date}."
            )

        df["sample_datetime"] = pd.to_datetime(df["sample_datetime"])
        df["pair_avg"] = pd.to_numeric(df["pair_avg"], errors="coerce")
        df = df.dropna(subset=["sample_datetime", "pair_avg"]).copy()
        if df.empty:
            raise ValueError(
                f"No plottable mole-fraction data found for {site_code_norm} "
                f"{analyte_info['analyte']['display_name']} between {start_date} and {end_date}."
            )
        sample_df = (
            df.rename(
                columns={
                    "pair_avg": "mean_mole_fraction",
                    "n": "num_analyses",
                    "pair_stdv": "stddev_mole_fraction",
                    "Wind_Speed": "wind_speed",
                    "Wind_Direction": "wind_direction",
                }
            )
            .sort_values(["sample_datetime", "pair_id_num"])
            .reset_index(drop=True)
        )

        plot_rows: list[dict[str, Any]]
        if aggregation_norm == "monthly_mean":
            monthly = (
                sample_df.assign(month_start=sample_df["sample_datetime"].dt.to_period("M").dt.to_timestamp())
                .groupby("month_start", as_index=False)
                .agg(
                    mean_mole_fraction=("mean_mole_fraction", "mean"),
                    stddev_mole_fraction=("mean_mole_fraction", "std"),
                    num_measurements=("mean_mole_fraction", "size"),
                )
            )
            monthly["stderr_mole_fraction"] = (
                monthly["stddev_mole_fraction"] / monthly["num_measurements"] ** 0.5
            )
            plot_rows = monthly.to_dict("records")
        else:
            plot_rows = sample_df.to_dict("records")

        self.PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        analyte_name = analyte_info["analyte"]["display_name"]
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in analyte_name)
        start_token = start_date.replace("-", "")
        end_token = str(end_date).replace("-", "")
        suffix = "monthly_mean" if aggregation_norm == "monthly_mean" else "raw"
        if overlay_recent_pairs_limit:
            suffix += f"_overlay{int(overlay_recent_pairs_limit)}"
        output_path = self.PLOT_OUTPUT_DIR / (
            f"{site_code_norm}_{safe_name}_{start_token}_{end_token}_{suffix}.png"
        )

        fig = Figure(figsize=(10, 4.8), facecolor="white")
        ax = fig.add_subplot(111)
        color = "#1f4f8a"
        overlay_rows: list[dict[str, Any]] = []

        if aggregation_norm == "monthly_mean":
            monthly = pd.DataFrame(plot_rows)
            monthly["month_start"] = pd.to_datetime(monthly["month_start"])
            ax.plot(
                monthly["month_start"],
                monthly["mean_mole_fraction"],
                marker="o",
                linewidth=1.8,
                color=color,
                label="Monthly mean",
            )
            stderr = monthly["stderr_mole_fraction"].fillna(0.0)
            ax.fill_between(
                monthly["month_start"],
                monthly["mean_mole_fraction"] - stderr,
                monthly["mean_mole_fraction"] + stderr,
                color=color,
                alpha=0.18,
                label="±1 standard error",
            )
            ax.set_title(
                f"{site_code_norm} {analyte_name} monthly means\n{start_date} to {end_date}"
            )
        else:
            ax.scatter(
                sample_df["sample_datetime"],
                sample_df["mean_mole_fraction"],
                marker="o",
                s=18,
                color=color,
                alpha=0.85,
                label="Flask sample mean",
            )
            ax.set_title(f"{site_code_norm} {analyte_name}\n{start_date} to {end_date}")

        if overlay_recent_pairs_limit:
            recent_limit = int(overlay_recent_pairs_limit)
            overlay_df = sample_df.sort_values("sample_datetime", ascending=False).head(recent_limit).copy()
            overlay_df = overlay_df.sort_values("sample_datetime").reset_index(drop=True)
            if not overlay_df.empty:
                overlay_rows = overlay_df.to_dict("records")
                ax.scatter(
                    overlay_df["sample_datetime"],
                    overlay_df["mean_mole_fraction"],
                    marker="o",
                    s=38,
                    color="#d97706",
                    edgecolors="#9a5a04",
                    linewidths=0.5,
                    alpha=0.95,
                    label=f"{recent_limit} most recent pair means",
                    zorder=4,
                )

        units = "ppt"
        ax.set_xlabel("Sample datetime")
        ax.set_ylabel(f"{analyte_name} mole fraction ({units})")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)

        return {
            "instrument": self.inst_id,
            "site_code": site_code_norm,
            "analyte": analyte_info,
            "start_date": start_date,
            "end_date": end_date,
            "aggregation": aggregation_norm,
            "output_format": output_format_norm,
            "image_path": str(output_path),
            "num_points": int(len(plot_rows)),
            "num_pairs": int(len(sample_df)),
            "overlay_recent_pairs_limit": int(overlay_recent_pairs_limit) if overlay_recent_pairs_limit else None,
            "overlay_num_points": int(len(overlay_rows)),
            "rows": [
                {
                    key: (value.isoformat() if hasattr(value, "isoformat") else value)
                    for key, value in row.items()
                }
                for row in plot_rows
            ],
            "overlay_rows": [
                {
                    key: (value.isoformat() if hasattr(value, "isoformat") else value)
                    for key, value in row.items()
                }
                for row in overlay_rows
            ],
        }

    def compare_site_year_to_recent(
        self,
        site_code: str,
        analyte: str,
        year: int,
        recent_limit: int = 10,
    ) -> dict[str, Any]:
        """
        Compare a site's yearly flask mean to the most recent flask values.
        """
        start_date = f"{int(year):04d}-01-01"
        end_date = f"{int(year) + 1:04d}-01-01"
        yearly = self.get_site_flask_mean(site_code, analyte, start_date, end_date)
        recent_payload = self.get_recent_flask_values(site_code, analyte, limit=recent_limit)
        recent_rows = recent_payload["rows"]

        recent_df = pd.DataFrame(recent_rows)
        recent_summary: dict[str, Any] = {
            "num_measurements": 0,
            "mean_mole_fraction": None,
            "first_sample_datetime": None,
            "last_sample_datetime": None,
        }
        if not recent_df.empty:
            recent_summary = {
                "num_measurements": int(len(recent_df)),
                "mean_mole_fraction": float(recent_df["mole_fraction"].mean()),
                "first_sample_datetime": recent_df["sample_datetime"].min(),
                "last_sample_datetime": recent_df["sample_datetime"].max(),
            }

        year_mean = yearly["summary"].get("mean_mole_fraction")
        recent_mean = recent_summary.get("mean_mole_fraction")
        mean_difference = None
        if year_mean is not None and recent_mean is not None:
            mean_difference = float(recent_mean) - float(year_mean)

        return {
            "site_code": _normalize_site_code(site_code),
            "instrument": self.inst_id,
            "year": int(year),
            "analyte": yearly["analyte"],
            "yearly_summary": yearly["summary"],
            "recent_summary": recent_summary,
            "recent_rows": recent_rows,
            "mean_difference_recent_minus_year": mean_difference,
        }
