#!/usr/bin/env python3
"""
Free-form chat agent for LOGOS data queries.

This layer sits above LOGOSDataAgentTools and lets a model choose which
read-only tools to call in order to answer user questions.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

from logos_agent_tools import LOGOSDataAgentTools


def _load_dotenv(dotenv_path: str | Path = ".env") -> None:
    """Minimal .env loader that only sets variables not already exported."""
    path = Path(dotenv_path)
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


_load_dotenv()


class LOGOSChatAgent:
    """Small OpenAI-backed agent loop over LOGOSDataAgentTools."""

    DEFAULT_MODEL = os.environ.get("LOGOS_AI_MODEL", "gpt-5-mini")
    DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def __init__(
        self,
        tools: LOGOSDataAgentTools,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        request_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.tools = tools
        self.model = model or self.DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.previous_response_id: str | None = None
        self.request_fn = request_fn or self._responses_create

    def reset_conversation(self) -> None:
        self.previous_response_id = None

    def ask(self, prompt: str, context: dict[str, Any] | None = None) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export an API key before using LOGOS AI."
            )

        instructions = self._build_instructions(context or {})
        response = self.request_fn(
            self._request_payload(
                input_items=[self._user_message(prompt)],
                instructions=instructions,
                previous_response_id=self.previous_response_id,
            )
        )
        image_markers: list[str] = []

        while True:
            function_calls = self._extract_function_calls(response)
            if not function_calls:
                self.previous_response_id = response.get("id", self.previous_response_id)
                text = self._extract_output_text(response).strip()
                if image_markers:
                    joined = "\n".join(image_markers)
                    text = f"{text}\n\n{joined}" if text else joined
                return text or "No text response returned."

            tool_outputs = []
            for call in function_calls:
                result = self._execute_tool(call["name"], call["arguments"])
                if isinstance(result, dict) and result.get("image_path"):
                    image_markers.append(f"[[image:{result['image_path']}]]")
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": json.dumps(result, default=str),
                    }
                )

            response = self.request_fn(
                self._request_payload(
                    input_items=tool_outputs,
                    instructions=instructions,
                    previous_response_id=response.get("id"),
                )
            )

    def _request_payload(
        self,
        input_items: list[dict[str, Any]],
        instructions: str,
        previous_response_id: str | None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": input_items,
            "instructions": instructions,
            "tools": self._tool_specs(),
        }
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        return payload

    def _responses_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI API connection error: {exc.reason}") from exc

    def _build_instructions(self, context: dict[str, Any]) -> str:
        current_analyte = context.get("current_analyte") or "unknown"
        current_run_time = context.get("current_run_time") or "unknown"
        current_channel = context.get("current_channel") or "unknown"
        return (
            "You are LOGOS AI, a read-only data assistant for NOAA/GML LOGOS workflows. "
            "Use tools for factual answers about sites, flask pairs, pair metadata, analytes, "
            "recent flask values, and summary comparisons. "
            "For geographic filters: north/south refer to latitude, east/west refer to longitude. "
            "Use positive numbers for north/east and negative numbers for south/west when calling tools. "
            "For continent questions, use the country-to-continent mapping derived from gmd.site.country values rather than guessing from memory. "
            "When the user phrases a range in words, normalize it in the reply, for example "
            "'between 20 north and 30 south' should be described as latitude -30 to 20 degrees. "
            "Be careful with the meaning of recency for flask pairs: "
            "'recent pairs' means recent sampled pairs by Status_MetData.sample_datetime_utc, "
            "while 'recent processed pairs' or 'recent analyzed pairs' means pairs with actual "
            "processing rows ranked by ng_data_processing_view.run_time. "
            "If the user asks for a plot and the site, analyte, and date window are reasonably clear, call the plotting tool directly instead of asking unnecessary follow-up questions. "
            "For plot requests, default to PNG output. "
            "If the user says 'to now', 'through today', or similar, use the current date as the end of the range. "
            "If the user says 'monthly means', 'monthly mean', or similar, use aggregation 'monthly_mean'. Otherwise default to aggregation 'raw'. "
            "If the user asks to overlay the most recent pair means on the same PNG, pass overlay_recent_pairs_limit with the requested count. "
            "Do not ask whether to use PNG vs CSV unless the user explicitly asks for export options. "
            "Do not ask whether to use the default resolved analyte channel unless the user explicitly requests multiple channels or a non-default channel. "
            "Only describe an overlay as present if the plotting tool returned overlay rows or a non-zero overlay point count. "
            "If a plot tool returns an image path, mention the plot briefly and assume the UI will render it inline. "
            "Ask a follow-up question for a plot only when a required item is genuinely missing or ambiguous, such as the site, analyte, or date window. "
            "Do not invent database results or analytes. If a site, analyte, year, or pair ID is "
            "missing, ask a concise follow-up question or use current UI context when appropriate. "
            "Prefer the current GUI context unless the user overrides it explicitly. "
            f"Current instrument: {self.tools.inst_id}. "
            f"Current date: {datetime.utcnow().date().isoformat()}. "
            f"Current analyte: {current_analyte}. "
            f"Current run time: {current_run_time}. "
            f"Current channel: {current_channel}. "
            "When comparing values, explain the basis of the comparison briefly and mention counts "
            "when available. For site-list questions, give a short heading that states the applied "
            "latitude/longitude filter, then list each site compactly as "
            "'CODE — Name (lat, lon, elev m)'. Keep answers concise and readable."
        )

    @staticmethod
    def _user_message(prompt: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": prompt,
                }
            ],
        }

    @staticmethod
    def _extract_function_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
        calls = []
        for item in response.get("output", []):
            if item.get("type") != "function_call":
                continue
            raw_args = item.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Model returned invalid JSON arguments for tool {item.get('name')}: {raw_args}"
                ) from exc
            calls.append(
                {
                    "name": item["name"],
                    "arguments": args,
                    "call_id": item.get("call_id") or item.get("id"),
                }
            )
        return calls

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        texts = []
        if response.get("output_text"):
            texts.append(str(response["output_text"]))
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if text:
                        texts.append(str(text))
        return "\n".join(t for t in texts if t).strip()

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        tool_map = {
            "site_info": lambda a: self.tools.get_site_info(a["site_query"]),
            "list_sites": lambda a: self.tools.list_sites(
                min_lat=a.get("min_lat"),
                max_lat=a.get("max_lat"),
                min_lon=a.get("min_lon"),
                max_lon=a.get("max_lon"),
                country=a.get("country"),
                continent=a.get("continent"),
                limit=int(a.get("limit", 500)),
            ),
            "site_countries": lambda a: self.tools.list_site_countries(),
            "supported_analytes": lambda a: self.tools.list_supported_analytes(),
            "resolve_analyte": lambda a: self.tools.resolve_analyte(a["analyte"]),
            "recent_pairs": lambda a: self.tools.get_recent_flask_pairs(
                a["site_code"], limit=int(a.get("limit", 10))
            ),
            "recent_processed_pairs": lambda a: self.tools.get_recent_processed_flask_pairs(
                a["site_code"], limit=int(a.get("limit", 10))
            ),
            "pair_metadata": lambda a: self.tools.get_pair_metadata(int(a["pair_id_num"])),
            "recent_pairs_met": lambda a: self.tools.get_recent_flask_pairs_with_metadata(
                a["site_code"], limit=int(a.get("limit", 10))
            ),
            "recent_processed_pairs_met": lambda a: self.tools.get_recent_processed_flask_pairs_with_metadata(
                a["site_code"], limit=int(a.get("limit", 10))
            ),
            "window_mean": lambda a: self.tools.get_site_flask_mean(
                a["site_code"], a["analyte"], a["start_date"], a["end_date"]
            ),
            "recent_values": lambda a: self.tools.get_recent_flask_values(
                a["site_code"], a["analyte"], limit=int(a.get("limit", 10))
            ),
            "plot_site_timeseries": lambda a: self.tools.plot_site_timeseries(
                a["site_code"],
                a["analyte"],
                a["start_date"],
                end_date=a.get("end_date"),
                aggregation=a.get("aggregation", "raw"),
                output_format=a.get("output_format", "png"),
                overlay_recent_pairs_limit=a.get("overlay_recent_pairs_limit"),
            ),
            "compare_year": lambda a: self.tools.compare_site_year_to_recent(
                a["site_code"],
                a["analyte"],
                int(a["year"]),
                recent_limit=int(a.get("recent_limit", 10)),
            ),
        }
        if tool_name not in tool_map:
            raise RuntimeError(f"Unknown tool requested by model: {tool_name}")
        return tool_map[tool_name](arguments)

    @staticmethod
    def _tool_specs() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "site_info",
                "description": "Return site description including code, name, lat, lon, and elevation. Resolve by site code or site name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_query": {"type": "string", "description": "Site code like SMO or site name like Harvard Forest."}
                    },
                    "required": ["site_query"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "list_sites",
                "description": "List sites with optional latitude, longitude, country, and continent filtering. North/south describe latitude; east/west describe longitude.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_lat": {"type": "number", "description": "Minimum latitude in degrees."},
                        "max_lat": {"type": "number", "description": "Maximum latitude in degrees."},
                        "min_lon": {"type": "number", "description": "Minimum longitude in degrees."},
                        "max_lon": {"type": "number", "description": "Maximum longitude in degrees."},
                        "country": {"type": "string", "description": "Country name as stored in gmd.site.country."},
                        "continent": {"type": "string", "description": "Continent name, for example Africa."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "site_countries",
                "description": "List distinct gmd.site.country values and their continent association.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "supported_analytes",
                "description": "List analytes available for the current instrument.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "resolve_analyte",
                "description": "Resolve a user analyte name like CFC-11 to a parameter number and channel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analyte": {"type": "string", "description": "Analyte name such as CFC-11."}
                    },
                    "required": ["analyte"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "recent_pairs",
                "description": "Return recent sampled flask pairs for a site, ranked by Status_MetData.sample_datetime_utc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["site_code"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "recent_processed_pairs",
                "description": "Return recent processed or analyzed flask pairs for a site, ranked by latest run_time in ng_data_processing_view.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["site_code"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "pair_metadata",
                "description": "Return flask type and meteorology for a pair ID using hats.Status_MetData.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pair_id_num": {"type": "integer"}
                    },
                    "required": ["pair_id_num"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "recent_pairs_met",
                "description": "Return recent sampled flask pairs for a site enriched with flask type and met data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["site_code"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "recent_processed_pairs_met",
                "description": "Return recent processed or analyzed flask pairs for a site enriched with flask type and met data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["site_code"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "window_mean",
                "description": "Return flask summary statistics for a site, analyte, and date window.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "analyte": {"type": "string"},
                        "start_date": {"type": "string", "description": "Inclusive YYYY-MM-DD date."},
                        "end_date": {"type": "string", "description": "Exclusive YYYY-MM-DD date."},
                    },
                    "required": ["site_code", "analyte", "start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "recent_values",
                "description": "Return recent good flask values for a site and analyte.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "analyte": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["site_code", "analyte"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "compare_year",
                "description": "Compare a yearly site mean to recent flask values for an analyte.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "analyte": {"type": "string"},
                        "year": {"type": "integer"},
                        "recent_limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["site_code", "analyte", "year"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "plot_site_timeseries",
                "description": "Create a local PNG plot for site/analyte flask data over a date window. Use aggregation='monthly_mean' for monthly means.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "site_code": {"type": "string"},
                        "analyte": {"type": "string"},
                        "start_date": {"type": "string", "description": "Inclusive YYYY-MM-DD date."},
                        "end_date": {"type": "string", "description": "Exclusive YYYY-MM-DD date. Omit for now/current."},
                        "aggregation": {"type": "string", "enum": ["raw", "monthly_mean"]},
                        "output_format": {"type": "string", "enum": ["png"]},
                        "overlay_recent_pairs_limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["site_code", "analyte", "start_date"],
                    "additionalProperties": False,
                },
            },
        ]
