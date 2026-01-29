# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 00:04:18 2025

@author: yurt3
"""
# pages/Mapping_service.py

import os
import json
import hashlib
from io import BytesIO
from typing import List, Dict, Any, Optional
import uuid
import requests
from datetime import datetime
try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    from langchain.callbacks.base import BaseCallbackHandler

import pandas as pd
import streamlit as st

from openrouter_models import openrouter_model_supports_tools, check_tools_capability_openrouter
from llm_factory import build_chat_llm
from config_utils import load_config, save_config, extract_structured_payload
from wikidata_agent_and_tools.deep_agent_wikidata import get_agent_wiki
from bioportal_agent_and_tools.deep_agent_bioportal import get_agent_bioportal
from bioportal_wikidata_system.multiagent_system import get_multiagent  # NEW
from pricing_registry import get_pricing, estimate_cost_usd, GeminiCostCallback


# ============================================================
# Page setup & guards
# ============================================================
st.set_page_config(page_title="Mapping service", layout="wide")
st.title("Mapping service")

# Persistent Cost Display in Sidebar
with st.sidebar:
    st.header("Usage & Costs")
    
    # Show active pricing for detected model
    provider = os.environ.get("LLM_PROVIDER", "openai")
    model = os.environ.get("LLM_MODEL", "")
    if model:
        st.subheader("Active Pricing")
        pricing = get_pricing(provider, model)
        if pricing:
            st.caption(f"Model: {model}")
            st.write(f"In: `${pricing.input_per_million:.2f}` / 1M")
            st.write(f"Out: `${pricing.output_per_million:.2f}` / 1M")
            st.caption(f"Last updated: {pricing.last_updated[:10]}")
        else:
            st.warning(f"No pricing found for {model}")

    # Single run cost
    if st.session_state.get("last_mapping_cost"):
        c = st.session_state["last_mapping_cost"]
        st.subheader("Last Mapping")
        st.metric("Cost", f"${c['cost']:.6f}")
        st.caption(f"Tokens: {c['prompt_tokens']} in / {c['completion_tokens']} out")
        if st.button("Clear Single Cost"):
            del st.session_state["last_mapping_cost"]
            st.rerun()

    # Batch run cost
    if st.session_state.get("last_batch_cost"):
        bc = st.session_state["last_batch_cost"]
        st.subheader("Last Batch")
        st.metric("Total Cost", f"${bc['total_cost']:.4f}")
        st.metric("Avg / Term", f"${bc['avg_cost']:.6f}")
        st.caption(f"Tokens: {bc['total_prompt_tokens']} in / {bc['total_completion_tokens']} out")
        if bc.get("is_sample"):
            st.warning(f"Projected Total: ${bc.get('projected_total_cost', 0):.2f}")
        if st.button("Clear Batch Cost"):
            del st.session_state["last_batch_cost"]
            st.rerun()
            
    # Rolling average
    cost_history = st.session_state.get("cost_history", [])
    if cost_history:
        avg = sum(cost_history[-10:]) / max(1, len(cost_history[-10:]))
        st.subheader("Statistics")
        st.metric("Rolling Avg (last 10)", f"${avg:.6f}")

# OpenRouter Tool Capability Check
provider = os.environ.get("LLM_PROVIDER", "openai")
model = os.environ.get("LLM_MODEL", "")
if provider == "openrouter":
    if not openrouter_model_supports_tools(model):
        st.error(
            f"The selected OpenRouter model '{model}' does not support tool calling, "
            "but this Mapping service requires a tool-using agent. "
            "Please go back to Home and pick a model that supports tools (e.g. gpt-4o, or a non-free model)."
        )
        if st.button("← Back to Home"):
            st.switch_page("Home.py")
        st.stop()

if not os.environ.get("LLM_API_KEY"):
    st.error("LLM API key is not set. Please go back to Home and enter it.")
    if st.button("← Back to Home"):
        st.switch_page("Home.py")
    st.stop()

st.info("Tool-based mapping runs with streaming disabled (provider limitation).")
if provider in ("openai_compatible", "openwebui"):
    st.write("Open WebUI quick presets (optional):")
    col1, col2 = st.columns(2)
    if col1.button("Use Open WebUI → Ollama (/ollama/v1)"):
        st.session_state["LLM_BASE_URL"] = "http://172.27.38.10:8081/ollama/v1"
        os.environ["LLM_BASE_URL"] = st.session_state["LLM_BASE_URL"]
    if col2.button("Use Open WebUI → OpenAI proxy (/openai)"):
        st.session_state["LLM_BASE_URL"] = "http://172.27.38.10:8081/openai"
        os.environ["LLM_BASE_URL"] = st.session_state["LLM_BASE_URL"]

if st.button("← Back to Home"):
    st.switch_page("Home.py")

st.divider()

# ============================================================
# Config & Session-state defaults
# ============================================================
cfg = load_config()
bio_cfg = cfg.get("bioportal", {})

defaults = {
    "mapping_term_input": "",
    "mapping_definition_input": "",
    "mapping_multi_input": False,
    "mapping_endpoints_input": [],

    # BioPortal inputs from config
    "bioportal_api_key_input": bio_cfg.get("api_key", ""),
    "trusted_ontologies_input": ",".join(bio_cfg.get("trusted_ontologies", ["MESH", "NCIT", "LOINC", "FOODON"])),
    "term_ontologies_input": ",".join(bio_cfg.get("term_ontologies", ["NCIT", "NIFSTD", "SNOMEDCT"])),

    # Single-term output
    "mapping_iri_out": "",
    "mapping_skos_out": "",
    "mapping_expl_out": "",

    # Batch output
    "mapping_batch_df": None,

    # Highlight only last re-evaluation
    "last_reeval_run_id": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ============================================================
# Helpers
# ============================================================

class ToolTraceHandler(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_tool_start(self, serialized, input_str=None, **kwargs):
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or "unknown_tool"
        self.events.append({
            "event": "tool_start",
            "tool": name,
            "input": input_str if input_str is not None else kwargs.get("input"),
        })

    def on_tool_end(self, output, **kwargs):
        out = output
        try:
            if isinstance(output, (dict, list)):
                out = json.dumps(output)[:2000]
            else:
                out = str(output)[:2000]
        except Exception:
            out = str(output)[:2000]

        self.events.append({
            "event": "tool_end",
            "output": out,
        })

    def on_llm_end(self, response, **kwargs):
        try:
            self.events.append({"event": "llm_end", "response": str(response)[:2000]})
        except Exception:
            pass


def _build_llm_for_probe():
    # IMPORTANT: streaming must be False
    return build_chat_llm(streaming=False)


def _parse_csv_list(text: str) -> List[str]:
    items = [x.strip() for x in (text or "").split(",") if x.strip()]
    # stable de-dupe
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_agent_json(raw_output: Any, tool_trace: Optional[list] = None) -> Dict[str, Any]:
    """
    Robust parsing:
    1) dict/BaseModel
    2) JSON in content
    3) Gemini/OpenAI function_call arguments (from message or from tool_trace)
    """
    # 1) Already structured
    if isinstance(raw_output, dict):
        return raw_output

    # Pydantic v2
    if hasattr(raw_output, "model_dump"):
        return raw_output.model_dump()

    # Pydantic v1
    if hasattr(raw_output, "dict"):
        return raw_output.dict()

    # 2) JSON in plain text content
    content = None
    if hasattr(raw_output, "content"):
        content = raw_output.content
    elif isinstance(raw_output, str):
        content = raw_output

    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            return json.loads(content)
        except Exception:
            # Try to extract {...} block if it's not pure JSON
            raw = content
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1 and e > s:
                try:
                    return json.loads(raw[s : e + 1])
                except Exception:
                    pass

    # 3a) Function call directly on message (Gemini)
    if hasattr(raw_output, "additional_kwargs"):
        fc = raw_output.additional_kwargs.get("function_call")
        if fc and isinstance(fc, dict) and "arguments" in fc:
            try:
                return json.loads(fc["arguments"])
            except Exception:
                pass

    # 3b) Function call from tool_trace (fallback; last call wins)
    if tool_trace:
        for e in reversed(tool_trace):
            # depending on how you store trace, adapt keys here
            # Trace event format: {"event": "llm_end", "response": str(response)}
            pass

    return {}


def _effective_bioportal_ontologies(term_onts: List[str], trusted_onts: List[str]) -> List[str]:
    """BioPortal should search both term_ontologies and trusted_ontologies (union)."""
    # stable union, keep order preference: term_onts first then trusted_onts not already present
    out = []
    seen = set()
    for x in (term_onts or []):
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    for x in (trusted_onts or []):
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _bioportal_preflight_search(
    term: str,
    ontologies: List[str],
    api_key: str,
    pagesize: int = 5,
) -> Dict[str, Any]:
    """
    Deterministic BioPortal call (no agents). Returns status + top hits.
    Uses BioPortal REST /search endpoint.
    """
    base_url = "https://data.bioontology.org"
    url = f"{base_url}/search"

    q = (term or "").strip()
    onts = [o.strip() for o in (ontologies or []) if o.strip()]
    key = (api_key or "").strip()

    if not q:
        return {"ok": False, "error": "Empty term", "status": None, "hits": []}
    if not key:
        return {"ok": False, "error": "Missing BIOPORTAL_API_KEY", "status": None, "hits": []}
    if not onts:
        return {"ok": False, "error": "No ontologies provided", "status": None, "hits": []}

    params = {
        "q": q,
        "ontologies": ",".join(onts),
        "pagesize": int(pagesize),
        # ask for fields that help debugging (may be ignored if not supported)
        "include": "prefLabel,synonym,definition",
        # some setups accept apikey as query param; keep it as a fallback
        "apikey": key,
    }

    headers = {
        # BioPortal commonly supports this header form
        "Authorization": f"apikey token={key}",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        status = resp.status_code
        text = resp.text or ""
        if status != 200:
            return {
                "ok": False,
                "status": status,
                "error": f"HTTP {status}: {text[:400]}",
                "hits": [],
            }

        data = resp.json() if resp.content else {}
        collection = data.get("collection") or data.get("results") or []

        hits = []
        for item in collection[:pagesize]:
            iri = item.get("@id") or item.get("id") or ""
            pref = item.get("prefLabel") or item.get("label") or ""
            onto = item.get("ontology") or item.get("links", {}).get("ontology") or ""
            # ontology can be a URL; keep as is
            defin = item.get("definition")
            if isinstance(defin, list):
                defin = defin[0] if defin else ""
            hits.append({"iri": iri, "prefLabel": pref, "ontology": onto, "definition": defin})

        return {"ok": True, "status": status, "hits": hits, "raw_count": len(collection)}
    except Exception as e:
        return {"ok": False, "status": None, "error": f"{type(e).__name__}: {e}", "hits": []}


def _wikidata_url(qid: str) -> str:
    return f"https://www.wikidata.org/wiki/{qid}"


def _qid_to_url_if_needed(identifier: str) -> str:
    """If identifier is a bare Wikidata QID (Q123), convert to Wikidata URL."""
    ident = (identifier or "").strip()
    if ident.startswith("Q") and ident[1:].isdigit():
        return _wikidata_url(ident)
    return ident


def _extract_multiagent_fields(parsed: Dict[str, Any]) -> Dict[str, str]:
    """
    Multiagent final formatting tool returns:
      {"ID": "...", "SKOS": "...", "SKOS_explanation": "..."}
    But we also accept fallback keys to be robust.
    """
    ident = (
        parsed.get("ID")
        or parsed.get("id")
        or parsed.get("qid")
        or parsed.get("IRI")
        or parsed.get("iri")
        or ""
    )
    skos = (
        parsed.get("SKOS")
        or parsed.get("skos")
        or ""
    )
    expl = (
        parsed.get("SKOS_explanation")
        or parsed.get("skos_explanation")
        or parsed.get("explanation")
        or ""
    )
    ident = _qid_to_url_if_needed(str(ident))
    return {
        "iri": str(ident).strip(),
        "skos": str(skos).strip(),
        "explanation": str(expl).strip(),
    }


def _question_wikidata(term: str, definition: str) -> str:
    return f"""What is the best fitting Q-identifier the term {term} which definition {definition} matches with the label from the wikidata .
As a reply only provide
(i) identificator number,
(ii) SKOS matching between original term definition and the definition/description of the identified label from wikidata
(iii) Explanation for SKOS matching.

SKOS matching and corresponding explanation should be identified with the corresponding tool -

If no proper match is found, you may adjust the search query and try with other identifiers.

If no proper identifier is found after 10 iterations, return "No wiki match". In that case SKOS matching is not needed
"""


def _question_bioportal(term: str, definition: str, term_onts: List[str], trusted_onts: List[str]) -> str:
    search_onts = _effective_bioportal_ontologies(term_onts, trusted_onts)

    return f"""
You must find the best BioPortal class IRI for the following:

TERM: {term}
DEFINITION: {definition}

Search ONLY these BioPortal ontologies (acronyms):
{", ".join(search_onts)}

Return JSON ONLY (no extra text) in exactly this schema:
{{
  "ID": "<IRI or 'No bioportal match'>",
  "SKOS": "exact|close|related|none",
  "SKOS_explanation": "<short reason>"
}}
""".strip()


def _question_multiagent(term: str, definition: str) -> str:
    return f"""Map the term "{term}" with definition "{definition}" to a valid identifier from BioPortal or Wikidata.
Return only the final JSON output.
"""

def _check_openai_compatible_backend(base_url: str, api_key: str) -> dict:
    """
    Quick connectivity + auth test for OpenAI-compatible backends.
    """
    base_url = (base_url or "").rstrip("/")
    url = f"{base_url}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        out = {"ok": r.status_code == 200, "status": r.status_code}
        try:
            out["json"] = r.json()
        except Exception:
            out["text_head"] = (r.text or "")[:300]
        return out
    except Exception as e:
        return {"ok": False, "status": None, "error": f"{type(e).__name__}: {e}"}


def _ensure_batch_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure RowID + OriginalTerm + last_updated_run columns exist."""
    out = df.copy()

    if "RowID" not in out.columns:
        out.insert(0, "RowID", [str(uuid.uuid4()) for _ in range(len(out))])

    if "OriginalTerm" not in out.columns:
        out["OriginalTerm"] = out["Term"]

    if "last_updated_run" not in out.columns:
        out["last_updated_run"] = ""

    return out


def extract_token_usage(result) -> tuple[int, int]:
    """
    Returns (prompt_tokens, completion_tokens). Falls back to (0,0) if unavailable.
    """
    try:
        msg = result["messages"][-1] if isinstance(result, dict) and "messages" in result else result
        meta = getattr(msg, "response_metadata", {}) or {}
        usage = meta.get("token_usage") or {}
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    except Exception:
        return 0, 0


def _extract_mapping_from_agent_result(result, tool_trace: Optional[list] = None) -> dict:
    """
    Returns {"qid": str, "skos": str, "explanation": str} or {} if not found.
    Uses extract_structured_payload for robustness.
    """
    return extract_structured_payload(result)


def _sum_usage_from_trace(tool_trace: list[dict]) -> tuple[int, int]:
    in_tok = 0
    out_tok = 0
    for ev in tool_trace or []:
        if ev.get("event") != "llm_end":
            continue
        # Adapt to langchain response metadata or usage_metadata
        resp = ev.get("response")
        if isinstance(resp, str):
            # Best effort to extract tokens if they were serialized into the trace string
            # In our ToolTraceHandler, response is str(response)
            pass 
        # Note: ToolTraceHandler should ideally store structured metadata if we want easy summation here.
    return 0, 0 # Fallback, we'll use the result usage instead if possible


# ============================================================
# Agent caches
# - multiagent cache includes ontology lists so it rebuilds on change
# ============================================================
@st.cache_resource
def _cached_wiki_agent(provider: str, model: str, base_url: str, api_key_hash: str, langsmith_key: str):
    return get_agent_wiki()


@st.cache_resource
def _cached_bio_agent(
    provider: str,
    model: str,
    base_url: str,
    api_key_hash: str,
    langsmith_key: str,
    bioportal_key: str,
    trusted_ontologies: tuple,
    term_ontologies: tuple,
):
    return get_agent_bioportal(
        trusted_ontologies=list(trusted_ontologies),
        term_ontologies=list(term_ontologies),
    )


@st.cache_resource
def _cached_multi_agent(
    provider: str,
    model: str,
    base_url: str,
    api_key_hash: str,
    langsmith_key: str,
    bioportal_key: str,
    trusted_ontologies: tuple,
    term_ontologies: tuple,
):
    return get_multiagent(
        trusted_ontologies=list(trusted_ontologies),
        term_ontologies=list(term_ontologies),
    )


def _get_llm_cache_keys():
    provider = os.environ.get("LLM_PROVIDER", "openai")
    model = os.environ.get("LLM_MODEL", "gpt-4o")
    base_url = os.environ.get("LLM_BASE_URL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    api_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    return provider, model, base_url, api_key_hash


def _get_wiki_agent():
    provider, model, base_url, api_key_hash = _get_llm_cache_keys()
    return _cached_wiki_agent(
        provider, model, base_url, api_key_hash,
        os.environ.get("LANGSMITH_API_KEY", ""),
    )


def _get_bio_agent(trusted_onts: List[str], term_onts: List[str]):
    provider, model, base_url, api_key_hash = _get_llm_cache_keys()
    return _cached_bio_agent(
        provider, model, base_url, api_key_hash,
        os.environ.get("LANGSMITH_API_KEY", ""),
        os.environ.get("BIOPORTAL_API_KEY", ""),
        tuple(trusted_onts),
        tuple(term_onts),
    )


def _get_multi_agent(trusted_onts: List[str], term_onts: List[str]):
    provider, model, base_url, api_key_hash = _get_llm_cache_keys()
    return _cached_multi_agent(
        provider, model, base_url, api_key_hash,
        os.environ.get("LANGSMITH_API_KEY", ""),
        os.environ.get("BIOPORTAL_API_KEY", ""),
        tuple(trusted_onts),
        tuple(term_onts),
    )


# ============================================================
# Inputs
# ============================================================
searched_term = st.text_input("Searched Term", key="mapping_term_input")
multiple_terms = st.checkbox("Multiple terms", key="mapping_multi_input")

uploaded_file = None
term_definition = ""

if not multiple_terms:
    term_definition = st.text_area("Term definition", key="mapping_definition_input")
else:
    uploaded_file = st.file_uploader(
        "Upload Excel file (must contain columns: Term, Definition)",
        type=["xlsx"],
        key="mapping_upload_excel",
    )

st.divider()

endpoints = st.multiselect(
    "Endpoints",
    ["Wikidata", "Bioportal"],
    key="mapping_endpoints_input",
)

# --- Endpoint selection with Multiagent override when BOTH selected ---
endpoint_to_run = None
if len(endpoints) == 1:
    endpoint_to_run = endpoints[0]
elif set(endpoints) == {"Wikidata", "Bioportal"}:
    endpoint_to_run = "Multiagent"
    st.info("Both endpoints selected → using the Multiagent system (BioPortal → Wikidata).")
elif len(endpoints) > 1:
    endpoint_to_run = st.radio("Run mapping against", endpoints, horizontal=True)
else:
    endpoint_to_run = None

trusted_ontologies: List[str] = []
term_ontologies: List[str] = []


# ============================================================
# BioPortal configuration (Bioportal OR Multiagent)
# ============================================================
if endpoint_to_run in {"Bioportal", "Multiagent"}:
    st.subheader("BioPortal configuration")

    if st.button("Load BioPortal Settings from config.yaml"):
        cfg = load_config()
        bio_cfg = cfg.get("bioportal", {})
        if bio_cfg:
            st.session_state["bioportal_api_key_input"] = bio_cfg.get("api_key", "")
            st.session_state["trusted_ontologies_input"] = ",".join(bio_cfg.get("trusted_ontologies", []))
            st.session_state["term_ontologies_input"] = ",".join(bio_cfg.get("term_ontologies", []))
            st.success("BioPortal settings loaded from config.yaml.")
            st.rerun()
        else:
            st.warning("No BioPortal settings found in config.yaml.")

    bio_key = st.text_input(
        "BIOPORTAL_API_KEY",
        type="password",
        key="bioportal_api_key_input",
        help="Use 'Load BioPortal Settings from config.yaml' to prefill from your configuration.",
    )
    os.environ["BIOPORTAL_API_KEY"] = (bio_key or "").strip()

    trusted_text = st.text_area(
        "trusted_ontologies (comma-separated)",
        key="trusted_ontologies_input",
    )
    term_text = st.text_area(
        "term_ontologies (comma-separated)",
        key="term_ontologies_input",
    )
    trusted_ontologies = _parse_csv_list(trusted_text)
    term_ontologies = _parse_csv_list(term_text)

    os.environ["BIOPORTAL_TRUSTED_ONTOLOGIES"] = trusted_text or ""
    os.environ["BIOPORTAL_TERM_ONTOLOGIES"] = term_text or ""

# -------------------------
# Tool calling capability (OpenRouter / Compatible specific)
# -------------------------
provider_clean = os.environ.get("LLM_PROVIDER", "").strip().lower()
report = {}
if provider_clean in ("openrouter", "openai_compatible"):
    st.subheader("Tool-calling capability check")

    if st.button("Run tool-calling self-test for selected model"):
        model_id = os.environ.get("LLM_MODEL", "")
        api_key = os.environ.get("LLM_API_KEY", "")

        if provider_clean == "openrouter":
            with st.spinner("Probing tool-calling capability (OpenRouter)..."):
                report = check_tools_capability_openrouter(
                    model_id=model_id,
                    api_key=api_key,
                    model_builder=_build_llm_for_probe,
                )
        else:
            with st.spinner("Probing tool-calling capability (Live probe only)..."):
                from openrouter_models import probe_tool_call_via_langchain
                probe_ok, probe_reason, probe_debug = probe_tool_call_via_langchain(_build_llm_for_probe)
                report = {
                    "metadata_ok": True,
                    "metadata_reason": "Skipped for generic compatible provider",
                    "probe_ok": probe_ok,
                    "probe_reason": probe_reason,
                    "probe_debug": probe_debug,
                    "final_ok": probe_ok,
                    "final_reason": probe_reason
                }
            st.session_state["tools_capability_report"] = report
            if provider_clean == "openai_compatible":
                # Backend connectivity test
                base_url = os.environ.get("LLM_BASE_URL", "")
                api_key = os.environ.get("LLM_API_KEY", "")
                if st.button("Test backend connectivity (/models)"):
                    st.session_state["backend_check"] = _check_openai_compatible_backend(base_url, api_key)
                bc = st.session_state.get("backend_check")
                if bc:
                    if bc.get("ok"):
                        st.success(f"Backend OK (HTTP {bc.get('status')})")
                    else:
                        st.error(f"Backend check failed: {bc}")
                    with st.expander("Backend raw response"):
                        st.json(bc)

        report = st.session_state.get("tools_capability_report")
if report:
    if provider_clean == "openrouter":
        st.write("Metadata check:", "✅" if report["metadata_ok"] else "❌")
        st.caption(report["metadata_reason"])
    
    st.write("Live probe:", "✅" if report["probe_ok"] else "❌")
    st.caption(report["probe_reason"])

    # Show detailed debug information for the probe
    with st.expander("Probe debug details (raw AIMessage fields)"):
        st.json(report.get("probe_debug", {}))

    if not report["final_ok"]:
        st.error("Selected model/route does NOT support tool calling reliably. Choose another model.")

# -------------------------
# Debug / Diagnostics UI
# -------------------------
debug_mode = st.checkbox("Debug mode (show raw agent output + BioPortal preflight)", value=False)

# Tell the user what will actually be searched
effective_onts = _effective_bioportal_ontologies(term_ontologies, trusted_ontologies)

if debug_mode:
    with st.expander("Debug: Effective BioPortal settings", expanded=True):
        st.write("LLM_PROVIDER:", os.environ.get("LLM_PROVIDER", ""))
        st.write("LLM_MODEL:", os.environ.get("LLM_MODEL", ""))
        st.write("BIOPORTAL_API_KEY set:", bool(os.environ.get("BIOPORTAL_API_KEY")))
        st.write("term_ontologies:", term_ontologies)
        st.write("trusted_ontologies:", trusted_ontologies)
        st.write("effective search ontologies (union):", effective_onts)

    # Deterministic preflight (no agent)
    st.markdown("**BioPortal preflight (no agent):**")
    preflight_term = st.text_input("Preflight term", value=st.session_state.get("mapping_term_input", ""))
    if st.button("Run BioPortal preflight search"):
        pf = _bioportal_preflight_search(
            term=preflight_term,
            ontologies=effective_onts,
            api_key=os.environ.get("BIOPORTAL_API_KEY", ""),
            pagesize=5,
        )
        st.session_state["debug_last_preflight"] = pf

    pf = st.session_state.get("debug_last_preflight")
    if pf:
        st.write("Preflight OK:", pf.get("ok"))
        st.write("HTTP status:", pf.get("status"))
        if not pf.get("ok"):
            st.error(pf.get("error", "Unknown error"))
        else:
            st.write("Raw result count:", pf.get("raw_count"))
            st.json(pf.get("hits", []))

st.divider()

# Verify the model being used (Debug)
debug_provider = os.environ.get("LLM_PROVIDER")
debug_model = os.environ.get("LLM_MODEL")
st.caption(f"Backend configured with: Provider={debug_provider}, Model={debug_model}")

# ============================================================
# Single-term mapping
# ============================================================
st.subheader("Mapping (single term)")

run_single_enabled = (not multiple_terms) and (endpoint_to_run in {"Wikidata", "Bioportal", "Multiagent"})

if st.button("Run mapping", disabled=not run_single_enabled, use_container_width=True):
    if not searched_term.strip():
        st.error("Provide a term.")
        st.stop()
    if not term_definition.strip():
        st.error("Provide a definition.")
        st.stop()

    # Hard-gate OpenRouter/Compatible on capability check
    if os.environ.get("LLM_PROVIDER", "").strip().lower() in ("openrouter", "openai_compatible"):
        report = st.session_state.get("tools_capability_report")
        if not report or not report.get("final_ok"):
            st.error("Tool calling capability not verified / failed. Run the self-test above or choose a different model.")
            st.stop()
        if os.environ.get("LLM_PROVIDER", "").strip().lower() == "openai_compatible":
            bc = st.session_state.get("backend_check")
            if not bc or not bc.get("ok"):
                st.error("Backend connectivity not verified. Run the backend test.")
                st.stop()

    cost_cb = GeminiCostCallback(provider, model)
    
    if endpoint_to_run == "Wikidata":
        agent = _get_wiki_agent()
        question = _question_wikidata(searched_term.strip(), term_definition.strip())
        with st.spinner("Running Wikidata agent..."):
            trace = ToolTraceHandler()
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"callbacks": [trace, cost_cb]}
            )
            st.session_state["debug_tool_trace"] = trace.events
        
        raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
        payload = extract_structured_payload(result)
        
        # Debugging info
        debug_info = {
            "has_output": result.get("output") is not None,
            "output_type": type(result.get("output")).__name__,
            "messages_count": len(result.get("messages", []) or []),
            "last_message_content_type": type(getattr((result.get("messages") or [])[-1], "content", None)).__name__
                if result.get("messages") else None,
            "extracted_payload": payload,
        }
        st.session_state["last_debug_info"] = debug_info

        # ✅ Extract fields robustly (Normalization)
        iri = (payload.get("iri") or payload.get("qid") or payload.get("ID") or payload.get("id") or "").strip()
        skos = (payload.get("skos") or payload.get("SKOS") or payload.get("mapping_type") or "").strip()
        expl = (payload.get("explanation") or payload.get("SKOS_explanation") or payload.get("skos_explanation") or "").strip()

        if iri and iri != "No wiki match":
            iri = _qid_to_url_if_needed(iri)
        else:
            iri = "No wiki match"
            skos, expl = "" , ""

    elif endpoint_to_run == "Bioportal":
        if not os.environ.get("BIOPORTAL_API_KEY"):
            st.error("BIOPORTAL_API_KEY is required for BioPortal.")
            st.stop()
        if not term_ontologies:
            st.error("Please provide term_ontologies for BioPortal search.")
            st.stop()

        agent = _get_bio_agent(trusted_ontologies, term_ontologies)
        question = _question_bioportal(searched_term.strip(), term_definition.strip(), term_ontologies, trusted_ontologies)
        with st.spinner("Running BioPortal agent..."):
            trace = ToolTraceHandler()
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"callbacks": [trace, cost_cb]}
            )
            st.session_state["debug_tool_trace"] = trace.events

        raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
        payload = extract_structured_payload(result, schema_name="Bioportalmapping")

        # Debugging info
        debug_info = {
            "has_output": result.get("output") is not None,
            "output_type": type(result.get("output")).__name__,
            "messages_count": len(result.get("messages", []) or []),
            "last_message_content_type": type(getattr((result.get("messages") or [])[-1], "content", None)).__name__
                if result.get("messages") else None,
            "extracted_payload": payload,
        }
        st.session_state["last_debug_info"] = debug_info

        # Enforce tool usage check
        events = st.session_state.get("debug_tool_trace", [])
        if not any(e.get("event") == "tool_start" for e in events):
            st.error("Agent produced an answer without calling any BioPortal tool. Mapping aborted.")
            st.stop()

        # ✅ Robust: accept ID/IRI/QID keys consistently
        iri = (payload.get("iri") or payload.get("qid") or payload.get("ID") or payload.get("id") or "").strip()
        skos = (payload.get("skos") or payload.get("SKOS") or payload.get("mapping_type") or "").strip()
        expl = (payload.get("explanation") or payload.get("SKOS_explanation") or payload.get("skos_explanation") or "").strip()

        if not iri or iri == "No bioportal match" or iri.lower().startswith("no "):
            iri = "No bioportal match"
            skos, expl = "", ""

        st.session_state["debug_last_agent_raw"] = raw
        st.session_state["debug_last_agent_parsed"] = payload

    else:  # Multiagent
        if not os.environ.get("BIOPORTAL_API_KEY"):
            st.error("BIOPORTAL_API_KEY is required for Multiagent.")
            st.stop()
        if not term_ontologies:
            st.error("Please provide term_ontologies for Multiagent.")
            st.stop()

        agent = _get_multi_agent(trusted_ontologies, term_ontologies)
        question = _question_multiagent(searched_term.strip(), term_definition.strip())
        with st.spinner("Running Multiagent system..."):
            trace = ToolTraceHandler()
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"callbacks": [trace, cost_cb]}
            )
            st.session_state["debug_tool_trace"] = trace.events

        raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
        payload = extract_structured_payload(result)

        # Debugging info
        debug_info = {
            "has_output": result.get("output") is not None,
            "output_type": type(result.get("output")).__name__,
            "messages_count": len(result.get("messages", []) or []),
            "last_message_content_type": type(getattr((result.get("messages") or [])[-1], "content", None)).__name__
                if result.get("messages") else None,
            "extracted_payload": payload,
        }
        st.session_state["last_debug_info"] = debug_info

        # Normalization
        iri = (payload.get("iri") or payload.get("qid") or payload.get("ID") or payload.get("id") or "").strip()
        skos = (payload.get("skos") or payload.get("SKOS") or payload.get("mapping_type") or "").strip()
        expl = (payload.get("explanation") or payload.get("SKOS_explanation") or payload.get("skos_explanation") or "").strip()

        if not iri or iri.lower().startswith("no "):
            iri = "No match"
            skos, expl = "", ""
        else:
            iri = _qid_to_url_if_needed(iri)

    # Always capture cost from callback
    term_cost = cost_cb.summary.total_cost_usd
    prompt_toks = cost_cb.summary.input_tokens
    compl_toks = cost_cb.summary.output_tokens
    
    if term_cost > 0 or prompt_toks > 0 or compl_toks > 0:
        st.session_state["last_mapping_cost"] = {
            "cost": term_cost,
            "prompt_tokens": prompt_toks,
            "completion_tokens": compl_toks,
        }
        st.session_state.setdefault("cost_history", []).append(term_cost)

    st.session_state["mapping_iri_out"] = iri
    st.session_state["mapping_skos_out"] = skos
    st.session_state["mapping_expl_out"] = expl
    
    # Debug info
    # Ensure raw and payload are available even if they weren't set in a specific branch
    if "raw" not in locals(): raw = ""
    if "payload" not in locals(): payload = {}
    st.session_state["debug_last_agent_raw"] = raw
    st.session_state["debug_last_agent_parsed"] = payload
    st.rerun()

st.write("**IRI:**", st.session_state.get("mapping_iri_out", "") or "—")
st.write("**SKOS:**", st.session_state.get("mapping_skos_out", "") or "—")
st.write("**Explanation:**")
st.code(st.session_state.get("mapping_expl_out", "") or "—", language="text")

# Show debug info
if st.session_state.get("last_debug_info"):
    with st.expander("Debug: Extraction logic details"):
        st.json(st.session_state["last_debug_info"])

# Show last agent output if debug mode was enabled earlier
if st.session_state.get("debug_last_agent_raw"):
    with st.expander("Debug: Last agent raw output / parsed JSON"):
        st.code(st.session_state.get("debug_last_agent_raw", ""), language="text")
        st.json(st.session_state.get("debug_last_agent_parsed", {}) or {})

if st.session_state.get("debug_tool_trace"):
    with st.expander("Debug: Tool trace (BioPortal/Wikidata)", expanded=False):
        st.json(st.session_state.get("debug_tool_trace", []))

single_payload = {
    "endpoint": endpoint_to_run or "",
    "term": st.session_state.get("mapping_term_input", ""),
    "definition": st.session_state.get("mapping_definition_input", ""),
    "iri": st.session_state.get("mapping_iri_out", ""),
    "skos": st.session_state.get("mapping_skos_out", ""),
    "explanation": st.session_state.get("mapping_expl_out", ""),
}
st.download_button(
    "Download single-term result (.json)",
    data=json.dumps(single_payload, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="mapping_result.json",
    mime="application/json",
    use_container_width=True,
)

# ============================================================
# Batch mapping
# ============================================================
st.subheader("Mapping (multiple terms)")

if multiple_terms and uploaded_file is not None:
    try:
        preview_df = pd.read_excel(uploaded_file)
        st.write("**Uploaded terms preview:**")
        st.dataframe(preview_df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
        st.stop()

run_batch_enabled = multiple_terms and (uploaded_file is not None) and (endpoint_to_run in {"Wikidata", "Bioportal", "Multiagent"})

# --- Sampling & Cost Projection ---
if run_batch_enabled:
    col1, col2 = st.columns([1, 1])
    with col1:
        do_sampling = st.checkbox("Sample & Estimate Cost", value=False, key="mapping_do_sampling", help="Run first 5 terms to estimate total cost.")
    
    if do_sampling:
        pricing = get_pricing(provider=provider, model=model)
        if pricing:
            st.info(f"Using pricing for {model}: ${pricing.input_per_million}/1M in, ${pricing.output_per_million}/1M out")
        else:
            st.warning(f"No pricing data for {model}. Cost estimation will be skipped.")

if st.button("Run batch mapping", disabled=not run_batch_enabled, use_container_width=True):
    # Hard-gate OpenRouter/Compatible on capability check
    if os.environ.get("LLM_PROVIDER", "").strip().lower() in ("openrouter", "openai_compatible"):
        report = st.session_state.get("tools_capability_report")
        if not report or not report.get("final_ok"):
            st.error("Tool calling capability not verified / failed. Run the self-test above or choose a different model.")
            st.stop()
        if os.environ.get("LLM_PROVIDER", "").strip().lower() == "openai_compatible":
            bc = st.session_state.get("backend_check")
            if not bc or not bc.get("ok"):
                st.error("Backend connectivity not verified. Run the backend test.")
                st.stop()

    try:
        input_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
        st.stop()

    required_cols = {"Term", "Definition"}
    missing = required_cols - set(input_df.columns)
    if missing:
        st.error(f"Missing required column(s): {', '.join(sorted(missing))}. Your file must have Term, Definition.")
        st.stop()

    input_df = input_df.copy()
    input_df["Term"] = input_df["Term"].astype(str).str.strip()
    input_df["Definition"] = input_df["Definition"].astype(str).str.strip()
    input_df = input_df[(input_df["Term"] != "") & (input_df["Term"].str.lower() != "nan")]
    total_in_file = len(input_df)

    if total_in_file == 0:
        st.warning("No valid rows found (Term is empty).")
        st.stop()

    if endpoint_to_run in {"Bioportal", "Multiagent"}:
        if not os.environ.get("BIOPORTAL_API_KEY"):
            st.error("BIOPORTAL_API_KEY is required for BioPortal / Multiagent.")
            st.stop()
        if not term_ontologies:
            st.error("Please provide term_ontologies for BioPortal / Multiagent.")
            st.stop()

    if endpoint_to_run == "Wikidata":
        agent = _get_wiki_agent()
    elif endpoint_to_run == "Bioportal":
        agent = _get_bio_agent(trusted_ontologies, term_ontologies)
    else:
        agent = _get_multi_agent(trusted_ontologies, term_ontologies)

    # Determine if we only run a sample
    sample_size = 5
    is_sampling_run = st.session_state.get("mapping_do_sampling", False)
    
    if is_sampling_run:
        input_df = input_df.head(sample_size)
        total = len(input_df)
    else:
        total = total_in_file

    total_prompt_toks = 0
    total_compl_toks = 0
    results_rows = []
    
    progress = st.progress(0)
    status = st.empty()

    with st.spinner(f"Running {endpoint_to_run} agent for uploaded terms..."):
        for i, row in enumerate(input_df.itertuples(index=False), start=1):
            term = getattr(row, "Term")
            definition = getattr(row, "Definition")

            status.write(f"Processing {i}/{total}: **{term}**")
            progress.progress(int((i / total) * 100))

            if endpoint_to_run == "Wikidata":
                question = _question_wikidata(term, definition)
            elif endpoint_to_run == "Bioportal":
                question = _question_bioportal(term, definition, term_ontologies, trusted_ontologies)
            else:
                question = _question_multiagent(term, definition)

            try:
                trace = ToolTraceHandler()
                row_cost_cb = GeminiCostCallback(provider, model)
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": question}]},
                    config={"callbacks": [trace, row_cost_cb]}
                )
                raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
                # We can store last batch row trace if helpful
                st.session_state["debug_tool_trace"] = trace.events
                
                # Update batch totals
                total_prompt_toks += row_cost_cb.summary.input_tokens
                total_compl_toks += row_cost_cb.summary.output_tokens
                current_cost = row_cost_cb.summary.total_cost_usd
                st.session_state.setdefault("cost_history", []).append(current_cost)
            except Exception as e:
                results_rows.append({
                    "Term": term,
                    "Definition": definition,
                    "Endpoint": endpoint_to_run,
                    "IRI": "",
                    "SKOS": "",
                    "explanation": f"ERROR: {e}",
                })
                continue

            payload = extract_structured_payload(result, schema_name="Bioportalmapping" if endpoint_to_run == "Bioportal" else None)

            # Normalization
            iri_raw = (payload.get("iri") or payload.get("qid") or payload.get("ID") or payload.get("id") or "").strip()
            skos = (payload.get("skos") or payload.get("SKOS") or payload.get("mapping_type") or "").strip()
            expl = (payload.get("explanation") or payload.get("SKOS_explanation") or payload.get("skos_explanation") or "").strip()

            if endpoint_to_run == "Wikidata":
                if iri_raw and not iri_raw.lower().startswith("no "):
                    iri = _qid_to_url_if_needed(iri_raw)
                else:
                    iri, skos, expl = "No wiki match", "", ""

            elif endpoint_to_run == "Bioportal":
                # Tool usage enforcement in batch
                events = st.session_state.get("debug_tool_trace", [])
                if not any(e.get("event") == "tool_start" for e in events):
                    iri, skos, expl = "No bioportal match", "", "ERROR: Agent didn't use tools"
                elif not iri_raw or iri_raw.lower().startswith("no "):
                    iri, skos, expl = "No bioportal match", "", ""
                else:
                    iri = iri_raw

                st.session_state["debug_last_agent_raw"] = raw
                st.session_state["debug_last_agent_parsed"] = payload
                st.session_state["debug_last_term"] = term

            else:  # Multiagent
                if not iri_raw or iri_raw.lower().startswith("no "):
                    iri, skos, expl = "No match", "", ""
                else:
                    iri = _qid_to_url_if_needed(iri_raw)

            results_rows.append({
                "Term": term,
                "Definition": definition,
                "Endpoint": endpoint_to_run,
                "IRI": iri,
                "SKOS": skos,
                "explanation": expl,
            })

    df_out = pd.DataFrame(results_rows, columns=["Term", "Definition", "Endpoint", "IRI", "SKOS", "explanation"])
    df_out = _ensure_batch_schema(df_out)

    # Clear highlight (no "last reevaluation" yet)
    st.session_state["last_reeval_run_id"] = None
    st.session_state["mapping_batch_df"] = df_out

    if total_prompt_toks or total_compl_toks:
        total_cost = estimate_cost_usd(provider, model, total_prompt_toks, total_compl_toks)
        avg_cost = total_cost / total if total > 0 else 0
        
        cost_data = {
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "total_prompt_tokens": total_prompt_toks,
            "total_completion_tokens": total_compl_toks,
            "is_sample": is_sampling_run,
            "total_in_file": total_in_file,
        }
        
        if is_sampling_run:
            cost_data["projected_total_cost"] = avg_cost * total_in_file
        
        st.session_state["last_batch_cost"] = cost_data
    st.rerun()


# ============================================================
# Batch results + re-evaluation (highlight last run only)
# ============================================================
batch_df = st.session_state.get("mapping_batch_df")
if isinstance(batch_df, pd.DataFrame) and len(batch_df) > 0:
    batch_df = _ensure_batch_schema(batch_df)
    st.session_state["mapping_batch_df"] = batch_df

    st.subheader("Batch results")

    # Rolling average tracking
    cost_history = st.session_state.get("cost_history", [])
    if cost_history:
        avg = sum(cost_history[-10:]) / max(1, len(cost_history[-10:]))
        st.info(f"Rolling Average Cost: ${avg:.6f} per term (last 10)")

    # Highlight only rows updated in the last re-evaluation run
    last_run_id = st.session_state.get("last_reeval_run_id") or ""

    def _highlight_last_run(row):
        if last_run_id and str(row.get("last_updated_run", "")) == str(last_run_id):
            return ["background-color: #fff59d"] * len(row)
        return [""] * len(row)

    display_cols = ["Term", "IRI", "SKOS", "explanation"]
    # If a term was changed, show OriginalTerm too
    show_original = any(batch_df["OriginalTerm"].astype(str) != batch_df["Term"].astype(str))
    if show_original:
        display_cols = ["OriginalTerm"] + display_cols

    st.dataframe(
        batch_df[display_cols].style.apply(_highlight_last_run, axis=1),
        use_container_width=True,
    )

    st.subheader("Re-evaluate selected terms")

    selected_rowids: List[str] = []
    for _, r in batch_df.iterrows():
        label = f"{r['Term']}"
        if st.checkbox(label, key=f"recheck_{r['RowID']}"):
            selected_rowids.append(r["RowID"])

    if selected_rowids:
        st.markdown("**Optional:** Provide a new term for re-evaluation (leave unchanged to re-run the same term).")
    new_term_by_rowid: Dict[str, str] = {}
    for rowid in selected_rowids:
        current_term = str(batch_df.loc[batch_df["RowID"] == rowid, "Term"].iloc[0])
        new_term_by_rowid[rowid] = st.text_input(
            f"New term for: {current_term}",
            value=current_term,
            key=f"new_term_{rowid}",
        )

    if st.button("Re-evaluate selected", disabled=(len(selected_rowids) == 0), use_container_width=True):
        # Hard-gate OpenRouter/Compatible on capability check
        if os.environ.get("LLM_PROVIDER", "").strip().lower() in ("openrouter", "openai_compatible"):
            report = st.session_state.get("tools_capability_report")
            if not report or not report.get("final_ok"):
                st.error("Tool calling capability not verified / failed. Run the self-test above or choose a different model.")
                st.stop()

        # Validate BioPortal config if needed for any selected rows
        endpoints_needed = set(batch_df.loc[batch_df["RowID"].isin(selected_rowids), "Endpoint"].astype(str).tolist())
        if ("Bioportal" in endpoints_needed) or ("Multiagent" in endpoints_needed):
            if not os.environ.get("BIOPORTAL_API_KEY"):
                st.error("BIOPORTAL_API_KEY is required for BioPortal / Multiagent re-evaluation.")
                st.stop()
            trusted_ontologies = _parse_csv_list(st.session_state.get("trusted_ontologies_input", ""))
            term_ontologies = _parse_csv_list(st.session_state.get("term_ontologies_input", ""))
            if not term_ontologies:
                st.error("Please provide term_ontologies for BioPortal / Multiagent re-evaluation.")
                st.stop()

        run_id = str(uuid.uuid4())
        st.session_state["last_reeval_run_id"] = run_id

        progress = st.progress(0)
        status = st.empty()
        total = len(selected_rowids)

        for i, rowid in enumerate(selected_rowids, start=1):
            row = batch_df.loc[batch_df["RowID"] == rowid].iloc[0]
            endpoint = str(row["Endpoint"])
            definition = str(row["Definition"])
            new_term = (new_term_by_rowid.get(rowid) or "").strip() or str(row["Term"])

            status.write(f"Re-evaluating {i}/{total}: **{new_term}** ({endpoint})")
            progress.progress(int((i / total) * 100))

            if endpoint == "Wikidata":
                agent = _get_wiki_agent()
                question = _question_wikidata(new_term, definition)

            elif endpoint == "Bioportal":
                trusted_ontologies = _parse_csv_list(st.session_state.get("trusted_ontologies_input", ""))
                term_ontologies = _parse_csv_list(st.session_state.get("term_ontologies_input", ""))
                agent = _get_bio_agent(trusted_ontologies, term_ontologies)
                question = _question_bioportal(new_term, definition, term_ontologies, trusted_ontologies)

            else:  # Multiagent
                trusted_ontologies = _parse_csv_list(st.session_state.get("trusted_ontologies_input", ""))
                term_ontologies = _parse_csv_list(st.session_state.get("term_ontologies_input", ""))
                agent = _get_multi_agent(trusted_ontologies, term_ontologies)
                question = _question_multiagent(new_term, definition)

            trace = ToolTraceHandler()
            reeval_cost_cb = GeminiCostCallback(provider, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"callbacks": [trace, reeval_cost_cb]}
            )
            st.session_state["debug_tool_trace"] = trace.events

            payload = extract_structured_payload(result, schema_name="Bioportalmapping" if endpoint == "Bioportal" else None)

            # Normalization
            iri_raw = (payload.get("iri") or payload.get("qid") or payload.get("ID") or payload.get("id") or "").strip()
            skos = (payload.get("skos") or payload.get("SKOS") or payload.get("mapping_type") or "").strip()
            expl = (payload.get("explanation") or payload.get("SKOS_explanation") or payload.get("skos_explanation") or "").strip()

            if endpoint == "Wikidata":
                if iri_raw and not iri_raw.lower().startswith("no "):
                    iri = _qid_to_url_if_needed(iri_raw)
                else:
                    iri, skos, expl = "No wiki match", "", ""

            elif endpoint == "Bioportal":
                # Tool usage enforcement in re-eval
                events = st.session_state.get("debug_tool_trace", [])
                if not any(e.get("event") == "tool_start" for e in events):
                    iri, skos, expl = "No bioportal match", "", "ERROR: Agent didn't use tools"
                elif not iri_raw or iri_raw.lower().startswith("no "):
                    iri, skos, expl = "No bioportal match", "", ""
                else:
                    iri = iri_raw

            else:  # Multiagent
                if not iri_raw or iri_raw.lower().startswith("no "):
                    iri, skos, expl = "No match", "", ""
                else:
                    iri = _qid_to_url_if_needed(iri_raw)

            # Capture cost for re-evaluation
            re_cost = reeval_cost_cb.summary.total_cost_usd
            st.session_state.setdefault("cost_history", []).append(re_cost)
            st.session_state["last_mapping_cost"] = {
                "cost": re_cost,
                "prompt_tokens": reeval_cost_cb.summary.input_tokens,
                "completion_tokens": reeval_cost_cb.summary.output_tokens,
            }

            # Update term (and preserve OriginalTerm)
            idx = batch_df.index[batch_df["RowID"] == rowid][0]
            if not batch_df.loc[idx, "OriginalTerm"]:
                batch_df.loc[idx, "OriginalTerm"] = batch_df.loc[idx, "Term"]
            if batch_df.loc[idx, "OriginalTerm"] == batch_df.loc[idx, "Term"] and new_term != batch_df.loc[idx, "Term"]:
                batch_df.loc[idx, "OriginalTerm"] = batch_df.loc[idx, "Term"]

            batch_df.loc[idx, "Term"] = new_term
            batch_df.loc[idx, "IRI"] = iri
            batch_df.loc[idx, "SKOS"] = skos
            batch_df.loc[idx, "explanation"] = expl
            batch_df.loc[idx, "last_updated_run"] = run_id

        status.write("✅ Re-evaluation complete. Updated rows are highlighted (only for this last run).")
        st.session_state["mapping_batch_df"] = batch_df
        st.rerun()

    # Batch download (requested 4 columns)
    output = BytesIO()
    export_df = batch_df[["Term", "IRI", "SKOS", "explanation"]].copy()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="mapping")
    output.seek(0)

    st.download_button(
        label="Download batch mapping results (.xlsx)",
        data=output,
        file_name="batch_mapping_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
