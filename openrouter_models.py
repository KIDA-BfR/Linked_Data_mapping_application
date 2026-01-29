import os
import time
import requests
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

def _extract_tool_calls_from_message(msg) -> list:
    """
    Extract tool/function calls from a LangChain AIMessage across multiple schemas.
    """
    calls = []

    # Standard LC AIMessage.tool_calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        return msg.tool_calls

    # Common fallback: additional_kwargs["tool_calls"]
    if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
        ak = msg.additional_kwargs or {}
        if ak.get("tool_calls"):
            return ak["tool_calls"]

        # Legacy function calling schema
        if ak.get("function_call"):
            return [{"type": "function_call", "function_call": ak["function_call"]}]

    return calls

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

def openrouter_model_supports_tools(model_id: str) -> bool:
    """
    Backward compatible wrapper.
    Checks if an OpenRouter model supports tool calling based on metadata.
    """
    ok, reason = model_metadata_says_tools_supported(model_id)
    return ok

def fetch_openrouter_model_supported_parameters(model_id: str, api_key: Optional[str] = None) -> Tuple[bool, List[str], str]:
    """
    Returns (ok, supported_parameters, error_message).
    ok=False if request failed or model not found.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        return False, [], "Empty model_id"

    headers = {"Accept": "application/json"}
    key = (api_key or os.getenv("LLM_API_KEY") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = requests.get(f"{OPENROUTER_API_BASE}/models", headers=headers, timeout=15)
        if resp.status_code != 200:
            return False, [], f"OpenRouter /models HTTP {resp.status_code}: {resp.text[:200]}"
        data = resp.json()
        models = data.get("data") or data.get("models") or []
        for m in models:
            if m.get("id") == model_id:
                params = m.get("supported_parameters") or []
                return True, params, ""
        return False, [], f"Model not found in OpenRouter /models: {model_id}"
    except Exception as e:
        return False, [], f"{type(e).__name__}: {e}"

def model_metadata_says_tools_supported(model_id: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
    ok, params, err = fetch_openrouter_model_supported_parameters(model_id, api_key=api_key)
    if not ok:
        # Fail closed: if we cannot verify, treat as not supported.
        return False, f"Cannot verify model metadata: {err}"

    params_set = set(params)
    if "tools" in params_set or "tool_choice" in params_set:
        return True, "Model metadata lists 'tools' in supported_parameters"
    return False, f"Model metadata does not list 'tools' (supported_parameters={params})"

def probe_tool_call_via_langchain(model_builder, timeout_s: int = 30) -> Tuple[bool, str, dict]:
    """
    Live probe: forces a tool call using bind_tools + tool_choice.

    Returns:
      (ok, summary, debug_dict)
    """
    from langchain_core.tools import tool

    @tool
    def ping() -> str:
        """Return 'pong'."""
        return "pong"

    try:
        llm = model_builder()
        llm_tools = llm.bind_tools([ping], tool_choice="ping")
        msg = llm_tools.invoke("Call the ping tool now. Do not answer directly.")
        calls = _extract_tool_calls_from_message(msg)

        debug = {
            "content": getattr(msg, "content", None),
            "tool_calls_attr": getattr(msg, "tool_calls", None) if hasattr(msg, "tool_calls") else None,
            "additional_kwargs": getattr(msg, "additional_kwargs", None),
            "response_metadata": getattr(msg, "response_metadata", None),
            "calls_extracted": calls,
        }

        if calls:
            return True, f"Tool call succeeded: {calls}", debug

        return False, "No tool calls detected in AIMessage fields", debug
    except Exception as e:
        debug = {"exception": f"{type(e).__name__}: {str(e)}"}
        return False, f"{type(e).__name__}: {str(e)[:300]}", debug

def check_tools_capability_openrouter(
    model_id: str,
    api_key: Optional[str],
    model_builder,
) -> Dict[str, Any]:
    """
    Returns a dict with:
    - metadata_ok, metadata_reason
    - probe_ok, probe_reason
    - final_ok, final_reason
    """
    meta_ok, meta_reason = model_metadata_says_tools_supported(model_id, api_key=api_key)
    probe_ok, probe_reason, probe_debug = probe_tool_call_via_langchain(model_builder)

    # Final decision: require probe success (hard gate).
    final_ok = bool(probe_ok)
    final_reason = "Probe succeeded" if final_ok else f"Probe failed: {probe_reason}"

    return {
        "metadata_ok": meta_ok,
        "metadata_reason": meta_reason,
        "probe_ok": probe_ok,
        "probe_reason": probe_reason,
        "probe_debug": probe_debug,
        "final_ok": final_ok,
        "final_reason": final_reason,
    }
