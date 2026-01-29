# gemini_models.py
from __future__ import annotations

from typing import List, Tuple
import requests

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def _strip_models_prefix(name: str) -> str:
    """Remove the leading 'models/' prefix if present."""
    return name.replace("models/", "", 1) if name.startswith("models/") else name


def list_gemini_models(api_key: str, *, only_generate_content: bool = True, timeout_s: int = 15) -> Tuple[List[str], dict]:
    """
    Fetch available Gemini model IDs via the Gemini API.

    Returns:
        (model_ids, raw_response)
        model_ids – list of normalized model IDs (without the "models/" prefix)
        raw_response – dictionary with status code and raw JSON (or error info)
    """
    if not api_key:
        return [], {"error": "Missing GEMINI_API_KEY"}

    url = f"{GEMINI_MODELS_URL}?key={api_key}"
    try:
        r = requests.get(url, timeout=timeout_s)
    except Exception as e:
        return [], {"error": str(e)}

    raw = {"status_code": r.status_code}
    try:
        data = r.json()
    except Exception:
        raw["text"] = r.text[:500]
        return [], raw

    raw["data"] = data
    if r.status_code != 200:
        return [], raw

    models = data.get("models", []) or []
    out: List[str] = []
    for m in models:
        name = m.get("name", "")
        if not name:
            continue
        # Handle possible field name variations for supported actions/methods
        methods = (
            m.get("supportedGenerationMethods")
            or m.get("supported_actions")
            or m.get("supportedActions")
            or []
        )
        if only_generate_content and "generateContent" not in methods:
            continue
        
        # Exclude Deep Research models as they only support Interactions API
        normalized_name = _strip_models_prefix(name)
        if "deep-research" in normalized_name.lower():
            continue
            
        out.append(normalized_name)

    # Ensure deterministic ordering
    out = sorted(set(out))
    return out, raw


def _supports_generate_content(m: dict) -> bool:
    methods = m.get("supportedGenerationMethods") or m.get("supported_generation_methods") or []
    return "generateContent" in methods
