from __future__ import annotations
from pathlib import Path
import yaml
import json
import re
from typing import Any, Dict, Optional

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def save_config(cfg: dict, path: Path = CONFIG_PATH) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def _content_to_text(content: Any) -> str:
    """Normalize LangChain/Gemini content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Gemini can return a list of content parts
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    return str(content)


def _try_parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON if present; otherwise return {}."""
    if not text:
        return {}
    text = text.strip()

    # Direct JSON
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Extract first JSON object from text if wrapped in other content
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return {}

    return {}


def _coerce_to_dict(obj: Any) -> Dict[str, Any]:
    """Support dict + Pydantic v1/v2 + JSON text."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, str):
        return _try_parse_json(obj)

    # Sometimes tools return objects with .content
    if hasattr(obj, "content"):
        return _try_parse_json(_content_to_text(getattr(obj, "content")))

    return {}


def extract_structured_payload(
    result: Any,
    messages: Optional[List[Any]] = None,
    schema_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Robust extraction for Gemini function-calling and other LLMs.
    Preferred order:
      1) result["output"] (Pydantic/dict)
      2) scan messages backwards for function_call (Gemini style)
      3) scan messages backwards for parseable JSON in content
    """
    # 1) Handle Pydantic objects directly if passed as result
    if hasattr(result, "model_dump"):
        return result.model_dump()

    if not isinstance(result, dict):
        result = {}

    # 2) Best case: agent returned a structured output object in 'output'
    output = result.get("output")
    payload = _coerce_to_dict(output)
    if payload and any(k in payload for k in ("iri", "qid", "ID", "skos", "explanation")):
        return payload

    # 3) Fallback: scan messages from back to front
    msgs = messages or result.get("messages", []) or []
    for msg in reversed(msgs):
        # 3a) Check for function_call (Gemini style)
        fc = getattr(msg, "additional_kwargs", {}).get("function_call")
        if fc and isinstance(fc, dict):
            # If schema_name is provided, skip if it doesn't match
            if schema_name and fc.get("name") != schema_name:
                continue
            args = fc.get("arguments")
            if isinstance(args, str):
                try:
                    return json.loads(args)
                except Exception:
                    pass
            elif isinstance(args, dict):
                return args

        # 3b) Check for parseable JSON in content
        text = _content_to_text(getattr(msg, "content", None))
        candidate = _try_parse_json(text)
        if candidate:
            return candidate

    return {}
