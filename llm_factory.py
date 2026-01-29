from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import requests

from langchain_openai import ChatOpenAI

# Attempt to import Gemini integration; if unavailable, keep placeholder
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


@dataclass(frozen=True)
class LLMSettings:
    provider: str               # "openai" | "openrouter" | "openai_compatible" | "openwebui" | "gemini"
    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = 0.0


def get_llm_settings_from_env() -> LLMSettings:
    """
    Reads LLM configuration from environment variables (or Streamlit session state)
    and returns a validated LLMSettings instance.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    model = os.getenv("LLM_MODEL", "gpt-4o").strip()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None

    # Allow OpenAI-compatible env var name as fallback
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

    # Gemini uses a distinct env var name
    if provider == "gemini" and not api_key:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY. Configure it in the UI or environment.")

    # Provider‑specific validation / defaults
    if provider == "openrouter" and not base_url:
        base_url = "https://openrouter.ai/api/v1"

    if provider == "openai_compatible" and not base_url:
        raise RuntimeError("LLM_BASE_URL is required for provider=openai_compatible")

    if provider == "openwebui" and not base_url:
        raise RuntimeError("LLM_BASE_URL is required for provider=openwebui")

    if provider not in {"openai", "openrouter", "openai_compatible", "openwebui", "gemini"}:
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {provider}")

    # Temperature handling – default to 0.0 on any conversion problem
    try:
        temp = float(os.getenv("LLM_TEMPERATURE", "0") or 0)
    except ValueError:
        temp = 0.0

    return LLMSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temp,
    )


def _gemini_supports_generate_content(api_key: str, model: str) -> bool:
    """Check if the selected Gemini model supports generateContent."""
    m = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{m}?key={api_key}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return False
        data = r.json()
        methods = data.get("supportedGenerationMethods") or []
        return "generateContent" in methods
    except Exception:
        return False


def build_chat_llm(settings: Optional[LLMSettings] = None, *, streaming: bool = False):
    """
    Builds a LangChain chat model appropriate for the selected provider.
    Tool‑using agents MUST run with streaming=False for compatibility with some providers.
    """
    s = settings or get_llm_settings_from_env()

    # Gemini provider handling
    if s.provider == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "Gemini provider selected but langchain-google-genai is not installed. "
                "Run: pip install -U langchain-google-genai"
            )

        # Preflight check for Gemini models
        if "deep-research" in s.model.lower():
            raise RuntimeError(
                "Selected model is Gemini Deep Research (Interactions API only). "
                "Pick a Gemini model that supports generateContent (e.g., gemini-2.0-flash)."
            )

        if not _gemini_supports_generate_content(s.api_key, s.model):
            raise RuntimeError(
                f"Selected Gemini model '{s.model}' does not support generateContent. "
                "It cannot be used with ChatGoogleGenerativeAI; choose another model."
            )

        # Gemini does not support streaming in the current agent setup
        return ChatGoogleGenerativeAI(
            model=s.model,
            google_api_key=s.api_key,
            temperature=s.temperature,
        )

    extra_body = None
    if s.provider == "openrouter":
        extra_body = {"provider": {"allow_fallbacks": True}}

    # Normalize base URL for OpenWebUI (avoid duplicate slashes)
    if s.provider == "openwebui" and s.base_url:
        s.base_url = s.base_url.rstrip("/")

    # Enforce streaming off unless explicitly requested
    return ChatOpenAI(
        model=s.model,
        api_key=s.api_key,
        base_url=s.base_url,
        temperature=s.temperature,
        streaming=bool(streaming),
        extra_body=extra_body,
    )
