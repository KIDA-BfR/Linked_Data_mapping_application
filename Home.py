# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 23:52:23 2025

@author: yurt3
"""

# Home.py
import os
import streamlit as st
from config_utils import load_config, save_config
from gemini_models import list_gemini_models

st.set_page_config(page_title="Services", layout="centered")
st.title("Service Portal")

# -------------------------------------------------
# CONFIG LOAD & SESSION GUARD
# -------------------------------------------------
if "LLM_API_KEY" not in st.session_state:
    # First time initialization: try loading from config.yaml
    config = load_config()
    llm_cfg = config.get("llm", {})
    if llm_cfg:
        st.session_state["LLM_PROVIDER"] = llm_cfg.get("provider", "openai")
        st.session_state["LLM_API_KEY"] = llm_cfg.get("api_key", "")
        st.session_state["LLM_MODEL"] = llm_cfg.get("model", "gpt-4o")
        st.session_state["LLM_BASE_URL"] = llm_cfg.get("base_url", "")
        st.session_state["LANGSMITH_API_KEY"] = llm_cfg.get("langsmith_api_key", "")
    
    # If still no key in session, ensure no leaked env vars exist
    if "LLM_API_KEY" not in st.session_state or not st.session_state["LLM_API_KEY"]:
        for k in [
            "LLM_PROVIDER",
            "LLM_API_KEY",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "LANGSMITH_API_KEY",
            "LANGSMITH_TRACING",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
        ]:
            os.environ.pop(k, None)

if "LLM_API_KEY" in st.session_state and st.session_state["LLM_API_KEY"]:
    # Sync env vars from session (in case of rerun)
    os.environ["LLM_PROVIDER"] = st.session_state.get("LLM_PROVIDER", "openai")
    os.environ["LLM_API_KEY"] = st.session_state["LLM_API_KEY"]
    os.environ["LLM_MODEL"] = st.session_state.get("LLM_MODEL", "gpt-4o")
    if st.session_state.get("LLM_BASE_URL"):
        os.environ["LLM_BASE_URL"] = st.session_state["LLM_BASE_URL"]

    if st.session_state.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = st.session_state["LANGSMITH_API_KEY"]
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "KIDA_data"
        os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

# Pre-fetch Gemini models on app startup if already configured
if st.session_state.get("LLM_PROVIDER") == "gemini" and st.session_state.get("LLM_API_KEY"):
    @st.cache_data(ttl=3600, show_spinner=False)
    def _initial_load_models(key: str):
        return list_gemini_models(key, only_generate_content=True)
    
    if "GEMINI_MODELS" not in st.session_state:
        try:
            models, _ = _initial_load_models(st.session_state["LLM_API_KEY"])
            if models:
                st.session_state["GEMINI_MODELS"] = models
        except Exception:
            pass

# -------------------------------------------------
# LLM CONFIGURATION
# -------------------------------------------------
st.subheader("LLM Configuration")

providers = ["openai", "openrouter", "openai_compatible", "gemini"]
default_provider = st.session_state.get("LLM_PROVIDER", "openai")
default_index = providers.index(default_provider) if default_provider in providers else 0

provider = st.selectbox("LLM Provider", providers, index=default_index)
st.session_state["LLM_PROVIDER"] = provider

api_key_input = st.text_input(
    "LLM API Key",
    type="password",
    value=st.session_state.get("LLM_API_KEY", ""),
    placeholder="sk-...",
)

# --- Model selection ---
st.markdown("### Model")

# default fallback
current_model = st.session_state.get("LLM_MODEL", "gpt-4o")

if provider == "gemini":
    gem_key = api_key_input.strip()

    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_list_models(key: str):
        return list_gemini_models(key, only_generate_content=True)

    # Auto-fetch models if API key is present
    gem_models = []
    gem_debug = None
    
    if gem_key:
        try:
            with st.spinner("Fetching Gemini models…"):
                gem_models, gem_debug = _cached_list_models(gem_key)
        except Exception as e:
            st.error(f"Failed to fetch models: {str(e)}")
            gem_models = []
    
    # Store in session for persistence
    if gem_models:
        st.session_state["GEMINI_MODELS"] = gem_models
    elif "GEMINI_MODELS" in st.session_state:
        gem_models = st.session_state["GEMINI_MODELS"]
    
    # Display dropdown
    if gem_models:
        selected = st.selectbox(
            "Gemini model",
            gem_models,
            index=gem_models.index(current_model) if current_model in gem_models else 0,
            key="gemini_model_selectbox"
        )
        model_input_value = selected
        st.session_state["LLM_MODEL"] = model_input_value
    else:
        st.warning("No Gemini models available. Check your API key and try refreshing.")
        model_input_value = st.text_input(
            "Model (manual fallback)",
            value=current_model if current_model else "gemini-2.0-flash",
            placeholder="e.g. gemini-2.0-flash",
        )
    
    # Optional: debug toggle below dropdown
    show_debug = st.checkbox("Show raw model list debug", value=False, key="gemini_debug")
    if show_debug and gem_debug is not None:
        st.json(gem_debug)

else:
    model_input_value = st.text_input(
        "Model",
        value=current_model,
        placeholder="e.g. gpt-4o or anthropic/claude-3-opus",
        key="non_gemini_model_input"
    )
    st.session_state["LLM_MODEL"] = model_input_value

base_url_input = ""
if provider == "openrouter":
    base_url_input = st.text_input(
        "Base URL",
        value=st.session_state.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
        placeholder="https://openrouter.ai/api/v1",
    )
elif provider == "openai_compatible":
    base_url_input = st.text_input(
        "Model Server Base URL (OpenAI-compatible /v1)",
        value=st.session_state.get("LLM_BASE_URL", ""),
        placeholder="http://localhost:11434/v1  (Ollama)  |  http://localhost:8000/v1 (vLLM)",
        help="Enter a backend model server URL ending in /v1, or an Open WebUI proxy URL ending in /ollama/v1 or /openai.",
    )
elif provider == "gemini":
    base_url_input = ""
st.caption(
        "Enter the base URL for the LLM provider. URLs ending in /v1, /ollama/v1, /openai, or /api are common, but other valid URLs are also accepted."
    )

st.divider()
st.subheader("Other Keys")

langsmith_key_input = st.text_input(
    "LangSmith API key (optional)",
    type="password",
    placeholder="lsv2_...",
)

col_save_session, col_save_yaml, col_clear = st.columns(3)

with col_save_session:
    if st.button("Save for this session", use_container_width=True):
        api_key = api_key_input.strip()
        langsmith_key = langsmith_key_input.strip()

        if not api_key:
            st.error("LLM API key is required.")
            st.stop()

        if provider == "gemini":
            # No base URL required; store GEMINI_API_KEY
            st.session_state["LLM_PROVIDER"] = provider
            st.session_state["LLM_API_KEY"] = api_key_input.strip()
            st.session_state["LLM_MODEL"] = model_input_value.strip()
            st.session_state.pop("LLM_BASE_URL", None)
            st.session_state["LANGSMITH_API_KEY"] = langsmith_key_input.strip()

            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_API_KEY"] = api_key_input.strip()
            os.environ["LLM_MODEL"] = st.session_state["LLM_MODEL"]
            os.environ.pop("LLM_BASE_URL", None)
            os.environ["GEMINI_API_KEY"] = api_key_input.strip()
            if langsmith_key_input.strip():
                os.environ["LANGSMITH_API_KEY"] = langsmith_key_input.strip()
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_PROJECT"] = "KIDA_data"
                os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

            st.success("Saved for session.")
        elif provider in ("openrouter", "openai_compatible", "openwebui"):
            bu = base_url_input.strip().rstrip("/")
            if provider in ("openrouter", "openai_compatible") and not bu:
                st.error("Base URL is required for the selected provider.")
                st.stop()
            # No strict pattern enforcement; any non‑empty URL is accepted.

            # Store in session
            st.session_state["LLM_PROVIDER"] = provider
            st.session_state["LLM_API_KEY"] = api_key_input.strip()
            st.session_state["LLM_MODEL"] = model_input_value.strip()
            if provider in ("openrouter", "openai_compatible"):
                st.session_state["LLM_BASE_URL"] = base_url_input.strip()
            else:
                st.session_state.pop("LLM_BASE_URL", None)
            st.session_state["LANGSMITH_API_KEY"] = langsmith_key_input.strip()

            # Sync to environment variables
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_API_KEY"] = api_key_input.strip()
            os.environ["LLM_MODEL"] = st.session_state["LLM_MODEL"]
            if provider in ("openrouter", "openai_compatible"):
                os.environ["LLM_BASE_URL"] = st.session_state["LLM_BASE_URL"]
            else:
                os.environ.pop("LLM_BASE_URL", None)
            if langsmith_key_input.strip():
                os.environ["LANGSMITH_API_KEY"] = langsmith_key_input.strip()
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_PROJECT"] = "KIDA_data"
                os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

            st.success("Saved for session.")

with col_save_yaml:
    if st.button("Save to config.yaml", use_container_width=True):
        api_key = api_key_input.strip()
        langsmith_key = langsmith_key_input.strip()

        if not api_key:
            st.error("LLM API key is required.")
            st.stop()

        if provider == "gemini":
            # No base URL needed; store GEMINI_API_KEY
            st.session_state["LLM_PROVIDER"] = provider
            st.session_state["LLM_API_KEY"] = api_key_input.strip()
            st.session_state["LLM_MODEL"] = model_input_value.strip()
            st.session_state.pop("LLM_BASE_URL", None)
            st.session_state["LANGSMITH_API_KEY"] = langsmith_key_input.strip()

            # Update config.yaml
            config = load_config()
            config["llm"] = {
                "provider": provider,
                "api_key": api_key_input.strip(),
                "model": model_input_value.strip(),
                "base_url": "",
                "langsmith_api_key": langsmith_key_input.strip()
            }
            save_config(config)

            # Sync to environment variables
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_API_KEY"] = api_key_input.strip()
            os.environ["LLM_MODEL"] = st.session_state["LLM_MODEL"]
            os.environ.pop("LLM_BASE_URL", None)
            os.environ["GEMINI_API_KEY"] = api_key_input.strip()
            if langsmith_key_input.strip():
                os.environ["LANGSMITH_API_KEY"] = langsmith_key_input.strip()
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_PROJECT"] = "KIDA_data"
                os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

            st.success("Saved to config.yaml.")
        elif provider in ("openrouter", "openai_compatible", "openwebui"):
            bu = base_url_input.strip().rstrip("/")
            if provider in ("openrouter", "openai_compatible") and not bu:
                st.error("Base URL is required for the selected provider.")
                st.stop()
            # No strict pattern enforcement; any non‑empty URL is accepted.

            # Update session state
            st.session_state["LLM_PROVIDER"] = provider
            st.session_state["LLM_API_KEY"] = api_key_input.strip()
            st.session_state["LLM_MODEL"] = model_input_value.strip()
            if provider in ("openrouter", "openai_compatible"):
                st.session_state["LLM_BASE_URL"] = base_url_input.strip()
            else:
                st.session_state.pop("LLM_BASE_URL", None)
            st.session_state["LANGSMITH_API_KEY"] = langsmith_key_input.strip()

            # Update config.yaml
            config = load_config()
            config["llm"] = {
                "provider": provider,
                "api_key": api_key_input.strip(),
                "model": model_input_value.strip(),
                "base_url": base_url_input.strip() if provider in ("openrouter", "openai_compatible") else "",
                "langsmith_api_key": langsmith_key_input.strip()
            }
            save_config(config)

            # Sync to environment variables
            os.environ["LLM_PROVIDER"] = provider
            os.environ["LLM_API_KEY"] = api_key_input.strip()
            os.environ["LLM_MODEL"] = st.session_state["LLM_MODEL"]
            if provider in ("openrouter", "openai_compatible"):
                os.environ["LLM_BASE_URL"] = st.session_state["LLM_BASE_URL"]
            else:
                os.environ.pop("LLM_BASE_URL", None)
            if langsmith_key_input.strip():
                os.environ["LANGSMITH_API_KEY"] = langsmith_key_input.strip()
                os.environ["LANGSMITH_TRACING"] = "true"
                os.environ["LANGSMITH_PROJECT"] = "KIDA_data"
                os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

            st.success("Saved to config.yaml.")

with col_clear:
    if st.button("Clear configuration", use_container_width=True):
        st.session_state.clear()
        for k in [
            "LLM_PROVIDER",
            "LLM_API_KEY",
            "LLM_MODEL",
            "LLM_BASE_URL",
            "LANGSMITH_API_KEY",
            "LANGSMITH_TRACING",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
        ]:
            os.environ.pop(k, None)

        st.info("Configuration cleared. Please re-enter to continue.")

st.divider()

# -------------------------------------------------
# NAVIGATION (GUARDED)
# -------------------------------------------------
has_llm_key = bool(st.session_state.get("LLM_API_KEY"))

if not has_llm_key:
    st.warning("You must provide an LLM API key to access services.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    if st.button("Mapping service", use_container_width=True):
        st.switch_page("pages/Mapping_service.py")

with col2:
    if st.button("Verification service", use_container_width=True):
        st.switch_page("pages/Verification_service.py")
