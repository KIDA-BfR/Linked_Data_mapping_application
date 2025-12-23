# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 23:52:23 2025

@author: yurt3
"""

# Home.py
import os
import streamlit as st

st.set_page_config(page_title="Services", layout="centered")
st.title("Service Portal")

# -------------------------------------------------
# SESSION GUARD (authoritative source = session_state)
# -------------------------------------------------
if "OPENAI_API_KEY" not in st.session_state:
    # Ensure no leaked env vars exist at session start
    for k in [
        "OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGSMITH_TRACING",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
    ]:
        os.environ.pop(k, None)
else:
    # Sync env vars from session (in case of rerun)
    os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]

    if st.session_state.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = st.session_state["LANGSMITH_API_KEY"]
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = "KIDA_data"
        os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

# -------------------------------------------------
# API KEY INPUTS (SESSION ONLY)
# -------------------------------------------------
st.subheader("API Keys (valid for this session only)")

openai_key_input = st.text_input(
    "OpenAI API key",
    type="password",
    placeholder="sk-...",
)

langsmith_key_input = st.text_input(
    "LangSmith API key (optional)",
    type="password",
    placeholder="lsv2_...",
)

col_save, col_clear = st.columns(2)

with col_save:
    if st.button("Save keys", use_container_width=True):
        openai_key = openai_key_input.strip()
        langsmith_key = langsmith_key_input.strip()

        if not openai_key:
            st.error("OpenAI API key is required.")
            st.stop()

        # Store in session (dies with session)
        st.session_state["OPENAI_API_KEY"] = openai_key
        st.session_state["LANGSMITH_API_KEY"] = langsmith_key

        # Set environment variables
        os.environ["OPENAI_API_KEY"] = openai_key

        if langsmith_key:
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_PROJECT"] = "KIDA_data"
            os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"

        st.success("Keys saved for this session.")

with col_clear:
    if st.button("Clear keys", use_container_width=True):
        st.session_state.clear()
        for k in [
            "OPENAI_API_KEY",
            "LANGSMITH_API_KEY",
            "LANGSMITH_TRACING",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
        ]:
            os.environ.pop(k, None)

        st.info("Keys cleared. Please re-enter to continue.")

st.divider()

# -------------------------------------------------
# NAVIGATION (GUARDED)
# -------------------------------------------------
has_openai_key = bool(st.session_state.get("OPENAI_API_KEY"))

if not has_openai_key:
    st.warning("You must provide an OpenAI API key to access services.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    if st.button("Mapping service", use_container_width=True):
        st.switch_page("pages/Mapping_service.py")

with col2:
    if st.button("Verification service", use_container_width=True):
        st.switch_page("pages/Verification_service.py")
