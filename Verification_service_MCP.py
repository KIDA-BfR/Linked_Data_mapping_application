# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 00:04:44 2025

@author: Iurii_Savvateev
"""

# -*- coding: utf-8 -*-
"""
Streamlit Verification service (via MCP tool call)

This version calls classify_skos_match through the MCP server tool
defined in mcp_skos_server.py (classify_skos_match_tool).
"""

import os
import sys
import json
import asyncio
import streamlit as st

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# ---------- MCP client helper ----------
def _extract_tool_payload(tool_result):
    """
    MCP tool calls often return an object whose `content` is a LIST of items.
    Those items may be TextContent objects (with `.text`) or dict-like objects.

    We try to recover a dict like: {"mapping_type": "...", "explanation": "..."}.
    """
    # Case 1: already a dict
    if isinstance(tool_result, dict):
        return tool_result

    # Case 2: has `.content`
    content = getattr(tool_result, "content", None)
    if content is None:
        # fallback: string
        return {"mapping_type": "none", "explanation": str(tool_result)}

    # Many SDKs: content is a list
    if isinstance(content, list) and len(content) > 0:
        first = content[0]

        # If it's a dict already
        if isinstance(first, dict):
            return first

        # If it has `.text` (TextContent)
        text = getattr(first, "text", None)
        if isinstance(text, str):
            # sometimes the tool returns JSON as text
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {"mapping_type": "none", "explanation": text}

        # Fallback: stringify first block
        return {"mapping_type": "none", "explanation": str(first)}

    # Case 3: content isn’t a list
    if isinstance(content, dict):
        return content

    return {"mapping_type": "none", "explanation": str(tool_result)}


async def _classify_skos_match_via_mcp_async(term_a: str, gen_def: str, term_b: str, onto_def: str):
    """
    Calls MCP tool `classify_skos_match_tool` by spawning the MCP server
    as a subprocess using stdio transport.
    """

    # IMPORTANT: In this MCP version, stdio_client expects StdioServerParameters
    params = StdioServerParameters(
        command=sys.executable,                 # use same Python as Streamlit (conda-safe)
        args=["mcp_skos_server.py"],
        env={**os.environ},                     # pass OPENAI_API_KEY etc. through
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tool_result = await session.call_tool(
                "classify_skos_match_tool",
                {
                    "term_a": term_a,
                    "gen_def": gen_def,
                    "term_b": term_b,
                    "onto_def": onto_def,
                },
            )

            return _extract_tool_payload(tool_result)


def classify_skos_match_via_mcp(term_a: str, gen_def: str, term_b: str, onto_def: str):
    """
    Sync wrapper for Streamlit callbacks.
    """
    return asyncio.run(_classify_skos_match_via_mcp_async(term_a, gen_def, term_b, onto_def))


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Verification service", layout="centered")
st.title("Verification service")

# Back button
if st.button("← Back to Home"):
    st.switch_page("Home.py")

st.divider()

term = st.text_input("Term")
term_def = st.text_area("Term Definition")

label = st.text_input("Label")
label_def = st.text_area("Label Definition")

provided_skos = st.selectbox(
    "Term–Label SKOS class (provided)",
    options=["exact", "close", "related", "none"],
    index=0,
)

st.divider()
st.subheader("Result")

if st.button("Verify SKOS match", use_container_width=True):
    if not term.strip():
        st.error("Please provide Term."); st.stop()
    if not term_def.strip():
        st.error("Please provide Term Definition."); st.stop()
    if not label.strip():
        st.error("Please provide Label."); st.stop()
    if not label_def.strip():
        st.error("Please provide Label Definition."); st.stop()

    try:
        out = classify_skos_match_via_mcp(
            term_a=term.strip(),
            gen_def=term_def.strip(),
            term_b=label.strip(),
            onto_def=label_def.strip(),
        )
    except Exception as e:
        st.error(f"Verification failed (MCP call): {e}")
        st.stop()

    predicted = (out.get("mapping_type") or "").strip().lower()
    explanation = (out.get("explanation") or "").strip()
    provided = (provided_skos or "").strip().lower()

    if predicted == provided:
        st.success(f"✅ The provided SKOS class is correct: **{provided}**")
    else:
        st.warning(
            f"⚠️ Provided SKOS class: **{provided}**\n\n"
            f"Predicted SKOS class (via MCP): **{predicted or 'unknown'}**\n\n"
            f"Explanation:\n{explanation or '—'}"
        )
else:
    st.info("Fill the fields above and click **Verify SKOS match**.")

