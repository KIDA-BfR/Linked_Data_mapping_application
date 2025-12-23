# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 12:24:07 2025

@author: Iurii Savvateev
"""

# mcp_skos_server.py
import os
from typing import Dict

# MCP (FastMCP) server
from mcp.server.fastmcp import FastMCP

# Your existing function
from general_tools.skos_tools import classify_skos_match  # adjust import to your project layout

mcp = FastMCP("skos-verification")

@mcp.tool()
def classify_skos_match_tool(
    term_a: str,
    gen_def: str,
    term_b: str,
    onto_def: str,
) -> Dict[str, str]:
    """
    Classify SKOS relationship between two concepts.
    Returns: {"mapping_type": "...", "explanation": "..."}
    """
    # classify_skos_match already checks OPENAI_API_KEY (in your skos_tools.py)
    return classify_skos_match(term_a=term_a, gen_def=gen_def, term_b=term_b, onto_def=onto_def)

if __name__ == "__main__":
    # Ensure env var is set (your code raises if missing)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (expected env var).")
    mcp.run()
