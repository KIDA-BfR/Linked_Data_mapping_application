# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 00:04:44 2025

@author: yurt3
"""

import streamlit as st

st.set_page_config(page_title="Verification service", layout="centered")

st.title("Verification service")

# Back button
if st.button("← Back to Home"):
    st.switch_page("Home.py")

st.divider()

# 3.1 Verified fields
verified_term = st.text_input("Verified term")
verified_iri = st.text_input("Verified IRI")
verified_skos = st.text_input("Verified SKOS")

# 3.2 Multiple terms checkbox (you wrote "under Searched Term", but on this page
# the closest equivalent is under "Verified term")
multiple_terms = st.checkbox("Multiple terms")

# 3.2.1 / 3.2.2 Show uploader only if checked
uploaded_file = None
if multiple_terms:
    uploaded_file = st.file_uploader("Upload file", type=["csv", "txt", "tsv", "xlsx"])

st.divider()

# 3.3 Result of IRI verification
result_iri_verification = st.text_input("Result of IRI verification")

# 3.4 Result of SKOS verification
result_skos_verification = st.text_input("Result of SKOS verification")
