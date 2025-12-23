"""
Created on Mon Dec 22 00:04:44 2025

@author: Iurii_Savvateev
"""


import streamlit as st

from general_tools.skos_tools import classify_skos_match  # adjust if your path differs


st.set_page_config(page_title="Verification service", layout="centered")
st.title("Verification service")

# Back button
if st.button("← Back to Home"):
    st.switch_page("Home.py")

st.divider()

# ---- Inputs (5 fields) ----
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

# ---- Result (one field) ----
st.subheader("Result")

if st.button("Verify SKOS match", use_container_width=True):
    # Basic guards
    if not term.strip():
        st.error("Please provide Term.")
        st.stop()
    if not term_def.strip():
        st.error("Please provide Term Definition.")
        st.stop()
    if not label.strip():
        st.error("Please provide Label.")
        st.stop()
    if not label_def.strip():
        st.error("Please provide Label Definition.")
        st.stop()

    try:
        out = classify_skos_match(
            term_a=term.strip(),
            gen_def=term_def.strip(),
            term_b=label.strip(),
            onto_def=label_def.strip(),
        )
    except Exception as e:
        st.error(f"Verification failed: {e}")
        st.stop()

    predicted = (out.get("mapping_type") or "").strip().lower()
    explanation = (out.get("explanation") or "").strip()
    provided = (provided_skos or "").strip().lower()

    if predicted == provided:
        st.success(f"✅ The provided SKOS class is correct: **{provided}**")
    else:
        # If mismatch, show predicted + reasoning
        msg = (
            f"⚠️ Provided SKOS class: **{provided}**\n\n"
            f"Predicted SKOS class (from classify_skos_match): **{predicted or 'unknown'}**\n\n"
            f"Explanation:\n{explanation or '—'}"
        )
        st.warning(msg)

else:
    st.info("Fill the fields above and click **Verify SKOS match** to validate the provided SKOS class.")

