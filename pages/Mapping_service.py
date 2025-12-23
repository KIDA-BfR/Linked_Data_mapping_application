# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 00:04:18 2025

@author: yurt3
"""
# pages/Mapping_service.py

import os
import json
from io import BytesIO
from typing import List, Dict, Any
import uuid

import pandas as pd
import streamlit as st

from wikidata_agent_and_tools.deep_agent_wikidata import get_agent_wiki
from bioportal_agent_and_tools.deep_agent_bioportal import get_agent_bioportal
from bioportal_wikidata_system.multiagent_system import get_multiagent  # NEW


# ============================================================
# Page setup & guards
# ============================================================
st.set_page_config(page_title="Mapping service", layout="centered")
st.title("Mapping service")

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API key is not set. Please go back to Home and enter it.")
    if st.button("← Back to Home"):
        st.switch_page("Home.py")
    st.stop()

if st.button("← Back to Home"):
    st.switch_page("Home.py")

st.divider()


# ============================================================
# Session-state defaults
# ============================================================
defaults = {
    "mapping_term_input": "",
    "mapping_definition_input": "",
    "mapping_multi_input": False,
    "mapping_endpoints_input": [],

    # BioPortal session-only inputs
    "bioportal_api_key_input": "",
    "trusted_ontologies_input": "MESH,NCIT,LOINC,FOODON",
    "term_ontologies_input": "NCIT,NIFSTD,SNOMEDCT",

    # Single-term output
    "mapping_iri_out": "",
    "mapping_skos_out": "",
    "mapping_expl_out": "",

    # Batch output
    "mapping_batch_df": None,

    # Highlight only last re-evaluation
    "last_reeval_run_id": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ============================================================
# Helpers
# ============================================================
def _parse_csv_list(text: str) -> List[str]:
    items = [x.strip() for x in (text or "").split(",") if x.strip()]
    # stable de-dupe
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _parse_agent_json(raw: str) -> Dict[str, Any]:
    """Parse dict from JSON; fallback extracts first {...} block."""
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                data = json.loads(raw[s:e+1])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}


def _wikidata_url(qid: str) -> str:
    return f"https://www.wikidata.org/wiki/{qid}"


def _qid_to_url_if_needed(identifier: str) -> str:
    """If identifier is a bare Wikidata QID (Q123), convert to Wikidata URL."""
    ident = (identifier or "").strip()
    if ident.startswith("Q") and ident[1:].isdigit():
        return _wikidata_url(ident)
    return ident


def _extract_multiagent_fields(parsed: Dict[str, Any]) -> Dict[str, str]:
    """
    Multiagent final formatting tool (agentmapping_format) returns:
      {"ID": "...", "SKOS": "...", "SKOS_explanation": "..."} :contentReference[oaicite:1]{index=1}
    But we also accept fallback keys to be robust.
    """
    ident = (
        parsed.get("ID")
        or parsed.get("id")
        or parsed.get("qid")
        or parsed.get("IRI")
        or parsed.get("iri")
        or ""
    )
    skos = (
        parsed.get("SKOS")
        or parsed.get("skos")
        or ""
    )
    expl = (
        parsed.get("SKOS_explanation")
        or parsed.get("skos_explanation")
        or parsed.get("explanation")
        or ""
    )
    ident = _qid_to_url_if_needed(str(ident))
    return {
        "iri": str(ident).strip(),
        "skos": str(skos).strip(),
        "explanation": str(expl).strip(),
    }


def _question_wikidata(term: str, definition: str) -> str:
    return f"""What is the best fitting Q-identifier the term {term} which definition {definition} matches with the label from the wikidata .
As a reply only provide
(i) identificator number,
(ii) SKOS matching between original term definition and the definition/description of the identified label from wikidata
(iii) Explanation for SKOS matching.

SKOS matching and corresponding explanation should be identified with the corresponding tool -

If no proper match is found, you may adjust the search query and try with other identifiers.

If no proper identifier is found after 10 iterations, return "No wiki match". In that case SKOS matching is not needed
"""


def _question_bioportal(term: str, definition: str, term_onts: List[str], trusted_onts: List[str]) -> str:
    # Minimal but functional prompt (your previous version was a stub).
    # Duplication of wikidata extensive prompt is not needed as the logic is more rigid and is defined in the system prompt
    return f"""Find the best BioPortal identifier/IRI for the term {term} with definition {definition}.

"""


def _question_multiagent(term: str, definition: str) -> str:
    # Keep minimal; multiagent_system's main prompt will drive the behavior.
    return f"""Map the term "{term}" with definition "{definition}" to a valid identifier from BioPortal or Wikidata.
Return only the final JSON output.
"""


def _ensure_batch_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure RowID + OriginalTerm + last_updated_run columns exist."""
    out = df.copy()

    if "RowID" not in out.columns:
        out.insert(0, "RowID", [str(uuid.uuid4()) for _ in range(len(out))])

    if "OriginalTerm" not in out.columns:
        out["OriginalTerm"] = out["Term"]

    if "last_updated_run" not in out.columns:
        out["last_updated_run"] = ""

    return out


# ============================================================
# Agent caches
# - multiagent cache includes ontology lists so it rebuilds on change
# ============================================================
@st.cache_resource
def _cached_wiki_agent(openai_key: str, langsmith_key: str):
    return get_agent_wiki()


@st.cache_resource
def _cached_bio_agent(
    openai_key: str,
    langsmith_key: str,
    bioportal_key: str,
    trusted_ontologies: tuple,
    term_ontologies: tuple,
):
    return get_agent_bioportal(
        trusted_ontologies=list(trusted_ontologies),
        term_ontologies=list(term_ontologies),
    )


@st.cache_resource
def _cached_multi_agent(
    openai_key: str,
    langsmith_key: str,
    bioportal_key: str,
    trusted_ontologies: tuple,
    term_ontologies: tuple,
):
    # Cache key includes ontology tuples (rebuild when they change)
    return get_multiagent(
        trusted_ontologies=list(trusted_ontologies),
        term_ontologies=list(term_ontologies),
    )


def _get_wiki_agent():
    return _cached_wiki_agent(
        os.environ.get("OPENAI_API_KEY", ""),
        os.environ.get("LANGSMITH_API_KEY", ""),
    )


def _get_bio_agent(trusted_onts: List[str], term_onts: List[str]):
    return _cached_bio_agent(
        os.environ.get("OPENAI_API_KEY", ""),
        os.environ.get("LANGSMITH_API_KEY", ""),
        os.environ.get("BIOPORTAL_API_KEY", ""),
        tuple(trusted_onts),
        tuple(term_onts),
    )


def _get_multi_agent(trusted_onts: List[str], term_onts: List[str]):
    return _cached_multi_agent(
        os.environ.get("OPENAI_API_KEY", ""),
        os.environ.get("LANGSMITH_API_KEY", ""),
        os.environ.get("BIOPORTAL_API_KEY", ""),
        tuple(trusted_onts),
        tuple(term_onts),
    )


# ============================================================
# Inputs
# ============================================================
searched_term = st.text_input("Searched Term", key="mapping_term_input")
multiple_terms = st.checkbox("Multiple terms", key="mapping_multi_input")

uploaded_file = None
term_definition = ""

if not multiple_terms:
    term_definition = st.text_area("Term definition", key="mapping_definition_input")
else:
    uploaded_file = st.file_uploader(
        "Upload Excel file (must contain columns: Term, Definition)",
        type=["xlsx"],
        key="mapping_upload_excel",
    )

st.divider()

endpoints = st.multiselect(
    "Endpoints",
    ["Wikidata", "Bioportal"],
    key="mapping_endpoints_input",
)

# --- Endpoint selection with Multiagent override when BOTH selected ---
endpoint_to_run = None
if len(endpoints) == 1:
    endpoint_to_run = endpoints[0]
elif set(endpoints) == {"Wikidata", "Bioportal"}:
    endpoint_to_run = "Multiagent"
    st.info("Both endpoints selected → using the Multiagent system (BioPortal → Wikidata).")
elif len(endpoints) > 1:
    endpoint_to_run = st.radio("Run mapping against", endpoints, horizontal=True)
else:
    endpoint_to_run = None

trusted_ontologies: List[str] = []
term_ontologies: List[str] = []

# ============================================================
# BioPortal configuration (Bioportal OR Multiagent)
# ============================================================
if endpoint_to_run in {"Bioportal", "Multiagent"}:
    st.subheader("BioPortal configuration")

    bio_key = st.text_input(
        "BIOPORTAL_API_KEY",
        type="password",
        key="bioportal_api_key_input",
        help="Stored only for this session.",
    )
    os.environ["BIOPORTAL_API_KEY"] = (bio_key or "").strip()

    trusted_text = st.text_area(
        "trusted_ontologies (comma-separated)",
        key="trusted_ontologies_input",
    )
    term_text = st.text_area(
        "term_ontologies (comma-separated)",
        key="term_ontologies_input",
    )
    trusted_ontologies = _parse_csv_list(trusted_text)
    term_ontologies = _parse_csv_list(term_text)

    # NEW: set env vars from UI (session-only)
    os.environ["BIOPORTAL_TRUSTED_ONTOLOGIES"] = trusted_text or ""
    os.environ["BIOPORTAL_TERM_ONTOLOGIES"] = term_text or ""

st.divider()


# ============================================================
# Single-term mapping
# ============================================================
st.subheader("Mapping (single term)")

run_single_enabled = (not multiple_terms) and (endpoint_to_run in {"Wikidata", "Bioportal", "Multiagent"})

if st.button("Run mapping", disabled=not run_single_enabled, use_container_width=True):
    if not searched_term.strip():
        st.error("Provide a term.")
        st.stop()
    if not term_definition.strip():
        st.error("Provide a definition.")
        st.stop()

    if endpoint_to_run == "Wikidata":
        agent = _get_wiki_agent()
        question = _question_wikidata(searched_term.strip(), term_definition.strip())
        with st.spinner("Running Wikidata agent..."):
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
        parsed = _parse_agent_json(raw)

        qid = (parsed.get("qid") or "").strip()
        skos = (parsed.get("skos") or "").strip()
        expl = (parsed.get("explanation") or "").strip()

        if qid and qid != "No wiki match":
            iri = _wikidata_url(qid)
        else:
            iri = "No wiki match"
            skos, expl = "", ""

    elif endpoint_to_run == "Bioportal":
        if not os.environ.get("BIOPORTAL_API_KEY"):
            st.error("BIOPORTAL_API_KEY is required for BioPortal.")
            st.stop()
        if not term_ontologies:
            st.error("Please provide term_ontologies for BioPortal search.")
            st.stop()

        agent = _get_bio_agent(trusted_ontologies, term_ontologies)
        question = _question_bioportal(searched_term.strip(), term_definition.strip(), term_ontologies, trusted_ontologies)
        with st.spinner("Running BioPortal agent..."):
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
        raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
        parsed = _parse_agent_json(raw)

        iri = (parsed.get("qid") or "").strip()  # agent returns IRI in field "qid"
        skos = (parsed.get("skos") or "").strip()
        expl = (parsed.get("explanation") or "").strip()

        if not iri or iri == "No bioportal match":
            iri = "No bioportal match"
            skos, expl = "", ""

    else:  # Multiagent
        if not os.environ.get("BIOPORTAL_API_KEY"):
            st.error("BIOPORTAL_API_KEY is required for Multiagent.")
            st.stop()
        if not term_ontologies:
            st.error("Please provide term_ontologies for Multiagent.")
            st.stop()

        agent = _get_multi_agent(trusted_ontologies, term_ontologies)
        question = _question_multiagent(searched_term.strip(), term_definition.strip())
        with st.spinner("Running Multiagent system..."):
            result = agent.invoke({"messages": [{"role": "user", "content": question}]})

        raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
        parsed = _parse_agent_json(raw)

        # ✅ FIX: Multiagent output is produced by agentmapping_format → keys: ID, SKOS, SKOS_explanation
        fields = _extract_multiagent_fields(parsed)
        iri = fields["iri"]
        skos = fields["skos"]
        expl = fields["explanation"]

        if not iri or iri.startswith("No "):
            skos, expl = "", ""

    st.session_state["mapping_iri_out"] = iri
    st.session_state["mapping_skos_out"] = skos
    st.session_state["mapping_expl_out"] = expl

# Single-term results
st.write("**IRI:**", st.session_state.get("mapping_iri_out", "") or "—")
st.write("**SKOS:**", st.session_state.get("mapping_skos_out", "") or "—")
st.write("**Explanation:**")
st.code(st.session_state.get("mapping_expl_out", "") or "—", language="text")

single_payload = {
    "endpoint": endpoint_to_run or "",
    "term": st.session_state.get("mapping_term_input", ""),
    "definition": st.session_state.get("mapping_definition_input", ""),
    "iri": st.session_state.get("mapping_iri_out", ""),
    "skos": st.session_state.get("mapping_skos_out", ""),
    "explanation": st.session_state.get("mapping_expl_out", ""),
}
st.download_button(
    "Download single-term result (.json)",
    data=json.dumps(single_payload, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="mapping_result.json",
    mime="application/json",
    use_container_width=True,
)

st.divider()


# ============================================================
# Batch mapping
# ============================================================
st.subheader("Mapping (multiple terms)")

if multiple_terms and uploaded_file is not None:
    try:
        preview_df = pd.read_excel(uploaded_file)
        st.write("**Uploaded terms preview:**")
        st.dataframe(preview_df.head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
        st.stop()

run_batch_enabled = multiple_terms and (uploaded_file is not None) and (endpoint_to_run in {"Wikidata", "Bioportal", "Multiagent"})
if st.button("Run batch mapping", disabled=not run_batch_enabled, use_container_width=True):
    try:
        input_df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the Excel file: {e}")
        st.stop()

    required_cols = {"Term", "Definition"}
    missing = required_cols - set(input_df.columns)
    if missing:
        st.error(f"Missing required column(s): {', '.join(sorted(missing))}. Your file must have Term, Definition.")
        st.stop()

    input_df = input_df.copy()
    input_df["Term"] = input_df["Term"].astype(str).str.strip()
    input_df["Definition"] = input_df["Definition"].astype(str).str.strip()
    input_df = input_df[(input_df["Term"] != "") & (input_df["Term"].str.lower() != "nan")]

    if len(input_df) == 0:
        st.warning("No valid rows found (Term is empty).")
        st.stop()

    if endpoint_to_run in {"Bioportal", "Multiagent"}:
        if not os.environ.get("BIOPORTAL_API_KEY"):
            st.error("BIOPORTAL_API_KEY is required for BioPortal / Multiagent.")
            st.stop()
        if not term_ontologies:
            st.error("Please provide term_ontologies for BioPortal / Multiagent.")
            st.stop()

    if endpoint_to_run == "Wikidata":
        agent = _get_wiki_agent()
    elif endpoint_to_run == "Bioportal":
        agent = _get_bio_agent(trusted_ontologies, term_ontologies)
    else:
        agent = _get_multi_agent(trusted_ontologies, term_ontologies)

    results_rows = []
    progress = st.progress(0)
    status = st.empty()
    total = len(input_df)

    with st.spinner(f"Running {endpoint_to_run} agent for uploaded terms..."):
        for i, row in enumerate(input_df.itertuples(index=False), start=1):
            term = getattr(row, "Term")
            definition = getattr(row, "Definition")

            status.write(f"Processing {i}/{total}: **{term}**")
            progress.progress(int((i / total) * 100))

            if endpoint_to_run == "Wikidata":
                question = _question_wikidata(term, definition)
            elif endpoint_to_run == "Bioportal":
                question = _question_bioportal(term, definition, term_ontologies, trusted_ontologies)
            else:
                question = _question_multiagent(term, definition)

            try:
                result = agent.invoke({"messages": [{"role": "user", "content": question}]})
                raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
            except Exception as e:
                results_rows.append({
                    "Term": term,
                    "Definition": definition,
                    "Endpoint": endpoint_to_run,
                    "IRI": "",
                    "SKOS": "",
                    "explanation": f"ERROR: {e}",
                })
                continue

            parsed = _parse_agent_json(raw)

            if endpoint_to_run == "Wikidata":
                qid = (parsed.get("qid") or "").strip()
                skos = (parsed.get("skos") or "").strip()
                expl = (parsed.get("explanation") or "").strip()

                if qid and qid != "No wiki match":
                    iri = _wikidata_url(qid)
                else:
                    iri, skos, expl = "No wiki match", "", ""

            elif endpoint_to_run == "Bioportal":
                iri = (parsed.get("qid") or "").strip()
                skos = (parsed.get("skos") or "").strip()
                expl = (parsed.get("explanation") or "").strip()

                if not iri or iri == "No bioportal match":
                    iri, skos, expl = "No bioportal match", "", ""

            else:  # Multiagent
                # ✅ FIX: Multiagent output produced by agentmapping_format → keys: ID, SKOS, SKOS_explanation
                fields = _extract_multiagent_fields(parsed)
                iri = fields["iri"]
                skos = fields["skos"]
                expl = fields["explanation"]

                if not iri or iri.startswith("No "):
                    skos, expl = "", ""

            results_rows.append({
                "Term": term,
                "Definition": definition,
                "Endpoint": endpoint_to_run,
                "IRI": iri,
                "SKOS": skos,
                "explanation": expl,
            })

    df_out = pd.DataFrame(results_rows, columns=["Term", "Definition", "Endpoint", "IRI", "SKOS", "explanation"])
    df_out = _ensure_batch_schema(df_out)

    # Clear highlight (no "last reevaluation" yet)
    st.session_state["last_reeval_run_id"] = None
    st.session_state["mapping_batch_df"] = df_out


# ============================================================
# Batch results + re-evaluation (highlight last run only)
# ============================================================
batch_df = st.session_state.get("mapping_batch_df")
if isinstance(batch_df, pd.DataFrame) and len(batch_df) > 0:
    batch_df = _ensure_batch_schema(batch_df)
    st.session_state["mapping_batch_df"] = batch_df

    st.subheader("Batch results")

    # Highlight only rows updated in the last re-evaluation run
    last_run_id = st.session_state.get("last_reeval_run_id") or ""

    def _highlight_last_run(row):
        if last_run_id and str(row.get("last_updated_run", "")) == str(last_run_id):
            return ["background-color: #fff59d"] * len(row)
        return [""] * len(row)

    display_cols = ["Term", "IRI", "SKOS", "explanation"]
    # If a term was changed, show OriginalTerm too
    show_original = any(batch_df["OriginalTerm"].astype(str) != batch_df["Term"].astype(str))
    if show_original:
        display_cols = ["OriginalTerm"] + display_cols

    st.dataframe(
        batch_df[display_cols].style.apply(_highlight_last_run, axis=1),
        use_container_width=True,
    )

    st.subheader("Re-evaluate selected terms")

    selected_rowids: List[str] = []
    for _, r in batch_df.iterrows():
        label = f"{r['Term']}"
        if st.checkbox(label, key=f"recheck_{r['RowID']}"):
            selected_rowids.append(r["RowID"])

    if selected_rowids:
        st.markdown("**Optional:** Provide a new term for re-evaluation (leave unchanged to re-run the same term).")
    new_term_by_rowid: Dict[str, str] = {}
    for rowid in selected_rowids:
        current_term = str(batch_df.loc[batch_df["RowID"] == rowid, "Term"].iloc[0])
        new_term_by_rowid[rowid] = st.text_input(
            f"New term for: {current_term}",
            value=current_term,
            key=f"new_term_{rowid}",
        )

    if st.button("Re-evaluate selected", disabled=(len(selected_rowids) == 0), use_container_width=True):
        # Validate BioPortal config if needed for any selected rows
        endpoints_needed = set(batch_df.loc[batch_df["RowID"].isin(selected_rowids), "Endpoint"].astype(str).tolist())
        if ("Bioportal" in endpoints_needed) or ("Multiagent" in endpoints_needed):
            if not os.environ.get("BIOPORTAL_API_KEY"):
                st.error("BIOPORTAL_API_KEY is required for BioPortal / Multiagent re-evaluation.")
                st.stop()
            trusted_ontologies = _parse_csv_list(st.session_state.get("trusted_ontologies_input", ""))
            term_ontologies = _parse_csv_list(st.session_state.get("term_ontologies_input", ""))
            if not term_ontologies:
                st.error("Please provide term_ontologies for BioPortal / Multiagent re-evaluation.")
                st.stop()

        run_id = str(uuid.uuid4())
        st.session_state["last_reeval_run_id"] = run_id

        progress = st.progress(0)
        status = st.empty()
        total = len(selected_rowids)

        for i, rowid in enumerate(selected_rowids, start=1):
            row = batch_df.loc[batch_df["RowID"] == rowid].iloc[0]
            endpoint = str(row["Endpoint"])
            definition = str(row["Definition"])
            new_term = (new_term_by_rowid.get(rowid) or "").strip() or str(row["Term"])

            status.write(f"Re-evaluating {i}/{total}: **{new_term}** ({endpoint})")
            progress.progress(int((i / total) * 100))

            if endpoint == "Wikidata":
                agent = _get_wiki_agent()
                question = _question_wikidata(new_term, definition)

            elif endpoint == "Bioportal":
                trusted_ontologies = _parse_csv_list(st.session_state.get("trusted_ontologies_input", ""))
                term_ontologies = _parse_csv_list(st.session_state.get("term_ontologies_input", ""))
                agent = _get_bio_agent(trusted_ontologies, term_ontologies)
                question = _question_bioportal(new_term, definition, term_ontologies, trusted_ontologies)

            else:  # Multiagent
                trusted_ontologies = _parse_csv_list(st.session_state.get("trusted_ontologies_input", ""))
                term_ontologies = _parse_csv_list(st.session_state.get("term_ontologies_input", ""))
                agent = _get_multi_agent(trusted_ontologies, term_ontologies)
                question = _question_multiagent(new_term, definition)

            result = agent.invoke({"messages": [{"role": "user", "content": question}]})
            raw = result["messages"][-1].content if isinstance(result, dict) and "messages" in result else str(result)
            parsed = _parse_agent_json(raw)

            if endpoint == "Wikidata":
                qid = (parsed.get("qid") or "").strip()
                skos = (parsed.get("skos") or "").strip()
                expl = (parsed.get("explanation") or "").strip()

                if qid and qid != "No wiki match":
                    iri = _wikidata_url(qid)
                else:
                    iri, skos, expl = "No wiki match", "", ""

            elif endpoint == "Bioportal":
                iri = (parsed.get("qid") or "").strip()
                skos = (parsed.get("skos") or "").strip()
                expl = (parsed.get("explanation") or "").strip()

                if not iri or iri == "No bioportal match":
                    iri, skos, expl = "No bioportal match", "", ""

            else:  # Multiagent
                # ✅ FIX: Multiagent output produced by agentmapping_format → keys: ID, SKOS, SKOS_explanation
                fields = _extract_multiagent_fields(parsed)
                iri = fields["iri"]
                skos = fields["skos"]
                expl = fields["explanation"]

                if not iri or iri.startswith("No "):
                    skos, expl = "", ""

            # Update term (and preserve OriginalTerm)
            idx = batch_df.index[batch_df["RowID"] == rowid][0]
            if not batch_df.loc[idx, "OriginalTerm"]:
                batch_df.loc[idx, "OriginalTerm"] = batch_df.loc[idx, "Term"]
            if batch_df.loc[idx, "OriginalTerm"] == batch_df.loc[idx, "Term"] and new_term != batch_df.loc[idx, "Term"]:
                batch_df.loc[idx, "OriginalTerm"] = batch_df.loc[idx, "Term"]

            batch_df.loc[idx, "Term"] = new_term
            batch_df.loc[idx, "IRI"] = iri
            batch_df.loc[idx, "SKOS"] = skos
            batch_df.loc[idx, "explanation"] = expl
            batch_df.loc[idx, "last_updated_run"] = run_id

        status.write("✅ Re-evaluation complete. Updated rows are highlighted (only for this last run).")
        st.session_state["mapping_batch_df"] = batch_df
        st.rerun()

    # Batch download (requested 4 columns)
    output = BytesIO()
    export_df = batch_df[["Term", "IRI", "SKOS", "explanation"]].copy()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="mapping")
    output.seek(0)

    st.download_button(
        label="Download batch mapping results (.xlsx)",
        data=output,
        file_name="batch_mapping_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
