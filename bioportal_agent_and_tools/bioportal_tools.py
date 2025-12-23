# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 00:17:00 2025

@author: yurt3
"""
from typing import Tuple,List, Dict, Any, Optional
import requests
import os

BASE_URL = "https://data.bioontology.org"


def find_term_in_ontology(
    term: str,
    ontology: str,
    exact: bool = True,
    case_sensitive: bool = False
) -> Tuple[str, str]:
    """
    Function to search for a given term in a specified ontology without retrieving
    the definition.

    Searches a specified ontology in the BioPortal API for a given term and returns:
      - the mapped ID
      - the mapping type: "exact" or "synonym"

    If no matches are found and exact=True, the search is retried with exact=False.

    Parameters
    ----------
    term : str
        Search term.
    ontology : str
        Ontology acronym.
    exact : bool, optional
        If True, attempt exact prefLabel match first.
    case_sensitive : bool, optional
        If True, case-sensitive matching is used and no .lower() transformations occur.

    Returns
    -------
    Tuple[str, str]
        (mapped_id, match_type)
        Returns ("", "") if no match found.
    """

    params = {
        "q": term,
        "ontologies": ontology,
        "require_exact_match": str(exact).lower(),
        "include": "prefLabel,definition,synonym,notation,cui,semanticType",
        "pagesize": 20,
        "apikey": os.environ.get("BIOPORTAL_API_KEY", "").strip()
    }

    resp = requests.get(f"{BASE_URL}/search", params=params, timeout=15)
    resp.raise_for_status()
    entries = resp.json().get("collection", [])

    # Apply case sensitivity rule
    if case_sensitive:
        term_cmp = term               # keep original case
    else:
        term_cmp = term.lower()       # lowercase for case-insensitive match

    # --- Filter logic ---
    filtered = []

    if exact:
        for e in entries:
            label = e.get("prefLabel", "")
            label_cmp = label if case_sensitive else label.lower()
            if label_cmp == term_cmp:
                filtered.append(e)
        match_type = "exact"
    else:
        for e in entries:
            syns = e.get("synonym") or []
            if isinstance(syns, str):
                syns = [syns]

            for s in syns:
                syn_cmp = s if case_sensitive else s.lower()
                if syn_cmp == term_cmp:
                    filtered.append(e)
                    break
        match_type = "synonym"

    # If nothing found and this was an exact search → retry as synonym search
    if not filtered and exact:
        return find_term_in_ontology(
            term,
            ontology,
            exact=False,
            case_sensitive=case_sensitive
        )

    # Still nothing found
    if not filtered:
        return "", ""

    # Extract mapped ID
    first = filtered[0]
    mapped_id = first.get("@id", first.get("id", ""))

    return mapped_id, match_type

def _extract_definition(e: Dict[str, Any]) -> str:
    """
    Extract a definition string from a BioPortal entry.
    Returns the first non-empty definition if available, else empty string.
    """
    defs = e.get("definition") or []

    if isinstance(defs, list):
        for d in defs:
            if isinstance(d, str) and d.strip():
                return d.strip()
        return ""

    if isinstance(defs, str) and defs.strip():
        return defs.strip()

    return ""


def find_term_in_ontology_with_definition(
    term: str,
    ontology: str,
    exact: bool = True,
    case_sensitive: bool = False
) -> str:
    """
    Searches an ontology in BioPortal and returns a *single output string* with:
        mapped_id: ...
        matching_type: ...
        definition: ...

    If no match found -> returns an empty string.
    """

    params = {
        "q": term,
        "ontologies": ontology,
        "require_exact_match": str(exact).lower(),
        "include": "prefLabel,definition,synonym,notation,cui,semanticType",
        "pagesize": 20,
        "apikey": os.environ.get("BIOPORTAL_API_KEY", "").strip()
    }

    resp = requests.get(f"{BASE_URL}/search", params=params, timeout=15)
    resp.raise_for_status()
    entries = resp.json().get("collection", [])

    # Case sensitivity
    term_cmp = term if case_sensitive else term.lower()

    filtered: List[Dict[str, Any]] = []

    # ---------------------------
    # MATCHING LOGIC
    # ---------------------------
    if exact:
        for e in entries:
            label = e.get("prefLabel", "")
            label_cmp = label if case_sensitive else label.lower()
            if label_cmp == term_cmp:
                filtered.append(e)
        match_type = "exact"
    else:
        for e in entries:
            syns = e.get("synonym") or []
            if isinstance(syns, str):
                syns = [syns]
            for s in syns:
                syn_cmp = s if case_sensitive else s.lower()
                if syn_cmp == term_cmp:
                    filtered.append(e)
                    break
        match_type = "synonym"

    # Retry using synonym search if exact failed
    if not filtered and exact:
        return find_term_in_ontology_with_definition(
            term,
            ontology,
            exact=False,
            case_sensitive=case_sensitive
        )

    if not filtered:
        return ""  # no match at all

    # ---------------------------
    # EXTRACT FIELDS
    # ---------------------------
    first = filtered[0]
    mapped_id = first.get("@id", first.get("id", ""))
    definition = _extract_definition(first)

    # ---------------------------
    # FORMAT FINAL OUTPUT STRING
    # ---------------------------
    result = (
        f"mapped_id: {mapped_id}, "
        f"mapped_type: {match_type}, "
        f"definition: {definition}"
    )

    return result

def find_indirect_definition(term: str, ontology: str) -> Optional[Dict[str, str]]:
    """
    Look up mappings for (term, ontology) and try to pull a definition
    from the *mapped* term in another ontology.

    Returns
    -------
    None or {
        "definition": <definition text>,
        "iri": <mapped class IRI>,
        "source_onto": <ontology link of mapped class>
    }
    """

    # 1) Search for the term in the given ontology to get its mappings link
    search_params = {
        "q": term,
        "ontologies": ontology,
        "require_exact_match": "true",
        "pagesize": 1,
        "apikey": os.environ.get("BIOPORTAL_API_KEY", "").strip()
    }
    try:
        r = requests.get(f"{BASE_URL}/search", params=search_params, timeout=15)
        r.raise_for_status()
    except requests.RequestException:
        return None

    coll = r.json().get("collection", [])
    if not coll:
        return None

    mappings_url = coll[0].get("links", {}).get("mappings")
    if not mappings_url:
        return None

    # 2) Fetch the mapping records
    try:
        mresp = requests.get(
            mappings_url, params={"apikey": os.environ.get("BIOPORTAL_API_KEY", "").strip()}, timeout=15
        )
        mresp.raise_for_status()
    except requests.RequestException:
        return None

    mdata = mresp.json()
    records: List[Dict[str, Any]] = (
        mdata if isinstance(mdata, list) else mdata.get("collection", [])
    )

    # 3) Iterate over mappings and follow the target class's self link
    for rec in records:
        classes = rec.get("classes", [])
        if len(classes) < 2:
            continue

        target = classes[1]  # mapped-to class
        iri = target.get("@id", "")
        links = target.get("links", {}) or {}

        self_link = links.get("self")
        onto_link = links.get("ontology", "")

        if not self_link:
            continue

        # 4) Fetch the full class record to get its definition
        try:
            c = requests.get(
                self_link, params={"apikey": os.environ.get("BIOPORTAL_API_KEY", "").strip()}, timeout=15
            )
            c.raise_for_status()
        except requests.RequestException:
            continue

        entry = c.json()
        defs = entry.get("definition") or []

        text = None
        if isinstance(defs, list) and defs:
            for d in defs:
                if isinstance(d, str) and d.strip():
                    text = d.strip()
                    break
        elif isinstance(defs, str) and defs.strip():
            text = defs.strip()

        if text:
            return {
                "definition": text,
                "iri": iri,
                "source_onto": onto_link,  # full ontology URL, no guessing
            }

    # No mapped term with a definition found
    return None

def _parse_mapped_output(output: str) -> Dict[str, str]:
    """
    Parse the string from find_term_in_ontology_with_definition:
      "mapped_id: ..., mapped_type: ..., definition: ..."
    into a dict.
    """
    result: Dict[str, str] = {"mapped_id": "", "mapped_type": "", "definition": ""}

    if not output:
        return result

    parts = [p.strip() for p in output.split(",")]
    for part in parts:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key in result:
            result[key] = value

    return result


def find_best_definition(
    term: str,
    ontology: str,
    exact: bool = True,
    case_sensitive: bool = False
) -> Optional[Dict[str, str]]:
    """
    Function to search for a given term in a specified ontology with retrieving the
    definition information

    Combines direct and indirect definition lookup.

    Output schema:

    {
        "mapped_id": ...,
        "mapped_type": ...,
        "definition": ...,
        "definition_source": "original" | "indirect" | "",
        "definition_source_ontology": <ontology URL or empty>,
    }
    """

    direct_str = find_term_in_ontology_with_definition(
        term, ontology, exact=exact, case_sensitive=case_sensitive
    )

    if not direct_str:
        return None

    direct = _parse_mapped_output(direct_str)
    mapped_id = direct.get("mapped_id", "").strip()
    mapped_type = direct.get("mapped_type", "").strip()
    definition = direct.get("definition", "").strip()

    if not mapped_id:
        return None

    # -------------------------------------------------
    # CASE 1: Direct definition found
    # -------------------------------------------------
    if definition:
        return {
            "mapped_id": mapped_id,
            "mapped_type": f'{mapped_type}+Definition',
            "definition": definition,
            "definition_source": "original",
            "definition_source_ontology": mapped_id
        }

    # -------------------------------------------------
    # CASE 2: No direct definition → try indirect
    # -------------------------------------------------
    indirect = find_indirect_definition(term, ontology)

    if indirect and indirect.get("definition"):
        return {
            "mapped_id": mapped_id,
            "mapped_type": f'{mapped_type}+Definition',
            "definition": indirect["definition"],
            "definition_source": "indirect",
            "definition_source_ontology": indirect.get("iri", "")
        }

    # -------------------------------------------------
    # CASE 3: No definition found anywhere
    # -------------------------------------------------
    return {
        "mapped_id": mapped_id,
        "mapped_type": f'{mapped_type}+Unverified',
        "definition": "",
        "definition_source": "",
        "definition_source_ontology": ""
    }