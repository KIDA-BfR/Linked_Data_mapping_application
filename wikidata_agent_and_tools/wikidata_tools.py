# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:44:34 2025

@author: yurt3
"""

from typing import Iterable, List, Dict, Any, Optional, Set
#from utils import load_wikidata_property_labels
from wikidata_agent_and_tools.utils import load_wikidata_property_labels

import re
import datetime
import requests

# Wikidata API base URL
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": "MyReActAgent/0.1 Linked_data"
}

PROPERTY_LABELS=load_wikidata_property_labels()

def _extract_time_string(wikidata_time: str) -> str:
    """
    Convert Wikidata time format '+1955-07-26T00:00:00Z' to '1955-07-26'.
    Falls back to the original string if parsing fails.
    """
    try:
        if wikidata_time.startswith("+") or wikidata_time.startswith("-"):
            wikidata_time = wikidata_time[1:]
        dt = datetime.datetime.fromisoformat(wikidata_time.replace("Z", "+00:00"))
        return dt.date().isoformat()
    except Exception:
        return wikidata_time


def _collect_referenced_item_ids(entities: Dict[str, Any]) -> List[str]:
    """
    From an entities dict (wbgetentities result), collect referenced item IDs (Qxxx)
    from the subset of properties we care about.
    """
    referenced_ids = set()
    for entity_id, entity in entities.items():
        claims = entity.get("claims", {})
        for pid in PROPERTY_LABELS.keys():
            for claim in claims.get(pid, []):
                mainsnak = claim.get("mainsnak", {})
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value")
                if (
                    isinstance(value, dict)
                    and value.get("entity-type") == "item"
                    and "id" in value
                ):
                    referenced_ids.add(value["id"])
    # remove the original entity ids; we only need labels for referenced ones
    referenced_ids -= set(entities.keys())
    return list(referenced_ids)


def _get_entities(ids: Iterable[str], language: str = "en") -> Dict[str, Any]:
    """
    Helper to call wbgetentities for a list of Q-ids and return the 'entities' map.
    """
    ids_list = list(ids)
    if not ids_list:
        return {}

    params = {
        "action": "wbgetentities",
        "ids": "|".join(ids_list),
        "format": "json",
        "languages": language,
        "props": "labels|descriptions|claims",
    }
    response = requests.get(
        WIKIDATA_API_URL,
        params=params,
        headers=HEADERS,  # <-- important for avoiding 403
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("entities", {})


def _get_entity_labels(ids: Iterable[str], language: str = "en") -> Dict[str, str]:
    """
    Get labels for item IDs (e.g. Q5, Q30) in the given language.
    """
    ids_list = list(ids)
    if not ids_list:
        return {}

    params = {
        "action": "wbgetentities",
        "ids": "|".join(ids_list),
        "format": "json",
        "languages": language,
        "props": "labels",
    }
    response = requests.get(
        WIKIDATA_API_URL,
        params=params,
        headers=HEADERS,  # <-- important for avoiding 403
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    entities = data.get("entities", {})

    labels = {}
    for eid, entity in entities.items():
        label_obj = entity.get("labels", {}).get(language)
        if label_obj:
            labels[eid] = label_obj.get("value")
    return labels

def get_wikidata_definition(
    entity_id: str,
    language: str = "en",
) -> Optional[Dict[str, Any]]:
    """
    Tool: Given a single Wikidata entity ID (e.g. 'Q42'),
    construct an enriched definition for that term using Wikidata entity data.

    Returns:
        A dict like:
        {
          "id": "Q42",
          "label": "...",
          "description": "...",
          "definition": "...",
          "url": "https://www.wikidata.org/wiki/Q42",
          "facts": { ... }
        }
        or None if the entity cannot be retrieved.
    """
    if not entity_id:
        return None

    # 1) Get entity data for this single ID
    entities = _get_entities([entity_id], language=language)
    if entity_id not in entities:
        # Nothing found for this ID
        return None

    entity = entities[entity_id]

    # 2) Collect referenced item ids for properties of interest, to get nicer labels
    referenced_item_ids = _collect_referenced_item_ids({entity_id: entity})
    referenced_labels = _get_entity_labels(referenced_item_ids, language=language)

    def _value_to_string(datavalue: Dict[str, Any]) -> str:
        """
        Convert a Wikidata datavalue into a human-readable string, using
        referenced_labels when possible.
        """
        if not datavalue:
            return ""

        value = datavalue.get("value")
        vtype = datavalue.get("type")

        if vtype == "wikibase-entityid" and isinstance(value, dict):
            eid = value.get("id")
            return referenced_labels.get(eid, eid or "")

        if vtype == "time" and isinstance(value, dict):
            return _extract_time_string(value.get("time", ""))

        if vtype == "globecoordinate" and isinstance(value, dict):
            lat = value.get("latitude")
            lon = value.get("longitude")
            if lat is not None and lon is not None:
                return f"{lat}, {lon}"
            return str(value)

        # Fallback: just stringify
        return str(value)

    labels = entity.get("labels", {})
    descriptions = entity.get("descriptions", {})
    claims = entity.get("claims", {})

    label = labels.get(language, {}).get("value") or entity_id
    description = descriptions.get(language, {}).get("value")

    # Build structured "facts" from selected properties
    facts: Dict[str, List[str]] = {}

    for pid, human_label in PROPERTY_LABELS.items():
        prop_claims = claims.get(pid, [])
        values: List[str] = []
        for claim in prop_claims:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue")
            if datavalue:
                s = _value_to_string(datavalue)
                if s:
                    values.append(s)
        if values:
            facts[human_label] = values

    # Construct definition text
    definition_parts: List[str] = []
    if description:
        definition_parts.append(description.rstrip("."))

    if facts:
        fact_strings = []
        for human_label, values in facts.items():
            joined_vals = ", ".join(values)
            fact_strings.append(f"{human_label}: {joined_vals}")
        definition_parts.append("Key facts: " + "; ".join(fact_strings))

    if not definition_parts:
        definition_parts.append(
            f"{label} (no textual description available in Wikidata)."
        )

    definition = ". ".join(definition_parts).strip()
    if not definition.endswith("."):
        definition += "."

    return {
        "id": entity_id,
        "label": label,
        "description": description,
        "definition": definition,
        "url": f"https://www.wikidata.org/wiki/{entity_id}",
        "facts": facts,
    }

def resolve_qids_and_pids_in_definition(
    enriched_result: Dict[str, Any],
    language: str = "en",
) -> Dict[str, Any]:
    """
    Given a single enriched entity from `get_wikidata_definition`,
    replace:
      - Q-IDs (e.g. 'Q183') with their Wikidata labels
      - P-IDs (e.g. 'P2076') with their property labels from PROPERTY_LABELS

    Replacement is applied to:
      - the 'definition' string
      - all string values inside the 'facts' dict

    Args:
        enriched_result: output of get_wikidata_definition (a dict for one entity)
        language: label language to use when resolving Q-IDs

    Returns:
        A NEW dict with Q- and P-IDs replaced by labels.
        (Original dict is not mutated.)
    """
    if not enriched_result:
        return enriched_result

    qid_pattern = re.compile(r"\bQ\d+\b")
    pid_pattern = re.compile(r"\bP\d+\b")

    # 1) Collect all Q-IDs that appear in definition or facts
    all_qids: Set[str] = set()

    # From definition text
    definition = enriched_result.get("definition") or ""
    all_qids.update(qid_pattern.findall(definition))

    # From facts (only string values)
    facts = enriched_result.get("facts") or {}
    if isinstance(facts, dict):
        for vals in facts.values():
            for v in vals:
                if isinstance(v, str):
                    all_qids.update(qid_pattern.findall(v))

    # 2) Resolve Q-IDs -> labels using Wikidata (in chunks, though usually few)
    entity_labels: Dict[str, str] = {}
    if all_qids:
        qid_list = list(all_qids)
        chunk_size = 50

        for i in range(0, len(qid_list), chunk_size):
            chunk = qid_list[i : i + chunk_size]
            chunk_labels = _get_entity_labels(chunk, language=language)
            entity_labels.update(chunk_labels)

    # 3) Helper to replace Q-IDs and P-IDs in any string
    def _replace_ids_in_text(text: str) -> str:
        # First replace Q-IDs with entity labels
        text = qid_pattern.sub(
            lambda m: entity_labels.get(m.group(0), m.group(0)),
            text,
        )
        # Then replace P-IDs with property labels from PROPERTY_LABELS
        text = pid_pattern.sub(
            lambda m: PROPERTY_LABELS.get(m.group(0), m.group(0)),
            text,
        )
        return text

    # 4) Build a new dict with replaced IDs
    new_item = dict(enriched_result)  # shallow copy

    # Replace in definition
    definition = enriched_result.get("definition")
    if isinstance(definition, str):
        new_item["definition"] = _replace_ids_in_text(definition)

    # Replace inside facts strings
    facts = enriched_result.get("facts")
    if isinstance(facts, dict):
        new_facts: Dict[str, list] = {}
        for key, vals in facts.items():
            new_vals: list = []
            for v in vals:
                if isinstance(v, str):
                    new_vals.append(_replace_ids_in_text(v))
                else:
                    new_vals.append(v)
            new_facts[key] = new_vals
        new_item["facts"] = new_facts

    return new_item

def WikidataEntityDetails (q: str):
     """
     Fetch full Wikidata details for a given entity (e.g. 'Q159').
     Input should be Q-ID only, and output is the JSON of that entity
     """

     raw = get_wikidata_definition(q) # Get definition containing Q-ids and P-ids
     resolved = resolve_qids_and_pids_in_definition(raw) # Enrich the definition
     return resolved


def get_nested_value(o: dict, path: list) -> any:
    """
    Safely walk through nested dicts and lists by keys/indexes.
    Returns None if any KeyError, IndexError, or TypeError occurs.
    """
    current = o
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return None
    return current


def WikidataEntitySearch(
    search: str,
    entity_type: str = "item",
    url: str = "https://www.wikidata.org/w/api.php",
    user_agent_header: str = 'DeepWikidataMapper/0.1',
    srqiprofile: str = None,
) -> Optional[str]:
    """
    Search Wikidata entities for a given query.

    Args:
        search: Text to search for (e.g. "Berlin").
        entity_type: Type of entity to search ('item' or 'property').
        url: Wikidata API URL.
        user_agent_header: User-Agent string for requests.
        srqiprofile: Search profile for Wikidata API.

    Returns:
        The Q-ID or P-ID of the top result, or an error message if not found.
    """
    headers = {"Accept": "application/json"}
    if user_agent_header is not None:
        headers["User-Agent"] = user_agent_header

    if entity_type == "item":
        srnamespace = 0
        srqiprofile = "classic_noboostlinks" if srqiprofile is None else srqiprofile
    elif entity_type == "property":
        srnamespace = 120
        srqiprofile = "classic" if srqiprofile is None else srqiprofile
    else:
        raise ValueError("entity_type must be either 'property' or 'item'")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": search,
        "srnamespace": srnamespace,
        "srlimit": 1,
        "srqiprofile": srqiprofile,
        "srwhat": "text",
        "format": "json",
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        title = get_nested_value(response.json(), ["query", "search", 0, "title"])
        if title is None:
            return f"I couldn't find any {entity_type} for '{search}'. Please rephrase your request and try again"
        # if there is a prefix, strip it off
        return title.split(":")[-1]
    else:
        return "Sorry, I got an error. Please try again."
