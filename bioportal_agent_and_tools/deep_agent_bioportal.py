# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 18:22:58 2025

@author: yurt3
"""

from pydantic import BaseModel, Field
from deepagents import create_deep_agent
# from wikidata_tools import WikidataEntitySearch, WikidataEntityDetails 
# from skos_tools import classify_skos_match

from bioportal_agent_and_tools.bioportal_tools import find_best_definition, find_term_in_ontology
from general_tools.skos_tools import classify_skos_match

import pandas as pd
from pathlib import Path
from io import StringIO
import os

from langchain.chat_models import init_chat_model
model = init_chat_model(model="gpt-5.1")

file_path = Path.cwd() / "auxiliary_files" / "autoreconcilitation_training_terms_20251203_formatted.xlsx"
df = pd.read_excel(file_path) 

#trusted_ontologies=['MESH', 'NCIT', 'LOINC', 'FOODON', 'NCBITAXON']
#term_ontologies =["NCIT","NIFSTD","BERO","OCHV","SNOMEDCT"] # for Independent variable list

#df=pd.read_excel("/content/autoreconcilitation_training_terms_20251203_formatted.xlsx")

def build_match_pairs(
    df: pd.DataFrame,
    match_name: str,
    label_col: str,
    desc_col: str,
    term_col: str = "term",
    def_col: str = "definition",
):
    """
    Build plain-text pairs for a given SKOS match type.
    Returns a full text block as a string instead of printing.
    """
    buffer = StringIO()
    buffer.write(f"========== {match_name} pairs ==========\n\n")

    subset = df.dropna(subset=[label_col])

    for n, row in enumerate(subset.itertuples(index=False), start=1):
        term_a = getattr(row, term_col)
        def_a  = getattr(row, def_col)
        term_b = getattr(row, label_col)
        def_b  = getattr(row, desc_col)

        buffer.write(f"{n}) Term A: {term_a}\n")
        buffer.write(f"   Definition A: {def_a}\n")
        buffer.write(f"   Term B: {term_b}\n")
        buffer.write(f"   Definition B: {def_b}\n\n")

    return buffer.getvalue()

exact_text = build_match_pairs(
    df,
    match_name="exactMatch",
    label_col="exactMatch_label",
    desc_col="exactMatch_description"
)

close_text = build_match_pairs(
    df,
    match_name="closeMatch",
    label_col="closeMatch_label",
    desc_col="closeMatch_description"
)

related_text = build_match_pairs(
    df,
    match_name="relatedMatch",
    label_col="relatedMatch_label",
    desc_col="relatedMatch_description"
)



# research_instructions_onto = f"""You task is to match the term with valid identifiers from bioportal by checking the following ontologies {term_ontologies}

# For the trusted ontologies such as {trusted_ontologies} you use the find_term_in_ontology tool to find the matches and do not need to check the definitions.

# If the ontology is not in the list of trusted, then use find_best_definition tool to get the term with its definition. Then compare the retrieved definition and the provided one.

# If these definitions match, then return the found identifier. If not, continue the search among other identifiers.

# The definitions do not to match exactly, but should be in one of these broad categories

# Exact matching: The two concepts can be used interchangeably across schemes.They denote the same real-world concept, even if the wording differs.
# {exact_text}

# Close matching: The two concepts are very similar and usually substitutable in most contexts, but not strictly equivalent.
# {close_text}

# Related matching: The two concepts are associated but not synonymous. Represents a non-hierarchical 'see also' relation.
# {related_text}


# Keep track on what identifiers you tried to avoid repetitive tries"""



class Bioportalmapping(BaseModel):
    """Bioportal mapping output structure, similar as for Wikidata"""
    qid: str = Field(description="IRI or ID of the label. Examples:  http://www.ncbi.nlm.nih.gov/gene/18125, http://purl.bioontology.org/ontology/SNOMEDCT/387390002 ")
    skos: str = Field(description="SKOS_matching. SKOS matching between original term definition and the definition/description of the identified label from bioportal. Example: exact")
    explanation: str = Field(description="SKOS_matching_logic. The explanation for SKOS matching logic retrieved from explanation field of SKOS matching tool" )

def get_agent_bioportal(trusted_ontologies: list[str], term_ontologies: list[str]):
    # Ensure the key is available in env (set by Streamlit Home page)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (expected env var).")
    if not os.environ.get("BIOPORTAL_API_KEY"):
        raise RuntimeError("BIOPORTAL_API_KEY is not set.")

    research_instructions_onto = f"""You task is to match the term with valid identifiers from bioportal by checking the following ontologies {term_ontologies}

    For the trusted ontologies such as {trusted_ontologies} you use the find_term_in_ontology tool to find the matches and do not need to check the definitions.

    If the ontology is not in the list of trusted, then use find_best_definition tool to get the term with its definition. Then compare the retrieved definition and the provided one.

    If these definitions match, then return the found identifier. If not, continue the search among other identifiers.

    The definitions do not to match exactly, but should be in one of these broad categories

    Exact matching: The two concepts can be used interchangeably across schemes.They denote the same real-world concept, even if the wording differs.
    {exact_text}

    Close matching: The two concepts are very similar and usually substitutable in most contexts, but not strictly equivalent.
    {close_text}

    Related matching: The two concepts are associated but not synonymous. Represents a non-hierarchical 'see also' relation.
    {related_text}


    Keep track on what identifiers you tried to avoid repetitive tries"""

    return create_deep_agent(
        model=model,
        tools=[find_best_definition, find_term_in_ontology, classify_skos_match],
        system_prompt=research_instructions_onto,
        response_format=Bioportalmapping,
    )
# agent_wiki = create_deep_agent(model="gpt-5.1",
#     tools=[WikidataEntitySearch, WikidataEntityDetails, classify_skos_match],
#     system_prompt=research_instructions,
#     response_format=Wikimapping
# )