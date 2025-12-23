# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:49:02 2025

@author: yurt3
"""
import pandas as pd
from io import StringIO
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import os

from pathlib import Path

file_path = Path.cwd() / "auxiliary_files" / "autoreconcilitation_training_terms_20251203_formatted.xlsx"
df = pd.read_excel(file_path) 

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

class SKOSMatch(BaseModel):
    """SKOS-style semantic relationship between two concepts."""

    exact_match: bool = Field(
        default=None,
        description=(
            f"""True if the two concepts can be used interchangeably across schemes.
They denote the same real-world concept, even if the wording differs.
This is symmetric and transitive.

EXACT MATCH EXAMPLES:
{exact_text}
"""
        ),
    )

    close_match: bool = Field(
        default=None,
        description=(
            f"""True if the two concepts are very similar and usually substitutable in most contexts,
but not strictly equivalent. Not transitive.

CLOSE MATCH EXAMPLES:
{close_text}
"""
        ),
    )

    related_match: bool = Field(
        default=None,
        description=(
            f"""True if the two concepts are associated but not synonymous.
Represents a non-hierarchical 'see also' relation.

RELATED MATCH EXAMPLES:
{related_text}
"""
        ),
    )

    explanation: Optional[str] = Field(
        default=None,
        description=(
            """Short explanation of why the chosen semantic relationship holds,
based on comparing terms and definitions."""
        ),
    )


# agent = create_agent(
#     model="gpt-5.1",
#     response_format=SKOSMatch,
# )

def _get_structured_llm():
    # IMPORTANT: read key from env; if missing, fail with clear message
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (expected env var).")

    llm_skos = ChatOpenAI(model="gpt-5.1", temperature=0)
    return llm_skos.with_structured_output(SKOSMatch)



def classify_skos_match(term_a: str, gen_def: str, term_b: str, onto_def: str):
    """
    Function to classify semantic relationships between two concepts into the
    SKOS concept: exact, close and related. The output is the matching type and
    the explanation
    """
    prompt=f"""
        You are comparing semantic similarities between two concepts. Each concept
        is represented by a term and its definition. Provide a concise summary
        explaining the semantic relationship between the concepts.

        Concept A:\n
          Term: {term_a}
          Definition (generated): {gen_def}
        Concept B:
          Term: {term_b}
          Definition (ontology): {onto_def}

      """

    messages = [
        HumanMessage(content=prompt),
    ]
    structured_llm_skos = _get_structured_llm()

    data: SKOSMatch = structured_llm_skos.invoke(messages)


     # Decide mapping_type with priority: exact > close > related
    if data.exact_match:
        mapping_type = "exact"
    elif data.close_match:
        mapping_type = "close"
    elif data.related_match:
        mapping_type = "related"
    else:
        mapping_type = "none"

    return {
        "mapping_type": mapping_type,
        "explanation": data.explanation or ""
    }

#### Tool for the formatting, fits better for the orchestrating agent compared 
# to structured output

class Agentmapping(BaseModel):
    """Wkidata mapping output"""
    id: str = Field(description="Either Q-id or iri")
    skos: str = Field(description="SKOS_matching class: exact, related or close ")
    explanation: str = Field(description="SKOS_matching logic: rationale of the SKOS_matchcing class" )

llm_format = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
)
structured_llm_format = llm_format.with_structured_output(Agentmapping)

def agentmapping_format(output: str):
    """
    Function to structure the agent output into
    id, skos and explanation
    """
    prompt=f"""
        You are provided with the following  output of agent performing the mapping.

        {output}

        From this text retrieve the following information:
        ID: for wikidata the Q-id number (e.g. Q159). For other sources the mapped id iri http://www.ncbi.nlm.nih.gov/gene/18125
        SKOS: SKOS_matching. SKOS matching between original term definition and the definition/description of the identified label from wikidata. Example: exact
        Explanation: The explanation for SKOS matching logic retrieved from explanation field of SKOS matching tool
      """

    messages = [
        HumanMessage(content=prompt),
    ]

    data: SKOSMatch = structured_llm_format.invoke(messages)


    return {
        "ID": data.id,
        "SKOS": data.skos,
        "SKOS_explanation": data.explanation or ""
    }