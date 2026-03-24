# Multi-Agent Wikidata & BioPortal Mapping Pipeline

### Overview

This repository contains a multi-agent system for automatically mapping
domain-specific terminology to structured knowledge identifiers from
**Wikidata** and **BioPortal ontologies**.\
The notebook integrates retrieval, ontology search, definition
extraction, reasoning, and identifier matching using Deep Agent
Langchain framework

The system is intended to support:

-   Semantic data integration
-   Terminology harmonization
-   Ontology-aware knowledge graph construction
-   Metadata enrichment for scientific and industrial datasets

------------------------------------------------------------------------

## 🚀 Features

### **1. Multi-Agent Architecture**

The pipeline is built using **DeepAgents**, **LangGraph**, and
**LangChain**, orchestrating multiple collaborating agents:

-   **Ontology Agent**
    Maps terms to the most appropriate Ontologies from the list suggested by Bioportal_recommender pipeline (see separate folder)

-   **Bioportal Agent**
    Maps terms to the most appropriate Wikidata QID 

-   **Orchestrating Agent**
    Formulate tasks for Ontology and Bioportal agents, integrate the results, provides SKOS matching.


------------------------------------------------------------------------

## 📘 BioPortal Integration

The notebook includes a requester for interacting with the
BioPortal REST API. Capabilities include:

-   Querying `/search` endpoint for preferred labels and synonyms\
-   Extracting:
    -   `prefLabel`
    -   `synonym`
    -   `definition`
-   Handling missing definitions through a **two-step definition
    resolution pipeline**:
    1.  **Direct definition lookup** for exact or synonym-based matches\
    2.  **Indirect lookup** via related ontology concepts when
        definitions are absent


------------------------------------------------------------------------

## 🗂 Wikidata Mapping System

The multi-agent system supports:

-   Wikidata QID retrieval
-   Definition formation based on QID
-   Deciding on the best fitting label, if any.

### Timeout Protection

All agent calls are wrapped in a timeout wrapper:

``` python
run_with_timeout(func, timeout, *args)
```

This ensures that long-running LLM calls do not halt batch execution.

------------------------------------------------------------------------

## 📊 Excel-Based Batch Processing

The notebook operates on a user-provided Excel file containing two
sheets:

### **Expected Input Sheets**
| Sheet Name          | Required Columns                 | Purpose                                                                 |
|---------------------|----------------------------------|-------------------------------------------------------------------------|
| **Mappings**        | `Term`, `Mapped ID`, `SKOS_matching` | Master list of terms to be processed                                     |
| **Used definitions** | `Term`, `Definition`               | Definitions generated using a single-shot call (see openFSMR definition procedure folder) |


### **Processing Logic**

The main batch function executes the pipeline:

1.  Load Excel sheets
2.  Build dictionary: **term → definition**
3.  Identify terms that:
    -   Have no mapping
    -   Or have a placeholder value `"No wiki match"`
4.  For each such term:
    -   Retrieve definition
    -   Query the multi-agent pipeline
    -   Parse JSON or text responses
    -   Record:
        -   Mapped ID
        -   SKOS matching type
        -   Explanation text
5.  Save updated sheets to Excel after each processed row

Terms with missing definitions are skipped 

------------------------------------------------------------------------

## ▶️ How to Use

1.  Prepare an Excel file containing:

    -   A **Mappings** sheet with a list of terms\
    -   A **Used definitions** sheet with definitions (optional but
        recommended)

2.  Upload the Excel file in your notebook session.

3.  Run all notebook cells 

4.  Execute (wikidata here is only the name and  does not mean that the mapping is to wikidata only):

``` python
batch_run_wikidata_match("input.xlsx", "output.xlsx")
```

5.  Review the generated output Excel file, which will now contain:
    -   Mapped ID
    -   SKOS match semantics
    -   SKOS Explanations

------------------------------------------------------------------------

## 📚 Output Specification

The updated **Mappings** sheet will contain:

| Column Name        | Description                         |
|--------------------|-------------------------------------|
| **Term**           | Original term                       |
| **Mapped ID**      | Wikidata QID or ontology URL      |
| **SKOS_matching**  | Chosen SKOS semantic match          |
| **SKOS_explanation** | Agent-generated justification       |

