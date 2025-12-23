# Full streamlit app 

Download the repository, install all required dependencies (see **environment.yml**) and run Home.py with `strreamlit run Home.py`

# Mapping Service (Streamlit) — README

This Streamlit page provides a “Mapping service” UI that maps **(Term + Definition)** to identifiers in:

- **Wikidata** (QIDs → Wikidata URLs)
- **BioPortal** (ontology term IRIs)
- **Multiagent (BioPortal → Wikidata)** when both endpoints are selected

It supports **single-term mapping** and **batch mapping from Excel**, plus **re-evaluation of selected terms**.

---

## 1) High-level flow

1. User selects an **endpoint**:
   - **Wikidata** → run Wikidata agent only
   - **Bioportal** → run BioPortal agent only
   - **Wikidata + Bioportal** → automatically use **Multiagent** (tries BioPortal, then Wikidata)

2. User chooses mapping mode:
   - **Single term**: enter `Searched Term` and `Term definition`
   - **Multiple terms**: upload an Excel file with columns: `Term`, `Definition`

3. The selected agent is invoked with a prompt created from the provided term and definition.

4. Results are displayed in Streamlit and can be downloaded:
   - Single-term → JSON download
   - Batch → Excel download

---

## 2) API keys and session behavior

### OpenAI
The mapping service requires `OPENAI_API_KEY` (set on the Home page).  
If it is missing, the mapping page stops with an error.

### BioPortal
When **BioPortal** or **Multiagent** is used, the UI asks for:

- `BIOPORTAL_API_KEY`
- `trusted_ontologies` (comma-separated list)
- `term_ontologies` (comma-separated list)

These values are stored **only for the current Streamlit session** and also exported to environment variables:

- `BIOPORTAL_API_KEY`
- `BIOPORTAL_TRUSTED_ONTOLOGIES`
- `BIOPORTAL_TERM_ONTOLOGIES`

---

## 3) Ontology lists and caching

When BioPortal or Multiagent is active, the ontology lists control which ontologies are searched.

The app caches agents with `st.cache_resource`, but the cache keys include:

- OpenAI key
- LangSmith key (if present)
- BioPortal key (if present)
- `trusted_ontologies`
- `term_ontologies`

So if ontology lists change in the UI, the cached agent **rebuilds automatically**.

---

## 4) Output formats

### Agent output
The Agent is expected to return JSON like:
```json
{"qid":"Q123","skos":"exact","explanation":"..."}
