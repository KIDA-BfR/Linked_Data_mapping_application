# 🧠 Deep Agent for Wikidata — ESWC Final Version

The notebook **`Deep_Agent_wikidata_ESWC_final.ipynb`** contains **all code components** used to generate the mapping runs whose traces are listed in:

📄 **`Links_to_all_run_traces.txt`**

---

## ⚙️ Execution Environment

This project is designed to run in **Google Colab**.  
The notebook includes cells that:

- 📦 Install all required Python packages  
- 🔧 Import all necessary modules  
- 🧹 Set up configuration for the agent and tools  

---

## 📁 Required Input Files

Before running the notebook, the user must **upload all files** from the `Files/` directory.

These include:

### 📘 `autoreconcilitation_training_terms_20251203_formatted.xlsx`
- Contains expert-generated examples  
- Used to train the **SKOS matching** component (exact / close / related)  
- Provides definitions and relationships for few-shot prompting

### 📙 `wikidata_properties.json`
- A dictionary of **Wikidata property IDs (P-ids)**  
- Snapshot generated on **03.12.2025**  
- Used for interpreting Wikidata claims in a readable form

### 📑 `Terms_wiki_test_agent_run_{1-4}.xlsx`
- Outputs of individual **agent runs**  
- Each run contains:
  - Mapped Q-IDs  
  - SKOS mapping decisions  
  - Explanations  
- Trace logs for these runs are listed in `Links_to_all_run_traces.txt`

### 📊 `Terms_wiki_test_agent_combined.xlsx`
- A **merged** result file combining outputs from all four runs  
- Useful for aggregated inspection and evaluation

---

## 🗂️ Summary Diagram

```
                ┌───────────────────────────┐
                │    Input Files (Files/)    │
                ├───────────────────────────┤
                │  training_terms.xlsx       │
                │  wikidata_properties.json  │
                │  Terms_run_1–4.xlsx        │
                │  combined_results.xlsx     │
                └──────────────┬────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │ Deep_Agent_wikidata_ESWC_2026 │
              │        (Google Colab)          │
              └────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │   Agent Runs + SKOS Matching   │
              └────────────────────────────────┘
                               │
                               ▼
                  📄 Links_to_all_run_traces.txt
```

---

## 🚀 Getting Started

1. Open the notebook in **Google Colab**  
2. Upload all required files from the `Files/` directory  
3. Run all notebook cells in order  
4. Inspect results using the provided UI for selecting run outputs  

