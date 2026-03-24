# Context-Based Definition Generation Workflow

To create context-based definitions for the mapped terms  
(see **Fig_2_dataset_and_workflow/Fig_2_Used_Raw_Data/Mapped_terms**),  
the following procedure was used.  
All resulting definitions are stored in **Terms_with_definitions.xlsx** and **Terms_with_definitions.csv**.

---

## ✔️ Step 1 — LightRAG-Based Definition Construction

A **LightRAG** (https://arxiv.org/abs/2410.05779) approach was used to build a local database and then query it to generate definitions.  
(See code and intermediate outputs in:  
`Fig_2_dataset_and_workflow/Deep_Agent/RAG_based_definition/Light_RAG_approach`)

### 🔍 1.1 Direct Database-Based Definitions  
The querying LLM (**GPT-5-nano**) was explicitly instructed to generate each term’s definition **only from the database knowledge**.  
- Terms successfully defined in this step are labeled **“Light RAG”** in the *Definition Type* column.  
- ✔️ Indicates the definition is grounded fully in the constructed database.

### 🔄 1.2 Definitions via Database Context + Few-Shot Examples  
For terms **not present** in the database and therefore not mapped in Step 1.1:  
- The database was provided as **context**, and  
- The LLM was asked to generate definitions using **its general knowledge** plus **a few examples** from Step 1.1.  
- Terms defined in this step are labeled **“Few-Shot Call GPT-5-nano”**.

📝 This hybrid method allows the model to generalize while remaining anchored to the available context.

---

## ✔️ Step 2 — Single-Shot GPT-5.1 Thinking Call for Remaining Terms

For all terms **not defined** in Step 1, a single-shot query was made to  
**GPT-5.1 (thinking mode)** to generate the missing definitions.  
(See implementation:  
`Fig_2_dataset_and_workflow/Deep_Agent/RAG_based_definition/General_GPT_5_1_Thinking_call`)

✨ These definitions serve as a fallback mechanism when the RAG-based pipeline cannot map a term.

---

## 📘 Summary of Definition Types

| Definition Type Label             | Source Method                                     | Icon |
|----------------------------------|---------------------------------------------------|:----:|
| **Light RAG**                    | Direct database-grounded definition               | ✅   |
| **Few-Shot Call GPT-5-nano**     | Definition using database context + example set   | 🔄   |
| **GPT-5.1 Thinking (Single Shot)** | General LLM knowledge without RAG support         | ✨   |

---

This workflow ensures that each mapped term receives a high-quality definition, starting from the most context-grounded approach (LightRAG) and progressively expanding to more general LLM reasoning when needed.
