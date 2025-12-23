# General Instructions

1. Download or clone this repository.
2. Install all required dependencies listed in **`environment.yml`**.
3. Run the application with:

   ```bash
   streamlit run Home.py
   ```
**Note**: One can run all codes directly in Colab without a pre-installation. See https://github.com/KIDA-BfR/Linked_Data

---

# Checking the MCP Option

As a demonstration of the MCP-based tool call (using the SKOS matching tool as an example), an alternative implementation is provided in:

- **`Verification_service_MCP.py`**

To try the MCP version:

1. Open `Verification_service_MCP.py`.
2. Copy its contents.
3. Replace the contents of `Verification_service.py` with it.
4. Run the application again using:

   ```bash
   streamlit run Home.py
   ```
---
# App Design 
Application provides an acces to the Mapping and verification services 
[Figure 0 - App design](https://github.com/KIDA-BfR/Linked_Data_mapping_application/blob/main/visuals/Application_design.png)

# Starter page

Once the application is run, the user sees the entry page shown below.

[Figure 1 – Application entry page](https://github.com/KIDA-BfR/Linked_Data_mapping_application/blob/main/visuals/Entry.PNG)

On this page, the user is prompted to provide an **OpenAI API key** and, optionally, a **LangSmith API key** for application tracing.

To obtain a LangSmith API key, see:  https://smith.langchain.com/

If a LangSmith API key is provided, a project named **`KIDA_data`** is initialized in LangSmith using the endpoint:

```
https://eu.api.smith.langchain.com
```

---

## Verification Service

The **verification service** allows the user to verify a provided mapping using verification tools from `skos_tools.py`.

The verification can be executed:
- either via **direct function calls**, or
- alternatively, via the **MCP-based implementation** (see the corresponding section for details).

[Figure 2 – Verification service interface](https://github.com/KIDA-BfR/Linked_Data_mapping_application/blob/main/visuals/Verification.PNG)

---

## Mapping Service

The **mapping service** provides the following options:

- Mapping a **single term** to:
  - Wikidata labels, or
  - BioPortal labels  
  using a **single specialized deep agent**
- Mapping simultaneously to **both Wikidata and BioPortal** using a **multi-agent system**

[Figure 3 – Mapping service options](https://github.com/KIDA-BfR/Linked_Data_mapping_application/blob/main/visuals/Mapping_single.PNG)

---

## Batch Mapping via Tables

The user can also upload **tables** to map multiple terms at once.

Example input files are provided in the **`auxiliary_files`** [folder](https://github.com/KIDA-BfR/Linked_Data_mapping_application/tree/main/auxiliary_files)

[Figure 4 – Batch mapping using tables](https://github.com/KIDA-BfR/Linked_Data_mapping_application/blob/main/visuals/Mapping_multiple.PNG)

---
