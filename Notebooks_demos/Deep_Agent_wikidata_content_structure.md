# **Deep_Agent_wikidata** folder structure

Contains the implementation and resources for the two principal stages of the term–mapping workflow:

📁**RAG_based_definition**  
   - Each mapped term is enriched with a context-derived definition.  
   - Definitions are generated using **LightRAG** followed by **few-shot LLM calls**.  
   - Resulting definitions are used downstream in SKOS matching and Wikidata disambiguation.

📁**Deep_Agent_implementation**  
   - **`Codes-and_files_to_run_agent`**  
     - Contains all code and files required to reproduce wikidata deep agent runs.  
     - Includes LangSmith trace links used in the paper.  
   - **`Wikidata_tools`**  
     - Demonstrates the use of custom tools for interacting with the Wikidata API.  
     - Provides examples of querying, retrieving entity facts, and property expansion.  
   - **`SKOS_matching_tool`**  
     - Presents how the SKOS matching component assigns  
       **exact**, **close**, or **related** mappings in a structured LLM-driven workflow.

