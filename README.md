# KARMA Mini: Automated Knowledge Graph Extraction

KARMA Mini is a streamlined, 3-agent natural language processing framework designed to automatically extract, standardize, and integrate biomedical knowledge from raw text abstracts into a structured Knowledge Graph. 

This project is a simplified reproduction of the original [KARMA architecture](https://github.com/YuxingLu613/KARMA), optimized for batch processing of short texts (like paper abstracts) and specifically designed to export data seamlessly into **Neo4j**.

## The 3-Agent Architecture

The pipeline is driven by three specialized Large Language Model (LLM) agents working in sequence:

### 1. Information Extraction Agent (IEA)
**Role:** The reader. It ingests the raw text of an abstract and extracts key biomedical entities (e.g., Diseases, Drugs, Genes) and the explicit relationships between them.
*   **Input:** Raw abstract text.
*   **Output:** Raw Triples (e.g., `["Aspirin", "lowers", "Headache"]`) along with the source text evidence.

### 2. Schema Alignment Agent (SAA)
**Role:** The standardizer. It takes the raw, messy triples and maps them to a strict, predefined vocabulary (ontology). This ensures that different ways of saying the same thing (e.g., "reduces", "lowers", "decreases") are grouped under a single relationship type (e.g., `INHIBITS`).
*   **Input:** Raw Triples.
*   **Output:** Aligned Triples (e.g., `["Aspirin", "INHIBITS", "Headache"]`).

### 3. Knowledge Integration Agent (KIA)
**Role:** The judge and synthesizer. After processing all abstracts, this agent reviews the global list of aligned triples. It merges duplicates, aggregates confidence scores, and crucially, resolves logical conflicts (e.g., Paper A says X inhibits Y, but Paper B says X activates Y) using LLM-based reasoning.
*   **Input:** Global list of Aligned Triples.
*   **Output:** The final, conflict-free Knowledge Graph.

## Tech Stack & Neo4j Integration
*   **LLM Provider:** Uses the `openai` Python client configured for the KIT API toolbox.
*   **Export:** The data structures are designed to export directly into `nodes.csv` and `relationships.csv`, optimized for `LOAD CSV` operations in Neo4j.

## Setup & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    Create a `.env` file in the root directory:
    ```env
    KIT_API_KEY=your_actual_api_key_here
    KIT_BASE_URL=https://ki-toolbox.scc.kit.edu/api/v1
    ```

3.  **Run Pipeline:**
    *(Implementation pending - see `main.py`)*
