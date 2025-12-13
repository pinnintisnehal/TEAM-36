# Medication Reminder Chatbot (Label-Aware RAG)

A **Label-Aware Medication Reminder Chatbot** built as a **Proof of Concept (POC)** using **Retrieval Augmented Generation (RAG)** over **openFDA drug labels**.

The chatbot answers user questions strictly from official drug label data and generates structured medication reminder schedules, avoiding hallucinations by restricting queries to known drugs.

---

## Problem Statement

Drug labels contain critical information such as dosage, warnings, and usage instructions, but they are difficult to interpret and remember.  
Misinterpretation or missed doses can lead to safety risks.

This project aims to:
- Answer questions over drug labels accurately
- Prevent hallucinated answers
- Generate medication reminder plans
- Work without phone/SMS integration

---

## Key Features

- Label-aware Q&A over drug labels
- Drug-restricted queries (only drugs present in database)
- Retrieval Augmented Generation (RAG)
- Section-wise label ingestion
- Medication reminder generation in JSON format
- Jupyter Notebook based POC

---

## Architecture Overview

User ->Jupyter Notebook Interface ->Retriever (ChromaDB)->Relevant Drug Label Sections->LLM (Ollama)->Answer / Reminder JSON


---

## RAG Pipeline

1. Drug label data is loaded from openFDA-style JSON
2. Labels are split into sections (dosage, warnings, usage, etc.)
3. Long sections are chunked to fit embedding limits
4. Chunks are embedded and stored in ChromaDB with metadata
5. Queries retrieve only relevant sections for the selected drug
6. LLM generates answers strictly from retrieved context

---

## Project Structure (POC)
├── sample.txt # Drug label data (openFDA style)<br>
├── chroma_poc/ # Persisted ChromaDB<br>
├── poc.ipynb # Main Jupyter Notebook<br>
└── README.md<br>


---

## Tech Stack

- Python
- LangChain
- ChromaDB
- Ollama
- LLM: llama3:8b
- Embeddings: nomic-embed-text
- Jupyter Notebook

---

## How to Run

### 1. Install dependencies
```bash
pip install langchain langchain-community chromadb
```
2. Pull Ollama models
```bash
ollama pull llama3:8b
ollama pull nomic-embed-text
```

