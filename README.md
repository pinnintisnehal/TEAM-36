# Medication Reminder Chatbot (Label-Aware RAG)

A **Label-Aware Medication Reminder Chatbot** built as a **Proof of Concept (POC)** using **Retrieval Augmented Generation (RAG)** over **openFDA drug labels**.

The chatbot answers user questions strictly from official drug label data and generates structured medication reminder schedules, avoiding hallucinations by restricting queries to known drugs.

This project is implemented as a Streamlit web application and uses a modular, agent-based architecture
---

## Problem Statement

Drug labels contain critical information such as dosage, warnings, and usage instructions, but they are difficult to interpret and remember.  
However labels are difficult to read, users may misinterpret instructions, and in many cases labels are available only as images, making the information even harder to access. Missing reminders can lead to unsafe usage.

This project aims to:
- Answer questions over drug labels accurately
- Prevent hallucinated answers
- Generate medication reminder plans
- Work without phone/SMS integration
- Works completely offline

---

## Solution Overview

- The chatbot uses Retrieval Augmented Generation (RAG) with functional AI agents and OCR-based drug identification:
- Drug names can be automatically identified from uploaded label images using OCR
- Drug labels are ingested and embedded
- Relevant sections are retrieved per query
- A local LLM answers only from retrieved context
- Reminder schedules are generated in JSON
---

## Architecture Overview

<img width="1408" height="768" alt="Gemini_Generated_Image_nneu4mnneu4mnneu" src="https://github.com/user-attachments/assets/6a3cc204-53ec-4d8f-98c3-d5b7fe062c21" />

The system also supports image-based input, where users can upload a drug label image. Text is extracted using OCR, the drug name is identified and matched against the vector database, and the existing RAG pipeline is applied.


---

## RAG Pipeline

1. Drug label data is loaded from sample.txt
2. Labels are split into meaningful sections
3. Long sections are chunked
4. Chunks are embedded using nomic-embed-text
5. Embeddings are stored in ChromaDB
6. User selects a medication or uploads a drug label image
7. OCR extracts text from the image and identifies the drug name
8. The identified drug is matched against the vector database
9. Retriever fetches relevant chunks
10. LLM generates grounded answers or reminders

---

## Project Structure (Streamlit)
```bash
medication_bot/
│
├── app.py               # Streamlit UI and interaction logic
├── agents.py            # Ingestion, QA, Reminder agents
├── vectorstore.py       # ChromaDB setup and helpers
├── guardrails.py        # Input validation and safety checks
├── ocr_utils.py         # OCR-based drug name identification
├── evals.py             # Basic evaluation utilities
├── ingest.py            # Manual ingestion script
├── sample.txt           # Drug label data (openFDA-style JSON)
├── chroma_poc/          # Persisted vector database
├── requirements.txt     # Dependencies
└── README.md
```

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- Ollama
- LLM: llama3:8b
- Embeddings: nomic-embed-text
- Streamlit
- EasyOCR (for image-based drug label text extraction)


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
3. Run the app
```bash
streamlit run app.py
```

---

## Agents used in this Project
This project uses simple task-based agents, where each agent is responsible for one specific function in the system.
These agents work together in a fixed and safe flow and do not act autonomously.

Agents in the System

- Ingestion Agent
Loads drug label data, splits it into meaningful sections, and stores it in the vector database for search.

- Retrieval Agent
Retrieves the most relevant drug label information based on the selected medication and user query.

- Question Answering Agent
Uses the retrieved drug label content to answer user questions accurately.

- Reminder Agent
Generates a structured medication reminder schedule based on dosage instructions from the label.

- Guardrail Agent
Validates user inputs such as medication selection and questions to ensure safe and correct usage.

Each agent performs a single, clearly defined task, making the system easy to understand, reliable, and safe for healthcare-related use.

## Evaluation

File: evals.py

- Checks if retrieval returns documents
- Verifies reminder output structure
- Ensures JSON validity where applicable
  
## 🔐 Safety & Guardrails

- Only drugs present in the vector DB can be queried
- Questions must be non-empty
- LLM is strictly constrained to label context
- No medical advice beyond label content
  
## OCR Support

- Users can upload images of drug labels
- Extracted text is displayed for transparency
- Identified drugs are auto-selected to avoid repeated user input

---
## Disclaimer
**This project is for educational and demonstration purposes only.
It does not replace professional medical advice.**
```

