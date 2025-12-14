import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# -----------------------------
# IMPORT YOUR EXISTING AGENT
# -----------------------------
from agents import ingestion_agent, extract_list_field


# -----------------------------
# LOAD DATA
# -----------------------------
with open("sample.txt", "r", encoding="utf-8") as f:
    records = json.load(f)

print(f"📄 Loaded {len(records)} records")


# -----------------------------
# RUN INGESTION AGENT
# -----------------------------
all_documents = []

for record in records:
    docs = ingestion_agent(record)
    all_documents.extend(docs)

print(f"🧩 Generated {len(all_documents)} documents")


# -----------------------------
# BUILD VECTOR STORE
# -----------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma.from_documents(
    documents=all_documents,
    embedding=embeddings,
    persist_directory="./chroma_poc"
)

vectorstore.persist()

print("✅ Ingestion completed and ChromaDB persisted")
