import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from vectorstore import get_vectorstore, is_db_empty

llm = Ollama(model="llama3:8b")
vectorstore = get_vectorstore()

from langchain_core.documents import Document

# -------------------------
# HELPER FUNCTION
# -------------------------
def extract_list_field(record, key):
    val = record.get(key, [])
    if isinstance(val, list):
        return "\n".join(val)
    return str(val)

def chunk_text(text, max_chars=800):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += max_chars
    return chunks


from langchain_core.documents import Document

def autonomous_ingestion():
    vectorstore = get_vectorstore()

    if not is_db_empty(vectorstore):
        print("✅ Vector DB already populated. Skipping ingestion.")
        return vectorstore

    print("⚙️ Vector DB empty. Running ingestion agent...")

    with open("sample.txt", "r", encoding="utf-8") as f:
        records = json.load(f)

    all_documents = []

    for record in records:
        docs = ingestion_agent(record)

        for doc in docs:
            chunks = chunk_text(doc.page_content, max_chars=800)

            for i, chunk in enumerate(chunks):
                all_documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk": i
                        }
                    )
                )

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory="./chroma_poc"
    )

    vectorstore.persist()

    print(f"✅ Autonomous ingestion completed. Chunks stored: {len(all_documents)}")
    return vectorstore


# -------------------------
# INGESTION AGENT
# -------------------------
def ingestion_agent(record):
    """Convert a single FDA JSON record into multiple Documents (one per section)."""

    brand = record.get("openfda", {}).get("brand_name", ["Unknown Product"])[0]
    product_id = record.get("id", "unknown")

    sections = {
        "indications_and_usage": extract_list_field(record, "indications_and_usage"),
        "warnings": extract_list_field(record, "warnings_and_cautions"),
        "active_ingredients": extract_list_field(record, "active_ingredients"),
        "boxed_warning": extract_list_field(record, "boxed_warning"),
        "dosage_and_administration": extract_list_field(record, "dosage_and_administration"),
        "adverse_reactions": extract_list_field(record, "adverse_reactions"),
        "interactions": extract_list_field(record, "drug_interactions"),
        "contraindications": extract_list_field(record, "contraindications"),
        "keep_out_of_reach_of_children": extract_list_field(record, "keep_out_of_reach_of_children"),
        "pregnancy": extract_list_field(record, "pregnancy"),
        "pediatric_use": extract_list_field(record, "pediatric_use"),
    }

    docs = []
    for section_name, text in sections.items():
        if text.strip():
            docs.append(
                Document(
                    page_content=f"PRODUCT: {brand}\nSECTION: {section_name}\n\n{text}",
                    metadata={
                        "product": brand,
                        "section": section_name,
                        "id": product_id,
                    },
                )
            )
    return docs

# -------------------------
# Q&A AGENT
# -------------------------
def ask_drug(product, question):
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"product": product}
        }
    )

    docs = retriever.invoke(question)

    if not docs:
        return "Information not available in the label."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a medication assistant.

RULES:
- Answer ONLY from the drug label context.
- Do NOT add medical advice.
- If information is missing, say:
  "Information not available in the label."

Drug Label Context:
{context}

Question:
{question}
"""
    return llm.invoke(prompt)

import json
import re

def parse_or_fallback(text):
    """
    Try to parse valid JSON.
    If parsing fails, return raw text safely.
    """

    # Attempt to extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        # No JSON detected at all
        return {
            "type": "raw",
            "content": text
        }

    json_str = match.group(0)

    try:
        parsed = json.loads(json_str)
        return {
            "type": "json",
            "content": parsed
        }

    except json.JSONDecodeError:
        # JSON present but malformed
        return {
            "type": "raw",
            "content": text
        }

# -------------------------
# REMINDER AGENT (JSON)
# -------------------------
def reminder_agent(product):
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "$and": [
                    {"product": product},
                    {"section": "dosage_and_administration"}
                ]
            }
        }
    )

    docs = retriever.invoke("dosage instructions")

    if not docs:
        return {"error": "No dosage information available."}

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are a medication reminder generator.

STRICT RULES (MANDATORY):
- You MUST return ONLY raw JSON.
- Do NOT include explanations.
- Do NOT include markdown.
- Do NOT include any text before or after JSON.
- Output must start with {{ and end with }}.

Use ONLY the dosage information below.
Do NOT guess missing details.
Do NOT give medical advice.

Dosage Information:
{context}

OUTPUT FORMAT (JSON ONLY):
{{
  "product": "{product}",
  "schedule": [
    {{
      "time": "...",
      "instruction": "..."
    }}
  ],
  "notes": "..."
}}
"""
    raw_output = llm.invoke(prompt)
    return parse_or_fallback(raw_output)
