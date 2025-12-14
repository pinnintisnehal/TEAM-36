from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory="./chroma_poc",
        embedding_function=embeddings
    )


def get_available_products(vectorstore):
    data = vectorstore._collection.get(include=["metadatas"])

    if not data or not data.get("metadatas"):
        return []

    products = sorted(
        set(
            m.get("product")
            for m in data["metadatas"]
            if m and "product" in m
        )
    )
    return products
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory="./chroma_poc",
        embedding_function=embeddings
    )


def is_db_empty(vectorstore):
    try:
        return vectorstore._collection.count() == 0
    except:
        return True
