# rag.py
# Light RAG retrieval helper.
import os
from typing import List, Tuple, Dict, Any

import chromadb
from openai import OpenAI

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "bittipuisto")

# Create Chroma client
_chroma = chromadb.PersistentClient(path=CHROMA_DIR)
_collection = _chroma.get_or_create_collection(name=COLLECTION_NAME)

def retrieve(
        client: OpenAI,
        query: str,
        k: int = 6
) -> List[Tuple[str, Dict[str, Any]]]:
    # Args:
    #   client: OpenAI client instance.
    #   query: user's question.
    #   k: number of chunks to return.

    # Embed the query into a vector.
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    # Vector search in Chroma.
    res = _collection.query(
        query_embeddings=[emb],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    documents = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]

    # Pair documents with metadatas.
    hits: List[Tuple[str, Dict[str, Any]]] = []
    for doc, meta in zip(documents, metadatas):
        if doc and isinstance(meta, dict):
            hits.append((doc, meta))

    return hits