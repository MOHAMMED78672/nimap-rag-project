import uuid
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

embedding_model = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANK_MODEL)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def create_chunks(text):
    return splitter.split_text(text)

def index_document(file_path):
    text = extract_text(file_path)
    chunks = create_chunks(text)

    embeddings = embedding_model.encode(chunks).tolist()

    doc_id = str(uuid.uuid4())

    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"document_id": doc_id} for _ in chunks]
    )

    return {
        "document_id": doc_id,
        "chunks_indexed": len(chunks)
    }

def semantic_search(query, top_k=20):
    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results["documents"][0]
    ids = results["ids"][0]

    pairs = [[query, doc] for doc in docs]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(ids, docs, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top_results = ranked[:5]

    return [
        {
            "chunk_id": r[0],
            "text": r[1],
            "rerank_score": float(r[2])
        }
        for r in top_results
    ]

def semantic_search_within_document(query, document_id, top_k=5):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    
    # ChromaDB where clause syntax
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        where={"document_id": {"$eq": document_id}}  # ← FIXED
    )
    
    # Check if any results found
    if not results["documents"] or len(results["documents"][0]) == 0:
        return []
    
    docs = results["documents"][0]
    ids = results["ids"][0]
    
    # Rerank results
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    
    # Sort by rerank score and take top_k
    ranked = sorted(
        zip(ids, docs, scores),
        key=lambda x: x[2],
        reverse=True
    )
    
    top_results = ranked[:top_k]
    
    return [
        {
            "chunk_id": r[0],
            "text": r[1],
            "rerank_score": float(r[2])
        }
        for r in top_results
    ]

def get_context(document_id):
    results = collection.get()

    matched = []

    for idx, meta in enumerate(results["metadatas"]):
        if meta["document_id"] == document_id:
            matched.append(results["documents"][idx])

    return matched

def remove_document(document_id):
    results = collection.get()

    delete_ids = []

    for idx, meta in enumerate(results["metadatas"]):
        if meta["document_id"] == document_id:
            delete_ids.append(results["ids"][idx])

    if delete_ids:
        collection.delete(ids=delete_ids)

    return {
        "removed_chunks": len(delete_ids)
    }