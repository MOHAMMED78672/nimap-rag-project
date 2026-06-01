import os

from fastapi import FastAPI, UploadFile, File, Query
from app.models import SearchRequest
from app.rag_pipeline import (
    index_document,
    semantic_search,
    get_context,
    remove_document,
    semantic_search_within_document  # NEW import
)

app = FastAPI(title="Financial RAG API")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Financial RAG API Running"}

@app.post("/rag/index-document")
async def upload_document(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = index_document(file_path)

    return {
        "status": "success",
        "data": result
    }

@app.post("/rag/search")
def search(request: SearchRequest):
    results = semantic_search(request.query)

    return {
        "query": request.query,
        "results": results
    }

# UPDATED endpoint - now accepts optional query parameter
@app.get("/rag/context/{document_id}")
def context(
    document_id: str, 
    query: str = Query(None, description="Optional query to search within document")
):
    if query:
        # Search within specific document
        results = semantic_search_within_document(query, document_id)
        return {
            "document_id": document_id,
            "query": query,
            "context": results
        }
    else:
        # Return all chunks (original behavior)
        data = get_context(document_id)
        return {
            "document_id": document_id,
            "context": data
        }

@app.delete("/rag/remove-document/{document_id}")
def delete_doc(document_id: str):
    result = remove_document(document_id)

    return {
        "status": "deleted",
        "data": result
    }