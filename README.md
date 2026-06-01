
# Financial RAG System (Nimap AI/ML Assignment)

## Features
- FastAPI backend
- PDF/Text document indexing
- Chroma vector database
- SentenceTransformer embeddings
- Financial semantic search
- Cross-encoder reranking
- REST APIs for indexing/searching/removing docs

## Project Structure
```
nimap_rag_project/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── models.py
│   └── rag_pipeline.py
├── requirements.txt
├── run.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Server

```bash
python run.py
OR
uvicorn app.main:app --reload
```

## APIs

### Index Document
POST `/rag/index-document`

Form Data:
- file: PDF/TXT file

### Search
POST `/rag/search`

```json
{
  "query": "financial risk related to high debt ratio"
}
```

### Context
GET `/rag/context/{document_id}`

### Remove Document
DELETE `/rag/remove-document/{id}`
