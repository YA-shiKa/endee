# Endee-Powered Document Analysis AI

**Retrieval Augmented Generation (RAG) using Endee Vector Database** 

---

## Project Overview

This project is a Document Question Answering System built using Endee (nD) Vector Database as the core vector search engine.

Users can:

- Upload a PDF document
- Automatically extract and chunk text
- Convert text into embeddings
- Store embeddings in Endee
- Ask questions about the document
- Get accurate answers strictly grounded in document content

The system uses a Retrieval Augmented Generation (RAG) pipeline where vector search retrieves relevant document chunks, and an LLM generates answers using only retrieved context.

---
## Preview

https://github.com/user-attachments/assets/b2bd0b4a-9d1c-4231-af2c-36f4e41aa9b2


---

## Problem Statement

Traditional document search relies on keyword matching, which:

- Misses semantic meaning
- Fails on paraphrased questions
- Performs poorly on large documents

This project solves that problem by:

- Converting document text into vector embeddings
- Storing them in Endee high-performance vector database
- Performing semantic similarity search
- Using retrieved context for accurate LLM-based answers

This ensures meaning-based retrieval instead of keyword matching.

---

## System Architecture

### Architecture Flow

1. User uploads PDF
2. PDF text is extracted using PyPDF
3. Text is chunked into smaller segments
4. SentenceTransformer generates embeddings (384-dim)
5. Embeddings stored in Endee
6. User asks question
7. Question embedding generated
8. Endee performs similarity search (top-k)
9. Retrieved context passed to Mistral (Ollama)
10. LLM generates grounded answer

### Tech Stack
- Vector Database: Endee (nD)
- Backend: FastAPI
- Frontend: HTML + JS
- Embedding Model: all-MiniLM-L6-v2 (SentenceTransformers)
- LLM: Mistral (via Ollama)
- PDF Parsing: PyPDF
- Precision Mode: FLOAT16

---

## System Design and Technical Approach
### System Design Overview

This project follows a modular Retrieval Augmented Generation (RAG) architecture where responsibilities are clearly separated into three core layers:

- Ingestion Layer
- Retrieval Layer (Endee Vector Search)
- Generation Layer (LLM Response Engine)

This layered design ensures scalability, maintainability, and clear separation of concerns.

#### 1️⃣ Ingestion Layer

The ingestion pipeline processes the uploaded document and prepares it for semantic retrieval.

Steps:

- PDF file uploaded via frontend
- Text extracted using PyPDF
- Extracted text cleaned and concatenated
- Text divided into smaller chunks (~800–900 characters)
- Each chunk converted into a 384-dimensional embedding
- Embeddings stored in Endee along with original text as metadata

Chunking:

- Improves retrieval granularity
- Reduces irrelevant context
- Increases Top-K relevance
- Prevents entire-document embedding noise

#### 2️⃣ Retrieval Layer (Core: Endee Vector Database)

This is the most critical component of the system.

When a user submits a question:

1. The question is converted into a 384-dimensional embedding.
2. Endee performs cosine similarity search.
3. Top-K most semantically similar chunks are retrieved.
4. Retrieved chunks are combined to form contextual knowledge.
```bash
results = index.query(vector=query_vector, top_k=6)
```
Design Choices:
- dimension=384 → Matches embedding model output
- space_type="cosine" → Suitable for semantic similarity
- precision=FLOAT16 → Optimized memory usage and faster computation
- top_k=6 → Balanced retrieval depth without excessive noise

Endee acts as the semantic memory layer of the system.

#### 3️⃣ Generation Layer (LLM Integration)

After retrieval, the system constructs a strictly controlled prompt that includes:
- Retrieved context
- User question
- Explicit instructions to avoid hallucination
  
The LLM (Mistral via Ollama):

- Uses only retrieved context
- Does not rely on external knowledge
- Returns deterministic responses (temperature = 0)
- If answer is not found → responds with a fixed fallback message

This ensures the system behaves like a grounded document QA engine, not a general chatbot.

### Embedding Strategy
- Model Used: sentence-transformers/all-MiniLM-L6-v2
- Output Dimension: 384
- Chosen because:
  - Lightweight and fast
  - Good semantic understanding
  - Efficient for CPU inference
  - Compatible with Endee configuration

### Query Processing Pipeline

The complete query lifecycle:

- User submits question
- Question converted to embedding
- Endee retrieves similar chunks
- Context constructed from Top-K results
- Strict prompt generated
- Mistral produces answer
- Response returned to frontend
- This ensures semantic retrieval + controlled generation.

### Hallucination Control Strategy

To prevent LLM hallucination:
- Strict prompt instructions
- Context-only answering
- No tool usage allowed
- Deterministic temperature setting
- Fixed fallback message when answer not found

This improves factual reliability.

### Scalability Considerations

The current architecture can be extended to support:

- Multi-document indexing
- Persistent vector storage
- Hybrid search (keyword + vector)
- Authentication-enabled Endee server
- Distributed retrieval
- Streaming LLM responses
- The modular design ensures easy extension without major restructuring.
  
---

## How Endee is Used 

Endee is used as the primary vector storage and retrieval engine.

### 1️⃣ Index Creation
```bash
client.create_index(
    name="pdf_rag_index",
    dimension=384,
    space_type="cosine",
    precision=Precision.FLOAT16
)
```

- Dimension: 384 (matches MiniLM embedding size)
- Similarity: Cosine
- Precision: FLOAT16 for optimized performance

### 2️⃣ Vector Upsert

Each document chunk is embedded and stored:

```bash
index.upsert([{
    "id": f"chunk_{i}",
    "vector": vector,
    "meta": {"text": chunk}
}])
```
- Stores embedding vector
- Stores original text in metadata
- Enables contextual retrieval

### 3️⃣ Semantic Query
```bash
results = index.query(vector=query_vector, top_k=6)
```
- Retrieves most semantically similar chunks
- Enables meaning-based search
- Acts as the retrieval layer of RAG

---

## Setup & Execution Guide

### Step 1 — Clone Repository

```bash
git clone https://github.com/YA-shiKa/endee.git
cd endee
```

### Step 2 — Run Endee Server

You can run Endee using Docker

### Step 3 — Install Python Dependencies
```bash
cd ai-app
pip install -r requirements.txt
```
Required packages:

- fastapi
- uvicorn
- sentence-transformers
- pypdf
- ollama

### Step 4 — Run Backend
```bash
uvicorn app:app --reload
```

Open browser:
```bash
http://127.0.0.1:8000
```
### Step 5 — Run Ollama (Required for LLM)
```bash
ollama pull mistral
ollama run mistral
```
Make sure Ollama server is running.

### 🧪 Testing Endee Integration

You can test vector insertion:
```bash
python test_endee.py
```
