import requests
from pypdf import PdfReader
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer

PDF_PATH = "sample.pdf"   
INDEX_NAME = "pdf_rag_index"
MODEL_NAME = "mistral"
client = Endee()

indexes = client.list_indexes()["indexes"]
index_names = [idx["name"] for idx in indexes]

if INDEX_NAME not in index_names:
    print("Creating index...")
    client.create_index(
        name=INDEX_NAME,
        dimension=384,
        space_type="cosine",
        precision=Precision.FLOAT16
    )

index = client.get_index(INDEX_NAME)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Reading PDF...")
reader = PdfReader(PDF_PATH)

full_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

def chunk_text(text, chunk_size=800):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n"

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

chunks = chunk_text(full_text)

print(f"Total chunks created: {len(chunks)}")

print("Embedding and storing chunks...")

for i, chunk in enumerate(chunks):
    vector = model.encode(chunk).tolist()
    index.upsert([
        {
            "id": f"chunk_{i}",
            "vector": vector,
            "meta": {"text": chunk}
        }
    ])

print("PDF stored successfully!")

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    
    if query.lower() == "exit":
        break

    query_vector = model.encode(query).tolist()

    results = index.query(vector=query_vector, top_k=3)

    context = "\n".join([r["meta"]["text"] for r in results])

    prompt = f"""
You are a strict document question answering system. 
You MUST follow these rules:

1. Use ONLY the exact information present in the CONTEXT section. 
2. Do NOT add technologies, tools, or assumptions not explicitly written.
3. If the answer is not fully in the document context, respond exactly with: "I could not find this information in the document."
4. Do NOT elaborate beyond what is written.

If the answer cannot be extracted from the context, say: "The answer is not provided in the document."

CONTEXT:
----------------
{context}
----------------

QUESTION:
{query}

ANSWER:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
    )

    print("\nAnswer:")
    print(response.json()["response"])