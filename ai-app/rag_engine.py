import os
import ollama
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

client = Endee()

INDEX_NAME = "pdf_rag_index"

indexes = client.list_indexes()["indexes"]
index_names = [i["name"] for i in indexes]

if INDEX_NAME in index_names:
    index = client.get_index(INDEX_NAME)
else:
    index = None

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=900):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_pdf(file_path):
    client.delete_index(INDEX_NAME)

    client.create_index(
        name=INDEX_NAME,
        dimension=384,
        space_type="cosine",
        precision=Precision.FLOAT16
    )

    global index
    index = client.get_index(INDEX_NAME)

    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        extracted = page.extract_text()
        
        if extracted:
            text += extracted + "\n"
            print(f"Text extracted from page {page_num + 1}:\n{extracted[:300]}...")  
        else:
            print(f"No text extracted from page {page_num + 1}")

    print("\nEXTRACTED TEXT PREVIEW")
    print(text[:1000])  

    if text:
        chunks = chunk_text(text)
        print(f"Total chunks created: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            index.upsert([{
                "id": f"chunk_{i}",
                "vector": vector,
                "meta": {"text": chunk}
            }])

        print("PDF stored successfully!")
        return len(chunks)
    else:
        print("No text extracted from the PDF. Check if the PDF is text-based.")
        return 0

def query_rag(question):
    if index is None:
        return "No document uploaded yet."

    query_vector = model.encode(question).tolist()

    results = index.query(vector=query_vector, top_k=6)

    context = "\n".join([r["meta"]["text"] for r in results])

    prompt = f"""
You are a strict document question answering system.

Rules:
1. Use ONLY information present inside <context>.
2. Do NOT add tools, technologies, or assumptions.
3. If answer is not clearly in context, reply exactly:
I could not find this information in the document.
4. Do NOT elaborate.

<context>
{context}
</context>

Question:
{question}

Answer:
"""

    response = ollama.generate(
        model="mistral",
        prompt=prompt,
        options={
            "temperature": 0
        }
    )

    return response["response"]