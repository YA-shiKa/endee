import requests
from endee import Endee, Precision
from sentence_transformers import SentenceTransformer
client = Endee()
indexes = client.list_indexes()["indexes"]
index_names = [idx["name"] for idx in indexes]

if "rag_index" not in index_names:
    print("Creating rag_index...")
    client.create_index(
        name="rag_index",
        dimension=384,
        space_type="cosine",
        precision=Precision.FLOAT16
    )
else:
    print("rag_index already exists. Skipping creation.")

index = client.get_index("rag_index")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Artificial Intelligence is transforming the world.",
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks."
]
for i, doc in enumerate(documents):
    vector = model.encode(doc).tolist()
    index.upsert([
        {
            "id": f"doc_{i}",
            "vector": vector,
            "meta": {"text": doc}
        }
    ])

print("Documents inserted successfully!")
query = "What is AI?"
query_vector = model.encode(query).tolist()

results = index.query(vector=query_vector, top_k=3)

print("\nQuery Results:")
print(results)

context = "\n".join([r["meta"]["text"] for r in results])

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
)

print("\nGenerated Answer:")
print(response.json()["response"])