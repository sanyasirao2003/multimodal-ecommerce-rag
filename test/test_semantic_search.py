import pickle
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# -----------------------------
# PATHS
# -----------------------------
DATA_FILE = "data/processed/products_clean_v1.csv"
INDEX_FILE = "data/processed/faiss_index.bin"
ID_FILE = "data/processed/faiss_product_ids.pkl"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)

with open(ID_FILE, "rb") as f:
    product_ids = pickle.load(f)

index = faiss.read_index(INDEX_FILE)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# USER QUERY
# -----------------------------
query = input("Enter search query: ")

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

# -----------------------------
# SEARCH
# -----------------------------
k = 5
distances, indices = index.search(query_embedding, k)

print("\nTop Results:\n")

for i, idx in enumerate(indices[0]):
    product_id = product_ids[idx]
    product = df[df["product_id"] == product_id].iloc[0]
    
    print(f"Rank {i+1}")
    print("Title:", product["title"])
    print("Category:", product["category"])
    print("Price:", product["price"])
    print("Similarity Score:", distances[0][i])
    print("-" * 50)
