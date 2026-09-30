import pickle
import faiss
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
EMBEDDING_FILE = "data/processed/image_embeddings.pkl"
INDEX_FILE = "data/processed/image_faiss_index.bin"
ID_FILE = "data/processed/image_faiss_product_ids.pkl"

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
print("Loading image embeddings...")

with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"].astype("float32")
product_ids = data["product_id"]

print(f"Embedding shape: {embeddings.shape}")

# -----------------------------
# CREATE FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]

print("Creating Image FAISS index...")
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print(f"Total image vectors indexed: {index.ntotal}")

# -----------------------------
# SAVE INDEX
# -----------------------------
faiss.write_index(index, INDEX_FILE)

with open(ID_FILE, "wb") as f:
    pickle.dump(product_ids, f)

print("✅ Image FAISS index built successfully")
print(f"Saved index to: {INDEX_FILE}")
