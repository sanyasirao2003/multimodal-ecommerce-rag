import pickle
import faiss
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
EMBEDDING_FILE = "data/processed/text_embeddings.pkl"
INDEX_FILE = "data/processed/faiss_index.bin"
ID_FILE = "data/processed/faiss_product_ids.pkl"

# -----------------------------
# LOAD EMBEDDINGS
# -----------------------------
print("Loading embeddings...")
with open(EMBEDDING_FILE, "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"].astype("float32")
product_ids = data["product_id"]

print(f"Embedding shape: {embeddings.shape}")

# -----------------------------
# CREATE FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]

print("Creating FAISS index...")

nlist = 100  # number of clusters (tune this)

quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

print("Training FAISS index...")
index.train(embeddings)

print("Adding embeddings...")
index.add(embeddings)

# 🔥 VERY IMPORTANT
index.nprobe = 10  # number of clusters to search

print(f"Total vectors indexed: {index.ntotal}")

# -----------------------------
# SAVE INDEX
# -----------------------------
faiss.write_index(index, INDEX_FILE)

with open(ID_FILE, "wb") as f:
    pickle.dump(product_ids, f)

print("✅ FAISS index built successfully")
print(f"Saved index to: {INDEX_FILE}")
