import pickle
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# PATHS
# -----------------------------
DATA_FILE = "data/processed/products_clean_v1.csv"

TEXT_INDEX_FILE = "data/processed/faiss_index.bin"
TEXT_ID_FILE = "data/processed/faiss_product_ids.pkl"

IMAGE_INDEX_FILE = "data/processed/image_faiss_index.bin"
IMAGE_ID_FILE = "data/processed/image_faiss_product_ids.pkl"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)

# Load text FAISS
text_index = faiss.read_index(TEXT_INDEX_FILE)
with open(TEXT_ID_FILE, "rb") as f:
    text_product_ids = pickle.load(f)

# Load image FAISS
image_index = faiss.read_index(IMAGE_INDEX_FILE)
with open(IMAGE_ID_FILE, "rb") as f:
    image_product_ids = pickle.load(f)

# Load text embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# USER INPUT
# -----------------------------
query = input("Enter search query: ")

# Hybrid weights
alpha = 0.7  # text weight
beta = 0.3   # image weight

# -----------------------------
# TEXT SEARCH
# -----------------------------
query_embedding = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

k = 20  # fetch more candidates
text_distances, text_indices = text_index.search(query_embedding, k)

# Convert to dict for quick lookup
text_scores = {
    text_product_ids[idx]: text_distances[0][i]
    for i, idx in enumerate(text_indices[0])
}

# -----------------------------
# IMAGE SCORES (for same products)
# -----------------------------
# Get image embeddings for candidate products
with open("data/processed/image_embeddings.pkl", "rb") as f:
    image_data = pickle.load(f)

image_embeddings = image_data["embeddings"].astype("float32")
image_product_ids_full = image_data["product_id"]

# Map product_id → image embedding index
image_id_to_index = {
    pid: i for i, pid in enumerate(image_product_ids_full)
}

# Hybrid scoring
hybrid_results = []

for pid, text_score in text_scores.items():
    img_index = image_id_to_index.get(pid)
    
    if img_index is not None:
        image_vector = image_embeddings[img_index].reshape(1, -1)
        image_score, _ = image_index.search(image_vector, 1)
        image_score = image_score[0][0]
    else:
        image_score = 0
    
    final_score = alpha * text_score + beta * image_score
    
    hybrid_results.append((pid, final_score))

# Sort by hybrid score
hybrid_results = sorted(hybrid_results, key=lambda x: x[1], reverse=True)[:5]

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\nTop Hybrid Results:\n")

for rank, (pid, score) in enumerate(hybrid_results, 1):
    product = df[df["product_id"] == pid].iloc[0]
    
    print(f"Rank {rank}")
    print("Title:", product["title"])
    print("Category:", product["category"])
    print("Price:", product["price"])
    print("Hybrid Score:", score)
    print("-" * 50)
