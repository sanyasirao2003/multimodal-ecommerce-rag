import pandas as pd
import pickle
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/processed/products_clean_v1.csv"
OUTPUT_FILE = "data/processed/text_embeddings.pkl"

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Clean
df["search_text"] = df["search_text"].fillna("").astype(str)
df = df[df["search_text"].str.strip() != ""]

texts = df["search_text"].tolist()
product_ids = df["product_id"].tolist()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)

print("Generating embeddings...")

all_embeddings = []
chunk_size = 5000

for i in range(0, len(texts), chunk_size):
    batch = texts[i:i+chunk_size]
    
    emb = model.encode(
        batch,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    
    all_embeddings.append(emb)

embeddings = np.vstack(all_embeddings).astype("float32")

# Save
embedding_data = {
    "product_id": product_ids,
    "embeddings": embeddings
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embedding_data, f)

print("✅ Embeddings generated")
print("Shape:", embeddings.shape)