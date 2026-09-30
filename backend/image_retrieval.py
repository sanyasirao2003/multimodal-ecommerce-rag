import faiss
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
import open_clip
import os

# -------------------------
# DEVICE
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LOAD CLIP MODEL
# -------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)
model.eval()

# -------------------------
# LOAD TEXT MODEL
# -------------------------
text_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -------------------------
# LOAD IMAGE FAISS INDEX
# -------------------------
image_index = faiss.read_index("data/processed/image_faiss_index.bin")

with open("data/processed/image_faiss_product_ids.pkl", "rb") as f:
    image_product_ids = pickle.load(f)

# -------------------------
# LOAD DATASET
# -------------------------
df = pd.read_csv("data/processed/products_clean_v1.csv")
df_indexed = df.set_index("product_id")

# -------------------------
# LOAD TEXT EMBEDDINGS
# -------------------------
with open("data/processed/text_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    text_embeddings = data["embeddings"]
    text_product_ids = data["product_id"]  # ✅ IMPORTANT FIX

# =========================
# HYBRID RETRIEVAL FUNCTION
# =========================
def retrieve_hybrid(image_file=None, query=""):

    print("📸 Image received:", image_file is not None)
    print("📝 Query:", query)

    image_scores = {}
    text_scores = {}

    # -------------------------
    # IMAGE SEARCH
    # -------------------------
    if image_file:
        try:
            image = Image.open(image_file).convert("RGB")
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_embedding = image_features.cpu().numpy()[0]

            D, I = image_index.search(
                np.array([image_embedding]).astype("float32"), 20
            )

            for score, idx in zip(D[0], I[0]):
                pid = image_product_ids[idx]   # ✅ correct mapping
                image_scores[pid] = float(score)

        except Exception as e:
            print("❌ Image error:", e)

    # -------------------------
    # TEXT SEARCH
    # -------------------------
    if query:
        try:
            text_embedding = text_model.encode(
                [query],
                normalize_embeddings=True
            )[0]

            scores = np.dot(text_embeddings, text_embedding)

            # Top-k indices
            top_indices = np.argsort(scores)[-20:]

            for idx in top_indices:
                pid = text_product_ids[idx]   # ✅ FIXED (no int(idx))
                text_scores[pid] = float(scores[idx])

        except Exception as e:
            print("❌ Text error:", e)

    # -------------------------
    # RANK FUSION (FIXED)
    # -------------------------
    # Sort scores
    image_ranked = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
    text_ranked = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)

    # Create rank maps (FAST lookup)
    image_rank_map = {pid: rank for rank, (pid, _) in enumerate(image_ranked)}
    text_rank_map = {pid: rank for rank, (pid, _) in enumerate(text_ranked)}

    all_ids = set(image_rank_map.keys()) | set(text_rank_map.keys())

    final_scores = {}

    for pid in all_ids:
        img_rank = image_rank_map.get(pid, 100)
        txt_rank = text_rank_map.get(pid, 100)

        # Rank-based scoring
        final_score = (1 / (img_rank + 1)) + (1 / (txt_rank + 1))
        final_scores[pid] = final_score

    # -------------------------
    # SORT RESULTS
    # -------------------------
    sorted_products = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # -------------------------
    # BUILD RESPONSE
    # -------------------------
    results = []

    for pid, score in sorted_products:
        try:
            product = df_indexed.loc[pid]
            if isinstance(product, pd.DataFrame):
                product = product.iloc[0]

            image_path = str(product["image_path"])

            if not image_path.startswith("http"):
                filename = os.path.basename(image_path)
                image_path = f"http://127.0.0.1:8000/images/{filename}"

            results.append({
                "title": product["title"],
                "price": product["price"],
                "brand": product["brand"],
                "category": product.get("category_clean", product.get("category", "Unknown")),  # ✅ FIX
                "color": product.get("color", "Unknown"),  # ✅ FIX
                "image_path": image_path,
                "rating": float(product.get("rating", 4.0)),
                "description": product.get("description", ""),
                "reviews": str(product.get("reviews", "")).split("|")
            })

        except Exception as e:
            print("❌ Mapping error:", e)

    return results