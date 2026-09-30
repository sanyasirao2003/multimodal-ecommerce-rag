import pandas as pd
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import re

# -------------------------
# PRICE EXTRACTION (UPDATED)
# -------------------------
def extract_price_limit(query):
    query = query.lower()

    match = re.search(r"(under|below|less than)\s*₹?\s*(\d+)", query)
    if match:
        return float(match.group(2))

    return None


# -------------------------
# LOAD MODEL
# -------------------------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("data/processed/products_clean_v1.csv")
df = df.set_index("product_id")

# -------------------------
# LOAD FAISS
# -------------------------
index = faiss.read_index("data/processed/faiss_index.bin")

with open("data/processed/faiss_product_ids.pkl", "rb") as f:
    text_product_ids = pickle.load(f)

SIMILARITY_THRESHOLD = 0.6

# -------------------------
# CATEGORY KEYWORDS
# -------------------------
CATEGORY_KEYWORDS = {
    "sports": "shoes",
    "running": "shoes",
    "shoe": "shoes",
    "shoes": "shoes",

    "shirt": "clothing",
    "tshirt": "clothing",
    "t-shirt": "clothing",
    "jeans": "clothing",
    "pant": "clothing",
    "clothing": "clothing"
}

# -------------------------
# COLOR KEYWORDS
# -------------------------
COLOR_KEYWORDS = [
    "black", "white", "red", "blue", "green",
    "yellow", "pink", "brown", "grey", "gray"
]

# -------------------------
# MAIN FUNCTION
# -------------------------
def retrieve_products(query, k=40):

    query_lower = query.lower()
    price_limit = extract_price_limit(query)

    # -------------------------
    # DETECT CATEGORY
    # -------------------------
    detected_category = None
    for word, cat in CATEGORY_KEYWORDS.items():
        if word in query_lower:
            detected_category = cat
            break

    # -------------------------
    # DETECT COLOR
    # -------------------------
    detected_color = None
    for color in COLOR_KEYWORDS:
        if color in query_lower:
            detected_color = color
            break

    # -------------------------
    # ENCODE QUERY
    # -------------------------
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    distances, indices = index.search(query_embedding, k)

    products = []

    # -------------------------
    # MAIN LOOP
    # -------------------------
    for distance, idx in zip(distances[0], indices[0]):

        similarity_score = float(distance)

        if similarity_score < SIMILARITY_THRESHOLD:
            continue

        pid = text_product_ids[idx]

        try:
            product = df.loc[pid]
            if isinstance(product, pd.DataFrame):
                product = product.iloc[0]
        except KeyError:
            continue

        # ✅ PRICE FILTER (SAFE)
        if price_limit is not None:
            try:
                if float(product["price"]) > price_limit:
                    continue
            except:
                continue

        title_lower = str(product["title"]).lower()
        category_lower = str(product["category_clean"]).lower()
        product_color = str(product["color"]).lower()
        tags = str(product.get("tags", "")).lower()

        # CATEGORY FILTER
        if detected_category:
            if detected_category not in category_lower:
                continue

        # COLOR FILTER
        if detected_color:
            if detected_color not in product_color:
                continue

        # KEYWORD SCORE
        keyword_score = 0.0
        for word in query_lower.split():
            if word in title_lower:
                keyword_score += 0.25
            if word in category_lower:
                keyword_score += 0.15
            if word in product_color:
                keyword_score += 0.1
            if word in tags:
                keyword_score += 0.4

        final_score = (0.7 * similarity_score) + (0.3 * keyword_score)

        # IMAGE FIX
        image_path = str(product["image_path"])
        if image_path.startswith("http"):
            image_url = image_path
        else:
            image_url = f"http://127.0.0.1:8000/images/{os.path.basename(image_path)}"

        products.append({
            "title": product["title"],
            "category": product["category_clean"],
            "price": float(product["price"]),
            "brand": product["brand"],
            "color": product["color"],
            "description": str(product.get("description", "")),
            "rating": float(product.get("rating", 4.0)),
            "reviews": str(product.get("reviews", "")).split("|"),
            "image_path": image_url,
            "similarity_score": similarity_score,
            "final_score": final_score
        })

    # -------------------------
    # SAFE FALLBACK (FIXED)
    # -------------------------
    if len(products) == 0:
        for distance, idx in zip(distances[0], indices[0]):

            pid = text_product_ids[idx]

            try:
                product = df.loc[pid]
                if isinstance(product, pd.DataFrame):
                    product = product.iloc[0]
            except KeyError:
                continue

            # ✅ APPLY PRICE FILTER HERE ALSO
            if price_limit is not None:
                try:
                    if float(product["price"]) > price_limit:
                        continue
                except:
                    continue

            image_url = f"http://127.0.0.1:8000/images/{os.path.basename(product['image_path'])}"

            products.append({
                "title": product["title"],
                "category": product["category_clean"],
                "price": float(product["price"]),
                "brand": product["brand"],
                "description": str(product.get("description", "")),
                "rating": float(product.get("rating", 4.0)),
                "reviews": str(product.get("reviews", "")).split("|"),
                "color": product["color"],
                "image_path": image_url,
                "similarity_score": float(distance),
                "final_score": float(distance)
            })

    # -------------------------
    # SORT
    # -------------------------
    products = sorted(products, key=lambda x: x["final_score"], reverse=True)

    return products[:20]