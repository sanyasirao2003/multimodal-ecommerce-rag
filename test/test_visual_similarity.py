import pickle
import faiss
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# PATHS
# -----------------------------
DATA_FILE = "data/processed/products_clean_v1.csv"
INDEX_FILE = "data/processed/image_faiss_index.bin"
ID_FILE = "data/processed/image_faiss_product_ids.pkl"
EMBEDDING_FILE = "data/processed/image_embeddings.pkl"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)

with open(ID_FILE, "rb") as f:
    product_ids = pickle.load(f)

with open(EMBEDDING_FILE, "rb") as f:
    embedding_data = pickle.load(f)

embeddings = embedding_data["embeddings"].astype("float32")

index = faiss.read_index(INDEX_FILE)

# -----------------------------
# SELECT A RANDOM PRODUCT
# -----------------------------
query_index = 100  # change this to test others

query_embedding = embeddings[query_index].reshape(1, -1)

k = 6  # include itself
distances, indices = index.search(query_embedding, k)

print("Query Product:")
print(df.iloc[query_index]["title"])

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
plt.figure(figsize=(15, 5))

for i, idx in enumerate(indices[0]):
    product_id = product_ids[idx]
    product = df[df["product_id"] == product_id].iloc[0]

    image = Image.open(product["image_path"])

    plt.subplot(1, k, i+1)
    plt.imshow(image)
    plt.title(f"{i}\n{product['category']}")
    plt.axis("off")

plt.show()
