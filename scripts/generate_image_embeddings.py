import os
import pickle
import torch
import open_clip
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
DATA_FILE = "data/processed/products_clean_v1.csv"
OUTPUT_FILE = "data/processed/image_embeddings.pkl"

# -----------------------------
# DEVICE SETUP
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(DATA_FILE)
image_paths = df["image_path"].tolist()
product_ids = df["product_id"].tolist()

print(f"Total images: {len(image_paths)}")

# -----------------------------
# LOAD CLIP MODEL
# -----------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model = model.to(device)
model.eval()

# -----------------------------
# GENERATE IMAGE EMBEDDINGS
# -----------------------------
batch_size = 32
all_embeddings = []

print("Generating image embeddings...")

with torch.no_grad():
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        images = []

        for path in batch_paths:
            try:
                image = preprocess(Image.open(path).convert("RGB"))
                images.append(image)
            except:
                # If image fails, use zero tensor
                images.append(torch.zeros(3, 224, 224))

        images = torch.stack(images).to(device)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        all_embeddings.append(image_features.cpu().numpy())

# -----------------------------
# SAVE EMBEDDINGS
# -----------------------------
image_embeddings = np.vstack(all_embeddings)

embedding_data = {
    "product_id": product_ids,
    "embeddings": image_embeddings
}

with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(embedding_data, f)

print("✅ Image embeddings generated successfully")
print(f"Embedding shape: {image_embeddings.shape}")
print(f"Saved to: {OUTPUT_FILE}")
