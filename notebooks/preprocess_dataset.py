import os
import pandas as pd

# -----------------------------
# PATH CONFIG
# -----------------------------
RAW_DATA_DIR = "data/raw"
IMAGE_DIR = "data/images"
OUTPUT_DIR = "data/processed"

PRODUCTS_FILE = os.path.join(RAW_DATA_DIR, "final_products_enhanced.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "products_clean_v1.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading product dataset...")
df = pd.read_csv(PRODUCTS_FILE)

print(f"Original shape: {df.shape}")

# -----------------------------
# STANDARDIZE COLUMN NAMES
# -----------------------------
df = df.rename(columns={"id": "product_id"})

# -----------------------------
# IMAGE PATH NORMALIZATION
# -----------------------------
def normalize_image_path(path):
    if pd.isna(path):
        return None
    filename = os.path.basename(path)
    return os.path.join(IMAGE_DIR, filename)

df["image_path"] = df["image_path"].apply(normalize_image_path)

# -----------------------------
# BASIC VALIDATION
# -----------------------------
print("Validating records...")

# Drop missing descriptions
df = df.dropna(subset=["description"])

# Ensure price is numeric
df = df[pd.to_numeric(df["price"], errors="coerce").notnull()]
df["price"] = df["price"].astype(float)

# Validate image existence
df = df[df["image_path"].apply(lambda x: x is not None and os.path.exists(x))]

# -----------------------------
# FINAL COLUMN SELECTION
# -----------------------------
final_columns = [
    "product_id",
    "title",
    "description",
    "category",
    "brand",
    "price",
    "gender",
    "color",
    "image_path"
]

final_df = df[final_columns]

# -----------------------------
# SAVE CLEAN DATASET
# -----------------------------
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Preprocessing complete")
print(f"Final dataset shape: {final_df.shape}")
print(f"Saved to: {OUTPUT_FILE}")
