import pandas as pd
import os

RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

def load_data(filename):
    path = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(path)
    print(f"✅ Loaded data from {filename} with shape {df.shape}")
    return df

def clean_data(df):
    # Example cleaning steps — you’ll adjust these to your dataset
    df = df.dropna()  # Remove missing rows
    df = df.drop_duplicates()  # Remove duplicates
    # Convert date columns if any
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col])
    print("🧹 Cleaned data")
    return df

def save_data(df, filename):
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(path, index=False)
    print(f"💾 Saved processed data to {path}")

if __name__ == "__main__":
    df = load_data(".csv")  # <-- replace with your actual filename
    df = clean_data(df)
    save_data(df, "processed_dataset.csv")