# Step 1: Import libraries (tools we need)
import pandas as pd
import numpy as np
import os

def preprocess():
    print("Loading data...")
    
    # Step 2: Load the CSV file into a table (called DataFrame)
    df = pd.read_csv("data/raw/weatherAUS.csv")
    
    print(f"Original data shape: {df.shape}")  # rows x columns
    print(df.head())                            # show first 5 rows
    
    # Step 3: Drop columns we don't need
    df = df.drop(columns=["Date", "Location"])
    
    # Step 4: Convert Yes/No text to numbers (1 and 0)
    # Machine learning only understands numbers, not words
    df["RainToday"]    = df["RainToday"].map({"Yes": 1, "No": 0})
    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].fillna("Unknown")
        df[col] = le.fit_transform(df[col])
    
    # Step 5: Fill missing values with average of that column
    # Missing values cause errors, so we fill them
    df = df.fillna(df.mean(numeric_only=True))
    
    # Step 6: Drop any remaining rows that still have missing values
    df = df.dropna()
    df["RainTomorrow"] = df["RainTomorrow"].round().astype(int)
    
    print(f"Cleaned data shape: {df.shape}")
    
    # Step 7: Save cleaned data to processed folder
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_data.csv", index=False)
    
    print("✅ Data saved to data/processed/cleaned_data.csv")

# This means: only run if we directly run this file
if __name__ == "__main__":
    preprocess()