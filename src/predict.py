# Import tools
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────
# STEP 1: Load the trained model we saved earlier
# (No need to retrain every time!)
# ─────────────────────────────────────────
print("📦 Loading trained model...")
model = joblib.load("models/rainfall_model.pkl")
print("✅ Model loaded!")

# ─────────────────────────────────────────
# STEP 2: Load cleaned data to get column names
# We need to know what columns the model expects
# ─────────────────────────────────────────
df = pd.read_csv("data/processed/cleaned_data.csv")

# Get the feature columns (everything except RainTomorrow)
feature_columns = [col for col in df.columns if col != "RainTomorrow"]
print(f"\n📋 Model expects these {len(feature_columns)} features:")
print(feature_columns)

# ─────────────────────────────────────────
# STEP 3: Create a FAKE sample weather reading
# In real life, this would come from a weather station
# We just use average values for testing
# ─────────────────────────────────────────
print("\n🌤️  Creating a sample weather reading...")

# Use the average of each column as a test sample
sample = df[feature_columns].mean().to_frame().T  # T = transpose (flip rows/cols)

print("Sample weather data:")
print(sample.to_string())

# ─────────────────────────────────────────
# STEP 4: Make a prediction!
# ─────────────────────────────────────────
prediction      = model.predict(sample)          # 0 or 1
prediction_prob = model.predict_proba(sample)    # probability %

# prediction[0] is 0 (No Rain) or 1 (Rain)
result = "🌧️  RAIN expected tomorrow!" if prediction[0] == 1 else "☀️  NO RAIN expected tomorrow"

print("\n" + "="*45)
print(f"  PREDICTION: {result}")
print(f"  Confidence: {max(prediction_prob[0])*100:.1f}%")
print(f"  Rain probability:    {prediction_prob[0][1]*100:.1f}%")
print(f"  No Rain probability: {prediction_prob[0][0]*100:.1f}%")
print("="*45)

# ─────────────────────────────────────────
# STEP 5: Batch prediction (many rows at once)
# Predict for first 5 rows of dataset
# ─────────────────────────────────────────
print("\n📊 Predicting for first 5 rows of dataset:")

sample_5      = df[feature_columns].head(5)
predictions_5 = model.predict(sample_5)
actual_5      = df["RainTomorrow"].head(5).values

print("\n  Row | Predicted  | Actual")
print("  ----|------------|--------")
for i in range(5):
    pred   = "Rain    " if predictions_5[i] == 1 else "No Rain "
    actual = "Rain"     if actual_5[i]       == 1 else "No Rain"
    match  = "✅" if predictions_5[i] == actual_5[i] else "❌"
    print(f"   {i+1}  | {pred}   | {actual}  {match}")

print("\n✅ Prediction complete!")

if __name__ == "__main__":
    pass