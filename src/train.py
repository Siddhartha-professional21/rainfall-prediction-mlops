# Import all tools we need
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import lightgbm as lgb
import yaml
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────
# STEP 1: Load our settings from params.yaml
# ─────────────────────────────────────────
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# ─────────────────────────────────────────
# STEP 2: Load the cleaned data we made earlier
# ─────────────────────────────────────────
print("📂 Loading cleaned data...")
df = pd.read_csv(params["data"]["processed_path"])
print(f"Data shape: {df.shape}")

# ─────────────────────────────────────────
# STEP 3: Separate INPUT features and TARGET
# X = what we give the model (weather info)
# y = what we want to predict (Rain tomorrow?)
# ─────────────────────────────────────────
X = df.drop(columns=["RainTomorrow"])   # everything except the answer
y = df["RainTomorrow"]                  # the answer (0 or 1)

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape:   {y.shape}")

# ─────────────────────────────────────────
# STEP 4: Split data into Train and Test sets
# 80% for training, 20% for testing
# (like giving 80% questions for study, 20% for exam)
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = params["model"]["test_size"],     # 0.2 = 20%
    random_state= params["model"]["random_state"]   # 42 = fixed random seed
)

print(f"\nTraining samples:  {len(X_train)}")
print(f"Testing samples:   {len(X_test)}")

# ─────────────────────────────────────────
# STEP 5: Create the LightGBM model
# Think of it as creating a "student" ready to learn
# ─────────────────────────────────────────
model = lgb.LGBMClassifier(
    n_estimators  = params["model"]["n_estimators"],   # 100 trees
    learning_rate = params["model"]["learning_rate"],  # how fast to learn
    random_state  = params["model"]["random_state"]
)

# ─────────────────────────────────────────
# STEP 6: MLflow — tracks everything automatically
# Like a lab notebook that records all experiments
# ─────────────────────────────────────────
mlflow.set_experiment("rainfall-prediction")  # name of our experiment

with mlflow.start_run():  # start recording

    print("\n🚀 Training model...")
    model.fit(X_train, y_train)   # TRAIN the model!
    print("✅ Training done!")

    # STEP 7: Test how good our model is
    y_pred = model.predict(X_test)  # model makes predictions
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\n📊 Detailed Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Rain", "Rain"]))

    # STEP 8: Log results to MLflow (saves automatically)
    mlflow.log_param("n_estimators",  params["model"]["n_estimators"])
    mlflow.log_param("learning_rate", params["model"]["learning_rate"])
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    print("📝 Results saved to MLflow!")

# ─────────────────────────────────────────
# STEP 9: Save the trained model to models/ folder
# So we can use it later without retraining
# ─────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rainfall_model.pkl")
print("\n💾 Model saved to models/rainfall_model.pkl")