import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

def train_and_evaluate_model():
    print("=" * 60)
    print("Heart Disease Prediction Model Training (Team Deadlock)")
    print("=" * 60)

    # 1. Load Dataset
    print("\n[1] Loading UCI Heart Disease Dataset (heart.csv)...")
    df = pd.read_csv('heart.csv')
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Features and Target
    X = df.drop('target', axis=1)
    y = df['target']

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # 4. Train Random Forest Classifier
    print("\n[2] Training Random Forest Classifier...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Display Terminal Output exactly as required
    print(f"\nAccuracy: {acc}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # 6. Save Trained Model
    model_filename = 'heart_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved as {model_filename}")

    # 7. Save Actual Calculated Metrics for Web UI Display
    metrics_data = {
        "accuracy_raw": float(acc),
        "accuracy_percent": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "total_records": int(df.shape[0]),
        "train_records": int(X_train.shape[0]),
        "test_records": int(X_test.shape[0]),
        "confusion_matrix": cm.tolist()
    }

    with open('metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print("Calculated metrics saved to metrics.json")
    print("=" * 60)

if __name__ == '__main__':
    train_and_evaluate_model()
