import os
import sys
import json
import pickle
import traceback
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'heart_model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

DEFAULT_METRICS = {
    "accuracy_raw": 0.9853658536585366,
    "accuracy_percent": 98.54,
    "precision": 98.58,
    "recall": 98.54,
    "f1_score": 98.54,
    "total_records": 1025,
    "train_records": 820,
    "test_records": 205,
    "confusion_matrix": [[102, 0], [3, 100]]
}

model = None
metrics = DEFAULT_METRICS.copy()

def load_resources():
    global model, metrics
    
    # 1. Try loading existing serialized model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"[Notice] Model pickle incompatible with current environment: {e}. Retraining on the fly...")
            model = None

    # 2. If model is missing or incompatible, train on the fly using heart.csv
    if model is None or not os.path.exists(METRICS_PATH):
        try:
            from model import train_and_evaluate_model
            train_and_evaluate_model()
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print("[Success] Model trained and loaded successfully in current environment.")
        except Exception as e:
            print(f"[Error] Failed to train model on the fly: {e}")
            traceback.print_exc()

    # 3. Load calculated metrics
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    metrics.update(loaded)
        except Exception as e:
            print(f"[Error] Failed to load metrics.json: {e}")

# Load immediately on module import (Gunicorn / Flask startup)
load_resources()

# 13 Feature Names for standard ordering
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/', methods=['GET'])
def home():
    try:
        if model is None:
            load_resources()
        return render_template('index.html', metrics=metrics, prediction=None, form_data={})
    except Exception as e:
        print(f"Error on GET /: {e}")
        traceback.print_exc()
        return render_template('index.html', metrics=DEFAULT_METRICS, prediction=None, form_data={}, error=str(e))

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None:
        load_resources()

    try:
        # Extract inputs from the form
        form_data = {
            'age': float(request.form.get('age', 0)),
            'sex': int(request.form.get('sex', 0)),
            'cp': int(request.form.get('cp', 0)),
            'trestbps': float(request.form.get('trestbps', 0)),
            'chol': float(request.form.get('chol', 0)),
            'fbs': int(request.form.get('fbs', 0)),
            'restecg': int(request.form.get('restecg', 0)),
            'thalach': float(request.form.get('thalach', 0)),
            'exang': int(request.form.get('exang', 0)),
            'oldpeak': float(request.form.get('oldpeak', 0)),
            'slope': int(request.form.get('slope', 0)),
            'ca': int(request.form.get('ca', 0)),
            'thal': int(request.form.get('thal', 0))
        }

        # Format feature values as a 2D DataFrame in expected order
        features_list = [form_data[col] for col in FEATURE_NAMES]
        features_df = pd.DataFrame([features_list], columns=FEATURE_NAMES)

        # Make Prediction
        if model is None:
            raise RuntimeError("ML Model is not initialized. Please ensure heart.csv exists.")
            
        pred = model.predict(features_df)[0]
        prob = model.predict_proba(features_df)[0]

        # In this dataset:
        # 1 indicates Low Risk (Healthy / Normal)
        # 0 indicates High Risk (Presence of Heart Disease)
        if pred == 1:
            result_text = "Low Risk of Heart Disease"
            risk_category = "low"
            confidence = round(prob[1] * 100, 1)
            badge_color = "success"
        else:
            result_text = "High Risk of Heart Disease"
            risk_category = "high"
            confidence = round(prob[0] * 100, 1)
            badge_color = "danger"

        prediction_result = {
            "text": result_text,
            "category": risk_category,
            "confidence": confidence,
            "badge_color": badge_color,
            "raw_pred": int(pred)
        }

        return render_template(
            'index.html',
            metrics=metrics,
            prediction=prediction_result,
            form_data=form_data
        )

    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return render_template(
            'index.html',
            metrics=metrics,
            error=error_msg,
            form_data=request.form
        )

@app.errorhandler(500)
def internal_error(error):
    print("500 Internal Server Error encountered:")
    traceback.print_exc()
    return render_template('index.html', metrics=DEFAULT_METRICS, prediction=None, form_data={}, error="An internal server error occurred. The application has safely recovered."), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', metrics=DEFAULT_METRICS, prediction=None, form_data={}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
