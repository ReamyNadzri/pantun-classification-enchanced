"""
Flask API Server for Malay Pantun Theme Classification
Endpoints:
  POST /api/classify   - Classify a pantun
  GET  /api/themes     - Get theme list with counts
  GET  /api/metrics    - Get model performance metrics
  GET  /api/models     - Get available models list
"""

import json
import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_text, preprocess_batch

app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(PARENT_DIR, "pantun_dataset.json")

# Cache for loaded models and data
_models_cache = {}
_dataset_cache = None
_metrics_cache = None


def load_dataset():
    """Load the pantun dataset."""
    global _dataset_cache
    if _dataset_cache is None:
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            _dataset_cache = json.load(f)
    return _dataset_cache


def load_metrics():
    """Load metrics summary."""
    global _metrics_cache
    if _metrics_cache is None:
        metrics_path = os.path.join(MODELS_DIR, "metrics_summary.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                _metrics_cache = json.load(f)
        else:
            _metrics_cache = {}
    return _metrics_cache


def get_model(model_key):
    """Load a model, vectorizer, and label encoder by key."""
    if model_key not in _models_cache:
        model_path = os.path.join(MODELS_DIR, f"{model_key}_model.joblib")
        vectorizer_path = os.path.join(MODELS_DIR, f"{model_key}_vectorizer.joblib")
        encoder_path = os.path.join(MODELS_DIR, f"{model_key}_encoder.joblib")
        
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]):
            return None
        
        _models_cache[model_key] = {
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(vectorizer_path),
            "label_encoder": joblib.load(encoder_path)
        }
    
    return _models_cache[model_key]


def get_available_models():
    """Get list of available model keys."""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith("_model.joblib"):
                key = f.replace("_model.joblib", "")
                models.append(key)
    return sorted(models)


def find_related_pantun(theme, count=5):
    """Find related pantun from the dataset with the same theme."""
    dataset = load_dataset()
    related = [item for item in dataset if item['tema'] == theme]
    
    # Random sample
    if len(related) > count:
        import random
        related = random.sample(related, count)
    
    return related


@app.route('/api/classify', methods=['POST'])
def classify():
    """
    Classify a Malay pantun.
    
    Request body:
    {
        "pantun": "line1; line2; line3; line4",
        "model": "svm_80-20_no_pembayang",  (optional, defaults to best)
        "use_pembayang": false,              (optional)
        "top_k": 3,                          (optional, default 1)
        "show_steps": false,                 (optional)
        "related_count": 5                   (optional)
    }
    """
    data = request.get_json()
    
    if not data or 'pantun' not in data:
        return jsonify({"error": "Missing 'pantun' field"}), 400
    
    pantun_text = data['pantun']
    use_pembayang = data.get('use_pembayang', False)
    top_k = data.get('top_k', 1)
    show_steps = data.get('show_steps', False)
    related_count = data.get('related_count', 5)
    confidence_threshold = data.get('confidence_threshold', 0)
    
    # Determine which model to use
    model_key = data.get('model')
    if not model_key:
        # Use best model
        best_path = os.path.join(MODELS_DIR, "best_model.json")
        if os.path.exists(best_path):
            with open(best_path, 'r') as f:
                best_info = json.load(f)
            model_key = best_info.get("best_model", "svm_80-20_no_pembayang")
        else:
            model_key = "svm_80-20_no_pembayang"
    
    # Adjust model key based on pembayang setting
    # Extract base key (e.g., "svm_90-10") and append correct suffix
    if "_no_pembayang" in model_key:
        base_key = model_key.replace("_no_pembayang", "")
    elif "_pembayang" in model_key:
        base_key = model_key.replace("_pembayang", "")
    else:
        base_key = model_key
    
    suffix = "_pembayang" if use_pembayang else "_no_pembayang"
    model_key = base_key + suffix
    
    # Load model
    model_data = get_model(model_key)
    if not model_data:
        return jsonify({"error": f"Model '{model_key}' not found"}), 404
    
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    label_encoder = model_data["label_encoder"]
    
    # Preprocess
    processed_text, steps = preprocess_text(pantun_text, use_pembayang=use_pembayang)
    
    # Vectorize
    X = vectorizer.transform([processed_text])
    
    # Predict with probabilities
    probabilities = model.predict_proba(X)[0]
    
    # Get top-K predictions
    top_indices = np.argsort(probabilities)[::-1][:top_k]
    
    predictions = []
    for idx in top_indices:
        theme = label_encoder.inverse_transform([idx])[0]
        confidence = float(probabilities[idx])
        predictions.append({
            "theme": theme,
            "confidence": round(confidence * 100, 2)
        })
    
    # Check confidence threshold
    top_prediction = predictions[0]
    is_uncertain = top_prediction["confidence"] < confidence_threshold
    
    # Get related pantun
    related = find_related_pantun(top_prediction["theme"], related_count) if related_count > 0 else []
    
    # Build response
    response = {
        "predictions": predictions,
        "top_prediction": {
            "theme": top_prediction["theme"],
            "confidence": top_prediction["confidence"],
            "is_uncertain": is_uncertain
        },
        "model_used": model_key,
        "use_pembayang": use_pembayang,
        "related_pantun": related
    }
    
    if show_steps:
        # Convert steps to serializable format
        serializable_steps = {}
        for key, value in steps.items():
            if isinstance(value, list):
                serializable_steps[key] = value
            else:
                serializable_steps[key] = str(value)
        response["preprocessing_steps"] = serializable_steps
    
    return jsonify(response)


@app.route('/api/themes', methods=['GET'])
def get_themes():
    """Get list of themes with sample counts."""
    dataset = load_dataset()
    
    theme_counts = {}
    for item in dataset:
        theme = item['tema']
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    themes = [
        {"theme": theme, "count": count}
        for theme, count in sorted(theme_counts.items(), key=lambda x: -x[1])
    ]
    
    return jsonify({
        "themes": themes,
        "total_pantun": len(dataset),
        "total_themes": len(themes)
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics."""
    metrics = load_metrics()
    
    # Get best model info
    best_path = os.path.join(MODELS_DIR, "best_model.json")
    best_info = {}
    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            best_info = json.load(f)
    
    return jsonify({
        "models": metrics,
        "best_model": best_info,
        "available_models": get_available_models()
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """Get list of available models."""
    models = get_available_models()
    metrics = load_metrics()
    
    model_list = []
    for key in models:
        info = {
            "key": key,
            "type": "SVM",
            "metrics": metrics.get(key, {}).get("metrics", {})
        }
        model_list.append(info)
    
    return jsonify({"models": model_list})


@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "models_loaded": len(_models_cache),
        "available_models": get_available_models()
    })


if __name__ == '__main__':
    print("Starting Malay Pantun Classification API...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Available models: {get_available_models()}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
