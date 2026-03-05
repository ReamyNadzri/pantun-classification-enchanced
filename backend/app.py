"""
Flask API Server for Malay Pantun Theme Classification
Supports 3 model types: SVM, TextCNN, MalayBERT
Endpoints:
  POST /api/classify   - Classify a pantun
  GET  /api/themes     - Get theme list with counts
  GET  /api/metrics    - Get model performance metrics
  GET  /api/models     - Get available models list
"""

import json
import os
import sys
import re
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

# ============================================================
# TEXTCNN SUPPORT
# ============================================================

def _load_textcnn():
    """Load TextCNN model if available."""
    import torch
    import torch.nn as nn

    pth_path = os.path.join(MODELS_DIR, "textcnn_best.pth")
    if not os.path.exists(pth_path):
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)

    class TextCNN(nn.Module):
        def __init__(self, vocab_size, embed_dim, num_classes,
                     kernel_sizes=[3, 4, 5], num_filters=128, dropout=0.5):
            super(TextCNN, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters, kernel_size=ks)
                for ks in kernel_sizes
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        def forward(self, x):
            x = self.embedding(x)
            x = x.permute(0, 2, 1)
            conv_outs = []
            for conv in self.convs:
                c = torch.relu(conv(x))
                c = torch.max(c, dim=2)[0]
                conv_outs.append(c)
            x = torch.cat(conv_outs, dim=1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    vocab = checkpoint['vocab']
    label_classes = checkpoint['label_encoder_classes']
    max_len = checkpoint['max_len']
    embed_dim = checkpoint['embed_dim']
    kernel_sizes = checkpoint['kernel_sizes']
    num_filters = checkpoint['num_filters']
    num_classes = checkpoint['num_classes']

    model = TextCNN(
        vocab_size=len(vocab), embed_dim=embed_dim,
        num_classes=num_classes, kernel_sizes=kernel_sizes,
        num_filters=num_filters, dropout=0.0
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return {
        "model": model, "vocab": vocab, "label_classes": label_classes,
        "max_len": max_len, "device": device, "type": "textcnn",
        "use_pembayang": checkpoint.get('use_pembayang', False),
    }


def _predict_textcnn(model_data, text, use_pembayang, top_k):
    """Run prediction using TextCNN model."""
    import torch

    vocab = model_data["vocab"]
    max_len = model_data["max_len"]
    label_classes = model_data["label_classes"]
    device = model_data["device"]
    model = model_data["model"]

    # Preprocess using our pipeline
    processed_text, steps = preprocess_text(text, use_pembayang=use_pembayang)
    tokens = processed_text.split()

    # Convert tokens to indices
    indices = [vocab.get(t, vocab.get('<UNK>', 1)) for t in tokens[:max_len]]
    while len(indices) < max_len:
        indices.append(0)

    x = torch.LongTensor([indices]).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = []
    for idx in top_indices:
        predictions.append({
            "theme": label_classes[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
        })

    return predictions, steps


# ============================================================
# MALAYBERT SUPPORT
# ============================================================

def _load_malaybert():
    """Load MalayBERT model if available."""
    bert_dir = os.path.join(MODELS_DIR, "malaybert_best")
    config_path = os.path.join(bert_dir, "config.json")
    label_path = os.path.join(bert_dir, "label_classes.json")

    if not os.path.exists(config_path):
        return None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(label_path, 'r', encoding='utf-8') as f:
        label_info = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(bert_dir)
    model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(device)
    model.eval()

    return {
        "model": model, "tokenizer": tokenizer,
        "label_classes": label_info['classes'],
        "max_len": label_info.get('max_len', 128),
        "device": device, "type": "malaybert",
        "use_pembayang": label_info.get('use_pembayang', False),
    }


def _predict_malaybert(model_data, text, use_pembayang, top_k):
    """Run prediction using MalayBERT model."""
    import torch

    tokenizer = model_data["tokenizer"]
    model = model_data["model"]
    label_classes = model_data["label_classes"]
    device = model_data["device"]
    max_len = model_data["max_len"]

    # Segment pantun (no stemming for BERT - it uses its own tokenization)
    lines = re.split(r'[;,]', text)
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) >= 4:
        if use_pembayang:
            segmented = ' '.join(lines[:4])
        else:
            segmented = ' '.join(lines[2:4])
    else:
        segmented = ' '.join(lines)

    # Also get preprocessing steps for display
    _, steps = preprocess_text(text, use_pembayang=use_pembayang)

    # Tokenize
    encoding = tokenizer(
        segmented, truncation=True, max_length=max_len,
        padding='max_length', return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = []
    for idx in top_indices:
        predictions.append({
            "theme": label_classes[idx],
            "confidence": round(float(probs[idx]) * 100, 2),
        })

    return predictions, steps


# ============================================================
# SVM SUPPORT (existing)
# ============================================================

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


def get_svm_model(model_key):
    """Load a SVM model, vectorizer, and label encoder by key."""
    if model_key not in _models_cache:
        model_path = os.path.join(MODELS_DIR, f"{model_key}_model.joblib")
        vectorizer_path = os.path.join(MODELS_DIR, f"{model_key}_vectorizer.joblib")
        encoder_path = os.path.join(MODELS_DIR, f"{model_key}_encoder.joblib")

        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]):
            return None

        _models_cache[model_key] = {
            "model": joblib.load(model_path),
            "vectorizer": joblib.load(vectorizer_path),
            "label_encoder": joblib.load(encoder_path),
            "type": "svm",
        }

    return _models_cache[model_key]


def get_available_models():
    """Get list of available model keys (SVM + DL models)."""
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith("_model.joblib"):
                key = f.replace("_model.joblib", "")
                models.append(key)
        # Check for TextCNN
        if os.path.exists(os.path.join(MODELS_DIR, "textcnn_best.pth")):
            models.append("textcnn")
        # Check for MalayBERT
        if os.path.exists(os.path.join(MODELS_DIR, "malaybert_best", "config.json")):
            models.append("malaybert")
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


# ============================================================
# API ROUTES
# ============================================================

@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify a Malay pantun using SVM, TextCNN, or MalayBERT."""
    data = request.get_json()

    if not data or 'pantun' not in data:
        return jsonify({"error": "Missing 'pantun' field"}), 400

    pantun_text = data['pantun']
    use_pembayang = data.get('use_pembayang', False)
    top_k = data.get('top_k', 1)
    show_steps = data.get('show_steps', False)
    related_count = data.get('related_count', 5)
    confidence_threshold = data.get('confidence_threshold', 0)

    # Determine model
    model_key = data.get('model')
    if not model_key:
        best_path = os.path.join(MODELS_DIR, "best_model.json")
        if os.path.exists(best_path):
            with open(best_path, 'r') as f:
                best_info = json.load(f)
            model_key = best_info.get("best_model", "svm_80-20_no_pembayang")
        else:
            model_key = "svm_80-20_no_pembayang"

    # ---- ROUTE TO CORRECT MODEL ----

    if model_key == "textcnn":
        # TextCNN model
        if "textcnn" not in _models_cache:
            loaded = _load_textcnn()
            if loaded:
                _models_cache["textcnn"] = loaded
        if "textcnn" not in _models_cache:
            return jsonify({"error": "TextCNN model not found. Train it in Colab first."}), 404
        predictions, steps = _predict_textcnn(
            _models_cache["textcnn"], pantun_text, use_pembayang, top_k
        )

    elif model_key == "malaybert":
        # MalayBERT model
        if "malaybert" not in _models_cache:
            loaded = _load_malaybert()
            if loaded:
                _models_cache["malaybert"] = loaded
        if "malaybert" not in _models_cache:
            return jsonify({"error": "MalayBERT model not found. Train it in Colab first."}), 404
        predictions, steps = _predict_malaybert(
            _models_cache["malaybert"], pantun_text, use_pembayang, top_k
        )

    else:
        # SVM model
        # Adjust model key based on pembayang setting
        if "_no_pembayang" in model_key:
            base_key = model_key.replace("_no_pembayang", "")
        elif "_pembayang" in model_key:
            base_key = model_key.replace("_pembayang", "")
        else:
            base_key = model_key
        suffix = "_pembayang" if use_pembayang else "_no_pembayang"
        model_key = base_key + suffix

        model_data = get_svm_model(model_key)
        if not model_data:
            return jsonify({"error": f"Model '{model_key}' not found"}), 404

        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        label_encoder = model_data["label_encoder"]

        processed_text, steps = preprocess_text(pantun_text, use_pembayang=use_pembayang)
        X = vectorizer.transform([processed_text])
        probabilities = model.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            theme = label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            predictions.append({
                "theme": theme,
                "confidence": round(confidence * 100, 2),
            })

    # Build response
    top_prediction = predictions[0]
    is_uncertain = top_prediction["confidence"] < confidence_threshold

    related = find_related_pantun(top_prediction["theme"], related_count) if related_count > 0 else []

    response = {
        "predictions": predictions,
        "top_prediction": {
            "theme": top_prediction["theme"],
            "confidence": top_prediction["confidence"],
            "is_uncertain": is_uncertain,
        },
        "model_used": model_key,
        "use_pembayang": use_pembayang,
        "related_pantun": related,
    }

    if show_steps:
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
        "total_themes": len(themes),
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get model performance metrics."""
    metrics = load_metrics()

    best_path = os.path.join(MODELS_DIR, "best_model.json")
    best_info = {}
    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            best_info = json.load(f)

    # Also load DL model metrics if available
    for name, filename in [("textcnn", "textcnn_metrics.json"), ("malaybert", "malaybert_metrics.json")]:
        metrics_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                dl_metrics = json.load(f)
            metrics[name] = {"metrics": dl_metrics}

    return jsonify({
        "models": metrics,
        "best_model": best_info,
        "available_models": get_available_models(),
    })


@app.route('/api/models', methods=['GET'])
def list_models():
    """Get list of available models."""
    models = get_available_models()
    metrics = load_metrics()

    # Load DL metrics
    dl_metrics = {}
    for name, filename in [("textcnn", "textcnn_metrics.json"), ("malaybert", "malaybert_metrics.json")]:
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            with open(path, 'r') as f:
                dl_metrics[name] = json.load(f)

    model_list = []
    for key in models:
        if key == "textcnn":
            info = {"key": key, "type": "TextCNN",
                    "metrics": dl_metrics.get("textcnn", {})}
        elif key == "malaybert":
            info = {"key": key, "type": "MalayBERT",
                    "metrics": dl_metrics.get("malaybert", {})}
        else:
            info = {"key": key, "type": "SVM",
                    "metrics": metrics.get(key, {}).get("metrics", {})}
        model_list.append(info)

    return jsonify({"models": model_list})


@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "models_loaded": len(_models_cache),
        "available_models": get_available_models(),
    })


if __name__ == '__main__':
    print("Starting Malay Pantun Classification API...")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Available models: {get_available_models()}")

    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
