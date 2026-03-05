"""
SVM Model Training for Malay Pantun Theme Classification
Trains SVM with 3 data split ratios (70-30, 80-20, 90-10),
evaluates performance, and saves the best model.
"""

import json
import os
import sys
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_batch

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PARENT_DIR, "pantun_dataset.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_dataset():
    """Load the cleaned pantun dataset."""
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item['pantun'] for item in data]
    labels = [item['tema'] for item in data]
    
    return texts, labels


def train_and_evaluate(texts, labels, test_size, split_name, use_pembayang=False):
    """
    Train SVM model with given split ratio and evaluate.
    
    Returns:
        dict with model, vectorizer, metrics, and performance data
    """
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SPLIT: {split_name} | Pembayang: {'ON' if use_pembayang else 'OFF'}")
    print(f"{'='*60}")
    
    # Preprocess all texts
    print("  Preprocessing texts...")
    processed_texts = preprocess_batch(texts, use_pembayang=use_pembayang)
    
    # Remove empty processed texts
    valid_indices = [i for i, t in enumerate(processed_texts) if t.strip()]
    processed_texts = [processed_texts[i] for i in valid_indices]
    filtered_labels = [labels[i] for i in valid_indices]
    
    print(f"  Valid texts after preprocessing: {len(processed_texts)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(filtered_labels)
    
    # TF-IDF Vectorization
    print("  Applying TF-IDF vectorization...")
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X = tfidf.fit_transform(processed_texts)
    y = encoded_labels
    
    print(f"  TF-IDF features: {X.shape[1]}")
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Testing samples: {X_test.shape[0]}")
    
    # Train SVM
    print("  Training SVM classifier...")
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        class_weight='balanced',
        probability=True,  # Enable probability estimates
        random_state=42
    )
    svm.fit(X_train, y_train)
    
    # Predict
    y_pred = svm.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class report
    class_names = label_encoder.classes_
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\n  RESULTS:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Classification Report:\n{report}")
    
    # Per-class metrics as dict
    per_class = {}
    report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    for cls_name in class_names:
        per_class[cls_name] = {
            "precision": round(report_dict[cls_name]["precision"], 4),
            "recall": round(report_dict[cls_name]["recall"], 4),
            "f1_score": round(report_dict[cls_name]["f1-score"], 4),
            "support": int(report_dict[cls_name]["support"])
        }
    
    return {
        "model": svm,
        "vectorizer": tfidf,
        "label_encoder": label_encoder,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "split": split_name,
            "use_pembayang": use_pembayang,
            "train_size": X_train.shape[0],
            "test_size": X_test.shape[0],
            "num_features": X.shape[1]
        },
        "per_class_metrics": per_class,
        "confusion_matrix": conf_matrix.tolist()
    }


def main():
    print("=" * 60)
    print("MALAY PANTUN SVM CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    texts, labels = load_dataset()
    print(f"  Total samples: {len(texts)}")
    
    # Define split ratios
    splits = [
        (0.30, "70-30"),
        (0.20, "80-20"),
        (0.10, "90-10"),
    ]
    
    all_results = {}
    best_f1 = 0
    best_split = None
    
    # Train for each split (without pembayang - default)
    for test_size, split_name in splits:
        result = train_and_evaluate(texts, labels, test_size, split_name, use_pembayang=False)
        all_results[f"svm_{split_name}_no_pembayang"] = result
        
        if result["metrics"]["f1_score"] > best_f1:
            best_f1 = result["metrics"]["f1_score"]
            best_split = f"svm_{split_name}_no_pembayang"
    
    # Also train with pembayang for comparison
    for test_size, split_name in splits:
        result = train_and_evaluate(texts, labels, test_size, split_name, use_pembayang=True)
        all_results[f"svm_{split_name}_pembayang"] = result
        
        if result["metrics"]["f1_score"] > best_f1:
            best_f1 = result["metrics"]["f1_score"]
            best_split = f"svm_{split_name}_pembayang"
    
    # Save all models
    print(f"\n{'='*60}")
    print("SAVING MODELS")
    print(f"{'='*60}")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save metrics summary
    metrics_summary = {}
    
    for name, result in all_results.items():
        model_path = os.path.join(MODELS_DIR, f"{name}_model.joblib")
        vectorizer_path = os.path.join(MODELS_DIR, f"{name}_vectorizer.joblib")
        encoder_path = os.path.join(MODELS_DIR, f"{name}_encoder.joblib")
        
        joblib.dump(result["model"], model_path)
        joblib.dump(result["vectorizer"], vectorizer_path)
        joblib.dump(result["label_encoder"], encoder_path)
        
        metrics_summary[name] = {
            "metrics": result["metrics"],
            "per_class_metrics": result["per_class_metrics"],
            "confusion_matrix": result["confusion_matrix"]
        }
        
        print(f"  Saved: {name}")
        print(f"    Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"    F1-Score: {result['metrics']['f1_score']:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(MODELS_DIR, "metrics_summary.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n  Best model: {best_split} (F1: {best_f1:.4f})")
    
    # Save best model info
    best_info = {"best_model": best_split, "best_f1": best_f1}
    with open(os.path.join(MODELS_DIR, "best_model.json"), 'w') as f:
        json.dump(best_info, f, indent=2)
    
    print(f"\n  All models and metrics saved to: {MODELS_DIR}")
    
    # Summary table
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Model':<35s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, result in all_results.items():
        m = result["metrics"]
        marker = " ★" if name == best_split else ""
        print(f"  {name:<35s} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1_score']:>10.4f}{marker}")


if __name__ == "__main__":
    main()
