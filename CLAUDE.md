# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack Malay Pantun Theme Classification System classifying pantun into 11 themes using three model types: SVM (baseline), TextCNN, and MalayBERT.

```
frontend/ (Next.js 16 + TypeScript)  →  backend/ (Flask + Python)
   Port 3000                              Port 5000
```

## Commands

### Running the Application

```bash
# Terminal 1: Flask backend
cd backend
python app.py

# Terminal 2: Next.js frontend
cd frontend
npm run dev
```

Open http://localhost:3000

### Training

```bash
# Train all 6 SVM variants (70/30, 80/20, 90/10 × pembayang/no-pembayang)
cd backend
python train_svm.py
```

TextCNN and MalayBERT are trained on Google Colab via `notebooks/train_textcnn.ipynb` and `notebooks/train_malaybert.ipynb`. Trained model files go into `backend/models/`.

### Testing (requires backend running)

```bash
cd backend
python test_5_pantun.py   # Tests 5 pantun × 3 models
python test_7_pantun.py   # Tests 7 pantun × 3 models
python test_user_pantun.py
```

### Frontend

```bash
cd frontend
npm run lint
npm run build
```

## Architecture

### Backend (`backend/`)

- **`app.py`** — Flask API server; loads and caches models; routes `/api/classify`, `/api/themes`, `/api/metrics`, `/api/models`, `/api/health`
- **`preprocess.py`** — NLP pipeline: segmentation → case folding → tokenization → stopword removal → stemming (PySastrawi). The `use_pembayang` flag controls whether all 4 pantun lines or only lines 3–4 (maksud) are used.
- **`train_svm.py`** — Trains 6 SVM models (3 splits × 2 pembayang settings), saves joblib files and `metrics_summary.json` into `backend/models/`

### Model files in `backend/models/`

| Model | Files |
|---|---|
| SVM | `{key}_model.joblib`, `{key}_vectorizer.joblib`, `{key}_encoder.joblib` |
| TextCNN | `textcnn_best.pth` |
| MalayBERT | `malaybert_best/` directory (HuggingFace format) |
| Metadata | `metrics_summary.json`, `best_model.json` |

SVM model keys follow the pattern `svm_{split}_{pembayang}`, e.g. `svm_90-10_no_pembayang`.

### Frontend (`frontend/`)

- **`app/page.tsx`** — Main classification UI: 4-line pantun input, multi-model selection, settings panel, results with confidence bars and related pantun
- **`app/insights/page.tsx`** — Static insights/analysis page
- **`next.config.ts`** — Rewrites `/api/*` to `http://localhost:5000/api/*`, so all API calls from the frontend use relative paths

### Dataset

`pantun_dataset.json` at repo root — 9,124 pantun, each with `pantun` and `tema` fields. 11 standardized themes.

## Key Design Decisions

**Pembayang vs. Maksud**: A pantun has 4 lines — lines 1–2 are the pembayang (imagery/metaphor), lines 3–4 are the maksud (meaning/theme). The `use_pembayang=False` setting (default) uses only lines 3–4 and consistently outperforms using all 4 lines, because theme meaning resides in the maksud.

**Model accuracy**: MalayBERT (~59.7%) > SVM 90/10 (~55.1%) > TextCNN (~46.9%). MalayBERT uses `mesolitica/bert-base-standard-bahasa-cased`.

**Model loading**: Models are lazy-loaded and cached in `_models_cache` dict in `app.py` on first request.

**SVM TF-IDF config**: `max_features=10000`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`. SVM uses `kernel='rbf'`, `C=10`, `class_weight='balanced'`.
