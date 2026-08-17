# ADAPT-CLONE

**ADAPT-CLONE** is a hybrid feature-based source code plagiarism detection and clone classification framework. It combines multiple code representation techniques (token-based, AST-based, PDG-based, semantic, and structural features) with machine learning models to detect and classify code clones across different clone types.

This repository contains the full pipeline: feature extraction, similarity computation, model training, ensemble-based clone type classification, and evaluation.

## 🚀 Features

- **Multi-representation feature extraction**:
  - Token-based similarity
  - AST-based features and similarity
  - PDG-based features and similarity
  - Semantic embeddings (e.g., UniXcoder)
  - Structural features
- **Hybrid similarity scoring**: Combines multiple feature-based similarities into a unified score.
- **Clone type classification**: Ensemble-based classifier to predict clone types (Type-1, Type-2, Type-3, etc.).
- **End-to-end runner**: Single entry point (`runner.py`) to orchestrate the full pipeline.
- **Reproducible results**: Serialized models, scalers, and metadata for exact re-runs.

## ▶️ How to Run

### Full pipeline (recommended)

From the repo root:

```bash
cd Approachs
python runner.py
```

`runner.py` orchestrates:

1. Feature extraction (token, AST, PDG, semantic, structure)
2. Similarity computation per feature type
3. Hybrid similarity aggregation
4. Clone detection and classification using trained models
5. Result export to `Results/ADAPT-CLONE_<timestamp>/`

Check the generated folder for:

- `pair_predictions.csv` — predicted clone labels per pair
- `evaluation_metrics.txt` — precision, recall, F1, ROC-AUC, etc.
- `cluster_report.csv` — clustering/ grouping of similar pairs (if applicable)
- `run_summary.json` — run configuration and summary stats

### Individual components

You can run parts of the pipeline independently but need to fix path accordingly:

```bash
# Token-based similarity
cd Approachs/Token_based
python token_similarity.py

# AST-based similarity
cd Approachs/AST_based
python Ast_similarity.py

# PDG-based similarity
cd Approachs/PDG_based
python pdg_similarity.py
# or
bash run_pdg_pipeline.sh

# Semantic similarity (UniXcoder)
cd Approachs/Semantic_based
python semantic_similarity.py

# Hybrid aggregation
cd Approachs/Hybrid_based
python hybrid_similarity.py

# Clone type classification
cd Approachs/Train_Model
python type_classifier_ensemble_compare.py
```

Results from each step are saved as CSV/PKL files in their respective folders.

## 🧠 Models and Artifacts

Pre-trained models and preprocessing artifacts are provided in:

- `Approachs/Adapt_model/` — ready-to-use models for inference
- `Approachs/Train_Model/` — training outputs, metrics, and comparison scripts

Key files:

- `final_mlp_mixed_8feat.pt` — MLP model for clone detection
- `type_classifier_ensemble_best.joblib` — best ensemble for clone type classification
- `*_scaler.joblib` — feature scalers for consistent preprocessing

To use these in your own scripts:

```python
import torch
import joblib

model = torch.load("Approachs/Adapt_model/final_mlp_mixed_8feat.pt")
scaler = joblib.load("Approachs/Adapt_model/final_mlp_mixed_8feat_scaler.joblib")
ensemble = joblib.load("Approachs/Adapt_model/type_classifier_ensemble_best.joblib")
```

## 📊 Results

Each run generates a timestamped folder under:

Example contents:

- `evaluation_metrics.txt` — detailed metrics (precision, recall, F1, ROC-AUC)
- `pair_predictions.csv` — per-pair predictions and confidence scores
- `cluster_report.csv` — cluster assignments for similar code pairs
- `run_summary.json` — hyperparameters, dataset stats, runtime info

## 🧪 Testing and Validation

- Use `Check_Results.py` in `Approachs/Hybrid_based/` to inspect and validate outputs.
- Compare ensemble variants using `type_classifier_ensemble_compare.py` in `Train_Model/`.
- External test predictions are available in `final_mlp_external_test_predictions.csv`.

## 📝 Notes

- Compiled parser binaries (`my-languages.so`, `.dll`, etc.) are included for AST/PDG extraction. Ensure they match your OS; rebuild if needed.
- For large-scale datasets (100M+ pairs), consider batching and parallelization adjustments in the similarity scripts.
- Paths and dataset formats may need minor tweaks depending on your environment.

## 🤝 Contributing

Feel free to fork, extend, or adapt this framework for your own plagiarism detection or clone classification research. If you use ADAPT-CLONE in your work, please cite appropriately.
