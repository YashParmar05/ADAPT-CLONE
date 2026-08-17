#!/usr/bin/env python3
"""
Type Classifier — Tree Ensemble Comparison
Compares Decision Tree, Random Forest, and Extra Trees on the same type-labelled clone pairs.
Saves separate outputs with distinct filenames under /Codenet_label/Classification.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo
from itertools import product
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# =====================================================
# PATH CONFIG
# =====================================================
timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d_%H-%M-%S")

PROJECT_DIR  = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based"
INPUT_PARQUET = os.path.join(PROJECT_DIR, "combined_labeled_8.parquet")
INPUT_CSV     = os.path.join(PROJECT_DIR, "combined_labeled_8.csv")

CLASSIFICATION_DIR = os.path.join(PROJECT_DIR, "Codenet_label", "Classification")
os.makedirs(CLASSIFICATION_DIR, exist_ok=True)

OUT_SEARCH    = os.path.join(CLASSIFICATION_DIR, "type_classifier_ensemble_search.csv")
OUT_JSON      = os.path.join(CLASSIFICATION_DIR, "type_classifier_ensemble_best.json")
OUT_METRICS   = os.path.join(CLASSIFICATION_DIR, "type_classifier_ensemble_metrics.txt")
OUT_PAIRS     = os.path.join(CLASSIFICATION_DIR, "type_classifier_ensemble_pairs.csv")
OUT_MODEL     = os.path.join(CLASSIFICATION_DIR, "type_classifier_ensemble_best.joblib")
OUT_SCALER    = os.path.join(CLASSIFICATION_DIR, "type_classifier_ensemble_scaler.joblib")

print("\n==========================================")
print("CLONE TYPE CLASSIFIER — ENSEMBLE COMPARISON")
print("Run:", timestamp)
print("==========================================\n")

# =====================================================
# FEATURES
# =====================================================
FEATURE_COLS = [
    "S_token",
    "S_AST",
    "S_structure",
    "unixcoder_cosine",
    "unixcoder_l2",
    "unixcoder_manhattan",
    "S_PDG",
    "pdg_valid",
]
DIST_COLS     = ["unixcoder_l2", "unixcoder_manhattan"]
PROBLEM1_COL  = "problem_1"
PROBLEM2_COL  = "problem_2"
FILE1_COL     = "file_1"
FILE2_COL     = "file_2"
GT_COL        = "GroundTruth"
TYPE_COL      = "clone_type"
TYPE_MAP      = {1: "Type-1", 2: "Type-2", 3: "Type-3", 4: "Type-4"}

# =====================================================
# MODEL GRIDS
# =====================================================
MODELS = {
    "DecisionTree": (
        DecisionTreeClassifier,
        [
            {"max_depth": d, "min_samples_leaf": l, "min_samples_split": s, "criterion": c, "class_weight": cw}
            for d in [3, 4, 5, 6, 7, None]
            for l in [1, 2, 3, 5]
            for s in [2, 4, 6, 10]
            for c in ["gini", "entropy"]
            for cw in [None, "balanced"]
        ]
    ),
    "RandomForest": (
        RandomForestClassifier,
        [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": l, "criterion": c, "class_weight": cw}
            for n in [100, 200, 300]
            for d in [4, 5, 6, None]
            for l in [1, 2, 3, 5]
            for c in ["gini", "entropy"]
            for cw in [None, "balanced"]
        ]
    ),
    "ExtraTrees": (
        ExtraTreesClassifier,
        [
            {"n_estimators": n, "max_depth": d, "min_samples_leaf": l, "criterion": c, "class_weight": cw}
            for n in [100, 200, 300]
            for d in [4, 5, 6, None]
            for l in [1, 2, 3, 5]
            for c in ["gini", "entropy"]
            for cw in [None, "balanced"]
        ]
    ),
}
CV_FOLDS = 5
RANDOM_STATE = 42
MAX_CONFIGS_PER_MODEL = 200  # cap for runtime control

# =====================================================
# LOAD DATA
# =====================================================
print("[INFO] Loading data...")
if os.path.exists(INPUT_PARQUET):
    df = pd.read_parquet(INPUT_PARQUET)
else:
    df = pd.read_csv(INPUT_CSV)

print(f"[INFO] Total rows      : {len(df):,}")
print(f"[INFO] Columns         : {df.columns.tolist()}")

# =====================================================
# NORMALIZE DISTANCES FIRST
# =====================================================
print("[INFO] Normalizing distance features...")
for c in FEATURE_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).replace([float('inf'), float('-inf')], 0.0)
for c in DIST_COLS:
    df[c] = 1.0 / (1.0 + df[c].clip(lower=0.0))
for c in FEATURE_COLS:
    df[c] = df[c].clip(0.0, 1.0).astype(np.float32)

# =====================================================
# EXTRACT TYPE-LABELLED PAIRS
# =====================================================
def strip_ext(f):
    f = str(f).strip()
    return f[:-4] if f.endswith('.cpp') else f

def get_type_from_file(f):
    base = strip_ext(f)
    if not base.startswith('s'):
        return None
    try:
        num = int(base[1:])
    except ValueError:
        return None
    if 10 <= num <= 19:
        return 1
    if 20 <= num <= 29:
        return 2
    if 30 <= num <= 39:
        return 3
    if 40 <= num <= 49:
        return 4
    return None

same_prob = df[(df[PROBLEM1_COL] == df[PROBLEM2_COL]) & (df[GT_COL] == 1)].copy()

def valid_type_pair(row):
    f1 = strip_ext(str(row[FILE1_COL]))
    f2 = strip_ext(str(row[FILE2_COL]))
    if f1 == 's1':
        return get_type_from_file(row[FILE2_COL])
    if f2 == 's1':
        return get_type_from_file(row[FILE1_COL])
    return None

labels = same_prob.apply(valid_type_pair, axis=1)
mask = labels.notna()
type_df = same_prob[mask].copy()
type_df[TYPE_COL] = labels[mask].astype(int)

print(f"[INFO] Type-labelled pairs found : {len(type_df):,}")
print(type_df[TYPE_COL].value_counts().sort_index().rename(index=TYPE_MAP).to_string())

X = type_df[FEATURE_COLS].values.astype(np.float32)
y = type_df[TYPE_COL].values.astype(int)

scaler = MinMaxScaler(feature_range=(0.0, 1.0))
X_scaled = scaler.fit_transform(X)
X_scaled = np.clip(X_scaled, 0.0, 1.0).astype(np.float32)
joblib.dump(scaler, OUT_SCALER)
print(f"[INFO] Scaler saved: {OUT_SCALER}")

# =====================================================
# SEARCH
# =====================================================
search_rows = []
best = None
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for model_name, (cls, grid) in MODELS.items():
    combos = grid[:MAX_CONFIGS_PER_MODEL]
    print(f"\n[INFO] Searching {model_name} over {len(combos):,} configs...")
    for i, params in enumerate(combos, 1):
        params = dict(params)
        params['random_state'] = RANDOM_STATE
        model = cls(**params)
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1_macro')
        row = {
            'model': model_name,
            **params,
            'cv_f1_macro_mean': float(scores.mean()),
            'cv_f1_macro_std': float(scores.std()),
        }
        search_rows.append(row)
        if best is None or row['cv_f1_macro_mean'] > best['cv_f1_macro_mean'] or (row['cv_f1_macro_mean'] == best['cv_f1_macro_mean'] and row['cv_f1_macro_std'] < best['cv_f1_macro_std']):
            best = row
        if i % 25 == 0 or i == len(combos):
            print(f"  tested {i}/{len(combos)} | best={best['model']} {best['cv_f1_macro_mean']:.4f} ± {best['cv_f1_macro_std']:.4f}")

search_df = pd.DataFrame(search_rows).sort_values(['cv_f1_macro_mean', 'cv_f1_macro_std'], ascending=[False, True])
search_df.to_csv(OUT_SEARCH, index=False)
print(f"\n[INFO] Search log saved: {OUT_SEARCH}")

# =====================================================
# TRAIN BEST MODEL
# =====================================================
print("\n[INFO] Training best model on all data...")
BestCls, _ = MODELS[best['model']]
best_params = {k: v for k, v in best.items() if k not in ['model', 'cv_f1_macro_mean', 'cv_f1_macro_std']}
best_model = BestCls(**best_params)
best_model.fit(X_scaled, y)
joblib.dump(best_model, OUT_MODEL)
print(f"[INFO] Best model saved: {OUT_MODEL}")

# =====================================================
# EVALUATION
# =====================================================
y_pred = best_model.predict(X_scaled)
acc  = accuracy_score(y, y_pred)
f1   = f1_score(y, y_pred, average='macro', zero_division=0)
prec = precision_score(y, y_pred, average='macro', zero_division=0)
rec  = recall_score(y, y_pred, average='macro', zero_division=0)
cm   = confusion_matrix(y, y_pred, labels=[1, 2, 3, 4])
report = classification_report(y, y_pred, labels=[1, 2, 3, 4], target_names=["Type-1", "Type-2", "Type-3", "Type-4"], zero_division=0)

print("\n==========================================")
print("TYPE CLASSIFICATION METRICS")
print("==========================================")
print(f"Accuracy   : {acc:.4f}")
print(f"F1 (macro) : {f1:.4f}")
print(f"Precision  : {prec:.4f}")
print(f"Recall     : {rec:.4f}")
print("\nConfusion Matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm, index=["T1","T2","T3","T4"], columns=["T1","T2","T3","T4"]).to_string())
print(f"\nClassification Report:\n{report}")

# feature importance if available
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n[INFO] Top feature importances:")
    print(imp.to_string())

if best['model'] == 'DecisionTree':
    rules = export_text(best_model, feature_names=FEATURE_COLS, max_depth=4)
else:
    rules = "N/A for ensemble model"

# =====================================================
# SAVE PAIRS
# =====================================================
type_df = type_df.copy()
type_df['pred_type'] = y_pred
type_df['pred_type_label'] = type_df['pred_type'].map(TYPE_MAP)
type_df['true_type_label'] = type_df[TYPE_COL].map(TYPE_MAP)
pair_cols = [PROBLEM1_COL, FILE1_COL, PROBLEM2_COL, FILE2_COL, TYPE_COL, 'true_type_label', 'pred_type', 'pred_type_label'] + FEATURE_COLS
type_df[pair_cols].to_csv(OUT_PAIRS, index=False)
print(f"\n[INFO] Pair predictions saved: {OUT_PAIRS}")

# =====================================================
# SAVE JSON / METRICS
# =====================================================
summary = {
    'timestamp': timestamp,
    'type': 'clone_type_classifier_ensemble_comparison',
    'cv_f1_macro_mean': round(float(best['cv_f1_macro_mean']), 4),
    'cv_f1_macro_std': round(float(best['cv_f1_macro_std']), 4),
    'train_metrics': {
        'accuracy': round(float(acc), 4),
        'f1_macro': round(float(f1), 4),
        'precision_macro': round(float(prec), 4),
        'recall_macro': round(float(rec), 4),
    },
    'best_model': best['model'],
    'best_params': best_params,
    'type_distribution': type_df[TYPE_COL].value_counts().sort_index().rename(index=TYPE_MAP).to_dict(),
    'features': FEATURE_COLS,
}
with open(OUT_JSON, 'w') as f:
    json.dump(summary, f, indent=2)

with open(OUT_METRICS, 'w') as f:
    f.write(f"Clone Type Classifier — Ensemble Comparison\n")
    f.write(f"Timestamp : {timestamp}\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Best Model         : {best['model']}\n")
    f.write(f"Best CV F1 (macro) : {best['cv_f1_macro_mean']:.4f} ± {best['cv_f1_macro_std']:.4f}\n\n")
    f.write(f"Accuracy   : {acc:.4f}\n")
    f.write(f"F1 (macro) : {f1:.4f}\n")
    f.write(f"Precision  : {prec:.4f}\n")
    f.write(f"Recall     : {rec:.4f}\n\n")
    f.write(f"Best Params : {json.dumps(best_params)}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(pd.DataFrame(cm, index=["T1","T2","T3","T4"], columns=["T1","T2","T3","T4"]).to_string())
    f.write("\n\nClassification Report:\n")
    f.write(report)
    f.write("\n\nRules / Notes:\n")
    f.write(rules)

print(f"[INFO] JSON saved   : {OUT_JSON}")
print(f"[INFO] Metrics saved: {OUT_METRICS}")
print(f"[INFO] Best model   : {best['model']}")
print(f"[INFO] Best CV F1   : {best['cv_f1_macro_mean']:.4f} ± {best['cv_f1_macro_std']:.4f}")
print(f"[INFO] Best params  : {best_params}")

print("\n==========================================")
print("✅ type_classifier_ensemble_compare.py complete")
print("==========================================\n")
