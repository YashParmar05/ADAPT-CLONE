#!/usr/bin/env python3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys
import json
import gc
from datetime import datetime
from zoneinfo import ZoneInfo
import networkx as nx
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings("ignore", category=Warning)

# =====================================================
# PATH CONFIG
# =====================================================
timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d_%H-%M-%S")

PROJECT_DIR = sys.argv[2]
# PROJECT_DIR = "/home/yash/SCPD_TEST_Final/Approachs"  
INPUT_PARQUET = os.path.join(PROJECT_DIR, "Hybrid_based", "combined.parquet")
INPUT_CSV     = os.path.join(PROJECT_DIR, "Hybrid_based", "combined.csv")

# INPUT_PARQUET = os.path.join(PROJECT_DIR, "Hybrid_based", "clone_classification.parquet")
# INPUT_CSV     = os.path.join(PROJECT_DIR, "Hybrid_based", "clone_classification.csv")


MODEL_PATH       = os.path.join(PROJECT_DIR, "Adapt_model", "final_mlp_mixed_8feat.pt")
SCALER_PATH      = os.path.join(PROJECT_DIR, "Adapt_model", "final_mlp_mixed_8feat_scaler.joblib")
META_PATH        = os.path.join(PROJECT_DIR, "Adapt_model", "final_mlp_mixed_8feat_meta.json")

TYPE_MODEL_PATH  = os.path.join(PROJECT_DIR, "Adapt_model", "type_classifier_ensemble_best.joblib")
TYPE_SCALER_PATH = os.path.join(PROJECT_DIR, "Adapt_model", "type_classifier_ensemble_scaler.joblib")
TYPE_META_PATH   = os.path.join(PROJECT_DIR, "Adapt_model", "type_classifier_ensemble_best.json")

RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

run_dir = os.path.join(RESULTS_DIR, f"ADAPT-CLONE: {timestamp}")
os.makedirs(run_dir, exist_ok=True)

OUTPUT_PAIR_CSV     = os.path.join(run_dir, "pair_predictions.csv")
OUTPUT_CLUSTER_CSV  = os.path.join(run_dir, "cluster_report.csv")
OUTPUT_METRICS_TXT  = os.path.join(run_dir, "evaluation_metrics.txt")
OUTPUT_SUMMARY_JSON = os.path.join(run_dir, "run_summary.json")

print("\n==========================================")
print("ADAPT-CLONE DETECTOR + TYPE CLASSIFIER RUN:", timestamp)
print("==========================================\n")

# =====================================================
# FEATURE LISTS
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
DIST_COLS  = ["unixcoder_l2", "unixcoder_manhattan"]
PRED_BATCH = 200_000
TYPE_MAP   = {1: "Type-1", 2: "Type-2", 3: "Type-3", 4: "Type-4"}

# =====================================================
# MODEL DEFINITION
# =====================================================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# =====================================================
# LOAD BINARY CLONE MODEL
# =====================================================
print("[INFO] Loading binary clone model, scaler and metadata...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
meta = {}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Binary model not found: {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Binary scaler not found: {SCALER_PATH}")

ckpt        = torch.load(MODEL_PATH, map_location=device)
HIDDEN_DIMS = ckpt.get("hidden_dims", [64, 32, 16, 8])
DROPOUT     = ckpt.get("dropout", 0.4)

model = MLPClassifier(len(FEATURE_COLS), HIDDEN_DIMS, DROPOUT).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

scaler         = joblib.load(SCALER_PATH)
PLAG_THRESHOLD = float(ckpt.get("best_threshold", 0.5))

if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)
    PLAG_THRESHOLD = float(meta.get("best_threshold", PLAG_THRESHOLD))
    print(f"[INFO] Binary model threshold : {PLAG_THRESHOLD}")
    print(f"[INFO] Best valid F1          : {meta.get('best_valid_f1')}")
else:
    print(f"[WARNING] Meta not found — using threshold {PLAG_THRESHOLD}")

CLUSTER_THRESHOLD = max(PLAG_THRESHOLD + 0.10, 0.30)
print(f"[INFO] Device                 : {device}")

# =====================================================
# LOAD TYPE CLASSIFIER
# =====================================================
print("\n[INFO] Loading ensemble type classifier...")
if not os.path.exists(TYPE_MODEL_PATH):
    raise FileNotFoundError(f"Type model not found: {TYPE_MODEL_PATH}")
if not os.path.exists(TYPE_SCALER_PATH):
    raise FileNotFoundError(f"Type scaler not found: {TYPE_SCALER_PATH}")

type_model  = joblib.load(TYPE_MODEL_PATH)
type_scaler = joblib.load(TYPE_SCALER_PATH)
type_meta   = {}
if os.path.exists(TYPE_META_PATH):
    with open(TYPE_META_PATH) as f:
        type_meta = json.load(f)
    print(f"[INFO] Type model             : {type_meta.get('best_model')}")
    print(f"[INFO] Type classifier CV F1  : {type_meta.get('cv_f1_macro_mean')}")
    print(f"[INFO] Type classifier train F1: {type_meta.get('train_metrics', {}).get('f1_macro')}")

# =====================================================
# LOAD DATA
# =====================================================
print("\n[INFO] Loading input data...")
if os.path.exists(INPUT_PARQUET):
    INPUT_FILE = INPUT_PARQUET
elif os.path.exists(INPUT_CSV):
    INPUT_FILE = INPUT_CSV
else:
    raise FileNotFoundError("Neither parquet nor csv found")

if INPUT_FILE.endswith(".parquet"):
    df = pd.read_parquet(INPUT_FILE)
else:
    df = pd.read_csv(INPUT_FILE)

print(f"[INFO] Total pairs   : {len(df):,}")
print(f"[INFO] Columns found : {df.columns.tolist()}")

# =====================================================
# NORMALIZE DISTANCES → SIMILARITY BEFORE SCALING
# =====================================================
print("[INFO] Normalizing distance features via 1/(1+d)...")
for c in FEATURE_COLS:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0).replace([np.inf, -np.inf], 0.0)

for c in DIST_COLS:
    df[c] = 1.0 / (1.0 + df[c].clip(lower=0.0))

for c in FEATURE_COLS:
    df[c] = df[c].clip(0.0, 1.0).astype(np.float32)

# =====================================================
# SCALE FOR BINARY MODEL
# =====================================================
X = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
X = np.clip(X, 0.0, 1.0).astype(np.float32)
print(f"[INFO] Feature range after binary scaler: min={X.min():.4f} max={X.max():.4f}")

# =====================================================
# BINARY CLONE INFERENCE
# =====================================================
print("\n[INFO] Running binary clone inference...")
X_tensor  = torch.tensor(X, dtype=torch.float32)
all_probs = []

with torch.no_grad():
    for start in range(0, len(X_tensor), PRED_BATCH):
        xb    = X_tensor[start:start + PRED_BATCH].to(device)
        probs = torch.sigmoid(model(xb)).detach().cpu().numpy()
        all_probs.append(probs)
        if start % 2_000_000 == 0 and start > 0:
            print(f"  {start:,} / {len(X_tensor):,} processed...")

plag_probs = np.clip(np.concatenate(all_probs), 0.0, 1.0).astype(np.float32)
del X_tensor, all_probs, X
gc.collect()

df["Plagiarism_Probability"] = plag_probs
df["Is_Plagiarized"]         = df["Plagiarism_Probability"] >= PLAG_THRESHOLD
print(f"[INFO] Predicted clone pairs : {df['Is_Plagiarized'].sum():,} / {len(df):,} ({df['Is_Plagiarized'].mean()*100:.2f}%)")

# =====================================================
# TYPE CLASSIFICATION — only on predicted clone pairs
# =====================================================
print("\n[INFO] Running ensemble type classification on predicted clone pairs...")
df["Predicted_Type"] = "Not-Clone"

clone_idx = df[df["Is_Plagiarized"]].index
if len(clone_idx) > 0:
    X_type = type_scaler.transform(df.loc[clone_idx, FEATURE_COLS].values.astype(np.float32))
    X_type = np.clip(X_type, 0.0, 1.0).astype(np.float32)
    type_preds = type_model.predict(X_type)
    df.loc[clone_idx, "Predicted_Type"] = pd.Series(type_preds, index=clone_idx).map(TYPE_MAP)
    print(f"[INFO] Type distribution on clone pairs:")
    print(df.loc[clone_idx, "Predicted_Type"].value_counts().sort_index().to_string())
else:
    print("[WARNING] No clone pairs predicted — type classifier not applied.")

# =====================================================
# SAVE PAIR-LEVEL OUTPUT
# =====================================================
pair_cols = [c for c in ["problem_1","file_1","problem_2","file_2"] if c in df.columns]
# pair_cols += ["Plagiarism_Probability", "Is_Plagiarized", "Predicted_Type"] + FEATURE_COLS
pair_cols += ["Plagiarism_Probability", "Is_Plagiarized", "Predicted_Type"]
df[pair_cols].to_csv(OUTPUT_PAIR_CSV, index=False)
print(f"\n[INFO] Pair predictions saved : {OUTPUT_PAIR_CSV}")

# =====================================================
# BUILD CLUSTERS
# =====================================================
print("\n[INFO] Building plagiarism clusters...")
cluster_rows = []
cluster_id   = 1

if all(c in df.columns for c in ["problem_1","problem_2","file_1","file_2"]):
    for problem in df["problem_1"].dropna().unique():
        sub_df = df[(df["problem_1"] == problem) & (df["problem_2"] == problem)]
        if sub_df.empty:
            continue

        G         = nx.Graph()
        all_files = set(sub_df["file_1"]).union(set(sub_df["file_2"]))
        G.add_nodes_from(all_files)

        for _, row in sub_df.iterrows():
            if row["Plagiarism_Probability"] >= CLUSTER_THRESHOLD:
                G.add_edge(
                    row["file_1"], row["file_2"],
                    probability=row["Plagiarism_Probability"],
                    clone_type=row["Predicted_Type"]
                )

        for cluster in nx.connected_components(G):
            if len(cluster) == 1:
                cluster_rows.append({
                    "Problem":            problem,
                    "Cluster_ID":         cluster_id,
                    "Cluster_Size":       1,
                    "Files":              list(cluster)[0],
                    "Pair_Relationships": "N/A — no clone pair detected",
                    "Avg_Probability":    0.0,
                    "Graph_Density":      0.0,
                    "Confidence":         "UNIQUE — no clone found"
                })
                cluster_id += 1
                continue

            subgraph  = G.subgraph(cluster)
            edge_data = list(subgraph.edges(data=True))
            if not edge_data:
                continue

            cluster_edges = [
                f"{u} - {v} ({d['clone_type']}, {round(d['probability'], 3)})"
                for u, v, d in edge_data
            ]
            avg_prob       = float(np.mean([d["probability"] for _, _, d in edge_data]))
            possible_pairs = len(cluster) * (len(cluster) - 1) // 2
            graph_density  = round(len(edge_data) / possible_pairs, 3) if possible_pairs > 0 else 0.0

            if avg_prob >= 0.80 and graph_density >= 0.80:
                confidence = "HIGH"
            elif avg_prob >= 0.50:
                confidence = "MEDIUM"
            else:
                confidence = "LOW — review manually"

            cluster_rows.append({
                "Problem":            problem,
                "Cluster_ID":         cluster_id,
                "Cluster_Size":       len(cluster),
                "Files":              ", ".join(sorted(cluster)),
                "Pair_Relationships": " | ".join(cluster_edges),
                "Avg_Probability":    round(avg_prob, 3),
                "Graph_Density":      graph_density,
                "Confidence":         confidence
            })
            cluster_id += 1
else:
    print("[WARNING] problem/file columns missing — cluster report skipped.")

cluster_df = pd.DataFrame(cluster_rows)
cluster_df.to_csv(OUTPUT_CLUSTER_CSV, index=False)
print(f"[INFO] Cluster report saved      : {OUTPUT_CLUSTER_CSV}")
print(f"[INFO] Total clusters            : {len(cluster_df):,}")

# =====================================================
# EVALUATION METRICS
# =====================================================
print("\n==========================================")
print("EVALUATION METRICS")
print("==========================================\n")

if "GroundTruth" in df.columns:
    df["label"] = df["GroundTruth"].astype(int)
elif all(c in df.columns for c in ["problem_1","problem_2"]):
    df["label"] = (df["problem_1"] == df["problem_2"]).astype(int)
else:
    raise ValueError("No GroundTruth column and cannot infer labels from problem columns.")

y_true = df["label"].values
y_pred = df["Is_Plagiarized"].astype(int).values

unique_true = np.unique(y_true)
unique_pred = np.unique(y_pred)
is_single_class = len(unique_true) < 2 or len(unique_pred) < 2

accuracy  = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0, labels=[0,1], average="binary")
recall    = recall_score(y_true, y_pred, zero_division=0, labels=[0,1], average="binary")
f1        = f1_score(y_true, y_pred, zero_division=0, labels=[0,1], average="binary")
cm        = confusion_matrix(y_true, y_pred, labels=[0,1])
tn, fp, fn, tp = cm.ravel()
report    = classification_report(y_true, y_pred, labels=[0,1], target_names=["Not Clone","Clone"], zero_division=0)

print("==== Binary Clone Metrics ====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
if is_single_class:
    present = "only CLONE" if unique_true[0] == 1 else "only NON-CLONE"
    print(f"\n[⚠️ ] Single-class dataset ({present}) — metrics may be limited")
print(f"\nConfusion Matrix:")
print(f"  TN={tn:,}  FP={fp:,}")
print(f"  FN={fn:,}  TP={tp:,}")
print(f"\nClassification Report:\n{report}")

if meta and "best_valid_f1" in meta:
    f1_drop = float(meta.get("best_valid_f1", f1)) - f1
    print("==== Generalization Check ====")
    if   f1_drop < 0.05: print(f"[✅] F1 drop : {f1_drop:+.4f}  — Stable generalization")
    elif f1_drop < 0.10: print(f"[⚠️ ] F1 drop : {f1_drop:+.4f}  — Mild degradation")
    else:                print(f"[❌] F1 drop : {f1_drop:+.4f}  — Significant drop, check for dataset shift")

# =====================================================
# OPTIONAL TYPE METRICS IF TRUE TYPE IS DERIVABLE
# =====================================================
print("\n==========================================")
print("TYPE CLASSIFICATION CHECK")
print("==========================================\n")

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

def derive_true_type(row):
    if row.get("problem_1") != row.get("problem_2"):
        return None
    f1 = strip_ext(row.get("file_1", ""))
    f2 = strip_ext(row.get("file_2", ""))
    if f1 == 's1':
        return get_type_from_file(row.get("file_2", ""))
    if f2 == 's1':
        return get_type_from_file(row.get("file_1", ""))
    return None

type_eval_df = df[df["Is_Plagiarized"]].copy()
if all(c in df.columns for c in ["problem_1","problem_2","file_1","file_2"]):
    true_types = type_eval_df.apply(derive_true_type, axis=1)
    type_eval_df = type_eval_df[true_types.notna()].copy()
    if len(type_eval_df) > 0:
        type_eval_df["true_type"] = true_types[true_types.notna()].astype(int)
        inv_type_map = {v:k for k,v in TYPE_MAP.items()}
        pred_num = type_eval_df["Predicted_Type"].map(inv_type_map)
        mask = pred_num.notna()
        type_eval_df = type_eval_df[mask].copy()
        pred_num = pred_num[mask].astype(int)

        if len(type_eval_df) > 0:
            type_f1 = f1_score(type_eval_df["true_type"], pred_num, average="macro", zero_division=0)
            type_report = classification_report(
                type_eval_df["true_type"], pred_num,
                labels=[1,2,3,4],
                target_names=["Type-1","Type-2","Type-3","Type-4"],
                zero_division=0
            )
            print(f"Type Macro-F1 : {type_f1:.4f}")
            print(f"\nType Classification Report:\n{type_report}")
        else:
            type_f1 = None
            type_report = "No type-evaluable predicted clone pairs."
            print(type_report)
    else:
        type_f1 = None
        type_report = "No rows with derivable true type."
        print(type_report)
else:
    type_f1 = None
    type_report = "Required pair columns missing for type evaluation."
    print(type_report)

# =====================================================
# SAVE METRICS TXT
# =====================================================
with open(OUTPUT_METRICS_TXT, "w") as f:
    f.write(f"FINAL MLP + ENSEMBLE TYPE CLASSIFIER RUN: {timestamp}\n")
    f.write("=" * 50 + "\n\n")
    if is_single_class:
        present = "only CLONE" if unique_true[0] == 1 else "only NON-CLONE"
        f.write(f"[WARNING] Single-class dataset ({present})\n\n")
    f.write("==== Binary Clone Metrics ====\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n\n")
    f.write(f"Threshold Used : {PLAG_THRESHOLD}\n")
    if meta:
        f.write(f"Best Valid F1  : {meta.get('best_valid_f1')}\n")
        f.write(f"Best Epoch     : {meta.get('best_epoch')}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"  TN={tn:,}  FP={fp:,}\n")
    f.write(f"  FN={fn:,}  TP={tp:,}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    if type_meta:
        f.write("\n==== Type Classifier Info ====\n")
        f.write(f"Best Model     : {type_meta.get('best_model')}\n")
        f.write(f"CV F1 (macro)  : {type_meta.get('cv_f1_macro_mean')}\n")
        f.write(f"Train F1 (macro): {type_meta.get('train_metrics', {}).get('f1_macro')}\n")
    f.write("\n==== Type Classification Check ====\n")
    if type_f1 is not None:
        f.write(f"Type Macro-F1 : {type_f1:.4f}\n\n")
    f.write(type_report)
print(f"[INFO] Metrics saved : {OUTPUT_METRICS_TXT}")

# =====================================================
# SAVE SUMMARY JSON
# =====================================================
summary = {
    "timestamp": timestamp,
    "input_file": INPUT_FILE,
    "binary_model": MODEL_PATH,
    "type_model": TYPE_MODEL_PATH,
    "threshold": float(PLAG_THRESHOLD),
    "cluster_threshold": float(CLUSTER_THRESHOLD),
    "n_rows": int(len(df)),
    "predicted_clones": int(df["Is_Plagiarized"].sum()),
    "predicted_clone_rate": float(df["Is_Plagiarized"].mean()),
    "clone_type_distribution": df.loc[df["Is_Plagiarized"], "Predicted_Type"].value_counts().to_dict(),
    "binary_metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    },
    "type_check_macro_f1": None if type_f1 is None else float(type_f1),
    "outputs": {
        "pair_predictions": OUTPUT_PAIR_CSV,
        "cluster_report": OUTPUT_CLUSTER_CSV,
        "evaluation_metrics": OUTPUT_METRICS_TXT,
    }
}
with open(OUTPUT_SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[INFO] Summary saved : {OUTPUT_SUMMARY_JSON}")

print(f"\n[✅] Pairs saved    : {OUTPUT_PAIR_CSV}")
print(f"[✅] Clusters saved : {OUTPUT_CLUSTER_CSV}")
print(f"[✅] Metrics saved  : {OUTPUT_METRICS_TXT}")
print(f"[✅] Summary saved  : {OUTPUT_SUMMARY_JSON}")
print("\n==========================================")
print("✅ final_mlp_deploy_test_ensemble.py complete")
print("==========================================\n")
