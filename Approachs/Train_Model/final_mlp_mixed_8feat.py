#!/usr/bin/env python3
"""
Final MLP Training Script — CodeNet + External-Train Mix, Eval on Held Test + SOCO
-----------------------------------------------------------------------------------
Design requested by user:
- Same MLP-style setup as SOCO
- Use ALL 8 features
- No column fallback logic; files are already clean
- GroundTruth is the label in every file
- Proper distance normalization
- Final probability in [0,1]
- Chunk-wise balanced CodeNet loading
- Use external labeled set in training with train/test split so model learns T4 patterns
- Evaluate on BOTH:
    1) held-out external test split
    2) SOCO evaluation file

Pipeline:
1) Load CodeNet chunk-wise, reservoir-sample balanced train pool
2) Load external labeled set, stratified split into ext_train / ext_valid / ext_test
3) Mix CodeNet train pool + ext_train to form MLP training set
4) Use ext_valid as model-selection / threshold-selection validation set
5) Keep ext_test untouched for final honest test
6) Separately evaluate the same saved model on SOCO file

Important:
- Distances unixcoder_l2 and unixcoder_manhattan are converted to similarity via 1/(1+d)
- MinMaxScaler fitted on TRAIN ONLY so all features stay in [0,1]
- MLP output is sigmoid probability in [0,1]
- Threshold selected on ext_valid only
- Final metrics reported on ext_test and SOCO
"""

import os, gc, json, time, math, signal, warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from copy import deepcopy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
CODENET_PARQUET = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based/combined_codenet.parquet"
CODENET_CSV     = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based/combined_codenet.csv"
EXTERNAL_PARQUET = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based/combined_labeled.parquet"
EXTERNAL_CSV     = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based/combined_labeled.csv"
SOCO_PARQUET     = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based/combined_soco.parquet"
SOCO_CSV         = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based/combined_soco.csv"

OUT_DIR = "/home/24mcmi06/DEMO_TEST/Approachs/Hybrid_based/Codenet_label"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH     = os.path.join(OUT_DIR, "final_mlp_mixed_8feat.pt")
SCALER_PATH    = os.path.join(OUT_DIR, "final_mlp_mixed_8feat_scaler.joblib")
META_JSON      = os.path.join(OUT_DIR, "final_mlp_mixed_8feat_meta.json")
TRAIN_LOG_CSV  = os.path.join(OUT_DIR, "final_mlp_train_history.csv")
EXT_TEST_CSV   = os.path.join(OUT_DIR, "final_mlp_external_test_predictions.csv")
SOCO_TEST_CSV  = os.path.join(OUT_DIR, "final_mlp_soco_predictions.csv")
VALID_CSV      = os.path.join(OUT_DIR, "final_mlp_valid_predictions.csv")
SESSION_LOG    = os.path.join(OUT_DIR, "final_mlp_session.log")

# ─────────────────────────────────────────────
# FEATURES / LABEL
# ─────────────────────────────────────────────
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
LABEL_COL = "GroundTruth"

DISTANCE_COLS = ["unixcoder_l2", "unixcoder_manhattan"]

# ─────────────────────────────────────────────
# EXTERNAL SPLIT
# ─────────────────────────────────────────────
EXT_TEST_SIZE  = 0.20
EXT_VALID_SIZE = 0.20   # of remaining 80%, so final ≈ 64/16/20 train/valid/test
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# CODENET CHUNKED BALANCE
# ─────────────────────────────────────────────
MAX_CODENET_POS = 500_000
NEG_POS_RATIO   = 5
CODENET_CHUNK_CSV = 250_000

# ─────────────────────────────────────────────
# MLP HYPERPARAMETERS (SOCO-like)
# ─────────────────────────────────────────────
HIDDEN_DIMS = [64, 32, 16, 8]
DROPOUT     = 0.4
LR          = 1e-3
EPOCHS      = 400
BATCH_SIZE  = 4096
PATIENCE    = 30
USE_SAMPLER = False
LOSS_TYPE   = "focal"   # focal or bce
POS_WEIGHT  = 200.0      # as user showed from SOCO run metadata
GAMMA       = 2.0        # focal gamma
ALPHA       = 0.25       # focal alpha

# ─────────────────────────────────────────────
# THRESHOLD SEARCH
# ─────────────────────────────────────────────
THRESH_MIN  = 0.30
THRESH_MAX  = 0.95
THRESH_STEP = 0.01

# ─────────────────────────────────────────────
# SYSTEM
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────
_log_file = open(SESSION_LOG, "a", buffering=1)
def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out, flush=True)
    _log_file.write(out + "\n")

# ─────────────────────────────────────────────
# CTRL+C SAFE EXIT
# ─────────────────────────────────────────────
INTERRUPTED = False
def _handle_sigint(sig, frame):
    global INTERRUPTED
    INTERRUPTED = True
    log("[WARN] Ctrl+C detected — will stop after current epoch and save best state.")
signal.signal(signal.SIGINT, _handle_sigint)

# ─────────────────────────────────────────────
# RESERVOIR SAMPLER
# ─────────────────────────────────────────────
class Reservoir:
    def __init__(self, capacity, seed=42):
        self.capacity = int(capacity)
        self.records = []
        self.n_seen = 0
        self.rng = np.random.default_rng(seed)

    def add_df(self, df):
        for rec in df.itertuples(index=False):
            self.n_seen += 1
            d = rec._asdict()
            if len(self.records) < self.capacity:
                self.records.append(d)
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j < self.capacity:
                    self.records[j] = d

    def to_df(self):
        return pd.DataFrame(self.records) if self.records else pd.DataFrame()

# ─────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────
def load_table(parquet_path, csv_path):
    if Path(parquet_path).exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)

def parquet_chunks(path):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        yield pf.read_row_group(rg).to_pandas()

def csv_chunks(path, chunksize=250_000):
    for chunk in pd.read_csv(path, chunksize=chunksize):
        yield chunk

def read_chunks(parquet_path, csv_path):
    if Path(parquet_path).exists():
        yield from parquet_chunks(parquet_path)
    else:
        yield from csv_chunks(csv_path, CODENET_CHUNK_CSV)

# ─────────────────────────────────────────────
# FEATURE PREP
# ─────────────────────────────────────────────
def prepare_df(df):
    # user explicitly said files are clean and no fallback needed
    df = df.copy()
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Proper distance normalization: convert distance -> similarity in [0,1]
    for c in DISTANCE_COLS:
        df[c] = 1.0 / (1.0 + df[c].clip(lower=0.0))

    # Final clip to [0,1] to guarantee normalized feature space
    for c in FEATURE_COLS:
        df[c] = df[c].clip(0.0, 1.0)

    df[LABEL_COL] = df[LABEL_COL].astype(np.int8)
    return df

# ─────────────────────────────────────────────
# CODENET BALANCED TRAIN POOL (chunk-wise)
# ─────────────────────────────────────────────
def build_codenet_balanced_pool():
    log("Streaming CodeNet chunk-wise and building balanced pool...")
    pos_res = Reservoir(MAX_CODENET_POS, seed=RANDOM_SEED)
    neg_res = Reservoir(MAX_CODENET_POS * NEG_POS_RATIO, seed=RANDOM_SEED + 1)

    total_rows = total_pos = total_neg = 0

    for i, raw in enumerate(read_chunks(CODENET_PARQUET, CODENET_CSV), start=1):
        df = prepare_df(raw)
        total_rows += len(df)
        total_pos += int((df[LABEL_COL] == 1).sum())
        total_neg += int((df[LABEL_COL] == 0).sum())

        pos_res.add_df(df[df[LABEL_COL] == 1])
        neg_res.add_df(df[df[LABEL_COL] == 0])

        if i % 10 == 0:
            log(f"  Chunk {i}: rows={total_rows:,} pos={total_pos:,} neg={total_neg:,}")
        del df, raw
        gc.collect()

    pos_df = pos_res.to_df()
    neg_df = neg_res.to_df()
    need_neg = min(len(neg_df), len(pos_df) * NEG_POS_RATIO)
    if len(neg_df) > need_neg:
        neg_df = neg_df.sample(n=need_neg, random_state=RANDOM_SEED)

    out = pd.concat([pos_df, neg_df], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    info = {
        "raw_rows": int(total_rows),
        "raw_pos": int(total_pos),
        "raw_neg": int(total_neg),
        "sampled_rows": int(len(out)),
        "sampled_pos": int(out[LABEL_COL].sum()),
        "sampled_neg": int((out[LABEL_COL] == 0).sum()),
        "neg_pos_ratio": NEG_POS_RATIO,
    }
    log(f"CodeNet balanced pool: rows={len(out):,} pos={int(out[LABEL_COL].sum()):,} neg={int((out[LABEL_COL]==0).sum()):,}")
    return out, info

# ─────────────────────────────────────────────
# EXTERNAL / SOCO
# ─────────────────────────────────────────────
def build_external_splits():
    log("Loading external labeled set...")
    ext_df = prepare_df(load_table(EXTERNAL_PARQUET, EXTERNAL_CSV))
    log(f"External total: rows={len(ext_df):,} pos={int(ext_df[LABEL_COL].sum()):,} neg={int((ext_df[LABEL_COL]==0).sum()):,}")

    train_valid_df, test_df = train_test_split(
        ext_df,
        test_size=EXT_TEST_SIZE,
        stratify=ext_df[LABEL_COL],
        random_state=RANDOM_SEED,
    )
    valid_fraction_of_train_valid = EXT_VALID_SIZE
    train_df, valid_df = train_test_split(
        train_valid_df,
        test_size=valid_fraction_of_train_valid,
        stratify=train_valid_df[LABEL_COL],
        random_state=RANDOM_SEED,
    )

    info = {
        "ext_total_rows": int(len(ext_df)),
        "ext_total_pos": int(ext_df[LABEL_COL].sum()),
        "ext_total_neg": int((ext_df[LABEL_COL] == 0).sum()),
        "ext_train_rows": int(len(train_df)),
        "ext_train_pos": int(train_df[LABEL_COL].sum()),
        "ext_train_neg": int((train_df[LABEL_COL] == 0).sum()),
        "ext_valid_rows": int(len(valid_df)),
        "ext_valid_pos": int(valid_df[LABEL_COL].sum()),
        "ext_valid_neg": int((valid_df[LABEL_COL] == 0).sum()),
        "ext_test_rows": int(len(test_df)),
        "ext_test_pos": int(test_df[LABEL_COL].sum()),
        "ext_test_neg": int((test_df[LABEL_COL] == 0).sum()),
    }
    log(f"External train: rows={len(train_df):,} pos={int(train_df[LABEL_COL].sum()):,} neg={int((train_df[LABEL_COL]==0).sum()):,}")
    log(f"External valid: rows={len(valid_df):,} pos={int(valid_df[LABEL_COL].sum()):,} neg={int((valid_df[LABEL_COL]==0).sum()):,}")
    log(f"External test : rows={len(test_df):,} pos={int(test_df[LABEL_COL].sum()):,} neg={int((test_df[LABEL_COL]==0).sum()):,}")
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True), info

def load_soco():
    log("Loading SOCO evaluation set...")
    soco_df = prepare_df(load_table(SOCO_PARQUET, SOCO_CSV))
    info = {
        "soco_rows": int(len(soco_df)),
        "soco_pos": int(soco_df[LABEL_COL].sum()),
        "soco_neg": int((soco_df[LABEL_COL] == 0).sum()),
    }
    log(f"SOCO eval: rows={len(soco_df):,} pos={int(soco_df[LABEL_COL].sum()):,} neg={int((soco_df[LABEL_COL]==0).sum()):,}")
    return soco_df.reset_index(drop=True), info

# ─────────────────────────────────────────────
# DATASET / MODEL
# ─────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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
        logits = self.net(x).squeeze(1)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * (1 - pt).pow(self.gamma) * bce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fit_scaler(train_df):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train_df[FEATURE_COLS].values.astype(np.float32))
    return scaler

def transform_df(df, scaler):
    x = scaler.transform(df[FEATURE_COLS].values.astype(np.float32))
    x = np.clip(x, 0.0, 1.0)
    y = df[LABEL_COL].values.astype(np.float32)
    return x, y

def make_loader(x, y, batch_size, shuffle=False, sampler=None):
    ds = PairDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
                      sampler=sampler, num_workers=0, pin_memory=False)

def build_sampler(y):
    y = np.asarray(y)
    class_counts = np.bincount(y.astype(int), minlength=2)
    weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = weights[y.astype(int)]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

def collect_probs(model, loader):
    model.eval()
    probs_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_all.append(probs)
            y_all.append(yb.numpy())
    probs_all = np.concatenate(probs_all)
    y_all = np.concatenate(y_all)
    probs_all = np.clip(probs_all, 0.0, 1.0)
    return probs_all, y_all

def eval_at_threshold(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0,1]).ravel()
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "preds": preds,
    }

def find_best_threshold(y_true, probs):
    best = None
    for t in np.arange(THRESH_MIN, THRESH_MAX + 1e-9, THRESH_STEP):
        t = float(round(t, 2))
        m = eval_at_threshold(y_true, probs, t)
        m["threshold"] = t
        if best is None:
            best = m
            continue
        if m["f1"] > best["f1"]:
            best = m
        elif m["f1"] == best["f1"] and m["precision"] > best["precision"]:
            best = m
        elif m["f1"] == best["f1"] and m["precision"] == best["precision"] and m["recall"] > best["recall"]:
            best = m
    return best

def save_predictions(df, probs, preds, path):
    out = df.copy()
    out["probability"] = np.clip(probs, 0.0, 1.0)
    out["pred"] = preds.astype(int)
    out.to_csv(path, index=False)

# ─────────────────────────────────────────────
# TRAIN ONE MODEL
# ─────────────────────────────────────────────
def train_model(train_loader, valid_loader, input_dim):
    model = MLPClassifier(input_dim=input_dim, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    pos_weight_t = torch.tensor([POS_WEIGHT], dtype=torch.float32, device=DEVICE)
    if LOSS_TYPE == "focal":
        criterion = FocalLoss(alpha=ALPHA, gamma=GAMMA, reduction="mean", pos_weight=pos_weight_t)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    history = []
    best_state = None
    best_valid_f1 = -1.0
    best_valid_threshold = 0.5
    best_epoch = -1
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        if INTERRUPTED:
            log("Interrupt requested. Stopping training loop after this epoch boundary.")
            break

        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        valid_probs, valid_y = collect_probs(model, valid_loader)
        valid_best = find_best_threshold(valid_y, valid_probs)

        record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "valid_f1": valid_best["f1"],
            "valid_precision": valid_best["precision"],
            "valid_recall": valid_best["recall"],
            "valid_threshold": valid_best["threshold"],
            "valid_tp": valid_best["tp"],
            "valid_fp": valid_best["fp"],
            "valid_fn": valid_best["fn"],
            "valid_tn": valid_best["tn"],
        }
        history.append(record)

        if valid_best["f1"] > best_valid_f1:
            best_valid_f1 = valid_best["f1"]
            best_valid_threshold = valid_best["threshold"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            no_improve = 0
            flag = "BEST"
        else:
            no_improve += 1
            flag = ""

        log(
            f"Epoch {epoch:03d}/{EPOCHS} | loss={avg_loss:.6f} | "
            f"valid_f1={valid_best['f1']:.4f} P={valid_best['precision']:.4f} "
            f"R={valid_best['recall']:.4f} @t={valid_best['threshold']:.2f} {flag}"
        )

        if no_improve >= PATIENCE:
            log(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(TRAIN_LOG_CSV, index=False)

    return model, hist_df, best_epoch, best_valid_f1, best_valid_threshold

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
start_ts = datetime.now().strftime("%Y-%m-%d %H:%M")
log("=" * 80)
log("FINAL MLP MIXED-TRAIN SCRIPT (8 features, SOCO-style)")
log(f"Started: {start_ts}")
log(f"Device: {DEVICE}")
log(f"Features: {FEATURE_COLS}")
log(f"Distance normalization: 1 / (1 + d)")
log("Final feature normalization: MinMaxScaler fitted on TRAIN only -> [0,1]")
log("Final model output: sigmoid probability -> [0,1]")
log("=" * 80)

# 1) CodeNet chunk-wise balanced pool
codenet_pool, codenet_info = build_codenet_balanced_pool()

# 2) External splits
ext_train, ext_valid, ext_test, external_info = build_external_splits()

# 3) SOCO eval set
soco_df, soco_info = load_soco()

# 4) Mixed training set: CodeNet balanced pool + external train
log("Mixing CodeNet pool + external train for final MLP training set...")
train_df = pd.concat([codenet_pool, ext_train], axis=0).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
log(f"Mixed train: rows={len(train_df):,} pos={int(train_df[LABEL_COL].sum()):,} neg={int((train_df[LABEL_COL]==0).sum()):,}")
log(f"Validation : rows={len(ext_valid):,} pos={int(ext_valid[LABEL_COL].sum()):,} neg={int((ext_valid[LABEL_COL]==0).sum()):,}")
log(f"Ext Test   : rows={len(ext_test):,} pos={int(ext_test[LABEL_COL].sum()):,} neg={int((ext_test[LABEL_COL]==0).sum()):,}")
log(f"SOCO Eval  : rows={len(soco_df):,} pos={int(soco_df[LABEL_COL].sum()):,} neg={int((soco_df[LABEL_COL]==0).sum()):,}")

# 5) Scaler fit on TRAIN only
log("Fitting MinMaxScaler on TRAIN only...")
scaler = fit_scaler(train_df)
joblib.dump(scaler, SCALER_PATH)

x_train, y_train = transform_df(train_df, scaler)
x_valid, y_valid = transform_df(ext_valid, scaler)
x_test, y_test   = transform_df(ext_test, scaler)
x_soco, y_soco   = transform_df(soco_df, scaler)

for name, x in [("train", x_train), ("valid", x_valid), ("test", x_test), ("soco", x_soco)]:
    mn = float(np.min(x))
    mx = float(np.max(x))
    log(f"Feature range check [{name}] -> min={mn:.4f}, max={mx:.4f}")

# 6) Loaders
sampler = build_sampler(y_train) if USE_SAMPLER else None
train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=not USE_SAMPLER, sampler=sampler)
valid_loader = make_loader(x_valid, y_valid, BATCH_SIZE, shuffle=False)
test_loader  = make_loader(x_test, y_test, BATCH_SIZE, shuffle=False)
soco_loader  = make_loader(x_soco, y_soco, BATCH_SIZE, shuffle=False)

# 7) Train MLP
log("Starting MLP training...")
model, hist_df, best_epoch, best_valid_f1, best_threshold = train_model(train_loader, valid_loader, input_dim=len(FEATURE_COLS))

# 8) Save model checkpoint
ckpt = {
    "model_state_dict": model.state_dict(),
    "feature_cols": FEATURE_COLS,
    "hidden_dims": HIDDEN_DIMS,
    "dropout": DROPOUT,
    "best_epoch": best_epoch,
    "best_valid_f1": best_valid_f1,
    "best_threshold": best_threshold,
    "distance_norm": "1/(1+d)",
}
torch.save(ckpt, MODEL_PATH)
log(f"Saved best model -> {MODEL_PATH}")

# 9) Validation predictions (for record)
valid_probs, valid_y = collect_probs(model, valid_loader)
valid_metrics = eval_at_threshold(valid_y, valid_probs, best_threshold)
save_predictions(ext_valid, valid_probs, valid_metrics["preds"], VALID_CSV)

# 10) External TEST evaluation
log("Evaluating on held-out external test...")
test_probs, test_y = collect_probs(model, test_loader)
test_metrics = eval_at_threshold(test_y, test_probs, best_threshold)
save_predictions(ext_test, test_probs, test_metrics["preds"], EXT_TEST_CSV)

# 11) SOCO evaluation
log("Evaluating on SOCO...")
soco_probs, soco_y = collect_probs(model, soco_loader)
soco_metrics = eval_at_threshold(soco_y, soco_probs, best_threshold)
save_predictions(soco_df, soco_probs, soco_metrics["preds"], SOCO_TEST_CSV)

# 12) Print reports
log("\n" + "=" * 80)
log("FINAL RESULTS")
log("=" * 80)
log(f"Best epoch      : {best_epoch}")
log(f"Best valid F1   : {best_valid_f1:.4f}")
log(f"Best threshold  : {best_threshold:.2f}")

log("\nValidation metrics:")
log(f"  F1={valid_metrics['f1']:.4f} P={valid_metrics['precision']:.4f} R={valid_metrics['recall']:.4f}")
log(f"  TP={valid_metrics['tp']:,} FP={valid_metrics['fp']:,} FN={valid_metrics['fn']:,} TN={valid_metrics['tn']:,}")

log("\nHeld-out External Test metrics:")
log(f"  F1={test_metrics['f1']:.4f} P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f}")
log(f"  TP={test_metrics['tp']:,} FP={test_metrics['fp']:,} FN={test_metrics['fn']:,} TN={test_metrics['tn']:,}")
log("\nExternal classification report:")
log(classification_report(test_y.astype(int), test_metrics['preds'], target_names=['Non-Clone','Clone'], zero_division=0))

log("\nSOCO metrics:")
log(f"  F1={soco_metrics['f1']:.4f} P={soco_metrics['precision']:.4f} R={soco_metrics['recall']:.4f}")
log(f"  TP={soco_metrics['tp']:,} FP={soco_metrics['fp']:,} FN={soco_metrics['fn']:,} TN={soco_metrics['tn']:,}")
log("\nSOCO classification report:")
log(classification_report(soco_y.astype(int), soco_metrics['preds'], target_names=['Non-Clone','Clone'], zero_division=0))

# 13) Save meta
meta = {
    "timestamp": datetime.now().isoformat(),
    "device": DEVICE,
    "feature_cols": FEATURE_COLS,
    "distance_normalization": "1/(1+d)",
    "final_feature_normalization": "MinMaxScaler(train_only)->[0,1]",
    "final_probability_normalization": "sigmoid->[0,1]",
    "hyperparameters": {
        "hidden_dims": HIDDEN_DIMS,
        "dropout": DROPOUT,
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "loss_type": LOSS_TYPE,
        "pos_weight": POS_WEIGHT,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "use_sampler": USE_SAMPLER,
    },
    "best_epoch": int(best_epoch),
    "best_valid_f1": float(best_valid_f1),
    "best_threshold": float(best_threshold),
    "codenet_info": codenet_info,
    "external_info": external_info,
    "soco_info": soco_info,
    "validation_metrics": {
        k: valid_metrics[k] for k in ["f1","precision","recall","tp","fp","fn","tn"]
    },
    "external_test_metrics": {
        k: test_metrics[k] for k in ["f1","precision","recall","tp","fp","fn","tn"]
    },
    "soco_metrics": {
        k: soco_metrics[k] for k in ["f1","precision","recall","tp","fp","fn","tn"]
    },
}
with open(META_JSON, "w") as f:
    json.dump(meta, f, indent=2)

log("\nSaved artifacts:")
log(f"  Model      : {MODEL_PATH}")
log(f"  Scaler     : {SCALER_PATH}")
log(f"  Meta       : {META_JSON}")
log(f"  Train hist : {TRAIN_LOG_CSV}")
log(f"  Valid pred : {VALID_CSV}")
log(f"  Ext pred   : {EXT_TEST_CSV}")
log(f"  SOCO pred  : {SOCO_TEST_CSV}")
log(f"  Session log: {SESSION_LOG}")

_log_file.close()
print("[INFO] Done.")
