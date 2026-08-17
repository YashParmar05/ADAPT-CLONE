import os
os.environ["HF_HUB_DISABLE_XET"] = "1"

import sys
import torch
import numpy as np
import re
import csv
import pickle
import hashlib
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

BATCH_SIZE  = 32
BATCH_WRITE = 10_000

MODELS_CONFIG = {
    "unixcoder": "microsoft/unixcoder-base",
}

# =====================================================
# DATASET FINGERPRINT
# =====================================================
def get_dataset_fingerprint(root_folder):
    entries = []
    for problem in sorted(os.listdir(root_folder)):
        prob_path = os.path.join(root_folder, problem)
        if not os.path.isdir(prob_path):
            continue
        for fname in sorted(os.listdir(prob_path)):
            if fname.endswith((".cpp", ".c++")):
                size = os.path.getsize(os.path.join(prob_path, fname))
                entries.append(f"{problem}/{fname}:{size}")
    return hashlib.md5("\n".join(entries).encode()).hexdigest()

# =====================================================
# CODE CLEANING
# =====================================================
def clean_code(code: str) -> str:
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    code = re.sub(r'//.*', '', code)
    code = re.sub(
        r'^\s*#\s*(include|define|pragma|ifndef|ifdef|endif|undef|else|elif|if)\b.*$',
        '', code, flags=re.MULTILINE | re.IGNORECASE
    )
    code = re.sub(
        r'^\s*(using\s+namespace\s+\w+|import\s+[\w.]+)\s*;?\s*$',
        '', code, flags=re.MULTILINE
    )
    code = re.sub(r'"[^"]*"',          'STRING_LITERAL', code)
    code = re.sub(r"'[^']*'",          'CHAR_LITERAL',   code)
    code = re.sub(r'\b\d+(\.\d+)?\b',  'NUM_LITERAL',    code)
    code = '\n'.join(line.rstrip() for line in code.splitlines())
    code = re.sub(r'\n{3,}', '\n\n', code)
    code = re.sub(r'[ \t]+', ' ',    code)
    return code.strip()

# =====================================================
# ENCODING - sliding window mean pooling
# =====================================================
def encode_single(text, tokenizer, model):
    tokens  = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    max_len = 512
    stride  = 256

    if len(tokens) <= max_len:
        enc = tokenizer(text, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb  = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(emb, dim=1)[0]
    else:
        chunk_embs = []
        for start in range(0, len(tokens), stride):
            chunk = tokens[start: start + max_len]
            ids   = chunk.unsqueeze(0).to(device)
            mask  = torch.ones_like(ids)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask)
            body = out.last_hidden_state[0, 1:-1]
            chunk_embs.append(body.mean(0))
            if start + max_len >= len(tokens):
                break
        emb = torch.stack(chunk_embs).mean(0, keepdim=True)
        return torch.nn.functional.normalize(emb, dim=1)[0]

def encode_batch(texts, tokenizer, model):
    embs = []
    for text in texts:
        embs.append(encode_single(text, tokenizer, model).cpu().numpy())
    return np.stack(embs)

# =====================================================
# PHASE 1 - Compute & cache embeddings
#   - sorted(os.listdir) for consistent pair order
#   - stores only folder name, not full path
#   - auto-invalidates old full-path caches
# =====================================================
def compute_and_save_embeddings(root_folder, embeddings_path, tokenizer, model, model_name):
    current_fp = get_dataset_fingerprint(root_folder)

    if os.path.exists(embeddings_path):
        with open(embeddings_path, "rb") as f:
            cached = pickle.load(f)
        cached_fp     = cached.get("dataset_fingerprint")
        sample_prob   = cached.get("cpp_files", [("",)])[0][0]
        has_full_path = os.sep in str(sample_prob)

        if cached_fp == current_fp and not has_full_path:
            print(f"[INFO] [{model_name}] Cache valid - skipping recomputation.")
            return
        elif has_full_path:
            print(f"[WARN] [{model_name}] Old cache stores full paths - recomputing with fix...")
        else:
            print(f"[INFO] [{model_name}] Dataset changed - recomputing...")

    cpp_files = []
    for problem in sorted(os.listdir(root_folder)):
        prob_path = os.path.join(root_folder, problem)
        if not os.path.isdir(prob_path):
            continue
        for fname in sorted(os.listdir(prob_path)):
            if fname.endswith((".cpp", ".c++")):
                cpp_files.append((
                    problem,                           # "p1"
                    fname,                             # "s1001.cpp"
                    os.path.join(prob_path, fname),    # full path for reading only
                ))

    print(f"[INFO] [{model_name}] Total C++ files : {len(cpp_files)}")
    if cpp_files:
        print(f"[INFO] [{model_name}] First : {cpp_files[0][0]}/{cpp_files[0][1]}")
        print(f"[INFO] [{model_name}] Last  : {cpp_files[-1][0]}/{cpp_files[-1][1]}")

    texts = []
    for _, _, path in cpp_files:
        with open(path, "r", errors="ignore") as f:
            texts.append(clean_code(f.read()))

    print(f"[INFO] [{model_name}] Computing embeddings...")
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        emb   = encode_batch(batch, tokenizer, model)
        embeddings.append(emb)
        print(f"  [{model_name}] [{i + len(batch)}/{len(texts)}]")

    embeddings = np.vstack(embeddings)
    print(f"[INFO] [{model_name}] Embeddings shape: {embeddings.shape}")

    with open(embeddings_path, "wb") as f:
        pickle.dump({
            "dataset_fingerprint": current_fp,
            "cpp_files":           cpp_files,
            "embeddings":          embeddings,
        }, f)

    local_model_path = os.path.join(
        os.path.dirname(embeddings_path), "models", model_name
    )
    if not os.path.exists(local_model_path):
        print(f"[INFO] [{model_name}] Saving model locally: {local_model_path}")
        tokenizer.save_pretrained(local_model_path)
        model.save_pretrained(local_model_path)

    print(f"[INFO] [{model_name}] Embeddings saved: {embeddings_path}")

# =====================================================
# PHASE 2 - Pairwise similarity -> CSV output
# =====================================================
def compute_pairwise_similarity(embeddings_path, output_csv, model_name):
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)

    cpp_files = data["cpp_files"]
    E         = data["embeddings"]
    N         = len(cpp_files)
    total     = N * (N - 1) // 2
    print(f"[INFO] [{model_name}] Files: {N} | Pairs: {total:,}")
    print(f"[INFO] [{model_name}] First file in cache: {cpp_files[0][0]}/{cpp_files[0][1]}")

    CHUNK = 500

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "problem_1", "file_1", "problem_2", "file_2",
            f"{model_name}_cosine",
            f"{model_name}_l2",
            f"{model_name}_manhattan",
        ])
        buffer  = []
        written = 0

        for i_start in range(0, N, CHUNK):
            i_end     = min(i_start + CHUNK, N)
            chunk_emb = E[i_start:i_end]
            sim_block = chunk_emb @ E[i_start:].T

            for local_i, global_i in enumerate(range(i_start, i_end)):
                p1, f1, _ = cpp_files[global_i]

                for j_offset in range(global_i - i_start + 1, N - i_start):
                    global_j  = i_start + j_offset
                    p2, f2, _ = cpp_files[global_j]

                    e1, e2    = E[global_i], E[global_j]
                    cos_sim   = float(sim_block[local_i, j_offset])
                    l2_dist   = float(np.linalg.norm(e1 - e2))
                    manhattan = float(np.sum(np.abs(e1 - e2)))

                    buffer.append([
                        p1, f1, p2, f2,
                        round(cos_sim,   4),
                        round(l2_dist,   4),
                        round(manhattan, 4),
                    ])
                    written += 1

                    if len(buffer) >= BATCH_WRITE:
                        writer.writerows(buffer)
                        buffer = []

            print(f"[INFO] [{model_name}] Outer chunk {i_start}-{i_end}/{N} done")

        if buffer:
            writer.writerows(buffer)

    print(f"[DONE] [{model_name}] CSV saved : {output_csv}")
    print(f"[DONE] [{model_name}] Total pairs: {written:,}")

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    DATASET_PATH = sys.argv[1]
    PROJECT_DIR  = sys.argv[2]

    SEMANTIC_DIR = os.path.join(PROJECT_DIR, "Semantic_based")
    MODELS_DIR   = os.path.join(SEMANTIC_DIR, "models")
    os.makedirs(SEMANTIC_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,   exist_ok=True)

    for model_name, model_hub_path in MODELS_CONFIG.items():
        print(f"\n{'='*55}")
        print(f"  PROCESSING: {model_name.upper()}")
        print(f"{'='*55}")

        local_model_path = os.path.join(MODELS_DIR, model_name)
        embeddings_path  = os.path.join(SEMANTIC_DIR, f"embeddings_{model_name}.pkl")
        output_csv       = os.path.join(SEMANTIC_DIR, f"semantic_{model_name}_results.csv")

        load_path = local_model_path if os.path.exists(local_model_path) else model_hub_path
        if not os.path.exists(local_model_path):
            print(f"[WARN] Downloading from HuggingFace: {model_hub_path}")

        print(f"[INFO] Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(load_path)
        model_obj = AutoModel.from_pretrained(load_path).to(device)
        model_obj.eval()

        compute_and_save_embeddings(
            DATASET_PATH, embeddings_path,
            tokenizer, model_obj, model_name
        )

        compute_pairwise_similarity(embeddings_path, output_csv, model_name)

        del tokenizer, model_obj
        torch.cuda.empty_cache()
