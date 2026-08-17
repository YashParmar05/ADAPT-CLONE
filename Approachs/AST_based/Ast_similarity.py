import os
import sys
import csv
import hashlib
import pickle
from collections import Counter
from tree_sitter import Language, Parser

DATASET_ROOT = sys.argv[1]
PROJECT_DIR  = sys.argv[2]
TS_DLL       = os.path.join(PROJECT_DIR, "AST_based", "build", "lib-my-languages.so")
OUTPUT_CSV   = os.path.join(PROJECT_DIR, "AST_based", "AST_similarity.csv")
CACHE_PATH   = os.path.join(PROJECT_DIR, "AST_based", "ast_features.pkl")

CPP_LANGUAGE = Language(TS_DLL, "cpp")
parser = Parser()
parser.set_language(CPP_LANGUAGE)

# =====================================================
# SUBTREE HASHING — rename invariant
# =====================================================
def hash_subtree(node, counter):
    child_hashes = []
    for child in node.children:
        h = hash_subtree(child, counter)
        if h:
            child_hashes.append(h)
    child_hashes.sort()

    # Normalize leaf nodes — ignore variable names and literal values
    if node.child_count == 0:
        if node.type in ("identifier", "number_literal",
                         "string_literal", "char_literal",
                         "float_literal", "true", "false"):
            label = "LEAF_" + node.type
        else:
            label = node.type
    else:
        label = node.type

    combined = label + "_" + "_".join(child_hashes)
    h = hashlib.md5(combined.encode()).hexdigest()
    counter[h] += 1
    return h

# =====================================================
# PHASE 1 — Extract and cache AST features per file
# =====================================================
def compute_and_save_ast_features(dataset_root, cache_path):
    # if os.path.exists(cache_path):
    #     print(f"[INFO] AST cache exists, skipping extraction.")
    #     return

    print("[INFO] Extracting AST features...")
    ast_data = {}

    for problem in sorted(os.listdir(dataset_root)):
        problem_path = os.path.join(dataset_root, problem)
        if not os.path.isdir(problem_path):
            continue
        for file in sorted(os.listdir(problem_path)):
            if not file.endswith(".cpp"):
                continue
            full_path = os.path.join(problem_path, file)
            with open(full_path, "r", errors="ignore") as f:
                code = f.read()
            tree    = parser.parse(bytes(code, "utf8"))
            counter = Counter()
            hash_subtree(tree.root_node, counter)
            ast_data[(problem, file)] = counter

    with open(cache_path, "wb") as f:
        pickle.dump(ast_data, f)
    print(f"[INFO] AST features saved: {len(ast_data)} files → {cache_path}")

# =====================================================
# PHASE 2 — Load cache and compute pairwise similarity
# =====================================================
def compute_pairwise_similarity(cache_path, output_csv):
    # if os.path.exists(output_csv):
    #     print(f"[INFO] CSV exists, skipping.")
    #     return

    with open(cache_path, "rb") as f:
        ast_data = pickle.load(f)

    keys = list(ast_data.keys())
    N    = len(keys)
    print(f"[INFO] Files: {N} | Pairs: {N*(N-1)//2}")

    def dice(c1, c2):
        if not c1 and not c2: return 0.0
        inter = sum((c1 & c2).values())
        total = sum(c1.values()) + sum(c2.values())
        return 2 * inter / total if total > 0 else 0.0

    BATCH_WRITE = 10000
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem_1","file_1","problem_2","file_2","S_AST"])
        buffer = []
        for i in range(N):
            p1, f1 = keys[i]
            for j in range(i+1, N):
                p2, f2 = keys[j]
                sim = dice(ast_data[(p1,f1)], ast_data[(p2,f2)])
                buffer.append([p1, f1, p2, f2, round(sim, 4)])
                if len(buffer) >= BATCH_WRITE:
                    writer.writerows(buffer)
                    buffer = []
        if buffer:
            writer.writerows(buffer)

    print(f"[INFO] AST similarity saved to: {output_csv}")

# =====================================================
# RUN
# =====================================================
compute_and_save_ast_features(DATASET_ROOT, CACHE_PATH)
compute_pairwise_similarity(CACHE_PATH, OUTPUT_CSV)