import os
import re
import csv
import sys
import math
from collections import Counter
import numpy as np

# ======================================================
# PATH CONFIG
# ======================================================

# ROOT = r"E:\Mtech AI\sem 3\Dataset\My_Dataset"
# OUTPUT_CSV = r"E:\Mtech AI\sem 3\Approachs\Structure_based\structure_similarity.csv"

# ROOT = "/home/yash/My_Dataset"
# OUTPUT_CSV = "/home/yash/Approachs/Structure_based/structure_similarity.csv"

ROOT = sys.argv[1]
PROJECT_DIR = sys.argv[2]
OUTPUT_CSV = os.path.join(PROJECT_DIR, "Structure_based", "structure_similarity.csv")

# ======================================================
# CLEAN CODE
# ======================================================

def clean_code(code):
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r'"[^"]*"', "", code)
    return code

# ======================================================
# STRUCTURAL TOKEN MAP
# ======================================================

STRUCT_MAP = [
    (r"\bfor\s*\(", "LOOP"),
    (r"\bwhile\s*\(", "LOOP"),
    (r"\bdo\s*\{?", "LOOP"),
    (r"\bif\s*\(", "BRANCH"),
    (r"\belse\b", "BRANCH"),
    (r"\bswitch\s*\(", "BRANCH"),
    (r"\bcase\b", "BRANCH"),
    (r"\breturn\b", "RETURN"),
    (r"\b(int|long|float|double|char|bool|string)\b", "DECL"),
    (r"[<>!=]=?", "COMP"),
    (r"[\+\-\*\/]", "ARITH"),
    (r"\=", "ASSIGN"),
    (r"&&|\|\|", "LOGIC"),
]

# ======================================================
# EXTRACT STRUCTURE SEQUENCE
# ======================================================

def extract_struct_sequence(code):

    tokens = []
    code = clean_code(code)

    for pattern, label in STRUCT_MAP:
        for m in re.finditer(pattern, code):
            tokens.append((m.start(), label))

    # detect function calls (exclude keywords)
    call_pattern = r"\b[a-zA-Z_]\w*\s*\("
    for m in re.finditer(call_pattern, code):
        word = m.group().strip()
        if not any(kw in word for kw in ["if(", "for(", "while(", "switch("]):
            tokens.append((m.start(), "CALL"))

    tokens.sort(key=lambda x: x[0])
    return [t[1] for t in tokens]

# ======================================================
# N-GRAM FEATURES
# ======================================================

def ngrams(seq, n=3):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

def extract_ngram_features(seq):
    if len(seq) < 3:
        return Counter()

    grams = ngrams(seq, 3)
    freq = Counter(grams)

    total = sum(freq.values())
    for k in freq:
        freq[k] /= total

    return freq

# ======================================================
# GLOBAL STRUCTURE FEATURES
# ======================================================

def extract_global_structure_features(code, seq):

    loop_count = seq.count("LOOP")
    branch_count = seq.count("BRANCH")
    return_count = seq.count("RETURN")
    call_count = seq.count("CALL")
    assign_count = seq.count("ASSIGN")
    logic_count = seq.count("LOGIC")

    # Improved cyclomatic complexity
    cyclomatic = branch_count + logic_count + 1

    # Nesting depth
    max_depth = 0
    depth = 0
    depth_profile = []

    for ch in code:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth -= 1
        depth_profile.append(depth)

    depth_hist = Counter(depth_profile)

    # Structural entropy
    token_freq = Counter(seq)
    total = sum(token_freq.values()) + 1e-6
    entropy = -sum((c/total) * math.log((c/total)+1e-6) for c in token_freq.values())

    global_vec = np.array([
        loop_count,
        branch_count,
        return_count,
        call_count,
        assign_count,
        logic_count,
        cyclomatic,
        max_depth,
        entropy
    ], dtype=float)

    return global_vec, depth_hist

# ======================================================
# LCS SIMILARITY
# ======================================================

def lcs_length(seq1, seq2):

    m, n = len(seq1), len(seq2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m):
        for j in range(n):
            if seq1[i] == seq2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])

    return dp[m][n]

def lcs_similarity(seq1, seq2):
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    lcs = lcs_length(seq1, seq2)
    return lcs / max(len(seq1), len(seq2))

# ======================================================
# COSINE SIMILARITY
# ======================================================

def cosine_sim_counter(v1: Counter, v2: Counter):
    all_keys = set(v1.keys()) | set(v2.keys())
    dot = sum(v1[k] * v2[k] for k in all_keys)
    mag1 = math.sqrt(sum(v1[k] ** 2 for k in all_keys))
    mag2 = math.sqrt(sum(v2[k] ** 2 for k in all_keys))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)

def l2_similarity(v1, v2):
    # normalize each vector first
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    v1 = v1 / (n1 + 1e-9)
    v2 = v2 / (n2 + 1e-9)
    dist = np.linalg.norm(v1 - v2)
    return 1 / (1 + dist)

# ======================================================
# LOAD ALL STRUCTURES
# ======================================================

def load_all_structures(root):

    data = {}

    for problem in sorted(os.listdir(root)):
        p_path = os.path.join(root, problem)
        if not os.path.isdir(p_path):
            continue

        for file in sorted(os.listdir(p_path)):
            if not file.endswith(".cpp"):
                continue

            full_path = os.path.join(p_path, file)
            code = open(full_path, "r", errors="ignore").read()
            code = clean_code(code)

            seq = extract_struct_sequence(code)
            ngram_feat = extract_ngram_features(seq)
            global_feat, depth_hist = extract_global_structure_features(code, seq)

            data[f"{problem}/{file}"] = (seq, ngram_feat, global_feat, depth_hist)

    return data

# ======================================================
# GENERATE CSV
# ======================================================

def generate_similarity_csv(data, out_csv):

    with open(out_csv, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["problem_1", "file_1", "problem_2", "file_2", "S_structure"])

        keys = list(data.keys())

        for i in range(len(keys)):
            for j in range(i+1, len(keys)):

                f1, f2 = keys[i], keys[j]
                seq1, ngram1, global1, depth1 = data[f1]
                seq2, ngram2, global2, depth2 = data[f2]

                sim_ngram = cosine_sim_counter(ngram1, ngram2)
                sim_global = l2_similarity(global1, global2)
                sim_lcs = lcs_similarity(seq1, seq2)

                # New weighted fusion
                sim_final = (
                    0.5 * sim_ngram +
                    0.3 * sim_global +
                    0.2 * sim_lcs
                )

                p1, s1 = f1.split("/")
                p2, s2 = f2.split("/")

                writer.writerow([p1, s1, p2, s2, round(sim_final, 4)])

    print("Improved structure CSV generated:", out_csv)

# ======================================================
# RUN
# ======================================================

data = load_all_structures(ROOT)
generate_similarity_csv(data, OUTPUT_CSV)