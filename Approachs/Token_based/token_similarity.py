import os
import sys
import re
import csv
import hashlib
from collections import Counter

CPP_KEYWORDS = {
    "int","long","float","double","char","bool","void",
    "if","else","for","while","return","switch","case",
    "break","continue","class","struct","public","private",
    "protected","include","using","namespace","new","delete",
    "try","catch","throw","static","const","sizeof","true","false",
    "std","cout","cin","main"
}

def normalize_cpp(code):
    code = re.sub(r"//.*?$|/\*.*?\*/", "", code, flags=re.S | re.MULTILINE)
    code = re.sub(r"^#.*$", "", code, flags=re.MULTILINE)
    code = re.sub(r'"[^"]*"', "STR", code)
    code = re.sub(r"'[^']*'", "STR", code)
    code = re.sub(r"\b\d+\b", "NUM", code)
    code = re.sub(r"\s+", " ", code)

    tokens = re.findall(r"[A-Za-z_]\w*|[+\-*/%=!&|(){}\[\];,.<>?:]", code)

    out = []
    for tok in tokens:
        if tok in CPP_KEYWORDS:
            out.append(tok)
        elif re.match(r"[A-Za-z_]\w*", tok):
            out.append("ID")
        elif tok in ["STR", "NUM"]:
            out.append(tok)
        else:
            out.append(tok)

    return out

# --------------------------------------------------------------
# Winnowing
# --------------------------------------------------------------

def winnow_hashes(tokens, k=4, window=3):

    if len(tokens) < k:
        return set()

    kgrams = [" ".join(tokens[i:i+k]) for i in range(len(tokens)-k+1)]
    hashes = [int(hashlib.md5(kg.encode()).hexdigest(), 16) & 0xffffffff for kg in kgrams]

    fingerprints = set()
    for i in range(len(hashes) - window + 1):
        window_slice = hashes[i:i+window]
        min_hash = min(window_slice)
        fingerprints.add(min_hash)

    return fingerprints

# --------------------------------------------------------------
# Dice Similarity (better for partial overlap)
# --------------------------------------------------------------

def dice_similarity(set1, set2):
    if not set1 and not set2:
        return 0.0
    inter = len(set1 & set2)
    return (2 * inter) / (len(set1) + len(set2)) if (len(set1)+len(set2)) > 0 else 0.0

# --------------------------------------------------------------
# Load fingerprints
# --------------------------------------------------------------

def load_all_fingerprints(root):
    fps = {}
    for problem in sorted(os.listdir(root)):
        p_path = os.path.join(root, problem)
        if not os.path.isdir(p_path):
            continue
        for file in sorted(os.listdir(p_path)):
            if not file.endswith(".cpp"):
                continue
            full_path = os.path.join(p_path, file)
            with open(full_path, "r", errors="ignore") as f:
                code = f.read()
            tokens = normalize_cpp(code)
            fps[f"{problem}/{file}"] = winnow_hashes(tokens)
    return fps

# --------------------------------------------------------------
# Generate CSV
# --------------------------------------------------------------

def generate_similarity_csv(fps, out_csv):
    with open(out_csv, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["problem_1", "file_1", "problem_2", "file_2", "S_token"])
        keys = list(fps.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                f1, f2 = keys[i], keys[j]
                sim = dice_similarity(fps[f1], fps[f2])
                p1, s1 = f1.split("/")
                p2, s2 = f2.split("/")
                writer.writerow([p1, s1, p2, s2, sim])
    print("Improved Token CSV GENERATED ->", out_csv)

# --------------------------------------------------------------
# RUN
# --------------------------------------------------------------

if __name__ == "__main__":

    # ROOT = r"E:\Mtech AI\sem 3\Dataset\My_Dataset"
    # OUTPUT_CSV = r"E:\Mtech AI\sem 3\Approachs\Token_based\token_similarity.csv"

    # ROOT = "/home/yash/My_Dataset"
    # OUTPUT_CSV = "/home/yash/Approachs/Token_based/token_similarity.csv"

    # use global path
    ROOT = sys.argv[1]
    PROJECT_DIR = sys.argv[2]
    OUTPUT_CSV = os.path.join(PROJECT_DIR, "Token_based", "token_similarity.csv")


    fps = load_all_fingerprints(ROOT)
    generate_similarity_csv(fps, OUTPUT_CSV)