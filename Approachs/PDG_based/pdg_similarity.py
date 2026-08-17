import os, sys, re, csv, hashlib, pickle
import networkx as nx
import numpy as np
from collections import Counter

DATASET_DIR = sys.argv[1]
PROJECT_DIR  = sys.argv[2]

# FIX 2: IR files now live under IR/<problem>/<fname>.ll (per-problem subdir)
IR_DIR     = os.path.join(PROJECT_DIR, "PDG_based", "IR")
OUT_CSV    = os.path.join(PROJECT_DIR, "PDG_based", "pdg_similarity.csv")
CACHE_PATH = os.path.join(PROJECT_DIR, "PDG_based", "pdg_features.pkl")

HASH_DIM = 8192
WL_ITERS = 3

# FIX 1: similarity assigned when one or both IRs are stubs or missing
FALLBACK_SIM = 0.0

# Marker written by the shell script into stub .ll files
STUB_MARKER = "; STUB_IR_FAILED"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def collect_cpp(dataset_root):
    items = []
    for p in sorted(os.listdir(dataset_root)):
        p_path = os.path.join(dataset_root, p)
        if not os.path.isdir(p_path):
            continue
        for f in sorted(os.listdir(p_path)):
            if f.endswith(".cpp"):
                items.append((p, f))
    return items


def ir_path_for(p, fname_no_ext):
    """FIX 2: build IR path using per-problem subdirectory."""
    return os.path.join(IR_DIR, p, fname_no_ext + ".ll")


def is_stub_or_missing(path):
    """FIX 1 + FIX 3: True if the IR file is absent OR is a stub placeholder."""
    if not os.path.exists(path):
        return True
    with open(path, "r", errors="ignore") as fh:
        first_line = fh.readline()
    return STUB_MARKER in first_line


def parse_llvm_ir(path):
    G = nx.DiGraph()
    if not os.path.exists(path):
        return G
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    current_block = None
    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith(";"):
            continue
        if re.match(r'^[a-zA-Z0-9_.]+:\s*(;.*)?$', line_s):
            current_block = line_s.split(":")[0].strip()
            if not G.has_node(current_block):
                G.add_node(current_block, label="block", type="block")
            continue
        if current_block is None:
            continue
        if line_s.startswith("br ") or line_s.startswith("switch ") or line_s.startswith("indirectbr "):
            for t in re.findall(r'label %([a-zA-Z0-9_.]+|\d+)', line_s):
                if not G.has_node(t):
                    G.add_node(t, label="block", type="block")
                G.add_edge(current_block, t, type="control")
        assign_match = re.match(r'^(%[a-zA-Z0-9_.]+|%\d+)\s*=\s*(.+)', line_s)
        if assign_match:
            lhs, rhs = assign_match.group(1), assign_match.group(2)
            if "alloca"      in rhs: label = "alloca"
            elif "load"      in rhs: label = "load"
            elif "phi"       in rhs: label = "phi"
            elif "call"      in rhs: label = "call"
            elif "icmp" in rhs or "fcmp" in rhs: label = "cmp"
            elif "getelementptr" in rhs: label = "gep"
            elif any(c in rhs for c in ["sitofp","bitcast","zext","sext","trunc"]): label = "cast"
            elif any(op in rhs for op in ["add","sub","mul","sdiv","udiv","srem","urem",
                                          "shl","lshr","ashr","and","or","xor",
                                          "fadd","fsub","fmul","fdiv"]): label = "arith"
            else: label = "inst"
            if not G.has_node(lhs):
                G.add_node(lhs, label=label, type="data", block=current_block)
            else:
                G.nodes[lhs]["label"] = label
            for u in re.findall(r'%([a-zA-Z0-9_.]+|\d+)', rhs):
                u_node = "%" + u
                if not G.has_node(u_node):
                    G.add_node(u_node, label="inst", type="data")
                G.add_edge(u_node, lhs, type="data")
        elif line_s.startswith("store "):
            ops = re.findall(r'%([a-zA-Z0-9_.]+|\d+)', line_s)
            if len(ops) >= 2:
                src, dst = "%" + ops[0], "%" + ops[1]
                for n in [src, dst]:
                    if not G.has_node(n):
                        G.add_node(n, label="inst", type="data")
                G.add_edge(src, dst, type="data")
        elif " call " in line_s or line_s.startswith("call "):
            call_id = "call_" + hashlib.md5(line_s.encode()).hexdigest()[:8]
            G.add_node(call_id, label="call", type="call")
            G.add_edge(current_block, call_id, type="control")
            args_part = line_s.split("(")[-1] if "(" in line_s else ""
            for a in re.findall(r'%([a-zA-Z0-9_.]+|\d+)', args_part):
                a_node = "%" + a
                if not G.has_node(a_node):
                    G.add_node(a_node, label="inst", type="data")
                G.add_edge(a_node, call_id, type="data")
    return G


def wl_refinement(G):
    labels = {n: f"{G.nodes[n].get('label','inst')}_deg{G.degree(n)}" for n in G.nodes()}
    all_counters = []
    for _ in range(WL_ITERS):
        all_counters.append(Counter(labels.values()))
        new_labels = {}
        for node in G.nodes():
            nbr_labels = (
                ["out_" + G[node][nbr].get("type","data") + "_" + labels[nbr]
                 for nbr in G.successors(node)] +
                ["in_"  + G[nbr][node].get("type","data") + "_" + labels[nbr]
                 for nbr in G.predecessors(node)]
            )
            combined = labels[node] + "_" + "_".join(sorted(nbr_labels))
            new_labels[node] = hashlib.md5(combined.encode()).hexdigest()
        labels = new_labels
    final = Counter()
    for c in all_counters:
        final.update(c)
    return final


def hashed_wl_vector(counter):
    vec = np.zeros(HASH_DIM, dtype=float)
    for label, count in counter.items():
        vec[int(hashlib.md5(label.encode()).hexdigest(), 16) % HASH_DIM] += count
    return vec


def cosine_sim(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm > 0 else 0.0


# =====================================================
# PHASE 1 -- Build and cache WL vectors
# FIX 1: track which files have stub/missing IR -> failed_set
# FIX 2: look up IR using per-problem subdirectory
# =====================================================
def compute_and_save_pdg_features(dataset_root, cache_path):
    SRC = collect_cpp(dataset_root)
    print(f"[INFO] Building PDG features for {len(SRC)} files...")

    wl_vectors = {}
    failed_set  = set()   # (p, f) keys whose IR is stub or missing

    for p, f in SRC:
        base = os.path.splitext(f)[0]
        path = ir_path_for(p, base)           # FIX 2

        if is_stub_or_missing(path):          # FIX 1 + FIX 3
            print(f"[WARN] Stub/missing IR, fallback sim will apply: {path}")
            failed_set.add((p, f))
            wl_vectors[(p, f)] = np.zeros(HASH_DIM, dtype=float)
        else:
            G = parse_llvm_ir(path)
            wl_vectors[(p, f)] = hashed_wl_vector(wl_refinement(G))

    with open(cache_path, "wb") as fh:
        pickle.dump({"src": SRC, "wl_vectors": wl_vectors, "failed_set": failed_set}, fh)

    print(f"[INFO] PDG features saved to: {cache_path}")
    print(f"[INFO] Files with stub/missing IR: {len(failed_set)}")


# =====================================================
# PHASE 2 -- Pairwise similarity
# FIX 1: pairs involving a failed/stub IR get FALLBACK_SIM (0.15)
#         CSV row order is preserved (same iteration order as before)
# =====================================================
def compute_pairwise_similarity(cache_path, output_csv):
    with open(cache_path, "rb") as fh:
        data = pickle.load(fh)

    SRC        = data["src"]
    wl_vectors = data["wl_vectors"]
    failed_set = data.get("failed_set", set())   # backwards-compatible

    N = len(SRC)
    print(f"[INFO] Files: {N} | Pairs: {N*(N-1)//2}")
    if failed_set:
        print(f"[INFO] {len(failed_set)} file(s) will use fallback sim = {FALLBACK_SIM}")

    BATCH_WRITE = 10000
    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["problem_1", "file_1", "problem_2", "file_2", "S_PDG", "pdg_valid"])
        buffer = []
        for i in range(N):
            p1, f1 = SRC[i]
            for j in range(i + 1, N):
                p2, f2 = SRC[j]

                # FIX 1: fallback sim when either file has stub/missing IR
                if (p1, f1) in failed_set or (p2, f2) in failed_set:
                    sim = FALLBACK_SIM
                else:
                    sim = cosine_sim(wl_vectors[(p1, f1)], wl_vectors[(p2, f2)])

                buffer.append([p1, f1, p2, f2, round(sim, 4), "0" if (p1, f1) in failed_set or (p2, f2) in failed_set else "1"])
                if len(buffer) >= BATCH_WRITE:
                    writer.writerows(buffer)
                    buffer = []
        if buffer:
            writer.writerows(buffer)

    print(f"[INFO] PDG similarity saved to: {output_csv}")


# =====================================================
# RUN
# =====================================================
compute_and_save_pdg_features(DATASET_DIR, CACHE_PATH)
compute_pairwise_similarity(CACHE_PATH, OUT_CSV)
