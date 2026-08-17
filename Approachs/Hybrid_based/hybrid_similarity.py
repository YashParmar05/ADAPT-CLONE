import pandas as pd
import sys
import os

PROJECT_DIR = sys.argv[2]
output_path = os.path.join(PROJECT_DIR, "Hybrid_based", "combined.csv")
is_training = sys.argv[3].lower() == "true"

semantic_list = ["unixcoder_cosine", "unixcoder_l2", "unixcoder_manhattan"]

paths = [
    (os.path.join(PROJECT_DIR, "Token_based", "token_similarity.csv"), ["S_token"]),
    (os.path.join(PROJECT_DIR, "AST_based", "AST_similarity.csv"), ["S_AST"]),
    (os.path.join(PROJECT_DIR, "PDG_based", "pdg_similarity.csv"), ["S_PDG", "pdg_valid"]),
    (os.path.join(PROJECT_DIR, "Structure_based", "structure_similarity.csv"), ["S_structure"]),
    (os.path.join(PROJECT_DIR, "Semantic_based", "semantic_unixcoder_results.csv"), semantic_list),
]

chunksize = 100000

keys = ["problem_1", "file_1", "problem_2", "file_2"]

readers = [pd.read_csv(path, chunksize=chunksize) for path, _ in paths]

with open(output_path, "w") as f:
    for chunk_group in zip(*readers):

        # Base chunk from first file
        base_chunk = chunk_group[0][keys].copy()

        # Add each file's score columns
        for i, (_, col_names) in enumerate(paths):
            for col in col_names:
                base_chunk[col] = chunk_group[i][col].values

        # Ground truth for training
        if is_training:
            base_chunk["GroundTruth"] = (
                base_chunk["problem_1"] == base_chunk["problem_2"]
            ).astype(int)
            cols = base_chunk.columns.tolist()
            cols.insert(4, cols.pop(cols.index("GroundTruth")))
            base_chunk = base_chunk[cols]

        base_chunk.to_csv(f, header=f.tell() == 0, index=False)

print(f"✅ Hybrid similarity results saved to {output_path}")