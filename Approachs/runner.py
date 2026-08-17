import os
import sys
import subprocess

# ===============================
# GLOBAL PATHS (EDIT HERE ONLY)
# ===============================

# DATASET_PATH = "E:/Mtech AI/sem 3/My_Dataset"
# PROJECT_PATH = "E:/Mtech AI/sem 3/Approachs"

# DATASET_PATH = "/home/yash/My_Dataset"
DATASET_PATH = "/home/yash/TEST_SET_6/"
PROJECT_PATH = "/home/yash/SCPD_TEST_Final/Approachs"
is_training = False   # Set to False for testing mode, True for training mode
# ===============================

def run_python_script(sub_folder,script_name, is_training=False):

    if(is_training):
        script_path = os.path.join(PROJECT_PATH, sub_folder, script_name)
        print(f"\n🚀 Running {script_name} in TRAINING mode")
        subprocess.run(
            [sys.executable, script_path, DATASET_PATH, PROJECT_PATH, "True"],
            check=True
        )
    else:
        script_path = os.path.join(PROJECT_PATH, sub_folder, script_name)
        print(f"\n🚀 Running {script_name} in TESTING mode")
        subprocess.run(
            [sys.executable, script_path, DATASET_PATH, PROJECT_PATH, "False"],
            check=True
        )

def run_bash_script(sub_folder,script_name):
    script_path = os.path.join(PROJECT_PATH, sub_folder, script_name)

    print(f"\n🚀 Running {script_name}")
    subprocess.run(
        ["bash", script_path, DATASET_PATH, PROJECT_PATH],
        check=True
    )


def main():
    print("===================================")
    print("  SOURCE CODE PLAG DETECTION")
    print("===================================")
    print("Dataset Path :", DATASET_PATH)
    print("Project Path :", PROJECT_PATH)

    run_python_script("Token_based", "token_similarity.py", is_training)
    run_python_script("AST_based", "Ast_similarity.py", is_training)
    run_bash_script("PDG_based", "run_pdg_pipeline.sh")
    run_python_script("PDG_based", "pdg_similarity.py", is_training)   
    run_python_script("Structure_based", "structure_similarity.py", is_training)
    run_python_script("Semantic_based", "semantic_similarity.py", is_training)  
    run_python_script("Hybrid_based", "hybrid_similarity.py", is_training)
    run_python_script("Hybrid_based", "Check_Results.py")  

    print("\n✅ All modules executed successfully!")

if __name__ == "__main__":
    main()
