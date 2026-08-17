#!/bin/bash

DATASET_DIR="$1"
PROJECT_DIR="$2"

OUTPUT_DIR="$PROJECT_DIR/PDG_based"
IR_DIR="$OUTPUT_DIR/IR"

# Delete old IR directory and recreate fresh
if [ -d "$IR_DIR" ]; then
    rm -rf "$IR_DIR"
fi

mkdir -p "$IR_DIR"

# echo "========================================"
# echo "[INFO] Starting PDG Pipeline"
# echo "[INFO] Dataset Dir : $DATASET_DIR"
# echo "[INFO] Project Dir : $PROJECT_DIR"
# echo "========================================"

IR_FAIL_COUNT=0

# ------------------------------------------------------
# STEP 1: Generate LLVM IR files
# Using new 6-step approach:
# 1-3: Try source AS-IS (C++17, C++11, C)
# 4-6: Header-prepend + sed (C++17, C++98, C)
# ------------------------------------------------------

echo "[STEP 1] Generating LLVM IR..."

while read FILE; do

    base=$(basename "$FILE")                      # e.g. s1.cpp
    fname="${base%.*}"                             # e.g. s1
    problem=$(basename "$(dirname "$FILE")")       # e.g. p1

    mkdir -p "$IR_DIR/$problem"
    OUT="$IR_DIR/$problem/$fname.ll"

    # ---------------------------------------------
    # 1) AS-IS: C++17
    # ---------------------------------------------
    clang++ -S -emit-llvm \
        -std=gnu++17 \
        -O2 \
        -w \
        "$FILE" -o "$OUT" 2>/dev/null || rm -f "$OUT"

    # ---------------------------------------------
    # 2) AS-IS: C++11
    # ---------------------------------------------
    if [ ! -f "$OUT" ]; then
        clang++ -S -emit-llvm \
            -std=gnu++11 \
            -O2 \
            -w \
            "$FILE" -o "$OUT" 2>/dev/null || rm -f "$OUT"
    fi

    # ---------------------------------------------
    # 3) AS-IS: C11
    # ---------------------------------------------
    if [ ! -f "$OUT" ]; then
        clang -x c -S -emit-llvm \
            -std=gnu11 \
            "$FILE" -o "$OUT" 2>/dev/null || rm -f "$OUT"
    fi

    # ---------------------------------------------
    # 4-6) Header prepend + sed (if AS-IS failed)
    # ---------------------------------------------
    if [ ! -f "$OUT" ]; then

        temp_file=$(mktemp --suffix=.cpp)

        {
            echo "#include <bits/stdc++.h>"
            echo "#include <stdio.h>"
            echo "using namespace std;"
            echo "static volatile int __anon_var__ = 0;"
            echo "#define __anon_call__(x) ((void)(x))"
            echo ""
            sed \
            -e 's/^int ()$/int main()/' \
            -e 's/^int (int argc, char \*argv\[\])$/int main(int argc, char *argv[])/' \
            -e 's/^[ \t]*= \(.*\);/__anon_var__ = \1;/' \
            -e 's/^[ \t]*(\([^)]*\));/__anon_call__(\1);/' \
            "$FILE"
        } > "$temp_file"

        # 4) C++17 with header
        clang++ -S -emit-llvm \
            -std=gnu++17 \
            -O2 \
            -w \
            "$temp_file" -o "$OUT" 2>/dev/null || rm -f "$OUT"

        # 5) C++98 fallback
        if [ ! -f "$OUT" ]; then
            clang++ -S -emit-llvm \
                -std=gnu++98 \
                -O2 \
                -w \
                "$temp_file" -o "$OUT" 2>/dev/null || rm -f "$OUT"
        fi

        # 6) C fallback
        if [ ! -f "$OUT" ]; then
            clang -x c -S -emit-llvm \
                -std=gnu11 \
                "$temp_file" -o "$OUT" 2>/dev/null || rm -f "$OUT"
        fi

        rm -f "$temp_file"
    fi

    # --------------------------------------------------
    # FIX 3: all attempts failed -> write stub IR
    # --------------------------------------------------
    if [ ! -f "$OUT" ]; then
        cat > "$OUT" << 'STUB'
; STUB_IR_FAILED -- placeholder, all compilation attempts failed
; pdg_similarity.py detects this marker and assigns FALLBACK_SIM
; for every pair involving this file. CSV row order is preserved.

define void @stub_failed() {
entry:
  ret void
}
STUB
        echo "[IR STUB] $FILE  -->  $problem/$fname.ll"
        IR_FAIL_COUNT=$((IR_FAIL_COUNT + 1))
    else
        echo "[IR OK]   $FILE  -->  $problem/$fname.ll"
    fi

done < <(find "$DATASET_DIR" -type f \( -name "*.cpp" -o -name "*.c++" \))

echo "========================================"
echo "[INFO] Total IR Failures (stub written): $IR_FAIL_COUNT"
echo "========================================"