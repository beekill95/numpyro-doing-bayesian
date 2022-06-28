#!/bin/bash

NOTEBOOKS_PY="notebooks/*.py"
OUTPUT_DIR="outputs"

# Create output directory if necessary.
mkdir -p "$OUTPUT_DIR"

for file in "$NOTEBOOKS_PY"; do
    filename=$(basename "$file")
    echo "Converting and Executing $file to notebook."
    jupytext --to notebook -o "$OUTPUT_DIR/${filename%.*}" --execute "$file"
done
