#!/bin/bash

NOTEBOOKS_PY=(notebooks/*.py)
OUTPUT_DIR="outputs"

# Create output directory if necessary.
mkdir -p "$OUTPUT_DIR"

for file in "${NOTEBOOKS_PY[@]}"; do
    filename=$(basename "$file")
    echo "Converting and executing $file."
    jupytext --to notebook -o "$OUTPUT_DIR/${filename%.*}.ipynb" --execute "$file"
done
