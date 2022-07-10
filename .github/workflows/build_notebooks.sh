#!/bin/bash

NOTEBOOKS_PY=(notebooks/*.py)

for file in "${NOTEBOOKS_PY[@]}"; do
    filename=$(basename "$file")
    echo "Converting and executing $file."
    jupytext --to notebook --execute "$file"
done
