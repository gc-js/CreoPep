#!/bin/bash

BASE_DIR="foldx"
PYTHON_SCRIPT="Interaction_Energy.py"
TASK_NAME="a9a10"


CASES=($(ls -d $BASE_DIR/*/))

for case_path in "${CASES[@]}"; do
    case_name=$(basename "$case_path")
    echo "Processing case: $case_name" 
    /home/ubuntu/anaconda3/envs/gecheng/bin/python "$PYTHON_SCRIPT" \
        --path "$case_path" \
        --task "$TASK_NAME" \
        --num "$case_name"
done
