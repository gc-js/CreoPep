#!/bin/bash

while getopts "d:s:t:" opt; do
  case $opt in
    d) BASE_DIR="$OPTARG" ;;
    s) PYTHON_SCRIPT="$OPTARG" ;;
    t) TASK_NAME="$OPTARG" ;;
  esac
done

echo "BASE_DIR: $BASE_DIR"
echo "PYTHON_SCRIPT: $PYTHON_SCRIPT"
echo "TASK_NAME: $TASK_NAME"

CASES=($(ls -d "$BASE_DIR"/*/))
for case_path in "${CASES[@]}"; do
    case_name=$(basename "$case_path")
    echo "Processing case: $case_name" 
    /home/ubuntu/anaconda3/envs/gecheng/bin/python "$PYTHON_SCRIPT" \
        --path "$case_path" \
        --task "$TASK_NAME" \
        --num "$case_name"
done
