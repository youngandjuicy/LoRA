#!/usr/bin/env bash

set -euo pipefail


export HF_HUB_OFFLINE=1


mkdir -p outputs/final_test
mkdir -p logs/final_test


echo "============================================================"
echo "S2 FINAL HELD-OUT EVALUATION"
echo "============================================================"


echo
echo "Running seed 42..."
echo

python run_occurrence_final_test.py \
    --adapter_path checkpoints/s2_occurrence_lora/epoch_3 \
    --output_path outputs/final_test/s2_seed42_test_predictions.jsonl \
    --seed 42 \
    2>&1 | tee logs/final_test/s2_seed42_test.log


echo
echo "Running seed 43..."
echo

python run_occurrence_final_test.py \
    --adapter_path checkpoints/s2_occurrence_lora_seed43/epoch_3 \
    --output_path outputs/final_test/s2_seed43_test_predictions.jsonl \
    --seed 43 \
    2>&1 | tee logs/final_test/s2_seed43_test.log


echo
echo "Running seed 44..."
echo

python run_occurrence_final_test.py \
    --adapter_path checkpoints/s2_occurrence_lora_seed44/epoch_3 \
    --output_path outputs/final_test/s2_seed44_test_predictions.jsonl \
    --seed 44 \
    2>&1 | tee logs/final_test/s2_seed44_test.log


echo
echo "============================================================"
echo "FINAL TEST FINISHED"
echo "============================================================"