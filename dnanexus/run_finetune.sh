#!/bin/bash

WAVEFORM_PATH="/ecgfounder_waveforms"
OUT_PATH="/ecgfounder_weights"
LABELS_FILE="cm_var_labels_ecgfounder.tsv"

dx mkdir -p $OUT_PATH

echo "Find waveforms"
dx find data --path "${WAVEFORM_PATH}/" --name "*.npy" --brief > waveform_inputs.txt
dx find data --path "/" --name "${LABELS_FILE}" --brief >> waveform_inputs.txt

echo "Create input json"
python generate_input_json.py waveform_inputs.txt

echo "Run workflow"
dx run app-swiss-army-knife \
    --name "Finetune ECGFounder" \
    --instance-type mem2_ssd2_gpu1_v2_x8 \
    --priority high \
    -f ecgfounder_finetune.json \
    --destination "${OUT_PATH}/" \
    -y --brief