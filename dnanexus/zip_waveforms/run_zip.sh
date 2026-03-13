#!/bin/bash

WAVEFORM_PATH="/waveforms"
OUT_PATH="/zipped_waveforms"

dx mkdir -p $OUT_PATH

dx find data --path "${WAVEFORM_PATH}/" --name "*.tar.gz" --brief > waveform_paths.txt
split -d -l 2188 waveform_paths.txt

for batch in {0..50}; do
    echo "Create input json ${batch}"
    python generate_input_json.py $batch

    echo "Run workflow"
    dx run app-swiss-army-knife \
        --name "tar waveforms" \
        --instance-type mem1_hdd1_v2_x2 \
        --priority high \
        -f zip_ecgs.json \
        --destination "${OUT_PATH}/" \
        -y --brief
done