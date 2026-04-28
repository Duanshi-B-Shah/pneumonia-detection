#!/usr/bin/env bash
# Download the Chest X-Ray Pneumonia dataset from Kaggle.
# Requires: pip install kaggle && kaggle API token at ~/.kaggle/kaggle.json

set -euo pipefail

DATASET="paultimothymooney/chest-xray-pneumonia"
OUTPUT_DIR="data/raw"

echo "═══ Downloading Chest X-Ray dataset ═══"
echo "Dataset: ${DATASET}"
echo "Target:  ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

# Download and unzip
kaggle datasets download -d "${DATASET}" -p "${OUTPUT_DIR}" --unzip

echo ""
echo "✓ Dataset downloaded to ${OUTPUT_DIR}/"
echo ""

# Verify structure
if [ -d "${OUTPUT_DIR}/chest_xray" ]; then
    echo "Verified structure:"
    echo "  ${OUTPUT_DIR}/chest_xray/train/NORMAL/     $(ls ${OUTPUT_DIR}/chest_xray/train/NORMAL/ | wc -l) images"
    echo "  ${OUTPUT_DIR}/chest_xray/train/PNEUMONIA/  $(ls ${OUTPUT_DIR}/chest_xray/train/PNEUMONIA/ | wc -l) images"
    echo "  ${OUTPUT_DIR}/chest_xray/val/NORMAL/       $(ls ${OUTPUT_DIR}/chest_xray/val/NORMAL/ | wc -l) images"
    echo "  ${OUTPUT_DIR}/chest_xray/val/PNEUMONIA/    $(ls ${OUTPUT_DIR}/chest_xray/val/PNEUMONIA/ | wc -l) images"
    echo "  ${OUTPUT_DIR}/chest_xray/test/NORMAL/      $(ls ${OUTPUT_DIR}/chest_xray/test/NORMAL/ | wc -l) images"
    echo "  ${OUTPUT_DIR}/chest_xray/test/PNEUMONIA/   $(ls ${OUTPUT_DIR}/chest_xray/test/PNEUMONIA/ | wc -l) images"
else
    echo "⚠ Expected directory ${OUTPUT_DIR}/chest_xray not found."
    echo "  Check Kaggle API setup: https://github.com/Kaggle/kaggle-api#api-credentials"
fi
