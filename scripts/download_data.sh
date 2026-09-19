#!/bin/bash
# VisDrone2019-DET train / val / test-dev (1.8 GB, from the Ultralytics asset mirror) -> YOLO
# labels and COCO JSON with ignore regions. Run on a login node (needs internet).
#   bash scripts/download_data.sh            # into $DATA/visdrone (cluster) or ./data/visdrone
set -euo pipefail
cd "$(dirname "$0")/.."
python -m aerial.visdrone --out "${DATA:-data}/visdrone"
