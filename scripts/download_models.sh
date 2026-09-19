#!/bin/bash
# Pre-fetch the COCO-pretrained YOLO26 weights on a login node (jobs run offline).
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
from ultralytics import YOLO

for name in ("yolo26n.pt", "yolo26s.pt"):
    YOLO(name)
    print("cached", name)
PY
