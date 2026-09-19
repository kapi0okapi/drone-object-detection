"""Train a YOLO26 detector with Ultralytics (resumes from last.pt after a time limit).

    aerial-train --data data/visdrone/visdrone.yaml --model yolo26s.pt --name visdrone_640 --imgsz 640

The weights start from COCO; the ten VisDrone classes get a new classification layer.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default=os.environ.get("RUNS", "runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.002)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    project = Path(os.path.expandvars(args.project)).resolve()  # relative paths would land under runs/
    last = project / args.name / "weights" / "last.pt"
    if last.exists():
        print(f"resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return
    YOLO(os.path.expandvars(args.model)).train(
        data=os.path.expandvars(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        optimizer="AdamW",
        cos_lr=True,
        workers=args.workers,
        device=args.device or None,
        project=str(project),
        name=args.name,
        exist_ok=True,
        patience=30,
        plots=True,
    )


if __name__ == "__main__":
    main()
