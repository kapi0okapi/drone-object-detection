"""COCO-style scores on VisDrone, with the toolkit's ignore regions and 500 detections per image.

    aerial-eval --model runs/visdrone_640/weights/best.pt --data data/visdrone --split test --imgsz 640
    aerial-eval --model runs/visdrone_tiles640/weights/best.pt --data data/visdrone --split test --tiled

Reports AP (0.50:0.95), AP50, AP75 and AP by object size (small < 32 x 32 px, medium < 96 x 96 px,
large above), measured in the original image, plus GPU time per image. ``--tiled`` predicts on
640 px tiles with 20 % overlap; ``--full-image`` adds a pass over the whole image to the tiles.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from aerial.tiling import predict_tiled

log = logging.getLogger("aerial.evaluate")
STATS = ("ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large")


def coco_scores(gt_path: Path, detections: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(gt_path))
        dt = gt.loadRes(detections) if detections else COCO()
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 100, 500]  # summarize() reports AP at the last value
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    per_class = {}
    precision = ev.eval["precision"][:, :, :, 0, -1]  # iou, recall, class, all areas, 500 dets
    for k, cat in enumerate(ev.params.catIds):
        p = precision[:, :, k]
        per_class[gt.cats[cat]["name"]] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return {**dict(zip(STATS, map(float, ev.stats[:6]), strict=True)), "per_class": per_class}


def predict_split(model, gt: dict, images: Path, imgsz: int, tiled: bool, full_image: bool, tile: int):
    import torch

    detections, seconds = [], []
    for info in gt["images"]:
        bgr = cv2.imread(str(images / info["file_name"]))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        if tiled:
            boxes, scores, classes = predict_tiled(model, bgr, imgsz, tile, full_image=full_image)
        else:
            b = model.predict(bgr, imgsz=imgsz, conf=0.001, max_det=500, verbose=False)[0].boxes
            boxes, scores, classes = (
                b.xyxy.cpu().numpy(),
                b.conf.cpu().numpy(),
                b.cls.cpu().numpy().astype(int),
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds.append(time.perf_counter() - started)
        for (x0, y0, x1, y1), s, c in zip(boxes, scores, classes, strict=True):
            detections.append(
                {
                    "image_id": info["id"],
                    "category_id": int(c) + 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "score": float(s),
                }
            )
    return detections, float(np.median(seconds[1:] if len(seconds) > 1 else seconds))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True, help="folder written by aerial-data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--full-image", action="store_true", help="with --tiled: add a whole-image pass")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--name", help="label for the result file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from ultralytics import YOLO

    gt_path = args.data / f"{args.split}.json"
    gt = json.loads(gt_path.read_text())
    detections, seconds = predict_split(
        YOLO(args.model),
        gt,
        args.data / "images" / args.split,
        args.imgsz,
        args.tiled,
        args.full_image,
        args.tile,
    )
    mode = f"tiled{args.tile}" + ("+full" if args.full_image else "") if args.tiled else f"full{args.imgsz}"
    report = {
        "model": args.model,
        "split": args.split,
        "mode": mode,
        "images": len(gt["images"]),
        **coco_scores(gt_path, detections),
        "ms_per_image": 1000 * seconds,
    }
    out = Path(args.model).resolve().parents[1] / f"eval_{args.split}_{args.name or mode}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "per_class"}, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
