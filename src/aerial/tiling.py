"""Slicing: train and predict on overlapping tiles instead of a downscaled whole image.

    aerial-slice --data data/visdrone --tile 640 --overlap 0.2     # train/val tiles + tiles640.yaml

A 2000 x 1500 drone photo shrunk to 640 px loses two thirds of every pedestrian's pixels. Cutting
it into 640 px tiles keeps full resolution (Akyon et al., "Slicing Aided Hyper Inference", ICIP
2022). Training tiles keep the boxes that are mostly inside; at inference, the detections of
all tiles (and optionally of the whole image, for large objects) are shifted back and merged with
class-wise NMS, because an object on a tile border is found twice.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms

from aerial.visdrone import write_yaml

log = logging.getLogger("aerial.tiling")


def starts(length: int, tile: int, max_step: int) -> list[int]:
    """Tile origins along one axis: the fewest tiles whose spacing is at most ``max_step``, spread
    evenly from border to border (no near-duplicate last tile)."""
    if length <= tile:
        return [0]
    n = math.ceil((length - tile) / max_step) + 1
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def tile_grid(
    width: int, height: int, tile: int = 640, overlap: float = 0.2
) -> list[tuple[int, int, int, int]]:
    step = max(1, round(tile * (1 - overlap)))
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height, tile, step)
        for x in starts(width, tile, step)
    ]


def crop_boxes(boxes: np.ndarray, window: tuple[int, int, int, int], min_visible: float = 0.5) -> np.ndarray:
    """(N, 5) class + xyxy in pixels -> boxes inside ``window``, in window coordinates; boxes
    with less than ``min_visible`` of their area inside are dropped."""
    if len(boxes) == 0:
        return boxes.reshape(0, 5)
    x0, y0, x1, y1 = window
    clipped = boxes.copy()
    clipped[:, [1, 3]] = boxes[:, [1, 3]].clip(x0, x1)
    clipped[:, [2, 4]] = boxes[:, [2, 4]].clip(y0, y1)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2])
    inside = (clipped[:, 3] - clipped[:, 1]) * (clipped[:, 4] - clipped[:, 2])
    keep = inside >= min_visible * np.maximum(area, 1e-9)
    clipped = clipped[keep]
    clipped[:, [1, 3]] -= x0
    clipped[:, [2, 4]] -= y0
    return clipped


def read_yolo(path: Path, width: int, height: int) -> np.ndarray:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    if not rows:
        return np.zeros((0, 5))
    a = np.array(rows, dtype=float)
    cx, cy, w, h = a[:, 1] * width, a[:, 2] * height, a[:, 3] * width, a[:, 4] * height
    return np.stack([a[:, 0], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def slice_split(data: Path, split: str, tile: int, overlap: float, min_visible: float = 0.5) -> int:
    """images/<split> -> images/<split>_tiles<tile> + labels; returns the number of tiles."""
    name = f"{split}_tiles{tile}"
    (data / "images" / name).mkdir(parents=True, exist_ok=True)
    (data / "labels" / name).mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted((data / "images" / split).glob("*.jpg")):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes = read_yolo(data / "labels" / split / f"{image_path.stem}.txt", width, height)
        for x0, y0, x1, y1 in tile_grid(width, height, tile, overlap):
            stem = f"{image_path.stem}_{x0}_{y0}"
            image.crop((x0, y0, x1, y1)).save(data / "images" / name / f"{stem}.jpg", quality=95)
            crop = crop_boxes(boxes, (x0, y0, x1, y1), min_visible)
            w, h = x1 - x0, y1 - y0
            lines = [
                f"{int(c)} {(bx0 + bx1) / 2 / w:.6f} {(by0 + by1) / 2 / h:.6f} "
                f"{(bx1 - bx0) / w:.6f} {(by1 - by0) / h:.6f}\n"
                for c, bx0, by0, bx1, by1 in crop
            ]
            (data / "labels" / name / f"{stem}.txt").write_text("".join(lines))
            count += 1
    return count


def predict_tiled(
    model,
    image: np.ndarray,
    imgsz: int = 640,
    tile: int = 640,
    overlap: float = 0.2,
    full_image: bool = False,
    conf: float = 0.001,
    iou: float = 0.5,
    max_det: int = 500,
    batch: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """xyxy boxes, scores and classes for a BGR image from overlapping tiles (+ the whole image)."""
    height, width = image.shape[:2]
    windows = tile_grid(width, height, tile, overlap)
    crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in windows]
    offsets = [(x0, y0) for x0, y0, _, _ in windows]
    if full_image and len(windows) > 1:
        crops.append(image)
        offsets.append((0, 0))
    boxes, scores, classes = [], [], []
    for i in range(0, len(crops), batch):
        results = model.predict(crops[i : i + batch], imgsz=imgsz, conf=conf, max_det=300, verbose=False)
        for result, (dx, dy) in zip(results, offsets[i : i + batch], strict=True):
            b = result.boxes
            boxes.append(b.xyxy.cpu() + torch.tensor([dx, dy, dx, dy], dtype=torch.float32))
            scores.append(b.conf.cpu())
            classes.append(b.cls.cpu())
    boxes, scores, classes = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
    keep = batched_nms(boxes, scores, classes.long(), iou)[:max_det]
    return boxes[keep].numpy(), scores[keep].numpy(), classes[keep].numpy().astype(int)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, required=True, help="folder written by aerial-data")
    parser.add_argument("--tile", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for split in ("train", "val"):  # val tiles pick best.pt the way the model will be used
        log.info("%s: %d tiles", split, slice_split(args.data, split, args.tile, args.overlap))
    path = write_yaml(args.data, train=f"images/train_tiles{args.tile}", val=f"images/val_tiles{args.tile}")
    path.rename(args.data / f"tiles{args.tile}.yaml")
    write_yaml(args.data)  # keep the whole-image yaml as it was


if __name__ == "__main__":
    main()
