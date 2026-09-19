"""VisDrone2019-DET -> YOLO labels for training and COCO JSON with ignore regions for scoring.

    aerial-data --out data/visdrone            # downloads train / val / test-dev (1.8 GB)

VisDrone annotations are ``x, y, w, h, score, category, truncation, occlusion`` per line. In the
ground truth, ``score`` 0 marks a box the official evaluation ignores; category 0 is an "ignored
region" (crowds, far-away clutter) and category 11 "others". Ultralytics' converter drops all of
them, which is fine for training but not for scoring: a detection inside an ignored region would
count as a false positive. Here they become ``iscrowd`` boxes in the COCO file, once per class,
so detections of any class inside them are neither rewarded nor punished, as in the official
toolkit. Scoring keeps the toolkit's limit of 500 detections per image.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import zipfile
from pathlib import Path

import requests
from PIL import Image

log = logging.getLogger("aerial.visdrone")
CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
SPLITS = {"train": "train", "val": "val", "test": "test-dev"}


def parse(
    path: Path,
) -> tuple[list[tuple[int, float, float, float, float]], list[tuple[float, float, float, float]]]:
    """Objects as (class 0-9, x, y, w, h) and ignore boxes as (x, y, w, h), in pixels."""
    objects, ignore = [], []
    for line in path.read_text().strip().splitlines():
        row = [v for v in line.strip().split(",") if v != ""]
        if len(row) < 6:
            continue
        x, y, w, h = map(float, row[:4])
        score, category = int(row[4]), int(row[5])
        if w <= 0 or h <= 0:
            continue
        if score == 0 or category in (0, 11):
            ignore.append((x, y, w, h))
        else:
            objects.append((category - 1, x, y, w, h))
    return objects, ignore


def yolo_lines(objects, width: int, height: int) -> list[str]:
    return [
        f"{c} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}\n"
        for c, x, y, w, h in objects
    ]


def coco_json(items: list[tuple[str, int, int, list, list]]) -> dict:
    """COCO ground truth; every ignore box is a crowd box for every class."""
    images, annotations = [], []
    for image_id, (name, width, height, objects, ignore) in enumerate(items, start=1):
        images.append({"id": image_id, "file_name": name, "width": width, "height": height})
        boxes = [(c, b, 0) for c, *b in objects] + [(c, b, 1) for b in ignore for c in range(len(CLASSES))]
        for c, (x, y, w, h), crowd in boxes:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": c + 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": crowd,
                }
            )
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(CLASSES)]
    return {"images": images, "annotations": annotations, "categories": categories}


def convert(source: Path, out: Path, split: str) -> int:
    """One extracted VisDrone split -> out/images/<split>, out/labels/<split>, out/<split>.json."""
    images_out, labels_out = out / "images" / split, out / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    items = []
    for ann in sorted((source / "annotations").glob("*.txt")):
        image = source / "images" / f"{ann.stem}.jpg"
        width, height = Image.open(image).size
        objects, ignore = parse(ann)
        shutil.move(image, images_out / image.name)
        (labels_out / ann.name).write_text("".join(yolo_lines(objects, width, height)))
        items.append((image.name, width, height, objects, ignore))
    (out / f"{split}.json").write_text(json.dumps(coco_json(items)))
    return len(items)


def write_yaml(out: Path, train: str = "images/train", val: str = "images/val") -> Path:
    lines = [f"path: {out.resolve()}", f"train: {train}", f"val: {val}", "test: images/test", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASSES)]
    path = out / "visdrone.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def download(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    for split, name in SPLITS.items():
        if (out / f"{split}.json").exists():
            continue
        archive = out / f"VisDrone2019-DET-{name}.zip"
        if not archive.exists():
            log.info("downloading %s", archive.name)
            with requests.get(URL.format(name), stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(archive.with_suffix(".part"), "wb") as f:
                    shutil.copyfileobj(r.raw, f, length=1 << 20)
            archive.with_suffix(".part").rename(archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
        source = next(
            p for p in out.iterdir() if p.is_dir() and p.name.startswith(f"VisDrone2019-DET-{name}")
        )
        log.info("%s: %d images", split, convert(source, out, split))
        shutil.rmtree(source)
        archive.unlink()
    return write_yaml(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(download(args.out))


if __name__ == "__main__":
    main()
