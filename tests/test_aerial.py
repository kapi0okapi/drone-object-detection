import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from aerial.evaluate import coco_scores
from aerial.tiling import crop_boxes, predict_tiled, tile_grid
from aerial.visdrone import CLASSES, coco_json, parse, yolo_lines


def test_tile_grid_covers_the_image_with_overlap():
    grid = tile_grid(2000, 1500, tile=640, overlap=0.2)
    covered = np.zeros((1500, 2000), bool)
    for x0, y0, x1, y1 in grid:
        assert x1 - x0 == 640 and y1 - y0 == 640
        covered[y0:y1, x0:x1] = True
    assert covered.all()
    assert grid[-1] == (1360, 860, 2000, 1500)  # last tile flush with the corner
    assert len(grid) == 4 * 3


def test_small_image_is_one_tile():
    assert tile_grid(500, 400) == [(0, 0, 500, 400)]


def test_crop_boxes_keeps_mostly_visible_boxes():
    boxes = np.array([[3, 10, 10, 30, 30], [1, 620, 100, 650, 120], [0, 600, 100, 700, 120]], float)
    out = crop_boxes(boxes, (0, 0, 640, 640), min_visible=0.5)
    assert out[:, 0].tolist() == [3, 1]  # the second is 67 % inside, the third only 40 %
    assert out[1].tolist() == [1, 620, 100, 640, 120]
    shifted = crop_boxes(boxes, (512, 0, 1152, 640))
    assert shifted[:, 0].tolist() == [1, 0] and shifted[1, 1] == 88


def test_parse_separates_objects_and_ignore_regions(tmp_path):
    ann = tmp_path / "a.txt"
    ann.write_text(
        "10,20,30,40,1,4,0,0\n5,5,50,50,0,0,0,0\n1,1,8,8,1,11,0,0\n7,7,3,3,0,1,0,0\n0,0,0,5,1,1,0,0\n"
    )
    objects, ignore = parse(ann)
    assert objects == [(3, 10, 20, 30, 40)]  # category 4 (car) -> class 3
    assert ignore == [(5, 5, 50, 50), (1, 1, 8, 8), (7, 7, 3, 3)]
    assert yolo_lines(objects, 100, 200) == ["3 0.250000 0.200000 0.300000 0.200000\n"]


def test_coco_json_turns_ignore_regions_into_crowds_of_every_class():
    gt = coco_json([("a.jpg", 100, 100, [(0, 10, 10, 5, 5)], [(50, 50, 20, 20)])])
    crowds = [a for a in gt["annotations"] if a["iscrowd"]]
    assert len(crowds) == len(CLASSES) and {a["category_id"] for a in crowds} == set(range(1, 11))


def test_coco_scores_ignore_detections_in_ignore_regions(tmp_path):
    gt = coco_json([("a.jpg", 200, 200, [(0, 10, 10, 20, 20), (3, 100, 20, 40, 40)], [(120, 120, 60, 60)])])
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt))
    perfect = [
        {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 4, "bbox": [100, 20, 40, 40], "score": 0.8},
    ]
    assert coco_scores(path, perfect)["ap"] == pytest.approx(1.0)
    in_ignore = {"image_id": 1, "category_id": 4, "bbox": [130, 130, 20, 20], "score": 0.95}
    assert coco_scores(path, [*perfect, in_ignore])["ap"] == pytest.approx(1.0)
    stray = {"image_id": 1, "category_id": 4, "bbox": [10, 150, 20, 20], "score": 0.95}
    assert coco_scores(path, [*perfect, stray])["ap"] < 1.0


class FakeModel:
    """Finds one fixed object (image coordinates) in every tile that fully contains it. Tiles
    arrive in the order of ``windows``, so the n-th crop has the n-th window's offset."""

    def __init__(self, box, windows):
        self.box = np.array(box, float)
        self.offsets = iter([(x0, y0) for x0, y0, _, _ in windows])

    def predict(self, crops, **_):
        results = []
        for crop in crops:
            ox, oy = next(self.offsets)
            x0, y0, x1, y1 = self.box - [ox, oy, ox, oy]
            h, w = crop.shape[:2]
            inside = x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h
            xyxy = torch.tensor([[x0, y0, x1, y1]] if inside else [], dtype=torch.float32).reshape(-1, 4)
            n = len(xyxy)
            boxes = SimpleNamespace(xyxy=xyxy, conf=torch.full((n,), 0.9), cls=torch.zeros(n))
            results.append(SimpleNamespace(boxes=boxes))
        return results


def test_predict_tiled_merges_duplicates_from_overlapping_tiles():
    image = np.zeros((640, 1200, 3), np.uint8)
    windows = tile_grid(1200, 640, 640, 0.2)
    assert [w[0] for w in windows] == [0, 280, 560]  # two tiles would overlap by only 80 px
    model = FakeModel([600, 300, 620, 320], windows)  # inside all three tiles
    boxes, scores, classes = predict_tiled(model, image)
    assert len(boxes) == 1  # found three times, merged into one
    np.testing.assert_allclose(boxes[0], [600, 300, 620, 320])
    assert classes.tolist() == [0]
