.PHONY: install dev lint format test data models jobs

install:
	pip install -e ".[train]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

data:     ## VisDrone2019-DET + COCO JSON with ignore regions
	bash scripts/download_data.sh

models:   ## COCO-pretrained YOLO26 weights
	bash scripts/download_models.sh

jobs:
	python3 slurm/generate.py --list
