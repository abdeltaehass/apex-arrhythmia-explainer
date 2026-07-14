.PHONY: help setup data data-meta data-sample eda manifests wandb-init api ui test lint

help:
	@echo "APEX targets:"
	@echo "  setup       create venv + install requirements"
	@echo "  data-meta   download only PTB-XL metadata CSVs (fast)"
	@echo "  data-sample download a small curated set of waveforms (Phase 2 notebook)"
	@echo "  data        download full PTB-XL dataset (several GB)"
	@echo "  manifests   build patient-level train/val/test manifests"
	@echo "  eda         run EDA -> docs/eda/"
	@echo "  wandb-init  initialize the W&B project"
	@echo "  api        run the FastAPI backend"
	@echo "  ui         run the Gradio frontend"
	@echo "  test       run pytest"
	@echo "  lint       run ruff"

setup:
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data-meta:
	python scripts/download_ptbxl.py --metadata-only

data-sample:
	python scripts/fetch_sample_records.py

data:
	python scripts/download_ptbxl.py

manifests:
	python -m src.data.manifests

eda:
	python scripts/run_eda.py

wandb-init:
	python scripts/init_wandb.py

api:
	uvicorn app.backend.main:app --reload

ui:
	python app/frontend/app.py

test:
	pytest -q

lint:
	ruff check .
