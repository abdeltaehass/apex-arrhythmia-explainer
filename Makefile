.PHONY: help setup data data-meta data-sample data-100 eda manifests train experiments compare grounding gen-data gen-train gen-train-smoke reliability wandb-init api ui test lint

help:
	@echo "APEX targets:"
	@echo "  setup       create venv + install requirements"
	@echo "  data-meta   download only PTB-XL metadata CSVs (fast)"
	@echo "  data-sample download a small curated set of waveforms (Phase 2 notebook)"
	@echo "  data-100    download only the 100 Hz waveforms (~1.7 GB, needed to train)"
	@echo "  data        download full PTB-XL dataset (both rates, ~20 GB)"
	@echo "  manifests   build patient-level train/val/test manifests"
	@echo "  eda         run EDA -> docs/eda/"
	@echo "  train       train the baseline detector (20 epochs) -> docs/baseline/"
	@echo "  experiments run the Phase 4 model sweep (cnn/transformer x bce/focal)"
	@echo "  compare     build docs/model_comparison/comparison.md from runs.jsonl"
	@echo "  grounding   Phase 5 saliency sanity sweep (AFIB + STTC) -> docs/grounding/"
	@echo "  gen-data    build the Phase 6 report dataset -> data/processed/generation/"
	@echo "  gen-train   LoRA fine-tune (default Mistral-7B-Instruct; needs a GPU)"
	@echo "  gen-train-smoke  tiny end-to-end LoRA smoke test, runs on CPU/MPS"
	@echo "  reliability Phase 7 consistency/grounding/confidence/mutex report -> docs/reliability/"
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

data-100:
	python scripts/download_ptbxl.py --records 100

data:
	python scripts/download_ptbxl.py

manifests:
	python -m src.data.manifests

eda:
	python scripts/run_eda.py

train:
	python -m src.detection.train

experiments:
	bash scripts/run_experiments.sh

compare:
	python scripts/build_comparison.py

grounding:
	python scripts/run_grounding.py --scan AFIB --n 60
	python scripts/run_grounding.py --scan STTC --n 60

gen-data:
	python scripts/build_gen_dataset.py

gen-train:
	python -m src.generation.train_lora --load-in-4bit --bf16

gen-train-smoke:
	python -m src.generation.train_lora --smoke

reliability:
	python scripts/run_reliability_report.py

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
