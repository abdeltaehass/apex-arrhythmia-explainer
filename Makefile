.PHONY: help setup data data-meta data-sample data-100 eda manifests train experiments compare grounding gen-data gen-train gen-train-smoke reliability benchmark digitize-eval eval-baselines edge-cases demographics subgroups stream-demo stream-eval calibrate distill distill-smoke distill-report federated federated-smoke federated-report rag-index rag-eval rag-report wandb-init api ui test lint

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
	@echo "  benchmark   Phase 9 latency/throughput benchmark -> docs/serving/benchmark.md"
	@echo "  digitize-eval Phase 10 image-digitization fidelity -> docs/digitization/report.md"
	@echo "  eval-baselines Phase 12 test-split eval vs published PTB-XL + GPT-4o -> docs/model_comparison/"
	@echo "  edge-cases  Phase 13 adversarial/edge-case cohorts + failure taxonomy -> docs/edge_cases/"
	@echo "  demographics Phase 14 AUROC by age/sex w/ bootstrap CIs -> docs/model_card/"
	@echo "  subgroups   Phase 18 per-label AUROC by sex/age + FDR -> docs/model_card/"
	@echo "  stream-demo Phase 16 live monitor in the terminal (normal -> AF playlist)"
	@echo "  stream-eval Phase 16 streaming behaviour + persistence trade-off -> docs/streaming/"
	@echo "  calibrate   Phase 17 temperature/vector scaling + reliability diagrams -> docs/calibration/"
	@echo "  distill     Phase 19 KD sweep (3 student sizes x distilled/from-scratch) + report"
	@echo "  distill-smoke  tiny end-to-end distillation check (sample records)"
	@echo "  distill-report rebuild docs/distillation/report.md from existing checkpoints"
	@echo "  federated   Phase 20 FedAvg sweep over PTB-XL device shards + report"
	@echo "  federated-smoke  tiny end-to-end federated check"
	@echo "  federated-report rebuild docs/federated/report.md from existing runs"
	@echo "  rag-index   Phase 21 fetch clinical corpus + build vector index + score retrieval"
	@echo "  rag-eval    Phase 21 paired RAG on/off hallucination comparison -> docs/rag/"
	@echo "  rag-report  rebuild docs/rag/report.md from existing eval JSON"
	@echo "  wandb-init  initialize the W&B project"
	@echo "  api        run the FastAPI service (/analyze /validate /health /metrics)"
	@echo "  ui         run the Gradio clinical dashboard (Phase 11; see docs/frontend/deploy.md)"
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

benchmark:
	python scripts/benchmark_api.py --http

digitize-eval:
	python scripts/eval_digitization.py

eval-baselines:
	python scripts/eval_baselines.py
	python scripts/gpt4o_baseline.py

edge-cases:
	python scripts/edge_case_report.py

demographics:
	python scripts/demographic_breakdown.py

subgroups:
	python scripts/subgroup_analysis.py

stream-demo:
	python scripts/stream_demo.py --playlist 9,598 --speed 4 --duration 60

stream-eval:
	python scripts/stream_eval.py

calibrate:
	python scripts/calibrate.py

distill:
	bash scripts/run_distillation.sh

distill-smoke:
	python -m src.detection.distill --smoke

distill-report:
	python scripts/distill_report.py

federated:
	bash scripts/run_federated.sh

federated-smoke:
	python -m src.federated.train --smoke

federated-report:
	python scripts/federated_report.py

rag-index:
	python scripts/build_rag_index.py

rag-eval:
	python scripts/rag_eval.py --n 150
	python scripts/rag_report.py

rag-report:
	python scripts/rag_report.py

wandb-init:
	python scripts/init_wandb.py

api:
	uvicorn app.backend.main:app --reload

ui:
	python app.py

test:
	pytest -q

lint:
	ruff check .
