# notebooks/

EDA and experiment logs.

- `01_eda.ipynb` — label frequency across the 71 SCP codes, class imbalance,
  demographics, multi-label density, and the patient-level train/val/test split.
  Runs on metadata only (`make data-meta`). Committed with outputs so it can be
  reviewed without a kernel.
- `02_preprocessing.ipynb` — Phase 2 sanity checks: raw vs. clean signals for 7
  records across all six diagnostic groups, baseline-wander removal, 500→100 Hz
  resampling, Pan-Tompkins internals, per-lead normalization, and the torch Dataset.
  Needs the curated waveforms (`make data-sample`). Committed with plots.
- `03_baseline_comparison.ipynb` — Phase 12: APEX on the PTB-XL test split vs the
  published benchmark (Strodthoff et al. 2021) + a GPT-4o zero-shot ECG-image baseline.
  Loads the precomputed result JSONs (`docs/model_comparison/`), so it runs in seconds
  without the model. Committed with tables + charts. Regenerate the results with
  `make eval-baselines`.

The reusable logic lives in `src/data/eda.py` (analysis + plots) and
`src/data/manifests.py` (splits). Regenerate the static artifacts with:

```bash
python scripts/run_eda.py     # -> docs/eda/ (figures, prevalence CSVs, summary.md)
python -m src.data.manifests  # -> data/manifests/ (gitignored, reproducible)
```

Keep exploratory notebooks out of the training path — anything reusable graduates
into `src/`.
