# notebooks/

EDA and experiment logs.

- `01_eda.ipynb` — label frequency across the 71 SCP codes, class imbalance,
  demographics, multi-label density, and the patient-level train/val/test split.
  Runs on metadata only (`make data-meta`). Committed with outputs so it can be
  reviewed without a kernel.

The reusable logic lives in `src/data/eda.py` (analysis + plots) and
`src/data/manifests.py` (splits). Regenerate the static artifacts with:

```bash
python scripts/run_eda.py     # -> docs/eda/ (figures, prevalence CSVs, summary.md)
python -m src.data.manifests  # -> data/manifests/ (gitignored, reproducible)
```

Keep exploratory notebooks out of the training path — anything reusable graduates
into `src/`.
