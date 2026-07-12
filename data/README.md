# data/

The dataset is **not** committed (see `.gitignore`). Populate it with the download
script.

```
data/
  raw/
    ptbxl/                  <- populated by scripts/download_ptbxl.py
      ptbxl_database.csv    record metadata + scp_codes + strat_fold
      scp_statements.csv    SCP code -> description / diagnostic superclass
      records100/           100 Hz waveforms (.dat/.hea)
      records500/           500 Hz waveforms (.dat/.hea)
  processed/                cached preprocessed tensors / split arrays
  manifests/                per-split record lists, label matrices, thresholds
```

## Source

**PTB-XL, a large publicly available electrocardiography dataset** (PhysioNet).
Open access — no credentialing or data-use agreement required. 12-lead, 10 s each,
at 100 Hz and 500 Hz, with demographics and SCP-ECG labels. Pinned to **v1.0.3**
(**21,799** records; the often-quoted 21,837 is the original v1.0.1 count).

- Get metadata only (fast): `python scripts/download_ptbxl.py --metadata-only`
- Get everything: `python scripts/download_ptbxl.py`

## Splits

Use PTB-XL's official `strat_fold` column: folds 1–8 train, fold 9 validation,
fold 10 test. Do not tune on fold 10.
