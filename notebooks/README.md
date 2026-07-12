# notebooks/

EDA and experiment logs.

- `01_eda.ipynb` — label frequency across the 71 SCP codes, class imbalance,
  demographics, per-fold distribution, sanity-check a few waveforms.

Suggested first pass (after `make data-meta`):

```python
from src.data.labels import load_database, load_scp_statements, build_label_space

db = load_database()
scp = load_scp_statements()
labels = build_label_space()
db["scp_codes"].apply(lambda d: list(d)).explode().value_counts()  # label frequencies
```

Keep exploratory notebooks out of the training path — anything reusable graduates
into `src/`.
