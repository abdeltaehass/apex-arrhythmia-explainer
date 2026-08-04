# Per-label demographic subgroup performance

`APEX (cnn_bce)` on the **PTB-XL test (fold 10)** (2198 records). Regenerate with `python scripts/subgroup_analysis.py`. This is the per-label companion to the macro-level breakdown in [`demographics.md`](demographics.md).

## Method, and why it is stricter than it looks

AUROC is computed per label within each subgroup. A label is only *tested* when it has at least **10 positives in both** subgroups of a comparison — an AUROC computed on three positive cases is noise wearing a number's clothes. Every gap carries a percentile bootstrap CI (400 resamples) and a two-sided bootstrap p-value, and all p-values within a comparison are **Benjamini-Hochberg FDR-corrected**. Only **q < 0.05** is called a finding (marked **\***).

Without that correction this table would be a disparity generator: testing ~30 labels at α = 0.05 produces one or two 'significant' gaps by chance alone.

**Subgroup sizes (test split):**

| male | female | <40 | 40-60 | 60+ | <60 |
|---|---|---|---|---|---|
| 1132 | 1066 | 284 | 641 | 1239 | 925 |

## By sex

**34 of 71 labels** had enough positives in both sexes to test. After FDR correction, **0** show a significant gap.

Positive gap = better on **male** patients. Sorted by absolute gap; powered labels only.

| label | description | pos M/F | AUROC ♂ | AUROC ♀ | gap | 95% CI | q |
|---|---|---:|---:|---:|---:|---|---:|
| `NST_` | non-specific ST changes | 28 / 49 | 0.885 | 0.792 | **+0.093** | +0.024, +0.176 | 0.340 |
| `ISCIN` | ischemic in inferior leads | 12 / 10 | 0.945 | 0.869 | **+0.075** | -0.016, +0.168 | 0.574 |
| `LAO/LAE` | left atrial overload/enlargeme | 20 / 22 | 0.883 | 0.809 | **+0.074** | -0.023, +0.168 | 0.574 |
| `SARRH` | sinus arrhythmia | 29 / 48 | 0.839 | 0.896 | **-0.057** | -0.148, +0.025 | 0.589 |
| `QWAVE` | Q waves present | 30 / 25 | 0.796 | 0.848 | **-0.052** | -0.147, +0.049 | 0.710 |
| `LVH` | left ventricular hypertrophy | 109 / 105 | 0.862 | 0.899 | **-0.038** | -0.083, +0.009 | 0.574 |
| `STD_` | non-specific ST depression | 40 / 61 | 0.840 | 0.870 | **-0.030** | -0.083, +0.024 | 0.680 |
| `NDT` | non-diagnostic T abnormalities | 74 / 108 | 0.926 | 0.898 | **+0.028** | -0.014, +0.070 | 0.589 |
| `ISC_` | non-specific ischemic | 61 / 67 | 0.962 | 0.939 | **+0.024** | -0.007, +0.054 | 0.574 |
| `PACE` | normal functioning artificial  | 17 / 11 | 0.958 | 0.981 | **-0.023** | -0.089, +0.035 | 0.710 |
| `ISCAL` | ischemic in anterolateral lead | 32 / 34 | 0.951 | 0.928 | **+0.022** | -0.006, +0.052 | 0.574 |
| `ABQRS` | abnormal QRS | 195 / 127 | 0.790 | 0.770 | **+0.020** | -0.039, +0.072 | 0.710 |
| `IMI` | inferior myocardial infarction | 160 / 107 | 0.900 | 0.883 | **+0.018** | -0.014, +0.050 | 0.710 |
| `INVT` | inverted T-waves | 11 / 18 | 0.955 | 0.937 | **+0.017** | -0.013, +0.050 | 0.589 |
| `AMI` | anterior myocardial infarction | 21 / 14 | 0.859 | 0.876 | **-0.017** | -0.104, +0.070 | 0.885 |
| `ILMI` | inferolateral myocardial infar | 25 / 23 | 0.942 | 0.926 | **+0.016** | -0.048, +0.093 | 0.885 |
| `PAC` | atrial premature complex | 13 / 27 | 0.964 | 0.949 | **+0.015** | -0.027, +0.054 | 0.710 |
| `LOWT` | low amplitude T-waves | 17 / 27 | 0.916 | 0.902 | **+0.015** | -0.050, +0.064 | 0.710 |
| `1AVB` | first degree AV block | 39 / 40 | 0.962 | 0.977 | **-0.015** | -0.040, +0.005 | 0.589 |
| `IVCD` | non-specific intraventricular  | 45 / 34 | 0.735 | 0.721 | **+0.014** | -0.113, +0.131 | 0.965 |

**No individual label survives FDR correction.** The macro-level sex gap reported in `demographics.md` (+0.019, CI excludes zero) is therefore best read as a *broad, diffuse* effect — many labels each slightly worse for female patients — rather than one or two badly-behaved findings. That is a meaningfully different deployment story: there is no single label to patch.

## By age bracket

The `<40` bracket is small, so the three-way split leaves most labels untestable. Each contrast below reports how many labels it could actually support; the `<60 vs 60+` row is a pragmatic secondary split with more power.

| contrast | labels powered | significant after FDR |
|---|---:|---:|
| <40 vs 60+ | 7 / 71 | 3 |
| 40-60 vs 60+ | 22 / 71 | 3 |
| <60 vs 60+ | 23 / 71 | 11 |

### <40 vs 60+

Positive gap = better on the **younger** group. 7 labels testable.

| label | description | pos y/o | AUROC young | AUROC old | gap | 95% CI | q |
|---|---|---:|---:|---:|---:|---|---:|
| `NORM` **\*** | normal ECG | 236 / 339 | 0.814 | 0.936 | **-0.122** | -0.194, -0.063 | 0.000 |
| `ABQRS` | abnormal QRS | 17 / 201 | 0.675 | 0.774 | **-0.099** | -0.276, +0.058 | 0.420 |
| `SR` **\*** | sinus rhythm | 229 / 894 | 0.827 | 0.914 | **-0.087** | -0.155, -0.027 | 0.000 |
| `SARRH` | sinus arrhythmia | 31 / 27 | 0.812 | 0.872 | **-0.060** | -0.177, +0.058 | 0.420 |
| `NDT` | non-diagnostic T abnormalities | 13 / 116 | 0.861 | 0.912 | **-0.052** | -0.207, +0.046 | 0.467 |
| `SBRAD` **\*** | sinus bradycardia | 11 / 29 | 0.999 | 0.948 | **+0.051** | +0.015, +0.101 | 0.000 |
| `IRBBB` | incomplete right bundle branch | 19 / 57 | 0.957 | 0.962 | **-0.005** | -0.045, +0.029 | 0.795 |

### 40-60 vs 60+

Positive gap = better on the **younger** group. 22 labels testable.

| label | description | pos y/o | AUROC young | AUROC old | gap | 95% CI | q |
|---|---|---:|---:|---:|---:|---|---:|
| `IVCD` | non-specific intraventricular  | 23 / 52 | 0.656 | 0.767 | **-0.112** | -0.274, +0.039 | 0.257 |
| `LAO/LAE` | left atrial overload/enlargeme | 11 / 30 | 0.896 | 0.789 | **+0.107** | -0.004, +0.219 | 0.165 |
| `STD_` | non-specific ST depression | 14 / 80 | 0.899 | 0.811 | **+0.088** | +0.023, +0.149 | 0.073 |
| `AMI` | anterior myocardial infarction | 10 / 23 | 0.899 | 0.817 | **+0.082** | -0.019, +0.183 | 0.210 |
| `SR` **\*** | sinus rhythm | 535 / 894 | 0.834 | 0.914 | **-0.080** | -0.134, -0.024 | 0.000 |
| `ISCAL` **\*** | ischemic in anterolateral lead | 13 / 51 | 0.979 | 0.912 | **+0.067** | +0.041, +0.091 | 0.000 |
| `NT_` | non-specific T-wave changes | 11 / 30 | 0.960 | 0.897 | **+0.062** | +0.009, +0.118 | 0.066 |
| `ASMI` **\*** | anteroseptal myocardial infarc | 35 / 188 | 0.979 | 0.932 | **+0.048** | +0.027, +0.073 | 0.000 |
| `AFIB` | atrial fibrillation | 14 / 129 | 0.938 | 0.983 | **-0.045** | -0.170, +0.020 | 0.689 |
| `IMI` | inferior myocardial infarction | 62 / 197 | 0.912 | 0.868 | **+0.044** | +0.009, +0.084 | 0.055 |
| `SARRH` | sinus arrhythmia | 19 / 27 | 0.912 | 0.872 | **+0.040** | -0.036, +0.131 | 0.440 |
| `VCLVH` | voltage criteria (QRS) for lef | 17 / 63 | 0.782 | 0.755 | **+0.027** | -0.092, +0.142 | 0.689 |
| `LVH` | left ventricular hypertrophy | 31 / 169 | 0.847 | 0.874 | **-0.026** | -0.098, +0.036 | 0.667 |
| `NORM` | normal ECG | 388 / 339 | 0.910 | 0.936 | **-0.026** | -0.051, +0.003 | 0.165 |
| `NDT` | non-diagnostic T abnormalities | 47 / 116 | 0.936 | 0.912 | **+0.023** | -0.013, +0.056 | 0.296 |

### <60 vs 60+

Positive gap = better on the **younger** group. 23 labels testable.

| label | description | pos y/o | AUROC young | AUROC old | gap | 95% CI | q |
|---|---|---:|---:|---:|---:|---|---:|
| `LAO/LAE` **\*** | left atrial overload/enlargeme | 11 / 30 | 0.920 | 0.789 | **+0.131** | +0.034, +0.231 | 0.023 |
| `AMI` **\*** | anterior myocardial infarction | 12 / 23 | 0.931 | 0.817 | **+0.115** | +0.026, +0.210 | 0.042 |
| `IVCD` | non-specific intraventricular  | 27 / 52 | 0.657 | 0.767 | **-0.111** | -0.257, +0.045 | 0.221 |
| `STD_` **\*** | non-specific ST depression | 17 / 80 | 0.910 | 0.811 | **+0.099** | +0.045, +0.152 | 0.000 |
| `SR` **\*** | sinus rhythm | 764 / 894 | 0.834 | 0.914 | **-0.080** | -0.125, -0.042 | 0.000 |
| `ISCAL` **\*** | ischemic in anterolateral lead | 13 / 51 | 0.985 | 0.912 | **+0.073** | +0.050, +0.103 | 0.000 |
| `NT_` **\*** | non-specific T-wave changes | 11 / 30 | 0.970 | 0.897 | **+0.072** | +0.022, +0.127 | 0.014 |
| `IMI` **\*** | inferior myocardial infarction | 65 / 197 | 0.932 | 0.868 | **+0.063** | +0.030, +0.099 | 0.000 |
| `ASMI` **\*** | anteroseptal myocardial infarc | 37 / 188 | 0.980 | 0.932 | **+0.048** | +0.025, +0.070 | 0.000 |
| `AFIB` | atrial fibrillation | 15 / 129 | 0.939 | 0.983 | **-0.044** | -0.190, +0.020 | 0.920 |
| `NORM` **\*** | normal ECG | 624 / 339 | 0.897 | 0.936 | **-0.038** | -0.064, -0.011 | 0.000 |
| `ISC_` | non-specific ischemic | 18 / 104 | 0.967 | 0.936 | **+0.031** | -0.002, +0.058 | 0.134 |
| `LVH` | left ventricular hypertrophy | 40 / 169 | 0.843 | 0.874 | **-0.030** | -0.096, +0.032 | 0.468 |
| `SBRAD` | sinus bradycardia | 34 / 29 | 0.969 | 0.948 | **+0.021** | -0.020, +0.072 | 0.697 |
| `ABQRS` | abnormal QRS | 117 / 201 | 0.792 | 0.774 | **+0.017** | -0.040, +0.074 | 0.697 |

### The age pattern is consistent, and it points the wrong way

**All 9 significant *pathology* labels are worse in the 60+ group** (gaps +0.008 to +0.131): `AMI`, `ASMI`, `IMI`, `ISCAL`, `LAFB`, `LAO/LAE`, `NT_`, `PVC`, `STD_`.

These are ischemia and infarction findings — anterior, anteroseptal and inferior MI, anterolateral ischemia, ST depression. **The model is weakest at detecting cardiac pathology in exactly the population that has the most of it**, and where a miss carries the most risk. That is the single most deployment-relevant result in this document.

The exceptions run the other way: `NORM` (-0.038), `SR` (-0.080) are *better* in older patients. That is a **case-mix** effect rather than a contradiction: `NORM` covers 83% of the under-40 cohort, so separating normal from abnormal in a nearly-all-normal group is a harder discrimination problem than in a mixed older one.

> **Read subgroup AUROC gaps with care.** AUROC depends on the difficulty mix within each subgroup, not only on model quality — a cohort whose positives are more advanced will score higher (spectrum bias). Some of the gap above is likely real model weakness on older, more co-morbid ECGs (consistent with the Phase-13 finding that multi-condition records lose secondary findings); some is case mix. This analysis cannot separate the two, and does not claim to.

## The two questions this phase was asked

**Does APEX detect atrial fibrillation equally well in older vs younger patients?**

Testable. Gap -0.045 (q = 0.689) — **no** difference that survives FDR correction. Support: male 81, female 71, <40 1, 40-60 14, 60+ 129.

**Does APEX perform differently by sex on ST-elevation detection?**

**Unanswerable from this test set.** `STE_` positives by subgroup: male 1, female 2, <40 2, 40-60 0, 60+ 1. That is below the 10-positive floor, so any AUROC quoted here would be an artifact of two or three cases. Reporting 'no disparity found' would be just as misleading as reporting one — the honest answer is that the measurement cannot be made, and would need a dataset enriched for this finding.

## What this means for deployment

- **The sex effect is diffuse, not localized.** No single label survives correction, while the macro gap does. There is no one finding to fix; a fairness intervention would have to act on the model as a whole (reweighting, stratified calibration), not on a patched label.
- **Age effects are mostly unmeasurable at this resolution.** PTB-XL's under-40 population is too small and too healthy to test most pathologies. Absence of a documented age disparity per label is **absence of evidence**, not evidence of absence.
- **The highest-consequence labels are the least measurable.** ST-elevation, the finding whose miss is most dangerous, has too few positives to audit for bias at all. Any deployment claiming equitable performance on acute findings needs a dataset built for that question.
- **Report subgroup performance alongside headline metrics, not instead of.** The macro AUROC of 0.920 is true and hides both a real diffuse sex gap and a large region of simply-unknown behaviour.

## Limitations

- PTB-XL records **no race or ethnicity**, so fairness here is verified along two axes only; disparities on unrecorded axes cannot be ruled out.
- Sex is recorded binary in the dataset; this analysis inherits that limitation and says nothing about intersex or transgender patients.
- Ground-truth labels are human annotations that carry their own historical biases; a model matching biased labels can look 'fair' while reproducing them.
- AUROC is threshold-free. Two subgroups with equal AUROC can still receive different *decisions* at a shared threshold if their score distributions differ — worth re-checking after the Phase-17 calibrator is applied in serving.
