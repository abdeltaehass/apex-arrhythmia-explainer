# Distilled student (kd) — per-label AUROC (validation fold)

_student width=32 blocks=1, 988,487 params, alpha=0.7, T=2.0, best epoch 17_

- **Macro-AUROC: 0.9206**  (over 71 labels with both classes present in val)
- Macro-F1 0.3674 · Micro-F1 0.5974 · ECE 0.0781 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.998395 |
| 2AVB   | second degree AV block                                 |              12 |             1 | 0.996334 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.996053 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.995719 |
| INJLA  | subendocardial injury in lateral leads                 |              13 |             2 | 0.995415 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.995    |
| INJIN  | subendocardial injury in inferior leads                |              14 |             2 | 0.992893 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.992676 |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.992194 |
| PVC    | ventricular premature complex                          |             915 |           114 | 0.992161 |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| PMI     | posterior myocardial infarction                              |              13 |             2 | 0.843879 |
| ISCLA   | ischemic in lateral leads                                    |             113 |            14 | 0.838438 |
| LVOLT   | low QRS voltages in the frontal and horizontal leads         |             145 |            19 | 0.81686  |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.802996 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.792302 |
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.790991 |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.759651 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.74029  |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.686668 |
| STE_    | non-specific ST elevation                                    |              22 |             3 | 0.675535 |
