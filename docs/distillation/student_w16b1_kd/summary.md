# Distilled student (kd) — per-label AUROC (validation fold)

_student width=16 blocks=1, 253,895 params, alpha=0.7, T=2.0, best epoch 19_

- **Macro-AUROC: 0.9143**  (over 71 labels with both classes present in val)
- Macro-F1 0.3478 · Micro-F1 0.5743 · ECE 0.0787 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.998395 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.995172 |
| AFLT   | atrial flutter                                         |              59 |             7 | 0.994945 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.994028 |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.992146 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.992059 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.990826 |
| PVC    | ventricular premature complex                          |             915 |           114 | 0.989091 |
| INJLA  | subendocardial injury in lateral leads                 |              13 |             2 | 0.986933 |
| AFIB   | atrial fibrillation                                    |            1211 |           151 | 0.986429 |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| ISCLA   | ischemic in lateral leads                                    |             113 |            14 | 0.835013 |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.803726 |
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.802815 |
| PMI     | posterior myocardial infarction                              |              13 |             2 | 0.795965 |
| LVOLT   | low QRS voltages in the frontal and horizontal leads         |             145 |            19 | 0.791152 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.787545 |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.785255 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.729447 |
| STE_    | non-specific ST elevation                                    |              22 |             3 | 0.636391 |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.625287 |
