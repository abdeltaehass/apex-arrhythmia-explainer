# CNN + focal — per-label AUROC (validation fold)

_cnn / focal loss, 8,778,055 params, best epoch 14_

- **Macro-AUROC: 0.9151**  (over 71 labels with both classes present in val)
- Macro-F1 0.3819 · Micro-F1 0.5587 · ECE 0.8163 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| INJIN  | subendocardial injury in inferior leads                |              14 |             2 | 0.998854 |
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.998166 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.996789 |
| AFLT   | atrial flutter                                         |              59 |             7 | 0.996389 |
| 2AVB   | second degree AV block                                 |              12 |             1 | 0.996334 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.995993 |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.99428  |
| PVC    | ventricular premature complex                          |             915 |           114 | 0.993407 |
| INJLA  | subendocardial injury in lateral leads                 |              13 |             2 | 0.993122 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.990287 |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| PRC(S)  | premature complex(es)                                        |               8 |             1 | 0.832264 |
| PMI     | posterior myocardial infarction                              |              13 |             2 | 0.830582 |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.811206 |
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.794947 |
| LVOLT   | low QRS voltages in the frontal and horizontal leads         |             145 |            19 | 0.794168 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.789555 |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.779543 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.733213 |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.729807 |
| STE_    | non-specific ST elevation                                    |              22 |             3 | 0.572477 |
