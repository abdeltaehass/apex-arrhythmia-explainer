# CNN + bce — per-label AUROC (validation fold)

_cnn / bce loss, 8,778,055 params, best epoch 14_

- **Macro-AUROC: 0.9174**  (over 71 labels with both classes present in val)
- Macro-F1 0.3688 · Micro-F1 0.6025 · ECE 0.8965 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.999083 |
| INJIN  | subendocardial injury in inferior leads                |              14 |             2 | 0.997707 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.995514 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.99546  |
| 2AVB   | second degree AV block                                 |              12 |             1 | 0.9945   |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.99276  |
| PVC    | ventricular premature complex                          |             915 |           114 | 0.992555 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.992528 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.991896 |
| AFLT   | atrial flutter                                         |              59 |             7 | 0.99094  |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| ISCLA   | ischemic in lateral leads                                    |             113 |            14 | 0.851248 |
| QWAVE   | Q waves present                                              |             438 |            55 | 0.830084 |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.810586 |
| LVOLT   | low QRS voltages in the frontal and horizontal leads         |             145 |            19 | 0.801634 |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.788209 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.784037 |
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.777733 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.742201 |
| STE_    | non-specific ST elevation                                    |              22 |             3 | 0.711315 |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.687701 |
