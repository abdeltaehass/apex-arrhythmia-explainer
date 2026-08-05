# Distilled student (scratch) — per-label AUROC (validation fold)

_student width=32 blocks=1, 988,487 params, alpha=0.0, T=2.0, best epoch 17_

- **Macro-AUROC: 0.9172**  (over 71 labels with both classes present in val)
- Macro-F1 0.3850 · Micro-F1 0.5548 · ECE 0.0786 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.999083 |
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.997707 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.996609 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.996078 |
| INJIN  | subendocardial injury in inferior leads                |              14 |             2 | 0.995186 |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.993609 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.991432 |
| PVC    | ventricular premature complex                          |             915 |           114 | 0.989146 |
| PACE   | normal functioning artificial pacemaker                |             237 |            29 | 0.98897  |
| SVTAC  | supraventricular tachycardia                           |              21 |             3 | 0.987003 |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| SARRH   | sinus arrhythmia                                             |             618 |            77 | 0.846783 |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.826226 |
| LVOLT   | low QRS voltages in the frontal and horizontal leads         |             145 |            19 | 0.821456 |
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.813801 |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.798876 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.793568 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.733942 |
| PMI     | posterior myocardial infarction                              |              13 |             2 | 0.713434 |
| STE_    | non-specific ST elevation                                    |              22 |             3 | 0.654281 |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.644564 |
