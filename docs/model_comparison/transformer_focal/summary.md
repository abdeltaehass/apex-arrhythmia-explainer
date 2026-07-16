# TRANSFORMER + focal — per-label AUROC (validation fold)

_transformer / focal loss, 1,859,399 params, best epoch 18_

- **Macro-AUROC: 0.8943**  (over 71 labels with both classes present in val)
- Macro-F1 0.3321 · Micro-F1 0.5214 · ECE 0.7901 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.997095 |
| INJLA  | subendocardial injury in lateral leads                 |              13 |             2 | 0.99679  |
| 3AVB   | third degree AV block                                  |              12 |             2 | 0.99679  |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.995062 |
| AFLT   | atrial flutter                                         |              59 |             7 | 0.993566 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.992607 |
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.991518 |
| INJIN  | subendocardial injury in inferior leads                |              14 |             2 | 0.990142 |
| IPMI   | inferoposterior myocardial infarction                  |              26 |             4 | 0.984741 |
| SVTAC  | supraventricular tachycardia                           |              21 |             3 | 0.984709 |

### Weakest 10 labels (scored)
| code   | description                                                  |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| ISCLA  | ischemic in lateral leads                                    |             113 |            14 | 0.784364 |
| VCLVH  | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.781878 |
| IVCD   | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.752835 |
| PAC    | atrial premature complex                                     |             318 |            40 | 0.750012 |
| PMI    | posterior myocardial infarction                              |              13 |             2 | 0.744613 |
| SEHYP  | septal hypertrophy                                           |              24 |             3 | 0.735474 |
| ABQRS  | abnormal QRS                                                 |            2683 |           322 | 0.718177 |
| SARRH  | sinus arrhythmia                                             |             618 |            77 | 0.694663 |
| TAB_   | T-wave abnormality                                           |              28 |             4 | 0.597866 |
| HVOLT  | high QRS voltage                                             |              49 |             7 | 0.48943  |
