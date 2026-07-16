# TRANSFORMER + bce — per-label AUROC (validation fold)

_transformer / bce loss, 1,859,399 params, best epoch 15_

- **Macro-AUROC: 0.8958**  (over 71 labels with both classes present in val)
- Macro-F1 0.3261 · Micro-F1 0.5755 · ECE 0.8860 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| 3AVB   | third degree AV block                                  |              12 |             2 | 0.998166 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.99478  |
| INJLA  | subendocardial injury in lateral leads                 |              13 |             2 | 0.994039 |
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.99381  |
| PRC(S) | premature complex(es)                                  |               8 |             1 | 0.991751 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.990673 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.987666 |
| INJIN  | subendocardial injury in inferior leads                |              14 |             2 | 0.986245 |
| IPMI   | inferoposterior myocardial infarction                  |              26 |             4 | 0.982905 |
| LAFB   | left anterior fascicular block                         |            1298 |           163 | 0.979223 |

### Weakest 10 labels (scored)
| code   | description                                                  |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| VCLVH  | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.78638  |
| IVCD   | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.77322  |
| ABQRS  | abnormal QRS                                                 |            2683 |           322 | 0.732095 |
| SEHYP  | septal hypertrophy                                           |              24 |             3 | 0.722936 |
| SARRH  | sinus arrhythmia                                             |             618 |            77 | 0.715556 |
| PAC    | atrial premature complex                                     |             318 |            40 | 0.70714  |
| STE_   | non-specific ST elevation                                    |              22 |             3 | 0.704128 |
| TAB_   | T-wave abnormality                                           |              28 |             4 | 0.685177 |
| HVOLT  | high QRS voltage                                             |              49 |             7 | 0.646731 |
| PMI    | posterior myocardial infarction                              |              13 |             2 | 0.63182  |
