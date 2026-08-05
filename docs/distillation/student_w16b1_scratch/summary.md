# Distilled student (scratch) — per-label AUROC (validation fold)

_student width=16 blocks=1, 253,895 params, alpha=0.0, T=2.0, best epoch 15_

- **Macro-AUROC: 0.9116**  (over 71 labels with both classes present in val)
- Macro-F1 0.3378 · Micro-F1 0.5688 · ECE 0.1040 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.997937 |
| AFLT   | atrial flutter                                         |              59 |             7 | 0.995601 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.995523 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.995115 |
| 3AVB   | third degree AV block                                  |              12 |             2 | 0.994727 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.992966 |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.992639 |
| 2AVB   | second degree AV block                                 |              12 |             1 | 0.992209 |
| IPMI   | inferoposterior myocardial infarction                  |              26 |             4 | 0.989101 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.988823 |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| ISCLA   | ischemic in lateral leads                                    |             113 |            14 | 0.825594 |
| SARRH   | sinus arrhythmia                                             |             618 |            77 | 0.80866  |
| PMI     | posterior myocardial infarction                              |              13 |             2 | 0.80353  |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.801973 |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.795253 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.774706 |
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.755673 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.715699 |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.709887 |
| STE_    | non-specific ST elevation                                    |              22 |             3 | 0.642508 |
