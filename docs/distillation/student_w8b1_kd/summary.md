# Distilled student (kd) — per-label AUROC (validation fold)

_student width=8 blocks=1, 66,887 params, alpha=0.7, T=2.0, best epoch 20_

- **Macro-AUROC: 0.8931**  (over 71 labels with both classes present in val)
- Macro-F1 0.3144 · Micro-F1 0.5324 · ECE 0.0825 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.997478 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.99526  |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.994344 |
| AFLT   | atrial flutter                                         |              59 |             7 | 0.993829 |
| SVTAC  | supraventricular tachycardia                           |              21 |             3 | 0.991131 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.990423 |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.987797 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.987299 |
| PACE   | normal functioning artificial pacemaker                |             237 |            29 | 0.984856 |
| INJLA  | subendocardial injury in lateral leads                 |              13 |             2 | 0.984411 |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.774332 |
| LMI     | lateral myocardial infarction                                |             161 |            20 | 0.773185 |
| PMI     | posterior myocardial infarction                              |              13 |             2 | 0.767079 |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.760491 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.74267  |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.728007 |
| SEHYP   | septal hypertrophy                                           |              24 |             3 | 0.712997 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.705017 |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.663148 |
| STE_    | non-specific ST elevation                                    |              22 |             3 | 0.484251 |
