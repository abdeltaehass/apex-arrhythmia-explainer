# Distilled student (scratch) — per-label AUROC (validation fold)

_student width=8 blocks=1, 66,887 params, alpha=0.0, T=2.0, best epoch 18_

- **Macro-AUROC: 0.8854**  (over 71 labels with both classes present in val)
- Macro-F1 0.2911 · Micro-F1 0.4632 · ECE 0.1275 · n_val 2183

> **Calibration caveat:** class-weighted training (`pos_weight` / focal) trades calibration for recall on rare labels, so raw probabilities run high and ECE is poor. AUROC (ranking) is the headline metric; probability calibration (temperature scaling) is follow-up work. F1 uses per-label thresholds tuned on this fold.

### Best 10 labels
| code   | description                                            |   train_support |   val_support |    auroc |
|:-------|:-------------------------------------------------------|----------------:|--------------:|---------:|
| TRIGU  | trigeminal pattern (unknown origin, SV or Ventricular) |              16 |             2 | 0.997478 |
| CRBBB  | complete right bundle branch block                     |             432 |            55 | 0.994318 |
| BIGU   | bigeminal pattern (unknown origin, SV or Ventricular)  |              66 |             8 | 0.992414 |
| 3AVB   | third degree AV block                                  |              12 |             2 | 0.991059 |
| PSVT   | paroxysmal supraventricular tachycardia                |              19 |             3 | 0.990214 |
| CLBBB  | complete left bundle branch block                      |             428 |            54 | 0.990119 |
| STACH  | sinus tachycardia                                      |             661 |            83 | 0.987711 |
| AFLT   | atrial flutter                                         |              59 |             7 | 0.986345 |
| PACE   | normal functioning artificial pacemaker                |             237 |            29 | 0.980405 |
| SVTAC  | supraventricular tachycardia                           |              21 |             3 | 0.979969 |

### Weakest 10 labels (scored)
| code    | description                                                  |   train_support |   val_support |    auroc |
|:--------|:-------------------------------------------------------------|----------------:|--------------:|---------:|
| SARRH   | sinus arrhythmia                                             |             618 |            77 | 0.751125 |
| LAO/LAE | left atrial overload/enlargement                             |             341 |            43 | 0.748261 |
| AMI     | anterior myocardial infarction                               |             282 |            36 | 0.742121 |
| SEHYP   | septal hypertrophy                                           |              24 |             3 | 0.735168 |
| IVCD    | non-specific intraventricular conduction disturbance (block) |             630 |            78 | 0.733358 |
| VCLVH   | voltage criteria (QRS) for left ventricular hypertrophy      |             701 |            87 | 0.700519 |
| ABQRS   | abnormal QRS                                                 |            2683 |           322 | 0.692552 |
| PMI     | posterior myocardial infarction                              |              13 |             2 | 0.686382 |
| HVOLT   | high QRS voltage                                             |              49 |             7 | 0.636226 |
| TAB_    | T-wave abnormality                                           |              28 |             4 | 0.501492 |
