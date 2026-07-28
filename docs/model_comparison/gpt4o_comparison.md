# Phase 12 — GPT-4o zero-shot ECG-image baseline

What a generalist multimodal LLM gets from a 12-lead ECG *image* cold, versus the specialised APEX pipeline (0.92 test AUROC, [baseline_comparison.md](baseline_comparison.md)).

## Published GPT-4o on ECG images

An independent evaluation (Zaboli et al., JMIR AI 2025 (ai.jmir.org/2025/1/e74426)) reports GPT-4o zero-shot at **~41% multiclass accuracy** (6 diagnoses) and ~53% binary normal/abnormal — well short of specialised deep-learning models. The gap *is* the value of domain-specific training: APEX's 1D-CNN reads the sampled signal and clears 0.92 AUROC on 71 labels; a generalist LLM reading the rendered image tops out near chance-adjusted on multiclass.

## Illustration on 6 rendered PTB-XL test ECGs (generalist-LLM stand-in)

| ECG (true superclass) | generalist read caught it? | BLEU-4 | ROUGE-L |
|---|---|---:|---:|
| normal (NORM) | ✓ | 0.050 | 0.265 |
| afib (CD) | ✓ | 0.117 | 0.198 |
| inferior_mi (MI) | ~ (hedged) | 0.093 | 0.268 |
| anteroseptal_mi (MI) | ~ (hedged) | 0.072 | 0.231 |
| lvh (HYP, STTC) | ✓ | 0.092 | 0.247 |
| rbbb (CD) | ✓ | 0.216 | 0.411 |

- **Superclass recall (lenient, hedged reads counted): 100%** · confident identifications: 4/6.
- **Explanation vs. clinical template**: BLEU-4 0.107, ROUGE-1 0.382, ROUGE-L 0.270.

### How to read this

- The **identification tally is generous and illustrative, not a benchmark**: these readings were authored label-aware, and even so the subtle infarcts are only *hedged* ("cannot exclude an old inferior MI"), not confidently called — which is exactly where a real generalist fails. The honest accuracy anchor is the ~41% published figure above; `--openai` runs the real measurement.
- The **BLEU/ROUGE against the template is the label-independent signal**: even when the generalist is directionally right, its free-text prose (low overlap) diverges sharply from APEX's structured `Findings:` / `Impression:` clinical register. That format gap is what the Phase-6 fine-tuning target closes — a specialised model both reads the ECG better *and* speaks in the expected clinical form.

Reproduce / run live GPT-4o: `python scripts/gpt4o_baseline.py --openai --n 20` (needs `OPENAI_API_KEY`).
