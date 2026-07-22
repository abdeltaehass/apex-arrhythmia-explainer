#!/usr/bin/env python3
"""LoRA supervised fine-tune of an open instruct model on the Phase-6 report dataset.

    # production run — needs a CUDA GPU w/ >= 24GB (QLoRA 4-bit) for the 7B default
    python -m src.generation.train_lora --load-in-4bit --bf16

    # a specific base model / GPU-poor setting
    python -m src.generation.train_lora --base Qwen/Qwen2.5-3B-Instruct --load-in-4bit

    # tiny end-to-end smoke test — runs anywhere (CPU/MPS), no GPU required
    python -m src.generation.train_lora --smoke

Default base is ``mistralai/Mistral-7B-Instruct-v0.3`` per the Phase-6 spec. Training
data is ``data/processed/generation/{train,val}.jsonl`` (build with
``scripts/build_gen_dataset.py`` / ``make gen-data``): each row is one
(structured-detection input, target report) pair from `src/generation/dataset.py`,
turned here into the exact chat-format conversation (`prompts.build_chat_example`) so
the model is trained on precisely the prompt it will see at inference.

Uses `trl.SFTTrainer`'s "prompt-completion" format (``prompt``/``completion`` message
lists, see `prompts.build_prompt_completion_example`) with ``completion_only_loss=True``
so the loss is computed only on the target report tokens, never on the system/user
turns — via a token-length diff between ``prompt`` and ``prompt + completion``, which
works regardless of whether the base model's chat template defines TRL's
``{% generation %}`` masking markers (many don't). LoRA adapters attach to every
linear projection (``target_modules="all-linear"``) so the same config works across
base architectures without hand-listing module names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import CFG, PROCESSED_DIR, ROOT
from src.generation.dataset import example_to_structured_input
from src.generation.prompts import build_prompt_completion_example

GEN_DATA_DIR = PROCESSED_DIR / "generation"
OUT_DIR = ROOT / "outputs"

DEFAULT_BASE = "mistralai/Mistral-7B-Instruct-v0.3"
SMOKE_BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"


def to_prompt_completion(example: dict, use_reference: bool = False) -> dict:
    """One dataset JSONL row -> ``{"prompt": [...], "completion": [...]}`` for `SFTTrainer`.

    ``use_reference=True`` trains against PTB-XL's own human report text instead of
    the template rendering, for the ablation described in the dataset card — off by
    default since the human report is free-form German prose, not the structured
    two-section target format.
    """
    si = example_to_structured_input(example)
    findings, impression = example["findings"], example["impression"]
    if use_reference and example.get("original_report"):
        findings, impression = example["original_report"], ""
    return build_prompt_completion_example(si, findings, impression)


def load_jsonl_dataset(path: Path, limit: int | None = None):
    from datasets import Dataset

    rows = [json.loads(line) for line in path.open()]
    if limit:
        rows = rows[:limit]
    ds = Dataset.from_list([to_prompt_completion(r) for r in rows])
    return ds


def build_lora_config(r: int, alpha: int, dropout: float):
    from peft import LoraConfig

    return LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules="all-linear", task_type="CAUSAL_LM", bias="none",
    )


def build_quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )


def check_max_length(dataset, tokenizer, max_length: int, name: str) -> None:
    """Fail loudly if any example's prompt+completion would be truncated.

    `SFTConfig`'s default `truncation_mode="keep_start"` truncates from the *end*,
    which silently drops the completion first if a sequence overruns ``max_length`` —
    every label goes to -100 and the reported loss/accuracy sit at exactly 0.0 for that
    example with no error at all (this is exactly what happened during development;
    see the smoke-mode comment in `main`). Checked once up front instead of trusting
    the training loop to surface it.
    """
    lengths = [
        len(tokenizer.apply_chat_template(ex["prompt"] + ex["completion"]))
        for ex in dataset
    ]
    n_over = sum(n > max_length for n in lengths)
    if n_over:
        raise ValueError(
            f"{n_over}/{len(lengths)} {name} examples exceed --max-length {max_length} "
            f"(max seen: {max(lengths)}). truncation_mode=\"keep_start\" would silently "
            f"drop their completion tokens (zero loss/accuracy, no error). Raise "
            f"--max-length to at least {max(lengths)}."
        )
    print(f"  {name}: max {max(lengths)} tokens, all within --max-length {max_length}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=DEFAULT_BASE, help="HF model id or local path")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--train-file", default=str(GEN_DATA_DIR / "train.jsonl"))
    ap.add_argument("--val-file", default=str(GEN_DATA_DIR / "val.jsonl"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA (needs CUDA + bitsandbytes)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--use-reference-text", action="store_true",
                    help="train on PTB-XL's own human report text instead of the template")
    ap.add_argument("--wandb-mode", default=CFG.wandb.mode, choices=("online", "offline", "disabled"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model + tiny data subset; runs end-to-end on CPU/MPS, no GPU needed")
    args = ap.parse_args()

    if args.smoke:
        args.base = SMOKE_BASE if args.base == DEFAULT_BASE else args.base
        args.epochs, args.batch_size, args.grad_accum = 1.0, 2, 1
        # NB: don't shrink this much further — real examples run ~400-700 tokens
        # (worst case ~9 findings), and `truncation_mode="keep_start"` silently drops
        # the *completion* first if max_length is too tight, zeroing the loss mask
        # without any error (bit us once; see the dataset card / PR notes).
        args.max_length = 768
        args.lora_r, args.lora_alpha = 8, 16
        args.train_limit, args.val_limit = args.train_limit or 40, args.val_limit or 10
        args.load_in_4bit, args.bf16 = False, False
        args.wandb_mode = "disabled"

    run_name = args.run_name or ("gen_lora_smoke" if args.smoke else f"gen_lora_{args.base.split('/')[-1]}")
    output_dir = Path(args.output_dir) if args.output_dir else OUT_DIR / run_name
    print(f"run={run_name}  base={args.base}  4bit={args.load_in_4bit}  out={output_dir}")

    train_ds = load_jsonl_dataset(Path(args.train_file), args.train_limit)
    val_ds = load_jsonl_dataset(Path(args.val_file), args.val_limit) if Path(args.val_file).exists() else None
    print(f"train examples: {len(train_ds)}" + (f"  val: {len(val_ds)}" if val_ds else ""))

    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    check_max_length(train_ds, tokenizer, args.max_length, "train")
    if val_ds is not None:
        check_max_length(val_ds, tokenizer, args.max_length, "val")

    model_kwargs = {}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = build_quantization_config()
        model_kwargs["dtype"] = "bfloat16"

    cfg = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        bf16=args.bf16,
        completion_only_loss=True,    # loss only on the target report tokens
        packing=False,                # keep examples separate for clean masking
        logging_steps=5,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=max(1, len(train_ds) // max(1, args.batch_size) // 2),
        save_strategy="epoch",
        report_to=[] if args.wandb_mode == "disabled" else ["wandb"],
        seed=args.seed,
        model_init_kwargs=model_kwargs or None,
    )
    peft_config = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)

    trainer = SFTTrainer(
        model=args.base,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    trainer.train()
    trainer.save_model(str(output_dir))
    print(f"adapter saved -> {output_dir}")

    if val_ds is not None and len(val_ds) > 0:
        _print_sample_generation(trainer, val_ds[0])
    return 0


def _print_sample_generation(trainer, example: dict) -> None:
    """Generate on one held-out example right after training, as a sanity spot-check."""
    tok = trainer.processing_class
    inputs = tok.apply_chat_template(example["prompt"], add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(trainer.model.device)
    out = trainer.model.generate(inputs, max_new_tokens=200, do_sample=False)
    text = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
    print("\n--- sample generation on a held-out example ---")
    print(text)
    print("--- reference target ---")
    print(example["completion"][0]["content"])


if __name__ == "__main__":
    raise SystemExit(main())
