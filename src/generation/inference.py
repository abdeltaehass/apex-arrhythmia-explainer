"""Explanation generation inference — three interchangeable backends, one contract.

Every backend takes a `templater.StructuredInput` (the same object
`src/generation/dataset.py` builds training pairs from) and returns raw two-section
text (``Findings:`` / ``Impression:``). None of them enforces the "only assert
provided findings" rule themselves — that check always happens downstream in
`src/eval/consistency.py` against `parse.asserted_findings`, so a backend that ignores
its prompt is still caught rather than trusted.

Backends:
    "template" — deterministic, `templater.render_report`. Always available, always
                 consistent by construction; the training target and the fallback.
    "claude"   — Anthropic API (`ANTHROPIC_API_KEY`). Used for the Phase-6 review
                 examples in lieu of a GPU to run the LoRA fine-tune end-to-end.
    "local"    — the LoRA-fine-tuned open model from `train_lora.py` (base model +
                 adapter directory), loaded once and reused across calls.
"""

from __future__ import annotations

from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt, target_text
from src.generation.templater import StructuredInput, render_report

DISCLAIMER = "Decision support only — verify against the full clinical picture."

BACKENDS = ("template", "claude", "local")


def generate_with_template(si: StructuredInput) -> str:
    """The deterministic renderer, wrapped in the same text contract as the LLM backends."""
    rep = render_report(si)
    return target_text(rep["findings"], rep["impression"])


def generate_with_claude(si: StructuredInput, model: str = "claude-fable-5", max_tokens: int = 600) -> str:
    """Requires ``ANTHROPIC_API_KEY``. Thin by design — see module docstring."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install anthropic to use the Claude backend") from e

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(si)}],
    )
    return resp.content[0].text


_LOCAL_CACHE: dict[str, tuple] = {}  # adapter_dir -> (model, tokenizer), loaded once per process


def _load_local(base_model: str, adapter_dir: str, device: str = "auto"):
    if adapter_dir in _LOCAL_CACHE:
        return _LOCAL_CACHE[adapter_dir]
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pip install transformers peft torch to use the local fine-tuned backend"
        ) from e

    if device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(base_model, dtype="auto").to(device)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    model.eval()
    _LOCAL_CACHE[adapter_dir] = (model, tok)
    return model, tok


def generate_with_local(
    si: StructuredInput,
    adapter_dir: str,
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3",
    max_new_tokens: int = 300,
    device: str = "auto",
) -> str:
    """Generate with a `train_lora.py` LoRA adapter. Loads the base + adapter once,
    cached in-process by ``adapter_dir`` (swap adapters to compare runs)."""
    import torch

    model, tok = _load_local(base_model, adapter_dir, device)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(si)},
    ]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def generate_explanation(si: StructuredInput, backend: str = "claude", **kwargs) -> str:
    """Dispatch to a backend by name. See module docstring for the three options."""
    if backend == "template":
        return generate_with_template(si)
    if backend == "claude":
        return generate_with_claude(si, **kwargs)
    if backend == "local":
        return generate_with_local(si, **kwargs)
    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
