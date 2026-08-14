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
    "hf"       — any local instruct model by Hub id, no adapter. Added in Phase 21 so the
                 RAG evaluation can run a competent open model offline: measuring whether
                 retrieval changes the hallucination rate needs a generator that actually
                 writes clinical prose, and neither the deterministic template (which
                 cannot hallucinate by construction) nor the 135M smoke adapter (which
                 emits no diagnoses at all) can serve as the subject.

Every backend takes an optional ``context`` string — the retrieved reference block from
`src.rag` — which is appended to the user turn by `prompts.build_user_prompt`. Passing
nothing reproduces the Phase-6 prompt exactly, which is what makes the RAG on/off
comparison a controlled one.
"""

from __future__ import annotations

from src.generation.prompts import (
    build_user_prompt,
    system_prompt,
    target_text,
)
from src.generation.templater import StructuredInput, render_report

DISCLAIMER = "Decision support only — verify against the full clinical picture."

BACKENDS = ("template", "claude", "local", "hf")


def generate_with_template(si: StructuredInput, lang: str = "en") -> str:
    """The deterministic renderer, wrapped in the same text contract as the LLM backends.

    Phase 27: with ``lang`` it renders through :mod:`src.i18n.render`, which shares one code
    path across languages — English output is byte-identical to the pre-Phase-27 renderer,
    asserted by test.
    """
    if lang and lang != "en":
        from src.i18n.languages import get_language
        from src.i18n.render import render_report as render_i18n

        bank = get_language(lang)
        rep = render_i18n(si, bank)
        return (f"{bank.findings_header}:\n{rep['findings']}\n\n"
                f"{bank.impression_header}:\n{rep['impression']}")
    rep = render_report(si)
    return target_text(rep["findings"], rep["impression"])


def generate_with_claude(si: StructuredInput, model: str = "claude-fable-5",
                         max_tokens: int = 600, context: str = "",
                         lang: str = "en") -> str:
    """Requires ``ANTHROPIC_API_KEY``. Thin by design — see module docstring."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install anthropic to use the Claude backend") from e

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt(lang),
        messages=[{"role": "user", "content": build_user_prompt(si, context)}],
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
    context: str = "",
    lang: str = "en",
) -> str:
    """Generate with a `train_lora.py` LoRA adapter. Loads the base + adapter once,
    cached in-process by ``adapter_dir`` (swap adapters to compare runs)."""
    import torch

    model, tok = _load_local(base_model, adapter_dir, device)
    messages = [
        {"role": "system", "content": system_prompt(lang)},
        {"role": "user", "content": build_user_prompt(si, context)},
    ]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


_HF_CACHE: dict[tuple[str, str], tuple] = {}


def _load_hf(model_id: str, device: str = "auto"):
    if device == "auto":
        import torch
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    key = (model_id, device)
    if key not in _HF_CACHE:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device).eval()
        _HF_CACHE[key] = (model, tok)
    return _HF_CACHE[key]


def generate_with_hf(
    si: StructuredInput,
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 320,
    device: str = "auto",
    context: str = "",
    lang: str = "en",
) -> str:
    """Generate with a plain local instruct model (no adapter), cached per process.

    Greedy decoding (``do_sample=False``): the RAG comparison is a paired before/after on
    the same records, so sampling noise would sit directly on top of the effect being
    measured.
    """
    import torch

    model, tok = _load_hf(model_id, device)
    messages = [
        {"role": "system", "content": system_prompt(lang)},
        {"role": "user", "content": build_user_prompt(si, context)},
    ]
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def generate_explanation(si: StructuredInput, backend: str = "claude",
                         context: str = "", lang: str = "en", **kwargs) -> str:
    """Dispatch to a backend by name. See module docstring for the four options.

    ``context`` is the optional retrieved reference block (Phase 21). The template backend
    ignores it — it renders from the structured input alone and so is consistent by
    construction, which is exactly why it is the wrong thing to measure RAG on.

    ``lang`` (Phase 27) selects the output language. The template backend renders it
    directly; the LLM backends receive a language clause appended to the system prompt.
    Whatever comes back must still clear the consistency gate in that language — see
    :mod:`src.i18n.parse`.
    """
    from src.i18n.languages import get_language

    lang = get_language(lang).code          # validates; raises on an unsupported language
    if backend == "template":
        return generate_with_template(si, lang=lang)
    if backend == "claude":
        return generate_with_claude(si, context=context, lang=lang, **kwargs)
    if backend == "local":
        return generate_with_local(si, context=context, lang=lang, **kwargs)
    if backend == "hf":
        return generate_with_hf(si, context=context, lang=lang, **kwargs)
    raise ValueError(f"unknown backend {backend!r}, expected one of {BACKENDS}")
