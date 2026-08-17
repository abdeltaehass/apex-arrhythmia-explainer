"""Phase 28 — the systems under comparison, behind one interface.

Every arm — the trained detector, the distilled student, a local generalist LLM, a hosted
foundation model — is asked the same question about the same recording and returns the same
:class:`BenchOutput`. That is what makes the columns of the comparison table commensurable;
without it the table is five different experiments printed side by side.

Each system reports its own latency, measured around the work it actually does. For the
API-backed systems that necessarily includes network time, which is a real cost of the
architecture rather than a measurement artefact and is reported as such.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.benchmark.features import extract
from src.benchmark.parse_scores import parse_interpretation, parse_scores


@dataclass
class BenchOutput:
    """One system's answer for one recording."""

    scores: dict[str, float | None] = field(default_factory=dict)
    explanation: str = ""
    latency_s: float = float("nan")
    tokens_in: int = 0
    tokens_out: int = 0
    raw: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class System:
    """Interface: name, provenance, and a per-record call."""

    name = "system"
    kind = "specialist"          # "specialist" | "generalist"
    hosting = "local"            # "local" | "api"

    def predict(self, signal: np.ndarray, fs: int = 100) -> BenchOutput:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "kind": self.kind, "hosting": self.hosting}


class ApexSystem(System):
    """The Phase-4 detector (or a Phase-19 distilled student) plus its template report."""

    kind = "specialist"
    hosting = "local"

    def __init__(self, checkpoint: str | None = None, name: str = "APEX (CNN)",
                 device: str = "cpu", backend: str = "template"):
        self.name = name
        self.checkpoint = checkpoint
        self.device = device
        self.backend = backend
        self._members = None

    def _superclass_members(self, label_space):
        if self._members is None:
            from src.data.labels import load_scp_statements
            from src.eval.superclass import superclass_member_indices

            self._members = superclass_member_indices(label_space, load_scp_statements())
        return self._members

    def predict(self, signal: np.ndarray, fs: int = 100) -> BenchOutput:
        import torch

        from src.preprocessing.pipeline import preprocess
        from src.serving.model_cache import get_detector

        model, label_space, _ = get_detector(self.checkpoint, device=self.device)
        start = time.perf_counter()
        try:
            clean, _ = preprocess(np.asarray(signal, dtype=np.float32), fs_in=fs,
                                  fs_out=100, detect_rpeaks=False)
            with torch.no_grad():
                probs = torch.sigmoid(
                    model(torch.from_numpy(clean).unsqueeze(0).to(self.device))
                )[0].cpu().numpy()
        except Exception as e:                                # noqa: BLE001
            return BenchOutput(error=f"{type(e).__name__}: {e}",
                               latency_s=time.perf_counter() - start)

        members = self._superclass_members(label_space)
        # "at least one member present" — the same max-over-members pooling the Phase-12
        # superclass comparison uses, so this column is directly comparable to it.
        scores = {s: (float(probs[idx].max()) if idx else None)
                  for s, idx in members.items()}

        from src.generation.inference import generate_explanation
        from src.generation.templater import build_structured_input
        from src.serving.model_cache import get_scp_statements

        surfaced = [label_space[j] for j in range(len(label_space)) if probs[j] >= 0.5]
        scp = get_scp_statements()
        si = build_structured_input(
            surfaced, confidences={c: float(probs[label_space.index(c)]) for c in surfaced},
            descriptions={c: (str(scp.loc[c, "description"]) if c in scp.index else "")
                          for c in surfaced})
        explanation = generate_explanation(si, backend=self.backend)
        latency = time.perf_counter() - start
        return BenchOutput(scores=scores, explanation=explanation, latency_s=latency,
                           raw=explanation)

    def describe(self) -> dict:
        d = super().describe()
        d["checkpoint"] = self.checkpoint or "default"
        return d


class LLMSystem(System):
    """A general-purpose language model reading the measured features.

    Subclasses supply :meth:`_complete`. The prompt, parsing, and scoring are shared so the
    only difference between a local model and a hosted one is where the tokens are produced.
    """

    kind = "generalist"

    def __init__(self, name: str):
        self.name = name

    def _complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        raise NotImplementedError

    def predict(self, signal: np.ndarray, fs: int = 100) -> BenchOutput:
        from src.benchmark.features import SYSTEM_PROMPT

        start = time.perf_counter()
        features = extract(signal, fs)
        try:
            text, tin, tout = self._complete(SYSTEM_PROMPT, features.text)
        except Exception as e:                                # noqa: BLE001
            return BenchOutput(error=f"{type(e).__name__}: {e}",
                               latency_s=time.perf_counter() - start)
        latency = time.perf_counter() - start
        return BenchOutput(scores=parse_scores(text),
                           explanation=parse_interpretation(text), latency_s=latency,
                           tokens_in=tin, tokens_out=tout, raw=text)


class LocalLLMSystem(LLMSystem):
    """A small instruct model run on this machine — no data leaves the host."""

    hosting = "local"

    def __init__(self, model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 name: str | None = None, device: str | None = None,
                 max_new_tokens: int = 120):
        super().__init__(name or f"{model_id.split('/')[-1]} (local)")
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._device = device
        self._model = None
        self._tok = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = self._device or (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.float32, local_files_only=True
            ).to(self._device).eval()
        return self._model, self._tok

    def _complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        import torch

        model, tok = self._load()
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt").to(self._device)
        attn = torch.ones_like(enc)
        with torch.no_grad():
            out = model.generate(enc, attention_mask=attn,
                                 max_new_tokens=self.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][enc.shape[-1]:], skip_special_tokens=True).strip()
        return text, int(enc.shape[-1]), int(out.shape[-1] - enc.shape[-1])

    def describe(self) -> dict:
        d = super().describe()
        d["model_id"] = self.model_id
        return d


class OpenAISystem(LLMSystem):
    """GPT-4o (or another OpenAI model) over the API. Needs ``OPENAI_API_KEY``."""

    hosting = "api"

    def __init__(self, model: str = "gpt-4o", name: str | None = None):
        super().__init__(name or f"{model} (API)")
        self.model = model

    def _complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model, temperature=0,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}])
        usage = resp.usage
        return (resp.choices[0].message.content or "",
                getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))

    def describe(self) -> dict:
        d = super().describe()
        d["model"] = self.model
        return d


class AnthropicSystem(LLMSystem):
    """A Claude model over the API. Needs ``ANTHROPIC_API_KEY``."""

    hosting = "api"

    def __init__(self, model: str = "claude-fable-5", name: str | None = None,
                 max_tokens: int = 300):
        super().__init__(name or f"{model} (API)")
        self.model = model
        self.max_tokens = max_tokens

    def _complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens

    def describe(self) -> dict:
        d = super().describe()
        d["model"] = self.model
        return d


def available_systems() -> dict[str, str]:
    """Registry name -> description, for the CLI."""
    return {
        "apex": "APEX 1D-CNN detector + template report (local)",
        "apex-student": "Phase-19 distilled student, 1.0 MB (local)",
        "local-llm": "Qwen2.5-1.5B-Instruct on measured features (local)",
        "gpt-4o": "OpenAI GPT-4o on measured features (API key required)",
        "claude": "Claude on measured features (API key required)",
    }


def build_system(key: str, device: str = "cpu") -> System:
    from src.config import ROOT

    if key == "apex":
        return ApexSystem(device=device)
    if key == "apex-student":
        return ApexSystem(checkpoint=str(ROOT / "outputs" / "student_w16b1_kd_best.pt"),
                          name="APEX distilled student", device=device)
    if key == "local-llm":
        return LocalLLMSystem()
    if key == "gpt-4o":
        return OpenAISystem()
    if key == "claude":
        return AnthropicSystem()
    raise ValueError(f"unknown system {key!r}; available: {sorted(available_systems())}")
