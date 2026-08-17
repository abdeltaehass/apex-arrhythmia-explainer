"""Phase 28 — benchmarking the specialist against general-purpose models.

Runs every system over the same recordings through one interface and scores them on the
axes a deployment decision actually turns on: discrimination, self-consistency, latency,
and cost.

    from src.benchmark import build_system, summarize

    system = build_system("apex")
    outputs = [system.predict(sig) for sig in signals]
    summarize(system, outputs, y_true)

Parts:

- :mod:`~src.benchmark.features`     the measurement-mediated protocol (a text model cannot
                                     read a waveform, so the protocol must be stated)
- :mod:`~src.benchmark.parse_scores` recovering numbers from a model's prose
- :mod:`~src.benchmark.systems`      APEX, the distilled student, local and hosted LLMs
- :mod:`~src.benchmark.metrics`      AUROC with abstention accounting, self-contradiction,
                                     latency, and a dated cost model
"""

from src.benchmark.features import SUPERCLASSES, build_prompt, describe, extract
from src.benchmark.metrics import (
    API_PRICING_PER_MTOK,
    LOCAL_HOURLY_USD,
    PRICING_AS_OF,
    SystemScores,
    cost_per_1k,
    coverage,
    latency_percentiles,
    per_superclass_auroc,
    self_contradiction_rate,
    summarize,
)
from src.benchmark.parse_scores import compliance, parse_interpretation, parse_scores
from src.benchmark.systems import (
    AnthropicSystem,
    ApexSystem,
    BenchOutput,
    LocalLLMSystem,
    OpenAISystem,
    System,
    available_systems,
    build_system,
)

__all__ = [
    "API_PRICING_PER_MTOK", "LOCAL_HOURLY_USD", "PRICING_AS_OF", "SUPERCLASSES",
    "AnthropicSystem", "ApexSystem", "BenchOutput", "LocalLLMSystem", "OpenAISystem",
    "System", "SystemScores", "available_systems", "build_prompt", "build_system",
    "compliance", "coverage", "cost_per_1k", "describe", "extract", "latency_percentiles",
    "parse_interpretation", "parse_scores", "per_superclass_auroc",
    "self_contradiction_rate", "summarize",
]
