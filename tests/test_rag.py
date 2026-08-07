"""Unit tests for the Phase 21 RAG layer (no network, no model download, no dataset)."""

import json

import numpy as np
import pytest

from src.generation.templater import build_structured_input
from src.rag.corpus import Passage, chunk_text, load_corpus, split_sections
from src.rag.index import Hit, HybridIndex, TfidfIndex
from src.rag.retrieve import (
    CONTEXT_INSTRUCTION,
    RetrievedContext,
    build_query,
    context_condition_names,
    format_context,
    retrieve_for_findings,
)


def _p(pid: str, text: str, title: str = "T", source: str = "S") -> Passage:
    return Passage(id=pid, text=text, title=title, source=source, license="CC BY 4.0",
                   url="https://example.org", retrieved="2026-08-07", codes=[])


@pytest.fixture
def tiny_corpus():
    return [
        _p("scp::AFIB", "AFIB denotes atrial fibrillation, an irregularly irregular rhythm."),
        _p("scp::IMI", "IMI denotes inferior myocardial infarction affecting leads II III aVF."),
        _p("scp::CLBBB", "CLBBB denotes complete left bundle branch block with a wide QRS."),
        _p("wiki::lvh", "Left ventricular hypertrophy increases R wave amplitude in aVL."),
    ]


# --- chunking ----------------------------------------------------------------
def test_split_sections_drops_boilerplate():
    extract = "Intro body.\n\n== Causes ==\nSome causes.\n\n== References ==\n[1] a paper."
    got = dict(split_sections(extract))
    assert "Summary" in got and "Causes" in got
    assert "References" not in got  # boilerplate is pure noise in a retrieval index


def test_chunk_text_respects_max_and_keeps_sentences_whole():
    para = " ".join(f"Sentence number {i} about the ECG." for i in range(60))
    chunks = chunk_text(para, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)          # some slack for the final sentence
    assert all(c.strip().endswith(".") for c in chunks)


def test_chunk_text_merges_runt_fragments():
    chunks = chunk_text("A full paragraph of reasonable length here.\n\nShort.", min_chars=100)
    assert len(chunks) == 1  # the 6-char fragment is merged, not emitted alone


def test_chunk_text_empty_input():
    assert chunk_text("   \n\n  ") == []


# --- sparse index ------------------------------------------------------------
def test_tfidf_retrieves_the_matching_passage(tiny_corpus):
    idx = TfidfIndex().build(tiny_corpus)
    top = idx.search("AFIB atrial fibrillation", k=1)[0]
    assert top.passage.id == "scp::AFIB"


def test_tfidf_ranks_and_scores_are_ordered(tiny_corpus):
    hits = TfidfIndex().build(tiny_corpus).search("bundle branch block", k=3)
    assert [h.rank for h in hits] == [0, 1, 2]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_search_before_build_raises(tiny_corpus):
    with pytest.raises(RuntimeError):
        TfidfIndex().search("anything")


def test_tfidf_roundtrips_through_disk(tiny_corpus, tmp_path):
    idx = TfidfIndex().build(tiny_corpus)
    path = idx.save(tmp_path / "t.pkl")
    reloaded = TfidfIndex().load(tiny_corpus, path)
    assert [h.passage.id for h in reloaded.search("inferior infarction", k=2)] == \
           [h.passage.id for h in idx.search("inferior infarction", k=2)]


# --- fusion ------------------------------------------------------------------
class _FakeIndex:
    """Returns a fixed ranking, so fusion can be tested without any model."""

    def __init__(self, passages):
        self.passages = passages

    def search(self, query, k=4):
        return [Hit(p, 1.0 - i * 0.1, i) for i, p in enumerate(self.passages[:k])]


def test_hybrid_fuses_both_rankings(tiny_corpus):
    a = _FakeIndex([tiny_corpus[0], tiny_corpus[1]])
    b = _FakeIndex([tiny_corpus[1], tiny_corpus[0]])
    ids = [h.passage.id for h in HybridIndex(a, b).search("q", k=2)]
    assert set(ids) == {"scp::AFIB", "scp::IMI"}


def test_hybrid_rewards_agreement(tiny_corpus):
    """A passage both retrievers rank first must outrank one only a single retriever likes."""
    a = _FakeIndex([tiny_corpus[0], tiny_corpus[2]])
    b = _FakeIndex([tiny_corpus[0], tiny_corpus[3]])
    assert HybridIndex(a, b).search("q", k=1)[0].passage.id == "scp::AFIB"


def test_hybrid_needs_at_least_one_subindex():
    with pytest.raises(ValueError):
        HybridIndex(None, None)


def test_hybrid_works_with_dense_missing(tiny_corpus):
    """Offline fallback: no embedding model available, sparse alone must still serve."""
    sparse = TfidfIndex().build(tiny_corpus)
    assert HybridIndex(None, sparse).search("atrial fibrillation", k=1)[0].passage.id == "scp::AFIB"


# --- query building / retrieval ---------------------------------------------
def test_build_query_includes_code_description_and_leads():
    q = build_query("IMI", "inferior myocardial infarction", ["II", "III", "aVF"])
    assert "IMI" in q and "inferior myocardial infarction" in q and "aVF" in q


def test_build_query_bare_code():
    assert build_query("AFIB") == "AFIB"


def test_retrieve_covers_every_finding_not_just_the_first(tiny_corpus):
    """One query per finding: a merged query would embed to the average of two unrelated
    conditions and reliably return neither."""
    si = build_structured_input(
        ["AFIB", "CLBBB"],
        descriptions={"AFIB": "atrial fibrillation",
                      "CLBBB": "complete left bundle branch block"},
    )
    ctx = retrieve_for_findings(si, TfidfIndex().build(tiny_corpus),
                                k_per_finding=1, max_passages=4)
    assert {"scp::AFIB", "scp::CLBBB"} <= set(ctx.passage_ids)
    assert len(ctx.queries) == 2


def test_retrieve_dedupes_and_caps(tiny_corpus):
    si = build_structured_input(["AFIB", "IMI", "CLBBB"])
    ctx = retrieve_for_findings(si, TfidfIndex().build(tiny_corpus),
                                k_per_finding=3, max_passages=2)
    assert len(ctx.hits) == 2
    assert len(set(ctx.passage_ids)) == 2       # no passage injected twice
    assert [h.rank for h in ctx.hits] == [0, 1]  # ranks renumbered after the cap


def test_retrieve_with_no_findings_is_empty(tiny_corpus):
    ctx = retrieve_for_findings(build_structured_input([]), TfidfIndex().build(tiny_corpus))
    assert ctx.hits == []
    assert format_context(ctx) == ""


# --- prompt block ------------------------------------------------------------
def test_format_context_numbers_cites_and_restates_the_boundary(tiny_corpus):
    ctx = RetrievedContext(hits=[Hit(tiny_corpus[0], 0.9, 0)], queries=["q"])
    block = format_context(ctx)
    assert "[1]" in block
    assert tiny_corpus[0].source in block
    # the anti-leak instruction must travel with the context, not live only in the system prompt
    assert CONTEXT_INSTRUCTION in block


def test_format_context_truncates_long_passages():
    long = RetrievedContext(hits=[Hit(_p("x", "word " * 500), 0.5, 0)], queries=["q"])
    assert "..." in format_context(long, max_chars_per_passage=100)


def test_context_condition_names_finds_only_named_conditions(tiny_corpus):
    ctx = RetrievedContext(hits=[Hit(tiny_corpus[0], 0.9, 0)], queries=["q"])
    terms = {"AFIB": "atrial fibrillation", "CLBBB": "left bundle branch block"}
    assert context_condition_names(ctx, terms) == {"AFIB"}


# --- prompt integration ------------------------------------------------------
def test_empty_context_reproduces_the_phase6_prompt_exactly():
    """The no-RAG arm of the comparison must be byte-identical to the original prompt,
    or the experiment is not controlled."""
    from src.generation.prompts import build_user_prompt, serialize_structured_input

    si = build_structured_input(["AFIB"], descriptions={"AFIB": "atrial fibrillation"})
    assert build_user_prompt(si) == serialize_structured_input(si) + "\n\nWrite the report now."
    assert build_user_prompt(si, "") == build_user_prompt(si)


def test_context_is_inserted_before_the_write_instruction():
    from src.generation.prompts import build_user_prompt

    si = build_structured_input(["AFIB"])
    prompt = build_user_prompt(si, "REFERENCE BLOCK")
    assert "REFERENCE BLOCK" in prompt
    assert prompt.index("REFERENCE BLOCK") < prompt.index("Write the report now.")


def test_template_backend_ignores_context():
    """The deterministic renderer is consistent by construction; context must not reach it."""
    from src.generation.inference import generate_explanation

    si = build_structured_input(["AFIB"], descriptions={"AFIB": "atrial fibrillation"})
    assert generate_explanation(si, backend="template", context="anything at all") == \
           generate_explanation(si, backend="template")


# --- corpus provenance -------------------------------------------------------
def test_passage_json_roundtrip():
    p = _p("a::b", "text")
    assert Passage(**json.loads(p.to_json())) == p


@pytest.mark.parametrize("field", ["source", "license", "url", "retrieved"])
def test_built_corpus_carries_provenance(field):
    """Every passage must be auditable — this is clinical reference text."""
    try:
        passages = load_corpus()
    except FileNotFoundError:
        pytest.skip("corpus not built in this environment")
    assert all(getattr(p, field) for p in passages)


def test_built_corpus_licenses_are_known_open_ones():
    try:
        passages = load_corpus()
    except FileNotFoundError:
        pytest.skip("corpus not built in this environment")
    assert {p.license for p in passages} <= {"CC BY 4.0", "CC BY-SA 4.0", "Public Domain"}


def test_dense_index_search_before_build_raises():
    from src.rag.index import DenseIndex

    with pytest.raises(RuntimeError):
        DenseIndex().search("q")


def test_dense_load_rejects_a_corpus_mismatch(tmp_path, tiny_corpus):
    """A stale index silently paired with a rebuilt corpus would retrieve wrong passages."""
    from src.rag.index import DenseIndex

    idx = DenseIndex()
    idx.passages = tiny_corpus
    idx.matrix = np.zeros((len(tiny_corpus), 8), dtype=np.float32)
    path = idx.save(tmp_path / "d.npz")
    with pytest.raises(ValueError, match="rebuild the index"):
        DenseIndex().load(tiny_corpus[:2], path)


# --- serving integration -----------------------------------------------------
def test_missing_index_degrades_instead_of_failing(monkeypatch):
    """A deployment without a built index must still analyze ECGs — retrieval is an
    enhancement, not a prerequisite."""
    from src.serving import model_cache

    model_cache.clear_caches()
    monkeypatch.setattr("src.rag.load_index",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no corpus")))
    assert model_cache.get_rag_index() is None
    assert model_cache.get_rag_index() is None  # cached failure, not retried every call
    model_cache.clear_caches()


def test_analyze_signal_exposes_with_rag_flag():
    import inspect

    from src.serving.serializer import analyze_signal

    assert "with_rag" in inspect.signature(analyze_signal).parameters
    assert inspect.signature(analyze_signal).parameters["with_rag"].default is False
