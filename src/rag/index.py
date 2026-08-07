"""Phase 21 — the vector index the RAG layer retrieves from.

A dense index over the reference corpus: passages are embedded once with a small
sentence-embedding model, L2-normalized, and stored as a single matrix, so retrieval is
one matrix-vector product and a top-k. At 894 passages an exact search is
microseconds — an approximate-nearest-neighbour library (FAISS, hnswlib) would add a
dependency and an accuracy caveat to solve a problem this corpus does not have.

**Two retrievers, because clinical text needs both.** Dense embeddings capture that
"irregularly irregular rhythm" and "atrial fibrillation" are the same idea. They are also
prone to missing rare literal tokens — and the queries here are full of those: SCP codes
(``ASMI``, ``LNGQT``), lead names (``aVF``, ``V1``). A sparse TF-IDF index matches those
exactly. :class:`HybridIndex` runs both and fuses the rankings, which is what makes a
query like "ASMI anteroseptal myocardial infarction" retrieve both the code definition and
the clinical description.

The embedding model (``all-MiniLM-L6-v2``, 384-d) is loaded through plain ``transformers``
with mean pooling rather than adding ``sentence-transformers`` as a dependency — it is the
same computation, and this repo already has ``transformers`` and ``torch``. If the model
cannot be loaded (offline, no cache), :class:`DenseIndex` raises and the caller can fall
back to :class:`TfidfIndex`, which needs nothing beyond scikit-learn.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import ROOT
from src.rag.corpus import Passage, load_corpus

INDEX_DIR = ROOT / "data" / "reference"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class Hit:
    """One retrieved passage plus the score that retrieved it."""

    passage: Passage
    score: float
    rank: int

    def cite(self) -> str:
        return f"{self.passage.title} ({self.passage.source})"


# --- dense ------------------------------------------------------------------
class DenseIndex:
    """Mean-pooled MiniLM embeddings + exact cosine search."""

    def __init__(self, model_name: str = EMBED_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._tok = None
        self._model = None
        self.passages: list[Passage] = []
        self.matrix: np.ndarray | None = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        return self._tok, self._model

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Mean-pooled, L2-normalized embeddings. Normalizing makes the dot product the
        cosine similarity, so search is a single matmul."""
        import torch

        tok, model = self._load_model()
        out = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            # Mean-pool over real tokens only — padding must not drag the vector toward zero.
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)

    def build(self, passages: list[Passage]) -> DenseIndex:
        self.passages = passages
        self.matrix = self.embed([p.text for p in passages])
        return self

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if self.matrix is None:
            raise RuntimeError("build() or load() first")
        q = self.embed([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self.passages[i], float(scores[i]), r) for r, i in enumerate(top)]

    def save(self, path: Path = INDEX_DIR / "dense_index.npz") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, matrix=self.matrix,
                            model=np.array(self.model_name),
                            ids=np.array([p.id for p in self.passages]))
        return path

    def load(self, passages: list[Passage],
             path: Path = INDEX_DIR / "dense_index.npz") -> DenseIndex:
        d = np.load(path, allow_pickle=False)
        ids = [str(x) for x in d["ids"]]
        by_id = {p.id: p for p in passages}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(
                f"index references {len(missing)} passages absent from the corpus "
                f"(first: {missing[0]!r}); rebuild the index")
        self.passages = [by_id[i] for i in ids]
        self.matrix = d["matrix"]
        self.model_name = str(d["model"])
        return self


# --- sparse -----------------------------------------------------------------
class TfidfIndex:
    """Word + character n-gram TF-IDF, cosine similarity.

    Character n-grams matter here: they let ``LNGQT`` partially match ``long QT`` and make
    the retriever robust to the abbreviation-heavy register of ECG text, which a
    word-level index alone handles badly.
    """

    def __init__(self):
        self.passages: list[Passage] = []
        self._word = None
        self._char = None
        self._wm = None
        self._cm = None

    def build(self, passages: list[Passage]) -> TfidfIndex:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.passages = passages
        texts = [f"{p.title}. {p.text}" for p in passages]
        self._word = TfidfVectorizer(sublinear_tf=True, stop_words="english",
                                     ngram_range=(1, 2), min_df=1)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     sublinear_tf=True, min_df=2)
        self._wm = self._word.fit_transform(texts)
        self._cm = self._char.fit_transform(texts)
        return self

    def search(self, query: str, k: int = 4) -> list[Hit]:
        from sklearn.preprocessing import normalize

        if self._wm is None:
            raise RuntimeError("build() first")
        qw = normalize(self._word.transform([query]))
        qc = normalize(self._char.transform([query]))
        scores = (normalize(self._wm) @ qw.T).toarray().ravel() * 0.7 \
            + (normalize(self._cm) @ qc.T).toarray().ravel() * 0.3
        top = np.argsort(-scores)[:k]
        return [Hit(self.passages[i], float(scores[i]), r) for r, i in enumerate(top)]

    def save(self, path: Path = INDEX_DIR / "tfidf_index.pkl") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"word": self._word, "char": self._char,
                         "wm": self._wm, "cm": self._cm,
                         "ids": [p.id for p in self.passages]}, f)
        return path

    def load(self, passages: list[Passage],
             path: Path = INDEX_DIR / "tfidf_index.pkl") -> TfidfIndex:
        with path.open("rb") as f:
            d = pickle.load(f)  # noqa: S301 - our own artifact, not untrusted input
        by_id = {p.id: p for p in passages}
        self.passages = [by_id[i] for i in d["ids"]]
        self._word, self._char, self._wm, self._cm = d["word"], d["char"], d["wm"], d["cm"]
        return self


# --- fusion -----------------------------------------------------------------
class HybridIndex:
    """Dense + sparse, fused by reciprocal rank.

    Reciprocal-rank fusion rather than a weighted score sum, because the two retrievers'
    scores are not on a comparable scale (cosine over unit embeddings vs TF-IDF cosine)
    and normalizing them against each other would be an arbitrary choice tuned on nothing.
    RRF only uses each retriever's *ordering*, which is the part both agree on the meaning
    of.
    """

    def __init__(self, dense: DenseIndex | None, sparse: TfidfIndex | None, k_rrf: int = 60):
        if dense is None and sparse is None:
            raise ValueError("HybridIndex needs at least one sub-index")
        self.dense, self.sparse, self.k_rrf = dense, sparse, k_rrf
        self.passages = (dense or sparse).passages

    def search(self, query: str, k: int = 4, pool: int = 20) -> list[Hit]:
        ranked: dict[str, float] = {}
        by_id: dict[str, Passage] = {}
        for idx in (self.dense, self.sparse):
            if idx is None:
                continue
            for hit in idx.search(query, k=pool):
                pid = hit.passage.id
                by_id[pid] = hit.passage
                ranked[pid] = ranked.get(pid, 0.0) + 1.0 / (self.k_rrf + hit.rank + 1)
        order = sorted(ranked, key=lambda pid: -ranked[pid])[:k]
        return [Hit(by_id[pid], ranked[pid], r) for r, pid in enumerate(order)]


# --- build / load helpers ---------------------------------------------------
def build_indexes(passages: list[Passage] | None = None, dense: bool = True,
                  verbose: bool = True) -> tuple[DenseIndex | None, TfidfIndex]:
    passages = load_corpus() if passages is None else passages
    if verbose:
        print(f"indexing {len(passages)} passages...")
    sparse = TfidfIndex().build(passages)
    sparse.save()
    d = None
    if dense:
        try:
            d = DenseIndex().build(passages)
            d.save()
        except Exception as e:
            print(f"  !! dense index unavailable ({type(e).__name__}: {e}); "
                  "falling back to TF-IDF only")
            d = None
    meta = {"n_passages": len(passages), "dense": d is not None,
            "embed_model": EMBED_MODEL if d is not None else None}
    (INDEX_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"  dense={'yes' if d is not None else 'no'}  sparse=yes -> {INDEX_DIR}")
    return d, sparse


def load_index(prefer_dense: bool = True) -> HybridIndex:
    """Load the built indexes into a hybrid retriever, degrading gracefully."""
    passages = load_corpus()
    sparse = TfidfIndex().load(passages)
    dense = None
    if prefer_dense and (INDEX_DIR / "dense_index.npz").exists():
        try:
            dense = DenseIndex().load(passages)
        except Exception:
            dense = None
    return HybridIndex(dense, sparse)
