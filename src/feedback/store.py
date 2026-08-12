"""Phase 24 — the feedback database.

A local SQLite file recording what reviewers said about what APEX claimed. Plain
``sqlite3`` from the standard library: the store has to survive being copied between
machines, opened by hand, and read by someone who has never seen this project, and a file
you can point ``sqlite3`` at satisfies all three better than any ORM would.

The schema is shaped by what turns out to be needed *later*, when the feedback is used to
move a decision threshold. Three things must be recorded at collection time or the data is
worthless for that purpose:

**The confidence at the moment of review.** A rating of "incorrect" means nothing on its
own. "Incorrect, at confidence 0.55" and "incorrect, at confidence 0.98" are entirely
different pieces of evidence — the first says the threshold is too low, the second says the
model is broken. Every rating is therefore stored against the confidence that produced it.

**The threshold in force at the time.** Feedback is collected under a policy, and that
policy determines *which* findings the reviewer could see at all. Re-analysing yesterday's
ratings without knowing yesterday's threshold silently mixes incompatible samples. The
threshold is snapshotted onto the report row.

**Whether the finding was shown deliberately below threshold.** See
:mod:`src.feedback.policy` — a feedback loop that only ever sees supra-threshold findings
can only learn to raise thresholds, never lower them. Breaking that requires occasionally
surfacing a sub-threshold finding on purpose, and those rows must be distinguishable from
ordinary ones or they will bias every precision estimate computed from the table.

**Missed findings are a separate table, not a rating.** A false negative is not a verdict
on something APEX said; it is a report of something APEX failed to say, and it is the only
route by which the loop can learn about recall. Squeezing it into ``ratings`` would mean
inventing a finding row for something the model never produced.

The database is deliberately *not* committed — feedback is clinical opinion attached to a
recording, and it belongs wherever the deployment's data governance says it belongs, not in
a public git repository.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.config import ROOT

DEFAULT_DB = ROOT / "outputs" / "feedback.db"

# The three verdicts a reviewer can give a surfaced finding.
VERDICT_CORRECT = "correct"
VERDICT_INCORRECT = "incorrect"
VERDICT_UNCERTAIN = "uncertain"
VERDICTS = (VERDICT_CORRECT, VERDICT_INCORRECT, VERDICT_UNCERTAIN)

SCHEMA_VERSION = 1

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    reviewer_id   TEXT,
    model_version TEXT,
    app_version   TEXT,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     INTEGER REFERENCES sessions(id),
    created_at     TEXT NOT NULL,
    record_ref     TEXT,
    -- the per-label thresholds in force when this report was produced, as JSON. Without
    -- it, ratings collected under different policies cannot be pooled correctly.
    thresholds     TEXT,
    review_recommended INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS findings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    description TEXT,
    confidence  REAL NOT NULL,
    threshold   REAL,           -- the threshold this label faced
    exploratory INTEGER NOT NULL DEFAULT 0,  -- shown deliberately below threshold
    -- Probability this finding was shown at all: 1.0 for supra-threshold, epsilon for an
    -- exploratory probe. Sub-threshold findings are sampled far more rarely than
    -- supra-threshold ones, so pooling them unweighted would badly under-represent the
    -- region exploration exists to measure. Stored per row because the rate is per-label
    -- and drifts as feedback accumulates.
    sampling_rate REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS ratings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id  INTEGER NOT NULL REFERENCES findings(id) ON DELETE CASCADE,
    verdict     TEXT NOT NULL CHECK (verdict IN ('correct','incorrect','uncertain')),
    reviewer_id TEXT,
    created_at  TEXT NOT NULL,
    comment     TEXT
);

-- False negatives: what the reviewer says APEX should have reported and did not. The only
-- channel through which the loop can learn anything about recall.
CREATE TABLE IF NOT EXISTS missed (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id   INTEGER NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    label       TEXT NOT NULL,
    confidence  REAL,           -- what the model actually scored it, if known
    reviewer_id TEXT,
    created_at  TEXT NOT NULL,
    comment     TEXT
);

CREATE INDEX IF NOT EXISTS idx_findings_label ON findings(label);
CREATE INDEX IF NOT EXISTS idx_ratings_finding ON ratings(finding_id);
CREATE INDEX IF NOT EXISTS idx_missed_label ON missed(label);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RatedFinding:
    """One finding as it was shown, with the verdict it received."""

    label: str
    confidence: float
    verdict: str
    threshold: float | None = None
    exploratory: bool = False
    sampling_rate: float = 1.0
    description: str = ""
    comment: str = ""


@dataclass
class LabelCounts:
    """Rating tallies for one label, used by :mod:`src.feedback.policy`."""

    label: str
    correct: int = 0
    incorrect: int = 0
    uncertain: int = 0
    missed: int = 0
    # (confidence, verdict, inverse sampling weight) per rating.
    confidences: list[tuple[float, str, float]] = field(default_factory=list)

    @property
    def rated(self) -> int:
        """Ratings that carry information — `uncertain` deliberately excluded.

        An uncertain verdict is not half a correct one. It says the reviewer could not
        tell, which is evidence about the *case*, not about the model, and averaging it in
        either direction would invent an opinion nobody expressed. It is counted and
        reported, and kept out of the precision estimate.
        """
        return self.correct + self.incorrect

    @property
    def observed_precision(self) -> float | None:
        return (self.correct / self.rated) if self.rated else None


class FeedbackStore:
    """Read/write access to the feedback database.

    Usable as a context manager. Safe to open concurrently for reads (WAL mode); writes are
    short and serialized by SQLite itself, which is ample for a review UI.
    """

    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                           (str(SCHEMA_VERSION),))
        self._conn.commit()

    # --- plumbing ------------------------------------------------------------
    def __enter__(self) -> FeedbackStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # --- writing -------------------------------------------------------------
    def start_session(self, reviewer_id: str | None = None, model_version: str | None = None,
                      app_version: str | None = None, notes: str | None = None) -> int:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO sessions(started_at, reviewer_id, model_version, app_version, notes)"
                " VALUES (?,?,?,?,?)",
                (_now(), reviewer_id, model_version, app_version, notes))
            return int(cur.lastrowid)

    def log_review(self, findings: Iterable[RatedFinding], reviewer_id: str | None = None,
                   record_ref: str | None = None, session_id: int | None = None,
                   thresholds: dict[str, float] | None = None,
                   review_recommended: bool = False,
                   missed: Iterable[str] | None = None,
                   missed_confidences: dict[str, float] | None = None) -> int:
        """Record one reviewed report. Returns the report id.

        ``missed`` are labels the reviewer says should have been reported. Supplying
        ``missed_confidences`` (what the model actually scored them) makes those rows far
        more useful — it is the difference between "we missed it" and "we missed it, and we
        had it at 0.41", and only the second tells you where to move a threshold.
        """
        now = _now()
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO reports(session_id, created_at, record_ref, thresholds,"
                " review_recommended) VALUES (?,?,?,?,?)",
                (session_id, now, record_ref,
                 json.dumps(thresholds) if thresholds else None, int(review_recommended)))
            report_id = int(cur.lastrowid)

            for f in findings:
                if f.verdict not in VERDICTS:
                    raise ValueError(f"unknown verdict {f.verdict!r}; expected one of {VERDICTS}")
                cur.execute(
                    "INSERT INTO findings(report_id, label, description, confidence,"
                    " threshold, exploratory, sampling_rate) VALUES (?,?,?,?,?,?,?)",
                    (report_id, f.label, f.description, float(f.confidence),
                     f.threshold, int(f.exploratory), float(f.sampling_rate)))
                cur.execute(
                    "INSERT INTO ratings(finding_id, verdict, reviewer_id, created_at, comment)"
                    " VALUES (?,?,?,?,?)",
                    (int(cur.lastrowid), f.verdict, reviewer_id, now, f.comment or None))

            for label in missed or ():
                cur.execute(
                    "INSERT INTO missed(report_id, label, confidence, reviewer_id, created_at,"
                    " comment) VALUES (?,?,?,?,?,?)",
                    (report_id, label, (missed_confidences or {}).get(label),
                     reviewer_id, now, None))
        return report_id

    # --- reading -------------------------------------------------------------
    def counts_by_label(self, min_confidence: float = 0.0,
                        include_exploratory: bool = True) -> dict[str, LabelCounts]:
        """Rating tallies per label, with the confidence of each rating retained."""
        out: dict[str, LabelCounts] = {}
        sql = ("SELECT f.label, f.confidence, f.exploratory, f.sampling_rate, r.verdict"
               " FROM findings f JOIN ratings r ON r.finding_id = f.id"
               " WHERE f.confidence >= ?")
        if not include_exploratory:
            sql += " AND f.exploratory = 0"
        for row in self._conn.execute(sql, (min_confidence,)):
            counts = out.setdefault(row["label"], LabelCounts(row["label"]))
            rate = float(row["sampling_rate"] or 1.0)
            counts.confidences.append((float(row["confidence"]), row["verdict"],
                                       1.0 / max(rate, 1e-6)))
            setattr(counts, row["verdict"], getattr(counts, row["verdict"]) + 1)

        for row in self._conn.execute("SELECT label, COUNT(*) n FROM missed GROUP BY label"):
            counts = out.setdefault(row["label"], LabelCounts(row["label"]))
            counts.missed = int(row["n"])
        return out

    def missed_confidences(self, label: str) -> list[float]:
        """Model confidences for findings a reviewer said were missed — where to look."""
        return [float(r["confidence"]) for r in self._conn.execute(
            "SELECT confidence FROM missed WHERE label = ? AND confidence IS NOT NULL",
            (label,))]

    def summary(self) -> dict:
        """Headline counts, for the UI footer and the report."""
        one = lambda q: int(self._conn.execute(q).fetchone()[0])  # noqa: E731
        by_verdict = {r["verdict"]: int(r["n"]) for r in self._conn.execute(
            "SELECT verdict, COUNT(*) n FROM ratings GROUP BY verdict")}
        return {
            "reports": one("SELECT COUNT(*) FROM reports"),
            "findings": one("SELECT COUNT(*) FROM findings"),
            "ratings": one("SELECT COUNT(*) FROM ratings"),
            "missed": one("SELECT COUNT(*) FROM missed"),
            "exploratory": one("SELECT COUNT(*) FROM findings WHERE exploratory = 1"),
            "reviewers": one("SELECT COUNT(DISTINCT reviewer_id) FROM ratings"),
            "labels": one("SELECT COUNT(DISTINCT label) FROM findings"),
            "by_verdict": by_verdict,
        }

    def agreement(self) -> dict:
        """Where two reviewers rated the same finding, how often did they agree?

        Returns ``{}`` until some finding has been rated twice. Worth watching before
        trusting any threshold move: if reviewers disagree with each other as often as they
        disagree with the model, the feedback is measuring the reviewers.
        """
        rows = list(self._conn.execute(
            "SELECT finding_id, COUNT(*) n, COUNT(DISTINCT verdict) d"
            " FROM ratings GROUP BY finding_id HAVING n > 1"))
        if not rows:
            return {}
        agreed = sum(1 for r in rows if r["d"] == 1)
        return {"double_rated": len(rows), "agreed": agreed,
                "agreement_rate": round(agreed / len(rows), 4)}
