"""Phase 24 — the reviewer feedback panel for the dashboard.

Kept out of `app.py` because it owns real state: which findings are currently on screen,
what the reviewer has said about them, and a database handle. The Gradio wiring is here;
everything it writes goes through :class:`~src.feedback.store.FeedbackStore`.

Three choices about the interaction that matter more than they look:

**Every finding is rated, or none are.** A partially completed review is worse than no
review — unrated findings are indistinguishable from findings the reviewer silently
accepted, and pooling those two into a precision estimate quietly inflates it. Submission
is blocked until every visible finding has a verdict.

**"Uncertain" is a first-class answer, not a skip button.** Reviewers will be unsure, most
often exactly at the decision boundary where the model is also unsure. Forcing a binary
choice there manufactures data. The policy drops uncertain ratings from the precision
estimate and counts them separately, so the honest answer costs the reviewer nothing.

**Missed findings are asked for explicitly.** A reviewer who is only shown what the model
said will only ever correct what the model said, and the loop learns nothing about what it
failed to say — the verification bias in
:mod:`src.feedback.policy`. The prompt for missed findings is therefore part of the form
rather than an afterthought, even though it is the question reviewers are least inclined to
answer.

Exploratory findings — surfaced deliberately below threshold — are labelled as such in the
row, because a reviewer who reads a probe as an ordinary assertion will rate it against the
wrong standard.
"""

from __future__ import annotations

import gradio as gr

from src.feedback.store import VERDICTS, FeedbackStore, RatedFinding

MAX_ROWS = 10       # findings shown for rating; more than this is rare and is noted


def _label_choices() -> list[str]:
    """SCP codes with their descriptions, for the missed-findings picker."""
    from src.generation.vocab import VOCAB

    return [f"{code} — {(e.impression or e.finding)}" for code, e in sorted(VOCAB.items())]


def _code_of(choice: str) -> str:
    return choice.split(" — ", 1)[0].strip()


class FeedbackPanel:
    """Builds the panel and wires its callbacks. One instance per demo."""

    def __init__(self, db_path=None):
        self._db_path = db_path
        self._store: FeedbackStore | None = None

    @property
    def store(self) -> FeedbackStore:
        # Opened lazily so importing the dashboard never creates a database as a
        # side effect — a Space that nobody reviews on should leave no file behind.
        if self._store is None:
            self._store = (FeedbackStore(self._db_path) if self._db_path
                           else FeedbackStore())
        return self._store

    # --- callbacks -----------------------------------------------------------
    def populate(self, report) -> list:
        """Show one row per finding in the current report; hide the rest."""
        findings = list(getattr(report, "findings", []) or [])[:MAX_ROWS]
        updates: list = []
        for i in range(MAX_ROWS):
            if i < len(findings):
                f = findings[i]
                tag = " · **exploratory** (below threshold, shown for review)" if getattr(
                    f, "exploratory", False) else ""
                updates.append(gr.update(visible=True))
                updates.append(gr.update(
                    value=f"**{f.label}** — {f.description or f.label} "
                          f"· confidence {f.confidence:.2f}{tag}"))
                updates.append(gr.update(value=None, visible=True))
            else:
                updates.append(gr.update(visible=False))
                updates.append(gr.update(value=""))
                updates.append(gr.update(value=None, visible=False))
        return updates

    def submit(self, report, reviewer_id, missed_choices, comment, *verdicts) -> str:
        """Validate and persist one review. Returns the status HTML."""
        findings = list(getattr(report, "findings", []) or [])[:MAX_ROWS]
        if not findings:
            return _status("Nothing to review — analyze a recording first.", "warn")
        if not (reviewer_id or "").strip():
            return _status("Enter a reviewer ID before submitting.", "warn")

        given = list(verdicts)[:len(findings)]
        unrated = [f.label for f, v in zip(findings, given, strict=True) if v not in VERDICTS]
        if unrated:
            return _status(
                "Rate every finding before submitting — unrated findings are "
                f"indistinguishable from accepted ones. Missing: {', '.join(unrated)}",
                "warn")

        rated = [
            RatedFinding(label=f.label, confidence=float(f.confidence), verdict=v,
                         description=f.description or "",
                         exploratory=bool(getattr(f, "exploratory", False)),
                         comment=(comment or "").strip())
            for f, v in zip(findings, given, strict=True)
        ]
        missed = [_code_of(c) for c in (missed_choices or [])]
        try:
            self.store.log_review(rated, reviewer_id=reviewer_id.strip(),
                                  review_recommended=bool(report.review_recommended),
                                  missed=missed)
        except Exception as e:                                # noqa: BLE001
            return _status(f"Could not save feedback: {e}", "error")

        s = self.store.summary()
        agree = self.store.agreement()
        extra = (f" · inter-rater agreement {agree['agreement_rate']:.0%} "
                 f"over {agree['double_rated']} double-rated findings" if agree else "")
        return _status(
            f"Saved {len(rated)} rating(s)"
            + (f" and {len(missed)} missed finding(s)" if missed else "")
            + f". Database now holds {s['ratings']} ratings across {s['reports']} reports "
              f"from {s['reviewers']} reviewer(s){extra}.", "ok")

    def stats(self) -> str:
        s = self.store.summary()
        if not s["ratings"]:
            return _status("No feedback collected yet.", "muted")
        v = s["by_verdict"]
        return _status(
            f"{s['ratings']} ratings · correct {v.get('correct', 0)}, "
            f"incorrect {v.get('incorrect', 0)}, uncertain {v.get('uncertain', 0)} · "
            f"{s['missed']} missed findings reported · {s['exploratory']} exploratory "
            f"· {s['reports']} reports · {s['reviewers']} reviewer(s)", "muted")


def _status(text: str, kind: str = "ok") -> str:
    colors = {"ok": ("#0b6b3a", "#e6f4ea"), "warn": ("#8a6d00", "#fef7e0"),
              "error": ("#b3261e", "#fce8e6"), "muted": ("#555", "#f3f3f3")}
    fg, bg = colors.get(kind, colors["muted"])
    return (f'<div style="padding:.6rem .8rem;border-radius:6px;background:{bg};'
            f'color:{fg};font-size:.9rem">{text}</div>')


def build_panel(panel: FeedbackPanel, report_state):
    """Render the panel. Returns ``(rows, verdicts, populate_outputs, submit_button, ...)``."""
    rows, labels, verdicts = [], [], []
    with gr.Accordion("Review this report (clinician feedback)", open=False):
        gr.Markdown(
            "Rate each finding APEX reported, then tell it what it **missed**. "
            "Feedback is stored locally and used to re-tune per-label decision thresholds "
            "— see `docs/feedback/report.md`. "
            "_Rating only what the model said would teach it to be more cautious and "
            "nothing else, which is why the missed-findings box matters._"
        )
        reviewer = gr.Textbox(label="Reviewer ID", placeholder="e.g. initials or staff ID",
                              max_lines=1)
        for _ in range(MAX_ROWS):
            with gr.Row(visible=False) as row:
                label = gr.Markdown("")
                verdict = gr.Radio(choices=list(VERDICTS), label="Verdict", visible=False,
                                   interactive=True, scale=0)
            rows.append(row)
            labels.append(label)
            verdicts.append(verdict)

        missed = gr.Dropdown(choices=_label_choices(), multiselect=True, value=[],
                             label="Findings APEX missed (false negatives)",
                             info="The only way the loop can learn to be less conservative.")
        comment = gr.Textbox(label="Comment (optional)", lines=2)
        submit = gr.Button("Submit review", variant="primary")
        status = gr.HTML(panel.stats())

    populate_outputs: list = []
    for row, label, verdict in zip(rows, labels, verdicts, strict=True):
        populate_outputs += [row, label, verdict]

    submit.click(panel.submit,
                 inputs=[report_state, reviewer, missed, comment, *verdicts],
                 outputs=status)
    return populate_outputs, status
