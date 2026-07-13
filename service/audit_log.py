"""Audit-log serialization for the frontend.

Turns the immutable ``VariantResult`` / ``Attempt`` dataclasses into plain JSON-safe
dicts. All serialization lives here (not on the models) so the data classes stay pure
and the UI-facing shape has one owner. Every attempt — LLM draft, error, or fallback —
is included so the frontend can render the expandable per-card attempt trail, and each
variant carries a human ``status_label`` flag ("AI-approved" vs "fallback-truncated").
"""

from __future__ import annotations

import json
from typing import Any

from service.critique_loop import Attempt, VariantResult

# Maps the internal status enum to the human label the UI displays on each result card.
STATUS_LABELS: dict[str, str] = {
    "ai_approved": "AI-approved",
    "fallback_truncated": "fallback-truncated",
}

# Maps the lexical (word-overlap) distinctness verdict to its UI badge label. ``None`` ("not
# assessed") maps to no label, so the frontend simply omits the badge in that case.
DISTINCT_LABELS: dict[bool, str] = {
    True: "Distinct",
    False: "Too similar",
}

# Maps the semantic judge verdict to its UI badge label. Worded differently from the lexical
# labels above so the two badges read as distinct signals (shared angle vs shared words), not
# duplicates. ``None`` = judge unavailable -> no badge, degrading to the lexical signal alone.
JUDGE_LABELS: dict[bool, str] = {
    True: "Angles distinct",
    False: "Same angle",
}


def _signals_agree(distinct: bool | None, judge_distinct: bool | None) -> bool | None:
    """Whether the two independent signals concur. ``None`` when either did not run.

    Surfaced so the UI can visibly flag a disagreement (e.g. lexical says distinct but the
    judge caught a paraphrase collision) rather than silently resolving it.
    """
    if distinct is None or judge_distinct is None:
        return None
    return distinct == judge_distinct


def attempt_to_dict(attempt: Attempt) -> dict[str, Any]:
    """Serialize one attempt row for the expandable audit trail."""
    return {
        "attempt_number": attempt.attempt_number,
        "source": attempt.source,
        "text": attempt.text,
        "char_count": attempt.char_count,
        "max_chars": attempt.max_chars,
        "passed": attempt.passed,
        "note": attempt.note,
    }


def variant_to_dict(result: VariantResult) -> dict[str, Any]:
    """Serialize one platform's result: the chosen text, the UI flags, and every attempt.

    Alongside the character-limit ``status``/``status_label`` pair, this emits the two
    independent distinctness signals as parallel trios — lexical ``distinct``/
    ``distinct_label``/``distinct_note`` (shared words) and semantic ``judge_distinct``/
    ``judge_label``/``judge_note`` (shared meaning) — plus ``signals_agree`` so the UI can
    render both badges and visibly flag when they disagree.
    """
    distinct = result.distinct
    judge_distinct = result.judge_distinct
    return {
        "platform": result.rule.name,
        "label": result.rule.label,
        "max_chars": result.rule.max_chars,
        "final_text": result.final_text,
        "final_char_count": result.final_char_count,
        "status": result.status,
        "status_label": STATUS_LABELS.get(result.status, result.status),
        "distinct": distinct,
        "distinct_label": DISTINCT_LABELS.get(distinct) if distinct is not None else None,
        "distinct_note": result.distinct_note,
        "judge_distinct": judge_distinct,
        "judge_label": JUDGE_LABELS.get(judge_distinct) if judge_distinct is not None else None,
        "judge_note": result.judge_note,
        "signals_agree": _signals_agree(distinct, judge_distinct),
        "attempts": [attempt_to_dict(attempt) for attempt in result.attempts],
    }


def build_audit_log(results: list[VariantResult]) -> list[dict[str, Any]]:
    """Serialize all variant results into a JSON-safe list, one entry per platform."""
    return [variant_to_dict(result) for result in results]


def to_json(results: list[VariantResult], *, indent: int = 2) -> str:
    """Render the full audit log as a JSON string ready to hand to the frontend.

    ``ensure_ascii=False`` keeps emojis (welcome in the social-ad style) readable
    instead of escaping them.
    """
    return json.dumps(build_audit_log(results), indent=indent, ensure_ascii=False)
