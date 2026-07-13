"""Persisted version of the manual verification scenarios for the critique loop.

Covers: (1) a stub VariantGenerator exercising the ai_approved, fallback-truncates-
shortest-attempt, and all-errors-still-falls-back paths; (2) truncate_to_limit against
adversarial input; (5) the audit log JSON round-trip. Rules validation (3) and file I/O
(4) live in their own test modules.
"""

from __future__ import annotations

import json

import pytest

from service.critique_loop import (
    Attempt,
    VariantResult,
    _build_feedback,
    run_all_platforms,
    run_critique_loop,
    truncate_to_limit,
)
from service.audit_log import to_json
from service.rules import PlatformRule

PRODUCT = (
    "The Aurora Desk Lamp is a smart, warm-to-cool tunable LED light with app control, "
    "USB-C charging, and a sleek aluminum body that fits any modern workspace."
)


@pytest.fixture
def search_rule() -> PlatformRule:
    # A tight limit (30 chars) makes it cheap to force overflow in test drafts.
    return PlatformRule(name="search_ad", label="Search Ad", max_chars=30, style="terse")


@pytest.fixture
def real_rules() -> list[PlatformRule]:
    # Mirrors the shipped data/platform_rules.json without touching the file.
    return [
        PlatformRule(name="search_ad", label="Search Ad", max_chars=30, style="terse"),
        PlatformRule(name="social_ad", label="Social Ad", max_chars=125, style="punchy"),
        PlatformRule(name="display_banner", label="Display Banner", max_chars=90, style="crisp"),
    ]


# --- Scenario 1: stub VariantGenerator behaviors ----------------------------------


def test_canned_short_returns_ai_approved(search_rule: PlatformRule) -> None:
    def gen_short(*, product_source: str, rule: PlatformRule, feedback: str | None = None) -> str:
        return "Buy now"

    result = run_critique_loop(product_source=PRODUCT, rule=search_rule, generate=gen_short)

    assert result.status == "ai_approved"
    assert result.final_char_count <= search_rule.max_chars
    assert len(result.attempts) == 1
    assert result.attempts[0].source == "llm"
    assert result.attempts[0].passed is True


def test_overflow_truncates_shortest_attempt(search_rule: PlatformRule) -> None:
    # Three overflowing drafts of decreasing length, all still over the 30-char limit.
    # The fallback must truncate the shortest one ("C"*60), not the last call made.
    drafts = ["A" * 120, "B" * 90, "C" * 60]
    calls = {"n": 0}

    def gen_overflow(*, product_source: str, rule: PlatformRule, feedback: str | None = None) -> str:
        draft = drafts[calls["n"]]
        calls["n"] += 1
        return draft

    result = run_critique_loop(product_source=PRODUCT, rule=search_rule, generate=gen_overflow)

    assert result.status == "fallback_truncated"
    assert result.final_char_count <= search_rule.max_chars
    assert set(result.final_text) == {"C"}  # proves the shortest draft, not the last, was kept
    assert len(result.attempts) == 4  # 3 llm attempts + 1 fallback
    assert [a.source for a in result.attempts] == ["llm", "llm", "llm", "fallback"]


def test_all_errors_fall_back_to_product_source(search_rule: PlatformRule) -> None:
    def gen_error(*, product_source: str, rule: PlatformRule, feedback: str | None = None) -> str:
        raise RuntimeError("simulated LLM/network outage")

    result = run_critique_loop(product_source=PRODUCT, rule=search_rule, generate=gen_error)

    assert result.status == "fallback_truncated"
    assert result.final_char_count <= search_rule.max_chars
    assert [a.source for a in result.attempts] == ["error", "error", "error", "fallback"]
    assert all(a.text == "" for a in result.attempts if a.source == "error")
    assert result.final_text != ""
    assert PRODUCT.startswith(result.final_text)  # fallback truncated the raw product source


# --- Scenario 2: truncate_to_limit adversarial input ------------------------------


@pytest.mark.parametrize(
    ("text", "max_chars"),
    [
        ("supercalifragilisticexpialidocious", 5),  # single word longer than the limit
        ("exactly8", 8),  # exact fit
        ("", 30),  # empty input
        ("one two three four five", 12),  # ordinary multi-word overflow
        ("   padded   text   ", 6),  # surrounding/interior whitespace
    ],
)
def test_truncate_to_limit_never_exceeds(text: str, max_chars: int) -> None:
    result = truncate_to_limit(text, max_chars)
    assert len(result) <= max_chars


def test_truncate_to_limit_exact_fit_is_unchanged() -> None:
    assert truncate_to_limit("exactly8", 8) == "exactly8"


def test_truncate_to_limit_empty_input_is_empty() -> None:
    assert truncate_to_limit("", 30) == ""


def test_truncate_to_limit_oversized_word_hard_slices() -> None:
    result = truncate_to_limit("supercalifragilisticexpialidocious", 5)
    assert result == "super"


def test_truncate_falls_back_to_hard_slice_when_no_boundary_exists() -> None:
    # A single unbroken word has no sentence ending, no clause boundary, and no space
    # to word-wrap on — none of the three boundary types can exist within the limit.
    # The function must still return a non-empty, in-limit result without crashing.
    word = "Pneumonoultramicroscopicsilicovolcanoconiosis"
    result = truncate_to_limit(word, 12)
    assert result == "Pneumonoultr"
    assert result != ""
    assert len(result) <= 12


def test_truncate_prefers_sentence_boundary_over_word_boundary() -> None:
    text = (
        "This jacket is fully waterproof. It also has a secure zip closure and "
        "interior pockets for gear."
    )
    result = truncate_to_limit(text, 60)
    # Ends on the sentence-ending period, not mid-word ("...zip cl") like a plain
    # word-boundary cut at 60 chars would.
    assert result == "This jacket is fully waterproof."
    assert result[-1] in ".!?"
    assert len(result) <= 60


def test_truncate_prefers_clause_boundary_when_no_sentence_ending_fits() -> None:
    text = (
        "Waterproof jacket with a secure zip closure, plus smart interior "
        "pockets for everyday carry."
    )
    result = truncate_to_limit(text, 50)
    # Ends on the comma, not mid-word ("...closure, plus") like a plain word-boundary
    # cut at 50 chars would produce.
    assert result == "Waterproof jacket with a secure zip closure,"
    assert result[-1] in ",;—"
    assert len(result) <= 50


# --- _build_feedback ratio branching (LARGE_OVERAGE_RATIO) ------------------------


def test_feedback_under_ratio_threshold_edits_in_place(search_rule: PlatformRule) -> None:
    # 37 chars against a 30-char limit is only slightly over (37 <= 30 * 1.5 = 45) —
    # within the threshold, so the edit-in-place instruction should be used.
    draft = "A genuinely excellent desk lamp today"
    feedback = _build_feedback(draft, search_rule)

    assert "Delete words, clauses, or punctuation from THIS EXACT TEXT" in feedback
    assert draft in feedback  # the exact prior draft is embedded for editing
    assert "Style: terse" in feedback


def test_feedback_over_ratio_threshold_writes_fresh_phrase(search_rule: PlatformRule) -> None:
    # A ~300-character draft against a 30-char limit mirrors the observed Search Ad
    # failure mode: well beyond 30 * 1.5 = 45, so pure editing is abandoned.
    draft = (
        "The Aurora Desk Lamp is a smart, warm-to-cool tunable LED light with full "
        "app control, USB-C fast charging, five brightness presets, and a sleek "
        "brushed-aluminum body that fits any modern workspace or studio setup."
    )
    assert len(draft) > search_rule.max_chars * 1.5  # sanity-check the scenario

    feedback = _build_feedback(draft, search_rule)

    assert "write a short, fresh phrase from scratch" in feedback
    assert "Delete words, clauses, or punctuation from THIS EXACT TEXT" not in feedback


# --- Scenario 5: audit log JSON round-trip -----------------------------------------


def test_to_json_round_trips(real_rules: list[PlatformRule]) -> None:
    # Fresh, non-stateful stub: always overflows every platform's limit, independent
    # of call count, so this test doesn't share behavior/state with scenario 1b's stub.
    def gen_always_overflow(
        *, product_source: str, rule: PlatformRule, feedback: str | None = None
    ) -> str:
        return product_source + " " + product_source

    results: list[VariantResult] = run_all_platforms(
        product_source=PRODUCT, rules=real_rules, generate=gen_always_overflow
    )

    payload = to_json(results)
    loaded = json.loads(payload)

    assert isinstance(loaded, list)
    assert len(loaded) == len(real_rules)

    expected_keys = {
        "platform",
        "label",
        "max_chars",
        "final_text",
        "final_char_count",
        "status",
        "status_label",
        "distinct",
        "distinct_label",
        "distinct_note",
        "judge_distinct",
        "judge_label",
        "judge_note",
        "signals_agree",
        "attempts",
    }
    for entry, rule in zip(loaded, real_rules):
        assert set(entry.keys()) == expected_keys
        assert entry["max_chars"] == rule.max_chars
        assert entry["final_char_count"] <= entry["max_chars"]
        assert entry["status_label"] in ("AI-approved", "fallback-truncated")
        assert isinstance(entry["attempts"], list) and len(entry["attempts"]) > 0
