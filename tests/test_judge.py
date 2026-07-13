"""Tests for the semantic distinctness judge — deterministic, no live API.

Three layers are covered without any model call:
  * service side (``assess_with_judge`` + ``run_all_platforms`` integration): a stub judge
    exercises the per-variant mapping, the always-fall-back-on-failure guarantee, and the
    parallel-signal serialization including ``signals_agree``;
  * agent side (``agent.judge._to_verdict`` / ``_resolve_platform_names``): mapping an
    already schema-validated ``JudgeVerdictSchema`` onto the internal ``JudgeVerdict`` shape;
  * the schema itself (``JudgeVerdictSchema``): this is where structural validity is now
    enforced (via ``with_structured_output`` forced tool-calling, not a "return ONLY JSON"
    prompt instruction) — a malformed field must raise ``pydantic.ValidationError`` here,
    since ``_to_verdict`` no longer does any of that checking itself.

The judge's actual reliability against human judgment is a separate, live concern handled by
tests/harness_judge.py against the labeled fixtures — that is the Q4 validation gate, not
something a unit test can assert.
"""

from __future__ import annotations

import pydantic
import pytest

from agent.judge import JudgeVerdictSchema, _resolve_platform_names, _to_verdict
from service.audit_log import variant_to_dict
from service.critique_loop import VariantResult, run_all_platforms
from service.distinctness import JudgeVerdict, assess_with_judge
from service.rules import PlatformRule


def _variant(name: str, label: str, text: str, angle: str = "") -> VariantResult:
    rule = PlatformRule(name=name, label=label, max_chars=100, style="s", angle=angle)
    return VariantResult(
        rule=rule,
        final_text=text,
        final_char_count=len(text),
        status="ai_approved",
        attempts=[],
    )


def _trio() -> list[VariantResult]:
    return [
        _variant("search_ad", "Search Ad", "alpha"),
        _variant("social_ad", "Social Ad", "beta"),
        _variant("display_banner", "Display Banner", "gamma"),
    ]


# --- assess_with_judge: mapping + fallback ----------------------------------------


def test_no_judge_yields_unavailable_findings() -> None:
    findings = assess_with_judge(_trio(), "product", None)
    assert all(f.distinct is None and f.note is None for f in findings)


def test_verdict_flags_only_named_platforms() -> None:
    def judge(*, variants: list[VariantResult], product_source: str) -> JudgeVerdict:
        return JudgeVerdict(
            distinct=False,
            rationale="Search and Display both argue the product is easy for beginners.",
            similar_platforms=("search_ad", "display_banner"),
        )

    search, social, display = assess_with_judge(_trio(), "product", judge)
    assert search.distinct is False and "beginners" in search.note
    assert social.distinct is True and social.note is None  # not named -> distinct
    assert display.distinct is False and display.note is not None


def test_verdict_distinct_flags_nobody() -> None:
    def judge(*, variants: list[VariantResult], product_source: str) -> JudgeVerdict:
        return JudgeVerdict(distinct=True, rationale="All three differ.", similar_platforms=())

    findings = assess_with_judge(_trio(), "product", judge)
    assert all(f.distinct is True and f.note is None for f in findings)


def test_not_distinct_without_named_platforms_flags_all() -> None:
    # Defensive: judge says the set is not distinct but names no platforms -> flag everyone.
    def judge(*, variants: list[VariantResult], product_source: str) -> JudgeVerdict:
        return JudgeVerdict(distinct=False, rationale="Two of these overlap.", similar_platforms=())

    findings = assess_with_judge(_trio(), "product", judge)
    assert all(f.distinct is False and f.note == "Two of these overlap." for f in findings)


def test_judge_failure_falls_back_to_unavailable() -> None:
    def judge(*, variants: list[VariantResult], product_source: str) -> JudgeVerdict:
        raise RuntimeError("simulated API/timeout/parse failure")

    findings = assess_with_judge(_trio(), "product", judge)
    assert all(f.distinct is None and f.note is None for f in findings)


# --- run_all_platforms integration -------------------------------------------------


def _rules() -> list[PlatformRule]:
    return [
        PlatformRule(name="search_ad", label="Search Ad", max_chars=100, style="s"),
        PlatformRule(name="social_ad", label="Social Ad", max_chars=100, style="s"),
        PlatformRule(name="display_banner", label="Display Banner", max_chars=100, style="s"),
    ]


def _distinct_generator(*, product_source: str, rule: PlatformRule, feedback: str | None = None) -> str:
    # Genuinely different words per platform so the lexical signal stays independent/clean.
    return {"search_ad": "buy fast today", "social_ad": "feel amazing forever", "display_banner": "trusted quality worldwide"}[rule.name]


def test_run_all_platforms_attaches_judge_verdict() -> None:
    def judge(*, variants: list[VariantResult], product_source: str) -> JudgeVerdict:
        return JudgeVerdict(
            distinct=False, rationale="Search collides with Display.", similar_platforms=("search_ad",)
        )

    results = run_all_platforms(
        product_source="x", rules=_rules(), generate=_distinct_generator, judge=judge
    )
    by_name = {r.rule.name: r for r in results}
    assert by_name["search_ad"].judge_distinct is False
    assert by_name["social_ad"].judge_distinct is True
    # Lexical signal is computed independently and still populated alongside the judge.
    assert all(r.distinct is not None for r in results)


def test_run_all_platforms_without_judge_leaves_judge_fields_none() -> None:
    results = run_all_platforms(product_source="x", rules=_rules(), generate=_distinct_generator)
    assert all(r.judge_distinct is None and r.judge_note is None for r in results)
    assert all(r.distinct is not None for r in results)  # lexical still runs


def test_run_all_platforms_survives_judge_failure() -> None:
    def judge(*, variants: list[VariantResult], product_source: str) -> JudgeVerdict:
        raise RuntimeError("outage")

    results = run_all_platforms(
        product_source="x", rules=_rules(), generate=_distinct_generator, judge=judge
    )
    assert all(r.judge_distinct is None for r in results)  # fell back
    assert all(r.distinct is not None for r in results)  # lexical unaffected


# --- audit_log: judge fields + signals_agree --------------------------------------


def _serialized(distinct: bool | None, judge_distinct: bool | None) -> dict:
    rule = PlatformRule(name="a", label="A", max_chars=100, style="s")
    result = VariantResult(
        rule=rule,
        final_text="t",
        final_char_count=1,
        status="ai_approved",
        attempts=[],
        distinct=distinct,
        judge_distinct=judge_distinct,
    )
    return variant_to_dict(result)


@pytest.mark.parametrize(
    ("distinct", "judge_distinct", "expected_agree"),
    [
        (True, True, True),
        (False, False, True),
        (True, False, False),  # the drone case: lexical distinct, judge catches collision
        (False, True, False),  # the jeans case: lexical collides on a shared noun, judge clears it
        (True, None, None),  # judge unavailable
        (None, True, None),  # lexical not assessed
    ],
)
def test_signals_agree(distinct, judge_distinct, expected_agree) -> None:
    assert _serialized(distinct, judge_distinct)["signals_agree"] is expected_agree


def test_judge_label_mapping() -> None:
    assert _serialized(True, True)["judge_label"] == "Angles distinct"
    assert _serialized(True, False)["judge_label"] == "Same angle"
    assert _serialized(True, None)["judge_label"] is None  # unavailable -> no badge


# --- agent.judge: mapping an already-validated schema instance --------------------


def test_to_verdict_resolves_labels_and_names_to_names() -> None:
    variants = _trio()
    parsed = JudgeVerdictSchema(
        distinct=False,
        rationale="same claim",
        similar_platforms=["Search Ad", "display_banner"],  # one label, one name
    )
    verdict = _to_verdict(parsed, variants)
    assert verdict.distinct is False
    assert verdict.rationale == "same claim"
    assert verdict.similar_platforms == ("search_ad", "display_banner")


def test_resolve_platform_names_is_case_insensitive_and_drops_unknowns() -> None:
    variants = _trio()
    resolved = _resolve_platform_names(["SEARCH AD", "nonsense", "Social Ad", "Social Ad"], variants)
    assert resolved == ("search_ad", "social_ad")  # deduped, unknown dropped


def test_to_verdict_missing_similar_platforms_defaults_empty() -> None:
    parsed = JudgeVerdictSchema(distinct=True, rationale="all different")
    verdict = _to_verdict(parsed, _trio())
    assert verdict.similar_platforms == ()


# --- JudgeVerdictSchema: structural validity now enforced here, not by prompt instruction --


def test_schema_accepts_well_formed_response() -> None:
    parsed = JudgeVerdictSchema(
        distinct=True, rationale="all different", similar_platforms=["Search Ad"]
    )
    assert parsed.distinct is True
    assert parsed.rationale == "all different"
    assert parsed.similar_platforms == ["Search Ad"]


def test_schema_rejects_blank_rationale() -> None:
    with pytest.raises(pydantic.ValidationError):
        JudgeVerdictSchema(distinct=True, rationale="   ")


def test_schema_requires_rationale() -> None:
    with pytest.raises(pydantic.ValidationError):
        JudgeVerdictSchema(distinct=True)  # rationale is a required field


def test_schema_rejects_non_list_similar_platforms() -> None:
    # A bare string is not auto-split into a list — must be an actual JSON array.
    with pytest.raises(pydantic.ValidationError):
        JudgeVerdictSchema(distinct=False, rationale="x", similar_platforms="Search Ad")


def test_schema_similar_platforms_defaults_to_empty_list() -> None:
    parsed = JudgeVerdictSchema(distinct=True, rationale="all different")
    assert parsed.similar_platforms == []
