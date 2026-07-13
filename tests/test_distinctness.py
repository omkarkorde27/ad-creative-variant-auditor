"""Tests for the cross-platform distinctness signal (service/distinctness.py).

Covers the metric (overlap coefficient + its empty-set guard), the tokenizer (emoji /
punctuation / stopword / short-token handling), the collection-level ``assess_distinctness``
verdict on both the observed real failure and a genuinely-distinct trio, the threshold
boundary, and the end-to-end path through ``run_all_platforms`` into the serialized audit
log. No API is hit — distinctness is pure lexical logic over already-final text.
"""

from __future__ import annotations

from service.audit_log import variant_to_dict
from service.critique_loop import VariantResult, run_all_platforms
from service.distinctness import (
    DISTINCTNESS_THRESHOLD,
    _extract_product_terms,
    _stem,
    _tokenize,
    assess_distinctness,
    overlap_coefficient,
)
from service.rules import PlatformRule

# The real observed failure: one idea rendered at three lengths, each obeying its limit.
OBSERVED = {
    "search_ad": "fits your whole day",
    "social_ad": "fits your whole life... better",
    "display_banner": "fits your full day",
}
# Three genuinely different creative angles sharing no content words.
DISTINCT = {
    "search_ad": "Warmest recycled fleece",
    "social_ad": "Cozy vibes all season long, friend",
    "display_banner": "Sustainable warmth for outdoor adventurers",
}


def _variant(name: str, label: str, text: str) -> VariantResult:
    """Minimal VariantResult carrying only what distinctness reads: rule label/name + text."""
    rule = PlatformRule(name=name, label=label, max_chars=100, style="s")
    return VariantResult(
        rule=rule,
        final_text=text,
        final_char_count=len(text),
        status="ai_approved",
        attempts=[],
    )


# --- overlap_coefficient ----------------------------------------------------------


def test_overlap_coefficient_full_containment_is_one() -> None:
    assert overlap_coefficient({"a", "b"}, {"a", "b", "c"}) == 1.0


def test_overlap_coefficient_partial() -> None:
    # min set size 2, one shared -> 0.5
    assert overlap_coefficient({"a", "b"}, {"a", "c", "d", "e"}) == 0.5


def test_overlap_coefficient_is_symmetric() -> None:
    a, b = {"x", "y", "z"}, {"y", "z"}
    assert overlap_coefficient(a, b) == overlap_coefficient(b, a)


def test_overlap_coefficient_empty_set_guard() -> None:
    # A variant with no content words cannot be "too similar" to anything.
    assert overlap_coefficient(set(), {"a"}) == 0.0
    assert overlap_coefficient(set(), set()) == 0.0


# --- _tokenize --------------------------------------------------------------------


def test_tokenize_strips_punctuation_and_emoji() -> None:
    # "vibes" stems to "vibe" (see stemming tests below).
    assert _tokenize("Cozy vibes 🧥 life... better") == {"cozy", "vibe", "life", "better"}


def test_tokenize_drops_stopwords_and_lowercases() -> None:
    # "your" is a stopword; the rest are content words, lowercased. "fits" stems to "fit".
    assert _tokenize("Fits Your Whole Day") == {"fit", "whole", "day"}


def test_tokenize_drops_single_character_tokens() -> None:
    # "USB-C" splits to usb + c; the 1-char "c" is noise and is dropped.
    assert _tokenize("USB-C port") == {"usb", "port"}


def test_tokenize_stems_plurals_so_they_match_singulars() -> None:
    # The observed false negative: "legs" and "leg" were treated as unrelated tokens,
    # letting a real collision escape detection. Stemming collapses them to one key.
    assert _tokenize("Straight legs") == _tokenize("Straight leg")
    assert _tokenize("Free returns, always") & _tokenize("a full return policy") == {"return"}


# --- _stem --------------------------------------------------------------------------


def test_stem_strips_simple_plural() -> None:
    assert _stem("legs") == "leg"
    assert _stem("returns") == "return"


def test_stem_handles_ies_suffix() -> None:
    assert _stem("berries") == "berry"


def test_stem_leaves_ss_endings_untouched() -> None:
    # Must not mangle non-plural words ending in "s" into meaningless fragments.
    assert _stem("glass") == "glass"
    assert _stem("dress") == "dress"


def test_stem_leaves_short_words_untouched() -> None:
    assert _stem("gas") == "gas"  # at/under MIN_STEMMABLE_LENGTH -> left alone


# --- _extract_product_terms ----------------------------------------------------------


def test_extract_product_terms_reads_leading_lines() -> None:
    # "magnus" stems to "magnu" (a disclosed limitation of a dictionary-free stemmer on
    # proper nouns — see _stem's docstring); still correct because it's applied consistently.
    source = "Magnus Straight Leg Jeans\nAGOLDE\n$238.00 Current Price\n\nDetails: ..."
    assert _extract_product_terms(source) == {"magnu", "straight", "leg", "jean", "agolde"}


def test_extract_product_terms_empty_source_yields_empty_set() -> None:
    assert _extract_product_terms("") == set()


# --- assess_distinctness ----------------------------------------------------------


def test_assess_flags_the_observed_failure() -> None:
    results = [_variant(name, name.replace("_", " ").title(), text) for name, text in OBSERVED.items()]
    findings = assess_distinctness(results, product_source="")

    # All three are the same idea, so none is distinct.
    assert all(f.distinct is False for f in findings)
    search = findings[0]
    assert search.score >= 0.66  # 2 of 3 content words reused
    assert search.note is not None
    assert "fit" in search.note and "whole" in search.note  # "fits" stems to "fit"
    assert search.nearest_label in {"Social Ad", "Display Banner"}


def test_assess_passes_genuinely_distinct_variants() -> None:
    results = [_variant(name, name.replace("_", " ").title(), text) for name, text in DISTINCT.items()]
    findings = assess_distinctness(results, product_source="")

    assert all(f.distinct is True for f in findings)
    assert all(f.note is None for f in findings)  # no explanation needed when distinct


def test_assess_threshold_boundary_is_inclusive() -> None:
    # Exactly at the threshold (0.5) counts as too similar: distinct = score < threshold.
    a = _variant("a", "A", "alpha beta")  # {alpha, beta}
    b = _variant("b", "B", "alpha gamma delta epsilon")  # {alpha, gamma, delta, epsilon}
    findings = assess_distinctness([a, b], product_source="")

    assert findings[0].score == DISTINCTNESS_THRESHOLD  # 1 / min(2,4) = 0.5
    assert findings[0].distinct is False  # at the threshold -> flagged


def test_assess_single_variant_is_trivially_distinct() -> None:
    findings = assess_distinctness([_variant("only", "Only", "anything at all")], product_source="")
    assert findings[0].distinct is True
    assert findings[0].nearest_label is None
    assert findings[0].note is None


def test_shared_product_name_words_are_not_flagged() -> None:
    # Regression test for the real 002_jeans false positive: Search Ad and Display Banner
    # both legitimately name the product ("straight leg"), which is NOT a creative-sameness
    # signal, while their actual content (urgency vs. trust framing) is genuinely distinct.
    product_source = "Magnus Straight Leg Jeans\nAGOLDE\n$238.00 Current Price"
    results = [
        _variant("search_ad", "Search Ad", "True-to-Size Straight Leg"),
        _variant(
            "display_banner",
            "Display Banner",
            "AGOLDE's regenerative cotton straight leg. Trusted by Nordstrom. Free returns.",
        ),
    ]
    findings = assess_distinctness(results, product_source=product_source)

    assert all(f.distinct is True for f in findings)


def test_product_name_exclusion_does_not_hide_a_real_collision() -> None:
    # The exclusion only removes product-name words; a genuine repeated *idea* beyond the
    # product name must still be caught.
    product_source = "Magnus Straight Leg Jeans\nAGOLDE"
    results = [
        _variant("search_ad", "Search Ad", "Straight leg jeans that fit your whole day"),
        _variant("social_ad", "Social Ad", "Straight leg jeans that fit your whole life"),
    ]
    findings = assess_distinctness(results, product_source=product_source)

    assert all(f.distinct is False for f in findings)


# --- integration: run_all_platforms + audit_log serialization ---------------------


def test_run_all_platforms_serializes_distinctness_flag() -> None:
    rules = [
        PlatformRule(name="search_ad", label="Search Ad", max_chars=100, style="s"),
        PlatformRule(name="social_ad", label="Social Ad", max_chars=100, style="s"),
        PlatformRule(name="display_banner", label="Display Banner", max_chars=100, style="s"),
    ]

    def gen_same(*, product_source: str, rule: PlatformRule, feedback: str | None = None) -> str:
        return "identical cozy fleece jacket copy for every platform"

    results = run_all_platforms(product_source="x", rules=rules, generate=gen_same)
    payload = [variant_to_dict(r) for r in results]

    # Identical text across platforms -> every card flagged too similar, with an explanation.
    for entry in payload:
        assert entry["distinct"] is False
        assert entry["distinct_label"] == "Too similar"
        assert entry["distinct_note"] is not None
