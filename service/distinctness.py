"""Cross-platform distinctness assessment for the three final variants.

Character-limit compliance is a *per-platform* property, validated inside
``run_critique_loop`` because ``max_chars`` belongs to one rule. Distinctness is the
opposite: it is only meaningful *across* the finished variants — the same idea rendered at
three lengths (Search "fits your whole day" / Social "fits your whole life… better" /
Display "fits your full day") satisfies every character limit yet fails the spec's
requirement that the three be "completely distinct". This module supplies the missing
cross-platform signal.

Design commitments (see the plan and CLAUDE.md):
  * Cheap and explainable — a lexical *overlap coefficient* on content-word sets, no
    embeddings, no model calls, no new dependencies. It is a heuristic: it catches repeated
    phrasing ("fits your whole"), not synonym-level paraphrase ("day" vs "life"). That
    honest limitation is why detection can only *flag*, never *guarantee*, distinctness.
  * Pure and side-effect free — it reads each result's ``final_text`` and never mutates
    anything, so it cannot affect the character-limit guarantee it runs after.
  * ``DISTINCTNESS_THRESHOLD`` is a system property hardcoded here, NOT read from
    ``platform_rules.json`` — mirroring how ``MAX_ATTEMPTS`` is deliberately not
    rule-sourced. Distinctness is a property of the set of platforms, not of any one rule.

Two refinements (added after live measurement surfaced both failure modes on a real
product — see ``002_jeans`` in the regression set):
  * A light heuristic stemmer collapses simple surface variants ("leg"/"legs",
    "return"/"returns") so a paraphrase does not escape detection purely by pluralizing a
    shared word — the false negative that let a Social Ad variant dodge a real collision.
  * ``assess_distinctness`` excludes the product's own name/title words (the leading lines
    of ``product_source``) from comparison. Every platform may legitimately need to name
    the product for relevance; two variants both saying "straight leg jeans" is not
    evidence of creative sameness, and without this exclusion it was a false positive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # annotations only — importing at runtime would be circular
    from service.critique_loop import VariantResult

# Two content-word sets are "too similar" when their overlap coefficient reaches this.
# Tunable; 0.5 flags the observed failure (0.67) while leaving genuinely different angles
# (which share few content words) comfortably below.
DISTINCTNESS_THRESHOLD: float = 0.5

# Tokens shorter than this are dropped as noise: single letters left over from splitting
# compounds ("USB-C" -> "usb", "c") carry no creative-angle signal.
MIN_TOKEN_LENGTH: int = 2

# At most this many shared terms are quoted in a finding's note, to keep it readable.
MAX_SHARED_TERMS_IN_NOTE: int = 4

# Below this length, stripping a trailing "s" risks mangling a short word into a
# meaningless fragment ("gas" -> "ga") rather than de-pluralizing it. Words at or under
# this length are left as-is by ``_stem``. Set just low enough that a 4-letter plural like
# "legs" (-> "leg") still gets stemmed.
MIN_STEMMABLE_LENGTH: int = 3

# Leading non-empty lines of ``product_source`` treated as the product's own name/brand,
# and therefore excluded from the distinctness comparison (see module docstring). Verified
# against every file in data/regression_set/: the scraped product name/brand consistently
# appears in the first two lines, before pricing and review noise begins.
PRODUCT_TITLE_LINES: int = 2

# Pure function words. Deliberately small and conservative: it strips grammatical glue
# ("your", "the", "with") but NOT meaning-bearing words like "better" or "more", which are
# exactly the kind of token that legitimately distinguishes one angle from another.
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "nor", "for", "to", "of", "in", "on",
        "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been",
        "being", "it", "its", "this", "that", "these", "those", "your", "you", "our",
        "we", "they", "their", "i", "my", "me", "so", "if", "then", "than", "too",
    }
)

# Alphanumeric runs only: this splits on and discards all punctuation and emoji in one
# step ("life… better" -> ["life", "better"], "cozy 🧥 layer" -> ["cozy", "layer"]).
_WORD_RE = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """Collapse a common plural suffix to a matching key — not a linguistically correct
    stem, just enough to stop "leg"/"legs" or "return"/"returns" from being treated as
    unrelated tokens. Conservative by design: skips words at or under
    ``MIN_STEMMABLE_LENGTH`` and words ending "ss" (so "glass", "dress" pass through
    untouched) rather than risk mangling a non-plural word into a meaningless fragment.

    Known, accepted limitation: a proper noun that happens to end in "s" (a brand name
    like "Magnus") is stemmed the same as a real plural ("magnus" -> "magnu"), since
    nothing here distinguishes proper nouns without a dictionary or NER model — out of
    scope for a cheap, dependency-free heuristic. This does not break correctness: the same
    mangled key is produced everywhere that word appears, so overlap comparisons stay
    consistent; it only makes that one token look odd if quoted in a finding's note.
    """
    if len(word) <= MIN_STEMMABLE_LENGTH:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokenize(text: str) -> set[str]:
    """Normalize ``text`` to a set of content words for overlap comparison.

    Lowercases, keeps only alphanumeric runs (dropping punctuation and emoji), removes
    stopwords and too-short tokens, then stems what's left so simple surface variants
    collapse to the same key. Returns a set — multiplicity does not matter for an overlap
    signal, only which content words two variants share.
    """
    return {
        _stem(token)
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= MIN_TOKEN_LENGTH and token not in STOPWORDS
    }


def _extract_product_terms(product_source: str) -> set[str]:
    """Tokenize the product's own name/brand (its leading lines) for exclusion.

    See ``PRODUCT_TITLE_LINES``. An empty or unusual ``product_source`` degrades gracefully
    to an empty exclusion set — comparison then proceeds exactly as if this refinement did
    not exist, rather than raising.
    """
    lines = [line for line in product_source.splitlines() if line.strip()]
    title_text = " ".join(lines[:PRODUCT_TITLE_LINES])
    return _tokenize(title_text)


def overlap_coefficient(a: set[str], b: set[str]) -> float:
    """Szymkiewicz–Simpson overlap coefficient: ``|a ∩ b| / min(|a|, |b|)``.

    Chosen over Jaccard because the variants differ hugely in length (a ~4-word Search
    headline vs a ~20-word Social caption); Jaccard's union denominator would dilute a real
    collision, while normalizing by the *smaller* set exposes it. Returns ``0.0`` when
    either set is empty — a variant with no content words cannot be "too similar".
    """
    smaller = min(len(a), len(b))
    if smaller == 0:
        return 0.0
    return len(a & b) / smaller


@dataclass(frozen=True)
class DistinctnessFinding:
    """One variant's distinctness outcome, aligned by position to the assessed results.

    ``distinct`` is the flag surfaced to the UI. ``nearest_label`` and ``score`` record the
    most-similar sibling and that similarity (kept even when distinct, for auditing).
    ``note`` is the human-readable explanation, populated only when a collision is flagged.
    """

    platform: str
    distinct: bool
    nearest_label: str | None
    score: float
    note: str | None


def _build_note(this_tokens: set[str], nearest_tokens: set[str], nearest_label: str, score: float) -> str:
    """Explain a flagged collision: which content words are shared, and with whom."""
    shared = sorted(this_tokens & nearest_tokens)[:MAX_SHARED_TERMS_IN_NOTE]
    shared_str = ", ".join(shared)
    return f"Shares '{shared_str}' with {nearest_label} ({score:.0%} word overlap)"


def assess_distinctness(
    results: list[VariantResult],
    product_source: str,
    threshold: float = DISTINCTNESS_THRESHOLD,
) -> list[DistinctnessFinding]:
    """Compare every final variant against the others and flag insufficiently distinct ones.

    For each variant, finds its maximum pairwise overlap coefficient against the other
    finals; it is ``distinct`` when that maximum is below ``threshold``. Returns one finding
    per input result, in the same order. With fewer than two results there are no pairs to
    compare, so every variant is trivially distinct.

    ``product_source`` supplies the product-name exclusion set (see
    ``_extract_product_terms``): words drawn from the product's own name are removed from
    every variant's tokens before comparison, so two variants both legitimately naming the
    product is never mistaken for creative sameness.
    """
    product_terms = _extract_product_terms(product_source)
    tokens = [_tokenize(result.final_text) - product_terms for result in results]
    findings: list[DistinctnessFinding] = []

    for i, result in enumerate(results):
        best_score = 0.0
        best_j: int | None = None
        for j in range(len(results)):
            if j == i:
                continue
            score = overlap_coefficient(tokens[i], tokens[j])
            if score > best_score:
                best_score = score
                best_j = j

        distinct = best_score < threshold
        nearest_label = results[best_j].rule.label if best_j is not None else None
        note = (
            _build_note(tokens[i], tokens[best_j], nearest_label, best_score)
            if best_j is not None and not distinct
            else None
        )
        findings.append(
            DistinctnessFinding(
                platform=result.rule.name,
                distinct=distinct,
                nearest_label=nearest_label,
                score=best_score,
                note=note,
            )
        )

    return findings


# --- Semantic second pass: the LLM judge -------------------------------------------
#
# The lexical pass above measures shared *words*. It is structurally blind to shared
# *meaning* expressed in different words (e.g. "ready today for beginners" vs "built for
# beginners who want reliable flight" — same claim, near-zero word overlap). The judge is a
# second, independent signal that reads the three finished variants against their assigned
# strategic angles and asks whether they commit to genuinely different arguments. It is an
# LLM call, so — like the generator — the *mechanism* lives in agent/ behind a Protocol,
# while this module owns the *contract*, the per-variant mapping, and the fallback policy.


@dataclass(frozen=True)
class JudgeVerdict:
    """The judge's set-level assessment of all three variants together.

    ``distinct`` is the overall call; ``similar_platforms`` names the platforms caught in a
    strategic collision (empty when ``distinct`` is True); ``rationale`` is the required
    human-readable explanation. This is the raw judge output — ``assess_with_judge`` maps it
    onto the per-variant shape the UI renders.
    """

    distinct: bool
    rationale: str
    similar_platforms: tuple[str, ...]


class DistinctnessJudge(Protocol):
    """The contract an LLM-backed judge must satisfy (implemented in ``agent/judge.py``).

    It receives all three finished variants together (so it can compare their strategic
    angles) and returns one ``JudgeVerdict``. It may raise on API/timeout/parse failure;
    ``assess_with_judge`` catches that and falls back to the lexical signal.
    """

    def __call__(
        self, *, variants: list[VariantResult], product_source: str
    ) -> JudgeVerdict: ...


@dataclass(frozen=True)
class JudgeFinding:
    """One variant's judge outcome, aligned by position to the assessed results.

    ``distinct`` is ``None`` when the judge did not run or failed (the fallback state, in
    which only the lexical signal is shown); otherwise it is this variant's per-platform
    verdict. ``note`` carries the judge's rationale, populated only when this variant is
    flagged — mirroring ``DistinctnessFinding.note``.
    """

    distinct: bool | None
    note: str | None


# A judge finding meaning "judge unavailable" — the pipeline degrades to lexical-only.
_JUDGE_UNAVAILABLE = JudgeFinding(distinct=None, note=None)


def _map_verdict(results: list[VariantResult], verdict: JudgeVerdict) -> list[JudgeFinding]:
    """Project a set-level ``JudgeVerdict`` onto one ``JudgeFinding`` per variant.

    A variant is flagged (``distinct=False``, rationale attached) when it is named in
    ``similar_platforms``. Defensive: if the judge reports the set is not distinct but names
    no platforms, every variant is flagged, since we cannot tell which pair collided but must
    not silently drop the verdict.
    """
    similar = set(verdict.similar_platforms)
    findings: list[JudgeFinding] = []
    for result in results:
        if verdict.distinct:
            findings.append(JudgeFinding(distinct=True, note=None))
            continue
        flagged = result.rule.name in similar if similar else True
        findings.append(
            JudgeFinding(distinct=not flagged, note=verdict.rationale if flagged else None)
        )
    return findings


def assess_with_judge(
    results: list[VariantResult],
    product_source: str,
    judge: DistinctnessJudge | None,
) -> list[JudgeFinding]:
    """Run the semantic judge and return one ``JudgeFinding`` per variant, aligned to order.

    Independent of and parallel to ``assess_distinctness``; it never gates or is gated by the
    lexical signal. When ``judge`` is ``None`` (not injected) or the call raises for ANY
    reason — API error, timeout, unparseable output — every variant gets the "unavailable"
    finding so the caller leaves the judge fields at their default and the lexical signal
    stands alone. The pipeline must never break or block generation on a judge failure.
    """
    if judge is None:
        return [_JUDGE_UNAVAILABLE for _ in results]
    try:
        verdict = judge(variants=results, product_source=product_source)
    except Exception:  # noqa: BLE001 - any judge failure falls back to the lexical signal
        return [_JUDGE_UNAVAILABLE for _ in results]
    return _map_verdict(results, verdict)
