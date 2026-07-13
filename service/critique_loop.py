"""The critique loop — the architectural centerpiece.

Guarantees that every emitted variant satisfies its platform's character limit, and
that the guarantee comes from code (``len()`` + deterministic truncation), never from
trusting the LLM's own claim about its output.

Design commitments (see CLAUDE.md):
  * The retry cap is ``MAX_ATTEMPTS``, hardcoded here — NOT read from the rules file.
  * After the LLM attempts are exhausted, a deterministic word-boundary truncation
    produces a provably compliant string, so the system cannot fail the limit.
  * The LLM is reached only through an injected ``VariantGenerator`` callable, so this
    module (and all of ``service/``) has no dependency on ``agent/`` and can be tested
    with a stub generator before the agent exists.
"""
from __future__ import annotations
from anthropic.types import bash_code_execution_tool_result_error_code

from dataclasses import dataclass, replace
from typing import Literal, Protocol

from service.distinctness import DistinctnessJudge, assess_distinctness, assess_with_judge
from service.rules import PlatformRule

# Hardcoded retry cap. This is the source of truth for how many times the LLM is asked
# to fix its own output before we truncate deterministically. It is intentionally NOT
# sourced from platform_rules.json.
MAX_ATTEMPTS: int = 3


class VariantGenerator(Protocol):
    """The contract the agent must satisfy. It produces text; it never counts characters.

    ``feedback`` is ``None`` on the first attempt and, on retries, carries the critique
    describing why the previous draft was rejected.
    """

    def __call__(
        self,
        *,
        product_source: str,
        rule: PlatformRule,
        feedback: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class Attempt:
    """One row in a variant's audit trail (an LLM draft, an error, or the fallback)."""

    attempt_number: int  # 1-indexed, monotonically increasing within a variant
    source: Literal["llm", "error", "fallback"]  # "error" == generate() raised
    text: str  # "" when source == "error"
    char_count: int  # len(text) — the single source of truth
    max_chars: int
    passed: bool  # char_count <= max_chars
    note: str | None = None  # critique fed back to the LLM, or the exception message


@dataclass(frozen=True)
class VariantResult:
    """The outcome for one platform: the chosen text plus the full attempt trail.

    ``distinct`` / ``distinct_note`` are the cross-platform *lexical* distinctness verdict,
    filled in by ``run_all_platforms`` after all variants exist (see ``service.distinctness``).
    They default to ``None`` because a single ``run_critique_loop`` call cannot know them — a
    variant is only comparable once its siblings are also final. ``None`` therefore means
    "not assessed" (e.g. a lone-platform test), distinct from ``True``/``False``.

    ``judge_distinct`` / ``judge_note`` are the parallel *semantic* verdict from the optional
    LLM judge. ``None`` means the judge did not run or failed — the pipeline degrades to the
    lexical signal alone, never breaking on a judge error.
    """

    rule: PlatformRule
    final_text: str
    final_char_count: int
    status: Literal["ai_approved", "fallback_truncated"]
    attempts: list[Attempt]
    distinct: bool | None = None
    distinct_note: str | None = None
    judge_distinct: bool | None = None
    judge_note: str | None = None


def count_chars(text: str) -> int:
    """The one place characters are counted. Everything else defers to this."""
    return len(text)


def fits(text: str, rule: PlatformRule) -> bool:
    """True iff ``text`` is within the platform's hard character limit."""
    return count_chars(text) <= rule.max_chars


# Boundaries truncate_to_limit prefers to cut on, most coherent first. Plain "-" is
# intentionally excluded from clause endings so compounds like "USB-C" are never split.
SENTENCE_ENDINGS: tuple[str, ...] = (".", "!", "?")
CLAUSE_ENDINGS: tuple[str, ...] = (",", ";", "—")

# A word-boundary cut is only accepted if it keeps at least this fraction of the
# character budget. Below it, a hard character slice (more content, even mid-word)
# beats discarding >30% of the limit just to land on a word boundary.
MIN_WORD_BOUNDARY_FRACTION: float = 0.7


def _last_punctuation_boundary(text: str, max_chars: int, endings: tuple[str, ...]) -> str:
    """Longest prefix of ``text`` ending on one of ``endings``, within ``max_chars``.

    A character only counts as a boundary if it is the last character of ``text`` or is
    followed by whitespace. This keeps in-token punctuation — a decimal point in
    "$129.95", a thousands separator in "7,307" — from being mistaken for a sentence or
    clause end. Returns ``""`` if no such boundary exists within the limit.
    """
    limit = min(max_chars, len(text))
    best = -1
    for i in range(limit):
        if text[i] in endings and (i + 1 == len(text) or text[i + 1].isspace()):
            best = i
    return text[: best + 1].strip() if best != -1 else ""


def _last_word_boundary(text: str, max_chars: int) -> str:
    """Longest prefix of ``text`` made of whole words that fits within ``max_chars``.

    Returns ``""`` if even the first word already exceeds ``max_chars``.
    """
    truncated = ""
    for word in text.split():
        candidate = word if not truncated else f"{truncated} {word}"
        if count_chars(candidate) <= max_chars:
            truncated = candidate
        else:
            break
    return truncated


def truncate_to_limit(text: str, max_chars: int) -> str:
    text = text.strip()
    if count_chars(text) <= max_chars:
        return text

    min_acceptable = max_chars * MIN_WORD_BOUNDARY_FRACTION

    sentence = _last_punctuation_boundary(text, max_chars, SENTENCE_ENDINGS)
    if sentence and count_chars(sentence) >= min_acceptable:
        return sentence

    clause = _last_punctuation_boundary(text, max_chars, CLAUSE_ENDINGS)
    if clause and count_chars(clause) >= min_acceptable:
        return clause

    word_boundary = _last_word_boundary(text, max_chars)
    if count_chars(word_boundary) >= min_acceptable:
        return word_boundary

    return text[:max_chars]


# When a draft overshoots by more than this multiple of the limit, editing it down word
# by word cannot coherently reach the target — a 4-word Search headline is a different
# artifact from a trimmed paragraph, not a shorter version of it. Past this ratio we stop
# asking the model to delete from the draft and ask for a fresh, ultra-short rewrite.
LARGE_OVERAGE_RATIO: float = 1.5
# Rough characters-per-word (including the trailing space), used only to translate a
# character limit into an approximate *word* ceiling for the fresh-rewrite instruction.
# A word ceiling gives the model the sense of scale a raw "too long" critique lacks. The
# agent never sees max_chars; service derives the word count here and passes it as prose.
AVG_CHARS_PER_WORD: int = 6


def _build_feedback(text: str, rule: PlatformRule) -> str:
    """Critique handed to the generator so its next attempt can fit the limit.

    The instruction is magnitude-aware. A modest overage is best fixed by editing the
    existing draft down — delete words, keep the angle. A large overage (more than
    ``LARGE_OVERAGE_RATIO``× the limit) cannot be reached by deletion without wrecking
    coherence, so instead we ask for a fresh, ultra-short rewrite bounded by an explicit
    word ceiling derived from the limit. ``service`` owns ``max_chars`` and the derived
    word count; the agent only relays this text verbatim and never counts anything.
    """
    length = count_chars(text)
    over_by = length - rule.max_chars

    if length <= rule.max_chars * LARGE_OVERAGE_RATIO:
        return (
            f"Your previous {rule.label} draft, which you must edit rather than rewrite "
            f"from scratch, was:\n\n{text}\n\n"
            f"That draft is {length} characters, {over_by} over the hard limit of "
            f"{rule.max_chars}. Delete words, clauses, or punctuation from THIS EXACT TEXT "
            f"until it fits within {rule.max_chars} characters. Do not introduce new "
            f"phrasing or a different angle. Style: {rule.style}"
        )

    word_ceiling = max(1, rule.max_chars // AVG_CHARS_PER_WORD)
    angle_reminder = f" Stay committed to this angle: {rule.angle}" if rule.angle else ""
    return (
        f"Your previous {rule.label} draft is {length} characters — far over the hard "
        f"limit of {rule.max_chars}. Do not edit it. Instead, write a short, fresh phrase "
        f"from scratch: a {rule.label} of at most {word_ceiling} words that keeps only the "
        f"single strongest benefit and fits within {rule.max_chars} characters. "
        f"Style: {rule.style}.{angle_reminder}"
    )


def run_critique_loop(
    *,
    product_source: str,
    rule: PlatformRule,
    generate: VariantGenerator,
    max_attempts: int = MAX_ATTEMPTS,
) -> VariantResult:
    """Generate → validate → retry → guaranteed fallback for a single platform.

    Calls ``generate`` up to ``max_attempts`` times. The first draft that fits (by
    ``len()``) is returned with status ``ai_approved``. A ``generate`` call that raises
    is caught, logged as an ``error`` attempt, and the loop continues. If no attempt
    fits, the shortest usable LLM draft (or ``product_source`` if none produced text)
    is truncated deterministically and returned with status ``fallback_truncated``.

    ``max_attempts`` defaults to the module constant and is exposed only so tests can
    shrink it; production always uses ``MAX_ATTEMPTS``.
    """
    attempts: list[Attempt] = []
    feedback: str | None = None

    for attempt_number in range(1, max_attempts + 1):
        try:
            text = generate(product_source=product_source, rule=rule, feedback=feedback)
        except Exception as exc:  # noqa: BLE001 - any generator failure must still fall back
            attempts.append(
                Attempt(
                    attempt_number=attempt_number,
                    source="error",
                    text="",
                    char_count=0,
                    max_chars=rule.max_chars,
                    passed=False,
                    note=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        char_count = count_chars(text)
        passed = fits(text, rule)
        note = None if passed else _build_feedback(text, rule)
        attempts.append(
            Attempt(
                attempt_number=attempt_number,
                source="llm",
                text=text,
                char_count=char_count,
                max_chars=rule.max_chars,
                passed=passed,
                note=note,
            )
        )
        if passed:
            return VariantResult(
                rule=rule,
                final_text=text,
                final_char_count=char_count,
                status="ai_approved",
                attempts=attempts,
            )
        feedback = note  # carry the critique into the next attempt

    # Every LLM attempt failed. Truncate the shortest usable draft; if the LLM never
    # produced text (all attempts errored), truncate the raw product source instead.
    usable = [a for a in attempts if a.source == "llm" and a.text]
    fallback_source = min(usable, key=lambda a: a.char_count).text if usable else product_source

    truncated = truncate_to_limit(fallback_source, rule.max_chars)
    char_count = count_chars(truncated)
    attempts.append(
        Attempt(
            attempt_number=len(attempts) + 1,
            source="fallback",
            text=truncated,
            char_count=char_count,
            max_chars=rule.max_chars,
            passed=fits(truncated, rule),  # True by construction of truncate_to_limit
            note="Deterministic word-boundary truncation after all LLM attempts failed.",
        )
    )
    return VariantResult(
        rule=rule,
        final_text=truncated,
        final_char_count=char_count,
        status="fallback_truncated",
        attempts=attempts,
    )


def run_all_platforms(
    *,
    product_source: str,
    rules: list[PlatformRule],
    generate: VariantGenerator,
    judge: DistinctnessJudge | None = None,
) -> list[VariantResult]:
    """Run the critique loop for every platform, then assess cross-platform distinctness.

    Each platform's variant is produced independently and its character-limit guarantee is
    unchanged. Only afterwards — the one point where all finals coexist — do we assess
    distinctness and attach the verdicts to each result. Two independent, parallel signals
    run here:
      * the always-on lexical ``assess_distinctness`` (shared words), and
      * the optional semantic ``assess_with_judge`` (shared meaning), run once over all three
        finals when a ``judge`` is injected — one call per product, never per attempt.
    Both are pure, read-only passes over the finished text; neither alters a variant or its
    limit compliance, and a judge failure falls back to the lexical signal without breaking.
    """
    results = [
        run_critique_loop(product_source=product_source, rule=rule, generate=generate)
        for rule in rules
    ]
    lexical = assess_distinctness(results, product_source)
    judged = assess_with_judge(results, product_source, judge)
    return [
        replace(
            result,
            distinct=lex.distinct,
            distinct_note=lex.note,
            judge_distinct=jud.distinct,
            judge_note=jud.note,
        )
        for result, lex, jud in zip(results, lexical, judged)
    ]
