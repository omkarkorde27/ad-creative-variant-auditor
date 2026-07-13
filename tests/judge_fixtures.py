"""Human-labeled ground-truth fixtures for validating the semantic distinctness judge.

Per the plan's Q4 validation gate: the lexical metric's own flaw was that it was never
checked against human judgment. The judge does not get to repeat that mistake — this module
is the ground truth ``tests/harness_judge.py`` scores the live judge against before its
verdict is trusted as anything more than a co-equal, human-checkable signal.

Every fixture is FIXED — real captured output or a hand-constructed control case, never
freshly generated — so judge behavior is isolated from generation nondeterminism. Fixtures
are built from the SHIPPED ``data/platform_rules.json`` angles (via ``load_platform_rules``)
rather than restating angle text here, so they stay in sync with the live rules rather than
silently drifting from them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from service.critique_loop import VariantResult  # noqa: E402
from service.rules import load_platform_rules  # noqa: E402

_RULES = {rule.name: rule for rule in load_platform_rules()}


def _variant(platform: str, text: str) -> VariantResult:
    """Build a fixture ``VariantResult`` using the shipped rule for ``platform``.

    Reusing the real ``PlatformRule`` (label, style, angle) means a fixture always reflects
    the angle text actually in production, so it can't silently go stale if
    ``platform_rules.json`` is edited without updating this file.
    """
    rule = _RULES[platform]
    return VariantResult(
        rule=rule,
        final_text=text,
        final_char_count=len(text),
        status="ai_approved",
        attempts=[],
    )


@dataclass(frozen=True)
class JudgeFixture:
    """One human-labeled ground-truth case.

    ``expected_distinct`` / ``expected_similar`` encode the human verdict (the platforms that
    SHOULD be flagged as colliding; empty when the trio is genuinely distinct). ``origin``
    documents where the case came from, so a failing fixture traces back to real evidence.
    """

    name: str
    product_source: str
    variants: list[VariantResult]
    expected_distinct: bool
    expected_similar: frozenset[str]
    origin: str


FIXTURES: list[JudgeFixture] = [
    JudgeFixture(
        name="drone_paraphrase_collision",
        product_source=(
            "Snaptain S5C Foldable Drone with 1080P HD Camera. Beginner-friendly with "
            "altitude hold, one-key takeoff/landing, and headless mode for easy control."
        ),
        variants=[
            _variant("search_ad", "Beginner Drone Ready Today"),
            _variant(
                "social_ad",
                "Take flight and feel like a pro from day one — no experience needed! 🚁",
            ),
            _variant(
                "display_banner", "Snaptain S5C — built for beginners who want reliable flight."
            ),
        ],
        expected_distinct=False,
        expected_similar=frozenset({"search_ad", "display_banner"}),
        origin=(
            "Real system output, currently marked DISTINCT by the lexical check (only "
            "'beginner' shared after stemming/product-name exclusion, against small token "
            "sets). Search Ad and Display Banner make the identical 'easy/safe for "
            "beginners' claim in different words — the exact failure mode the judge exists "
            "to catch. Social Ad's lifestyle framing is a genuinely different angle and must "
            "NOT be flagged."
        ),
    ),
    JudgeFixture(
        name="jeans_shared_noun_not_collision",
        product_source="Magnus Straight Leg Jeans\nAGOLDE\n$238.00 Current Price",
        variants=[
            _variant("search_ad", "True-to-Size Straight Leg"),
            _variant(
                "social_ad",
                "That '90s vibe that just *feels* right. Straight legs, pure comfort, "
                "zero regrets. 👖✨",
            ),
            _variant(
                "display_banner",
                "AGOLDE's regenerative cotton straight leg. Trusted by Nordstrom. "
                "Free returns, always.",
            ),
        ],
        expected_distinct=True,
        expected_similar=frozenset(),
        origin=(
            "Real system output. Originally a lexical FALSE POSITIVE (Search/Display both "
            "flagged purely for sharing 'straight leg', the product's own category name) "
            "before the product-term exclusion fix. The judge must not repeat that mistake: "
            "sharing a product-name word is not a strategic collision when the actual "
            "arguments — urgent fact vs. lifestyle vs. brand trust — genuinely differ."
        ),
    ),
    JudgeFixture(
        name="fits_your_day_observed_collision",
        product_source=(
            "A cozy, all-day layer that moves with you — from morning commute to evening "
            "errands, indoors or out."
        ),
        variants=[
            _variant("search_ad", "fits your whole day"),
            _variant("social_ad", "fits your whole life... better"),
            _variant("display_banner", "fits your full day"),
        ],
        expected_distinct=False,
        expected_similar=frozenset({"search_ad", "social_ad", "display_banner"}),
        origin=(
            "The original observed failure that motivated the entire distinctness effort: "
            "one idea rendered at three lengths. Both signals should agree here — this is "
            "the case the lexical metric was explicitly built to catch, and the judge must "
            "catch it too."
        ),
    ),
    JudgeFixture(
        name="clean_distinct_fleece",
        product_source=(
            "Recycled polyester fleece jacket, warm-weight, for casual everyday wear or "
            "outdoor layering. Machine washable, Fair Trade Certified factory."
        ),
        variants=[
            _variant("search_ad", "Warmest Recycled Fleece"),
            _variant("social_ad", "Cozy vibes all season long, friend 🧡"),
            _variant("display_banner", "Sustainable warmth for outdoor adventurers."),
        ],
        expected_distinct=True,
        expected_similar=frozenset(),
        origin=(
            "Hand-constructed control case: no shared content words and three clearly "
            "different strategic angles (product fact / lifestyle feeling / audience-fit "
            "positioning). The 'nothing to catch' baseline — as important as the collision "
            "cases, since a judge that flags everything would ace the collision fixtures "
            "while being useless."
        ),
    ),
    JudgeFixture(
        name="clean_distinct_necklace",
        product_source=(
            "Black Onyx Stone & 18K Gold Plating pendant necklace. Adjustable chain, "
            "hypoallergenic, gift box included."
        ),
        variants=[
            _variant("search_ad", "18K Gold Onyx Necklace"),
            _variant("social_ad", "The piece that finishes every outfit ✨"),
            _variant(
                "display_banner", "Hypoallergenic gold-plated jewelry, gift-ready in every box."
            ),
        ],
        expected_distinct=True,
        expected_similar=frozenset(),
        origin=(
            "Second hand-constructed control case, a different product category (jewelry) "
            "than the fleece control, to reduce the chance the judge is pattern-matching one "
            "product type rather than genuinely evaluating strategic distinctness."
        ),
    ),
]
