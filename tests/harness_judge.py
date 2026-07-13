"""Validation harness for the semantic distinctness judge (agent/judge.py) — the Q4 gate.

This is a *runner*, not a unit test (named ``harness_*`` so pytest's default collection
ignores it). ``tests/test_judge.py`` proves the judge is correctly *plumbed* (a stub verdict
maps to the right fields, a failure falls back cleanly); it cannot prove the judge is
*right*. That is what this harness measures, matching the plan's Q4 validation gate:

  * ``--mode fixtures`` (default): runs the judge against ``tests/judge_fixtures.py`` — a
    small, FIXED, human-labeled set (the real drone paraphrase collision, the real jeans
    shared-noun non-collision, the original observed "fits your day" collision, plus two
    hand-constructed clean control cases). Reports accuracy against the human labels and, in
    ``--live`` mode, verdict CONSISTENCY across repeated calls on identical input — a judge
    that flips its answer on the same input cannot be trusted even when it's sometimes right.

  * ``--mode regression``: runs the FULL live pipeline (generation + both distinctness
    signals) over ``data/regression_set/``, and reports the lexical-vs-judge AGREEMENT rate
    plus every disagreement, for human review. This is where a systematic error in either
    signal would surface on fresh, real text rather than the fixed fixtures above.

Ship-gate stated in the plan: "looks more sophisticated" is not evidence the judge should be
trusted — passing the fixtures (especially drone and jeans) is.

Usage::

    python tests/harness_judge.py                              # stub, free, plumbing check
    python tests/harness_judge.py --live --repeats 3            # real judge vs fixtures
    python tests/harness_judge.py --mode regression --live --n 1   # real pipeline, agreement
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
# tests/ has no __init__.py (it's a script directory, not a package, matching
# harness_convergence.py's own pattern) — put both the repo root (for service/agent) and
# tests/ itself (for judge_fixtures, harness_convergence) on the path.
for path in (_REPO_ROOT, _TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from service.critique_loop import run_all_platforms  # noqa: E402
from service.distinctness import DistinctnessJudge, JudgeVerdict  # noqa: E402
from service.rules import load_platform_rules  # noqa: E402

from harness_convergence import _make_generator, load_products  # noqa: E402
from judge_fixtures import FIXTURES  # noqa: E402

# --- Named constants ----------------------------------------------------------------

DEFAULT_CONSISTENCY_REPEATS: int = 3  # K repeats per fixture in --live fixtures mode
DEFAULT_REGRESSION_TRIALS: int = 1  # trials per product in --mode regression (live, costly)


class StubJudge:
    """Deterministic ``DistinctnessJudge`` for free plumbing checks: always says distinct.

    Not meant to pass the fixtures (most expect a collision) — it exists to prove the harness
    itself runs end to end without spending API budget.
    """

    def __call__(self, *, variants: list[Any], product_source: str) -> JudgeVerdict:
        return JudgeVerdict(distinct=True, rationale="stub: always distinct", similar_platforms=())


def _make_judge(live: bool) -> DistinctnessJudge:
    if not live:
        return StubJudge()
    from agent.judge import AnthropicDistinctnessJudge  # noqa: PLC0415 - lazy, live-only

    return AnthropicDistinctnessJudge()


# --- Mode: fixtures -------------------------------------------------------------------


def run_fixture_mode(*, live: bool, repeats: int) -> None:
    """Score the judge against every labeled fixture; in --live mode, also check consistency."""
    judge = _make_judge(live)
    mode = "LIVE (real API)" if live else "STUB (no API)"
    print(f"\nJudge fixture validation — {mode}\n")

    distinct_correct = 0
    exact_set_correct = 0
    n_repeats = repeats if live else 1  # the stub is deterministic; repeats add no signal
    # Raw call attempt/failure counts — the same measurement that originally surfaced the
    # free-text JSON parsing bug (roughly 1 in 5 calls failed before the structured-output fix).
    raw_attempts = 0
    raw_failures = 0

    for fixture in FIXTURES:
        verdicts: list[JudgeVerdict] = []
        for _ in range(n_repeats):
            raw_attempts += 1
            try:
                verdicts.append(
                    judge(variants=fixture.variants, product_source=fixture.product_source)
                )
            except Exception as exc:  # noqa: BLE001 - report and continue, don't crash the run
                raw_failures += 1
                print(f"  [{fixture.name}] judge call raised: {exc!r}")

        if not verdicts:
            print(f"  [{fixture.name}] SKIPPED — every judge call failed")
            continue

        first = verdicts[0]
        got_similar = frozenset(first.similar_platforms)
        distinct_ok = first.distinct == fixture.expected_distinct
        set_ok = got_similar == fixture.expected_similar
        distinct_correct += int(distinct_ok)
        exact_set_correct += int(set_ok)

        print(f"  [{fixture.name}] {'OK' if distinct_ok else 'WRONG'}")
        print(f"    distinct={first.distinct} (expected {fixture.expected_distinct})")
        print(
            f"    similar_platforms={sorted(got_similar)} "
            f"(expected {sorted(fixture.expected_similar)})"
        )
        print(f"    rationale: {first.rationale}")

        if len(verdicts) > 1:
            distinct_values = {v.distinct for v in verdicts}
            if len(distinct_values) > 1:
                print(f"    ⚠ verdict FLIPPED across {len(verdicts)} repeats: "
                      f"{[v.distinct for v in verdicts]}")
            else:
                print(f"    consistent across {len(verdicts)} repeats")

    n = len(FIXTURES)
    print(f"\ndistinct-label accuracy: {distinct_correct}/{n}")
    print(f"exact similar_platforms accuracy: {exact_set_correct}/{n}")
    raw_failure_rate = raw_failures / raw_attempts if raw_attempts else 0.0
    print(
        f"\nraw judge call failure rate: {raw_failure_rate:.0%} "
        f"({raw_failures}/{raw_attempts} calls raised — parse errors, timeouts, refusals, etc.)"
    )
    print(
        "\nShip-gate: drone_paraphrase_collision and jeans_shared_noun_not_collision "
        "must both be OK above before the judge is trusted as anything more than a "
        "co-equal, shown-alongside signal."
    )


# --- Mode: regression (live pipeline, lexical vs judge agreement) ---------------------


def run_regression_mode(*, live: bool, trials: int, out_dir: Path) -> None:
    """Run the full pipeline over the regression set; report lexical-vs-judge agreement."""
    judge = _make_judge(live)
    generate = _make_generator(live)
    rules = load_platform_rules()
    products = load_products(_REPO_ROOT / "data" / "regression_set")

    mode = "LIVE (real API)" if live else "STUB (no API)"
    print(f"\nJudge/lexical agreement — {mode}, {len(products)} product(s) × {trials} trial(s)\n")

    agree = 0
    total = 0
    disagreements: list[dict[str, Any]] = []
    # One judge call per product×trial (never per platform — see run_all_platforms). Track
    # fallback at that call granularity, not per-variant, so 1 failed call reads as 1 failure
    # rather than 3 (all three variants uniformly get judge_distinct=None on a fallback).
    judge_calls = 0
    judge_fallbacks = 0

    for product_name, product_source in products:
        for trial in range(trials):
            results = run_all_platforms(
                product_source=product_source, rules=rules, generate=generate, judge=judge
            )
            judge_calls += 1
            if all(result.judge_distinct is None for result in results):
                judge_fallbacks += 1

            for result in results:
                if result.distinct is None or result.judge_distinct is None:
                    continue  # one signal unavailable this run -> not comparable
                total += 1
                if result.distinct == result.judge_distinct:
                    agree += 1
                else:
                    disagreements.append(
                        {
                            "product": product_name,
                            "trial": trial,
                            "platform": result.rule.name,
                            "text": result.final_text,
                            "lexical_distinct": result.distinct,
                            "lexical_note": result.distinct_note,
                            "judge_distinct": result.judge_distinct,
                            "judge_note": result.judge_note,
                        }
                    )

    fallback_rate = judge_fallbacks / judge_calls if judge_calls else 0.0
    print(
        f"judge fallback rate: {fallback_rate:.0%} "
        f"({judge_fallbacks}/{judge_calls} product calls fell back to lexical-only)"
    )
    rate = agree / total if total else 0.0
    print(f"agreement rate: {rate:.0%} ({agree}/{total} per-variant comparisons)")
    print(f"disagreements: {len(disagreements)}\n")
    for d in disagreements:
        print(f"  [{d['product']} / {d['platform']}] lexical={d['lexical_distinct']} judge={d['judge_distinct']}")
        print(f"    text: {d['text']!r}")
        print(f"    judge note: {d['judge_note']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "disagreements.json"
    out_path.write_text(json.dumps(disagreements, indent=2), encoding="utf-8")
    print(f"\ndisagreements written to: {out_path}")


# --- Driver -----------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["fixtures", "regression"], default="fixtures")
    parser.add_argument(
        "--live", action="store_true", help="use the real Anthropic judge (costs money)"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_CONSISTENCY_REPEATS,
        help="fixtures mode: consistency repeats per fixture (live only)",
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_REGRESSION_TRIALS, help="regression mode: trials per product"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "variant_auditor_judge",
        help="regression mode: output dir for disagreements.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.mode == "fixtures":
        run_fixture_mode(live=args.live, repeats=args.repeats)
    else:
        run_regression_mode(live=args.live, trials=args.n, out_dir=args.out)


if __name__ == "__main__":
    main()
