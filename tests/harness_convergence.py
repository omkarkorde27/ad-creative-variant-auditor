"""Offline convergence-measurement harness for the critique loop.

This is a *runner*, not a unit test. It is named ``harness_*`` (not ``test_*``) so
pytest's default collection ignores it: it runs the full ``run_all_platforms`` pipeline
many times and, in ``--live`` mode, calls the real Anthropic API — which costs money and
is non-deterministic. Its purpose is to replace manual spot-checks of a handful of runs
with aggregate metrics that can actually adjudicate whether a change helped.

What it measures, per platform, across N trials (all extracted from the existing
``VariantResult.attempts`` trail — no new data shape is invented):
  * fallback rate           — fraction of runs ending ``fallback_truncated``.
  * first-attempt overshoot — ``attempts[0].char_count / max_chars`` (p50 / p90).
  * attempts-to-pass        — among ``ai_approved`` runs, how many LLM drafts it took.
  * convergence trajectory  — the per-attempt ``char_count`` sequence, classified as
                              pass-first / shrinking-pass / shrinking-insufficient /
                              flat-and-far / oscillating.

Usage::

    python tests/harness_convergence.py                 # stub generator, free, plumbing check
    python tests/harness_convergence.py --live --n 30   # real API, costs money

To compare code versions (baseline vs a candidate change), run ``--live`` on the current
tree, apply the change, and run ``--live`` again with the same ``--n`` and ``--seed``;
compare the two printed tables. ``git stash`` recovers the pre-change tree for a clean
baseline if the change was already applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# This file lives in tests/; put the repo root on the path so `service`/`agent` import
# whether it is run as `python tests/harness_convergence.py` or `python -m tests.…`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from service.critique_loop import (  # noqa: E402 - after sys.path shim
    Attempt,
    VariantGenerator,
    VariantResult,
    run_all_platforms,
)
from service.rules import PlatformRule, load_platform_rules  # noqa: E402

# --- Named constants (no magic numbers) -------------------------------------------

DEFAULT_TRIALS: int = 8  # small by default so an accidental --live run is cheap
DEFAULT_SEED: int = 42  # reproducible stub trajectories / product sampling

# Input products. The default is the regression set (one .txt per product) when present,
# so distinctness collision rate is measured across a realistic corpus; a single-file path
# still works for a one-off measurement.
DEFAULT_REGRESSION_DIR: Path = _REPO_ROOT / "data" / "regression_set"
DEFAULT_PRODUCT_FILE: Path = _REPO_ROOT / "data" / "product_source.txt"

# Stub-only knobs. The stub reproduces the observed failure shape so the harness's own
# metrics and classifier can be validated without spending API budget. It is measurement
# infrastructure, NOT the shipped agent, so it is allowed to read rule.max_chars.
_STUB_FIRST_OVERSHOOT: dict[str, float] = {
    # First-draft length as a multiple of the limit: tight limits overshoot hugely
    # (the 270-vs-30 Search case), loose limits land near or under.
    "search_ad": 9.0,
    "display_banner": 1.4,
    "social_ad": 1.1,
}
_STUB_DEFAULT_OVERSHOOT: float = 2.0  # for any platform not named above (e.g. a 4th)
_STUB_RETRY_SHRINK: float = 0.5  # each retry halves the previous draft's length

# Classifier thresholds.
_UPTICK_FRACTION: float = 0.05  # a rise > 5% of the limit counts as "went up"
_FLAT_DROP_FRACTION: float = 0.25  # total drop < 25% of the first draft == barely moved


# --- Deterministic stub generator (free; validates plumbing + the classifier) -----


class StubGenerator:
    """A ``VariantGenerator`` that fakes the observed convergence shape, no API needed.

    A fresh instance is created per run so each run has an independent call counter per
    platform, yielding a deterministic shrinking trajectory: a large first draft (scaled
    off the limit) that halves on every retry. Tight limits never reach the limit within
    the budget (exercising the fallback path); loose limits converge in one or two steps.
    """

    def __init__(self) -> None:
        self._calls: dict[str, int] = {}

    def __call__(
        self,
        *,
        product_source: str,
        rule: PlatformRule,
        feedback: str | None = None,
    ) -> str:
        n = self._calls.get(rule.name, 0) + 1
        self._calls[rule.name] = n
        factor = _STUB_FIRST_OVERSHOOT.get(rule.name, _STUB_DEFAULT_OVERSHOOT)
        length = round(rule.max_chars * factor * (_STUB_RETRY_SHRINK ** (n - 1)))
        length = max(length, 1)
        # Build filler text of exactly `length` characters; content is irrelevant, only
        # len() matters to the loop.
        filler = ("value benefit deal offer now buy save more today ") * length
        return filler[:length]


# --- Per-run record + aggregation --------------------------------------------------


@dataclass(frozen=True)
class RunMetrics:
    """One platform's outcome within a single trial, distilled from its attempt trail."""

    platform: str
    max_chars: int
    status: str
    first_overshoot_ratio: float | None  # None if the first attempt errored (no draft)
    attempts_to_pass: int | None  # None if it fell back
    llm_trajectory: tuple[int, ...]  # char_count of each LLM draft, in order
    trajectory_class: str


def _llm_char_counts(attempts: list[Attempt]) -> tuple[int, ...]:
    """The char_count of each LLM draft, in order (errors and the fallback excluded)."""
    return tuple(a.char_count for a in attempts if a.source == "llm")


def classify_trajectory(trajectory: tuple[int, ...], max_chars: int) -> str:
    """Label a per-run LLM char_count sequence — the core diagnostic distinction.

    Discriminates the mechanisms that manual spot-checks blur together: whether retries
    shrink toward the limit, wander (oscillate), or sit far away barely moving.
    """
    if not trajectory:
        return "no-llm-draft"
    if len(trajectory) == 1:
        return "pass-first" if trajectory[0] <= max_chars else "single-over"

    deltas = [trajectory[i + 1] - trajectory[i] for i in range(len(trajectory) - 1)]
    went_up = any(d > _UPTICK_FRACTION * max_chars for d in deltas)
    reached = trajectory[-1] <= max_chars
    total_drop = trajectory[0] - trajectory[-1]

    if went_up:
        return "oscillating"
    if reached:
        return "shrinking-pass"
    if total_drop < _FLAT_DROP_FRACTION * trajectory[0]:
        return "flat-and-far"
    return "shrinking-insufficient"


def metrics_for_result(result: VariantResult) -> RunMetrics:
    """Distill one ``VariantResult`` into the metrics we aggregate across trials."""
    attempts = result.attempts
    first = attempts[0] if attempts else None
    first_ratio = (
        first.char_count / result.rule.max_chars
        if first is not None and first.source == "llm"
        else None
    )
    trajectory = _llm_char_counts(attempts)
    attempts_to_pass = len(trajectory) if result.status == "ai_approved" else None
    return RunMetrics(
        platform=result.rule.name,
        max_chars=result.rule.max_chars,
        status=result.status,
        first_overshoot_ratio=first_ratio,
        attempts_to_pass=attempts_to_pass,
        llm_trajectory=trajectory,
        trajectory_class=classify_trajectory(trajectory, result.rule.max_chars),
    )


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in [0, 100]); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


def summarize(platform: str, rows: list[RunMetrics]) -> dict[str, object]:
    """Aggregate all trials for one platform into the headline metrics."""
    n = len(rows)
    fallbacks = sum(1 for r in rows if r.status == "fallback_truncated")
    overshoots = [r.first_overshoot_ratio for r in rows if r.first_overshoot_ratio is not None]
    passes = [r.attempts_to_pass for r in rows if r.attempts_to_pass is not None]
    classes = Counter(r.trajectory_class for r in rows)
    return {
        "platform": platform,
        "max_chars": rows[0].max_chars if rows else None,
        "trials": n,
        "fallback_rate": fallbacks / n if n else None,
        "overshoot_p50": _percentile(overshoots, 50),
        "overshoot_p90": _percentile(overshoots, 90),
        "attempts_to_pass_mean": statistics.fmean(passes) if passes else None,
        "attempts_to_pass_median": statistics.median(passes) if passes else None,
        "trajectory_classes": dict(classes),
    }


# --- Output ------------------------------------------------------------------------


def _fmt(value: object) -> str:
    """Compact cell formatter for the console table."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def print_table(summaries: list[dict[str, object]]) -> None:
    """Print the per-platform aggregate table to stdout."""
    cols = [
        ("platform", "platform"),
        ("max_chars", "limit"),
        ("trials", "N"),
        ("fallback_rate", "fallback%"),
        ("overshoot_p50", "over_p50"),
        ("overshoot_p90", "over_p90"),
        ("attempts_to_pass_mean", "att_mean"),
    ]
    header = "  ".join(f"{title:>12}" for _key, title in cols)
    print(header)
    print("-" * len(header))
    for s in summaries:
        print("  ".join(f"{_fmt(s[key]):>12}" for key, _title in cols))
    print()
    for s in summaries:
        print(f"{s['platform']}: trajectory classes -> {s['trajectory_classes']}")


def write_outputs(
    out_dir: Path,
    per_attempt: list[dict[str, object]],
    summaries: list[dict[str, object]],
) -> tuple[Path, Path]:
    """Persist the per-attempt CSV and the aggregate JSON; return both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_attempt.csv"
    json_path = out_dir / "summary.json"

    fieldnames = [
        "product",
        "run",
        "platform",
        "max_chars",
        "attempt_number",
        "source",
        "char_count",
        "passed",
        "over_by",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_attempt)

    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return csv_path, json_path


# --- Driver ------------------------------------------------------------------------


def load_products(path: Path) -> list[tuple[str, str]]:
    """Load one or many product sources as ``(name, text)`` pairs.

    A directory yields every ``*.txt`` in it (sorted, keyed by filename stem) — the
    regression set. A single file yields one product keyed by its stem. Empty files are
    skipped. Raises ``FileNotFoundError`` if nothing usable is found, so a mis-pointed path
    fails loudly rather than measuring nothing.
    """
    if path.is_dir():
        products = [
            (f.stem, f.read_text(encoding="utf-8").strip())
            for f in sorted(path.glob("*.txt"))
        ]
        products = [(name, text) for name, text in products if text]
        if not products:
            raise FileNotFoundError(f"No non-empty .txt product files in {path}")
        return products

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise FileNotFoundError(f"Product file {path} is empty")
    return [(path.stem, text)]


def _make_generator(live: bool) -> VariantGenerator:
    """Return the real Anthropic generator (``--live``) or the free deterministic stub.

    The real generator is imported and constructed lazily so a stub run needs neither the
    ``langchain_anthropic`` dependency loaded for API validation nor an API key present.
    """
    if not live:
        return StubGenerator()
    from agent.generator import AnthropicVariantGenerator  # noqa: PLC0415 - lazy, live-only

    return AnthropicVariantGenerator()


def run_harness(
    *, trials: int, live: bool, out_dir: Path, seed: int, product_path: Path
) -> list[dict[str, object]]:
    """Run the pipeline ``trials`` times per product, aggregate, persist, and return summaries.

    Convergence metrics are pooled across every product/trial (the pipeline's behavior over
    a realistic corpus), while distinctness collision rate is reported both per product and
    overall — the per-product breakdown is the signal that gates the phase-2 corrective-loop
    decision.
    """
    random.seed(seed)  # only the stub/product sampling is stochastic; keeps runs reproducible
    products = load_products(product_path)
    rules = load_platform_rules()

    per_platform: dict[str, list[RunMetrics]] = {rule.name: [] for rule in rules}
    per_attempt_rows: list[dict[str, object]] = []

    # Cross-platform distinctness telemetry: per product, how many runs produced at least one
    # too-similar variant, plus which platforms get flagged. Gathered by reusing this run.
    collision_by_product: dict[str, tuple[int, int]] = {}
    nondistinct_by_platform: Counter[str] = Counter()

    for product_name, product_source in products:
        runs_with_collision = 0
        for run_idx in range(trials):
            # A fresh generator per run so the stub's per-platform counters reset; harmless
            # for the live generator (its chains are stateless across calls).
            generate = _make_generator(live)
            results = run_all_platforms(
                product_source=product_source, rules=rules, generate=generate
            )
            if any(result.distinct is False for result in results):
                runs_with_collision += 1
            for result in results:
                if result.distinct is False:
                    nondistinct_by_platform[result.rule.name] += 1
                per_platform[result.rule.name].append(metrics_for_result(result))
                for attempt in result.attempts:
                    per_attempt_rows.append(
                        {
                            "product": product_name,
                            "run": run_idx,
                            "platform": result.rule.name,
                            "max_chars": attempt.max_chars,
                            "attempt_number": attempt.attempt_number,
                            "source": attempt.source,
                            "char_count": attempt.char_count,
                            "passed": attempt.passed,
                            "over_by": max(0, attempt.char_count - attempt.max_chars),
                        }
                    )
        collision_by_product[product_name] = (runs_with_collision, trials)

    summaries = [summarize(rule.name, per_platform[rule.name]) for rule in rules]
    csv_path, json_path = write_outputs(out_dir, per_attempt_rows, summaries)

    mode = "LIVE (real API)" if live else "STUB (no API)"
    print(f"\nConvergence harness — {mode}, {len(products)} product(s) × {trials} trials\n")
    print_table(summaries)

    total_collisions = sum(c for c, _ in collision_by_product.values())
    total_runs = trials * len(products)
    overall_rate = total_collisions / total_runs if total_runs else 0.0
    print(
        f"\ndistinctness collision rate: overall {overall_rate:.0%} "
        f"({total_collisions}/{total_runs} product-runs had >=1 too-similar variant)"
    )
    print("  by product:")
    for name, (collisions, runs) in collision_by_product.items():
        rate = collisions / runs if runs else 0.0
        print(f"    {name:16} {rate:>4.0%} ({collisions}/{runs})")
    if nondistinct_by_platform:
        flagged = ", ".join(f"{name}={count}" for name, count in nondistinct_by_platform.most_common())
        print(f"  flagged-too-similar counts by platform: {flagged}")

    print(f"\nper-attempt rows: {csv_path}\naggregate summary: {json_path}")
    return summaries


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=DEFAULT_TRIALS, help="number of trials")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use the real Anthropic generator (costs money); default is the free stub",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "variant_auditor_convergence",
        help="output directory for the per-attempt CSV and summary JSON",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed")
    parser.add_argument(
        "--product",
        type=Path,
        default=None,
        help=(
            "product source: a .txt file, or a directory of .txt files (default: the "
            f"{DEFAULT_REGRESSION_DIR.name}/ directory if present, else product_source.txt)"
        ),
    )
    return parser.parse_args(argv)


def _resolve_product_path(explicit: Path | None) -> Path:
    """Pick the product path: an explicit arg wins, else the regression set, else the file."""
    if explicit is not None:
        return explicit
    return DEFAULT_REGRESSION_DIR if DEFAULT_REGRESSION_DIR.is_dir() else DEFAULT_PRODUCT_FILE


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_harness(
        trials=args.n,
        live=args.live,
        out_dir=args.out,
        seed=args.seed,
        product_path=_resolve_product_path(args.product),
    )


if __name__ == "__main__":
    main()
