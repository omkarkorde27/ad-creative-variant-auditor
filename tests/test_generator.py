"""Tests for the agent generation layer.

These do NOT hit the live Anthropic API. The LCEL chain is replaced with a fake so we
can assert the generator's contract without cost or network: (1) feedback formatting,
(2) that ``AnthropicVariantGenerator`` is a drop-in ``VariantGenerator`` (right prompt
payload, output stripped), and (3) that it plugs into ``run_critique_loop`` unchanged.
The one thing we never assert here is anything about ``max_chars`` — the agent must not
know or care about it; that guarantee is proven by the service-layer tests.
"""

from __future__ import annotations

from typing import Any

import pytest

import agent.generator as generator_module
from agent.generator import AnthropicVariantGenerator, _format_angle, _format_feedback
from service.critique_loop import run_critique_loop
from service.rules import PlatformRule

PRODUCT = (
    "The Aurora Desk Lamp is a smart, warm-to-cool tunable LED light with app control, "
    "USB-C charging, and a sleek aluminum body that fits any modern workspace."
)


@pytest.fixture
def social_rule() -> PlatformRule:
    return PlatformRule(
        name="social_ad", label="Social Ad", max_chars=125, style="punchy, emojis welcome"
    )


class _FakeChain:
    """Stand-in for the LCEL chain. Records the last invoke payload, returns canned text."""

    def __init__(self, output: str) -> None:
        self.output = output
        self.last_payload: dict[str, Any] | None = None

    def invoke(self, payload: dict[str, Any]) -> str:
        self.last_payload = payload
        return self.output


@pytest.fixture
def fake_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[AnthropicVariantGenerator, _FakeChain]:
    """An AnthropicVariantGenerator whose chain is a _FakeChain (no API, no dotenv needed)."""
    fake = _FakeChain(output="  Light your desk, any mood.  ")  # padded to prove .strip()
    monkeypatch.setattr(generator_module, "_build_chain", lambda *a, **k: fake)
    return AnthropicVariantGenerator(), fake


# --- _format_feedback -------------------------------------------------------------


def test_format_feedback_none_is_empty() -> None:
    assert _format_feedback(None) == ""


def test_format_feedback_includes_critique_verbatim() -> None:
    critique = "Your previous draft was 140 characters, 15 over the hard limit of 125."
    section = _format_feedback(critique)
    assert critique in section  # included verbatim, unaltered


# --- __call__ contract (drop-in VariantGenerator) ---------------------------------


def test_call_returns_stripped_chain_output(
    fake_generator: tuple[AnthropicVariantGenerator, _FakeChain], social_rule: PlatformRule
) -> None:
    gen, _fake = fake_generator
    result = gen(product_source=PRODUCT, rule=social_rule)
    assert result == "Light your desk, any mood."  # surrounding whitespace stripped


def test_call_builds_expected_prompt_payload(
    fake_generator: tuple[AnthropicVariantGenerator, _FakeChain], social_rule: PlatformRule
) -> None:
    gen, fake = fake_generator
    gen(product_source=PRODUCT, rule=social_rule)

    payload = fake.last_payload
    assert payload is not None
    assert set(payload) == {
        "label",
        "style",
        "angle_section",
        "product_source",
        "feedback_section",
    }
    assert payload["label"] == social_rule.label
    assert payload["style"] == social_rule.style
    assert payload["product_source"] == PRODUCT
    # First attempt: no feedback -> empty section. And max_chars never enters the payload.
    assert payload["feedback_section"] == ""
    assert str(social_rule.max_chars) not in "".join(str(v) for v in payload.values())


def test_call_passes_feedback_into_payload(
    fake_generator: tuple[AnthropicVariantGenerator, _FakeChain], social_rule: PlatformRule
) -> None:
    gen, fake = fake_generator
    critique = "Rewrite it to fit within 125 characters. Style: punchy"
    gen(product_source=PRODUCT, rule=social_rule, feedback=critique)

    assert fake.last_payload is not None
    assert critique in fake.last_payload["feedback_section"]


# --- Creative angle (distinctness) ------------------------------------------------


def test_format_angle_renders_explicit_and_empties() -> None:
    assert _format_angle("") == ""  # no angle -> model derives its own
    section = _format_angle("Lead with the strongest benefit")
    assert "Assigned strategic angle" in section
    assert "Lead with the strongest benefit" in section  # included verbatim


def test_explicit_angle_reaches_payload(
    fake_generator: tuple[AnthropicVariantGenerator, _FakeChain]
) -> None:
    gen, fake = fake_generator
    angled = PlatformRule(
        name="social_ad",
        label="Social Ad",
        max_chars=125,
        style="punchy",
        angle="make it playful",
    )
    gen(product_source=PRODUCT, rule=angled)

    assert fake.last_payload is not None
    assert "make it playful" in fake.last_payload["angle_section"]


def test_missing_angle_yields_empty_section(
    fake_generator: tuple[AnthropicVariantGenerator, _FakeChain], social_rule: PlatformRule
) -> None:
    gen, fake = fake_generator
    gen(product_source=PRODUCT, rule=social_rule)  # social_rule has no angle (defaults "")

    assert fake.last_payload is not None
    assert fake.last_payload["angle_section"] == ""


# --- Retry routing (convergence) --------------------------------------------------


def test_first_attempt_and_retry_use_different_chains(
    monkeypatch: pytest.MonkeyPatch, social_rule: PlatformRule
) -> None:
    """No feedback -> creative chain; feedback present -> near-deterministic retry chain."""
    first = _FakeChain(output="first-draft")
    retry = _FakeChain(output="retry-draft")
    # Constructor builds self._chain, then self._retry_chain — hand them out in that order.
    chains = iter([first, retry])
    monkeypatch.setattr(generator_module, "_build_chain", lambda *a, **k: next(chains))
    gen = AnthropicVariantGenerator()

    assert gen(product_source=PRODUCT, rule=social_rule) == "first-draft"
    assert first.last_payload is not None and retry.last_payload is None

    assert gen(product_source=PRODUCT, rule=social_rule, feedback="too long") == "retry-draft"
    assert retry.last_payload is not None


# --- Integration with the service critique loop -----------------------------------


def test_generator_drops_into_critique_loop(
    fake_generator: tuple[AnthropicVariantGenerator, _FakeChain], social_rule: PlatformRule
) -> None:
    gen, _fake = fake_generator
    result = run_critique_loop(product_source=PRODUCT, rule=social_rule, generate=gen)

    assert result.status == "ai_approved"  # 26-char output fits the 125 limit
    assert result.final_text == "Light your desk, any mood."
    assert result.final_char_count <= social_rule.max_chars
    assert result.attempts[0].source == "llm"
