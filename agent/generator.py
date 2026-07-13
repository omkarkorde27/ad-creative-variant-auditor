"""The LLM generation layer — an implementation of the ``VariantGenerator`` Protocol.

Design commitments (see CLAUDE.md, "Agent Layer Constraints"):
  * This module NEVER counts characters, checks length, or references ``rule.max_chars``.
    ``service/`` owns all validation and the retry/fallback guarantee. The agent's only
    job is to produce ad copy and, on a retry, obey the critique it is handed.
  * Compliance with a platform's character limit comes from the *critique loop*, not from
    a prompt instruction. The first-attempt prompt therefore leans on ``rule.style`` and a
    "shortest phrasing" nudge; the exact limit reaches the model only via ``feedback``,
    which the loop assembles and this module includes verbatim.
  * An LCEL chain (``prompt | model | parser``) — single-shot generation per call, not a
    multi-tool agent. Two instances of the *same* chain shape are built once per generator,
    differing only in sampling temperature: a creative one for the first attempt and a
    near-deterministic one for retries, so corrections converge instead of drifting. This is
    still single-shot generation, not multi-step tool use.
  * No error handling here: ``run_critique_loop`` catches any exception a ``generate`` call
    raises, logs it as an ``error`` attempt, and still falls back to deterministic
    truncation. So API/auth failures must propagate rather than be swallowed.
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from service.rules import PlatformRule

# --- Model configuration (named, no magic numbers) --------------------------------

# Small/cheap Claude model per CLAUDE.md; ad copy is a short, well-scoped task.
DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"
# First-attempt temperature: creative variety is welcome when there is no correction yet.
DEFAULT_TEMPERATURE: float = 0.4
# Retry temperature: once a length correction exists, the job is convergence, not
# exploration. Low sampling makes the model reliably *shorten* the rejected draft
# instead of re-sampling to an even longer one (the observed 84->127 divergence).
# Kept low but NON-zero: at exactly 0.0 a retry that still overshoots reproduces an
# identical draft, wasting the remaining attempts before fallback; a small value lets
# each retry make a genuinely different shortening attempt while staying obedient.
RETRY_TEMPERATURE: float = 0.2
# Outputs are at most ~125 characters (~40 tokens); this caps cost/runaway with headroom.
DEFAULT_MAX_TOKENS: int = 256


# --- Prompt (the exact text sent to the model) ------------------------------------

SYSTEM_PROMPT: str = (
    "You are an expert direct-response advertising copywriter. You write one ad variant "
    "for a single specific placement, using only the product information provided.\n\n"
    "Rules for your output:\n"
    "- Return ONLY the ad copy itself — no preamble, no quotation marks, no labels, no "
    "explanation.\n"
    "- Write for the named placement and follow its style guidance exactly.\n"
    "- Lead with ONE distinct angle. If an angle to take is given, commit to it. Otherwise "
    "choose the single most compelling angle for THIS placement, drawn from the product "
    "information — do not default to restating the same price, discount, or rating facts.\n"
    "- Make every word earn its place; prefer the shortest phrasing that still sells.\n"
    "- Base every claim on the product information; never invent facts, prices, or features.\n"
    "- When a correction about length is present, meeting the stated character limit is "
    "non-negotiable. This does not mean abandoning your assigned angle: a short rewrite "
    "must still commit to that angle, not revert to the most generic, most obviously-"
    "present facts in the product information. Length and angle adherence are both "
    "required; only extra cleverness or completeness may be sacrificed. Aim to land "
    "comfortably UNDER the limit, not exactly at it."
)

HUMAN_PROMPT: str = (
    "Placement: {label}\n"
    "Style guidance: {style}\n"
    "{angle_section}"
    "\nProduct information:\n"
    "{product_source}\n"
    "{feedback_section}\n"
    "Write the {label} ad copy now."
)


def _format_angle(angle: str) -> str:
    if not angle:
        return ""
    return (
        f"\nAssigned strategic angle (follow this, not your own instinct about what's "
        f"most compelling): {angle}\n"
        f"Do not lead with a different angle just because it seems more salient in the "
        f"product information. If, and only if, the product information genuinely gives "
        f"you nothing to work with for this angle, fall back to the strongest "
        f"genuinely-present differentiator instead — but that is the exception, not the "
        f"default.\n"
    )


def _format_feedback(feedback: str | None) -> str:
    """Render the optional critique block for the prompt.

    ``None`` (first attempt) yields an empty string. Otherwise the loop's critique is
    included **verbatim** so the model corrects against the same text the loop produced.
    The agent-owned lead-in is deliberately forceful: it makes fitting the limit the
    dominant goal so the qualitative style guidance stops competing with the numeric
    constraint on retries. It is kept operation-neutral — it does not presume the fix is
    editing the prior draft — because ``service`` decides per overage whether to ask for a
    trim or a fresh, much shorter rewrite, and this lead-in must not fight that choice. The
    critique's exact character numbers come from ``service/``; this module never computes
    or parses them.
    """
    if feedback is None:
        return ""
    return (
        "\nYour previous draft was rejected for being too long. Fitting the character "
        "limit is now your single most important goal — more important than style, "
        "completeness, or cleverness. Follow the correction below exactly, whether it asks "
        f"you to trim the previous draft or to write a shorter fresh one:\n{feedback}\n"
    )


def _build_chain(model: str, temperature: float, max_tokens: int) -> Any:
    """Build the single LCEL chain: prompt | model | string parser.

    ``ChatAnthropic`` reads ``ANTHROPIC_API_KEY`` from the environment (loaded via
    ``load_dotenv`` by the caller). Returns a ``Runnable`` whose ``.invoke`` maps the
    prompt variables to the model's raw text output.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    model_client = ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return prompt | model_client | StrOutputParser()


class AnthropicVariantGenerator:
    """A ``VariantGenerator`` backed by Anthropic Claude via LangChain.

    Instances satisfy ``service.critique_loop.VariantGenerator`` structurally (the
    ``__call__`` signature matches), so an instance can be passed directly as the
    ``generate`` argument of ``run_critique_loop`` / ``run_all_platforms``. Two chains are
    built once here (not per call): a creative first-attempt chain and a near-deterministic
    retry chain. ``__call__`` picks between them by whether a correction is present, so the
    model explores on attempt one but converges on the shortening critique afterwards.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        retry_temperature: float = RETRY_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        load_dotenv()  # make ANTHROPIC_API_KEY from .env visible to ChatAnthropic
        self._chain = _build_chain(model, temperature, max_tokens)
        self._retry_chain = _build_chain(model, retry_temperature, max_tokens)

    def __call__(
        self,
        *,
        product_source: str,
        rule: PlatformRule,
        feedback: str | None = None,
    ) -> str:
        """Generate one ad variant for ``rule``'s placement.

        Uses ``rule.label``, ``rule.style``, and ``rule.angle`` only — never
        ``rule.max_chars``. On a retry, ``feedback`` (the loop's critique) is included
        verbatim and the near-deterministic retry chain is used so the model reliably
        shortens toward the limit. Returns the model's text with surrounding whitespace
        stripped; this is cleanup, not validation — no length is measured or compared here.
        """
        # First attempt (no correction) -> creative chain; retry -> obedient chain.
        chain = self._retry_chain if feedback is not None else self._chain
        text: str = chain.invoke(
            {
                "label": rule.label,
                "style": rule.style,
                "angle_section": _format_angle(rule.angle),
                "product_source": product_source,
                "feedback_section": _format_feedback(feedback),
            }
        )
        return text.strip()
