"""The semantic distinctness judge — an implementation of the ``DistinctnessJudge`` Protocol.

Sibling of ``agent/generator.py``: the *mechanism* of the second, semantic distinctness
signal. Where ``service/distinctness.py`` owns the cheap lexical pass and the contract +
fallback policy, this module owns the LLM call that judges whether three finished variants
commit to genuinely different strategic angles — the thing the word-overlap metric is
structurally blind to.

Design commitments (mirroring ``generator.py`` and the plan):
  * One LCEL chain (``prompt | model.with_structured_output(...)``); one call per product,
    evaluating all three variants together — never per platform, never inside a critique-loop
    retry.
  * The judge reads shared *meaning/strategy*, not shared words: two variants making the same
    underlying claim in different wording are NOT distinct, even with zero word overlap.
  * A capable model (Sonnet, not the generator's Haiku): catching strategic paraphrase is the
    whole reason the judge exists, so under-powering it would repeat the very blind spot it is
    meant to cover.
  * No ``temperature`` is passed. Claude Sonnet 5 is in the family of models (alongside Opus
    4.7/4.8 and Fable 5) that reject sampling parameters outright — even an explicit
    ``temperature=0.0`` — with a 400 ("`temperature` is deprecated for this model"), so the
    request omits the field entirely rather than picking a "safe" value.
  * The verdict shape is enforced structurally, not by prompt instruction. An earlier version
    asked the model in prose to "return ONLY a JSON object" and parsed free text with
    ``JsonOutputParser`` — about 1 in 5 live calls failed to parse (stray preamble, markdown
    fences) despite the instruction. This project already learned that lesson once, for the
    character-limit requirement: natural-language formatting instructions are not a structural
    guarantee. ``with_structured_output`` binds ``JudgeVerdictSchema`` via forced tool-calling
    (``method="function_calling"``), so the API itself constrains the output shape — there is
    no free-text JSON to mis-format.
  * No error handling that swallows failures silently: a genuine failure (network, timeout,
    refusal, rate limit, or the rare case the model still can't be coerced into the schema)
    raises here, and ``service.distinctness.assess_with_judge`` catches it and falls back to
    the lexical signal so the pipeline never breaks.
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from service.critique_loop import VariantResult
from service.distinctness import JudgeVerdict

# --- Model configuration (named, no magic numbers) --------------------------------

# A capable model for the judge — see module docstring. Deliberately NOT the generator's
# Haiku default: judging strategic-argument equivalence is the hard part this call exists for.
JUDGE_MODEL: str = "claude-sonnet-5"
# A verdict is a small tool-call payload plus a one-to-three-sentence rationale.
JUDGE_MAX_TOKENS: int = 512


class JudgeVerdictSchema(BaseModel):
    """Structured-output schema bound to the model via forced tool-calling.

    Field descriptions are the model's *only* instruction on what each field means — the
    system prompt no longer describes a JSON shape in prose, since the schema itself is what
    constrains the response. ``rationale`` keeps the same non-blank requirement the old manual
    check enforced, now as a validator instead of a hand-rolled ``isinstance``/``strip`` check.
    """

    distinct: bool = Field(
        description=(
            "True if all three variants commit to genuinely different strategic angles; "
            "false if any two share the same underlying argument."
        )
    )
    similar_platforms: list[str] = Field(
        default_factory=list,
        description=(
            "Placement labels caught making the same underlying argument, e.g. "
            "['Search Ad', 'Display Banner']. Empty when distinct is true."
        ),
    )
    rationale: str = Field(
        description=(
            "One to three sentences, in plain language, explaining the verdict — which "
            "arguments overlap or why the three are genuinely different."
        )
    )

    @field_validator("rationale")
    @classmethod
    def _rationale_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("rationale must not be blank")
        return stripped


# --- Prompt (the exact text sent to the model) ------------------------------------

SYSTEM_PROMPT: str = (
    "You are a marketing creative director auditing a set of three ad variants written for "
    "the SAME product, one per placement (Search Ad, Social Ad, Display Banner). Each "
    "placement was assigned a distinct strategic angle. Your job is to judge whether the "
    "three variants actually commit to genuinely different strategic angles and arguments.\n\n"
    "Judge shared MEANING and strategy, not shared words. Two variants are NOT distinct if "
    "they make the same underlying claim or promise, even when worded completely differently "
    "(e.g. 'ready today for beginners' and 'built for beginners who want reliable flight' are "
    "the same 'easy/safe for beginners' argument). Conversely, two variants that share "
    "unavoidable product-name words but make genuinely different arguments ARE distinct."
)

HUMAN_PROMPT: str = (
    "Product (for context on what claims are available):\n{product_source}\n\n"
    "The three variants and their assigned strategic angles:\n{variants_block}\n\n"
    "Assess whether the three commit to genuinely different strategic angles."
)


def _format_variants(variants: list[VariantResult]) -> str:
    """Render each variant as an assigned-angle + final-text block for the prompt."""
    blocks = []
    for variant in variants:
        rule = variant.rule
        angle = rule.angle or "(no explicit angle assigned — a distinct angle is expected)"
        blocks.append(
            f"- Placement: {rule.label}\n"
            f"  Assigned strategic angle: {angle}\n"
            f"  Variant text: {variant.final_text}"
        )
    return "\n".join(blocks)


def _build_chain(model: str, max_tokens: int) -> Any:
    """Build the single LCEL chain: prompt | model bound to the structured-output schema.

    ``with_structured_output(JudgeVerdictSchema, method="function_calling")`` forces the model
    to answer via a tool call whose arguments are validated against the schema before this
    chain's ``.invoke()`` returns — the returned value is already a ``JudgeVerdictSchema``
    instance, not text to parse. A response the API can't coerce into the schema, or any
    genuine call failure, raises out of ``.invoke()`` and propagates to the service-layer
    fallback exactly as a parse failure used to. No ``temperature`` is passed — see the module
    docstring: Sonnet 5 rejects sampling parameters outright, so the field must be omitted
    entirely, not set to a "safe" value.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    )
    model_client = ChatAnthropic(model=model, max_tokens=max_tokens)
    structured_model = model_client.with_structured_output(
        JudgeVerdictSchema, method="function_calling"
    )
    return prompt | structured_model


class AnthropicDistinctnessJudge:
    """A ``DistinctnessJudge`` backed by Anthropic Claude (Sonnet) via LangChain.

    Satisfies ``service.distinctness.DistinctnessJudge`` structurally, so an instance can be
    passed directly as the ``judge`` argument of ``run_all_platforms``. The chain is built
    once here, not per call.
    """

    def __init__(
        self,
        *,
        model: str = JUDGE_MODEL,
        max_tokens: int = JUDGE_MAX_TOKENS,
    ) -> None:
        load_dotenv()  # make ANTHROPIC_API_KEY from .env visible to ChatAnthropic
        self._chain = _build_chain(model, max_tokens)

    def __call__(
        self, *, variants: list[VariantResult], product_source: str
    ) -> JudgeVerdict:
        """Judge whether the three finished variants commit to distinct strategic angles.

        Invokes the model once over all three; the chain returns an already schema-validated
        ``JudgeVerdictSchema``, which ``_to_verdict`` maps onto ``JudgeVerdict`` (resolving
        ``similar_platforms`` from whatever labels or names the model used to platform *names*,
        the machine ids). Any call failure — network, timeout, refusal, rate limit, or the rare
        response the API can't coerce into the schema — propagates from ``.invoke()`` to the
        service-layer fallback.
        """
        parsed = self._chain.invoke(
            {
                "product_source": product_source,
                "variants_block": _format_variants(variants),
            }
        )
        return _to_verdict(parsed, variants)


def _resolve_platform_names(
    reported: list[Any], variants: list[VariantResult]
) -> tuple[str, ...]:
    """Map whatever the model named (labels or names, any case) back to platform names.

    Builds a lookup from both ``rule.label`` and ``rule.name`` so the judge can answer in
    natural labels; unrecognized entries are dropped rather than trusted blindly.
    """
    lookup: dict[str, str] = {}
    for variant in variants:
        lookup[variant.rule.label.strip().lower()] = variant.rule.name
        lookup[variant.rule.name.strip().lower()] = variant.rule.name

    resolved: list[str] = []
    for item in reported:
        name = lookup.get(str(item).strip().lower())
        if name is not None and name not in resolved:
            resolved.append(name)
    return tuple(resolved)


def _to_verdict(parsed: JudgeVerdictSchema, variants: list[VariantResult]) -> JudgeVerdict:
    """Map a schema-validated model response onto the internal ``JudgeVerdict`` shape.

    Type and shape validity are already guaranteed by ``JudgeVerdictSchema`` (Pydantic ran
    before this function is reached — see ``_build_chain``), so there is nothing left to
    validate here. What remains is semantic, not structural: the model may still name a
    platform by its human label ("Search Ad") rather than its internal name ("search_ad"),
    so ``_resolve_platform_names`` still resolves case-insensitively and drops unrecognized
    entries rather than trusting them blindly.
    """
    return JudgeVerdict(
        distinct=parsed.distinct,
        rationale=parsed.rationale,
        similar_platforms=_resolve_platform_names(parsed.similar_platforms, variants),
    )
