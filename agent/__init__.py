"""Agent layer for the Ad Creative Variant Auditor.

The LLM generation layer. Exposes ``AnthropicVariantGenerator``, whose instances
satisfy the ``service.critique_loop.VariantGenerator`` Protocol. The agent only
produces text: it never counts characters, checks length, or references ``max_chars``.
All validation and the retry/fallback guarantee live in ``service/``.
"""

from agent.generator import AnthropicVariantGenerator

__all__ = ["AnthropicVariantGenerator"]
