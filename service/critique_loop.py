from dataclasses import dataclass, field
from typing import Callable, Optional

RETRY_CAP = 3  # hardcoded by design, not marketer-tunable


@dataclass
class Attempt:
    attempt_number: int
    text: str
    char_count: int
    passed: bool
    mechanism: str  # "llm" or "fallback_truncation"


@dataclass
class VariantResult:
    platform_name: str
    final_text: str
    final_char_count: int
    mechanism: str
    attempts: list = field(default_factory=list)


def truncate_to_limit(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    last_space = trimmed.rfind(" ")
    return (trimmed[:last_space] if last_space > 0 else trimmed).rstrip()


def run_critique_loop(platform_rule: dict, product_copy: str,
                       generate_fn: Callable[[str, dict, Optional[str]], str]) -> VariantResult:
    max_chars = platform_rule["max_chars"]
    attempts = []
    feedback = None

    for i in range(1, RETRY_CAP + 1):
        draft = generate_fn(product_copy, platform_rule, feedback)
        char_count = len(draft)
        passed = char_count <= max_chars
        attempts.append(Attempt(i, draft, char_count, passed, "llm"))

        if passed:
            return VariantResult(platform_rule["name"], draft, char_count, "llm", attempts)

        overage = char_count - max_chars
        feedback = f"Your last draft was {char_count} characters. Limit is {max_chars}. Cut {overage}."

    fallback = truncate_to_limit(attempts[-1].text, max_chars)
    attempts.append(Attempt(RETRY_CAP + 1, fallback, len(fallback), True, "fallback_truncation"))
    return VariantResult(platform_rule["name"], fallback, len(fallback), "fallback_truncation", attempts)