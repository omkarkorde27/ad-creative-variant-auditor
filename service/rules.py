"""Platform-rule schema validation.

Owns the ``PlatformRule`` data model and the logic that turns the raw JSON produced
by ``data_loader`` into validated, immutable rule objects. Validation is fully generic
over the list of platforms: no platform name (``search_ad``, ``social_ad``, ...) is
ever referenced by literal, so a 4th platform can be added to ``platform_rules.json``
with zero code changes here or in ``agent/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from service.data_loader import DEFAULT_PLATFORM_RULES, load_platform_rules_file

# The exact set of fields every platform entry must provide. Adding a platform means
# adding a JSON object with these keys — not editing this tuple.
REQUIRED_FIELDS: tuple[str, ...] = ("name", "label", "max_chars", "style")


@dataclass(frozen=True)
class PlatformRule:
    """One platform's generation contract.

    ``max_chars`` is the hard character limit enforced by the critique loop via
    ``len()``. ``style`` is guidance passed to the generator; it never affects
    validation or counting. ``angle`` is an optional creative-angle hint passed to
    the generator to keep variants distinct; when empty (the default) the generator
    derives an angle from the product itself. Being optional, it does not appear in
    ``REQUIRED_FIELDS``, so a new platform can still be added with zero code changes.
    """

    name: str
    label: str
    max_chars: int
    style: str
    angle: str = ""


class RuleValidationError(ValueError):
    """Raised when the platform rules do not conform to the expected schema."""


def parse_platform_rules(raw: Mapping[str, Any]) -> list[PlatformRule]:
    """Validate raw decoded JSON and return the list of ``PlatformRule`` objects.

    Enforced schema:
      * top-level ``platforms`` key present and a non-empty list;
      * every entry is a mapping containing all ``REQUIRED_FIELDS``;
      * ``max_chars`` is an ``int`` strictly greater than 0 (``bool`` rejected);
      * ``name`` values are unique across platforms.

    Raises ``RuleValidationError`` with a precise, position-aware message on the
    first violation found.
    """
    if not isinstance(raw, Mapping):
        raise RuleValidationError(
            f"Platform rules must be a JSON object, got {type(raw).__name__}."
        )

    platforms = raw.get("platforms")
    if platforms is None:
        raise RuleValidationError("Platform rules must contain a 'platforms' key.")
    if not isinstance(platforms, list) or not platforms:
        raise RuleValidationError("'platforms' must be a non-empty list.")

    rules: list[PlatformRule] = []
    seen_names: set[str] = set()

    for index, entry in enumerate(platforms):
        location = f"platforms[{index}]"
        if not isinstance(entry, Mapping):
            raise RuleValidationError(f"{location} must be an object.")

        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise RuleValidationError(
                f"{location} is missing required field(s): {', '.join(missing)}."
            )

        name = entry["name"]
        max_chars = entry["max_chars"]

        # bool is a subclass of int; reject it explicitly so True/False can't pose as a limit.
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise RuleValidationError(
                f"{location}.max_chars must be an integer, got {max_chars!r}."
            )
        if max_chars <= 0:
            raise RuleValidationError(
                f"{location}.max_chars must be greater than 0, got {max_chars}."
            )

        if not isinstance(name, str) or not name.strip():
            raise RuleValidationError(f"{location}.name must be a non-empty string.")
        if name in seen_names:
            raise RuleValidationError(f"Duplicate platform name {name!r} at {location}.")
        seen_names.add(name)

        rules.append(
            PlatformRule(
                name=name,
                label=str(entry["label"]),
                max_chars=max_chars,
                style=str(entry["style"]),
                # Optional: absent -> "" -> generator derives the angle itself.
                angle=str(entry.get("angle", "")),
            )
        )

    return rules


def load_platform_rules(path: Path = DEFAULT_PLATFORM_RULES) -> list[PlatformRule]:
    """Convenience: read the rules file from disk and validate it in one call."""
    return parse_platform_rules(load_platform_rules_file(path))
