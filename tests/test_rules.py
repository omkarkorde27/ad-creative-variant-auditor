"""Persisted version of the manual verification scenarios for platform rule validation.

Covers scenario 3: parse_platform_rules against a 4-platform copy of the real schema
(proving a new platform needs zero code changes) and against a battery of malformed
configs (proving RuleValidationError is raised precisely). Operates on in-memory dicts
only — never reads or writes the real data/platform_rules.json.
"""

from __future__ import annotations

import pytest

from service.rules import RuleValidationError, parse_platform_rules

# Mirrors the shipped data/platform_rules.json exactly, as a plain dict.
REAL_PLATFORMS = [
    {"name": "search_ad", "label": "Search Ad", "max_chars": 30, "style": "terse"},
    {"name": "social_ad", "label": "Social Ad", "max_chars": 125, "style": "punchy"},
    {"name": "display_banner", "label": "Display Banner", "max_chars": 90, "style": "crisp"},
]


def test_real_platforms_parse() -> None:
    rules = parse_platform_rules({"platforms": REAL_PLATFORMS})
    assert [r.name for r in rules] == ["search_ad", "social_ad", "display_banner"]
    assert rules[0].max_chars == 30


def test_fourth_platform_zero_code_change() -> None:
    four_platforms = {
        "platforms": [
            *REAL_PLATFORMS,
            {
                "name": "email_subject",
                "label": "Email Subject",
                "max_chars": 50,
                "style": "curiosity-driven",
            },
        ]
    }

    rules = parse_platform_rules(four_platforms)

    assert len(rules) == 4
    assert rules[3].name == "email_subject"
    assert rules[3].max_chars == 50


def test_angle_is_optional() -> None:
    """`angle` parses when present and defaults to "" when absent (zero-code 4th platform)."""
    rules = parse_platform_rules(
        {
            "platforms": [
                {
                    "name": "search_ad",
                    "label": "Search Ad",
                    "max_chars": 30,
                    "style": "terse",
                    "angle": "lead with the strongest benefit",
                },
                {"name": "social_ad", "label": "Social Ad", "max_chars": 125, "style": "punchy"},
            ]
        }
    )
    assert rules[0].angle == "lead with the strongest benefit"
    assert rules[1].angle == ""  # omitted -> default, generator derives its own angle


@pytest.mark.parametrize(
    "raw",
    [
        {"nope": []},  # missing 'platforms' key entirely
        {"platforms": []},  # empty list
        {"platforms": "not-a-list"},  # wrong type for 'platforms'
        {"platforms": [{"name": "x", "label": "X", "style": "s"}]},  # missing max_chars
        {"platforms": [{"name": "x", "label": "X", "max_chars": 0, "style": "s"}]},  # zero limit
        {"platforms": [{"name": "x", "label": "X", "max_chars": -5, "style": "s"}]},  # negative
        {"platforms": [{"name": "x", "label": "X", "max_chars": True, "style": "s"}]},  # bool
        {"platforms": [{"name": "x", "label": "X", "max_chars": "30", "style": "s"}]},  # string
        {"platforms": ["not-a-mapping"]},  # entry isn't an object
        {
            "platforms": [
                {"name": "x", "label": "X", "max_chars": 10, "style": "s"},
                {"name": "x", "label": "Y", "max_chars": 10, "style": "s"},
            ]
        },  # duplicate name
    ],
)
def test_malformed_configs_rejected(raw: object) -> None:
    with pytest.raises(RuleValidationError):
        parse_platform_rules(raw)  # type: ignore[arg-type]
