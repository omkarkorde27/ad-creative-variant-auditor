"""Disk I/O for the service layer.

This module knows *how to read files*, not *what they mean*. It returns raw text and
raw parsed JSON; schema validation of the platform rules lives in ``service.rules``.
Keeping I/O and validation separate makes each independently testable and keeps the
"add a 4th platform with zero code changes" guarantee in exactly one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Path defaults are derived from this module's location so callers need no arguments
# and the service works regardless of the process's current working directory.
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PRODUCT_SOURCE: Path = DATA_DIR / "product_source.txt"
DEFAULT_PLATFORM_RULES: Path = DATA_DIR / "platform_rules.json"


class DataLoadError(RuntimeError):
    """Raised when a source file is missing, empty, or not valid JSON."""


def load_product_source(path: Path = DEFAULT_PRODUCT_SOURCE) -> str:
    """Read the long-form product description from disk.

    Returns the file's text with surrounding whitespace stripped. Raises
    ``DataLoadError`` if the file does not exist or contains no non-whitespace
    text — an empty product source cannot produce ad variants, so we fail loudly
    with an actionable message rather than generating from nothing.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataLoadError(
            f"Product source not found at {path}. Add the product copy to that file."
        ) from exc
    except OSError as exc:  # pragma: no cover - unusual filesystem errors
        raise DataLoadError(f"Could not read product source at {path}: {exc}") from exc

    stripped = text.strip()
    if not stripped:
        raise DataLoadError(
            f"Product source at {path} is empty. Paste the product description into it."
        )
    return stripped


def load_platform_rules_file(path: Path = DEFAULT_PLATFORM_RULES) -> dict[str, Any]:
    """Read and JSON-parse the platform rules file.

    Returns the raw decoded object (no schema checking — see
    ``service.rules.parse_platform_rules``). Raises ``DataLoadError`` if the file
    is missing or is not valid JSON.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DataLoadError(f"Platform rules not found at {path}.") from exc
    except OSError as exc:  # pragma: no cover - unusual filesystem errors
        raise DataLoadError(f"Could not read platform rules at {path}: {exc}") from exc

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DataLoadError(f"Platform rules at {path} are not valid JSON: {exc}") from exc
