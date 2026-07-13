"""Service layer for the Ad Creative Variant Auditor.

Pure-stdlib modules that own file I/O, platform-rule validation, the character-limit
critique loop, and audit-log serialization. Nothing here imports the agent or the
frontend; the LLM is reached only through an injected ``VariantGenerator`` callable.
"""

from service.audit_log import (
    attempt_to_dict,
    build_audit_log,
    to_json,
    variant_to_dict,
)
from service.critique_loop import (
    MAX_ATTEMPTS,
    Attempt,
    VariantGenerator,
    VariantResult,
    fits,
    run_all_platforms,
    run_critique_loop,
    truncate_to_limit,
)
from service.data_loader import (
    DEFAULT_PLATFORM_RULES,
    DEFAULT_PRODUCT_SOURCE,
    DataLoadError,
    load_platform_rules_file,
    load_product_source,
)
from service.rules import (
    PlatformRule,
    RuleValidationError,
    load_platform_rules,
    parse_platform_rules,
)

__all__ = [
    # data_loader
    "DataLoadError",
    "DEFAULT_PLATFORM_RULES",
    "DEFAULT_PRODUCT_SOURCE",
    "load_platform_rules_file",
    "load_product_source",
    # rules
    "PlatformRule",
    "RuleValidationError",
    "load_platform_rules",
    "parse_platform_rules",
    # critique_loop
    "MAX_ATTEMPTS",
    "Attempt",
    "VariantGenerator",
    "VariantResult",
    "fits",
    "run_all_platforms",
    "run_critique_loop",
    "truncate_to_limit",
    # audit_log
    "attempt_to_dict",
    "build_audit_log",
    "to_json",
    "variant_to_dict",
]
