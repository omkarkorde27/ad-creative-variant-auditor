"""FastAPI HTTP boundary for the Ad Creative Variant Auditor.

Thin wrapper only: it validates nothing about ad copy, counts no characters,
and owns no retry logic. Every guarantee (character limits, the critique loop,
the deterministic fallback, the audit trail) already lives in ``service/`` and
``agent/`` and is exercised here exactly as-is. In production this same process
also serves the built React frontend so the whole app runs on one port.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.generator import AnthropicVariantGenerator
from agent.judge import AnthropicDistinctnessJudge
from service.audit_log import build_audit_log
from service.critique_loop import run_all_platforms
from service.rules import load_platform_rules

app = FastAPI(title="Ad Creative Variant Auditor")

# Built once and reused across requests. The generator and judge each hold a compiled LCEL
# chain; the rules come from the swappable data/platform_rules.json. The judge runs one
# semantic-distinctness call per product; if it fails, the pipeline falls back to the lexical
# signal, so a judge outage never blocks generation.
_generator = AnthropicVariantGenerator()
_judge = AnthropicDistinctnessJudge()
_rules = load_platform_rules()

# Uvicorn dispatches sync endpoints on a threadpool, so two requests can land
# concurrently on the single shared generator. Serialize the pipeline call so
# only one thread drives the generator's chain at a time.
_generation_lock = threading.Lock()

# Where `npm run build` emits the frontend. Present only in shipped mode.
_DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class GenerateRequest(BaseModel):
    """Request body for POST /api/generate."""

    product_source: str


@app.post("/api/generate")
def generate(req: GenerateRequest) -> list[dict[str, Any]]:
    """Run the critique loop for every platform and return the audit log.

    The return value is `build_audit_log(...)` verbatim — the same serialization
    the frontend renders and the tests assert on. No shape is reinvented here.
    """
    if not req.product_source.strip():
        raise HTTPException(status_code=422, detail="product_source is empty")

    with _generation_lock:
        results = run_all_platforms(
            product_source=req.product_source,
            rules=_rules,
            generate=_generator,
            judge=_judge,
        )
    return build_audit_log(results)


# Shipped mode: serve the built SPA on the same origin/port as the API. The
# specific /api/generate route above is registered first, so it always wins;
# this catch-all mount only handles everything else (index.html, assets). The
# guard keeps dev mode (no build present) booting cleanly. On Vercel (`VERCEL`
# is always set in that runtime) the frontend is deployed separately as a
# static site by vercel.json's outputDirectory, so this function should only
# ever answer /api/generate — mounting StaticFiles here too would let it swallow
# any request whose path doesn't hit that route with a misleading 404/405
# instead of FastAPI's own routing error.
if _DIST_DIR.is_dir() and not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory=_DIST_DIR, html=True), name="frontend")
