# Ad Creative Variant Auditor

Ingests a long-form product description and produces three character-limited ad
variants — **Search Ad** (30), **Social Ad** (125), **Display Banner** (90) —
guaranteeing 100% compliance with each platform's character limit through a
**critique loop**, not a prompt instruction.

Every variant is generated, then validated with Python's `len()` (never trusting
the model's claim about its own output). If a draft is over the limit, the
critique is fed back to the model, up to 3 attempts. After 3 failures the system
falls back to deterministic word-boundary truncation, so it *provably* cannot
emit an over-limit variant. Every attempt — LLM or fallback, pass or fail — is
recorded and surfaced in the UI.

## Architecture

| Layer | Path | Responsibility |
|---|---|---|
| `service/` | file I/O, rules, **critique loop**, audit log | Owns all validation and character counting. The graded centerpiece. |
| `agent/` | LangChain generation (`ChatAnthropic`, single LCEL chain) | Generates text only. Never counts characters. |
| `api/` | FastAPI | Thin HTTP wrapper. No business logic — calls `service/` + `agent/` as-is. |
| `frontend/` | React + Vite + TypeScript | Paste box, generate button, three result cards with expandable attempt trails. Talks to `api/` only. |

Platform rules live in `data/platform_rules.json`. Adding a 4th platform there
requires **zero code changes** — the API loads whatever rules exist and the
frontend renders one card per result.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python) and Node 18+ / npm.

```bash
# 1. Python deps
uv sync

# 2. API key — copy the example and add your Anthropic key
cp .env.example .env
#   then edit .env:  ANTHROPIC_API_KEY=sk-ant-...

# 3. Frontend deps
cd frontend && npm install && cd ..
```

## Run — shipped mode (single command, one port)

Build the frontend, then serve both the API and the built SPA from one FastAPI
process on one port:

```bash
cd frontend && npm run build && cd .. && uv run uvicorn api.main:app --port 8000
```

Open **http://localhost:8000** — the page and the API are same-origin.

## Run — dev mode (two processes, hot reload)

Terminal 1 — API:

```bash
uv run uvicorn api.main:app --reload --port 8000
```

Terminal 2 — Vite dev server (proxies `/api` → `http://localhost:8000`):

```bash
cd frontend && npm run dev
```

Open the URL Vite prints (default **http://localhost:5173**).

## API

`POST /api/generate`

Request:

```json
{ "product_source": "Your long-form product description…" }
```

Response — a JSON array, one object per platform:

```json
[
  {
    "platform": "search_ad",
    "label": "Search Ad",
    "max_chars": 30,
    "final_text": "…",
    "final_char_count": 28,
    "status": "ai_approved",
    "status_label": "AI-approved",
    "attempts": [
      { "attempt_number": 1, "source": "llm", "text": "…", "char_count": 28, "max_chars": 30, "passed": true, "note": null }
    ]
  }
]
```

`status` is `"ai_approved"` or `"fallback_truncated"`. An empty `product_source`
returns `422`.

## Tests

```bash
uv run pytest
```

## Dependencies

Python deps are managed with `uv` (`pyproject.toml` + `uv.lock`).
`requirements.txt` is exported from the lock for evaluator convenience:

```bash
uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
```
