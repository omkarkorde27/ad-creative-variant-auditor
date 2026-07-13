# Ad Creative Variant Auditor

Ingests a long-form product description and produces three character-limited ad
variants — **Search Ad** (30), **Social Ad** (125), **Display Banner** (90) — with two
independent quality gates:

1. **Character-limit compliance is guaranteed by code, not a prompt.** Every variant is
   generated, then validated with Python's `len()` (never trusting the model's claim). Over-limit
   drafts are critiqued back to the model up to 3 attempts; after that the system falls back to
   deterministic word-boundary truncation, so it *provably* cannot emit an over-limit variant.
2. **Cross-platform distinctness is checked** so the three variants aren't one idea at three
   lengths. Two independent signals run and are shown side by side: a cheap **lexical**
   word-overlap check (always on) and a **semantic LLM judge** (Claude Sonnet 5) that reads all
   three against their assigned strategic angles. Distinctness is *flagged and surfaced*, not
   guaranteed — see `design-decisions.md` for why.

Every attempt (LLM or fallback, pass or fail) and both distinctness verdicts are recorded and
surfaced in the UI.

## Architecture

| Layer | Path | Responsibility |
|---|---|---|
| `service/` | file I/O, rules, **critique loop**, **distinctness**, audit log | Owns all validation, character counting, and the distinctness contracts. The graded centerpiece. |
| `agent/` | LangChain LLM calls: `generator.py` (Haiku) + `judge.py` (Sonnet 5) | Generates text and judges distinctness. Never counts characters. |
| `api/` | FastAPI | Thin HTTP wrapper. No business logic — injects `agent/` into `service/` and returns the audit log. |
| `frontend/` | React + Vite + TypeScript | Paste box, generate button, three result cards (char-limit + lexical + judge badges, disagreement banner, expandable attempt trail). Talks to `api/` only. |

Platform rules live in `data/platform_rules.json` (name, label, `max_chars`, `style`, optional
`angle`). Adding a 4th platform there requires **zero code changes** — the API loads whatever
rules exist and the frontend renders one card per result.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) (Python 3.10+) and Node 18+ / npm.

```bash
uv sync                                   # 1. Python deps (creates .venv from uv.lock)
cp .env.example .env                      # 2. API key — then edit .env: ANTHROPIC_API_KEY=sk-ant-...
cd frontend && npm install && cd ..       # 3. Frontend deps
```

The app calls two Anthropic models: Claude Haiku 4.5 (generation) and Claude Sonnet 5 (the
distinctness judge). A judge outage degrades gracefully to the lexical signal — it never blocks
generation.

## Run — shipped mode (single command, one port)

```bash
cd frontend && npm run build && cd .. && uv run uvicorn api.main:app --port 8000
```

Open **http://localhost:8000** — page and API are same-origin (FastAPI serves `frontend/dist/`).

## Run — dev mode (two processes, hot reload)

```bash
# Terminal 1 — API
uv run uvicorn api.main:app --reload --port 8000
# Terminal 2 — Vite (proxies /api → http://localhost:8000)
cd frontend && npm run dev
```

Open the URL Vite prints (default **http://localhost:5173**).

## API

`POST /api/generate` — body `{ "product_source": "..." }`; empty source returns `422`.

Response is a JSON array, one object per platform:

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
    "distinct": true,
    "distinct_label": "Distinct",
    "distinct_note": null,
    "judge_distinct": false,
    "judge_label": "Same angle",
    "judge_note": "Search and Display both lead with the same benefit claim…",
    "signals_agree": false,
    "attempts": [
      { "attempt_number": 1, "source": "llm", "text": "…", "char_count": 28, "max_chars": 30, "passed": true, "note": null }
    ]
  }
]
```

- `status`: `"ai_approved"` | `"fallback_truncated"`.
- `distinct` / `judge_distinct`: `true` = distinct, `false` = flagged too similar, `null` = not
  assessed (lexical) or judge unavailable (judge). `signals_agree` is `false` when the two
  disagree, `null` when the judge didn't run.

## Tests

```bash
uv run pytest          # deterministic unit tests, no API calls
```

Two offline **measurement harnesses** (named `harness_*` so pytest ignores them; `--live` hits
the real API and costs money):

```bash
uv run python tests/harness_convergence.py            # char-limit convergence + distinctness collision rate (stub)
uv run python tests/harness_judge.py                  # judge accuracy vs labeled fixtures (stub)
uv run python tests/harness_judge.py --live --repeats 3   # real judge vs fixtures
```

They run against `data/regression_set/` (real scraped products) and `tests/judge_fixtures.py`
(human-labeled cases). See `design-decisions.md` for how they were used.

## Dependencies

Python deps are managed with `uv` (`pyproject.toml` + `uv.lock`). `requirements.txt` is exported
from the lock for evaluator convenience:

```bash
uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
```
