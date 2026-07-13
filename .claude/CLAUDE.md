# Ad Creative Variant Auditor

## Project Goal
Ingest a long-form product description and output three character-limited ad variants (Search,
Social, Display). Two guarantees, both from code rather than prompt instruction: (1) 100%
character-limit compliance via a critique loop + deterministic fallback truncation; (2) a
cross-platform distinctness check — a cheap always-on lexical signal plus an optional semantic
LLM judge — that flags and surfaces (does not silently guarantee) whether the three variants
commit to genuinely different creative angles.

## Tech Stack
- Python 3.10+ (`.python-version` = 3.10; `pyproject.toml` `requires-python = ">=3.10"`),
  managed with uv (`pyproject.toml` + `uv.lock`; `requirements.txt` exported from the lock).
- LangChain (`langchain`, `langchain-anthropic`) for the LLM calls; Pydantic for the judge's
  structured-output schema.
- Anthropic API, two models: **Claude Haiku 4.5** for generation (small/cheap; short task) and
  **Claude Sonnet 5** for the distinctness judge (semantic judgment is the hard part). Key read
  from `.env` via `python-dotenv`.
- `api/` — FastAPI, thin HTTP wrapper only. No business logic duplicated; injects `agent/` into
  `service/` and returns the existing `audit_log` serialization.
- `frontend/` — React + Vite (TypeScript). Talks to `api/` only via `fetch` to `/api/generate`.
  Never imports Python. Never hardcodes product copy.
- Node 18+ / npm for frontend tooling — separate toolchain from uv; do not conflate the two.

## Folder Structure
```
data/       -> platform_rules.json (swappable rules incl. optional angle) + regression_set/ (real products)
service/    -> file I/O, rule validation, critique loop, distinctness (lexical + judge contracts),
               audit log. The graded architectural centerpiece. Owns all validation + char counting.
agent/      -> LLM mechanisms behind service Protocols: generator.py (Haiku) + judge.py (Sonnet 5).
               Never counts characters.
api/        -> FastAPI. POST /api/generate {product_source} -> run_all_platforms(generate=..., judge=...)
               -> build_audit_log -> JSON. Serves frontend/dist/ in shipped mode (one port).
frontend/   -> React SPA: paste box, generate button, 3 cards. Each card shows the char-limit
               status badge + two distinctness badges (lexical / judge) + a "signals disagree"
               banner, and an expandable attempt trail.
tests/      -> unit-test modules (deterministic, no API) + 3 support files: harness_convergence.py,
               harness_judge.py (offline measurement runners), judge_fixtures.py (labeled ground truth).
```

## Constraints (do not deviate without asking)
- Retry cap for the critique loop: 3 attempts, hardcoded as `MAX_ATTEMPTS` in
  `service/critique_loop.py`, NOT read from `platform_rules.json` — intentional.
- After 3 failed LLM attempts, fall back to deterministic word-boundary truncation
  (`truncate_to_limit`) so the system provably cannot emit an over-limit variant.
- Character limits are counted with Python's `len()`, never trusted from the LLM's claim.
- `platform_rules.json` must support adding a 4th platform with zero code changes to `service/`
  or `agent/`. `angle` is optional (absent -> generator derives its own).
- Frontend accepts pasted text in a textarea, never hardcoded to one product.
- Every attempt (pass/fail, LLM/fallback) is logged and surfaced in the UI, flagged
  "AI-approved" vs "fallback-truncated".
- **Distinctness is a separate, cross-platform concern that runs after all variants exist and is
  read-only over `final_text` — it must never alter a variant or its char-limit compliance.**
  - The lexical signal (`assess_distinctness`) is always on and is never a gate on the judge.
  - `DISTINCTNESS_THRESHOLD` is hardcoded in `service/distinctness.py`, NOT rule-sourced
    (mirrors `MAX_ATTEMPTS`).
  - The judge is optional; on ANY failure (API/timeout/unparseable) the pipeline falls back to
    the lexical signal and never breaks or blocks generation.
  - The two signals are shown side by side; neither overrides the other. `signals_agree` surfaces
    disagreement rather than silently resolving it.

## Agent Layer Constraints
- `agent/generator.py` implements the `VariantGenerator` Protocol defined in
  `service/critique_loop.py` — read that Protocol; don't restate its signature here.
- Generation: LangChain `ChatAnthropic`, a single LCEL chain (`prompt | model | parser`), NOT a
  multi-tool agent. Two chains of the same shape differ only in sampling temperature (creative
  first attempt / near-deterministic retry).
- Generation model: small/cheap Claude (Haiku); API key via `.env` / `python-dotenv`.
- `generator.py` must NEVER count characters, check length, or reference `rule.max_chars` —
  `service/` owns all validation; the agent only generates and relays feedback text verbatim.
  `rule.style` and `rule.angle` are generation guidance; loop `feedback` (when not None) is
  included verbatim so the model can correct a prior over-limit draft.
- `agent/judge.py` implements the `DistinctnessJudge` Protocol defined in
  `service/distinctness.py` — one call per product over all three finished variants; never per
  platform, never inside a critique-loop retry.
- Judge model: Claude **Sonnet 5** (semantic judgment, not the generator's Haiku). **No
  `temperature` is passed** — Sonnet 5 rejects sampling params outright (400), so the field is
  omitted, not set to a "safe" value.
- The judge's output shape is a **structural guarantee, not a prompt instruction**: it uses
  `with_structured_output(JudgeVerdictSchema, method="function_calling")`, not a "return ONLY
  JSON" instruction. It raises on failure; `service.distinctness.assess_with_judge` catches that
  and falls back to lexical.

## Definition of Done
- [x] `service/data_loader.py` reads files from disk (product source + `platform_rules.json`).
- [x] `service/rules.py` loads and validates `platform_rules.json`; supports an optional `angle`
      field and a zero-code-change 4th platform.
- [x] `service/critique_loop.py`: generate -> validate (`len()`) -> retry (max 3) -> deterministic
      fallback truncation. Truncation prefers coherent boundaries but keeps ≥70% of the budget
      (`MIN_WORD_BOUNDARY_FRACTION`) and treats in-token punctuation ($129.95, 7,307) correctly.
      Retry feedback is magnitude-aware (large overshoots trigger a fresh short rewrite with a
      word ceiling, not a doomed edit).
- [x] `service/distinctness.py`: lexical cross-platform check (overlap coefficient + light
      stemming + product-name exclusion) AND the `DistinctnessJudge` Protocol + `assess_with_judge`
      fallback policy.
- [x] `agent/generator.py` calls the LLM (Haiku), has zero character-counting logic, and commits
      to the assigned `angle`.
- [x] `agent/judge.py`: Sonnet 5 semantic judge via `with_structured_output`; one call per
      product; raises on failure so `service/` falls back.
- [x] `service/audit_log.py` serializes attempts + both distinctness verdicts + `signals_agree`.
- [x] `api/main.py`: POST `/api/generate`, zero business logic, injects generator + judge into the
      existing pipeline and serialization.
- [x] `frontend/`: three cards, each with char-limit + lexical + judge badges, a disagreement
      banner, and an expandable attempt trail.
- [x] One documented command builds `frontend/` and runs `api/main.py` on one port (see README).
- [x] `tests/`: deterministic unit tests + two offline measurement harnesses
      (`harness_convergence.py`, `harness_judge.py`) + labeled `judge_fixtures.py`, run against
      `data/regression_set/`.
- [x] `README.md` (setup, API key, single run command) and `requirements.txt` exported from the lock.
