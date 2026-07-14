# Design Decisions — Ad Creative Variant Auditor

## Overview

The system ingests a long-form product description and a JSON platform-rules file, and
produces three character-limited ad variants (Search Ad, Social Ad, Display Banner),
guaranteeing 100% compliance with each platform's character limit. The central
constraint from the spec: an LLM reasons in tokens, not characters, so compliance
cannot come from a prompt like "write 30 characters." It has to come from an engineered
iterative critique loop, backed by a deterministic fallback.

## Architecture

Four layers, deliberately decoupled:

- `data/` — static product copy and swappable platform rules (`platform_rules.json`).
  Adding a fourth platform requires zero changes to `service/` or `agent/`.
- `service/` — file I/O, validation, the critique loop, and the audit log. This is the
  architectural centerpiece: it owns character counting, the retry/fallback guarantee,
  and every pass/fail decision.
- `agent/` — LangChain generation logic. Produces text; never validates it.
- `frontend/` — paste box, generate button, three result cards, full attempt trail.

`service/critique_loop.py` reaches the LLM only through an injected `VariantGenerator`
callable (a `Protocol`), so the service layer has no import-time dependency on the agent
layer and was tested with a stub generator before the agent existed.

## The core guarantee: compliance is a code guarantee, not a prompt instruction

`MAX_ATTEMPTS` (currently 3) is hardcoded in `critique_loop.py`, intentionally not read
from `platform_rules.json`. Retry budget is a system property, not a per-platform
config knob.

After all LLM attempts are exhausted, `truncate_to_limit` deterministically produces a
compliant string. It searches backward from the character limit in priority order:
last sentence-ending punctuation, then last clause-boundary punctuation, then last
whole-word boundary, then a hard character slice as a final guarantee of a non-empty
result. This exists because an earlier version cut at the nearest word boundary
unconditionally, producing outputs that ended mid-clause with no terminal punctuation.
The current priority order was written specifically so a truncated result still reads
as a complete thought wherever possible. Two further, code-level bugs were found and
fixed in this same function after it was believed done — see below.

`len()` is the single source of truth for character counts (`count_chars`), used
identically for pass/fail decisions and for the fallback. The LLM's own claim about its
output length is never trusted or consulted.

### Two more `truncate_to_limit` bugs, found after the fact

**Bug — in-token punctuation was mistaken for a sentence/clause boundary.**
`_last_punctuation_boundary` scans backward for the last `.`/`!`/`?` (sentence) or
`,`/`;`/`—` (clause) character within the limit. Without a guard, a price like
`$129.95` or a thousands separator like `7,307` reads as a sentence or clause ending —
the scanner can't tell a decimal point from a period by character alone. The fix: a
character only counts as a real boundary if it is the very last character of the text
or is immediately followed by whitespace. This is a silent correctness bug, not a
crash — the truncated output was still within the character limit, it was just cut in
the wrong place (`"$129."` instead of `"$129.95"` truncated cleanly elsewhere).

**Bug — the retention floor was checked at the wrong step (git `ddba70a`, "fixed the
truncation logic").** `MIN_WORD_BOUNDARY_FRACTION` (0.7 — keep at least 70% of the
character budget) was originally checked only when falling through to the *word*-
boundary step; the sentence- and clause-boundary steps were accepted unconditionally,
however little of the budget they preserved. A draft whose first sentence-ending
period landed at, say, 15% of the character limit was truncated to that 15% —
technically "a complete thought," but discarding most of the available message. The
fix computes `min_acceptable = max_chars * MIN_WORD_BOUNDARY_FRACTION` once and checks
it at *every* boundary step (sentence, clause, and word), not just the last one, so a
coherent-but-too-short boundary is now rejected in favor of a longer, less-coherent
one — consistent with what the function is actually for: fit as much of the message as
possible, and prefer coherence only among the options that don't waste the budget.

## The agent layer never sees the character limit

`agent/generator.py` never references `rule.max_chars`, checks length, or counts
characters, by design. This was deliberately tested against two alternatives and both
were rejected:

1. Passing `max_chars` directly into the first-attempt prompt. Rejected because it sits
   too close to the exact shortcut the spec rules out ("write 30 characters"), even
   though the actual guarantee would still come from validation and fallback.
2. Deriving a qualitative length tier from `max_chars` (e.g. "extremely short, 3-5
   words") and folding it into `rule.style` at the data layer. Rejected on reflection:
   this is the same numeric target run through a unit conversion, not independently
   useful guidance. It would reintroduce exactly what the critique loop is supposed to
   be doing instead.

The character limit reaches the model only through `feedback`, assembled by the
service layer after a real failed attempt, and included verbatim by the agent. This
keeps the actual mechanism honest: compliance is earned through iteration against a
real failure, not anticipated from a hint.

## Magnitude-aware retry feedback

Two bugs were found and fixed in `_build_feedback`, in sequence:

**Bug 1 — no anchor to correct against.** The original feedback reported the character
count and overage but never included the rejected draft's actual text. Retries were
therefore independent re-draws from the same distribution with a number attached, not
corrections. This explained both a same-length stall (two attempts landing within one
character of each other) and a regression (a retry that got *longer* than the draft it
was supposed to shorten). Raising `MAX_ATTEMPTS` from 3 to 5 did not help, which was
itself diagnostic: more independent draws don't converge better than fewer.

Fix: feedback now includes the literal previous draft and instructs editing it down,
for modest overages.

**Bug 2 — one strategy doesn't fit all overage sizes.** Tight limits (Search Ad, 30
chars) frequently produced first drafts 5–10x over limit. Instructing the model to
edit a sentence-length draft down to a 4–6 word headline through pure deletion failed
reliably; it isn't a shorter version of the same artifact, it's a different artifact.

Fix: `_build_feedback` now branches on `LARGE_OVERAGE_RATIO` (1.5×). Below the
threshold, the instruction is edit-in-place. Above it, the instruction is a fresh
short rewrite, bounded by an explicit word ceiling derived from
`max_chars // AVG_CHARS_PER_WORD` (≈6 chars/word), so "write something short" has a
concrete target instead of being left to the model's judgment.

**A false negative worth recording:** an early version of this branching logic was
reverted after informal testing suggested it performed worse than the simpler
single-instruction version. The actual root cause, found later by having Claude Code
investigate rather than prescribing a fix, was in `agent/generator.py`, not
`critique_loop.py`: the agent-layer lead-in wrapping the feedback still said "cut
words... as needed," an editing assumption that fought the service layer's large-overage
instruction to write a fresh, shorter phrase instead of editing. The branching concept
was correct from the start; the harness surrounding it was undermining it. The fix was
making the agent-layer lead-in operation-neutral, explicitly deferring to whatever the
service layer's instruction actually asks for. Lesson: a correct decision in one layer
can still fail if a neighboring layer silently assumes something different.

## Retry temperature

First-attempt temperature is 0.4 (creative exploration is fine with no correction to
converge on yet). Retry temperature is 0.2, kept deliberately non-zero: at exactly 0.0,
a retry that still overshoots reproduces an identical draft, wasting the remaining
attempts before fallback with no chance of a better outcome.

## Known limitation: emoji and character counting

`len()` counts Unicode code points, not what a human perceives as one character.
Compound, flag, and skin-tone-modified emoji are multiple code points per visual
glyph. Real ad platforms vary in how they count length (code points vs. UTF-16 code
units vs. grapheme clusters), and there is no single universal standard to target.
Social Ad style guidance explicitly invites emoji, so this is a live caveat, not
theoretical: an output that passes this system's `len()` check is not guaranteed to
pass every real platform's own character-counting rule for emoji-bearing text. Flagged
as an acknowledged limitation rather than resolved, since resolving it would mean
picking one platform's counting convention arbitrarily.

## Creative distinctness: detection, not correction

The spec requires three *completely distinct* creative variants, not merely three
compliant ones. Character-limit compliance and distinctness are separate guarantees:
the former is validated per platform inside the critique loop; the latter can only be
evaluated after all three platform results exist, by comparing them to each other. A
lexical-overlap check runs post-hoc across the three final variants and surfaces a
`Distinct` / `Too similar` signal in the UI, the same visible, audited pattern already
used for `ai_approved` / `fallback_truncated`.

The comparison metric is the Szymkiewicz–Simpson overlap coefficient
(`|A ∩ B| / min(|A|, |B|)`) on stemmed content-word sets, deliberately **not** Jaccard.
Search Ad (~4 words) and Social Ad (~20 words) differ enormously in length; Jaccard's
union denominator would dilute a real collision on the shorter variant, while
normalizing by the *smaller* set instead exposes it.

**Two metric bugs were found and fixed before the signal could be trusted.** An early
version flagged `002_jeans` as colliding on every trial: Search Ad and Display Banner
both scored 50%+ overlap purely because both legitimately named the product's category
("straight leg jeans"), while Social Ad escaped detection by writing "legs" against the
others' "leg" — a stemming gap, not genuine distinctness. Fix: an auto-derived stoplist
excludes the product's own title/category terms from the comparison (a Search Ad
legitimately needs to name what it's selling), and light stemming normalizes plural/
singular surface forms so paraphrases aren't missed. Both fixes were necessary; each
addressed a different, independent failure mode of the same metric.

**After the fix, a 7-product regression run came back 5/7 fully distinct.** The two
remaining flags were examined individually rather than treated as one problem, because
they turned out to be different failure types:

- *Agentforce Sales* (a sales-automation product) flagged all three variants for
  sharing "close/deal." This is the same structural pattern as the original jeans
  case: the product's entire stated value proposition is "closing deals faster," so
  the phrase isn't one angle among several, it's close to the only accurate way to
  state what the product does. A corrective retry instructed to avoid it would trade
  accuracy for a passing badge.
- The Cavalier shoes case flagged Search Ad and Display Banner for sharing "honest"
  and "premium," ordinary adjectives, not the product's category name. Unlike the
  Salesforce case, unused source material existed (customer count, construction
  details) that a corrective retry could plausibly have pulled instead — this is the
  one case in the regression set that looks genuinely correctable rather than
  structurally unavoidable.

**Decision: do not build corrective regeneration (retrying a variant to force
distinctness).** With only 1 of 7 products showing a plausibly correctable collision,
and a real risk that forcing correction on the other (Agentforce Sales) would degrade
copy accuracy just to clear a badge, the cost of a corrective loop (retry-budget
interaction, prompt design for "avoid this content," re-validating the full regression
set again) isn't justified by the residual it would fix. Distinctness ships as a
disclosed diagnostic: detected and honestly reported, not silently ignored and not
force-corrected at a cost the system can't defend. This mirrors the emoji-counting
caveat above — an acknowledged limitation is preferred over a fix that would cost more
than the problem it solves.

## LLM-judge layer: a second, angle-aware distinctness signal

The lexical check's own documented limitation, no concept of paraphrase or synonymy,
was confirmed in production data, not just theorized. A beginner-drone product's Search
Ad ("Beginner Drone Ready Today") and Display Banner ("Snaptain S5C Elite—built for
beginners who want reliable flight") make the identical underlying claim in different
words; both scored comfortably `DISTINCT` because they shared only one content word
after stemming and product-term exclusion. Since this failure mode by definition
produces a lexical pass, a judge gated behind a lexical flag would never see it. This
ruled out the originally-planned two-tier design (judge only on flagged cases) in favor
of an always-on judge: one call per product, evaluating all three finished variants
together against each platform's `rule.angle`, not gated behind the lexical result.

Implementation: `agent/judge.py`, one additional Anthropic call per product. Lexical
stays as an independent, always-on, parallel signal, not replaced, not a gate, and
doubles as the fallback if the judge call fails for any reason.

**JSON-parsing reliability.** The first implementation asked the model to return
free-text JSON with a "return ONLY JSON" instruction; ~20% of raw calls (3 of 13–15 in
a test batch) failed to parse. This is the same category of mistake the character-limit
guarantee itself was built to avoid, natural-language formatting instructions are not a
reliable mechanism for a structural guarantee. Fixed by switching to LangChain's
`with_structured_output` (forced tool-calling against a Pydantic schema) instead of
prompt-and-hope. Result: raw failure rate dropped to ~7% (1/15), and a full
regression-set batch came back with 0% fallback.

**Signal disagreement is real and not fully characterized.** Across a session-level
sample, lexical and judge disagreed on roughly a third of platform-instances. Read by
hand: several were the judge correctly catching paraphrase collisions lexical
structurally cannot see, the intended improvement. Others ran the opposite direction,
lexical flagged a collision the judge dismissed, and those have not been independently
checked against human judgment. The judge inherits the same "never validated against
real human review" weakness already named for the lexical metric; a more fluent-sounding
rationale is not the same thing as a correct one.

## Evaluation approach

Early iterations were evaluated by reading UI screenshots of individual runs. This is
acknowledged as weak evidence: a single stochastic sample (temperature-sampled LLM
calls) cannot reliably distinguish a genuine improvement from a lucky or unlucky draw.
At least one real regression was caused by exactly this: a correct change was reverted
based on a single unfavorable comparison, before a planned refinement was ever tested
on its own. Subsequent changes were evaluated against a fixed regression set of
real product descriptions, tracking fallback rate and attempt count per platform, to
give a measurable before/after rather than a subjective read of generated copy.