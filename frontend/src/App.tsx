import { useState } from "react";
import type { Attempt, VariantResult } from "./types";

// Relative URL: proxied to :8000 by Vite in dev, same-origin in shipped mode.
const GENERATE_URL = "/api/generate";

async function requestVariants(productSource: string): Promise<VariantResult[]> {
  const res = await fetch(GENERATE_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ product_source: productSource }),
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      // non-JSON error body — keep the status-based message
    }
    throw new Error(detail);
  }
  return (await res.json()) as VariantResult[];
}

function AttemptRow({ attempt }: { attempt: Attempt }) {
  return (
    <li className={`attempt ${attempt.passed ? "attempt--pass" : "attempt--fail"}`}>
      <div className="attempt__head">
        <span className="attempt__num">Attempt {attempt.attempt_number}</span>
        <span className="attempt__source">{attempt.source}</span>
        <span className="attempt__count">
          {attempt.char_count} / {attempt.max_chars}
        </span>
        <span className="attempt__verdict">{attempt.passed ? "PASS" : "FAIL"}</span>
      </div>
      {attempt.text && <p className="attempt__text">{attempt.text}</p>}
      {attempt.note && <p className="attempt__note">{attempt.note}</p>}
    </li>
  );
}

function VariantCard({ result }: { result: VariantResult }) {
  const overLimit = result.final_char_count > result.max_chars;
  return (
    <article className="card">
      <header className="card__head">
        <h2 className="card__title">{result.label}</h2>
        {/* Badge text comes straight from the API fields, not re-derived here. Two
            independent distinctness signals sit beside the char-limit status badge: the
            lexical word-overlap check and the semantic judge. Both are shown; neither
            overrides the other. The judge badge is omitted when the judge was unavailable. */}
        <div className="card__badges">
          <span className={`badge badge--${result.status}`}>{result.status_label}</span>
          {result.distinct !== null && (
            <span className={`badge badge--${result.distinct ? "distinct" : "similar"}`}>
              lexical: {result.distinct_label}
            </span>
          )}
          {result.judge_distinct !== null && (
            <span className={`badge badge--judge-${result.judge_distinct ? "distinct" : "similar"}`}>
              judge: {result.judge_label}
            </span>
          )}
        </div>
      </header>

      {result.signals_agree === false && (
        <p className="card__disagree">
          ⚠ Signals disagree — the word-overlap check and the semantic judge reached
          different verdicts. Review both below.
        </p>
      )}

      <p className="card__text">{result.final_text}</p>

      <div className={`card__count ${overLimit ? "card__count--over" : ""}`}>
        {result.final_char_count} / {result.max_chars} chars
      </div>

      {result.distinct_note && (
        <p className="card__distinct-note">Lexical: {result.distinct_note}</p>
      )}

      {result.judge_note && (
        <p className="card__distinct-note">Judge: {result.judge_note}</p>
      )}

      <details className="trail">
        <summary>
          Attempt trail ({result.attempts.length})
        </summary>
        <ol className="trail__list">
          {result.attempts.map((a) => (
            <AttemptRow key={a.attempt_number} attempt={a} />
          ))}
        </ol>
      </details>
    </article>
  );
}

export default function App() {
  const [productSource, setProductSource] = useState("");
  const [results, setResults] = useState<VariantResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = productSource.trim().length > 0 && !loading;

  async function handleGenerate() {
    if (!canSubmit) return;
    setLoading(true);
    setError(null);
    try {
      setResults(await requestVariants(productSource));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1 className="app__title">Ad Creative Variant Auditor</h1>
      <p className="app__lede">
        Paste a long-form product description. Every variant is validated against
        its platform's character limit by a critique loop — never by trusting the
        model's own claim.
      </p>

      <textarea
        className="app__input"
        placeholder="Paste product copy here…"
        value={productSource}
        onChange={(e) => setProductSource(e.target.value)}
        rows={10}
      />

      <div className="app__actions">
        <button className="app__button" onClick={handleGenerate} disabled={!canSubmit}>
          {loading ? "Generating…" : "Generate"}
        </button>
      </div>

      {error && <div className="app__error">{error}</div>}

      {results && (
        <section className="cards">
          {results.map((r) => (
            <VariantCard key={r.platform} result={r} />
          ))}
        </section>
      )}
    </main>
  );
}
