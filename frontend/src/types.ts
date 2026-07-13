// TypeScript mirror of the JSON returned by POST /api/generate. These types
// describe the wire shape only — they hold no logic. The Python side
// (service/audit_log.py) is the single source of truth for the values.

export type VariantStatus = "ai_approved" | "fallback_truncated";

export interface Attempt {
  attempt_number: number;
  source: "llm" | "error" | "fallback";
  text: string;
  char_count: number;
  max_chars: number;
  passed: boolean;
  note: string | null;
}

export interface VariantResult {
  platform: string;
  label: string;
  max_chars: number;
  final_text: string;
  final_char_count: number;
  status: VariantStatus;
  status_label: string; // "AI-approved" | "fallback-truncated"
  // Lexical (word-overlap) distinctness verdict (parallel to status). null = not assessed.
  distinct: boolean | null;
  distinct_label: string | null; // "Distinct" | "Too similar" | null
  distinct_note: string | null; // explanation, present only when flagged too similar
  // Semantic judge verdict (parallel to the lexical one). null = judge unavailable/failed.
  judge_distinct: boolean | null;
  judge_label: string | null; // "Angles distinct" | "Same angle" | null
  judge_note: string | null; // judge rationale, present only when flagged
  signals_agree: boolean | null; // false = the two signals disagree; null = judge unavailable
  attempts: Attempt[];
}
