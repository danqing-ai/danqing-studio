/** Normalize script-parse SSE / HTTP errors for Long Video UI. */

export type ScriptParseQualityIssue = {
  code?: string;
  message: string;
  severity?: 'warning' | 'critical';
  shot_index?: number | null;
  beat_index?: number | null;
};

export class ScriptParseError extends Error {
  qualityIssues: ScriptParseQualityIssue[];

  constructor(message: string, qualityIssues: ScriptParseQualityIssue[] = []) {
    super(message);
    this.name = 'ScriptParseError';
    this.qualityIssues = qualityIssues;
  }
}

export function isScriptParseError(err: unknown): err is ScriptParseError {
  return err instanceof ScriptParseError;
}

export function criticalIssuesFromList(issues: ScriptParseQualityIssue[]): ScriptParseQualityIssue[] {
  return issues.filter((i) => i.severity === 'critical');
}

export function mergeQualityIssues(
  existing: ScriptParseQualityIssue[] | undefined,
  incoming: ScriptParseQualityIssue[],
): ScriptParseQualityIssue[] {
  const base = (existing ?? []).filter((i) => i.code !== 'parse_blocked');
  const seen = new Set(base.map((i) => `${i.code}|${i.message}`));
  const merged = [...base];
  for (const row of incoming) {
    const key = `${row.code ?? ''}|${row.message}`;
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(row);
  }
  return merged;
}
