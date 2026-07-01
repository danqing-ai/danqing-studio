/** Task statuses that mean the composer pipeline is still busy. */
export const ACTIVE_COMPOSER_TASK_STATUSES = new Set([
  'queued',
  'running',
  'pending',
  'submitting',
]);

/** Split multiline composer prompts into separate queue tasks (Image / Video / Audio). */
export function splitComposerPromptLines(text: string): string[] {
  return String(text || '')
    .split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}
