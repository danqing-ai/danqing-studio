import { setItem, type StorageKey } from '@/utils/storage';

export type PromptDraftMode = 'replace' | 'append';

export interface PromptDraftHandoff {
  text: string;
  mode: PromptDraftMode;
}

/** Join user prompt with LLM output (InvokeAI-style append). */
export function mergePromptText(base: string, addition: string): string {
  const b = base.trim();
  const a = addition.trim();
  if (!a) return base;
  if (!b) return a;
  return `${b}\n\n${a}`;
}

export function setPromptDraft(key: StorageKey, text: string, mode: PromptDraftMode = 'replace'): void {
  setItem(key, JSON.stringify({ text, mode }));
}

/** Read-once handoff from copilot → create views (plain string = replace). */
export function consumePromptDraft(key: StorageKey): PromptDraftHandoff | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    localStorage.removeItem(key);
    try {
      const parsed = JSON.parse(raw) as { text?: string; mode?: string };
      if (parsed && typeof parsed.text === 'string') {
        return {
          text: parsed.text,
          mode: parsed.mode === 'append' ? 'append' : 'replace',
        };
      }
    } catch {
      /* legacy plain string */
    }
    return { text: raw, mode: 'replace' };
  } catch {
    return null;
  }
}

export function applyPromptDraft(current: string, draft: PromptDraftHandoff): string {
  if (draft.mode === 'append') {
    return mergePromptText(current, draft.text);
  }
  return draft.text;
}
