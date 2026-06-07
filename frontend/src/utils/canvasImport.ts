import type { CanvasSessionState } from '@/types';

export type CanvasImportFile = {
  media?: string;
  sessionId?: string;
  title?: string;
  state?: CanvasSessionState;
};

function assertRecord(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`invalid ${label}`);
  }
  return value as Record<string, unknown>;
}

export function parseCanvasImportJson(text: string): CanvasImportFile {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('invalid JSON');
  }
  const root = assertRecord(parsed, 'canvas JSON');
  const state = assertRecord(root.state, 'state');
  const items = state.items;
  if (!items || typeof items !== 'object' || Array.isArray(items)) {
    throw new Error('missing items');
  }
  return parsed as CanvasImportFile;
}
