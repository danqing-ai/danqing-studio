/** Handoff from Video Create → Long Video studio (router history.state). */

export const LONG_VIDEO_HANDOFF_STATE_KEY = 'longVideoHandoff';

export type LongVideoHandoffState = {
  script_text?: string;
  target_duration_sec?: number;
};

export function readLongVideoHandoffState(): LongVideoHandoffState | null {
  const raw = history.state?.[LONG_VIDEO_HANDOFF_STATE_KEY];
  if (!raw || typeof raw !== 'object') return null;
  const script = String((raw as LongVideoHandoffState).script_text || '').trim();
  if (!script) return null;
  const dur = Number((raw as LongVideoHandoffState).target_duration_sec);
  return {
    script_text: script,
    target_duration_sec: Number.isFinite(dur) && dur > 0 ? dur : undefined,
  };
}

export function clearLongVideoHandoffState(): void {
  if (!history.state?.[LONG_VIDEO_HANDOFF_STATE_KEY]) return;
  const next = { ...history.state };
  delete next[LONG_VIDEO_HANDOFF_STATE_KEY];
  window.history.replaceState(next, '');
}
