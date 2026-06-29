import type { LongVideoProjectActivityItem } from '@/utils/api';

const TERMINAL_EVENTS = new Set(['task_completed', 'task_failed', 'task_cancelled']);

export function activityEventStatus(eventType: string): string {
  if (eventType === 'task_submitted') return 'queued';
  if (eventType.startsWith('task_')) return eventType.slice('task_'.length);
  return eventType;
}

export function activityStatusLabelKey(status: string): string {
  const map: Record<string, string> = {
    queued: 'video.longVideoActivityStatusQueued',
    running: 'video.longVideoActivityStatusRunning',
    completed: 'video.longVideoActivityStatusCompleted',
    failed: 'video.longVideoActivityStatusFailed',
    cancelled: 'video.longVideoActivityStatusCancelled',
  };
  return map[status] ?? 'video.longVideoActivityStatusUnknown';
}

/** Latest generation task per phase for one shot. */
export function latestShotTasksByPhase(
  items: LongVideoProjectActivityItem[],
  phases: string[],
): Map<string, { taskId: string; status: string; at: string; summary: string }> {
  const out = new Map<string, { taskId: string; status: string; at: string; summary: string }>();
  const byPhaseTask = new Map<string, LongVideoProjectActivityItem[]>();

  for (const item of items) {
    const phase = item.phase || '';
    const taskId = String(item.task_id || '').trim();
    if (!taskId || !phases.includes(phase)) continue;
    const key = `${phase}:${taskId}`;
    const bucket = byPhaseTask.get(key) ?? [];
    bucket.push(item);
    byPhaseTask.set(key, bucket);
  }

  for (const [key, events] of byPhaseTask) {
    const [phase] = key.split(':');
    events.sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
    const latest = events[0];
    const terminal = events.find((e) => TERMINAL_EVENTS.has(e.event_type));
    const status = terminal
      ? activityEventStatus(terminal.event_type)
      : activityEventStatus(latest.event_type);
    const at = terminal?.created_at ?? latest.created_at;
    const cur = out.get(phase);
    if (!cur || String(at).localeCompare(String(cur.at)) > 0) {
      out.set(phase, {
        taskId: String(latest.task_id),
        status,
        at: String(at),
        summary: latest.summary || '',
      });
    }
  }
  return out;
}

/** Latest task for one entity (cast look / scene look) matched via activity detail fields. */
export function latestEntityTask(
  items: LongVideoProjectActivityItem[],
  phase: string,
  match: Record<string, string>,
): { taskId: string; status: string; at: string } | null {
  const phases = [phase];
  const filtered = items.filter((item) => {
    if ((item.phase || '') !== phase) return false;
    const detail = item.detail ?? {};
    return Object.entries(match).every(([k, v]) => String(detail[k] ?? '') === v);
  });
  if (!filtered.length) return null;
  const map = latestShotTasksByPhase(filtered, phases);
  const row = map.get(phase);
  if (!row) return null;
  return { taskId: row.taskId, status: row.status, at: row.at };
}

export function shortActivityId(id: string, head = 8, tail = 4): string {
  const s = String(id || '').trim();
  if (s.length <= head + tail + 1) return s;
  return `${s.slice(0, head)}…${s.slice(-tail)}`;
}

export function formatActivityTime(iso: string, locale: string): string {
  if (!iso) return '';
  try {
    return new Intl.DateTimeFormat(locale.startsWith('zh') ? 'zh-CN' : 'en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}
