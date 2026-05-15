/**
 * Media queue composable — replaces window.DQMediaQueue
 */
import type { Task } from '@/types';
import type { LiveTaskProgress } from '@/stores/tasks';

interface NormalizedTask {
  id: string;
  kind: string;
  progress: number;
  step?: number;
  total?: number;
  progressMessage?: string;
  priority: number;
  estimated_wait_seconds?: number;
  params: {
    model: string;
    prompt: string;
  };
}

export function normalizeTaskRow(t: Task, liveById: Record<string, LiveTaskProgress>): NormalizedTask {
  const pr = t.priority;
  let progressMessage = t.progressMessage;
  const live = t?.id != null ? liveById[t.id] : null;
  let progress = typeof t.progress === 'number' ? t.progress : 0;
  let step = t.step;
  let total = t.total;
  if (live) {
    if (typeof live.progress === 'number') progress = live.progress;
    if (live.step != null) step = live.step;
    if (live.total != null) total = live.total;
    if (Object.prototype.hasOwnProperty.call(live, 'progressMessage')) {
      progressMessage = live.progressMessage;
    }
  }
  return {
    id: t.id,
    kind: t.kind,
    progress,
    step,
    total,
    progressMessage,
    priority: typeof pr === 'number' ? pr : 100,
    estimated_wait_seconds: t.estimated_wait_seconds,
    params: {
      model: (t.params?.model as string) || '',
      prompt: (t.params?.prompt as string) || '',
    },
  };
}

export function filterByKindPrefix(
  arr: Task[],
  prefix: string,
  liveById: Record<string, LiveTaskProgress>
): NormalizedTask[] {
  const p = String(prefix || '');
  return (arr || [])
    .filter((t) => String(t.kind || '').startsWith(p))
    .map((t) => normalizeTaskRow(t, liveById));
}

export function snapshotFullQueue(
  running: Task[],
  queued: Task[],
  liveProgress: Record<string, LiveTaskProgress>
): { running: NormalizedTask[]; queued: NormalizedTask[] } {
  return {
    running: (running || []).map((t) => normalizeTaskRow(t, liveProgress)),
    queued: (queued || []).map((t) => normalizeTaskRow(t, liveProgress)),
  };
}

export function tasksForMedia(
  running: Task[],
  queued: Task[],
  liveProgress: Record<string, LiveTaskProgress>,
  media: 'image' | 'video'
): { running: NormalizedTask[]; queued: NormalizedTask[] } {
  const prefix = media === 'video' ? 'video' : 'image';
  return {
    running: filterByKindPrefix(running, prefix, liveProgress),
    queued: filterByKindPrefix(queued, prefix, liveProgress),
  };
}