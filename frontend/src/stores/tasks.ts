import { reactive } from 'vue';
import { defineStore } from 'pinia';
import { api } from '@/utils/api';
import type { QueueState } from '@/types';

export interface LiveTaskProgress {
  progress?: number;
  step?: number;
  total?: number;
  eta_seconds?: number;
  progressMessage?: string;
}

export const useTasksStore = defineStore('tasks', () => {
  const queueState = reactive<QueueState>({
    running: [],
    queued: [],
  });

  const liveTaskProgress = reactive<Record<string, LiveTaskProgress>>({});

  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let pollRefCount = 0;

  function patchLiveTaskProgress(taskId: string, patch: LiveTaskProgress) {
    if (!taskId || !patch || Object.keys(patch).length === 0) return;
    const prev = liveTaskProgress[taskId] || {};
    const next = { ...prev };
    if (typeof patch.progress === 'number') next.progress = patch.progress;
    if (patch.step != null) next.step = patch.step;
    if (patch.total != null) next.total = patch.total;
    if (patch.eta_seconds != null) next.eta_seconds = patch.eta_seconds;
    if (Object.prototype.hasOwnProperty.call(patch, 'progressMessage')) {
      next.progressMessage = patch.progressMessage;
    }
    liveTaskProgress[taskId] = next;
  }

  function clearLiveTaskProgress(taskId: string) {
    if (taskId && liveTaskProgress[taskId] != null) {
      delete liveTaskProgress[taskId];
    }
  }

  const logStreams = new Map<string, EventSource>();

  function closeTaskLogStream(taskId: string) {
    const es = logStreams.get(taskId);
    if (es) {
      try {
        es.close();
      } catch {
        /* ignore */
      }
      logStreams.delete(taskId);
    }
  }

  function openTaskLogStream(
    taskId: string,
    callbacks: {
      onLog?: (data: unknown) => void;
      onStatus?: (data: unknown) => void;
      onProgress?: (data: unknown) => void;
      onResult?: (data: unknown) => void;
      onDone?: (data: unknown) => void;
      onError?: (event: Event) => void;
    }
  ): EventSource {
    closeTaskLogStream(taskId);
    const url = api.tasks.logStreamUrl(taskId);
    const eventSource = new EventSource(url);
    logStreams.set(taskId, eventSource);

    eventSource.addEventListener('log', (event) => {
      const data = JSON.parse(event.data);
      callbacks.onLog?.(data);
    });

    eventSource.addEventListener('progress', (event) => {
      const data = JSON.parse(event.data);
      const patch: LiveTaskProgress = {};
      if (typeof data.progress === 'number') patch.progress = data.progress;
      if (data.step != null) patch.step = data.step;
      if (data.total != null) patch.total = data.total;
      if (data.eta_seconds != null) patch.eta_seconds = data.eta_seconds;
      if (Object.prototype.hasOwnProperty.call(data, 'message')) {
        patch.progressMessage = data.message;
      }
      patchLiveTaskProgress(taskId, patch);
      callbacks.onProgress?.(data);
      if (!callbacks.onProgress && callbacks.onStatus) {
        callbacks.onStatus?.({
          status: 'running',
          progress: data.progress,
          step: data.step,
          total: data.total,
          eta_seconds: data.eta_seconds,
        });
      }
    });

    eventSource.addEventListener('status', (event) => {
      const data = JSON.parse(event.data);
      if (typeof data.progress === 'number') {
        patchLiveTaskProgress(taskId, { progress: data.progress });
      }
      callbacks.onStatus?.(data);
    });

    eventSource.addEventListener('result', (event) => {
      const data = JSON.parse(event.data);
      callbacks.onResult?.(data);
    });

    eventSource.addEventListener('done', (event) => {
      const data = JSON.parse(event.data);
      callbacks.onDone?.(data);
      clearLiveTaskProgress(taskId);
      closeTaskLogStream(taskId);
    });

    eventSource.addEventListener('error', (event) => {
      callbacks.onError?.(event);
      clearLiveTaskProgress(taskId);
      closeTaskLogStream(taskId);
    });

    return eventSource;
  }

  async function pollQueueOnce(): Promise<void> {
    try {
      const data = await api.gen.getQueue();
      queueState.running = data.running || [];
      queueState.queued = data.queued || [];
      for (const t of queueState.running || []) {
        if (t?.id && !logStreams.has(t.id)) {
          openTaskLogStream(t.id, {});
        }
      }
    } catch (e) {
      console.error('TasksStore: queue poll failed', e);
    }
  }

  function ensureQueuePoller(): void {
    pollRefCount += 1;
    if (!pollTimer) {
      pollQueueOnce();
      pollTimer = setInterval(pollQueueOnce, 2000);
    }
  }

  function releaseQueuePoller(): void {
    pollRefCount = Math.max(0, pollRefCount - 1);
    if (pollRefCount === 0 && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  return {
    queueState,
    liveTaskProgress,
    patchLiveTaskProgress,
    clearLiveTaskProgress,
    openTaskLogStream,
    closeTaskLogStream,
    pollQueueOnce,
    ensureQueuePoller,
    releaseQueuePoller,
  };
});