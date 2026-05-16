/**
 * Typed in-app event bus (no `window.dispatchEvent` / `CustomEvent`).
 * Extend `AppEventMap` and `ensureKnownKeys` when adding signals.
 */
export type AppEventMap = {
  'open-global-task-queue': void;
};

type Handler<K extends keyof AppEventMap> = (payload: AppEventMap[K]) => void;

type AnyHandler = (payload: unknown) => void;

const store = new Map<keyof AppEventMap, Set<AnyHandler>>();

function bucket<K extends keyof AppEventMap>(type: K): Set<AnyHandler> {
  let s = store.get(type);
  if (!s) {
    s = new Set();
    store.set(type, s);
  }
  return s;
}

export const appEvents = {
  on<K extends keyof AppEventMap>(type: K, fn: Handler<K>): void {
    bucket(type).add(fn as AnyHandler);
  },
  off<K extends keyof AppEventMap>(type: K, fn: Handler<K>): void {
    bucket(type).delete(fn as AnyHandler);
  },
  emit<K extends keyof AppEventMap>(type: K, payload: AppEventMap[K]): void {
    for (const fn of bucket(type)) {
      try {
        (fn as Handler<K>)(payload);
      } catch (e) {
        console.error('[appEvents]', type, e);
      }
    }
  },
};

/** Open the global task queue drawer (shell). */
export function openGlobalTaskQueue(): void {
  appEvents.emit('open-global-task-queue', undefined as void);
}
