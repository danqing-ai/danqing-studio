import { onUnmounted } from 'vue';

/**
 * Subscribe to a `document` event for the lifetime of the calling component.
 * Prefer over `window` listeners for page-scoped shortcuts (still global DOM, not `window`).
 */
export function useDocumentEvent<K extends keyof DocumentEventMap>(
  type: K,
  listener: (ev: DocumentEventMap[K]) => void,
  options?: boolean | AddEventListenerOptions
): void {
  const bound = listener as EventListener;
  document.addEventListener(type, bound, options);
  onUnmounted(() => {
    document.removeEventListener(type, bound, options);
  });
}
