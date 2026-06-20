import { onBeforeUnmount, ref, type Ref } from 'vue';

type DurationHint = () => number | null | undefined;

/** Sync media element currentTime / duration / buffer for player scrub UI. */
export function useMediaTimeline(
  mediaEl: Ref<HTMLMediaElement | null>,
  options?: { durationHint?: DurationHint },
) {
  const currentTime = ref(0);
  const duration = ref(0);
  const bufferEnd = ref(0);

  let rafId = 0;

  function resolveDuration(el: HTMLMediaElement): number {
    const fromEl = el.duration;
    if (Number.isFinite(fromEl) && fromEl > 0) return fromEl;
    const hint = options?.durationHint?.();
    if (hint != null && Number.isFinite(hint) && hint > 0) return hint;
    return 0;
  }

  function syncFromElement() {
    const el = mediaEl.value;
    if (!el) return;
    currentTime.value = el.currentTime;
    duration.value = resolveDuration(el);
    if (el.buffered.length > 0) {
      bufferEnd.value = el.buffered.end(el.buffered.length - 1);
    }
  }

  function stopProgressLoop() {
    if (!rafId) return;
    cancelAnimationFrame(rafId);
    rafId = 0;
  }

  function progressLoop() {
    const el = mediaEl.value;
    if (!el || el.paused || el.ended) {
      rafId = 0;
      return;
    }
    syncFromElement();
    rafId = requestAnimationFrame(progressLoop);
  }

  function startProgressLoop() {
    stopProgressLoop();
    syncFromElement();
    rafId = requestAnimationFrame(progressLoop);
  }

  function onTimeUpdate() {
    syncFromElement();
  }

  function onLoadedMetadata() {
    syncFromElement();
  }

  function onDurationChange() {
    syncFromElement();
  }

  function onPlay() {
    startProgressLoop();
  }

  function onPause() {
    stopProgressLoop();
    syncFromElement();
  }

  function onEnded() {
    stopProgressLoop();
    currentTime.value = 0;
  }

  function resetTimeline(clearDuration = true) {
    stopProgressLoop();
    currentTime.value = 0;
    if (clearDuration) duration.value = 0;
    bufferEnd.value = 0;
  }

  onBeforeUnmount(stopProgressLoop);

  return {
    currentTime,
    duration,
    bufferEnd,
    syncFromElement,
    resetTimeline,
    stopProgressLoop,
    onTimeUpdate,
    onLoadedMetadata,
    onDurationChange,
    onPlay,
    onPause,
    onEnded,
  };
}
