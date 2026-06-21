import { ref, watch } from 'vue';
import type { LongVideoProjectState, LongVideoSelection } from '@/types';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { defaultLongVideoProject } from '@/utils/longVideoProject';

const DEBOUNCE_MS = 400;
let persistTimer: ReturnType<typeof setTimeout> | null = null;

function loadFromStorage(): LongVideoProjectState | null {
  try {
    const raw = getItem(DQ_STORAGE.LONG_VIDEO_PROJECT);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as LongVideoProjectState;
    if (parsed?.version === 1) {
      return defaultLongVideoProject(parsed);
    }
  } catch {
    /* ignore */
  }
  return null;
}

function schedulePersist(state: LongVideoProjectState | null) {
  if (persistTimer) clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    persistNow(state);
  }, DEBOUNCE_MS);
}

function persistNow(state: LongVideoProjectState | null) {
  if (persistTimer) {
    clearTimeout(persistTimer);
    persistTimer = null;
  }
  try {
    if (state) {
      setItem(DQ_STORAGE.LONG_VIDEO_PROJECT, JSON.stringify(state));
    }
  } catch {
    /* ignore */
  }
}

export function useLongVideoProject() {
  const project = ref<LongVideoProjectState | null>(loadFromStorage());

  function setProject(next: LongVideoProjectState | null) {
    project.value = next;
    schedulePersist(next);
  }

  function patchProject(patch: Partial<LongVideoProjectState>) {
    if (!project.value) {
      setProject(defaultLongVideoProject(patch));
      return;
    }
    setProject({ ...project.value, ...patch });
  }

  function setSelection(selection: LongVideoSelection) {
    patchProject({ selection });
  }

  function initProject(partial: Partial<LongVideoProjectState> = {}) {
    if (!project.value) {
      setProject(defaultLongVideoProject(partial));
    }
  }

  watch(
    project,
    (v) => schedulePersist(v),
    { deep: true },
  );

  return {
    project,
    setProject,
    patchProject,
    setSelection,
    initProject,
    persistNow: () => persistNow(project.value),
  };
}
