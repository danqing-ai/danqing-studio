import { nextTick, type Ref } from 'vue';
import { toast } from '@/utils/feedback';
import { $tt } from '@/utils/i18n';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';

export function maybeShowCanvasWorkspaceHint() {
  if (getItem(DQ_STORAGE.CANVAS_WORKSPACE_HINT)) return;
  setItem(DQ_STORAGE.CANVAS_WORKSPACE_HINT, '1');
  toast.success($tt('canvas.workspaceEntered'));
}

/** Switch to canvas for post-generation auto-add (hint + wait for InfiniteCanvas mount). */
export async function activateCanvasViewForResults(
  viewMode: Ref<'grid' | 'canvas'>,
  onReady?: () => void
) {
  if (viewMode.value !== 'canvas') {
    viewMode.value = 'canvas';
    maybeShowCanvasWorkspaceHint();
  }
  await nextTick();
  await nextTick();
  onReady?.();
}
