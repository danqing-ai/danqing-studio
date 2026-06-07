import { ref } from 'vue';
import { toast } from '@/utils/feedback';
import { $tt } from '@/utils/i18n';
import { mergePromptText } from '@/utils/promptApply';

export interface PromptApplyPending {
  previous: string;
  result: string;
}

/** After LLM returns text: auto-fill when empty, else show replace/append strip. */
export function usePromptApplyOffer() {
  const pending = ref<PromptApplyPending | null>(null);

  function offer(previous: string, result: string, onAutoApply: (text: string) => void): void {
    const prev = previous.trim();
    const next = result.trim();
    if (!next) return;
    if (prev === next) {
      toast.success($tt('create.llmApplySame'));
      return;
    }
    if (!prev) {
      onAutoApply(next);
      toast.success($tt('create.llmApplyDone'));
      return;
    }
    pending.value = { previous, result: next };
  }

  function clear() {
    pending.value = null;
  }

  function applyReplace(onApply: (text: string) => void) {
    if (!pending.value) return;
    onApply(pending.value.result);
    clear();
    toast.success($tt('create.llmApplyDone'));
  }

  function applyAppend(getter: () => string, onApply: (text: string) => void) {
    if (!pending.value) return;
    onApply(mergePromptText(getter(), pending.value.result));
    clear();
    toast.success($tt('create.llmApplyDone'));
  }

  return { pending, offer, clear, applyReplace, applyAppend };
}
