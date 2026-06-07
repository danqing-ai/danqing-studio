import { ref } from 'vue';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import { toast } from '@/utils/feedback';

/** In-composer LLM actions — primary battlefield; copilot is for vision + batch. */
export function useComposerLlm() {
  const isEnhancing = ref(false);
  const isReversing = ref(false);
  const isGeneratingLyrics = ref(false);

  async function enhance(
    prompt: string,
    stylePositive?: string,
    targetAction?: string,
    modelId?: string,
    options?: { quietSuccess?: boolean },
  ): Promise<string | null> {
    isEnhancing.value = true;
    try {
      const result = await api.gen.enhancePrompt({
        prompt,
        style_positive: stylePositive,
        target_action: targetAction,
        model_id: modelId,
      });
      if (!options?.quietSuccess) {
        toast.success($tt('create.enhanceComplete'));
      }
      return result.enhanced_prompt;
    } catch (e) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as Error).message
        || String(e);
      toast.error($tt('create.enhanceFailed', { msg }));
      return null;
    } finally {
      isEnhancing.value = false;
    }
  }

  async function reversePrompt(assetId: string, options?: { quietSuccess?: boolean }): Promise<string | null> {
    isReversing.value = true;
    try {
      const result = await api.gen.imageToPrompt(assetId);
      if (!options?.quietSuccess) {
        toast.success($tt('create.reverseComplete'));
      }
      return result.prompt;
    } catch (e) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as Error).message
        || String(e);
      toast.error($tt('create.reverseFailed', { msg }));
      return null;
    } finally {
      isReversing.value = false;
    }
  }

  async function generateLyrics(prompt: string, options?: { quietSuccess?: boolean }): Promise<string | null> {
    isGeneratingLyrics.value = true;
    try {
      const result = await api.gen.generateLyrics({ prompt });
      if (!options?.quietSuccess) {
        toast.success($tt('audio.lyricsGenerated'));
      }
      return result.lyrics;
    } catch (e) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as Error).message
        || String(e);
      toast.error($tt('audio.lyricsGenFailed', { msg }));
      return null;
    } finally {
      isGeneratingLyrics.value = false;
    }
  }

  return {
    isEnhancing,
    isReversing,
    isGeneratingLyrics,
    enhance,
    reversePrompt,
    generateLyrics,
  };
}
