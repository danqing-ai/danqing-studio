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
      const lyrics = (result.lyrics || '').trim();
      if (!lyrics) {
        toast.error($tt('audio.lyricsGenFailed', { msg: $tt('audio.lyricsGenEmpty') }));
        return null;
      }
      if (!options?.quietSuccess) {
        toast.success($tt('audio.lyricsGenerated'));
      }
      return lyrics;
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

  const isStoryboardExpanding = ref(false);

  async function storyboardLongVideo(
    body: {
      prompt: string;
      target_duration_sec: number;
      initial_duration_sec?: number;
      segment_extend_sec?: number;
      segment_duration_sec?: number;
      reference_duration_sec?: number;
      style_positive?: string;
      locale?: string;
      use_shot_plan?: boolean;
    },
    opts?: { quietSuccess?: boolean },
  ) {
    isStoryboardExpanding.value = true;
    try {
      const result = await api.gen.longVideoStoryboard(body);
      if (!opts?.quietSuccess) {
        toast.success($tt('video.storyboardComplete'));
      }
      return result;
    } catch (e) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as Error).message
        || String(e);
      toast.error($tt('video.storyboardFailed', { msg }));
      return null;
    } finally {
      isStoryboardExpanding.value = false;
    }
  }

  return {
    isEnhancing,
    isReversing,
    isGeneratingLyrics,
    isStoryboardExpanding,
    enhance,
    reversePrompt,
    generateLyrics,
    storyboardLongVideo,
  };
}
