import { ref } from 'vue';
import { api, type LongVideoChapterAnalyzeResult } from '@/utils/api';
import {
  enhancePromptViaChat,
  generateLyricsViaChat,
  imageToPromptViaChat,
} from '@/utils/llmMessages';
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
    _modelId?: string,
    options?: { quietSuccess?: boolean },
  ): Promise<string | null> {
    isEnhancing.value = true;
    try {
      const text = await enhancePromptViaChat(prompt, { stylePositive, targetAction });
      if (!options?.quietSuccess) {
        toast.success($tt('create.enhanceComplete'));
      }
      return text;
    } catch (e) {
      const err = e as { code?: string; message?: string; response?: { data?: { detail?: string } } };
      const msg = err.response?.data?.detail
        || (err.code === 'ECONNABORTED' ? $tt('create.enhanceTimeout') : '')
        || err.message
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
      const text = await imageToPromptViaChat(assetId);
      if (!options?.quietSuccess) {
        toast.success($tt('create.reverseComplete'));
      }
      return text;
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
      const lyrics = await generateLyricsViaChat(prompt);
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
  const isChapterAnalyzing = ref(false);

  async function analyzeLongVideoChapter(
    body: {
      chapter_text: string;
      chapter_title?: string;
      locale?: string;
      target_duration_sec?: number;
      segment_duration_sec?: number;
      max_clip_sec?: number;
      long_video_project_id?: string;
    },
    opts?: { quietSuccess?: boolean; onProgress?: (phase: string, message: string) => void },
  ): Promise<LongVideoChapterAnalyzeResult | null> {
    isChapterAnalyzing.value = true;
    try {
      const result = await api.gen.longVideoChapterAnalyzeStream(body, opts?.onProgress);
      if (!opts?.quietSuccess) {
        toast.success($tt('video.longVideoChapterAnalyzeComplete'));
      }
      return result;
    } catch (e) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as Error).message
        || String(e);
      toast.error($tt('video.longVideoChapterAnalyzeFailed', { msg }));
      return null;
    } finally {
      isChapterAnalyzing.value = false;
    }
  }

  async function storyboardLongVideo(
    body: Parameters<typeof api.gen.longVideoStoryboard>[0],
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
    isChapterAnalyzing,
    enhance,
    reversePrompt,
    generateLyrics,
    analyzeLongVideoChapter,
    storyboardLongVideo,
  };
}
