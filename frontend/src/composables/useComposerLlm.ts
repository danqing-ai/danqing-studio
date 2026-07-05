import { ref } from 'vue';
import { api, type ScriptParseDecomposeResult, type ScriptParseExpandResult } from '@/utils/api';
import {
  enhancePromptViaChat,
  generateLyricsViaChat,
  imageToPromptViaChat,
} from '@/utils/llmMessages';
import { $tt } from '@/utils/i18n';
import { toast } from '@/utils/feedback';
import { isScriptParseError } from '@/utils/scriptParseError';

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

  const isStoryboardGenerating = ref(false);
  const isChapterAnalyzing = ref(false);

  async function scriptParseDecompose(
    body: {
      script_text: string;
      title?: string;
      locale?: string;
      long_video_project_id?: string;
      model?: string;
    },
    opts?: { quietSuccess?: boolean; onProgress?: (phase: string, message: string) => void },
  ): Promise<ScriptParseDecomposeResult | null> {
    isChapterAnalyzing.value = true;
    try {
      const result = await api.gen.scriptParseDecomposeStream(body, opts?.onProgress);
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

  async function scriptParseExpand(
    body: {
      script_artifact: Record<string, unknown>;
      locale?: string;
      target_duration_sec?: number;
      segment_duration_sec?: number;
      max_clip_sec?: number;
      long_video_project_id?: string;
      model?: string;
      beat_indices?: number[];
    },
    opts?: {
      quietSuccess?: boolean;
      onProgress?: (phase: string, message: string) => void;
      onError?: (msg: string, qualityIssues?: import('@/utils/scriptParseError').ScriptParseQualityIssue[]) => void;
    },
  ): Promise<ScriptParseExpandResult | null> {
    isStoryboardGenerating.value = true;
    try {
      const result = await api.gen.scriptParseExpandStream(body, opts?.onProgress);
      if (!opts?.quietSuccess) {
        toast.success($tt('video.longVideoScriptGenerateStoryboardDone'));
      }
      return result;
    } catch (e) {
      const msg = isScriptParseError(e)
        ? e.message
        : (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
          || (e as Error).message
          || String(e);
      const qualityIssues = isScriptParseError(e) ? e.qualityIssues : undefined;
      opts?.onError?.(msg, qualityIssues);
      toast.error($tt('video.longVideoScriptGenerateStoryboardFailed', { msg }));
      return null;
    } finally {
      isStoryboardGenerating.value = false;
    }
  }

  async function scriptParseExpandBeat(
    body: {
      script_artifact: Record<string, unknown>;
      beat_index: number;
      existing_shots?: Array<Record<string, unknown>>;
      locale?: string;
      target_duration_sec?: number;
      segment_duration_sec?: number;
      max_clip_sec?: number;
      long_video_project_id?: string;
      model?: string;
    },
    opts?: {
      quietSuccess?: boolean;
      onProgress?: (phase: string, message: string) => void;
      onError?: (msg: string, qualityIssues?: import('@/utils/scriptParseError').ScriptParseQualityIssue[]) => void;
    },
  ): Promise<ScriptParseExpandResult | null> {
    isStoryboardGenerating.value = true;
    try {
      const result = await api.gen.scriptParseExpandBeatStream(body, opts?.onProgress);
      if (!opts?.quietSuccess) {
        toast.success($tt('video.longVideoBeatExpandDone'));
      }
      return result;
    } catch (e) {
      const msg = isScriptParseError(e)
        ? e.message
        : (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
          || (e as Error).message
          || String(e);
      const qualityIssues = isScriptParseError(e) ? e.qualityIssues : undefined;
      opts?.onError?.(msg, qualityIssues);
      toast.error($tt('video.longVideoBeatExpandFailed', { msg }));
      return null;
    } finally {
      isStoryboardGenerating.value = false;
    }
  }

  /** Decompose + expand (legacy one-shot). */
  async function analyzeLongVideoChapter(
    body: {
      chapter_text: string;
      chapter_title?: string;
      locale?: string;
      target_duration_sec?: number;
      segment_duration_sec?: number;
      max_clip_sec?: number;
      long_video_project_id?: string;
      model?: string;
    },
    opts?: { quietSuccess?: boolean; onProgress?: (phase: string, message: string) => void },
  ): Promise<ScriptParseExpandResult | null> {
    const decomposed = await scriptParseDecompose(
      {
        script_text: body.chapter_text,
        title: body.chapter_title,
        locale: body.locale,
        long_video_project_id: body.long_video_project_id,
        model: body.model,
      },
      { quietSuccess: true, onProgress: opts?.onProgress },
    );
    if (!decomposed) return null;
    return scriptParseExpand(
      {
        script_artifact: decomposed.script_artifact,
        locale: body.locale,
        target_duration_sec: body.target_duration_sec,
        segment_duration_sec: body.segment_duration_sec,
        max_clip_sec: body.max_clip_sec,
        long_video_project_id: body.long_video_project_id,
        model: body.model,
      },
      opts,
    );
  }

  return {
    isEnhancing,
    isReversing,
    isGeneratingLyrics,
    isChapterAnalyzing,
    isStoryboardGenerating,
    enhance,
    reversePrompt,
    generateLyrics,
    scriptParseDecompose,
    scriptParseExpand,
    scriptParseExpandBeat,
    analyzeLongVideoChapter,
  };
}
