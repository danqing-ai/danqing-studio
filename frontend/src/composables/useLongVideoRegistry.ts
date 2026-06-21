import { computed } from 'vue';
import { useRegistryStore } from '@/stores/registry';
import {
  buildResolutionSizeOptions,
  pickResolutionForModel,
  type ResolutionSizeOption,
} from '@/utils/registryParamSchema';

export interface LongVideoProfile {
  keyframe_models: string[];
  segment_models: string[];
  max_target_duration_sec: number;
  segment_model_id: string;
}

export function useLongVideoRegistry() {
  const registryStore = useRegistryStore();

  const profile = computed((): LongVideoProfile | null => {
    const reg = registryStore.registry;
    if (!reg?.models) return null;
    for (const m of Object.values(reg.models)) {
      const cfg = m as {
        id?: string;
        parameters?: {
          long_video_profile?: {
            keyframe_models?: string[];
            segment_models?: string[];
            max_target_duration_sec?: number;
          };
        };
      };
      const p = cfg.parameters?.long_video_profile;
      if (p && typeof p === 'object') {
        return {
          keyframe_models: p.keyframe_models ?? ['z-image-turbo', 'z-image', 'flux2'],
          segment_models: p.segment_models ?? ['wan-2.2-i2v-14b'],
          max_target_duration_sec: p.max_target_duration_sec ?? 120,
          segment_model_id: cfg.id || p.segment_models?.[0] || 'wan-2.2-i2v-14b',
        };
      }
    }
    return {
      keyframe_models: ['z-image-turbo', 'z-image', 'flux2'],
      segment_models: ['wan-2.2-i2v-14b', 'wan-2.2-i2v-14b-distill'],
      max_target_duration_sec: 120,
      segment_model_id: 'wan-2.2-i2v-14b',
    };
  });

  const supported = computed(() => profile.value != null);

  function modelLabel(modelId: string): string {
    const reg = registryStore.registry;
    const m = reg?.models?.[modelId] as { name?: string | { zh?: string; en?: string } } | undefined;
    if (!m?.name) return modelId;
    if (typeof m.name === 'string') return m.name;
    return m.name.zh || m.name.en || modelId;
  }

  function durationOptions(maxSec?: number): { label: string; value: number }[] {
    const max = maxSec ?? profile.value?.max_target_duration_sec ?? 120;
    const opts = [30, 60, 90, 120].filter((s) => s <= max);
    if (!opts.length) opts.push(30);
    return opts.map((value) => ({ value, label: `${value}s` }));
  }

  function modelParameters(modelId: string): Record<string, unknown> | undefined {
    const reg = registryStore.registry;
    const m = reg?.models?.[modelId] as { parameters?: Record<string, unknown> } | undefined;
    return m?.parameters;
  }

  function resolutionOptionsForModel(modelId: string): ResolutionSizeOption[] {
    return buildResolutionSizeOptions(modelParameters(modelId));
  }

  function pickOutputSizeForModel(modelId: string, saved?: string | null): string | null {
    return pickResolutionForModel(modelParameters(modelId), saved);
  }

  return {
    profile,
    supported,
    modelLabel,
    durationOptions,
    modelParameters,
    resolutionOptionsForModel,
    pickOutputSizeForModel,
    loadRegistry: () => registryStore.load(),
  };
}
