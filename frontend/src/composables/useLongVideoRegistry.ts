import { computed } from 'vue';
import { useRegistryStore } from '@/stores/registry';
import { $mn } from '@/utils/i18n';
import {
  buildResolutionSizeOptions,
  pickResolutionForModel,
  type ResolutionSizeOption,
} from '@/utils/registryParamSchema';
import { isBerniniRenderer } from '@/utils/videoEditSource';

export interface LongVideoProfile {
  keyframe_models: string[];
  segment_models: string[];
  max_target_duration_sec: number;
  segment_model_id: string;
}

type RegistryModelRow = {
  media?: string;
  actions?: Record<string, unknown>;
  name?: string | { zh?: string; en?: string };
  parameters?: Record<string, unknown>;
  commercial_use_allowed?: boolean;
};

function modelRow(reg: ReturnType<typeof useRegistryStore>['registry'], modelId: string): RegistryModelRow | undefined {
  return reg?.models?.[modelId] as RegistryModelRow | undefined;
}

function profileFromRegistry(reg: NonNullable<ReturnType<typeof useRegistryStore>['registry']>): LongVideoProfile | null {
  for (const m of Object.values(reg.models ?? {})) {
    const cfg = m as RegistryModelRow & { id?: string };
    const p = cfg.parameters?.long_video_profile as
      | {
          keyframe_models?: string[];
          segment_models?: string[];
          max_target_duration_sec?: number;
        }
      | undefined;
    if (p && typeof p === 'object') {
      return {
        keyframe_models: p.keyframe_models ?? [],
        segment_models: p.segment_models ?? [],
        max_target_duration_sec: p.max_target_duration_sec ?? 120,
        segment_model_id: cfg.id || p.segment_models?.[0] || 'wan-2.2-i2v-14b',
      };
    }
  }
  return null;
}

function sortWithSuggestions(ids: string[], suggestions: string[]): string[] {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const id of suggestions) {
    if (ids.includes(id) && !seen.has(id)) {
      ordered.push(id);
      seen.add(id);
    }
  }
  for (const id of ids.sort()) {
    if (!seen.has(id)) ordered.push(id);
  }
  return ordered;
}

export function useLongVideoRegistry() {
  const registryStore = useRegistryStore();

  const profile = computed((): LongVideoProfile | null => {
    const reg = registryStore.registry;
    if (!reg?.models) return null;
    const fromReg = profileFromRegistry(reg);
    const keyframeIds = listImageCreateModelIds();
    const segmentIds = listVideoAnimateModelIds();
    return {
      keyframe_models: fromReg?.keyframe_models?.length
        ? sortWithSuggestions(keyframeIds, fromReg.keyframe_models)
        : keyframeIds,
      segment_models: fromReg?.segment_models?.length
        ? sortWithSuggestions(segmentIds, fromReg.segment_models)
        : segmentIds,
      max_target_duration_sec: fromReg?.max_target_duration_sec ?? 120,
      segment_model_id: fromReg?.segment_model_id || segmentIds[0] || 'wan-2.2-i2v-14b',
    };
  });

  const supported = computed(() => profile.value != null && profile.value.keyframe_models.length > 0);

  function listImageCreateModelIds(): string[] {
    const reg = registryStore.registry?.models ?? {};
    return Object.keys(reg).filter((id) => {
      const m = reg[id] as RegistryModelRow;
      return m?.media === 'image' && Boolean(m.actions?.create);
    });
  }

  function listVideoAnimateModelIds(): string[] {
    const reg = registryStore.registry?.models ?? {};
    return Object.keys(reg).filter((id) => {
      const m = reg[id] as RegistryModelRow;
      return m?.media === 'video' && Boolean(m.actions?.animate);
    });
  }

  function modelOptions(ids: string[]) {
    const reg = registryStore.registry?.models ?? {};
    return ids.map((id) => {
      const m = reg[id] as RegistryModelRow | undefined;
      return {
        label: m ? $mn(m, id) : id,
        value: id,
        commercialUseAllowed: Boolean(m?.commercial_use_allowed),
      };
    });
  }

  function modelLabel(modelId: string): string {
    const m = modelRow(registryStore.registry, modelId);
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
    return modelRow(registryStore.registry, modelId)?.parameters;
  }

  function modelHasBerniniRenderer(modelId: string): boolean {
    return isBerniniRenderer(modelParameters(modelId));
  }

  function defaultShiftForModel(modelId: string, fallback = 12): number {
    const params = modelParameters(modelId);
    const shift = params?.shift;
    if (typeof shift === 'number') return shift;
    const schema = params?.parameters as Record<string, { default?: number }> | undefined;
    const nested = schema?.shift?.default;
    return typeof nested === 'number' ? nested : fallback;
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
    modelHasBerniniRenderer,
    defaultShiftForModel,
    listImageCreateModelIds,
    listVideoAnimateModelIds,
    modelOptions,
    resolutionOptionsForModel,
    pickOutputSizeForModel,
    loadRegistry: () => registryStore.load(),
  };
}
