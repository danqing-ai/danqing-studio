import { computed, reactive, ref, watch, type Ref } from 'vue';
import { api } from '@/utils/api';
import { useRegistryStore } from '@/stores/registry';
import { applyDefaults, hasDeviation, normalizeParamsDef } from '@/utils/registryParamSchema';
import {
  applyLoraComposeOverrides,
  findCompatibleLora,
} from '@/utils/loraAdapterMeta';

export type SegmentComposeParams = {
  steps: number;
  guide_scale: number;
  seed: string;
  fps: number;
  shift: number;
  negative_prompt: string;
  lora: string;
  lora_scale: number;
};

function defaultParams(): SegmentComposeParams {
  return {
    steps: 40,
    guide_scale: 3.0,
    seed: '',
    fps: 16,
    shift: 12.0,
    negative_prompt: '',
    lora: '',
    lora_scale: 1.0,
  };
}

export function useLongVideoSegmentCompose(modelId: Ref<string>) {
  const registryStore = useRegistryStore();
  const params = reactive<SegmentComposeParams>(defaultParams());
  const defaultSnapshot = ref<SegmentComposeParams>(defaultParams());
  const compatibleLoras = ref<Record<string, unknown>[]>([]);

  const currentModelConfig = computed(() => {
    const id = modelId.value;
    if (!id) return null;
    return (registryStore.registry?.models?.[id] as Record<string, unknown> | undefined) ?? null;
  });

  const paramSchema = computed(() =>
    normalizeParamsDef(currentModelConfig.value?.parameters as Record<string, unknown> | undefined),
  );

  const hasCustomParams = computed(() => {
    const cfg = currentModelConfig.value?.parameters as Record<string, unknown> | undefined;
    if (!cfg) return false;
    return hasDeviation(cfg, params as unknown as Record<string, unknown>);
  });

  const showNegativePrompt = computed(() =>
    Boolean((currentModelConfig.value?.parameters as { negative_prompt_support?: boolean } | undefined)?.negative_prompt_support),
  );

  const showSeedField = computed(() => {
    const raw = currentModelConfig.value?.parameters as { seed_support?: boolean } | undefined;
    return raw?.seed_support !== false;
  });

  const showLora = computed(
    () =>
      Boolean((currentModelConfig.value?.parameters as { lora_support?: boolean } | undefined)?.lora_support) &&
      compatibleLoras.value.length > 0,
  );

  function applyModelDefaults() {
    const cfg = currentModelConfig.value?.parameters as Record<string, unknown> | undefined;
    if (!cfg) return;
    applyDefaults(cfg, params as unknown as Record<string, unknown>);
    defaultSnapshot.value = { ...params };
  }

  async function loadCompatibleAdapters(id: string) {
    if (!id) {
      compatibleLoras.value = [];
      params.lora = '';
      return;
    }
    try {
      const loras = await api.settings.getCompatibleLoras(id);
      compatibleLoras.value = (loras as Record<string, unknown>[]) || [];
      if (params.lora && !findCompatibleLora(compatibleLoras.value, params.lora)) {
        params.lora = '';
      } else if (params.lora) {
        applyLoraComposeOverrides(params as Record<string, unknown>, findCompatibleLora(compatibleLoras.value, params.lora));
      }
    } catch {
      compatibleLoras.value = [];
      params.lora = '';
    }
  }

  function resetToDefaults() {
    applyModelDefaults();
  }

  watch(
    modelId,
    (id) => {
      applyModelDefaults();
      void loadCompatibleAdapters(id);
    },
    { immediate: true },
  );

  watch(
    () => params.lora,
    (loraId) => {
      if (!loraId) return;
      const row = findCompatibleLora(compatibleLoras.value, loraId);
      if (!row) {
        params.lora = '';
        return;
      }
      applyLoraComposeOverrides(params as Record<string, unknown>, row);
    },
  );

  return {
    params,
    compatibleLoras,
    currentModelConfig,
    paramSchema,
    hasCustomParams,
    showNegativePrompt,
    showSeedField,
    showLora,
    resetToDefaults,
  };
}
