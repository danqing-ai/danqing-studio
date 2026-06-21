import { computed, reactive, ref, watch, type Ref } from 'vue';
import { api } from '@/utils/api';
import { useRegistryStore } from '@/stores/registry';
import { useCompatibleControlNets } from '@/composables/useStructuralGuide';
import {
  applyDefaults,
  buildResolutionSizeOptions,
  hasDeviation,
  normalizeParamsDef,
} from '@/utils/registryParamSchema';
import { $mn } from '@/utils/i18n';

export type KeyframeComposeParams = {
  steps: number;
  guidance: number;
  seed: string;
  strength: number;
  negative_prompt: string;
  scheduler?: string;
  lora?: string;
  lora_scale?: number;
  controlnet?: string;
  controlnet_strength?: number;
  lemica_mode?: string;
  latent_refine_scale?: number;
  latent_refine_denoise?: number;
};

function defaultParams(): KeyframeComposeParams {
  return {
    steps: 28,
    guidance: 4,
    seed: '',
    strength: 0.65,
    negative_prompt: '',
    scheduler: undefined,
    lora: '',
    lora_scale: 0.8,
    controlnet: '',
    controlnet_strength: 0.8,
    lemica_mode: 'none',
    latent_refine_scale: 1,
    latent_refine_denoise: 0.3,
  };
}

export function useLongVideoKeyframeCompose(modelId: Ref<string>, outputSize: Ref<string>) {
  const registryStore = useRegistryStore();
  const params = reactive<KeyframeComposeParams>(defaultParams());
  const defaultSnapshot = ref<KeyframeComposeParams>(defaultParams());
  const compatibleLoras = ref<Record<string, unknown>[]>([]);
  const presets = ref<Record<string, Record<string, unknown>>>({});
  const controlImage = ref<{ previewUrl: string; path: string } | null>(null);
  const inpaintSourceImage = ref<{ previewUrl: string; path: string } | null>(null);
  const inpaintMaskImage = ref<{ previewUrl: string; path: string } | null>(null);

  const {
    compatibleControlNets,
    loadCompatibleControlNets,
    clearIfIncompatible: clearControlNetIfIncompatible,
  } = useCompatibleControlNets('create');

  const currentModelConfig = computed(() => {
    const id = modelId.value;
    if (!id) return null;
    return (registryStore.registry?.models?.[id] as Record<string, unknown> | undefined) ?? null;
  });

  const modelSelectOptions = computed(() => {
    const reg = registryStore.registry?.models ?? {};
    return Object.keys(reg)
      .filter((id) => {
        const m = reg[id] as { media?: string; actions?: Record<string, unknown> };
        return m?.media === 'image' && m.actions?.create;
      })
      .map((id) => {
        const m = reg[id] as { commercial_use_allowed?: boolean };
        return {
          label: $mn(reg[id] as { name?: string | { zh?: string; en?: string } }, id),
          value: id,
          commercialUseAllowed: Boolean(m?.commercial_use_allowed),
        };
      });
  });

  const sizeOptions = computed(() => buildResolutionSizeOptions(currentModelConfig.value?.parameters as Record<string, unknown>));

  const filteredPresets = computed(() => {
    const result: Record<string, Record<string, unknown>> = {};
    for (const [name, preset] of Object.entries(presets.value)) {
      const applies = preset.applies_to as string[] | undefined;
      if (preset.media_scope === 'image' && Array.isArray(applies) && applies.includes('create')) {
        result[name] = preset;
      }
    }
    return result;
  });

  const hasCustomParams = computed(() => {
    const cfg = currentModelConfig.value?.parameters as Record<string, unknown> | undefined;
    if (!cfg) return false;
    return hasDeviation(cfg, params as unknown as Record<string, unknown>);
  });

  const showNegativePrompt = computed(() =>
    Boolean((currentModelConfig.value?.parameters as Record<string, unknown> | undefined)?.negative_prompt_support),
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
      compatibleControlNets.value = [];
      return;
    }
    try {
      const loras = await api.settings.getCompatibleLoras(id);
      compatibleLoras.value = (loras as Record<string, unknown>[]) || [];
      await loadCompatibleControlNets(id);
      clearControlNetIfIncompatible(params as unknown as Record<string, unknown>, () => {
        controlImage.value = null;
      });
    } catch {
      compatibleLoras.value = [];
      compatibleControlNets.value = [];
    }
  }

  async function loadPresets() {
    try {
      const data = await api.settings.getPresets();
      presets.value = (data as Record<string, Record<string, unknown>>) || {};
    } catch {
      presets.value = {};
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

  void loadPresets();

  return {
    params,
    compatibleLoras,
    compatibleControlNets,
    controlImage,
    inpaintSourceImage,
    inpaintMaskImage,
    currentModelConfig,
    modelSelectOptions,
    sizeOptions,
    filteredPresets,
    hasCustomParams,
    showNegativePrompt,
    outputSize,
    resetToDefaults,
    paramSchema: computed(() => normalizeParamsDef(currentModelConfig.value?.parameters as Record<string, unknown>)),
  };
}
