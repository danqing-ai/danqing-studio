import { ref, type Ref } from 'vue';
import { api } from '@/utils/api';
import { $mn } from '@/utils/i18n';
import {
  appendActiveEnumFields,
  appendLatentRefineField,
  COMPOSER_INFERENCE_ENUM_KEYS,
} from '@/utils/registryParamSchema';

export type StructuralGuideType = 'canny' | 'depth' | 'pose' | 'hed' | 'mlsd' | 'scribble' | 'gray' | 'auto' | 'redux';
export type ControlNetScope = 'create' | 'retouch' | 'extend';

/** Mirrors backend ``CONTROLNET_DECLARED_BACKENDS`` until CUDA batch lands. */
export const CONTROLNET_DECLARED_BACKENDS = ['mlx'] as const;

/** Prefer explicit API flag; infer MLX on Apple Silicon while system info is loading. */
export function resolveControlNetHostRuntimeAvailable(
  hostRuntimeAvailable?: boolean | null,
  compatibleControlNets?: Record<string, unknown>[],
  systemInfo?: { platform?: string; architecture?: string; dependencies?: Record<string, string> } | null,
): boolean {
  if (typeof hostRuntimeAvailable === 'boolean') return hostRuntimeAvailable;
  if (
    systemInfo?.platform === 'Darwin' &&
    systemInfo?.architecture === 'arm64'
  ) {
    const mlx = systemInfo.dependencies?.mlx;
    if (mlx && mlx !== 'not installed') return true;
  }
  const nets = compatibleControlNets;
  if (!nets?.length) return false;
  return nets.some((n) => n.runtime_available === true);
}

/** Host-level ControlNet runtime (prefer system info; fall back to compatible-net rows). */
export function isControlNetHostRuntimeAvailable(
  hostRuntimeAvailable?: boolean | null,
  compatibleControlNets?: Record<string, unknown>[],
  systemInfo?: { platform?: string; architecture?: string; dependencies?: Record<string, string> } | null,
): boolean {
  return resolveControlNetHostRuntimeAvailable(
    hostRuntimeAvailable,
    compatibleControlNets,
    systemInfo,
  );
}

export type ControlImageRef = {
  path: string;
  previewUrl: string;
  assetId?: string;
} | null;

export function inferStructuralGuideType(controlnetKey: string): StructuralGuideType {
  const k = String(controlnetKey || '').toLowerCase();
  if (k.includes('depth')) return 'depth';
  if (k.includes('pose')) return 'pose';
  if (k.includes('hed')) return 'hed';
  if (k.includes('mlsd')) return 'mlsd';
  if (k.includes('scribble')) return 'scribble';
  if (k.includes('gray')) return 'gray';
  if (k.includes('redux')) return 'redux';
  if (k.includes('z-image') && k.includes('union')) return 'auto';
  return 'canny';
}

export function isFillControlNet(controlnetKey: string): boolean {
  return String(controlnetKey || '').toLowerCase().includes('fill');
}

export function isReduxControlNet(controlnetKey: string): boolean {
  return String(controlnetKey || '').toLowerCase().includes('redux');
}

export function isCannyOrDepthControlNet(controlnetKey: string): boolean {
  const k = String(controlnetKey || '').toLowerCase();
  return k.includes('canny') || k.includes('depth');
}

export function isFluxStructuralBaseModel(modelKey: string): boolean {
  return String(modelKey || '').toLowerCase().startsWith('flux1');
}

export function isZImageUnionControlNet(controlnetKey: string): boolean {
  const k = String(controlnetKey || '').toLowerCase();
  return k.includes('z-image') && k.includes('union');
}

export function isZImageStructuralBaseModel(modelKey: string): boolean {
  const k = String(modelKey || '').toLowerCase();
  return k === 'z-image' || k === 'z-image-turbo' || k.startsWith('z-image');
}

export function supportsStructuralGuideBaseModel(modelKey: string): boolean {
  return isFluxStructuralBaseModel(modelKey) || isZImageStructuralBaseModel(modelKey);
}

export function controlNetDisplayName(n: Record<string, unknown>): string {
  return $mn(
    n as { name?: string | { zh?: string; en?: string }; name_en?: string },
    String(n.key || ''),
  );
}

export function controlNetReady(n: Record<string, unknown>): boolean {
  if (n.ready === true) return true;
  const vr = n.versions_ready as Record<string, boolean> | undefined;
  if (vr && Object.values(vr).some(Boolean)) return true;
  return false;
}

export function buildStructuralGuidePayload(
  controlnetKey: string,
  assetId: string,
  strength: number,
  inpaint?: { sourceAssetId?: string; maskAssetId?: string },
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    asset_id: assetId,
    model_id: controlnetKey,
    type: inferStructuralGuideType(controlnetKey),
    weight: Number(strength) || 0.8,
  };
  const src = String(inpaint?.sourceAssetId || '').trim();
  const msk = String(inpaint?.maskAssetId || '').trim();
  if (src && msk) {
    payload.inpaint_source_asset_id = src;
    payload.inpaint_mask_asset_id = msk;
  }
  return payload;
}

/** Attach structural guide + registry inference options to create/rewrite bodies. */
export function appendZImageEnhancementFields(
  body: Record<string, unknown>,
  opts: {
    parameters?: Record<string, unknown> | null;
    params?: Record<string, unknown>;
    controlnet?: string;
    controlAssetId?: string | null;
    controlnetStrength?: number;
    inpaintSourceId?: string;
    inpaintMaskId?: string;
  },
): { ok: true } | { ok: false; code: 'inpaint_pair_required' } {
  const controlnet = String(opts.controlnet || '');
  const controlAssetId = String(opts.controlAssetId || '').trim();
  if (controlAssetId && controlnet) {
    const inpSrc = String(opts.inpaintSourceId || '').trim();
    const inpMsk = String(opts.inpaintMaskId || '').trim();
    if (Boolean(inpSrc) !== Boolean(inpMsk)) {
      return { ok: false, code: 'inpaint_pair_required' };
    }
    body.structural_guide = buildStructuralGuidePayload(
      controlnet,
      controlAssetId,
      Number(opts.controlnetStrength) || 0.8,
      inpSrc && inpMsk ? { sourceAssetId: inpSrc, maskAssetId: inpMsk } : undefined,
    );
  }
  const composerParams = opts.params || {};
  appendActiveEnumFields(body, composerParams, COMPOSER_INFERENCE_ENUM_KEYS, opts.parameters);
  appendLatentRefineField(body, composerParams, opts.parameters);
  return { ok: true };
}

/** Attach registry inference enums (TeaCache / LeMiCa) + latent refine without structural guide. */
export function appendImageInferenceFields(
  body: Record<string, unknown>,
  composerParams: Record<string, unknown>,
  parameters?: Record<string, unknown> | null,
): void {
  appendActiveEnumFields(body, composerParams, COMPOSER_INFERENCE_ENUM_KEYS, parameters);
  appendLatentRefineField(body, composerParams, parameters);
}

export function fillModelRegistryDefaultsPatch(
  modelKey: string,
  registry: Record<string, Record<string, unknown>>,
): Partial<{ guidance: number; steps: number }> {
  if (!isFillControlNet(modelKey)) return {};
  const raw = registry[modelKey]?.parameters as Record<string, { default?: number }> | undefined;
  if (!raw) return {};
  const patch: Partial<{ guidance: number; steps: number }> = {};
  if (typeof raw.guidance?.default === 'number') patch.guidance = raw.guidance.default;
  if (typeof raw.steps?.default === 'number') patch.steps = raw.steps.default;
  return patch;
}

export function controlNetRegistryDefaultsPatch(
  key: string,
  compatibleControlNets: Record<string, unknown>[] | undefined,
): Partial<{ guidance: number; steps: number; controlnet_strength: number }> {
  const net = compatibleControlNets?.find((n) => String(n.key) === key);
  const raw = net?.parameters as Record<string, { default?: number }> | undefined;
  if (!raw) return {};
  const patch: Partial<{ guidance: number; steps: number; controlnet_strength: number }> = {};
  if (typeof raw.guidance?.default === 'number') patch.guidance = raw.guidance.default;
  if (typeof raw.steps?.default === 'number') patch.steps = raw.steps.default;
  const strength =
    typeof raw.controlnet_strength?.default === 'number'
      ? raw.controlnet_strength.default
      : typeof raw.redux_strength?.default === 'number'
        ? raw.redux_strength.default
        : undefined;
  if (typeof strength === 'number') patch.controlnet_strength = strength;
  return patch;
}

export function resolveControlAssetId(
  controlImage: ControlImageRef,
  messages: { assetRequired: string; required: string },
): string {
  if (controlImage?.assetId) return controlImage.assetId;
  if (controlImage?.path?.startsWith('asset:')) {
    return controlImage.path.slice('asset:'.length);
  }
  if (controlImage) throw new Error(messages.assetRequired);
  throw new Error(messages.required);
}

export type StructuralGuideValidationCode =
  | 'fill_edit_only'
  | 'flux_only'
  | 'no_img2img'
  | 'missing_control_image'
  | 'controlnet_not_ready'
  | 'runtime_unavailable';

/** First installed structural guide controlnet (excludes Fill). */
export function pickDefaultStructuralControlNet(
  nets: Record<string, unknown>[] | undefined,
): string {
  if (!nets?.length) return '';
  const ready = nets.find(
    (n) => controlNetReady(n) && !isFillControlNet(String(n.key || '')),
  );
  if (ready) return String(ready.key || '');
  const any = nets.find((n) => !isFillControlNet(String(n.key || '')));
  return any ? String(any.key || '') : '';
}

export function applyControlNetRegistryDefaults(
  controlnetKey: string,
  compatibleControlNets: Record<string, unknown>[] | undefined,
  params: {
    guidance?: number;
    steps?: number;
    controlnet_strength?: number;
  },
): void {
  const patch = controlNetRegistryDefaultsPatch(controlnetKey, compatibleControlNets);
  if (patch.guidance != null) params.guidance = patch.guidance;
  if (patch.steps != null) params.steps = patch.steps;
  if (patch.controlnet_strength != null) params.controlnet_strength = patch.controlnet_strength;
}

export function validateFillEditPrompt(prompt: string): boolean {
  return Boolean(String(prompt || '').trim());
}

export function validateStructuralGuideForCreate(opts: {
  controlnet: string;
  baseModel: string;
  hasReferenceImage: boolean;
  hasControlImage?: boolean;
  hostRuntimeAvailable?: boolean | null;
  compatibleControlNets?: Record<string, unknown>[];
  systemInfo?: { platform?: string; architecture?: string; dependencies?: Record<string, string> } | null;
}): { ok: true } | { ok: false; code: StructuralGuideValidationCode } {
  const key = String(opts.controlnet || '');
  if (!key) return { ok: true };
  if (
    !isControlNetHostRuntimeAvailable(
      opts.hostRuntimeAvailable,
      opts.compatibleControlNets,
      opts.systemInfo,
    )
  ) {
    return { ok: false, code: 'runtime_unavailable' };
  }
  if (isFillControlNet(key)) return { ok: false, code: 'fill_edit_only' };
  if (!supportsStructuralGuideBaseModel(opts.baseModel)) return { ok: false, code: 'flux_only' };
  if (opts.hasReferenceImage) {
    const zImg = isZImageStructuralBaseModel(opts.baseModel);
    const zUnion = isZImageUnionControlNet(key);
    if (!(zImg && zUnion)) {
      return { ok: false, code: 'no_img2img' };
    }
  }
  if (!opts.hasControlImage) return { ok: false, code: 'missing_control_image' };
  const entry = opts.compatibleControlNets?.find((n) => String(n.key) === key);
  if (entry && !controlNetReady(entry)) return { ok: false, code: 'controlnet_not_ready' };
  return { ok: true };
}

export function useCompatibleControlNets(scope: ControlNetScope = 'create') {
  const compatibleControlNets = ref<Record<string, unknown>[]>([]);
  const loading = ref(false);

  async function loadCompatibleControlNets(modelKey: string): Promise<void> {
    if (!modelKey) {
      compatibleControlNets.value = [];
      return;
    }
    loading.value = true;
    try {
      const rows = await api.settings.getCompatibleControlNets(modelKey, scope);
      compatibleControlNets.value = (rows as Record<string, unknown>[]) || [];
    } finally {
      loading.value = false;
    }
  }

  function clearIfIncompatible(
    params: { controlnet?: unknown },
    onClear: () => void,
  ): void {
    const key = String(params.controlnet || '');
    if (!key) return;
    const listed = compatibleControlNets.value.some((n) => String(n.key) === key);
    if (!listed) {
      params.controlnet = '';
      onClear();
    }
  }

  return {
    compatibleControlNets: compatibleControlNets as Ref<Record<string, unknown>[]>,
    loading,
    loadCompatibleControlNets,
    clearIfIncompatible,
  };
}
