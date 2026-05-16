/**
 * models_registry.json parameters — inference and utilities aligned with
 * backend/core/registry_format.typed_parameters.
 */

export type InferredParamType = 'int' | 'float' | 'enum' | 'bool' | 'object' | 'skip';

export type RawParamSpec = Record<string, unknown>;

export type NormalizedParamSpec = RawParamSpec & { type: InferredParamType };

export function inferType(spec: unknown, key?: string): InferredParamType {
  if (!spec || typeof spec !== 'object') return 'skip';
  const s = spec as RawParamSpec;
  if (typeof s.type === 'string') return s.type as InferredParamType;
  if (typeof s.default === 'boolean' || (key && String(key).endsWith('_support'))) {
    return 'bool';
  }
  if (Array.isArray(s.options)) return 'enum';
  if (typeof s.min === 'number' && typeof s.max === 'number') {
    const d = s.default;
    return Number.isInteger(d) && typeof d !== 'boolean' ? 'int' : 'float';
  }
  return 'object';
}

export function normalizeParamsDef(parameters: Record<string, unknown> | undefined | null): Record<string, NormalizedParamSpec> {
  const out: Record<string, NormalizedParamSpec> = {};
  for (const [key, spec] of Object.entries(parameters || {})) {
    const t = inferType(spec, key);
    if (t === 'skip') continue;
    out[key] = { ...(spec as RawParamSpec), type: t };
  }
  return out;
}

export function isCapabilityOnly(key: string, spec: NormalizedParamSpec): boolean {
  return spec.type === 'bool' && String(key).endsWith('_support');
}

export function isRenderableScalar(key: string, spec: NormalizedParamSpec): boolean {
  if (isCapabilityOnly(key, spec)) return false;
  if (spec.type === 'object') return false;
  return spec.type === 'int' || spec.type === 'float' || spec.type === 'enum';
}

const PREFERRED_ORDER = [
  'steps',
  'guidance',
  'scheduler',
  'scheduler_sigma_schedule',
  'enable_cfg_renorm',
  'cfg_renorm_min',
  'strength',
  'controlnet_strength',
  'redux_strength',
];

export function sortParamKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const ia = PREFERRED_ORDER.indexOf(a);
    const ib = PREFERRED_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
}

export function resolutionPair(
  normalized: Record<string, NormalizedParamSpec>
): { width: NormalizedParamSpec; height: NormalizedParamSpec } | null {
  const w = normalized.width;
  const h = normalized.height;
  if (w && h && w.type === 'enum' && h.type === 'enum') {
    return { width: w, height: h };
  }
  return null;
}

export function scalarKeysForForm(normalized: Record<string, NormalizedParamSpec>): string[] {
  const pair = resolutionPair(normalized);
  const skip = new Set<string>();
  if (pair) {
    skip.add('width');
    skip.add('height');
  }
  const keys = Object.keys(normalized).filter((k) => {
    if (skip.has(k)) return false;
    return isRenderableScalar(k, normalized[k]!);
  });
  return sortParamKeys(keys);
}

export function applyDefaults(parameters: Record<string, unknown> | undefined | null, target: Record<string, unknown>): void {
  const n = normalizeParamsDef(parameters || {});
  for (const [key, spec] of Object.entries(n)) {
    if (isCapabilityOnly(key, spec) || spec.type === 'object') continue;
    if ('default' in spec) {
      target[key] = spec.default;
    }
  }
  if (typeof target.seed !== 'undefined') target.seed = '';
  if (typeof target.lora !== 'undefined') target.lora = '';
  if (typeof target.controlnet !== 'undefined') {
    target.controlnet = '';
    target.controlnet_strength =
      n.controlnet_strength && 'default' in n.controlnet_strength
        ? (n.controlnet_strength as NormalizedParamSpec).default
        : 0.8;
  }
  const ls = n.lora_scale;
  if (typeof target.lora_scale !== 'undefined') {
    target.lora_scale = ls && 'default' in ls ? (ls as NormalizedParamSpec).default : 0.8;
  }
}

export function hasDeviation(
  parameters: Record<string, unknown> | undefined | null,
  target: Record<string, unknown>,
  opts?: { ignoreKeys?: string[] }
): boolean {
  const ignore = new Set(opts?.ignoreKeys || ['strength', 'controlnet_strength']);
  const n = normalizeParamsDef(parameters || {});
  for (const [key, spec] of Object.entries(n)) {
    if (ignore.has(key)) continue;
    if (!isRenderableScalar(key, spec)) continue;
    const def = spec.default;
    if (def !== undefined && target[key] !== def) return true;
  }
  const pair = resolutionPair(n);
  if (pair) {
    if (target.width !== pair.width.default || target.height !== pair.height.default) {
      return true;
    }
  }
  if (target.lora) return true;
  if (target.seed != null && String(target.seed).trim() !== '') return true;
  return false;
}
