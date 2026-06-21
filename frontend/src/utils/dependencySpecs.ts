/** Registry ``distribution.dependencies`` — string model id or versioned object. */

export interface DependencySpec {
  modelId: string;
  version: string | null;
}

export type RegistryDependency =
  | string
  | {
      model_id?: string;
      model?: string;
      version?: string;
    };

export function parseDependencies(raw: unknown): DependencySpec[] {
  if (!Array.isArray(raw)) return [];
  const out: DependencySpec[] = [];
  for (const item of raw) {
    if (typeof item === 'string') {
      const modelId = item.trim();
      if (modelId) out.push({ modelId, version: null });
      continue;
    }
    if (item && typeof item === 'object') {
      const obj = item as Record<string, unknown>;
      const modelId = String(obj.model_id ?? obj.model ?? '').trim();
      if (!modelId) continue;
      const versionRaw = obj.version;
      const version =
        typeof versionRaw === 'string' && versionRaw.trim() ? versionRaw.trim() : null;
      out.push({ modelId, version });
    }
  }
  return out;
}

export function isDependencyReady(
  spec: DependencySpec,
  detailedStatus: Record<string, { versions?: Record<string, { ready?: boolean }> }> | undefined,
  modelReady: (modelId: string) => boolean,
): boolean {
  if (!spec.version) {
    return modelReady(spec.modelId);
  }
  const versions = detailedStatus?.[spec.modelId]?.versions;
  if (versions?.[spec.version]?.ready) return true;
  // ``shared`` encoders satisfied when full ``original`` is installed.
  if (spec.version === 'shared' && versions?.original?.ready) return true;
  return false;
}
