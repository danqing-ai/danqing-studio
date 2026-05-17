/** Resolve a registry model id from app settings (id or localized display name). */
export function resolveDefaultModelRegistryKey(
  preferred: string,
  registry: Record<string, { media?: string; name?: string | { zh?: string; en?: string } }>,
  media: 'image' | 'video' | 'audio',
): string | null {
  const dm = (preferred || '').trim();
  if (!dm || !registry || !Object.keys(registry).length) return null;

  if (registry[dm]) {
    const entryMedia = registry[dm].media || 'image';
    if (entryMedia === media) return dm;
  }

  for (const [k, cfg] of Object.entries(registry)) {
    const entryMedia = cfg.media || 'image';
    if (entryMedia !== media) continue;
    const n = cfg.name;
    if (typeof n === 'string' && n === dm) return k;
    if (n && typeof n === 'object') {
      const o = n as { zh?: string; en?: string };
      if (o.zh === dm || o.en === dm) return k;
    }
  }
  return null;
}

/** Pick registry default version key; skip versions known to be not ready. */
export function pickDefaultVersionKey(
  modelKey: string,
  registry: Record<string, { versions?: Record<string, { default?: boolean }> }>,
  detailedVersions?: Record<string, { ready?: boolean }>,
): string | null {
  const cfg = registry[modelKey];
  if (!cfg?.versions) return null;
  const versionKeys = Object.keys(cfg.versions);
  const defaultVK = versionKeys.find((vk) => cfg.versions![vk]?.default) || versionKeys[0];
  if (!defaultVK) return null;
  const stRow = detailedVersions?.[defaultVK];
  if (stRow && stRow.ready === false) return null;
  return defaultVK;
}
