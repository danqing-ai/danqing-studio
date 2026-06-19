import { DQ_STORAGE, getItem, removeItem, setItem } from '@/utils/storage';

type SizeMap = Record<string, string>;

function readMap(): SizeMap {
  try {
    const raw = getItem(DQ_STORAGE.VIDEO_SIZE_BY_MODEL);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== 'object') return {};
    const out: SizeMap = {};
    for (const [modelId, size] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof modelId === 'string' && typeof size === 'string' && size.includes('x')) {
        out[modelId] = size;
      }
    }
    return out;
  } catch {
    return {};
  }
}

function writeMap(map: SizeMap): void {
  setItem(DQ_STORAGE.VIDEO_SIZE_BY_MODEL, JSON.stringify(map));
}

/** Last resolution chosen for a given video model id (e.g. ``wan-2.2-t2v-14b``). */
export function getVideoSizeForModel(modelId: string): string | null {
  if (!modelId) return null;
  return readMap()[modelId] ?? null;
}

export function setVideoSizeForModel(modelId: string, size: string): void {
  if (!modelId || !size) return;
  const map = readMap();
  map[modelId] = size;
  writeMap(map);
}

/** One-time cleanup of legacy global size key (do not copy to per-model). */
export function migrateLegacyVideoLastSize(_modelId: string): void {
  removeItem(DQ_STORAGE.VIDEO_LAST_SIZE);
}
