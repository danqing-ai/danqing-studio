/**
 * Pre-submit memory risk soft hint (non-blocking).
 * Heuristic: registry version `size` vs system RAM / MLX limit.
 */
import { ElMessage } from 'element-plus';
import type { SystemInfo } from '@/types';

export function parseHumanSizeToGb(s: unknown): number | null {
  if (s == null || s === '') return null;
  const str = String(s)
    .trim()
    .toLowerCase()
    .replace(/[,~≈]/g, '')
    .replace(/\s+/g, '');
  const m = str.match(/([\d.]+)\s*(tb|t|gb|g|mb|m)?/);
  if (!m) return null;
  let n = parseFloat(m[1]);
  if (!Number.isFinite(n) || n <= 0) return null;
  const u = m[2] || 'gb';
  if (u === 'tb' || u === 't') n *= 1024;
  else if (u === 'mb' || u === 'm') n /= 1024;
  return n;
}

export function warnIfRiskyMemory(opts: {
  systemInfo?: SystemInfo | null;
  versionSizeHuman: string;
  $tt: (key: string, params?: Record<string, string | number>) => string;
}): void {
  const $tt = opts.$tt;
  const si = opts.systemInfo;
  const mem = Number(si && si.memory_gb) || 0;
  const mlxRaw = Number(si && si.mlx_memory_limit);
  const capFromMlx = Number.isFinite(mlxRaw) && mlxRaw > 0 ? mlxRaw : null;

  let refGb = 0;
  if (mem > 0 && capFromMlx != null) {
    refGb = Math.min(mem, capFromMlx);
  } else if (mem > 0) {
    refGb = mem;
  } else if (capFromMlx != null) {
    refGb = capFromMlx;
  }
  if (!(refGb > 0)) return;

  const modelGb = parseHumanSizeToGb(opts.versionSizeHuman);
  if (modelGb == null || modelGb <= 0) return;

  if (modelGb > refGb * 0.88) {
    ElMessage.warning(
      $tt('studio.submitOomHint', {
        modelGb: modelGb.toFixed(1),
        refGb: refGb.toFixed(1),
      }),
    );
  }
}
