/**
 * Pre-submit memory risk soft hint (non-blocking).
 * Heuristic: registry version `size` vs system RAM / MLX limit.
 */
import { toast } from '@/utils/feedback';
import type { SystemInfo } from '@/types';
import { parseHumanSizeToGb } from '@/utils/sizeParse';

export { parseHumanSizeToGb };

export function warnIfRiskyMemory(opts: {
  systemInfo?: SystemInfo | null;
  versionSizeHuman: string;
  minUnifiedMemoryGb?: number | null;
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
  const minGb = opts.minUnifiedMemoryGb != null && opts.minUnifiedMemoryGb > 0
    ? Number(opts.minUnifiedMemoryGb)
    : null;
  const effectiveGb = minGb != null ? Math.max(modelGb ?? 0, minGb) : modelGb;
  if (effectiveGb == null || effectiveGb <= 0) return;

  if (effectiveGb > refGb * 0.88) {
    toast.warning(
      $tt('studio.submitOomHint', {
        modelGb: effectiveGb.toFixed(1),
        refGb: refGb.toFixed(1),
      }),
    );
  }
}
