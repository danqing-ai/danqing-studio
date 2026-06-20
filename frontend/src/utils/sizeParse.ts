/**
 * Parse human-readable size strings to GB (shared with memoryHint).
 */
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
