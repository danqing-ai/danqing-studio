/** User-facing label for gallery / preview / queue (title → prompt → filename). */

export interface AssetLabelSource {
  title?: string;
  prompt?: string;
  name?: string;
  metadata?: Record<string, unknown>;
}

export function assetTitleFrom(item: AssetLabelSource): string {
  const fromMeta = item.metadata?.title;
  const raw = item.title ?? fromMeta;
  return String(raw ?? '').trim();
}

export function assetDisplayLabel(item: AssetLabelSource, fallback = '—'): string {
  const title = assetTitleFrom(item);
  if (title) return title;
  const prompt = String(item.prompt ?? '').trim();
  if (prompt) return prompt;
  const name = String(item.name ?? '').trim();
  if (name) return name;
  return fallback;
}

export function previewDisplayCaption(title: string, prompt: string): string {
  const t = String(title ?? '').trim();
  if (t) return t;
  return String(prompt ?? '').trim();
}

export function truncateDisplayLabel(text: string, maxLen = 60): string {
  const s = String(text ?? '').trim();
  if (!s) return '—';
  if (s.length <= maxLen) return s;
  return `${s.substring(0, maxLen)}...`;
}
