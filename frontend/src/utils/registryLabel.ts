import { $vn, getLocale } from '@/utils/i18n';

type Bilingual = string | { zh?: string; en?: string };

/** Normalize download task / legacy API labels that embedded a Python dict repr. */
export function formatDownloadDisplayName(raw: string): string {
  if (!raw || typeof raw !== 'string') return raw || '';

  const dictMatch = raw.match(/\{['"]zh['"]\s*:\s*['"]([^'"]+)['"]/);
  if (dictMatch) {
    const prefix = raw.slice(0, raw.indexOf('{')).trim();
    const locale = getLocale();
    const enMatch = raw.match(/\{['"]zh['"]\s*:\s*['"]([^'"]+)['"]\s*,\s*['"]en['"]\s*:\s*['"]([^'"]+)['"]/);
    let suffix = dictMatch[1];
    if (enMatch) {
      suffix = locale === 'en' ? enMatch[2] || enMatch[1] : enMatch[1];
    }
    return prefix ? `${prefix} ${suffix}`.trim() : suffix;
  }

  return raw;
}

export function formatVersionLabel(
  modelName: string,
  version?: { name?: Bilingual } | null,
  versionKey = ''
): string {
  const suffix = version ? $vn(version, versionKey) : versionKey;
  if (!suffix) return modelName;
  if (suffix.trim().toLowerCase() === modelName.trim().toLowerCase()) return modelName;
  return `${modelName} ${suffix}`;
}
