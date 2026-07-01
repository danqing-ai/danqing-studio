import { $tt } from '@/utils/i18n';

/** Append registry preset positive text with a localized prefix. */
export function appendStyleBoost(current: string, positive: string): string {
  const boost = String(positive || '').trim();
  if (!boost) return current;
  const prefix = $tt('create.styleBoostPrefix');
  const base = String(current || '').trim();
  return base ? `${base}\n${prefix}${boost}` : boost;
}
