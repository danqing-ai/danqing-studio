import { $tt } from '@/utils/i18n';

/** Localized label for asset lineage / canvas edge relation_type. */
export function lineageRelationLabel(relation: string | null | undefined): string {
  const rt = String(relation || '').trim();
  if (!rt) return '';
  const key = `canvas.relation.${rt}`;
  const translated = $tt(key);
  return translated !== key ? translated : rt;
}
