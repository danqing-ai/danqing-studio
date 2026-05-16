import { $tt } from '@/utils/i18n';

/** 从日志行或进度文案解析去噪步 key，如 "3/10"。 */
export function parseDenoiseStepKey(msg: string): string | null {
  const text = String(msg || '').trim();
  const en = text.match(/^Step\s+(\d+)\s*\/\s*(\d+)/i);
  if (en) return `${en[1]}/${en[2]}`;
  const zh = text.match(/去噪\s*(\d+)\s*\/\s*(\d+)/);
  if (zh) return `${zh[1]}/${zh[2]}`;
  return null;
}

/** 将后端英文 Step 行统一为 i18n 去噪进度文案。 */
export function formatGenLogMessage(raw: string): string {
  const key = parseDenoiseStepKey(raw);
  if (!key) return raw;
  const [current, total] = key.split('/');
  return $tt('studio.queueDenoiseProgress', { current, total });
}

export function isDuplicateDenoiseStepLog(
  recentLogs: { message: string }[],
  rawMessage: string,
): boolean {
  const key = parseDenoiseStepKey(rawMessage);
  if (!key) return false;
  const last = recentLogs[recentLogs.length - 1];
  if (!last) return false;
  return parseDenoiseStepKey(last.message) === key;
}
