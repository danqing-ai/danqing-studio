import { $tt } from '@/utils/i18n';

export type LogDisplayKind = 'milestone' | 'progress' | 'technical' | 'error' | 'warning';

export interface LogDisplayChip {
  key: string;
  label: string;
  value: string;
}

export interface LogDisplayItem {
  index: number;
  time: string;
  kind: LogDisplayKind;
  title: string;
  detail?: string;
  chips?: LogDisplayChip[];
}

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

export function progressPhaseLabel(phase: string, message: string): string {
  const key =
    phase ||
    (message === 'denoise' ? 'denoising' : message === 'post' ? 'decoding' : message);
  if (!key || key === 'denoise' || key === 'post') {
    if (key === 'denoise') return $tt('studio.phase.denoising');
    if (key === 'post') return $tt('studio.phase.decoding');
    return '';
  }
  const i18nKey = `studio.phase.${key}`;
  const out = $tt(i18nKey);
  return out !== i18nKey ? out : '';
}

export function formatLogTimestamp(): string {
  const now = new Date();
  return (
    String(now.getHours()).padStart(2, '0') +
    ':' +
    String(now.getMinutes()).padStart(2, '0') +
    ':' +
    String(now.getSeconds()).padStart(2, '0')
  );
}

const GRAPH_STEP_RE = /^\[([^\]]+)\]\s*(.*)$/;

const TECHNICAL_HINT_RE =
  /^(infer |vae_|preview |batch |sigma_|t_embed|scheduler_|cfg_|supports_|image_seq_len|use_empirical|requires_sigma)/i;

function parseGraphStep(message: string): { node: string; detail: string } | null {
  const m = String(message || '').trim().match(GRAPH_STEP_RE);
  if (!m) return null;
  return { node: m[1], detail: (m[2] || '').trim() };
}

function graphStepTitle(node: string): string {
  const key = `studio.logGraph.${node}`;
  const out = $tt(key);
  return out !== key ? out : node;
}

function isPhaseMilestoneMessage(message: string): boolean {
  const phases = ['encoding', 'loading_model', 'denoising', 'decoding', 'saving'];
  return phases.some((p) => {
    const label = $tt(`studio.phase.${p}`);
    return label && message === label;
  });
}

export function parseInferParams(message: string): Record<string, string> | null {
  const text = String(message || '').trim();
  if (!text.startsWith('infer ')) return null;
  const out: Record<string, string> = {};
  for (const token of text.split(/\s+/)) {
    const eq = token.indexOf('=');
    if (eq > 0) {
      out[token.slice(0, eq)] = token.slice(eq + 1);
    }
  }
  return Object.keys(out).length > 0 ? out : null;
}

function inferParamChips(params: Record<string, string>): LogDisplayChip[] {
  const order: { key: string; labelKey: string }[] = [
    { key: 'model', labelKey: 'studio.logChip.model' },
    { key: 'size', labelKey: 'studio.logChip.size' },
    { key: 'steps', labelKey: 'studio.logChip.steps' },
    { key: 'guidance', labelKey: 'studio.logChip.guidance' },
    { key: 'base_seed', labelKey: 'studio.logChip.seed' },
  ];
  const chips: LogDisplayChip[] = [];
  for (const { key, labelKey } of order) {
    const value = params[key];
    if (value == null || value === '') continue;
    const labelKeyFull = labelKey;
    const label = $tt(labelKeyFull);
    chips.push({
      key,
      label: label !== labelKeyFull ? label : key,
      value,
    });
  }
  return chips;
}

export function classifyLogEntry(message: string, level: string): LogDisplayKind {
  const lvl = String(level || 'info').toLowerCase();
  if (lvl === 'error') return 'error';
  if (lvl === 'warning') return 'warning';
  if (lvl === 'success') return 'milestone';
  if (parseDenoiseStepKey(message)) return 'progress';
  if (message === $tt('studio.queuePostProcessHint')) return 'progress';
  if (isPhaseMilestoneMessage(message)) return 'milestone';
  if (message === $tt('studio.startingGen') || message === $tt('studio.genComplete')) {
    return 'milestone';
  }
  if (parseGraphStep(message)) return 'milestone';
  if (parseInferParams(message)) return 'milestone';
  if (/^preview step /i.test(message)) return 'progress';
  if (TECHNICAL_HINT_RE.test(message)) return 'technical';
  return 'info' === lvl ? 'technical' : 'milestone';
}

function buildDisplayTitle(message: string, kind: LogDisplayKind): string {
  const graph = parseGraphStep(message);
  if (graph) return graphStepTitle(graph.node);
  const infer = parseInferParams(message);
  if (infer) return $tt('studio.logInferTitle');
  if (kind === 'progress' && parseDenoiseStepKey(message)) {
    return formatGenLogMessage(message);
  }
  if (/^preview step /i.test(message)) {
    const m = message.match(/preview step (\d+)\/(\d+)/i);
    if (m) {
      return $tt('studio.livingCanvasStep', { step: m[1], total: m[2] });
    }
  }
  return message;
}

function buildDisplayDetail(message: string, kind: LogDisplayKind, showTechnical: boolean): string | undefined {
  if (!showTechnical && kind === 'technical') return undefined;
  const graph = parseGraphStep(message);
  if (graph) {
    const d = graph.detail;
    if (!d || d === 'start') return undefined;
    return d;
  }
  if (parseInferParams(message)) return showTechnical ? message : undefined;
  if (kind === 'technical') return message;
  if (kind === 'milestone' || kind === 'progress') {
    return showTechnical ? message : undefined;
  }
  return showTechnical ? message : undefined;
}

export function buildLogDisplayItems(
  entries: { time: string; message: string; level: string }[],
  showTechnical: boolean,
): LogDisplayItem[] {
  const items: LogDisplayItem[] = [];
  entries.forEach((entry, index) => {
    const kind = classifyLogEntry(entry.message, entry.level);
    if (kind === 'technical' && !showTechnical) return;

    const infer = parseInferParams(entry.message);
    items.push({
      index,
      time: entry.time,
      kind,
      title: buildDisplayTitle(entry.message, kind),
      detail: buildDisplayDetail(entry.message, kind, showTechnical),
      chips: infer ? inferParamChips(infer) : undefined,
    });
  });
  return items;
}

export function latestProgressItem(items: LogDisplayItem[]): LogDisplayItem | null {
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (items[i].kind === 'progress') return items[i];
  }
  return null;
}
