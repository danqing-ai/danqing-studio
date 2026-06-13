export type LoraQualityLevel = 'good' | 'fair' | 'poor';

export type LoraQualityHint = {
  code: string;
  severity: 'info' | 'warning' | 'error';
  params?: Record<string, unknown>;
  source?: string;
};

export type LoraVlmSummary = {
  audit_kind?: 'concept' | 'style';
  avg_score?: number;
  audited_count?: number;
  samples?: Array<VlmImageSample>;
};

export type VlmImageSample = {
  file: string;
  score?: number | null;
  reason?: string;
  issues?: string[];
};

export function vlmScoreLevel(score: number | null | undefined): 'good' | 'fair' | 'poor' | null {
  if (score == null || Number.isNaN(Number(score))) return null;
  const s = Number(score);
  if (s >= 4) return 'good';
  if (s >= 3) return 'fair';
  return 'poor';
}

export function buildVlmSampleMap(samples: VlmImageSample[] | undefined): Map<string, VlmImageSample> {
  const map = new Map<string, VlmImageSample>();
  if (!samples?.length) return map;
  for (const sample of samples) {
    const file = String(sample.file || '').trim();
    if (!file) continue;
    map.set(file, sample);
    const base = file.split('/').pop();
    if (base && base !== file) map.set(base, sample);
  }
  return map;
}

export function lookupVlmSample(
  map: Map<string, VlmImageSample>,
  datasetFile: string
): VlmImageSample | undefined {
  const key = (datasetFile || '').trim();
  if (!key) return undefined;
  return map.get(key) ?? map.get(key.split('/').pop() || '');
}

export type LoraDatasetHealthReport = {
  level: LoraQualityLevel;
  score: number;
  stats: Record<string, number>;
  hints: LoraQualityHint[];
  vision_available?: boolean;
  vlm_audited?: boolean;
  audit_kind?: 'concept' | 'style';
  vlm?: LoraVlmSummary;
};

export type LoraTrainingQualityReport = {
  level: LoraQualityLevel;
  score: number;
  metrics: Record<string, unknown>;
  hints: LoraQualityHint[];
  dataset_health?: LoraDatasetHealthReport | null;
  vision_available?: boolean;
  vlm_audited?: boolean;
  audit_kind?: 'concept' | 'style';
  vlm?: LoraVlmSummary;
};

export function qualityAlertType(level: LoraQualityLevel | undefined): 'success' | 'warning' | 'error' | 'info' {
  if (level === 'poor') return 'error';
  if (level === 'fair') return 'warning';
  if (level === 'good') return 'success';
  return 'info';
}
