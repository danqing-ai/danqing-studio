export interface ModelVersion {
  name: string;
  size?: string;
  source?: string;
  source_type?: string;
  from_version?: string;
  default?: boolean;
}

export interface ModelVersionLayoutInput {
  id: string;
  source?: string;
  versions?: Record<string, ModelVersion>;
}

export interface VersionRow {
  verKey: string;
  ver: ModelVersion;
  vstatus: string;
}

export interface FullVersionBlock extends VersionRow {
  derivedVariants: VersionRow[];
  parentReady: boolean;
}

export interface ModelVersionLayout {
  fullBlocks: FullVersionBlock[];
  lightweightRows: VersionRow[];
  fullBlockSource: string | null;
  lightweightBlockSource: string | null;
}

export interface LightweightSplit {
  primary: VersionRow | null;
  extra: VersionRow[];
  hasExtra: boolean;
}

function isFourBitLightweight(row: VersionRow): boolean {
  return /(?:^|-)4bit$|int4/i.test(row.verKey) || /4\s*bit|int4/i.test(row.ver.name);
}

/** 轻量版默认只展示推荐位（default → 4bit → 首个），其余收进「更多」。 */
export function splitLightweightRows(rows: VersionRow[]): LightweightSplit {
  if (rows.length === 0) {
    return { primary: null, extra: [], hasExtra: false };
  }
  if (rows.length === 1) {
    return { primary: rows[0], extra: [], hasExtra: false };
  }

  const primary =
    rows.find((row) => row.ver.default === true) ||
    rows.find(isFourBitLightweight) ||
    rows[0];
  const extra = rows.filter((row) => row.verKey !== primary.verKey);
  return { primary, extra, hasExtra: extra.length > 0 };
}

export function resolveVersionSource(
  model: ModelVersionLayoutInput,
  ver: ModelVersion,
): string {
  return ver.source || model.source || '';
}

export function uniformDownloadSource(model: ModelVersionLayoutInput): string {
  if (!model.versions) return model.source || '';
  const downloadable = Object.values(model.versions).filter(
    (ver) => ver.source_type !== 'derived',
  );
  if (downloadable.length === 0) return model.source || '';
  const sources = downloadable.map((ver) => resolveVersionSource(model, ver));
  const unique = [...new Set(sources.filter(Boolean))];
  return unique.length === 1 ? unique[0] : '';
}

function uniformBlockSource(
  model: ModelVersionLayoutInput,
  rows: VersionRow[],
): string | null {
  if (rows.length === 0) return null;
  const sources = rows.map((row) => resolveVersionSource(model, row.ver));
  const unique = [...new Set(sources.filter(Boolean))];
  return unique.length === 1 ? unique[0] : null;
}

function derivedChipStatus(
  getVersionStatus: (modelId: string, verKey: string) => string,
  modelId: string,
  verKey: string,
  parentReady: boolean,
): string {
  const st = getVersionStatus(modelId, verKey);
  if (st === 'ready') return 'ready';
  if (parentReady) return 'quantize';
  return 'waiting';
}

export function buildModelVersionLayout(
  model: ModelVersionLayoutInput,
  getVersionStatus: (modelId: string, verKey: string) => string,
): ModelVersionLayout {
  if (!model.versions) {
    return {
      fullBlocks: [],
      lightweightRows: [],
      fullBlockSource: null,
      lightweightBlockSource: null,
    };
  }

  const fullRows: VersionRow[] = [];
  const lightweightRows: VersionRow[] = [];
  const derivedByParent: Record<string, VersionRow[]> = {};

  for (const [verKey, ver] of Object.entries(model.versions)) {
    const row: VersionRow = {
      verKey,
      ver,
      vstatus: getVersionStatus(model.id, verKey),
    };
    if (ver.source_type === 'derived') {
      const parent = ver.from_version;
      if (!parent) continue;
      if (!derivedByParent[parent]) derivedByParent[parent] = [];
      derivedByParent[parent].push(row);
    } else if (ver.source_type === 'prequantized') {
      lightweightRows.push(row);
    } else {
      fullRows.push(row);
    }
  }

  fullRows.sort((a, b) => {
    if (a.ver.default === true && b.ver.default !== true) return -1;
    if (b.ver.default === true && a.ver.default !== true) return 1;
    return a.ver.name.localeCompare(b.ver.name);
  });

  lightweightRows.sort((a, b) => a.ver.name.localeCompare(b.ver.name));

  const fullBlocks: FullVersionBlock[] = fullRows.map((row) => {
    const parentReady = row.vstatus === 'ready';
    const derivedVariants = (derivedByParent[row.verKey] || [])
      .sort((a, b) => a.verKey.localeCompare(b.verKey))
      .map((derived) => ({
        ...derived,
        vstatus: derivedChipStatus(getVersionStatus, model.id, derived.verKey, parentReady),
      }));
    return {
      ...row,
      parentReady,
      derivedVariants,
    };
  });

  return {
    fullBlocks,
    lightweightRows,
    fullBlockSource: uniformBlockSource(model, fullRows),
    lightweightBlockSource: uniformBlockSource(model, lightweightRows),
  };
}

export function simplifyPrequantizedName(name: string, source: string): string {
  let text = name.trim();
  if (source === 'modelscope') {
    text = text.replace(/^魔搭\s*[·•]\s*/u, '');
  }
  text = text.replace(/预量化\s*/gu, '');
  return text.replace(/\s+/g, ' ').trim();
}

export function shortDerivedLabel(verKey: string, name: string): string {
  const key = verKey.toUpperCase();
  if (/^INT[48]$/.test(key)) return key;
  const match = name.match(/INT[48]/i);
  if (match) return match[0].toUpperCase();
  return name.replace(/\s*量化版\s*/u, '').trim() || name;
}
