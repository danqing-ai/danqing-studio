import { isCommercialUseAllowed, type CommercialUseFlag } from '@/utils/modelLicense';

export interface ModelVersionFilterRow {
  ready?: boolean;
  commercialUseAllowed?: boolean;
}

export function applyModelVersionFilters<T extends ModelVersionFilterRow>(
  rows: T[],
  opts: { installedOnly: boolean; commercialOnly: boolean },
): T[] {
  let out = rows;
  if (opts.installedOnly) {
    out = out.filter((r) => Boolean(r.ready));
  }
  if (opts.commercialOnly) {
    out = out.filter((r) => r.commercialUseAllowed === true);
  }
  return out;
}

export function modelPassesRegistryFilters(
  model: { ready?: boolean; commercial_use_allowed?: CommercialUseFlag; successor?: string | null },
  opts: { installedOnly: boolean; commercialOnly: boolean; currentModelsOnly?: boolean },
): boolean {
  if (opts.installedOnly && !model.ready) {
    return false;
  }
  if (opts.commercialOnly && !isCommercialUseAllowed(model.commercial_use_allowed)) {
    return false;
  }
  if (opts.currentModelsOnly && model.successor) {
    return false;
  }
  return true;
}
