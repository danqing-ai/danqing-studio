/** Registry model row: commercial_use_allowed from models_registry.json */

export type CommercialUseFlag = boolean | null | undefined;

export function isCommercialUseAllowed(flag: CommercialUseFlag): boolean {
  return flag === true;
}
