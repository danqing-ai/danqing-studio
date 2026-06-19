import { ref, watch, type Ref } from 'vue';
import { DQ_STORAGE, getItem, setItem, type StorageKey } from '@/utils/storage';

function readStoredBool(key: StorageKey, fallback: boolean): boolean {
  const raw = getItem(key);
  if (raw === '1') return true;
  if (raw === '0') return false;
  return fallback;
}

export interface ModelRegistryFilterDefaults {
  installedOnly?: boolean;
  commercialOnly?: boolean;
  currentModelsOnly?: boolean;
}

export function useModelRegistryFilters(defaults: ModelRegistryFilterDefaults = {}) {
  const installedOnly = ref(
    readStoredBool(DQ_STORAGE.MODEL_FILTER_INSTALLED, defaults.installedOnly ?? false),
  );
  const commercialOnly = ref(
    readStoredBool(DQ_STORAGE.MODEL_FILTER_COMMERCIAL, defaults.commercialOnly ?? false),
  );
  const currentModelsOnly = ref(
    readStoredBool(DQ_STORAGE.MODEL_FILTER_CURRENT_ONLY, defaults.currentModelsOnly ?? false),
  );

  watch(installedOnly, (v) => setItem(DQ_STORAGE.MODEL_FILTER_INSTALLED, v ? '1' : '0'));
  watch(commercialOnly, (v) => setItem(DQ_STORAGE.MODEL_FILTER_COMMERCIAL, v ? '1' : '0'));
  watch(currentModelsOnly, (v) => setItem(DQ_STORAGE.MODEL_FILTER_CURRENT_ONLY, v ? '1' : '0'));

  return { installedOnly, commercialOnly, currentModelsOnly };
}

export interface VersionPickerFields {
  model: string | undefined;
  version: string | undefined;
}

/** Re-select first row when current model|version drops out of filtered picker list. */
export function reconcileVersionPickerSelection<T extends { modelKey: string; versionKey: string; ready?: boolean }>(
  rows: T[],
  fields: VersionPickerFields,
  selectedKey: Ref<string | undefined>,
): boolean {
  const model = fields.model;
  const version = fields.version;
  const stillValid =
    model &&
    version &&
    rows.some((r) => r.modelKey === model && r.versionKey === version);
  if (stillValid) {
    return false;
  }
  const pick = rows.find((r) => r.ready) || rows[0];
  if (!pick) {
    return false;
  }
  fields.model = pick.modelKey;
  fields.version = pick.versionKey;
  selectedKey.value = `${pick.modelKey}|${pick.versionKey}`;
  return true;
}
