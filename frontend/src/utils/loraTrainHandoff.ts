import type { Router } from 'vue-router';
import type { GalleryItem } from '@/types';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { isImageGalleryItem } from '@/utils/canvasAssets';
import { DQ_STORAGE, setItem } from '@/utils/storage';

export interface LoraTrainHandoffOptions {
  datasetName?: string;
  datasetId?: string;
}

export function assetIdsFromGalleryPaths(paths: Iterable<string>): string[] {
  const ids: string[] = [];
  for (const path of paths) {
    const id = assetIdFromGalleryPath(path);
    if (id) ids.push(id);
  }
  return [...new Set(ids)];
}

export function assetIdsFromGalleryItems(items: GalleryItem[]): string[] {
  return assetIdsFromGalleryPaths(
    items.filter(isImageGalleryItem).map((item) => item.path)
  );
}

export function navigateToLoraTrainWithAssets(
  router: Router,
  assetIds: string[],
  opts: LoraTrainHandoffOptions = {}
): boolean {
  const unique = [...new Set(assetIds.map((id) => id.trim()).filter(Boolean))];
  if (!unique.length) return false;
  const query: Record<string, string> = { import: unique.join(',') };
  if (opts.datasetName?.trim()) query.dataset_name = opts.datasetName.trim();
  if (opts.datasetId?.trim()) query.dataset_id = opts.datasetId.trim();
  void router.push({ name: 'lora_train', query });
  return true;
}

export function openLoraTrainingRun(router: Router, taskId: string): void {
  if (!taskId.trim()) return;
  void router.push({ name: 'lora_train', query: { run: taskId.trim() } });
}

export function openModelsUserLoras(router: Router): void {
  setItem(DQ_STORAGE.MODELS_CATEGORY, 'trained_loras');
  void router.push({ name: 'models' });
}
