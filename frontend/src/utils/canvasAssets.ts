import type { CanvasItemState, GalleryItem } from '@/types';

export function isAudioGalleryItem(item: GalleryItem | null | undefined): boolean {
  if (!item) return false;
  return (
    item.metadata?.asset_kind === 'audio' ||
    (item.path.startsWith('asset:') && !!item.mime_type?.startsWith('audio/'))
  );
}

export function isVideoGalleryItem(item: GalleryItem | null | undefined): boolean {
  if (!item) return false;
  if (isAudioGalleryItem(item)) return false;
  return (
    item.metadata?.asset_kind === 'video' ||
    (item.duration_seconds != null &&
      item.duration_seconds > 0 &&
      !item.mime_type?.startsWith('audio/'))
  );
}

export function isImageGalleryItem(item: GalleryItem | null | undefined): boolean {
  if (!item) return false;
  return !isAudioGalleryItem(item) && !isVideoGalleryItem(item);
}

export function assetIdFromGalleryPath(path: string): string {
  if (!path.startsWith('asset:')) return '';
  return path.slice('asset:'.length);
}

/** Preview URL for composer / canvas overlays (thumbnail when available). */
/** mm:ss for canvas badges and layer panel. */
export function canvasNodeDisplayName(
  path: string,
  state: CanvasItemState,
  galleryItem?: GalleryItem | null
): string {
  const custom = (state.label || '').trim();
  if (custom) return custom;
  if (galleryItem?.name) return galleryItem.name;
  if (galleryItem?.title) return galleryItem.title;
  return path.startsWith('asset:') ? path.slice('asset:'.length) : path;
}

export function formatGalleryDuration(sec: number): string {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}

/** Grid / picker thumbnail (small WebP from /thumbnail; never full-res file). */
export function galleryThumbnailUrl(item: GalleryItem | null | undefined): string {
  if (!item) return '';
  if (item.thumbnail) return item.thumbnail;
  const id = assetIdFromGalleryPath(item.path);
  if (!id) return item.path;
  return `/api/assets/${id}/thumbnail`;
}

export function previewUrlForGalleryItem(item: GalleryItem | null | undefined): string {
  if (!item) return '';
  const id = assetIdFromGalleryPath(item.path);
  if (!id) return item.path;
  if (isVideoGalleryItem(item) || isAudioGalleryItem(item)) {
    return item.thumbnail || `/api/assets/${id}/thumbnail`;
  }
  return galleryThumbnailUrl(item);
}
