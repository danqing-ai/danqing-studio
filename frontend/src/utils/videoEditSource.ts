import type { GalleryItem } from '@/types';
import { isVideoGalleryItem } from '@/utils/canvasAssets';

export type VideoEditSourceMode = 'first_frame' | 'image_only';

export type AnimateReferenceAcceptKind = 'image' | 'video' | 'image_or_video';

export function resolveVideoEditSourceMode(
  parameters: Record<string, unknown> | undefined | null,
): VideoEditSourceMode {
  return String(parameters?.video_edit_source_mode || 'image_only') === 'first_frame'
    ? 'first_frame'
    : 'image_only';
}

export function animateReferenceAcceptKind(
  parameters: Record<string, unknown> | undefined | null,
): AnimateReferenceAcceptKind {
  return 'image';
}

export function videoSupportsVideoEdit(
  actions: Record<string, unknown> | undefined | null,
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  if (!actions || typeof actions !== 'object') return false;
  if (!Object.prototype.hasOwnProperty.call(actions, 'animate') || actions.animate == null) {
    return false;
  }
  return resolveVideoEditSourceMode(parameters) === 'first_frame';
}

export function galleryPathIsVideo(path: string, item?: GalleryItem | null): boolean {
  if (item) return isVideoGalleryItem(item);
  if (!path.startsWith('asset:')) {
    const base = path.split(/[/\\]/).pop()?.toLowerCase() || '';
    return /\.(mp4|mov|webm|mkv|avi)$/.test(base);
  }
  return false;
}
