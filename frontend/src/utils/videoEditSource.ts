import type { GalleryItem } from '@/types';
import { isVideoGalleryItem } from '@/utils/canvasAssets';

export type VideoEditSourceMode = 'first_frame' | 'image_only' | 'source_video';

export type AnimateReferenceAcceptKind = 'image' | 'video' | 'image_or_video';

const BERNINI_MAX_REFERENCE_IMAGES = 5;

export function berniniMaxReferenceImages(): number {
  return BERNINI_MAX_REFERENCE_IMAGES;
}

export function resolveVideoEditSourceMode(
  parameters: Record<string, unknown> | undefined | null,
): VideoEditSourceMode {
  const raw = String(parameters?.video_edit_source_mode || 'image_only');
  if (raw === 'source_video') return 'source_video';
  if (raw === 'first_frame') return 'first_frame';
  return 'image_only';
}

export function isBerniniRenderer(
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  return Boolean(parameters?.bernini_renderer);
}

export function supportsBerniniReferenceImages(
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  return isBerniniRenderer(parameters);
}

export function videoRequiresSourceVideo(
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  return resolveVideoEditSourceMode(parameters) === 'source_video';
}

export function animateReferenceAcceptKind(
  parameters: Record<string, unknown> | undefined | null,
): AnimateReferenceAcceptKind {
  if (videoRequiresSourceVideo(parameters)) {
    return 'image';
  }
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
  const mode = resolveVideoEditSourceMode(parameters);
  return mode === 'first_frame' || mode === 'source_video';
}

export function galleryPathIsVideo(path: string, item?: GalleryItem | null): boolean {
  if (item) return isVideoGalleryItem(item);
  if (!path.startsWith('asset:')) {
    const base = path.split(/[/\\]/).pop()?.toLowerCase() || '';
    return /\.(mp4|mov|webm|mkv|avi)$/.test(base);
  }
  return false;
}
