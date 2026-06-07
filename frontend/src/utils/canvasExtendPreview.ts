import type {
  CanvasExtendDirection,
  CanvasExtendPreviewState,
  CanvasItemState,
  GalleryItem,
} from '@/types';

export interface ExtendPreviewZone {
  dir: CanvasExtendDirection;
  x: number;
  y: number;
  width: number;
  height: number;
}

export function computeExtendPreviewZones(
  preview: CanvasExtendPreviewState,
  editingPath: string,
  items: Record<string, CanvasItemState>,
  galleryItems: GalleryItem[]
): ExtendPreviewZone[] {
  const state = items[editingPath];
  const gi = galleryItems.find((g) => g.path === editingPath);
  if (!state || !gi) return [];

  const w = (gi.width || 512) * state.scale;
  const h = (gi.height || 512) * state.scale;
  const x = state.x;
  const y = state.y;
  const px = Math.min(2048, Math.max(64, Number(preview.pixels) || 256));
  const dirs = preview.directions.filter((d) =>
    ['top', 'bottom', 'left', 'right'].includes(d)
  ) as CanvasExtendDirection[];

  return dirs.map((dir) => {
    let left = x;
    let top = y;
    let width = w;
    let height = h;
    if (dir === 'top') {
      top = y - px;
      height = px;
    } else if (dir === 'bottom') {
      top = y + h;
      height = px;
    } else if (dir === 'left') {
      left = x - px;
      width = px;
    } else if (dir === 'right') {
      left = x + w;
      width = px;
    }
    return { dir, x: left, y: top, width, height };
  });
}
