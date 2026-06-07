import type { CanvasItemState, GalleryItem } from '@/types';

export interface ItemBounds {
  path: string;
  x: number;
  y: number;
  w: number;
  h: number;
  cx: number;
  cy: number;
}

export function itemBounds(
  path: string,
  state: CanvasItemState,
  item: GalleryItem | undefined
): ItemBounds | null {
  if (!state || !item) return null;
  const baseW = item.width || 512;
  const baseH = item.height || 512;
  const w = baseW * state.scale;
  const h = baseH * state.scale;
  return {
    path,
    x: state.x,
    y: state.y,
    w,
    h,
    cx: state.x + w / 2,
    cy: state.y + h / 2,
  };
}

export type AlignMode = 'left' | 'center' | 'right' | 'top' | 'middle' | 'bottom';
export type DistributeMode = 'horizontal' | 'vertical';

export function computeAlignPositions(
  bounds: ItemBounds[],
  mode: AlignMode
): Record<string, { x: number; y: number }> {
  if (bounds.length < 2) return {};
  const out: Record<string, { x: number; y: number }> = {};
  const minX = Math.min(...bounds.map((b) => b.x));
  const maxX = Math.max(...bounds.map((b) => b.x + b.w));
  const minY = Math.min(...bounds.map((b) => b.y));
  const maxY = Math.max(...bounds.map((b) => b.y + b.h));
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;

  for (const b of bounds) {
    let x = b.x;
    let y = b.y;
    switch (mode) {
      case 'left':
        x = minX;
        break;
      case 'center':
        x = midX - b.w / 2;
        break;
      case 'right':
        x = maxX - b.w;
        break;
      case 'top':
        y = minY;
        break;
      case 'middle':
        y = midY - b.h / 2;
        break;
      case 'bottom':
        y = maxY - b.h;
        break;
      default:
        break;
    }
    out[b.path] = { x, y };
  }
  return out;
}

export function computeDistributePositions(
  bounds: ItemBounds[],
  mode: DistributeMode
): Record<string, { x: number; y: number }> {
  if (bounds.length < 3) return {};
  const sorted = [...bounds].sort((a, b) =>
    mode === 'horizontal' ? a.x - b.x : a.y - b.y
  );
  const out: Record<string, { x: number; y: number }> = {};
  const first = sorted[0];
  const last = sorted[sorted.length - 1];

  if (mode === 'horizontal') {
    const span = last.x + last.w - first.x;
    const totalW = sorted.reduce((s, b) => s + b.w, 0);
    const gap = (span - totalW) / (sorted.length - 1);
    let cursor = first.x;
    for (const b of sorted) {
      out[b.path] = { x: cursor, y: b.y };
      cursor += b.w + gap;
    }
  } else {
    const span = last.y + last.h - first.y;
    const totalH = sorted.reduce((s, b) => s + b.h, 0);
    const gap = (span - totalH) / (sorted.length - 1);
    let cursor = first.y;
    for (const b of sorted) {
      out[b.path] = { x: b.x, y: cursor };
      cursor += b.h + gap;
    }
  }
  return out;
}

export function rectIntersects(
  ax: number,
  ay: number,
  aw: number,
  ah: number,
  bx: number,
  by: number,
  bw: number,
  bh: number
): boolean {
  return ax < bx + bw && ax + aw > bx && ay < by + bh && ay + ah > by;
}
