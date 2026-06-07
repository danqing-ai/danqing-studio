/**
 * Unit checks for canvas extend preview zones.
 * Run: npx tsx frontend/src/utils/canvasExtendPreview.unit.ts
 */
import type { CanvasItemState, GalleryItem } from '@/types';
import { computeExtendPreviewZones } from './canvasExtendPreview';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const editingPath = 'asset:src';
const items: Record<string, CanvasItemState> = {
  [editingPath]: { x: 100, y: 200, scale: 1, visible: true, zIndex: 1, note: '' },
};
const galleryItems = [
  { path: editingPath, name: 'src', width: 512, height: 512, metadata: {} },
] as GalleryItem[];

const rightZones = computeExtendPreviewZones(
  { directions: ['right'], pixels: 128 },
  editingPath,
  items,
  galleryItems
);
assert(rightZones.length === 1, 'expected one right zone');
assert(rightZones[0].dir === 'right', 'zone dir');
assert(rightZones[0].x === 100 + 512, `right x expected ${100 + 512}, got ${rightZones[0].x}`);
assert(rightZones[0].width === 128, 'right width');

const multi = computeExtendPreviewZones(
  { directions: ['top', 'left'], pixels: 64 },
  editingPath,
  items,
  galleryItems
);
assert(multi.length === 2, 'expected two zones');

console.log('canvasExtendPreview.unit: OK');
