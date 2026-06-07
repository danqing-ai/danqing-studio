/**
 * Lightweight unit checks for canvas staging / lineage placement.
 * Run: npx tsx frontend/src/utils/canvasStaging.unit.ts
 */
import type { CanvasItemState, GalleryItem } from '@/types';
import { lineageSlotBesideParent, parentAssetPath } from './canvasStaging';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const galleryItems = [
  { path: 'asset:parent', name: 'parent.png', width: 512, height: 512, metadata: {} },
  {
    path: 'asset:child',
    name: 'child.png',
    width: 512,
    height: 512,
    metadata: { parent_asset_id: 'parent', relation_type: 'retouch' },
  },
] as GalleryItem[];

const parent = galleryItems[0];
const child = galleryItems[1];

const items: Record<string, CanvasItemState> = {
  'asset:parent': { x: 100, y: 200, scale: 1, visible: true, zIndex: 1, note: '' },
};

assert(parentAssetPath(child) === 'asset:parent', 'child parent path');
assert(parentAssetPath(parent) === null, 'root has no parent');

const slot = lineageSlotBesideParent(child, galleryItems, items, 0);
assert(!!slot, 'slot exists');
assert(slot!.x === 100 + 512 + 48, `unexpected x ${slot!.x}`);
assert(slot!.y === 200, `unexpected y ${slot!.y}`);

const withSibling = [
  ...galleryItems,
  {
    path: 'asset:child2',
    name: 'child2.png',
    width: 512,
    height: 512,
    metadata: { parent_asset_id: 'parent', relation_type: 'retouch' },
  },
] as GalleryItem[];
const sibling = withSibling[2];
const slot2 = lineageSlotBesideParent(sibling, withSibling, {
  ...items,
  'asset:child': { x: 660, y: 200, scale: 1, visible: true, zIndex: 2, note: '' },
});
assert(!!slot2, 'sibling slot exists');
assert(slot2!.y === 200 + Math.max(80, 512 * 0.35), `unexpected sibling y ${slot2!.y}`);

console.log('canvasStaging.unit.ts: ok');
