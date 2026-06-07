/**
 * Lightweight unit checks for canvas edge utilities.
 * Run: npx tsx frontend/src/utils/canvasEdges.unit.ts
 */
import type { CanvasEdge, CanvasItemState } from '@/types';
import { buildSessionEdges, computeNodeDepths } from './canvasEdges';
import { countItemsInStaging, layoutInStaging } from './canvasStaging';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const items: Record<string, CanvasItemState> = {
  'asset:a': { x: 0, y: 0, scale: 1, visible: true, zIndex: 1, note: '' },
  'asset:b': { x: 100, y: 0, scale: 1, visible: true, zIndex: 2, note: '' },
  'asset:c': { x: 200, y: 0, scale: 1, visible: true, zIndex: 3, note: '' },
};

const galleryItems = [
  { path: 'asset:a', name: 'a', width: 512, height: 512, metadata: {} },
  {
    path: 'asset:b',
    name: 'b',
    width: 512,
    height: 512,
    metadata: { parent_asset_id: 'a', relation_type: 'retouch' },
  },
  {
    path: 'asset:c',
    name: 'c',
    width: 512,
    height: 512,
    metadata: { parent_asset_id: 'b', relation_type: 'create' },
  },
] as import('@/types').GalleryItem[];

const edges = buildSessionEdges(items, galleryItems);
assert(edges.length === 2, `expected 2 edges, got ${edges.length}`);
assert(edges.some((e) => e.from === 'asset:a' && e.to === 'asset:b'), 'missing a->b');
assert(edges.some((e) => e.from === 'asset:b' && e.to === 'asset:c'), 'missing b->c');

const depths = computeNodeDepths(items, edges as CanvasEdge[]);
assert(depths.get('asset:a') === 0, 'root depth');
assert(depths.get('asset:b') === 1, 'child depth');
assert(depths.get('asset:c') === 2, 'grandchild depth');

const staging = { x: 100, y: 80, width: 512, height: 512, visible: true };
const slot0 = layoutInStaging(0, galleryItems[0], staging);
const slot1 = layoutInStaging(1, galleryItems[1], staging);
assert(slot0.x >= staging.x, 'staging slot inside box');
assert(slot1.y >= slot0.y, 'second slot below first');
assert(countItemsInStaging(items, staging) === 0, 'empty staging count');

console.log('canvasEdges.unit: OK');
