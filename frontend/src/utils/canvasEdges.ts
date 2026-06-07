import type { CanvasEdge, CanvasItemState, GalleryItem } from '@/types';

export interface CanvasEdgeLine {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  lx: number;
  ly: number;
  relation: string;
  from: string;
  to: string;
}

/** Screen-space connector geometry for lineage edges between canvas nodes. */
export function computeEdgeLines(
  edges: CanvasEdge[],
  items: Record<string, CanvasItemState>,
  galleryItems: GalleryItem[]
): CanvasEdgeLine[] {
  const lines: CanvasEdgeLine[] = [];

  for (const edge of edges) {
    const fromState = items[edge.from];
    const toState = items[edge.to];
    const fromGi = galleryItems.find((g) => g.path === edge.from);
    const toGi = galleryItems.find((g) => g.path === edge.to);
    if (!fromState?.visible || !toState?.visible || !fromGi || !toGi) continue;

    const fw = (fromGi.width || 512) * fromState.scale;
    const fh = (fromGi.height || 512) * fromState.scale;
    const th = (toGi.height || 512) * toState.scale;

    const x1 = fromState.x + fw;
    const y1 = fromState.y + fh / 2;
    const x2 = toState.x;
    const y2 = toState.y + th / 2;

    lines.push({
      x1,
      y1,
      x2,
      y2,
      lx: (x1 + x2) / 2,
      ly: (y1 + y2) / 2 - 6,
      relation: edge.relation,
      from: edge.from,
      to: edge.to,
    });
  }

  return lines;
}

/** Build lineage edges among nodes currently on the canvas session. */
export function buildSessionEdges(
  canvasItems: Record<string, CanvasItemState>,
  galleryItems: GalleryItem[]
): CanvasEdge[] {
  const onCanvas = new Set(Object.keys(canvasItems));
  const pathByAssetId = new Map<string, string>();
  for (const path of onCanvas) {
    if (path.startsWith('asset:')) {
      pathByAssetId.set(path.slice('asset:'.length), path);
    }
  }

  const edges: CanvasEdge[] = [];
  const seen = new Set<string>();

  for (const path of onCanvas) {
    const gi = galleryItems.find((g) => g.path === path);
    if (!gi) continue;
    const parentId = String(gi.metadata?.parent_asset_id || '').trim();
    if (!parentId) continue;
    const parentPath = pathByAssetId.get(parentId);
    if (!parentPath) continue;
    const relation = String(gi.metadata?.relation_type || 'create');
    const key = `${parentPath}->${path}`;
    if (seen.has(key)) continue;
    seen.add(key);
    edges.push({ from: parentPath, to: path, relation });
  }
  return edges;
}

/** BFS depth from lineage roots (nodes with no parent on canvas). */
export function computeNodeDepths(
  canvasItems: Record<string, CanvasItemState>,
  edges: CanvasEdge[]
): Map<string, number> {
  const children = new Map<string, string[]>();
  const hasParent = new Set<string>();
  for (const edge of edges) {
    const list = children.get(edge.from) || [];
    list.push(edge.to);
    children.set(edge.from, list);
    hasParent.add(edge.to);
  }

  const depth = new Map<string, number>();
  const queue: Array<{ path: string; d: number }> = [];
  for (const path of Object.keys(canvasItems)) {
    if (!hasParent.has(path)) queue.push({ path, d: 0 });
  }
  if (queue.length === 0) {
    for (const path of Object.keys(canvasItems)) depth.set(path, 0);
    return depth;
  }

  while (queue.length > 0) {
    const { path, d } = queue.shift()!;
    if (depth.has(path)) continue;
    depth.set(path, d);
    for (const child of children.get(path) || []) {
      queue.push({ path: child, d: d + 1 });
    }
  }
  for (const path of Object.keys(canvasItems)) {
    if (!depth.has(path)) depth.set(path, 0);
  }
  return depth;
}
