import type {
  CanvasItemState,
  CanvasSessionState,
  CanvasStagingState,
  CanvasOverlaysState,
  CanvasOverlayLayer,
  CanvasEdge,
  CanvasMaskPreviewState,
  CanvasExtendPreviewState,
  GalleryItem,
} from '@/types';
import { api } from '@/utils/api';
import { describeCanvasNodeViaChat } from '@/utils/llmMessages';
import {
  assetIdFromGalleryPath,
  isAudioGalleryItem,
  previewUrlForGalleryItem,
} from '@/utils/canvasAssets';
import { CANVAS_OVERLAY_KINDS } from '@/utils/canvasOverlays';
import { computeEdgeLines } from '@/utils/canvasEdges';
import { computeExtendPreviewZones } from '@/utils/canvasExtendPreview';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';

export interface CanvasPngExportOptions {
  includeStaging?: boolean;
  includeOverlays?: boolean;
  includeEdges?: boolean;
  includeNotes?: boolean;
  includeComposerPreview?: boolean;
}

export const DEFAULT_CANVAS_PNG_EXPORT_OPTIONS: Required<CanvasPngExportOptions> = {
  includeStaging: true,
  includeOverlays: true,
  includeEdges: true,
  includeNotes: true,
  includeComposerPreview: false,
};

export function loadCanvasPngExportOptions(): Required<CanvasPngExportOptions> {
  try {
    const raw = getItem(DQ_STORAGE.CANVAS_PNG_EXPORT_OPTS);
    if (!raw) return { ...DEFAULT_CANVAS_PNG_EXPORT_OPTIONS };
    const parsed = JSON.parse(raw) as Partial<CanvasPngExportOptions>;
    return { ...DEFAULT_CANVAS_PNG_EXPORT_OPTIONS, ...parsed };
  } catch {
    return { ...DEFAULT_CANVAS_PNG_EXPORT_OPTIONS };
  }
}

export function saveCanvasPngExportOptions(opts: CanvasPngExportOptions) {
  setItem(DQ_STORAGE.CANVAS_PNG_EXPORT_OPTS, JSON.stringify(opts));
}

function loadImageFromObjectUrl(objectUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Failed to decode image blob'));
    img.src = objectUrl;
  });
}

async function loadGalleryThumbnail(item: GalleryItem): Promise<HTMLImageElement | null> {
  const url = previewUrlForGalleryItem(item);
  try {
    const blob = await api.gen.urlToBlob(url);
    const objectUrl = URL.createObjectURL(blob);
    try {
      return await loadImageFromObjectUrl(objectUrl);
    } finally {
      URL.revokeObjectURL(objectUrl);
    }
  } catch {
    return null;
  }
}

function drawAudioPlaceholder(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  item: GalleryItem
) {
  ctx.fillStyle = '#25252c';
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = 'rgba(91, 141, 239, 0.45)';
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
  ctx.font = '600 15px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const label = item.duration_seconds
    ? `♪ ${Math.round(item.duration_seconds)}s`
    : '♪ Audio';
  ctx.fillText(label, x + w / 2, y + h / 2);
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportCanvasJson(payload: {
  sessionId: string;
  title: string;
  media: string;
  state: CanvasSessionState;
}) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json',
  });
  const safe = (payload.title || 'canvas').replace(/[^\w\u4e00-\u9fff-]+/g, '_');
  downloadBlob(blob, `${safe}-${payload.sessionId.slice(0, 12)}.canvas.json`);
}

function expandBounds(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  x: number,
  y: number,
  w: number,
  h: number
): [number, number, number, number] {
  return [
    Math.min(minX, x),
    Math.min(minY, y),
    Math.max(maxX, x + w),
    Math.max(maxY, y + h),
  ];
}

function overlayBounds(
  layer: CanvasOverlayLayer | null | undefined,
  galleryItems: GalleryItem[]
): { x: number; y: number; w: number; h: number } | null {
  if (!layer?.path || layer.visible === false) return null;
  const gi = galleryItems.find((g) => g.path === layer.path);
  const scale = layer.scale || 0.5;
  const w = (gi?.width || 512) * scale;
  const h = (gi?.height || 512) * scale;
  return { x: layer.x, y: layer.y, w, h };
}

function truncateText(text: string, max: number): string {
  const s = text.replace(/\s+/g, ' ').trim();
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}…`;
}

function drawEdgeArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number
) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = 9;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - head * Math.cos(angle - Math.PI / 7),
    y2 - head * Math.sin(angle - Math.PI / 7)
  );
  ctx.lineTo(
    x2 - head * Math.cos(angle + Math.PI / 7),
    y2 - head * Math.sin(angle + Math.PI / 7)
  );
  ctx.closePath();
  ctx.fill();
}

export async function exportCanvasPng(
  items: Record<string, CanvasItemState>,
  galleryItems: GalleryItem[],
  staging: CanvasStagingState,
  opts?: CanvasPngExportOptions & {
    background?: string;
    overlays?: CanvasOverlaysState | null;
    edges?: CanvasEdge[];
    maskPreview?: CanvasMaskPreviewState | null;
    extendPreview?: CanvasExtendPreviewState | null;
    editingPath?: string;
  }
): Promise<Blob> {
  const exportOpts = { ...DEFAULT_CANVAS_PNG_EXPORT_OPTIONS, ...opts };
  const pad = 64;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  const entries: Array<{ path: string; state: CanvasItemState; item: GalleryItem }> = [];
  for (const [path, state] of Object.entries(items)) {
    if (!state.visible) continue;
    const item = galleryItems.find((g) => g.path === path);
    if (!item) continue;
    entries.push({ path, state, item });
    const w = (item.width || 512) * state.scale;
    const h = (item.height || 512) * state.scale;
    [minX, minY, maxX, maxY] = expandBounds(minX, minY, maxX, maxY, state.x, state.y, w, h);
  }

  if (exportOpts.includeComposerPreview && opts?.editingPath) {
    if (opts.extendPreview) {
      for (const zone of computeExtendPreviewZones(
        opts.extendPreview,
        opts.editingPath,
        items,
        galleryItems
      )) {
        [minX, minY, maxX, maxY] = expandBounds(
          minX,
          minY,
          maxX,
          maxY,
          zone.x,
          zone.y,
          zone.width,
          zone.height
        );
      }
    }
    if (opts.maskPreview) {
      const state = items[opts.editingPath];
      const gi = galleryItems.find((g) => g.path === opts.editingPath);
      if (state && gi) {
        const w = (gi.width || 512) * state.scale;
        const h = (gi.height || 512) * state.scale;
        [minX, minY, maxX, maxY] = expandBounds(
          minX,
          minY,
          maxX,
          maxY,
          state.x,
          state.y,
          w,
          h
        );
      }
    }
  }

  if (exportOpts.includeStaging && staging.visible) {
    [minX, minY, maxX, maxY] = expandBounds(
      minX,
      minY,
      maxX,
      maxY,
      staging.x,
      staging.y,
      staging.width,
      staging.height
    );
  }

  const overlayEntries: Array<{
    layer: CanvasOverlayLayer;
    item: GalleryItem;
  }> = [];
  if (exportOpts.includeOverlays && opts?.overlays) {
    for (const kind of CANVAS_OVERLAY_KINDS) {
      const layer = opts.overlays[kind];
      const ob = overlayBounds(layer, galleryItems);
      if (!ob || !layer) continue;
      const item = galleryItems.find((g) => g.path === layer.path);
      if (!item) continue;
      overlayEntries.push({ layer, item });
      [minX, minY, maxX, maxY] = expandBounds(
        minX,
        minY,
        maxX,
        maxY,
        ob.x,
        ob.y,
        ob.w,
        ob.h
      );
    }
  }

  if (!isFinite(minX) || entries.length === 0) {
    throw new Error('No visible items to export');
  }

  const cw = Math.ceil(maxX - minX + pad * 2);
  const ch = Math.ceil(maxY - minY + pad * 2);
  const canvas = document.createElement('canvas');
  canvas.width = Math.min(8192, Math.max(256, cw));
  canvas.height = Math.min(8192, Math.max(256, ch));
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2D unavailable');

  ctx.fillStyle = opts?.background || '#1a1a1e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const edgeLines =
    exportOpts.includeEdges && opts?.edges?.length
      ? computeEdgeLines(opts.edges, items, galleryItems)
      : [];

  if (edgeLines.length > 0) {
    ctx.save();
    ctx.strokeStyle = 'rgba(91, 141, 239, 0.55)';
    ctx.fillStyle = 'rgba(91, 141, 239, 0.75)';
    ctx.lineWidth = 2;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const line of edgeLines) {
      const x1 = line.x1 - minX + pad;
      const y1 = line.y1 - minY + pad;
      const x2 = line.x2 - minX + pad;
      const y2 = line.y2 - minY + pad;
      drawEdgeArrow(ctx, x1, y1, x2, y2);
      const label = lineageRelationLabel(line.relation);
      if (label) {
        ctx.fillStyle = 'rgba(26, 26, 30, 0.72)';
        const lw = ctx.measureText(label).width + 8;
        const lx = line.lx - minX + pad;
        const ly = line.ly - minY + pad;
        ctx.fillRect(lx - lw / 2, ly - 7, lw, 14);
        ctx.fillStyle = 'rgba(220, 220, 230, 0.92)';
        ctx.fillText(label, lx, ly);
        ctx.fillStyle = 'rgba(91, 141, 239, 0.75)';
      }
    }
    ctx.restore();
  }

  if (exportOpts.includeStaging && staging.visible) {
    ctx.strokeStyle = 'rgba(91, 141, 239, 0.6)';
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(
      staging.x - minX + pad,
      staging.y - minY + pad,
      staging.width,
      staging.height
    );
    ctx.setLineDash([]);
  }

  for (const { state, item } of entries) {
    const w = (item.width || 512) * state.scale;
    const h = (item.height || 512) * state.scale;
    const dx = state.x - minX + pad;
    const dy = state.y - minY + pad;

    if (isAudioGalleryItem(item)) {
      drawAudioPlaceholder(ctx, dx, dy, w, h, item);
      continue;
    }

    const img = await loadGalleryThumbnail(item);
    if (img) {
      ctx.drawImage(img, dx, dy, w, h);
    }

    if (exportOpts.includeNotes) {
      const note = (state.label || state.note || item.prompt || '').trim();
      if (note) {
        const text = truncateText(note, 72);
        ctx.font = '11px system-ui, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        const ty = dy + h + 6;
        const tw = ctx.measureText(text).width + 8;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillRect(dx, ty - 2, Math.min(tw, w), 16);
        ctx.fillStyle = 'rgba(235, 235, 245, 0.92)';
        ctx.fillText(text, dx + 4, ty);
      }
    }
  }

  if (exportOpts.includeComposerPreview && opts?.editingPath) {
    if (opts.extendPreview) {
      ctx.save();
      ctx.strokeStyle = 'rgba(91, 141, 239, 0.75)';
      ctx.fillStyle = 'rgba(91, 141, 239, 0.14)';
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      for (const zone of computeExtendPreviewZones(
        opts.extendPreview,
        opts.editingPath,
        items,
        galleryItems
      )) {
        const zx = zone.x - minX + pad;
        const zy = zone.y - minY + pad;
        ctx.fillRect(zx, zy, zone.width, zone.height);
        ctx.strokeRect(zx, zy, zone.width, zone.height);
      }
      ctx.setLineDash([]);
      ctx.restore();
    }
    if (opts.maskPreview?.dataUrl) {
      const state = items[opts.editingPath];
      const gi = galleryItems.find((g) => g.path === opts.editingPath);
      if (state && gi) {
        const w = (gi.width || 512) * state.scale;
        const h = (gi.height || 512) * state.scale;
        const dx = state.x - minX + pad;
        const dy = state.y - minY + pad;
        try {
          const maskImg = await loadImageFromObjectUrl(opts.maskPreview.dataUrl);
          ctx.save();
          ctx.globalAlpha = 0.72;
          ctx.drawImage(maskImg, dx, dy, w, h);
          ctx.strokeStyle = 'rgba(233, 69, 96, 0.85)';
          ctx.lineWidth = 2;
          ctx.strokeRect(dx, dy, w, h);
          ctx.restore();
        } catch {
          /* skip mask preview */
        }
      }
    }
  }

  for (const { layer, item } of overlayEntries) {
    const ob = overlayBounds(layer, galleryItems);
    if (!ob) continue;
    const img = await loadGalleryThumbnail(item);
    if (!img) continue;
    ctx.save();
    ctx.globalAlpha = layer.opacity ?? 0.5;
    ctx.drawImage(
      img,
      ob.x - minX + pad,
      ob.y - minY + pad,
      ob.w,
      ob.h
    );
    ctx.strokeStyle = 'rgba(91, 141, 239, 0.75)';
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(
      ob.x - minX + pad,
      ob.y - minY + pad,
      ob.w,
      ob.h
    );
    ctx.setLineDash([]);
    ctx.restore();
  }

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('PNG export failed'))),
      'image/png'
    );
  });
}

export function assetIdFromPath(path: string): string {
  return assetIdFromGalleryPath(path);
}

export async function describeCanvasNode(
  assetId: string,
  opts?: { preferVision?: boolean; locale?: string },
): Promise<{ note: string; visionUsed: boolean }> {
  return describeCanvasNodeViaChat(assetId, { preferVision: opts?.preferVision });
}
