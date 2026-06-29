<template>
  <svg v-if="visible && staging?.visible" class="canvas-region-guides" aria-hidden="true">
    <rect
      :x="staging.x"
      :y="staging.y"
      :width="staging.width"
      :height="staging.height"
      class="canvas-region-guides__staging"
      rx="12"
      ry="12"
    />
    <text
      :x="staging.x + staging.width / 2"
      :y="staging.y - 10"
      class="canvas-region-guides__staging-label"
      text-anchor="middle"
    >
      {{ $t('canvas.guideStagingZone') }}
    </text>

    <g v-for="(link, idx) in overlayLinks" :key="idx">
      <line
        :x1="link.x1"
        :y1="link.y1"
        :x2="link.x2"
        :y2="link.y2"
        class="canvas-region-guides__link"
      />
      <text
        :x="link.mx"
        :y="link.my"
        class="canvas-region-guides__link-label"
        text-anchor="middle"
      >
        {{ link.badge }}
      </text>
    </g>
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type {
  CanvasOverlaysState,
  CanvasOverlayKind,
  CanvasStagingState,
  GalleryItem,
} from '@/types';
import { OVERLAY_BADGES, OVERLAY_KINDS_BY_MEDIA } from '@/utils/canvasOverlays';
import type { CanvasMedia } from '@/composables/useCanvasStore';

const props = defineProps<{
  visible: boolean;
  staging: CanvasStagingState | null | undefined;
  overlays?: CanvasOverlaysState | null;
  galleryItems: GalleryItem[];
  media?: CanvasMedia;
}>();

const staging = computed(() => props.staging);

const overlayLinks = computed(() => {
  const st = props.staging;
  if (!st?.visible || !props.overlays) return [];
  const media = props.media || 'image';
  const kinds = OVERLAY_KINDS_BY_MEDIA[media];
  const cx = st.x + st.width / 2;
  const cy = st.y + st.height / 2;
  const links: Array<{
    x1: number;
    y1: number;
    x2: number;
    y2: number;
    mx: number;
    my: number;
    badge: string;
  }> = [];

  for (const kind of kinds) {
    const layer = props.overlays[kind as CanvasOverlayKind];
    if (!layer?.path || layer.visible === false) continue;
    const gi = props.galleryItems.find((g) => g.path === layer.path);
    if (!gi) continue;
    const w = (gi.width || 512) * (layer.scale || 0.5);
    const h = (gi.height || 512) * (layer.scale || 0.5);
    const ox = layer.x + w / 2;
    const oy = layer.y + h / 2;
    links.push({
      x1: ox,
      y1: oy,
      x2: cx,
      y2: cy,
      mx: (ox + cx) / 2,
      my: (oy + cy) / 2 - 6,
      badge: OVERLAY_BADGES[kind as CanvasOverlayKind] || kind,
    });
  }
  return links;
});
</script>

<style scoped>
.canvas-region-guides {
  position: absolute;
  top: 0;
  left: 0;
  width: 10000px;
  height: 10000px;
  pointer-events: none;
  z-index: 2;
  overflow: visible;
}

.canvas-region-guides__staging {
  fill: color-mix(in srgb, var(--dq-warning) 6%, transparent);
  stroke: var(--dq-warning);
  stroke-width: 1.5;
  stroke-dasharray: 6 4;
  stroke-opacity: 0.75;
}

.canvas-region-guides__staging-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  fill: var(--dq-warning);
  opacity: 0.9;
}

.canvas-region-guides__link {
  stroke: var(--dq-warning);
  stroke-width: 1.5;
  stroke-dasharray: 5 4;
  stroke-opacity: 0.65;
}

.canvas-region-guides__link-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  fill: var(--dq-warning);
  opacity: 0.85;
}
</style>
