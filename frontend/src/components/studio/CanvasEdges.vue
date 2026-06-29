<template>
  <svg class="canvas-edges" aria-hidden="true">
    <defs>
      <marker
        id="canvas-edge-arrow"
        markerWidth="8"
        markerHeight="8"
        refX="6"
        refY="3"
        orient="auto"
        markerUnits="strokeWidth"
      >
        <path d="M0,0 L6,3 L0,6 Z" fill="var(--dq-accent)" opacity="0.7" />
      </marker>
      <marker
        id="canvas-edge-arrow-active"
        markerWidth="10"
        markerHeight="10"
        refX="7"
        refY="3.5"
        orient="auto"
        markerUnits="strokeWidth"
      >
        <path d="M0,0 L7,3.5 L0,7 Z" fill="var(--dq-accent)" opacity="1" />
      </marker>
    </defs>
    <g v-for="(edge, idx) in edgeLines" :key="idx">
      <line
        :x1="edge.x1"
        :y1="edge.y1"
        :x2="edge.x2"
        :y2="edge.y2"
        class="canvas-edges__line"
        :class="{ 'canvas-edges__line--active': edge.active }"
        :marker-end="edge.active ? 'url(#canvas-edge-arrow-active)' : 'url(#canvas-edge-arrow)'"
      />
      <text
        v-if="edge.label"
        :x="edge.lx"
        :y="edge.ly"
        class="canvas-edges__label"
      >
        {{ edge.label }}
      </text>
    </g>
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { CanvasEdge, CanvasItemState, GalleryItem } from '@/types';
import { computeEdgeLines } from '@/utils/canvasEdges';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';

const props = defineProps<{
  edges: CanvasEdge[];
  items: Record<string, CanvasItemState>;
  galleryItems: GalleryItem[];
  focusPaths?: string[];
}>();

const edgeLines = computed(() => {
  const focusSet = new Set(props.focusPaths || []);
  return computeEdgeLines(props.edges, props.items, props.galleryItems).map((line) => ({
    ...line,
    label: lineageRelationLabel(line.relation),
    active: focusSet.size > 0 && (focusSet.has(line.from) || focusSet.has(line.to)),
  }));
});
</script>

<style scoped>
.canvas-edges {
  position: absolute;
  top: 0;
  left: 0;
  width: 10000px;
  height: 10000px;
  pointer-events: none;
  z-index: 1;
  overflow: visible;
}

.canvas-edges__line {
  stroke: var(--dq-accent);
  stroke-width: 2;
  stroke-opacity: 0.55;
  fill: none;
  transition: stroke-width 0.15s ease, stroke-opacity 0.15s ease;
}

.canvas-edges__line--active {
  stroke-width: 3;
  stroke-opacity: 0.95;
}

.canvas-edges__label {
  font-size: var(--dq-font-size-caption);
  fill: var(--dq-label-secondary);
  text-anchor: middle;
}
</style>
