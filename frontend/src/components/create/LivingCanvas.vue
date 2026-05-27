<!-- @ts-nocheck -->
<template>
  <div class="living-canvas" :class="{ 'living-canvas--active': !!src }">
    <CreateImagePreview
      v-if="src"
      :src="src"
      :hue="hue"
      :alt="alt"
      eager-load
      class="living-canvas__image"
      @download="$emit('download')"
      @expand="$emit('expand')"
    />
    <DqEmpty v-else class="studio-preview-pane__empty" :description="emptyText" />
    <p v-if="stepHint" class="living-canvas__step-hint">{{ stepHint }}</p>
  </div>
</template>

<script setup>
import CreateImagePreview from '@/components/create/CreateImagePreview.vue';

defineProps({
  src: { type: String, default: '' },
  hue: { type: Number, default: 210 },
  alt: { type: String, default: '' },
  emptyText: { type: String, default: '' },
  stepHint: { type: String, default: '' },
});

defineEmits(['download', 'expand']);
</script>

<style scoped>
.living-canvas {
  position: relative;
  width: 100%;
}

.living-canvas__image {
  transition: opacity 0.2s ease;
}

.living-canvas--active .living-canvas__image :deep(.studio-preview-stage__media) {
  animation: living-canvas-fade 0.25s ease;
}

.living-canvas__step-hint {
  display: inline-block;
  margin: 8px auto 0;
  padding: 4px 10px;
  font-size: 11px;
  line-height: 1.35;
  color: hsl(var(--dq-text-secondary));
  background: hsl(var(--dq-surface-elevated, 220 12% 14%) / 0.7);
  border: 0.5px solid var(--dq-border-subtle, transparent);
  border-radius: 999px;
  text-align: center;
}

@keyframes living-canvas-fade {
  from {
    opacity: 0.65;
  }
  to {
    opacity: 1;
  }
}
</style>
