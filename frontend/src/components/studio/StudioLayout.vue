<template>
  <div class="studio-layout">
    <!-- Filter bar -->
    <header class="studio-filter-bar dq-glass--bar">
      <div class="studio-filter-bar__inner">
        <slot name="filters" />
      </div>
    </header>

    <!-- Canvas area -->
    <div ref="canvasRef" class="studio-canvas-area" @scroll="onScroll">
      <slot name="canvas" />
    </div>

    <!-- Composer -->
    <div class="studio-composer-bar">
      <div class="studio-composer-bar__scrim" aria-hidden="true" />
      <div class="studio-composer-bar__inner">
        <slot name="composer" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

const emit = defineEmits<{
  (e: 'scroll', event: Event): void;
}>();

const canvasRef = ref<HTMLElement | null>(null);

function onScroll(event: Event) {
  emit('scroll', event);
}

function scrollToTop() {
  canvasRef.value?.scrollTo({ top: 0, behavior: 'smooth' });
}

defineExpose({ scrollToTop, canvasRef });
</script>

<style scoped>
.studio-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  position: relative;
  overflow: hidden;
  background-color: var(--dq-bg-page);
  background-image:
    radial-gradient(ellipse 120% 420px at 50% -8%, rgba(10, 132, 255, 0.07) 0%, transparent 55%),
    var(--dq-bg-page-glow);
}

.studio-filter-bar {
  flex-shrink: 0;
  z-index: 10;
}

.studio-filter-bar__inner {
  display: flex;
  align-items: center;
  gap: 10px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  padding: 10px 20px;
  box-sizing: border-box;
}

.studio-canvas-area {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 20px 20px 220px;
  scroll-behavior: smooth;
}

.studio-composer-bar {
  position: fixed;
  bottom: 0;
  left: var(--dq-shell-sidebar-width, 60px);
  right: 0;
  padding: 0 24px 24px;
  z-index: 100;
  pointer-events: none;
}

.studio-composer-bar__scrim {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: min(220px, 38vh);
  pointer-events: none;
  background: linear-gradient(
    to top,
    color-mix(in srgb, var(--dq-bg-page) 96%, transparent) 0%,
    color-mix(in srgb, var(--dq-bg-page) 72%, transparent) 42%,
    transparent 100%
  );
}

.studio-composer-bar__inner {
  position: relative;
  max-width: 920px;
  margin: 0 auto;
  pointer-events: auto;
}

.studio-canvas-area::-webkit-scrollbar {
  width: 6px;
}

.studio-canvas-area::-webkit-scrollbar-track {
  background: transparent;
}

.studio-canvas-area::-webkit-scrollbar-thumb {
  background: var(--dq-scrollbar-thumb);
  border-radius: 3px;
}

.studio-canvas-area::-webkit-scrollbar-thumb:hover {
  background: var(--dq-scrollbar-thumb-hover);
}

@media (prefers-reduced-transparency: reduce) {
  .studio-layout {
    background-image: none;
  }

  .studio-composer-bar__scrim {
    background: linear-gradient(to top, var(--dq-bg-page) 0%, transparent 100%);
  }
}
</style>
