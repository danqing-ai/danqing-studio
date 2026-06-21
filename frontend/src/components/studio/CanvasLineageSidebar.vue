<template>
  <aside v-if="open" class="canvas-lineage-sidebar dq-glass--popover">
    <header class="canvas-lineage-sidebar__head">
      <div class="canvas-lineage-sidebar__head-text">
        <span>{{ $t('canvas.lineageSidebar') }}</span>
        <span v-if="assetId" class="canvas-lineage-sidebar__hint">{{ $t('canvas.lineageClickHint') }}</span>
      </div>
      <DqIconButton type="text" size="xs" :label="$t('gallery.close')" @click="$emit('close')">
        <DqIcon :size="14"><Close /></DqIcon>
      </DqIconButton>
    </header>

    <div v-if="!assetId" class="canvas-lineage-sidebar__empty">
      {{ $t('canvas.lineageSelectHint') }}
    </div>

    <div v-else-if="loading" class="canvas-lineage-sidebar__loading">
      <DqIcon :size="18"><Loading /></DqIcon>
      <span>{{ $t('common.loading') }}</span>
    </div>

    <div v-else-if="error" class="canvas-lineage-sidebar__error">{{ error }}</div>

    <div v-else-if="!rawData" class="canvas-lineage-sidebar__empty">
      {{ $t('gallery.lineageEmpty') }}
    </div>

    <LineageGraph
      v-else
      :data="rawData"
      :current-node-id="rawData.id"
      :on-canvas-ids="onCanvasIds"
      class="canvas-lineage-sidebar__graph"
      @focus-asset="$emit('focus-asset', $event)"
      @request-close="$emit('close')"
    />
  </aside>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import { Close, Loading } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import type { LineageNode } from '@/types';
import LineageGraph from '@/components/studio/LineageGraph.vue';

const props = defineProps<{
  open: boolean;
  assetId: string;
  onCanvasIds?: string[];
}>();

defineEmits<{
  (e: 'close'): void;
  (e: 'focus-asset', assetId: string): void;
}>();

const loading = ref(false);
const error = ref('');
const rawData = ref<LineageNode | null>(null);

watch(
  () => [props.open, props.assetId] as const,
  async ([open, assetId]) => {
    if (!open || !assetId) {
      rawData.value = null;
      error.value = '';
      return;
    }
    loading.value = true;
    error.value = '';
    rawData.value = null;
    try {
      rawData.value = (await api.gen.getAssetLineage(assetId)) as LineageNode;
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      error.value = msg;
    } finally {
      loading.value = false;
    }
  },
  { immediate: true }
);
</script>

<style scoped>
.canvas-lineage-sidebar {
  position: absolute;
  top: 52px;
  right: 12px;
  bottom: calc(16px + var(--dq-composer-reserve, min(200px, 36vh)) + 56px);
  width: min(360px, 38vw);
  z-index: 45;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--dq-border-subtle);
  border-radius: 10px;
  overflow: hidden;
  pointer-events: auto;
}

.canvas-lineage-sidebar__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  font-size: 13px;
  font-weight: 600;
  border-bottom: 1px solid var(--dq-border-subtle);
}

.canvas-lineage-sidebar__head-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.canvas-lineage-sidebar__hint {
  font-size: 10px;
  font-weight: 400;
  color: var(--dq-label-tertiary);
}

.canvas-lineage-sidebar__graph {
  flex: 1;
  min-height: 0;
}

.canvas-lineage-sidebar__empty,
.canvas-lineage-sidebar__loading,
.canvas-lineage-sidebar__error {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 24px 16px;
  font-size: 12px;
  color: var(--dq-label-secondary);
  text-align: center;
}
</style>
