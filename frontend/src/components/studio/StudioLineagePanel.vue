<template>
  <DqDrawer
    :open="modelValue"
    :title="$t('gallery.lineage')"
    direction="rtl"
    size="540px"
    class="studio-lineage-drawer"
    @update:open="$emit('update:modelValue', $event)"
  >
    <div v-if="loading" class="studio-lineage__loading">
      <DqIcon :size="20"><Loading /></DqIcon>
      <span>{{ $t('common.loading') }}</span>
    </div>

    <div v-else-if="error" class="studio-lineage__error">
      {{ error }}
    </div>

    <div v-else-if="!rawData" class="studio-lineage__empty">
      {{ $t('gallery.lineageEmpty') }}
    </div>

    <template v-else>
      <p class="studio-lineage__hint">{{ $t('canvas.lineageDrawerHint') }}</p>
      <LineageGraph
        :data="rawData"
        :current-node-id="rawData.id"
        :on-canvas-ids="onCanvasIds"
        class="studio-lineage__graph"
        @focus-asset="$emit('focus-asset', $event)"
        @request-close="$emit('update:modelValue', false)"
      />
    </template>
  </DqDrawer>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import { Loading } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import type { LineageNode } from '@/types';
import LineageGraph from '@/components/studio/LineageGraph.vue';

const props = defineProps<{
  modelValue: boolean;
  assetId: string;
  onCanvasIds?: string[];
}>();

defineEmits<{
  (e: 'update:modelValue', value: boolean): void;
  (e: 'focus-asset', assetId: string): void;
}>();

const loading = ref(false);
const error = ref('');
const rawData = ref<LineageNode | null>(null);

watch(
  () => props.modelValue,
  async (open) => {
    if (open && props.assetId) {
      loading.value = true;
      error.value = '';
      rawData.value = null;
      try {
        const data = await api.gen.getAssetLineage(props.assetId) as LineageNode;
        rawData.value = data;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : 'Unknown error';
        if (msg.includes('404') || msg.includes('not found')) {
          error.value = 'Asset not found';
        } else {
          error.value = msg;
        }
      } finally {
        loading.value = false;
      }
    }
  }
);
</script>

<style scoped>
.studio-lineage-drawer {
  --dq-drawer-padding: 0;
}

.studio-lineage-drawer :deep(.dq-drawer__body) {
  display: flex;
  flex-direction: column;
  min-height: 0;
  height: 100%;
}

.studio-lineage__hint {
  margin: 0;
  padding: 8px 16px 0;
  font-size: 11px;
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.studio-lineage__graph {
  flex: 1;
  min-height: 0;
}

.studio-lineage__loading,
.studio-lineage__error,
.studio-lineage__empty {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 32px 24px;
  color: var(--dq-label-secondary);
  font-size: 13px;
}
</style>
