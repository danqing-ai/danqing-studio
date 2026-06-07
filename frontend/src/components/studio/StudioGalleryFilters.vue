<template>
  <DqSelect
    :model-value="filterTime"
    size="small"
    class="studio-gallery-filters__time"
    @update:model-value="$emit('update:filterTime', $event)"
  >
    <DqOption
      v-for="opt in timeOptions"
      :key="opt.value"
      :label="opt.label"
      :value="opt.value"
    />
  </DqSelect>

  <DqSelect
    :model-value="filterModels"
    size="small"
    multiple
    collapse-tags
    class="studio-gallery-filters__models"
    :placeholder="$t('gallery.filterModel')"
    @update:model-value="$emit('update:filterModels', $event)"
  >
    <DqOption v-for="m in modelOptions" :key="m" :label="m" :value="m" />
  </DqSelect>

  <div class="studio-gallery-filters__spacer" />

  <div
    v-if="viewMode && supportsCanvas !== false"
    class="studio-gallery-filters__view-mode"
    :aria-label="$t('gallery.viewMode')"
  >
    <span class="studio-gallery-filters__view-label">{{ $t('gallery.viewMode') }}</span>
    <StudioViewModeSwitch
      :model-value="viewMode"
      @update:model-value="onViewModeChange"
    />
  </div>

  <DqButton
    v-if="!viewMode || viewMode === 'grid'"
    size="sm"
    :type="selectionMode ? 'primary' : 'default'"
    class="studio-gallery-filters__select"
    @click="$emit('toggle-selection-mode')"
  >
    {{ selectionMode ? $t('gallery.exitSelectMode') : $t('gallery.selectMode') }}
  </DqButton>

  <DqIconButton
    type="text"
    size="sm"
    :label="$t('common.refresh')"
    class="studio-gallery-filters__refresh"
    @click="$emit('refresh')"
  >
    <DqIcon><Refresh /></DqIcon>
  </DqIconButton>
</template>

<script setup lang="ts">
import { Refresh } from '@danqing/dq-shell';
import StudioViewModeSwitch from '@/components/studio/StudioViewModeSwitch.vue';

const props = defineProps<{
  filterTime: string;
  filterModels: string[];
  timeOptions: { label: string; value: string }[];
  modelOptions: string[];
  selectionMode?: boolean;
  viewMode?: 'grid' | 'canvas';
  supportsCanvas?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:filterTime', value: string): void;
  (e: 'update:filterModels', value: string[]): void;
  (e: 'refresh'): void;
  (e: 'toggle-selection-mode'): void;
  (e: 'update:viewMode', value: 'grid' | 'canvas'): void;
}>();

function onViewModeChange(mode: 'grid' | 'canvas') {
  if (props.viewMode === mode) return;
  emit('update:viewMode', mode);
}
</script>

<style scoped>
.studio-gallery-filters__time {
  width: 112px;
  flex-shrink: 0;
}

.studio-gallery-filters__models {
  flex: 1 1 auto;
  min-width: 0;
  max-width: 280px;
}

.studio-gallery-filters__spacer {
  flex: 1 1 auto;
  min-width: 8px;
}

.studio-gallery-filters__select {
  flex-shrink: 0;
}

.studio-gallery-filters__refresh {
  flex-shrink: 0;
}

.studio-gallery-filters__view-mode {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.studio-gallery-filters__view-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--dq-color-text-secondary);
  white-space: nowrap;
}
</style>
