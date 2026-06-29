<template>
  <div class="studio-gallery-batch-actions">
    <template v-if="selectedCount > 0">
      <span class="studio-gallery-batch-actions__count">
        {{ $tt('gallery.selectedCount', { count: selectedCount }) }}
      </span>
      <DqButton type="text" size="sm" @click="$emit('select-all')">
        {{ allSelected ? $t('gallery.deselectAll') : $t('gallery.selectAll') }}
      </DqButton>
      <DqButton type="danger" size="sm" @click="$emit('batch-delete')">
        <DqIcon><Delete /></DqIcon>
        {{ $t('gallery.batchDelete') }}
      </DqButton>
      <DqButton size="sm" type="secondary" @click="$emit('batch-train-lora')">
        {{ $t('loraTrain.trainFromSelection') }}
      </DqButton>
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('common.close')"
        @click="$emit('clear-selection')"
      >
        <DqIcon><Close /></DqIcon>
      </DqIconButton>
    </template>
    <DqButton
      v-else
      size="sm"
      :type="selectionMode ? 'primary' : 'default'"
      @click="$emit('toggle-selection-mode')"
    >
      {{ selectionMode ? $t('gallery.exitSelectMode') : $t('gallery.selectMode') }}
    </DqButton>
  </div>
</template>

<script setup lang="ts">
import { Close, Delete } from '@danqing/dq-shell';
import { $tt } from '@/utils/i18n';

defineProps<{
  selectionMode?: boolean;
  selectedCount: number;
  allSelected?: boolean;
}>();

defineEmits<{
  (e: 'toggle-selection-mode'): void;
  (e: 'select-all'): void;
  (e: 'batch-delete'): void;
  (e: 'batch-train-lora'): void;
  (e: 'clear-selection'): void;
}>();
</script>

<style scoped>
.studio-gallery-batch-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 0 0 auto;
  flex-shrink: 0;
  justify-content: flex-end;
}

.studio-gallery-batch-actions__count {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-primary);
  white-space: nowrap;
  padding-right: 2px;
}
</style>
