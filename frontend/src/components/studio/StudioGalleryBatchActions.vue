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
  (e: 'clear-selection'): void;
}>();
</script>

<style scoped>
.studio-gallery-batch-actions {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  flex: 1;
  justify-content: flex-end;
}

.studio-gallery-batch-actions__count {
  font-size: 12px;
  font-weight: 600;
  color: var(--dq-label-primary);
  white-space: nowrap;
  padding-right: 2px;
}
</style>
