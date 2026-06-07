<template>
  <DqDialog
    :open="open"
    :title="$t('canvas.exportPngOptions')"
    width="400px"
    @update:open="$emit('update:open', $event)"
  >
    <div class="canvas-export-png-dialog">
      <DqCheckbox v-model="local.includeStaging">
        {{ $t('canvas.exportIncludeStaging') }}
      </DqCheckbox>
      <DqCheckbox v-model="local.includeOverlays">
        {{ $t('canvas.exportIncludeOverlays') }}
      </DqCheckbox>
      <DqCheckbox v-model="local.includeEdges">
        {{ $t('canvas.exportIncludeEdges') }}
      </DqCheckbox>
      <DqCheckbox v-model="local.includeNotes">
        {{ $t('canvas.exportIncludeNotes') }}
      </DqCheckbox>
      <DqCheckbox v-model="local.includeComposerPreview">
        {{ $t('canvas.exportIncludeComposerPreview') }}
      </DqCheckbox>
    </div>
    <template #footer>
      <DqButton @click="$emit('update:open', false)">{{ $t('common.cancel') }}</DqButton>
      <DqButton type="primary" @click="onConfirm">{{ $t('canvas.exportPng') }}</DqButton>
    </template>
  </DqDialog>
</template>

<script setup lang="ts">
import { reactive, watch } from 'vue';
import type { CanvasPngExportOptions } from '@/utils/canvasExport';

const props = defineProps<{
  open: boolean;
  modelValue: CanvasPngExportOptions;
}>();

const emit = defineEmits<{
  (e: 'update:open', value: boolean): void;
  (e: 'update:modelValue', value: CanvasPngExportOptions): void;
  (e: 'confirm', value: CanvasPngExportOptions): void;
}>();

const local = reactive<CanvasPngExportOptions>({ ...props.modelValue });

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) Object.assign(local, props.modelValue);
  }
);

function onConfirm() {
  const next = { ...local };
  emit('update:modelValue', next);
  emit('confirm', next);
  emit('update:open', false);
}
</script>

<style scoped>
.canvas-export-png-dialog {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 8px 4px 4px;
}
</style>
