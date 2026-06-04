<template>
  <DqDialog
    v-model:open="open"
    class="gen-task-log-dialog"
    :title="$t('studio.logs')"
    width="min(520px, 92vw)"
    variant="glass"
    destroy-on-close
  >
    <GenTaskLogPanel
      v-if="taskId"
      :task-id="taskId"
      in-dialog
    />
    <div v-else class="gen-task-log-dialog__empty">
      {{ $t('studio.logsEmpty') }}
    </div>
  </DqDialog>
</template>

<script setup lang="ts">
import GenTaskLogPanel from './GenTaskLogPanel.vue';

const open = defineModel<boolean>('open', { default: false });

defineProps<{
  taskId?: string | null;
}>();
</script>

<style scoped>
.gen-task-log-dialog__empty {
  padding: 24px 8px;
  text-align: center;
  font-size: 13px;
  color: var(--dq-label-tertiary);
}

:deep(.gen-task-log-panel--dialog .gen-task-log-panel__body) {
  min-height: min(420px, 58vh);
  max-height: min(520px, 70vh);
}
</style>
