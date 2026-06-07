<template>
  <div class="canvas-session-bar dq-glass--popover">
    <span class="canvas-session-bar__label">{{ $t('canvas.sessionBarLabel') }}</span>
    <DqSelect
      :model-value="sessionId"
      size="small"
      class="canvas-session-bar__select"
      :placeholder="$t('canvas.sessionSelect')"
      @update:model-value="$emit('switch-session', $event)"
    >
      <DqOption
        v-for="s in sessions"
        :key="s.id"
        :label="sessionLabel(s)"
        :value="s.id"
      />
    </DqSelect>

    <DqInput
      v-if="editingTitle"
      v-model="titleDraft"
      size="small"
      class="canvas-session-bar__title-input"
      @keydown.enter="commitRename"
      @blur="commitRename"
    />
    <DqIconButton
      v-else
      type="text"
      size="xs"
      :label="$t('canvas.renameSession')"
      @click="startRename"
    >
      <DqIcon :size="14"><Document /></DqIcon>
    </DqIconButton>

    <DqIconButton
      type="text"
      size="xs"
      :label="$t('canvas.newSession')"
      @click="$emit('create-session')"
    >
      <DqIcon :size="14"><Plus /></DqIcon>
    </DqIconButton>

    <DqIconButton
      v-if="sessions.length > 1"
      type="text"
      size="xs"
      :label="$t('canvas.deleteSession')"
      @click="$emit('delete-session', sessionId)"
    >
      <DqIcon :size="14"><Delete /></DqIcon>
    </DqIconButton>

    <span v-if="syncing" class="canvas-session-bar__sync">{{ $t('canvas.syncing') }}</span>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import { Plus, Delete, Document } from '@danqing/dq-shell';
import type { CanvasSessionSummary } from '@/types';

const props = defineProps<{
  sessionId: string;
  sessionTitle?: string;
  sessions: CanvasSessionSummary[];
  syncing?: boolean;
}>();

const emit = defineEmits<{
  (e: 'switch-session', id: string): void;
  (e: 'create-session'): void;
  (e: 'delete-session', id: string): void;
  (e: 'rename-session', title: string): void;
}>();

const editingTitle = ref(false);
const titleDraft = ref('');

watch(
  () => props.sessionTitle,
  (t) => {
    if (!editingTitle.value) titleDraft.value = t || '';
  },
  { immediate: true }
);

function sessionLabel(s: CanvasSessionSummary): string {
  const n = s.item_count ?? 0;
  return `${s.title} (${n})`;
}

function startRename() {
  titleDraft.value = props.sessionTitle || '';
  editingTitle.value = true;
}

function commitRename() {
  editingTitle.value = false;
  const t = titleDraft.value.trim();
  if (t && t !== props.sessionTitle) {
    emit('rename-session', t);
  }
}
</script>

<style scoped>
.canvas-session-bar {
  position: absolute;
  top: 12px;
  left: 12px;
  z-index: 50;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 8px;
  border: 1px solid var(--dq-color-border);
  pointer-events: auto;
  max-width: calc(100% - 24px);
  flex-wrap: wrap;
}

.canvas-session-bar__label {
  font-size: 11px;
  font-weight: 600;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  padding-right: 2px;
}

.canvas-session-bar__select {
  width: 160px;
}

.canvas-session-bar__title-input {
  width: 120px;
}

.canvas-session-bar__sync {
  font-size: 11px;
  color: var(--dq-color-text-secondary);
}
</style>
