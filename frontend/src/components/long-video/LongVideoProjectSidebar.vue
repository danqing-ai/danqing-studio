<template>
  <DqSurfaceCard class="lv-sidebar-card">
    <div class="lv-sidebar-card__head">
      <span class="lv-sidebar-card__title">{{ $tt('video.longVideoProjectList') }}</span>
      <DqButton type="primary" size="sm" block @click="$emit('new-project')">
        + {{ $tt('video.longVideoNewProject') }}
      </DqButton>
    </div>

    <div class="lv-sidebar-card__body">
      <p v-if="loading" class="lv-sidebar-card__hint">{{ $tt('common.loading') }}</p>
      <p v-else-if="!projects.length" class="lv-sidebar-card__hint">
        {{ $tt('video.longVideoProjectListEmpty') }}
      </p>

      <ul v-else class="lv-sidebar-card__list" role="list">
        <li
          v-for="item in projects"
          :key="item.id"
          class="lv-sidebar-card__item"
          :class="{ 'is-active': item.id === activeProjectId }"
        >
          <button type="button" class="lv-sidebar-card__open" @click="$emit('open', item.id)">
            <span class="lv-sidebar-card__name">{{ item.title }}</span>
            <span class="lv-sidebar-card__meta">
              {{ $tt('video.longVideoProjectListMeta', { shots: item.shot_count }) }}
              · {{ formatUpdated(item.updated_at) }}
            </span>
          </button>
          <button
            type="button"
            class="lv-sidebar-card__delete"
            :title="$tt('common.delete')"
            @click.stop="$emit('delete', item.id)"
          >
            <DqIcon :size="14"><Delete /></DqIcon>
          </button>
        </li>
      </ul>
    </div>
  </DqSurfaceCard>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n';
import { Delete } from '@danqing/dq-shell';
import type { LongVideoProjectSummary } from '@/types';

defineProps<{
  projects: LongVideoProjectSummary[];
  activeProjectId?: string;
  loading?: boolean;
}>();

defineEmits<{
  (e: 'open', projectId: string): void;
  (e: 'new-project'): void;
  (e: 'delete', projectId: string): void;
}>();

const { t: $tt, locale } = useI18n();

function formatUpdated(iso: string): string {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleString(locale.value === 'zh' ? 'zh-CN' : 'en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}
</script>

<style scoped>
.lv-sidebar-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.lv-sidebar-card :deep(.dq-surface-card__body) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.lv-sidebar-card__head {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 2px 2px 12px;
  flex-shrink: 0;
}

.lv-sidebar-card__title {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.lv-sidebar-card__body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.lv-sidebar-card__hint {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  text-align: center;
  padding: 24px 8px;
}

.lv-sidebar-card__list {
  list-style: none;
  margin: 0;
  padding: 0;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lv-sidebar-card__item {
  display: flex;
  align-items: stretch;
  gap: 4px;
  border-radius: 10px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-surface-elevated) 60%, transparent);
  overflow: hidden;
}

.lv-sidebar-card__item.is-active {
  border-color: color-mix(in srgb, var(--dq-accent) 45%, transparent);
  background: color-mix(in srgb, var(--dq-accent) 8%, var(--dq-surface-elevated));
}

.lv-sidebar-card__open {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  padding: 10px 12px;
  border: none;
  background: transparent;
  cursor: pointer;
  text-align: left;
  color: inherit;
}

.lv-sidebar-card__open:hover {
  background: color-mix(in srgb, var(--dq-accent) 6%, transparent);
}

.lv-sidebar-card__name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
}

.lv-sidebar-card__meta {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lv-sidebar-card__delete {
  flex-shrink: 0;
  width: 36px;
  border: none;
  border-left: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  background: transparent;
  color: var(--dq-label-tertiary);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.lv-sidebar-card__delete:hover {
  color: var(--dq-danger);
  background: color-mix(in srgb, var(--dq-danger) 8%, transparent);
}
</style>
