<template>
  <div class="lora-train-history" :class="`lora-train-history--${variant}`">
    <div v-if="!hideHeader" class="lora-train-history__head">
      <div class="lora-train-history__head-row">
        <span class="lora-train-history__title">{{ $t('loraTrain.recentRuns') }}</span>
        <DqButton
          v-if="!hideRefresh"
          size="xs"
          type="text"
          :loading="loading"
          @click="refresh()"
        >
          {{ $t('loraTrain.refreshHistory') }}
        </DqButton>
      </div>
    </div>

    <div v-if="loading && !runs.length" class="lora-train-history__skeleton">
      <div v-for="i in skeletonCount" :key="i" class="lora-train-history__skeleton-row" />
    </div>

    <p v-else-if="!loading && !runs.length" class="lora-train-history__empty-hint">
      {{ $t('loraTrain.noRecentRuns') }}
    </p>

    <div v-else class="lora-train-history__list">
      <button
        v-for="run in runs"
        :key="runTaskId(run)"
        type="button"
        class="lora-train-history__item"
        :class="[
          `is-${String(run.status || 'unknown')}`,
          { 'is-active': runTaskId(run) === activeId },
        ]"
        @click="emit('select', runTaskId(run))"
      >
        <span class="lora-train-history__status-bar" aria-hidden="true" />

        <span class="lora-train-history__item-content">
          <span class="lora-train-history__item-top">
            <span class="lora-train-history__item-name">{{ runOutputName(run) }}</span>
            <DqTag size="small" :type="statusType(String(run.status || ''))" effect="plain">
              {{ statusLabel(String(run.status || '')) }}
            </DqTag>
            <DqTag
              v-if="userLoraForRun(run)"
              size="small"
              type="success"
              effect="plain"
              class="lora-train-history__registered"
            >
              {{ $t('loraTrain.registeredBadge') }}
            </DqTag>
          </span>
          <span class="lora-train-history__item-meta">
            <span class="lora-train-history__base">{{ runBaseModel(run) }}</span>
            <span class="lora-train-history__sep" aria-hidden="true">·</span>
            <span class="lora-train-history__when">{{ formatWhen(run) }}</span>
          </span>
        </span>

        <DqIcon v-if="variant === 'page'" class="lora-train-history__chevron" aria-hidden="true">
          <arrow-right />
        </DqIcon>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import { ArrowRight } from '@danqing/dq-shell';
import {
  runBaseModel,
  runOutputName,
  runTaskId,
  useLoraTrainLibrary,
} from '@/composables/useLoraTrainLibrary';

const props = withDefaults(
  defineProps<{
    activeId?: string;
    hideRefresh?: boolean;
    hideHeader?: boolean;
    variant?: 'rail' | 'page';
    limit?: number;
  }>(),
  {
    activeId: '',
    hideRefresh: false,
    hideHeader: false,
    variant: 'rail',
    limit: 12,
  }
);

const emit = defineEmits<{
  (e: 'select', taskId: string): void;
}>();

const { t, locale } = useI18n();
const { runs, loading, refresh, userLoraForRun } = useLoraTrainLibrary();

const skeletonCount = computed(() => (props.variant === 'page' ? 5 : 4));

function statusType(status: string): string {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'running') return 'primary';
  if (status === 'queued') return 'info';
  return 'info';
}

function statusLabel(status: string): string {
  const key = `loraTrain.runStatus.${status}`;
  const translated = t(key);
  return translated !== key ? translated : status;
}

function formatWhen(run: Record<string, unknown>): string {
  const raw = run.created_at || run.started_at || run.updated_at;
  if (!raw) return '';
  try {
    return new Date(String(raw)).toLocaleString(locale.value === 'zh' ? 'zh-CN' : undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return String(raw);
  }
}

function reload() {
  return refresh({ limit: props.limit });
}

onMounted(() => {
  void reload();
});

defineExpose({
  refresh: reload,
});
</script>

<style scoped>
.lora-train-history {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 0;
  height: 100%;
}

.lora-train-history__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-shrink: 0;
}

.lora-train-history__head-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
  width: 100%;
}

.lora-train-history__title {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  white-space: nowrap;
  flex-shrink: 0;
}

.lora-train-history--rail .lora-train-history__title {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.lora-train-history--page .lora-train-history__title {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  letter-spacing: normal;
  text-transform: none;
  color: var(--dq-label-primary);
}

.lora-train-history__empty-hint {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  text-align: center;
  padding: 24px 8px;
}

.lora-train-history--page .lora-train-history__empty-hint {
  padding: 32px 0;
}

.lora-train-history__skeleton {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lora-train-history__skeleton-row {
  height: 52px;
  border-radius: 10px;
  background: linear-gradient(
    90deg,
    var(--dq-fill-tertiary) 0%,
    var(--dq-fill-secondary) 50%,
    var(--dq-fill-tertiary) 100%
  );
  background-size: 200% 100%;
  animation: lora-train-history-shimmer 1.2s ease-in-out infinite;
}

@keyframes lora-train-history-shimmer {
  0% {
    background-position: 100% 0;
  }
  100% {
    background-position: -100% 0;
  }
}

.lora-train-history__list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.lora-train-history--page .lora-train-history__list {
  gap: 8px;
}

.lora-train-history__item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 10px 12px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  border-radius: 10px;
  background: color-mix(in srgb, var(--dq-surface-elevated) 60%, transparent);
  cursor: pointer;
  text-align: left;
  color: inherit;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.lora-train-history--page .lora-train-history__item {
  padding: 12px 14px;
  border-radius: 12px;
  background: var(--dq-fill-control);
}

.lora-train-history__item:hover {
  background: color-mix(in srgb, var(--dq-accent) 6%, var(--dq-surface-elevated));
  border-color: color-mix(in srgb, var(--dq-accent) 22%, var(--dq-border-subtle));
}

.lora-train-history--page .lora-train-history__item:hover {
  background: var(--dq-surface-inset-hover, var(--dq-fill-tertiary));
}

.lora-train-history__item.is-active {
  border-color: color-mix(in srgb, var(--dq-accent) 45%, transparent);
  background: color-mix(in srgb, var(--dq-accent) 8%, var(--dq-surface-elevated));
}

.lora-train-history__status-bar {
  flex-shrink: 0;
  width: 3px;
  align-self: stretch;
  border-radius: 999px;
  background: var(--dq-label-quaternary);
}

.lora-train-history__item.is-completed .lora-train-history__status-bar {
  background: var(--dq-success);
}

.lora-train-history__item.is-failed .lora-train-history__status-bar {
  background: var(--dq-danger);
}

.lora-train-history__item.is-running .lora-train-history__status-bar {
  background: var(--dq-accent);
}

.lora-train-history__item.is-queued .lora-train-history__status-bar {
  background: var(--dq-info, var(--dq-accent));
}

.lora-train-history__item-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 0;
}

.lora-train-history__item-top {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.lora-train-history--rail .lora-train-history__item-top {
  flex-wrap: nowrap;
}

.lora-train-history__item-name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.lora-train-history--rail .lora-train-history__item-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  word-break: normal;
}

.lora-train-history--rail .lora-train-history__item-top :deep(.dq-tag) {
  flex-shrink: 0;
  max-width: 100%;
  white-space: nowrap;
}

.lora-train-history__registered {
  flex-shrink: 0;
}

.lora-train-history__item-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-train-history--rail .lora-train-history__item-meta {
  flex-wrap: nowrap;
  min-width: 0;
}

.lora-train-history--rail .lora-train-history__base,
.lora-train-history--rail .lora-train-history__when {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.lora-train-history--rail .lora-train-history__base {
  flex-shrink: 1;
  min-width: 0;
}

.lora-train-history--rail .lora-train-history__when {
  flex-shrink: 0;
}

.lora-train-history__base {
  color: var(--dq-label-secondary);
}

.lora-train-history__chevron {
  flex-shrink: 0;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-quaternary);
  transition: transform 0.15s ease, color 0.15s ease;
}

.lora-train-history__item:hover .lora-train-history__chevron {
  color: var(--dq-accent);
  transform: translateX(2px);
}
</style>
