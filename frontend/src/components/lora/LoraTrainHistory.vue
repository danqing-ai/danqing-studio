<template>
  <div v-if="runs.length" class="lora-train-history">
    <div class="lora-train-history__head">
      <span class="lora-train-history__title">{{ $t('loraTrain.recentRuns') }}</span>
      <DqButton size="xs" type="text" :loading="loading" @click="loadRuns">
        {{ $t('loraTrain.refreshHistory') }}
      </DqButton>
    </div>
    <div class="lora-train-history__list">
      <button
        v-for="run in runs"
        :key="run.id"
        type="button"
        class="lora-train-history__item"
        :class="{ 'is-active': run.id === activeId }"
        @click="emit('select', run.id)"
      >
        <DqTag size="small" :type="statusType(run.status)" effect="plain">
          {{ statusLabel(run.status) }}
        </DqTag>
        <span class="lora-train-history__item-model">{{ runModel(run) }}</span>
        <span class="lora-train-history__item-meta">{{ formatWhen(run) }}</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { api } from '@/utils/api';

defineProps<{ activeId?: string }>();
const emit = defineEmits<{ (e: 'select', taskId: string): void }>();

const { t, locale } = useI18n();
const runs = ref<Array<Record<string, any>>>([]);
const loading = ref(false);

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

function runModel(run: Record<string, any>): string {
  const params = run.params || {};
  return String(params.base_model || run.model || '—').split(':', 1)[0];
}

function formatWhen(run: Record<string, any>): string {
  const raw = run.created_at || run.started_at || run.updated_at;
  if (!raw) return '';
  try {
    return new Date(raw).toLocaleString(locale.value === 'zh' ? 'zh-CN' : undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return String(raw);
  }
}

async function loadRuns() {
  loading.value = true;
  try {
    const res = (await api.gen.listMediaTasks({
      kind: 'lora.training',
      limit: 8,
    })) as { tasks?: Array<Record<string, any>> };
    runs.value = res.tasks || [];
  } catch {
    runs.value = [];
  } finally {
    loading.value = false;
  }
}

onMounted(() => {
  void loadRuns();
});

defineExpose({ loadRuns });
</script>

<style scoped>
.lora-train-history {
  margin-top: 6px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
}

.lora-train-history__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.lora-train-history__title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.lora-train-history__list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-height: 180px;
  overflow-y: auto;
}

.lora-train-history__item {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 7px 8px;
  border: 0.5px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.lora-train-history__item:hover {
  background: var(--dq-fill-tertiary);
}

.lora-train-history__item.is-active {
  background: color-mix(in srgb, var(--dq-accent) 10%, var(--dq-fill-secondary));
  border-color: color-mix(in srgb, var(--dq-accent) 30%, transparent);
}

.lora-train-history__item-model {
  font-size: 12px;
  font-weight: 500;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.lora-train-history__item-meta {
  font-size: 10px;
  color: var(--dq-label-tertiary);
}
</style>
