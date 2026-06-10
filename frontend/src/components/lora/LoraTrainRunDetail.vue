<template>
  <DqSurfaceCard class="copilot-page__workspace-card studio-surface-card lora-run-detail">
    <div class="lora-run-detail__head">
      <DqButton size="sm" @click="$emit('back')">{{ $t('common.back') }}</DqButton>
      <div class="lora-run-detail__head-main">
        <h3 class="lora-run-detail__title">{{ $t('loraTrain.runTitle') }}</h3>
        <TaskIdBadge :task-id="taskId" />
      </div>
    </div>

    <div v-if="task" class="lora-run-detail__status">
      <DqTag :type="statusTagType">{{ task.status }}</DqTag>
      <DqProgress :percentage="Math.round((task.progress || 0) * 100)" />
      <p v-if="task.step != null" class="lora-run-detail__step-hint">
        {{ task.step }} / {{ task.total }}
      </p>
    </div>

    <div v-if="artifacts?.loss_history?.length" class="lora-run-detail__section">
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.lossCurve') }}</h4>
      <div class="lora-run-loss-bars">
        <div
          v-for="pt in artifacts.loss_history"
          :key="pt.step"
          class="lora-run-loss-bar"
          :style="{ height: `${Math.min(100, pt.loss * 200)}px` }"
          :title="`step ${pt.step}: ${pt.loss}`"
        />
      </div>
    </div>

    <div v-if="artifacts?.progress_images?.length" class="lora-run-detail__section">
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.progressGallery') }}</h4>
      <div class="lora-run-gallery-grid">
        <a
          v-for="name in artifacts.progress_images"
          :key="name"
          :href="artifactUrl(name)"
          target="_blank"
          rel="noopener"
        >
          <img :src="artifactUrl(name)" :alt="name" loading="lazy" />
        </a>
      </div>
    </div>

    <div
      v-if="artifacts?.checkpoints?.length && task?.status === 'completed'"
      class="lora-run-detail__section"
    >
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.checkpoints') }}</h4>
      <div class="lora-run-checkpoint-row">
        <DqSelect v-model="selectedCheckpoint" class="lora-run-checkpoint-select">
          <DqOption
            v-for="ck in artifacts.checkpoints"
            :key="ck"
            :label="ck"
            :value="ck"
          />
        </DqSelect>
        <DqInput v-model="registerName" :placeholder="$t('loraTrain.registerName')" size="sm" />
        <DqButton size="sm" type="secondary" :loading="registering" @click="registerSelected">
          {{ $t('loraTrain.registerCheckpoint') }}
        </DqButton>
      </div>
      <p v-if="registeredLoraId" class="lora-run-registered">
        {{ $t('loraTrain.registeredAs') }}: {{ registeredLoraId }}
      </p>
    </div>

    <div v-if="task?.status === 'completed'" class="lora-run-detail__actions">
      <DqButton type="primary" size="sm" @click="verifyGenerate">
        {{ $t('loraTrain.verifyGenerate') }}
      </DqButton>
    </div>
  </DqSurfaceCard>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { api } from '@/utils/api';
import { toast } from '@/utils/feedback';
import TaskIdBadge from '@/components/studio/TaskIdBadge.vue';

const props = defineProps<{ taskId: string }>();
const emit = defineEmits<{
  (e: 'back'): void;
  (e: 'verify', payload: { prompt: string; loraId: string; baseModel: string }): void;
}>();

const { t } = useI18n();
const task = ref<any>(null);
const artifacts = ref<any>(null);
const selectedCheckpoint = ref('');
const registerName = ref('');
const registering = ref(false);
const registeredLoraId = ref('');
let pollTimer: ReturnType<typeof setInterval> | null = null;

const statusTagType = computed(() => {
  const s = task.value?.status;
  if (s === 'completed') return 'success';
  if (s === 'failed') return 'danger';
  if (s === 'running') return 'primary';
  return 'info';
});

function artifactUrl(name: string): string {
  return api.loras.artifactFileUrl(props.taskId, name);
}

async function refresh() {
  task.value = await api.gen.getMediaTask(props.taskId);
  try {
    artifacts.value = await api.loras.trainingArtifacts(props.taskId);
    if (!selectedCheckpoint.value && artifacts.value?.checkpoints?.length) {
      const finals = artifacts.value.checkpoints.filter((c: string) => c.includes('final'));
      selectedCheckpoint.value = finals[finals.length - 1] || artifacts.value.checkpoints.at(-1);
    }
  } catch {
    artifacts.value = null;
  }
}

watch(
  () => task.value?.result?.metadata?.user_lora_id,
  (id) => {
    if (id && !registeredLoraId.value) registeredLoraId.value = String(id);
  },
  { immediate: true }
);

function verifyLoraId(): string {
  return (
    registeredLoraId.value ||
    String(task.value?.result?.metadata?.user_lora_id || task.value?.result?.metadata?.training?.user_lora_id || '')
  );
}

function verifyGenerate() {
  const meta = task.value?.result?.metadata?.training || {};
  const params = task.value?.params || {};
  emit('verify', {
    prompt: String(params.progress_prompt || meta.progress_prompt || ''),
    loraId: verifyLoraId(),
    baseModel: String(params.base_model || 'flux1-dev'),
  });
}

async function registerSelected() {
  if (!selectedCheckpoint.value) return;
  registering.value = true;
  try {
    const res = (await api.loras.registerCheckpoint(props.taskId, {
      checkpoint: selectedCheckpoint.value,
      name: registerName.value.trim(),
    })) as { user_lora?: { id?: string } };
    registeredLoraId.value = String(res.user_lora?.id || '');
    toast.success(t('loraTrain.registerSuccess'));
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    registering.value = false;
  }
}

onMounted(() => {
  refresh();
  pollTimer = setInterval(refresh, 4000);
});

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer);
});
</script>

<style scoped>
.lora-run-detail :deep(.dq-surface-card__body) {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.lora-run-detail__head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.lora-run-detail__head-main {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.lora-run-detail__title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-run-detail__status {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lora-run-detail__step-hint {
  margin: 0;
  font-size: 12px;
  color: var(--dq-label-tertiary);
}

.lora-run-detail__section-title {
  margin: 0 0 10px;
  font-size: 13px;
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-run-gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
}

.lora-run-gallery-grid img {
  width: 100%;
  border-radius: var(--radius-md);
  border: 0.5px solid var(--dq-border);
}

.lora-run-loss-bars {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 80px;
  padding: 8px;
  border-radius: var(--radius-md);
  background: var(--dq-fill-secondary);
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-run-loss-bar {
  flex: 1;
  min-width: 4px;
  background: var(--dq-accent);
  opacity: 0.75;
  border-radius: 2px 2px 0 0;
}

.lora-run-checkpoint-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.lora-run-checkpoint-select {
  min-width: min(280px, 100%);
}

.lora-run-registered {
  margin: 8px 0 0;
  font-size: 12px;
  color: var(--dq-label-tertiary);
}

.lora-run-detail__actions {
  padding-top: 4px;
  border-top: 0.5px solid var(--dq-border-subtle);
}
</style>
