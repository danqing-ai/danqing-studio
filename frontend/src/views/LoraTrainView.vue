<template>
  <div class="copilot-page lora-train-page">
    <!-- Left: step navigation (models / assistant pattern) -->
    <div class="copilot-page__sidebar">
      <DqSurfaceCard class="copilot-page__sidebar-card studio-surface-card">
        <div class="card-title">
          <DqIcon><MagicStick /></DqIcon>
          {{ $t('loraTrain.title') }}
        </div>

        <nav class="copilot-page__nav" role="navigation" :aria-label="$t('loraTrain.title')">
          <button
            v-for="(item, i) in stepNavItems"
            :key="item.key"
            type="button"
            class="dq-download-menu__item"
            :class="{ 'is-active': step === i, 'is-done': i < step }"
            :disabled="i > maxReachableStep"
            @click="goToStep(i)"
          >
            <DqIcon class="dq-download-menu__icon"><component :is="item.icon" /></DqIcon>
            <span class="dq-download-menu__label">{{ item.label }}</span>
          </button>
        </nav>

        <div v-if="requirements" class="copilot-page__status-panel lora-train-page__status">
          <DqTag
            size="small"
            :type="requirements.can_submit ? 'success' : 'warning'"
            effect="plain"
          >
            {{ requirements.can_submit ? $t('loraTrain.memoryOk') : $t('loraTrain.memoryLow') }}
          </DqTag>
          <span class="copilot-page__status-model">
            {{ $t('loraTrain.memoryHint', {
              required: requirements.min_memory_gb,
              detected: requirements.detected_memory_gb || '?',
            }) }}
          </span>
        </div>
      </DqSurfaceCard>
    </div>

    <!-- Right: wizard workspace -->
    <div class="copilot-page__main">
      <header class="page-header copilot-page__page-header">
        <div class="copilot-page__header-main">
          <h2 class="page-title copilot-page__page-title">
            <DqIcon class="copilot-page__title-icon" aria-hidden="true">
              <component :is="currentStepIcon" />
            </DqIcon>
            {{ currentStepTitle }}
          </h2>
          <p class="copilot-page__page-desc">{{ currentStepDesc }}</p>
        </div>
        <div v-if="!activeRunId" class="copilot-page__header-actions lora-train-page__header-actions">
          <DqButton v-if="step > 0" size="small" @click="step -= 1">
            {{ $t('common.back') }}
          </DqButton>
          <DqButton
            v-if="step < 3"
            size="small"
            type="primary"
            :disabled="!canNext"
            @click="step += 1"
          >
            {{ $t('common.next') }}
          </DqButton>
          <DqButton
            v-else
            size="small"
            type="primary"
            :loading="submitting"
            :disabled="!canSubmit"
            @click="submitTraining"
          >
            {{ $t('loraTrain.startTraining') }}
          </DqButton>
        </div>
      </header>

      <LoraTrainRunDetail
        v-if="activeRunId"
        :task-id="activeRunId"
        @back="activeRunId = ''"
        @verify="onVerifyGenerate"
      />

      <DqSurfaceCard v-else class="copilot-page__workspace-card studio-surface-card lora-train-page__workspace">
        <!-- Step 1: base model -->
        <div v-show="step === 0" class="lora-train-page__panel">
          <DqRow :gutter="16" class="model-grid model-grid--fluid">
            <DqCol
              v-for="m in trainableModels"
              :key="m.id"
              :xs="24"
              :sm="12"
              :md="8"
              class="models-page__col-mb"
            >
              <button
                type="button"
                class="lora-train-page__model-pick"
                :class="{ 'is-selected': form.base_model === m.id }"
                :disabled="!m.trainable || !m.ready"
                :aria-pressed="form.base_model === m.id"
                @click="selectBase(m)"
              >
                <DqSurfaceCard
                  class="model-card"
                  :class="{
                    'model-ready': m.trainable && m.ready,
                    'is-disabled': !m.trainable || !m.ready,
                    'is-selected': form.base_model === m.id,
                  }"
                >
                  <div class="model-card-header">
                    <div class="model-icon">{{ modelInitials(m) }}</div>
                    <div class="lora-train-page__model-pick-badges">
                      <DqTag
                        v-if="form.base_model === m.id"
                        type="primary"
                        effect="dark"
                        size="small"
                        class="lora-train-page__selected-tag"
                      >
                        {{ $t('loraTrain.selectedBase') }}
                      </DqTag>
                      <div
                        class="model-status-dot"
                        :class="{
                          'is-ready': m.trainable && m.ready,
                          'is-missing': !m.trainable || !m.ready,
                        }"
                      />
                    </div>
                  </div>
                  <div class="model-card-content">
                    <div class="model-card-name">{{ modelDisplayName(m) }}</div>
                    <div class="model-card-meta">
                      <DqTag v-if="m.trainable && m.ready" type="success" effect="plain" size="small">
                        {{ $t('loraTrain.dreamboothReady') }}
                      </DqTag>
                      <DqTag v-else-if="!m.trainable" type="info" effect="plain" size="small">
                        {{ $t('loraTrain.phase2') }}
                      </DqTag>
                      <DqTag v-else type="warning" effect="plain" size="small">
                        {{ $t('loraTrain.notInstalled') }}
                      </DqTag>
                    </div>
                  </div>
                </DqSurfaceCard>
              </button>
            </DqCol>
          </DqRow>
        </div>

        <!-- Step 2: dataset -->
        <div v-show="step === 1" class="lora-train-page__panel">
          <div class="lora-train-page__toolbar">
            <DqSelect
              v-model="form.dataset_id"
              :placeholder="$t('loraTrain.selectDataset')"
              class="lora-train-page__dataset-select"
            >
              <DqOption
                v-for="d in datasets"
                :key="d.id"
                :label="`${d.name} (${d.image_count})`"
                :value="d.id"
              />
            </DqSelect>
            <DqButton
              size="sm"
              type="secondary"
              :disabled="!form.dataset_id"
              @click="deleteSelectedDataset"
            >
              {{ $t('loraTrain.deleteDataset') }}
            </DqButton>
            <div class="lora-train-page__toolbar-actions">
              <DqButton size="sm" @click="createEmptyDataset">{{ $t('loraTrain.newDataset') }}</DqButton>
              <DqButton size="sm" type="secondary" :loading="importingDog6" @click="importDog6">
                {{ $t('loraTrain.importDog6') }}
              </DqButton>
              <DqButton
                size="sm"
                type="secondary"
                :disabled="!form.dataset_id"
                @click="showGalleryImport = true"
              >
                {{ $t('loraTrain.importGallery') }}
              </DqButton>
              <DqButton
                size="sm"
                type="secondary"
                :disabled="!form.dataset_id || !selectedDataset?.images?.length"
                :loading="autoCaptioning"
                @click="runAutoCaption"
              >
                {{ $t('loraTrain.autoCaption') }}
              </DqButton>
              <input
                ref="uploadInputRef"
                type="file"
                multiple
                accept="image/*"
                hidden
                @change="onUploadImages"
              />
              <DqButton
                size="sm"
                type="secondary"
                :disabled="!form.dataset_id"
                @click="openUploadPicker"
              >
                {{ $t('loraTrain.uploadImages') }}
              </DqButton>
            </div>
          </div>

          <div v-if="selectedDataset?.images?.length" class="lora-train-page__image-grid">
            <div v-for="img in selectedDataset.images" :key="img.file" class="lora-train-page__image-cell">
              <img
                v-if="datasetImageUrl(img.file)"
                :src="datasetImageUrl(img.file)"
                :alt="img.file"
                loading="lazy"
              />
              <DqInput
                v-model="captionEdits[img.file]"
                size="xs"
                :placeholder="$t('loraTrain.captionPlaceholder')"
                @blur="saveCaptions"
              />
            </div>
          </div>
          <DqEmpty v-else :description="$t('loraTrain.noImages')" />

          <div class="lora-train-page__field">
            <label class="lora-train-page__label">{{ $t('loraTrain.defaultPrompt') }}</label>
            <DqInput v-model="form.default_prompt" @blur="patchDatasetMeta" />
          </div>
        </div>

        <!-- Step 3: config -->
        <div v-show="step === 2" class="lora-train-page__panel">
          <div class="lora-train-page__preset-row">
            <DqButton
              v-for="p in presetKeys"
              :key="p"
              size="sm"
              :type="form.preset === p ? 'primary' : 'secondary'"
              @click="form.preset = p"
            >
              {{ $t(`loraTrain.preset.${p}`) }}
            </DqButton>
          </div>
          <div class="lora-train-page__field">
            <label class="lora-train-page__label">{{ $t('loraTrain.progressPrompt') }}</label>
            <DqInput v-model="form.progress_prompt" type="textarea" :rows="3" />
          </div>
          <details class="lora-train-page__advanced">
            <summary>{{ $t('loraTrain.advanced') }}</summary>
            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.iterations') }}</label>
              <DqInput v-model.number="form.iterations" type="number" />
            </div>
            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.loraRank') }}</label>
              <DqInput v-model.number="form.lora_rank" type="number" />
            </div>
            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.learningRate') }}</label>
              <DqInput v-model.number="form.learning_rate" type="number" step="0.00001" />
            </div>
          </details>
        </div>

        <!-- Step 4: confirm -->
        <div v-show="step === 3" class="lora-train-page__panel">
          <ul class="lora-train-page__summary">
            <li><span>{{ $t('loraTrain.summaryBase') }}</span><strong>{{ form.base_model }}</strong></li>
            <li>
              <span>{{ $t('loraTrain.summaryDataset') }}</span>
              <strong>{{ selectedDataset?.name }} ({{ selectedDataset?.image_count }})</strong>
            </li>
            <li><span>{{ $t('loraTrain.summaryPreset') }}</span><strong>{{ form.preset }}</strong></li>
            <li><span>{{ $t('loraTrain.summaryPrompt') }}</span><strong>{{ form.progress_prompt }}</strong></li>
          </ul>
        </div>
      </DqSurfaceCard>
    </div>

    <DqDialog
      v-model:open="showGalleryImport"
      :title="$t('loraTrain.importGalleryTitle')"
      width="min(560px, 92vw)"
      destroy-on-close
    >
      <p class="lora-train-page__import-hint">{{ $t('loraTrain.importGalleryHint') }}</p>
      <AssetPicker accept-kind="image" @pick="onGalleryPick" />
      <div v-if="pendingGalleryAssets.length" class="lora-train-page__pending-tags">
        <DqTag v-for="a in pendingGalleryAssets" :key="a" size="sm">{{ a }}</DqTag>
      </div>
      <template #footer>
        <DqButton size="sm" @click="showGalleryImport = false">{{ $t('common.cancel') }}</DqButton>
        <DqButton
          size="sm"
          type="primary"
          :disabled="!pendingGalleryAssets.length || !form.dataset_id"
          :loading="importingGallery"
          @click="confirmGalleryImport"
        >
          {{ $t('loraTrain.importGalleryConfirm', { count: pendingGalleryAssets.length }) }}
        </DqButton>
      </template>
    </DqDialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { FolderChecked, MagicStick, PictureFilled, Setting } from '@danqing/dq-shell';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { toast, confirm } from '@/utils/feedback';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import LoraTrainRunDetail from '@/components/lora/LoraTrainRunDetail.vue';
import AssetPicker from '@/components/asset/AssetPicker.vue';

const { t } = useI18n();
const router = useRouter();
const route = useRoute();

const step = ref(0);
const submitting = ref(false);
const importingDog6 = ref(false);
const importingGallery = ref(false);
const autoCaptioning = ref(false);
const showGalleryImport = ref(false);
const pendingGalleryAssets = ref<string[]>([]);
const activeRunId = ref('');
const trainableModels = ref<any[]>([]);
const datasets = ref<any[]>([]);
const requirements = ref<any>(null);
const captionEdits = reactive<Record<string, string>>({});
const uploadInputRef = ref<HTMLInputElement | null>(null);
const presetKeys = ['quick', 'standard', 'quality', 'custom'];

const stepNavItems = computed(() => [
  { key: 'base', label: t('loraTrain.stepBase'), icon: MagicStick },
  { key: 'dataset', label: t('loraTrain.stepDataset'), icon: PictureFilled },
  { key: 'config', label: t('loraTrain.stepConfig'), icon: Setting },
  { key: 'confirm', label: t('loraTrain.stepConfirm'), icon: FolderChecked },
]);

const stepDescKeys = ['stepBaseDesc', 'stepDatasetDesc', 'stepConfigDesc', 'stepConfirmDesc'];

const currentStepTitle = computed(() => stepNavItems.value[step.value]?.label || '');
const currentStepDesc = computed(() => t(`loraTrain.${stepDescKeys[step.value] || 'stepBaseDesc'}`));
const currentStepIcon = computed(() => stepNavItems.value[step.value]?.icon || MagicStick);

const form = reactive({
  base_model: 'flux1-dev',
  dataset_id: '',
  preset: 'standard' as string,
  progress_prompt: '',
  default_prompt: 'A photo of sks dog',
  output_name: '',
  iterations: null as number | null,
  lora_rank: null as number | null,
  learning_rate: null as number | null,
});

const selectedDataset = computed(() =>
  datasets.value.find((d) => d.id === form.dataset_id)
);

const maxReachableStep = computed(() => {
  let max = 0;
  if (form.base_model) max = 1;
  if ((selectedDataset.value?.image_count || 0) >= 3) max = Math.max(max, 2);
  if (form.progress_prompt.trim()) max = Math.max(max, 3);
  return max;
});

const canNext = computed(() => {
  if (step.value === 0) return !!form.base_model;
  if (step.value === 1) return (selectedDataset.value?.image_count || 0) >= 3;
  if (step.value === 2) return !!form.progress_prompt.trim();
  return true;
});

const canSubmit = computed(
  () => canNext.value && requirements.value?.can_submit !== false
);

function goToStep(i: number) {
  if (i <= maxReachableStep.value) step.value = i;
}

function modelDisplayName(m: any): string {
  const n = m.name;
  if (n && typeof n === 'object') return n.en || n.zh || m.id;
  return m.id;
}

function modelInitials(m: any): string {
  const name = modelDisplayName(m);
  const parts = name.replace(/[\[\]()]/g, ' ').split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function apiErrorMessage(e: unknown): string {
  const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
  const detail = err?.response?.data?.detail;
  if (detail && typeof detail === 'object' && detail.message) return detail.message;
  if (typeof detail === 'string') return detail;
  return err?.message || String(e);
}

async function refreshDatasets(selectId?: string) {
  const dsRes = await api.loras.listDatasets();
  datasets.value = (dsRes as any).items || [];
  if (selectId) form.dataset_id = selectId;
}

function openUploadPicker() {
  if (!form.dataset_id) {
    toast.error(t('loraTrain.selectDatasetFirst'));
    return;
  }
  uploadInputRef.value?.click();
}

function selectBase(m: any) {
  if (!m.trainable || !m.ready) return;
  form.base_model = m.id;
}

async function loadMeta() {
  const [modelsRes, dsRes, reqRes] = await Promise.all([
    api.loras.trainableModels(),
    api.loras.listDatasets(),
    api.loras.trainingRequirements(),
  ]);
  trainableModels.value = (modelsRes as any).items || [];
  datasets.value = (dsRes as any).items || [];
  requirements.value = reqRes;
}

watch(
  () => form.dataset_id,
  async (id) => {
    if (!id) return;
    try {
      const ds = await api.loras.getDataset(id);
      Object.keys(captionEdits).forEach((k) => delete captionEdits[k]);
      for (const img of (ds as any).images || []) {
        captionEdits[img.file] = img.prompt || '';
      }
      datasets.value = datasets.value.map((d) => (d.id === id ? { ...d, ...ds } : d));
    } catch (e: unknown) {
      toast.error(apiErrorMessage(e));
    }
  }
);

function datasetImageUrl(file: string): string {
  if (!form.dataset_id || !file) return '';
  return api.loras.datasetImageUrl(form.dataset_id, file);
}

async function deleteSelectedDataset() {
  const id = form.dataset_id;
  if (!id) return;
  const name = selectedDataset.value?.name || id;
  const ok = await confirm({
    title: t('loraTrain.deleteDataset'),
    message: t('loraTrain.deleteDatasetConfirm', { name }),
    type: 'warning',
  });
  if (!ok) return;
  try {
    await api.loras.deleteDataset(id);
    if (form.dataset_id === id) form.dataset_id = '';
    Object.keys(captionEdits).forEach((k) => delete captionEdits[k]);
    await refreshDatasets();
    toast.success(t('loraTrain.deleteDatasetDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function createEmptyDataset() {
  const name = window.prompt(t('loraTrain.newDatasetName')) || t('loraTrain.newDatasetDefault');
  if (!name.trim()) return;
  try {
    const ds = await api.loras.createDataset({ name: name.trim(), default_prompt: form.default_prompt });
    await refreshDatasets((ds as any).id);
    toast.success(t('loraTrain.datasetCreated'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function importDog6() {
  importingDog6.value = true;
  try {
    const ds = await api.loras.importDog6();
    await refreshDatasets((ds as any).id);
    toast.success(t('loraTrain.dog6Imported'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    importingDog6.value = false;
  }
}

function onGalleryPick(payload: { path: string }) {
  const p = payload.path?.trim();
  if (!p || pendingGalleryAssets.value.includes(p)) return;
  pendingGalleryAssets.value.push(p);
}

async function confirmGalleryImport() {
  if (!form.dataset_id || !pendingGalleryAssets.value.length) return;
  importingGallery.value = true;
  try {
    const ds = await api.loras.importAssets(
      form.dataset_id,
      pendingGalleryAssets.value,
      form.default_prompt
    );
    datasets.value = datasets.value.map((d) =>
      d.id === form.dataset_id ? { ...d, ...(ds as object) } : d
    );
    pendingGalleryAssets.value = [];
    showGalleryImport.value = false;
    toast.success(t('loraTrain.galleryImported'));
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    importingGallery.value = false;
  }
}

async function runAutoCaption() {
  if (!form.dataset_id) return;
  autoCaptioning.value = true;
  try {
    const ds = await api.loras.autoCaption(form.dataset_id);
    for (const img of (ds as any).images || []) {
      captionEdits[img.file] = img.prompt || '';
    }
    datasets.value = datasets.value.map((d) =>
      d.id === form.dataset_id ? { ...d, ...(ds as object) } : d
    );
    toast.success(t('loraTrain.autoCaptionDone'));
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    autoCaptioning.value = false;
  }
}

async function importFromRouteQuery() {
  const raw = route.query.import;
  const ids = (Array.isArray(raw) ? raw.join(',') : String(raw || ''))
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (!ids.length) return;
  if (!form.dataset_id) {
    const ds = await api.loras.createDataset({
      name: t('loraTrain.canvasImportDataset'),
      default_prompt: form.default_prompt,
    });
    datasets.value.unshift(ds);
    form.dataset_id = (ds as any).id;
  }
  try {
    const ds = await api.loras.importAssets(form.dataset_id, ids, form.default_prompt);
    datasets.value = datasets.value.map((d) =>
      d.id === form.dataset_id ? { ...d, ...(ds as object) } : d
    );
    step.value = 1;
    toast.success(t('loraTrain.galleryImported'));
    const q = { ...route.query };
    delete q.import;
    router.replace({ query: q });
  } catch (e: any) {
    toast.error(e?.message || String(e));
  }
}

async function onUploadImages(ev: Event) {
  const input = ev.target as HTMLInputElement;
  const files = input.files;
  if (!files?.length || !form.dataset_id) return;
  try {
    await api.loras.uploadImages(form.dataset_id, Array.from(files), form.default_prompt);
    const ds = (await api.loras.getDataset(form.dataset_id)) as any;
    datasets.value = datasets.value.map((d) => (d.id === form.dataset_id ? { ...d, ...ds } : d));
    for (const img of ds.images || []) {
      captionEdits[img.file] = img.prompt || form.default_prompt || '';
    }
    toast.success(t('loraTrain.uploadDone', { count: files.length }));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    input.value = '';
  }
}

async function saveCaptions() {
  if (!form.dataset_id) return;
  const captions = Object.entries(captionEdits).map(([file, prompt]) => ({ file, prompt }));
  try {
    await api.loras.updateCaptions(form.dataset_id, captions);
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function patchDatasetMeta() {
  if (!form.dataset_id) return;
  await api.loras.patchDataset(form.dataset_id, {
    name: selectedDataset.value?.name || '',
    default_prompt: form.default_prompt,
  });
}

async function submitTraining() {
  submitting.value = true;
  try {
    const body: Record<string, unknown> = {
      base_model: form.base_model,
      dataset_id: form.dataset_id,
      progress_prompt: form.progress_prompt,
      preset: form.preset,
      output_name: form.output_name,
      auto_register: true,
    };
    if (form.preset === 'custom') {
      if (form.iterations) body.iterations = form.iterations;
      if (form.lora_rank) body.lora_rank = form.lora_rank;
      if (form.learning_rate) body.learning_rate = form.learning_rate;
    }
    const res = await api.loras.submitTraining(body);
    const tid = taskIdFromSubmitResponse(res);
    toast.success(t('loraTrain.submitted'));
    openGlobalTaskQueue();
    if (tid) activeRunId.value = tid;
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    submitting.value = false;
  }
}

function onVerifyGenerate(payload: { prompt: string; loraId: string; baseModel: string }) {
  router.push({
    name: 'image_create',
    query: {
      model: payload.baseModel,
      prompt: payload.prompt,
      lora: payload.loraId,
    },
  });
}

onMounted(async () => {
  try {
    await loadMeta();
    await importFromRouteQuery();
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
});
</script>

<style scoped>
.lora-train-page__workspace :deep(.dq-surface-card__body) {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.lora-train-page__panel {
  flex: 1;
  min-height: 0;
}

.lora-train-page__header-actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.lora-train-page__status {
  margin-top: auto;
}

.lora-train-page__model-pick {
  display: block;
  width: 100%;
  padding: 0;
  border: none;
  background: transparent;
  cursor: pointer;
  text-align: inherit;
  border-radius: var(--radius-md);
  overflow: hidden;
  -webkit-backface-visibility: hidden;
  backface-visibility: hidden;
}

.lora-train-page__model-pick:disabled {
  cursor: not-allowed;
}

.lora-train-page__model-pick.is-selected {
  outline: 2px solid var(--dq-accent);
  outline-offset: 2px;
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--dq-accent) 22%, transparent);
}

.lora-train-page__model-pick :deep(.model-card) {
  border-radius: inherit;
  transition:
    border-color 0.15s ease,
    box-shadow 0.15s ease,
    background 0.15s ease;
}

.lora-train-page__model-pick-badges {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-left: auto;
}

.lora-train-page__selected-tag {
  font-weight: 600;
  letter-spacing: 0.02em;
}

.lora-train-page__model-pick.is-selected :deep(.model-card) {
  border-color: var(--dq-accent);
  box-shadow:
    inset 0 0 0 1px color-mix(in srgb, var(--dq-accent) 35%, transparent),
    0 12px 32px var(--dq-shadow-md);
}

.lora-train-page__model-pick.is-selected :deep(.model-card.model-ready) {
  border-color: var(--dq-accent);
}

.lora-train-page__model-pick.is-selected :deep(.model-card-header) {
  background: color-mix(in srgb, var(--dq-accent) 16%, var(--dq-surface-inset)) !important;
}

.lora-train-page__model-pick :deep(.model-card.is-disabled) {
  opacity: 0.62;
}

.lora-train-page__toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: flex-start;
  margin-bottom: 16px;
}

.lora-train-page__dataset-select {
  min-width: min(280px, 100%);
}

.lora-train-page__toolbar-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  flex: 1;
}

.lora-train-page__dataset-select :deep(.dq-select__content) {
  min-width: var(--reka-select-trigger-width, 280px);
}

.lora-train-page__dataset-select :deep(.dq-select__viewport) {
  height: auto !important;
  max-height: 240px !important;
  flex: none !important;
}

.lora-train-page__image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.lora-train-page__image-cell img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--radius-md);
  border: 0.5px solid var(--dq-border);
  background: var(--dq-bg-base);
}

.lora-train-page__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-train-page__label {
  font-size: 11px;
  font-weight: 500;
  color: var(--dq-label-secondary);
}

.lora-train-page__preset-row {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.lora-train-page__advanced {
  margin-top: 8px;
  font-size: 13px;
  color: var(--dq-label-secondary);
}

.lora-train-page__advanced summary {
  cursor: pointer;
  margin-bottom: 12px;
}

.lora-train-page__summary {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.lora-train-page__summary li {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  background: var(--dq-fill-secondary);
  border: 0.5px solid var(--dq-border-subtle);
  font-size: 13px;
}

.lora-train-page__summary span {
  color: var(--dq-label-tertiary);
}

.lora-train-page__summary strong {
  color: var(--dq-label-primary);
  font-weight: 600;
  text-align: right;
  word-break: break-word;
}

.lora-train-page__import-hint {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--dq-label-tertiary);
}

.lora-train-page__pending-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}
</style>
