<template>
  <div class="lora-train-page">
    <aside class="lora-train-page__rail">
      <div class="lora-train-page__rail-head">
        <div class="lora-train-page__rail-icon" aria-hidden="true">
          <DqIcon><MagicStick /></DqIcon>
        </div>
        <div>
          <h1 class="lora-train-page__rail-title">{{ $t('loraTrain.title') }}</h1>
          <p class="lora-train-page__rail-sub">{{ $t('loraTrain.railSubtitle') }}</p>
        </div>
      </div>

      <nav class="lora-train-page__steps" role="navigation" :aria-label="$t('loraTrain.title')">
        <button
          v-for="(item, i) in stepNavItems"
          :key="item.key"
          type="button"
          class="lora-train-page__step"
          :class="{
            'is-active': step === i && !activeRunId,
            'is-done': stepDone(i),
            'is-locked': i > maxReachableStep,
          }"
          :disabled="i > maxReachableStep || !!activeRunId"
          @click="goToStep(i)"
        >
          <span class="lora-train-page__step-rail" aria-hidden="true">
            <span class="lora-train-page__step-dot">
              <span v-if="stepDone(i)">✓</span>
              <span v-else>{{ i + 1 }}</span>
            </span>
            <span v-if="i < stepNavItems.length - 1" class="lora-train-page__step-line" />
          </span>
          <span class="lora-train-page__step-body">
            <span class="lora-train-page__step-label">{{ item.label }}</span>
          </span>
        </button>
      </nav>

      <div v-if="wizardSummary.length && !activeRunId" class="lora-train-page__chip-panel">
        <div class="lora-train-page__chip-title">{{ $t('loraTrain.wizardSummary') }}</div>
        <div class="lora-train-page__chips">
          <span v-for="row in wizardSummary" :key="row.key" class="lora-train-page__chip">
            {{ row.label }} · {{ row.value }}
          </span>
        </div>
      </div>

      <details v-if="!activeRunId" class="lora-train-page__history-fold">
        <summary>{{ $t('loraTrain.recentRuns') }}</summary>
        <LoraTrainHistory
          ref="historyRef"
          :active-id="activeRunId"
          show-models-link
          @select="openRun"
          @open-models="openModelsUserLorasPage"
        />
      </details>

      <div v-if="requirements" class="lora-train-page__mem">
        <span
          class="lora-train-page__mem-dot"
          :class="requirements.can_submit ? 'is-ok' : 'is-warn'"
          aria-hidden="true"
        />
        <div class="lora-train-page__mem-text">
          <strong>{{ requirements.can_submit ? $t('loraTrain.memoryOk') : $t('loraTrain.memoryLow') }}</strong>
          <span>{{
            $t('loraTrain.memoryHint', {
              required: requirements.min_memory_gb,
              detected: requirements.detected_memory_gb || '?',
            })
          }}</span>
        </div>
      </div>
    </aside>

    <section class="lora-train-page__stage">
      <LoraTrainRunDetail
        v-if="activeRunId"
        :task-id="activeRunId"
        @back="onRunBack"
        @verify="onVerifyGenerate"
      />

      <template v-else>
        <div class="lora-train-page__progress" aria-hidden="true">
          <span
            v-for="i in 4"
            :key="i"
            class="lora-train-page__progress-seg"
            :class="{ 'is-done': step >= i - 1, 'is-current': step === i - 1 }"
          />
        </div>

        <header class="lora-train-page__stage-head">
          <div class="lora-train-page__stage-head-main">
            <DqIcon class="lora-train-page__stage-icon" aria-hidden="true">
              <component :is="currentStepIcon" />
            </DqIcon>
            <div>
              <h2 class="lora-train-page__stage-title">{{ currentStepTitle }}</h2>
              <p class="lora-train-page__stage-desc">{{ currentStepDesc }}</p>
            </div>
          </div>
        </header>

        <div class="lora-train-page__surface">
          <!-- Step 1: base model -->
          <div v-show="step === 0" class="lora-train-page__panel">
            <div class="lora-train-page__model-grid">
              <button
                v-for="m in trainableModels"
                :key="m.id"
                type="button"
                class="lora-model-tile"
                :class="{
                  'is-selected': form.base_model === m.id,
                  'is-ready': m.trainable && m.ready,
                  'is-disabled': !m.trainable || !m.ready,
                }"
                :disabled="!m.trainable || !m.ready"
                :aria-pressed="form.base_model === m.id"
                @click="selectBase(m)"
              >
                <span class="lora-model-tile__avatar">{{ modelInitials(m) }}</span>
                <span class="lora-model-tile__body">
                  <span class="lora-model-tile__name">{{ modelDisplayName(m) }}</span>
                  <span class="lora-model-tile__meta">
                    <template v-if="m.trainable && m.ready">{{ $t('loraTrain.dreamboothReady') }}</template>
                    <template v-else-if="!m.trainable">{{ $t('loraTrain.phase2') }}</template>
                    <template v-else>{{ $t('loraTrain.notInstalled') }}</template>
                  </span>
                </span>
                <span class="lora-model-tile__check" aria-hidden="true">
                  <span v-if="form.base_model === m.id">✓</span>
                </span>
              </button>
            </div>
          </div>

          <!-- Step 2: dataset -->
          <div v-show="step === 1" class="lora-train-page__panel lora-train-page__panel--flush">
            <LoraDatasetPanel
              ref="datasetPanelRef"
              :selected-id="form.dataset_id"
              :datasets="datasets"
              :default-prompt="form.default_prompt"
              :caption-edits="captionEdits"
              @update:selected-id="form.dataset_id = $event"
              @update:default-prompt="form.default_prompt = $event"
              @update:caption-edits="onCaptionEditsUpdate"
              @datasets-changed="datasets = $event"
            />
          </div>

          <!-- Step 3: config -->
          <div v-show="step === 2" class="lora-train-page__panel">
            <div class="lora-train-page__preset-grid">
              <button
                v-for="p in presetKeys"
                :key="p"
                type="button"
                class="lora-train-page__preset-card"
                :class="{ 'is-selected': form.preset === p }"
                @click="form.preset = p"
              >
                <span class="lora-train-page__preset-name">{{ $t(`loraTrain.preset.${p}`) }}</span>
                <span class="lora-train-page__preset-desc">{{ $t(`loraTrain.presetDesc.${p}`) }}</span>
                <span v-if="p !== 'custom' && presetStats(p)" class="lora-train-page__preset-stats">
                  {{ presetStats(p) }}
                </span>
              </button>
            </div>
            <div v-if="activePresetDetail" class="lora-train-page__preset-detail">
              <span class="lora-train-page__preset-detail-label">{{ $t('loraTrain.presetDetail') }}</span>
              <div class="lora-train-page__preset-detail-grid">
                <span>{{ $t('loraTrain.iterations') }}: {{ activePresetDetail.iterations }}</span>
                <span>{{ $t('loraTrain.loraRank') }}: {{ activePresetDetail.lora_rank }}</span>
                <span v-if="presetResolution(activePresetDetail)">
                  {{ $t('loraTrain.resolution') }}: {{ presetResolution(activePresetDetail) }}
                  <em class="lora-train-page__preset-crop-hint">({{ $t('loraTrain.resolutionAutoCrop') }})</em>
                </span>
                <span v-if="activePresetDetail.learning_rate">
                  {{ $t('loraTrain.learningRate') }}: {{ activePresetDetail.learning_rate }}
                </span>
              </div>
            </div>
            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.progressPrompt') }}</label>
              <DqInput v-model="form.progress_prompt" type="textarea" :rows="3" />
              <p class="lora-train-page__field-hint">{{ $t('loraTrain.progressPromptDesc') }}</p>
            </div>
            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.outputName') }}</label>
              <DqInput v-model="form.output_name" :placeholder="$t('loraTrain.outputNamePlaceholder')" />
              <p class="lora-train-page__field-hint">{{ $t('loraTrain.outputNameDesc') }}</p>
            </div>
            <details v-if="form.preset === 'custom'" class="lora-train-page__advanced">
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
            <p class="lora-train-page__confirm-intro">{{ $t('loraTrain.confirmIntro') }}</p>
            <div v-if="confirmPreviewImages.length" class="lora-train-page__confirm-preview">
              <span class="lora-train-page__confirm-preview-label">{{ $t('loraTrain.confirmPreview') }}</span>
              <div class="lora-train-page__confirm-preview-grid">
                <img
                  v-for="img in confirmPreviewImages"
                  :key="img.file"
                  :src="confirmImageUrl(img.file)"
                  :alt="img.file"
                  loading="lazy"
                />
                <span
                  v-if="(selectedDataset?.image_count || 0) > confirmPreviewImages.length"
                  class="lora-train-page__confirm-preview-more"
                >
                  +{{ (selectedDataset?.image_count || 0) - confirmPreviewImages.length }}
                </span>
              </div>
            </div>
            <ul class="lora-train-page__summary">
              <li><span>{{ $t('loraTrain.summaryBase') }}</span><strong>{{ form.base_model }}</strong></li>
              <li>
                <span>{{ $t('loraTrain.summaryDataset') }}</span>
                <strong>{{ selectedDataset?.name }} ({{ selectedDataset?.image_count }})</strong>
              </li>
              <li><span>{{ $t('loraTrain.summaryPreset') }}</span><strong>{{ $t(`loraTrain.preset.${form.preset}`) }}</strong></li>
              <li><span>{{ $t('loraTrain.summaryPrompt') }}</span><strong>{{ form.progress_prompt }}</strong></li>
              <li v-if="form.output_name.trim()">
                <span>{{ $t('loraTrain.summaryOutput') }}</span><strong>{{ form.output_name }}</strong>
              </li>
            </ul>
          </div>
        </div>

        <footer class="lora-train-page__footer">
          <div class="lora-train-page__footer-left">
            <DqButton v-if="step > 0" size="small" @click="step -= 1">
              {{ $t('common.back') }}
            </DqButton>
          </div>
          <span class="lora-train-page__footer-step">{{ step + 1 }} / 4</span>
          <div class="lora-train-page__footer-right">
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
        </footer>
      </template>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { FolderChecked, MagicStick, PictureFilled, Setting } from '@danqing/dq-shell';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { toast } from '@/utils/feedback';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import { getItem, setItem, DQ_STORAGE } from '@/utils/storage';
import LoraTrainRunDetail from '@/components/lora/LoraTrainRunDetail.vue';
import LoraDatasetPanel from '@/components/lora/LoraDatasetPanel.vue';
import LoraTrainHistory from '@/components/lora/LoraTrainHistory.vue';
import { openModelsUserLoras } from '@/utils/loraTrainHandoff';

const { t } = useI18n();
const router = useRouter();
const route = useRoute();

const step = ref(0);
const submitting = ref(false);
const activeRunId = ref('');
const trainableModels = ref<any[]>([]);
const datasets = ref<any[]>([]);
const requirements = ref<any>(null);
const presetsByModel = ref<Record<string, Record<string, Record<string, unknown>>>>({});
const captionEdits = reactive<Record<string, string>>({});
const datasetPanelRef = ref<InstanceType<typeof LoraDatasetPanel> | null>(null);
const historyRef = ref<InstanceType<typeof LoraTrainHistory> | null>(null);
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
  default_prompt: 'A photo of sks person',
  output_name: '',
  iterations: null as number | null,
  lora_rank: null as number | null,
  learning_rate: null as number | null,
});

const selectedDataset = computed(() =>
  datasets.value.find((d) => d.id === form.dataset_id)
);

const modelPresets = computed(
  () => presetsByModel.value[form.base_model] || presetsByModel.value['flux1-dev'] || {}
);

const activePresetDetail = computed(() => {
  if (form.preset === 'custom') return null;
  return (modelPresets.value[form.preset] || null) as Record<string, unknown> | null;
});

const confirmPreviewImages = computed(() =>
  (selectedDataset.value?.images || []).slice(0, 8)
);

const wizardSummary = computed(() => {
  const rows: Array<{ key: string; label: string; value: string }> = [];
  if (form.base_model) {
    const m = trainableModels.value.find((x) => x.id === form.base_model);
    rows.push({ key: 'base', label: t('loraTrain.summaryBase'), value: modelDisplayName(m || { id: form.base_model }) });
  }
  if (selectedDataset.value) {
    rows.push({
      key: 'dataset',
      label: t('loraTrain.summaryDataset'),
      value: `${selectedDataset.value.name} (${selectedDataset.value.image_count || 0})`,
    });
  }
  if (form.preset && step.value >= 2) {
    rows.push({
      key: 'preset',
      label: t('loraTrain.summaryPreset'),
      value: t(`loraTrain.preset.${form.preset}`),
    });
  }
  return rows;
});

function stepDone(i: number): boolean {
  if (i === 0) return !!form.base_model && step.value > 0;
  if (i === 1) return (selectedDataset.value?.image_count || 0) >= 3 && step.value > 1;
  if (i === 2) return !!form.progress_prompt.trim() && step.value > 2;
  return false;
}

function presetStats(p: string): string {
  const cfg = modelPresets.value[p] as Record<string, unknown> | undefined;
  if (!cfg) return '';
  const it = cfg.iterations;
  const rank = cfg.lora_rank;
  if (it == null || rank == null) return '';
  return t('loraTrain.presetStatsShort', { iterations: it, rank });
}

function presetResolution(cfg: Record<string, unknown> | null): string {
  const res = cfg?.resolution;
  if (!Array.isArray(res) || res.length < 2) return '';
  return `${res[0]}×${res[1]}`;
}

function confirmImageUrl(file: string): string {
  if (!form.dataset_id || !file) return '';
  return api.loras.datasetImageUrl(form.dataset_id, file);
}

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

function openRun(taskId: string) {
  if (!taskId) return;
  activeRunId.value = taskId;
}

function onRunBack() {
  activeRunId.value = '';
  historyRef.value?.refresh();
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

function onCaptionEditsUpdate(next: Record<string, string>) {
  Object.keys(captionEdits).forEach((k) => delete captionEdits[k]);
  Object.assign(captionEdits, next);
}

function selectBase(m: any) {
  if (!m.trainable || !m.ready) return;
  form.base_model = m.id;
}

async function loadMeta() {
  const [modelsRes, dsRes, reqRes] = await Promise.all([
    api.loras.trainableModels(),
    api.loras.listDatasets(),
    api.loras.trainingRequirements(form.base_model),
  ]);
  trainableModels.value = (modelsRes as any).items || [];
  presetsByModel.value = (modelsRes as any).presets_by_model || {};
  datasets.value = (dsRes as any).items || [];
  requirements.value = reqRes;
}

async function refreshRequirements() {
  try {
    requirements.value = await api.loras.trainingRequirements(form.base_model);
  } catch {
    // ignore
  }
}

function restoreDraft() {
  try {
    const raw = getItem(DQ_STORAGE.LORA_TRAIN_DRAFT);
    if (!raw) return;
    const draft = JSON.parse(raw) as Partial<typeof form> & { step?: number };
    if (draft.base_model) form.base_model = draft.base_model;
    if (draft.dataset_id) form.dataset_id = draft.dataset_id;
    if (draft.preset) form.preset = draft.preset;
    if (draft.progress_prompt) form.progress_prompt = draft.progress_prompt;
    if (draft.default_prompt) form.default_prompt = draft.default_prompt;
    if (draft.output_name) form.output_name = draft.output_name;
    if (typeof draft.step === 'number' && draft.step >= 0 && draft.step <= 3) {
      step.value = Math.min(draft.step, maxReachableStep.value);
    }
  } catch {
    // ignore corrupt draft
  }
}

function persistDraft() {
  setItem(
    DQ_STORAGE.LORA_TRAIN_DRAFT,
    JSON.stringify({
      base_model: form.base_model,
      dataset_id: form.dataset_id,
      preset: form.preset,
      progress_prompt: form.progress_prompt,
      default_prompt: form.default_prompt,
      output_name: form.output_name,
      step: step.value,
    })
  );
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
      datasets.value = datasets.value.map((d) => (d.id === id ? { ...d, ...(ds as Record<string, unknown>) } : d));
    } catch (e: unknown) {
      toast.error(apiErrorMessage(e));
    }
  }
);

async function importFromRouteQuery() {
  const raw = route.query.import;
  const ids = (Array.isArray(raw) ? raw.join(',') : String(raw || ''))
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (!ids.length) return;

  const explicitDatasetId = String(route.query.dataset_id || '').trim();
  const datasetName =
    String(route.query.dataset_name || '').trim() || t('loraTrain.canvasImportDataset');

  if (explicitDatasetId) {
    form.dataset_id = explicitDatasetId;
    if (!datasets.value.some((d) => d.id === explicitDatasetId)) {
      try {
        const ds = await api.loras.getDataset(explicitDatasetId);
        datasets.value = [ds as Record<string, unknown>, ...datasets.value];
      } catch (e: unknown) {
        toast.error(apiErrorMessage(e));
        return;
      }
    }
  } else if (datasetPanelRef.value) {
    await datasetPanelRef.value.createDatasetWithName(datasetName);
  } else {
    const ds = (await api.loras.createDataset({
      name: datasetName,
      default_prompt: form.default_prompt,
    })) as Record<string, any>;
    datasets.value.unshift(ds);
    form.dataset_id = ds.id;
  }

  if (!form.dataset_id) return;
  try {
    const ds = await api.loras.importAssets(form.dataset_id, ids, form.default_prompt);
    datasets.value = datasets.value.map((d) =>
      d.id === form.dataset_id ? { ...d, ...(ds as object) } : d
    );
    Object.keys(captionEdits).forEach((k) => delete captionEdits[k]);
    for (const img of (ds as any).images || []) {
      captionEdits[img.file] = img.prompt || form.default_prompt || '';
    }
    step.value = 1;
    toast.success(t('loraTrain.galleryImported'));
    const q = { ...route.query };
    delete q.import;
    delete q.dataset_name;
    delete q.dataset_id;
    router.replace({ query: q });
  } catch (e: any) {
    toast.error(e?.message || String(e));
  }
}

function openRunFromQuery() {
  const runId = String(route.query.run || '').trim();
  if (!runId) return;
  activeRunId.value = runId;
  const q = { ...route.query };
  delete q.run;
  router.replace({ query: q });
}

watch(step, (s) => {
  if (s === 2 && !form.progress_prompt.trim() && form.default_prompt.trim()) {
    form.progress_prompt = form.default_prompt.trim();
  }
  persistDraft();
});

watch(
  () => [form.base_model, form.dataset_id, form.preset, form.progress_prompt, form.default_prompt, form.output_name],
  () => persistDraft()
);

watch(() => form.base_model, () => {
  void refreshRequirements();
});

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
    if (tid) {
      activeRunId.value = tid;
      historyRef.value?.refresh();
    }
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    submitting.value = false;
  }
}

function openModelsUserLorasPage() {
  openModelsUserLoras(router);
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
    restoreDraft();
    if (!form.dataset_id && datasets.value.length === 1) {
      form.dataset_id = datasets.value[0].id;
    }
    openRunFromQuery();
    await importFromRouteQuery();
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
});
</script>

<style scoped>
.lora-train-page {
  display: flex;
  gap: 20px;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
  padding: 4px 6px 4px 0;
}

/* —— Sidebar rail —— */
.lora-train-page__rail {
  width: 272px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 18px 16px;
  border-radius: var(--dq-radius-group, 16px);
  border: 0.5px solid color-mix(in srgb, var(--dq-border) 55%, transparent);
  background: linear-gradient(
    165deg,
    color-mix(in srgb, var(--dq-fill-secondary) 88%, var(--dq-accent) 4%) 0%,
    var(--dq-fill-secondary) 42%,
    color-mix(in srgb, var(--dq-bg-base) 92%, transparent) 100%
  );
  box-shadow: 0 12px 40px color-mix(in srgb, var(--dq-shadow-md) 65%, transparent);
  overflow: hidden;
}

.lora-train-page__rail-head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.lora-train-page__rail-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  border-radius: 10px;
  background: color-mix(in srgb, var(--dq-accent) 18%, var(--dq-fill-tertiary));
  color: var(--dq-accent);
  box-shadow: inset 0 0 0 0.5px color-mix(in srgb, var(--dq-accent) 35%, transparent);
}

.lora-train-page__rail-title {
  margin: 0;
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--dq-label-primary);
}

.lora-train-page__rail-sub {
  margin: 4px 0 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lora-train-page__steps {
  display: flex;
  flex-direction: column;
  gap: 0;
  margin: 4px 0 0;
  padding: 0;
}

.lora-train-page__step {
  display: flex;
  align-items: stretch;
  gap: 10px;
  width: 100%;
  padding: 6px 8px;
  border: none;
  border-radius: var(--radius-md);
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease;
}

.lora-train-page__step:hover:not(:disabled) {
  background: color-mix(in srgb, var(--dq-fill-tertiary) 80%, transparent);
}

.lora-train-page__step.is-active {
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-fill-secondary));
}

.lora-train-page__step.is-locked {
  opacity: 0.38;
  cursor: not-allowed;
}

.lora-train-page__step-rail {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 24px;
  flex-shrink: 0;
}

.lora-train-page__step-dot {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  background: var(--dq-fill-tertiary);
  color: var(--dq-label-secondary);
  border: 0.5px solid var(--dq-border-subtle);
  transition: all 0.15s ease;
}

.lora-train-page__step.is-active .lora-train-page__step-dot {
  background: var(--dq-accent);
  color: var(--dq-accent-contrast, #fff);
  border-color: transparent;
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.lora-train-page__step.is-done .lora-train-page__step-dot {
  background: color-mix(in srgb, var(--dq-success) 16%, var(--dq-fill-secondary));
  color: var(--dq-success);
  border-color: color-mix(in srgb, var(--dq-success) 35%, transparent);
}

.lora-train-page__step-line {
  flex: 1;
  width: 2px;
  min-height: 14px;
  margin: 4px 0;
  border-radius: 1px;
  background: var(--dq-border-subtle);
}

.lora-train-page__step.is-done .lora-train-page__step-line {
  background: color-mix(in srgb, var(--dq-success) 45%, var(--dq-border-subtle));
}

.lora-train-page__step-body {
  display: flex;
  align-items: center;
  min-height: 22px;
  padding: 2px 0 14px;
}

.lora-train-page__step:last-child .lora-train-page__step-body {
  padding-bottom: 2px;
}

.lora-train-page__step-label {
  font-size: 13px;
  font-weight: 500;
  color: var(--dq-label-secondary);
}

.lora-train-page__step.is-active .lora-train-page__step-label {
  color: var(--dq-label-primary);
  font-weight: 600;
}

.lora-train-page__chip-panel {
  padding: 10px 12px;
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-train-page__chip-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  margin-bottom: 8px;
}

.lora-train-page__chips {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-train-page__chip {
  font-size: 11px;
  line-height: 1.4;
  color: var(--dq-label-primary);
  padding: 4px 8px;
  border-radius: var(--radius-sm);
  background: var(--dq-fill-tertiary);
}

.lora-train-page__history-fold {
  margin-top: auto;
  min-height: 0;
}

.lora-train-page__history-fold summary {
  cursor: pointer;
  font-size: 11px;
  font-weight: 600;
  color: var(--dq-label-tertiary);
  letter-spacing: 0.03em;
  list-style: none;
  padding: 6px 0;
}

.lora-train-page__history-fold summary::-webkit-details-marker {
  display: none;
}

.lora-train-page__mem {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius-md);
  background: color-mix(in srgb, var(--dq-bg-base) 50%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-train-page__mem-dot {
  width: 8px;
  height: 8px;
  margin-top: 4px;
  border-radius: 999px;
  flex-shrink: 0;
}

.lora-train-page__mem-dot.is-ok {
  background: var(--dq-success);
  box-shadow: 0 0 8px color-mix(in srgb, var(--dq-success) 55%, transparent);
}

.lora-train-page__mem-dot.is-warn {
  background: var(--dq-warning, #f5a623);
}

.lora-train-page__mem-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 11px;
  line-height: 1.4;
  color: var(--dq-label-tertiary);
}

.lora-train-page__mem-text strong {
  font-size: 12px;
  color: var(--dq-label-primary);
}

/* —— Main stage —— */
.lora-train-page__stage {
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow: hidden;
}

.lora-train-page__progress {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  flex-shrink: 0;
}

.lora-train-page__progress-seg {
  height: 3px;
  border-radius: 999px;
  background: var(--dq-fill-tertiary);
  transition: background 0.2s ease, box-shadow 0.2s ease;
}

.lora-train-page__progress-seg.is-done {
  background: color-mix(in srgb, var(--dq-accent) 55%, var(--dq-fill-tertiary));
}

.lora-train-page__progress-seg.is-current {
  background: var(--dq-accent);
  box-shadow: 0 0 10px color-mix(in srgb, var(--dq-accent) 45%, transparent);
}

.lora-train-page__stage-head {
  flex-shrink: 0;
}

.lora-train-page__stage-head-main {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.lora-train-page__stage-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  flex-shrink: 0;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-fill-secondary));
  color: var(--dq-accent);
}

.lora-train-page__stage-title {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  letter-spacing: -0.035em;
  color: var(--dq-label-primary);
}

.lora-train-page__stage-desc {
  margin: 6px 0 0;
  font-size: 13px;
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  max-width: 56ch;
}

.lora-train-page__surface {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 20px 22px;
  border-radius: var(--dq-radius-group, 16px);
  border: 0.5px solid color-mix(in srgb, var(--dq-border) 50%, transparent);
  background: color-mix(in srgb, var(--dq-fill-secondary) 72%, var(--dq-bg-base));
  box-shadow: inset 0 1px 0 color-mix(in srgb, #fff 4%, transparent);
}

.lora-train-page__panel {
  max-width: 880px;
}

.lora-train-page__panel--flush {
  max-width: none;
}

.lora-train-page__footer {
  flex-shrink: 0;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: var(--radius-md);
  border: 0.5px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-fill-secondary) 90%, var(--dq-bg-base));
}

.lora-train-page__footer-left {
  justify-self: start;
}

.lora-train-page__footer-right {
  justify-self: end;
}

.lora-train-page__footer-step {
  font-size: 12px;
  font-weight: 600;
  color: var(--dq-label-tertiary);
  font-variant-numeric: tabular-nums;
}

/* —— Compact model tiles —— */
.lora-train-page__model-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  max-width: 640px;
}

.lora-model-tile {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  padding: 14px 14px 14px 12px;
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: 14px;
  background: color-mix(in srgb, var(--dq-bg-base) 65%, var(--dq-fill-secondary));
  cursor: pointer;
  text-align: left;
  transition:
    border-color 0.15s ease,
    background 0.15s ease,
    box-shadow 0.15s ease,
    transform 0.12s ease;
}

.lora-model-tile:hover:not(:disabled) {
  border-color: color-mix(in srgb, var(--dq-accent) 35%, var(--dq-border));
  transform: translateY(-1px);
  box-shadow: 0 8px 24px color-mix(in srgb, var(--dq-shadow-md) 70%, transparent);
}

.lora-model-tile.is-selected {
  border-color: color-mix(in srgb, var(--dq-accent) 65%, var(--dq-border));
  background: color-mix(in srgb, var(--dq-accent) 10%, var(--dq-fill-secondary));
  box-shadow:
    inset 0 0 0 1px color-mix(in srgb, var(--dq-accent) 25%, transparent),
    0 8px 28px color-mix(in srgb, var(--dq-accent) 18%, transparent);
}

.lora-model-tile.is-disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.lora-model-tile__avatar {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px;
  height: 44px;
  flex-shrink: 0;
  border-radius: 12px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.02em;
  background: var(--dq-fill-tertiary);
  color: var(--dq-label-secondary);
}

.lora-model-tile.is-ready .lora-model-tile__avatar {
  background: color-mix(in srgb, var(--dq-success) 14%, var(--dq-fill-secondary));
  color: var(--dq-success);
}

.lora-model-tile.is-selected .lora-model-tile__avatar {
  background: color-mix(in srgb, var(--dq-accent) 20%, var(--dq-fill-secondary));
  color: var(--dq-accent);
}

.lora-model-tile__body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.lora-model-tile__name {
  font-size: 14px;
  font-weight: 600;
  color: var(--dq-label-primary);
  letter-spacing: -0.02em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.lora-model-tile__meta {
  font-size: 11px;
  color: var(--dq-label-tertiary);
}

.lora-model-tile.is-ready .lora-model-tile__meta {
  color: color-mix(in srgb, var(--dq-success) 80%, var(--dq-label-tertiary));
}

.lora-model-tile__check {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  flex-shrink: 0;
  border-radius: 999px;
  border: 1.5px solid var(--dq-border);
  font-size: 12px;
  font-weight: 700;
  color: transparent;
  transition: all 0.15s ease;
}

.lora-model-tile.is-selected .lora-model-tile__check {
  border-color: var(--dq-accent);
  background: var(--dq-accent);
  color: var(--dq-accent-contrast, #fff);
}

.lora-train-page__preset-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}

.lora-train-page__preset-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  padding: 14px 16px;
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
  cursor: pointer;
  text-align: left;
  transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}

.lora-train-page__preset-card:hover {
  border-color: var(--dq-border);
}

.lora-train-page__preset-card.is-selected {
  border-color: color-mix(in srgb, var(--dq-accent) 55%, var(--dq-border));
  background: color-mix(in srgb, var(--dq-accent) 9%, var(--dq-fill-secondary));
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--dq-accent) 22%, transparent);
}

.lora-train-page__preset-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-train-page__preset-desc {
  font-size: 11px;
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lora-train-page__preset-stats {
  font-size: 10px;
  font-weight: 600;
  color: var(--dq-accent);
  margin-top: 2px;
}

.lora-train-page__preset-detail {
  margin-bottom: 16px;
  padding: 12px 14px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 50%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-train-page__preset-detail-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: var(--dq-label-secondary);
  margin-bottom: 8px;
}

.lora-train-page__preset-detail-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  font-size: 12px;
  color: var(--dq-label-primary);
}

.lora-train-page__preset-crop-hint {
  font-style: normal;
  color: var(--dq-label-tertiary);
}

.lora-train-page__field-hint {
  margin: 0;
  font-size: 11px;
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.lora-train-page__confirm-intro {
  margin: 0 0 14px;
  font-size: 13px;
  color: var(--dq-label-secondary);
  line-height: 1.5;
}

.lora-train-page__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 14px;
}

.lora-train-page__label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
  color: var(--dq-label-secondary);
}

.lora-train-page__advanced {
  margin-top: 4px;
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
  gap: 8px;
}

.lora-train-page__summary li {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 14px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 50%, var(--dq-fill-secondary));
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

.lora-train-page__confirm-preview {
  margin-bottom: 16px;
}

.lora-train-page__confirm-preview-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  color: var(--dq-label-secondary);
  margin-bottom: 8px;
}

.lora-train-page__confirm-preview-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.lora-train-page__confirm-preview-grid img {
  width: 64px;
  height: 64px;
  object-fit: cover;
  border-radius: 10px;
  border: 0.5px solid var(--dq-border);
}

.lora-train-page__confirm-preview-more {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 64px;
  height: 64px;
  border-radius: 10px;
  background: var(--dq-fill-tertiary);
  border: 0.5px dashed var(--dq-border);
  font-size: 13px;
  font-weight: 600;
  color: var(--dq-label-tertiary);
}

@media (max-width: 960px) {
  .lora-train-page {
    flex-direction: column;
    overflow: auto;
  }

  .lora-train-page__rail {
    width: 100%;
    max-height: none;
  }

  .lora-train-page__model-grid {
    grid-template-columns: 1fr;
  }
}
</style>
