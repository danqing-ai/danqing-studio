<template>
  <StudioLayout class="studio-create-page" @scroll="onCanvasScroll">
    <template #filters>
      <StudioGalleryFilters
        :filter-time="filterTime"
        :filter-models="filterModels"
        :time-options="timeOptions"
        :model-options="allModelOptions"
        :selection-mode="selectionMode"
        @update:filter-time="filterTime = $event"
        @update:filter-models="filterModels = $event"
        @refresh="refreshGallery"
        @toggle-selection-mode="toggleSelectionMode"
      />
    </template>
    <template #canvas>
      <StudioCanvas
        :items="galleryItems"
        :active-tasks="activeAudioTasks"
        :loading="galleryLoading"
        :has-more="galleryHasMore"
        media="audio"
        :has-active-filters="hasActiveFilters"
        :selection-mode="selectionMode"
        :selected-paths="selectedPaths"
        :all-selected="allLoadedSelected"
        @select="onGallerySelect"
        @card-action="onCardAction"
        @reset-filters="resetGalleryFilters"
        @load-more="loadGallery(false)"
        @toggle-select="toggleSelect"
        @select-all="selectAllLoaded"
        @batch-delete="batchDeleteSelected"
        @clear-selection="clearSelection"
      />
    </template>
    <template #composer>
      <AudioComposer
        v-model="params.prompt"
        v-model:title="params.title"
        v-model:work-mode="audioWorkTab"
        v-model:model="selectedModelVersion"
        v-model:duration="params.duration"
        v-model:batch-count="params.n"
        :generating="generating"
        :can-generate="audioWorkTab === 'create' ? !submitDisabled : !coverSubmitDisabled"
        :generate-label="generateLabel"
        :model-options="audioModelSelectOptions"
        :duration-options="durationOptions"
        :styles="filteredPresets"
        :params="audioComposerParams"
        @update:params="applyAudioComposerParams"
        :has-custom-params="hasCustomParams"
        :show-negative-prompt="supportsNegativePrompt"
        :show-lyrics="true"
        :show-codec-controls="false"
        :show-cover-fidelity="audioWorkTab === 'cover'"
        :reference-media="coverReferenceMedia"
        :current-model-config="advancedParamModelConfig"
        @generate="onAudioComposerGenerate"
        @pick-reference="onPickCoverSource"
        @remove-reference="clearCoverSource"
        @model-change="onModelChange"
        @reset-defaults="restoreDefaults"
      />
    </template>
  </StudioLayout>

  <!-- Audio preview dialog -->
  <GalleryPreviewDialog
    v-model:visible="audioPreviewVisible"
    v-model:index="selectedAudioIndex"
    :items="galleryItems"
    media="audio"
  />
</template>

<script setup lang="ts">
// @ts-nocheck — legacy create view; narrow types in a follow-up pass
import { ref, reactive, computed, watch, onMounted, nextTick, inject, unref, type Ref } from 'vue';
import { useRouter } from 'vue-router';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt, $mn, $md, $mvn, $pn } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import { DQ_STORAGE, getItem, setItem, type StorageKey } from '@/utils/storage';

import { useModelRegistryFilters } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import { warnIfRiskyMemory } from '@/composables/memoryHint';
import type { SystemInfo, Task } from '@/types';
import { assetDisplayLabel, previewDisplayCaption } from '@/utils/assetDisplay';
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvas from '@/components/studio/StudioCanvas.vue';
import StudioGalleryFilters from '@/components/studio/StudioGalleryFilters.vue';
import AudioComposer from '@/components/studio/AudioComposer.vue';
import GalleryPreviewDialog from '@/components/gallery/GalleryPreviewDialog.vue';
import { useStudioGallery } from '@/composables/useStudioGallery';

const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const router = useRouter();
const systemInfo = inject<Ref<SystemInfo>>('systemInfo');

// ---- Helpers (migrated from window globals) ----
const STORAGE_KEY = 'dq-studio.audio-create-prompt-draft.v3';

function parseModelVersion(s: string) {
  const [m, v] = (s || '').split('|');
  return { model: m || '', version: v || '', modelKey: m || '' };
}

function isAudioModel(config: any) {
  return config && config.media === 'audio';
}

/** Registry v2: `actions` is `{ create: {}, ... }`, not a string array. */
function supportsAction(actions: unknown, action: string): boolean {
  if (actions == null) return false;
  if (Array.isArray(actions)) {
    return actions.includes(action);
  }
  if (typeof actions === 'object') {
    const rec = actions as Record<string, unknown>;
    return Object.prototype.hasOwnProperty.call(rec, action) && rec[action] != null;
  }
  return false;
}

const TSU = {
  tagType: (s: string) => {
    if (s === 'running') return 'primary';
    if (s === 'completed') return 'success';
    if (s === 'failed') return 'danger';
    return 'info';
  },
  statusText: (s: string) => s || ''
};

// ---- Tab navigation ----
const audioWorkTab = ref('create');
const audioWorkSegmentOptions = computed(() => [
  { label: $tt('action.audio.create'), value: 'create' },
  { label: $tt('action.audio.cover'), value: 'cover' },
]);

// ---- State ----
const generating = ref(false);
const modelsLoading = ref(false);
const previewPlayerRef = ref(null);
const previewIsPlaying = ref(false);
const previewDurationSec = ref(0);

const modelRegistry = ref<Record<string, any>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});

const params = reactive({
  model: '',
  title: '',
  prompt: '',
  negative_prompt: '',
  duration: 30,
  instrumental: false,
  lyrics: '',
  vocal_language: '',
  vocal_type: '',
  bpm: null as number | null,
  key_scale: '',
  time_signature: '',
  steps: 8,
  guidance: 3.0,
  seed: null as number | null,
  n: 1,
  audio_format: 'wav',
});

const coverParams = reactive({
  prompt: '',
  source_fidelity: 1.0,
  seed: null as number | null,
  audio_format: 'wav',
});

const coverSourceSrc = ref('');
const coverSourceFile = ref<File | null>(null);
const coverSourceName = ref('');

const currentTask = reactive({ id: '', progress: 0, step: null as number | null, total: null as number | null, status: '' });
const previewAudioSrc = ref('');
const previewPrompt = ref('');
const previewLyrics = ref('');
const previewLyricsDownload = ref('');
const previewAudioKey = ref(0);
const recentAudio = reactive<Array<any>>([]);
const previewArtHue = computed(() => artHueForPrompt(previewPrompt.value));

const previewPlayerSubtitle = computed(() => {
  const parts: string[] = [];
  if (currentModelDisplayName.value) parts.push(currentModelDisplayName.value);
  if (previewDurationSec.value > 0) parts.push(formatTime(previewDurationSec.value));
  else if (params.duration) parts.push(formatTime(params.duration));
  return parts.join(' · ');
});

function artHueForPrompt(text: string) {
  let h = 0;
  const s = String(text || 'audio');
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) % 360;
  }
  return h;
}

// ---- Model display name ----
const currentModelKey = computed(() => {
  if (!params.model) return '';
  const colon = params.model.lastIndexOf(':');
  return colon > 0 ? params.model.slice(0, colon) : params.model;
});

const currentModelConfig = computed(() => {
  const mk = currentModelKey.value;
  if (!mk) return null;
  return modelRegistry.value[mk] || null;
});

const currentModelDisplayName = computed(() => {
  const c = currentModelConfig.value;
  const mk = currentModelKey.value;
  return $mn(c, mk) || params.model || '';
});

const modelReady = computed(() => {
  const mk = currentModelKey.value;
  if (!mk) return false;
  const ds = modelsDetailedStatus.value[mk];
  if (ds && ds.status) return ds.status === 'ready';
  return true;
});

const selectedModelNotReady = computed(() => {
  return !!params.model && !modelReady.value;
});

const supportsNegativePrompt = computed(() => {
  const c = currentModelConfig.value;
  return c && c.parameters && c.parameters.negative_prompt_support === true;
});

const hasCustomParams = computed(() => {
  const p = currentModelConfig.value?.parameters || {};
  const stepsDef = p.steps?.default ?? 8;
  const guidanceDef = p.guidance?.default ?? 3.0;
  const durationDef = p.duration?.default ?? 30;
  const formatDef = (p.audio_formats && p.audio_formats[0]) || audioFormats.value[0] || 'wav';
  return params.steps !== stepsDef || params.guidance !== guidanceDef
    || params.seed !== null || params.n !== 1 || params.duration !== durationDef
    || params.audio_format !== formatDef;
});

// ---- Registry-driven capability flags ----
const supportsBpm = computed(() => currentModelConfig.value?.parameters?.supports_bpm === true);
const supportsKeyScale = computed(() => currentModelConfig.value?.parameters?.supports_key_scale === true);
const supportsTimeSignature = computed(
  () => currentModelConfig.value?.parameters?.supports_time_signature === true,
);

const audioFormats = computed(() => {
  const allowed = currentModelConfig.value?.parameters?.audio_formats;
  if (Array.isArray(allowed) && allowed.length > 0) {
    return allowed.map((f) => String(f).toLowerCase());
  }
  return ['mp3', 'flac', 'wav', 'opus', 'aac'];
});

const promptPlaceholder = computed(() => $tt('audio.promptPlaceholder'));

const advancedParamModelConfig = computed(() => currentModelConfig.value);
const vocalLanguages = [
  { label: 'English', value: 'en' }, { label: '中文', value: 'zh' }, { label: '日本語', value: 'ja' },
  { label: '한국어', value: 'ko' }, { label: 'Français', value: 'fr' }, { label: 'Deutsch', value: 'de' },
  { label: 'Español', value: 'es' }, { label: 'Português', value: 'pt' },
];

const supportsVocalType = computed(() => {
  return currentModelConfig.value?.parameters?.supports_vocal_type === true;
});

const vocalTypes = computed(() => [
  { label: $tt('audio.vocalTypeMale'), value: 'male' },
  { label: $tt('audio.vocalTypeFemale'), value: 'female' },
  { label: $tt('audio.vocalTypeChorus'), value: 'chorus' },
  { label: $tt('audio.vocalTypeDuet'), value: 'duet' },
]);
const musicalKeys = [
  'C Major', 'C# Major', 'D Major', 'D# Major', 'E Major', 'F Major', 'F# Major',
  'G Major', 'G# Major', 'A Major', 'A# Major', 'B Major',
  'C Minor', 'C# Minor', 'D Minor', 'D# Minor', 'E Minor', 'F Minor', 'F# Minor',
  'G Minor', 'G# Minor', 'A Minor', 'A# Minor', 'B Minor',
];
const timeSignatures = [
  { label: '2/4', value: '2' }, { label: '3/4', value: '3' },
  { label: '4/4', value: '4' }, { label: '6/8', value: '6' },
];

// ---- Model selector ----
const selectedModelVersion = ref('');

const { commercialOnly: modelFilterCommercialOnly } = useModelRegistryFilters();

const allAudioVersions = computed(() => {
  const rows: Array<Record<string, unknown>> = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!isAudioModel(config)) continue;
    const act = audioWorkTab.value === 'cover' ? 'cover' : 'create';
    if (act === 'cover' && !supportsAction(config.actions, 'cover')) continue;
    if (act === 'create' && !supportsAction(config.actions, 'create')) continue;
    const versions = config.versions || {};
    const ds = modelsDetailedStatus.value[modelKey] || {};
    const versionStatuses = ds.versions || {};
    for (const [versionKey, versionConfig] of Object.entries(versions)) {
      const vst = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
      const ready =
        vst.ready === true || (ds.status === 'ready' && vst.ready !== false);
      rows.push({
        modelKey,
        versionKey,
        name: $mvn(modelKey, config, versionConfig),
        description: $md(config as { description?: string | { zh?: string; en?: string }; description_en?: string }),
        size: (versionConfig as Record<string, unknown>).size || '',
        status: ready ? 'ready' : (vst.status || 'not_downloaded'),
        ready,
        recommended: config.recommended === true && (versionConfig as Record<string, unknown>).default === true,
        commercialUseAllowed: config.commercial_use_allowed === true,
      });
    }
  }
  return rows;
});

const selectedModelPickerItem = computed(() => {
  const key = selectedModelVersion.value;
  if (!key) return null;
  return allAudioVersions.value.find((item) => `${item.modelKey}|${item.versionKey}` === key) ?? null;
});

const filteredModelPickerVersions = computed(() => {
  const rows = applyModelVersionFilters(allAudioVersions.value, {
    installedOnly: true,
    commercialOnly: modelFilterCommercialOnly.value,
  });
  rows.sort((a, b) => {
    const ar = a.recommended ? 1 : 0;
    const br = b.recommended ? 1 : 0;
    if (ar !== br) return br - ar;
    const an = String(a.name || '');
    const bn = String(b.name || '');
    try {
      return an.localeCompare(bn, 'zh');
    } catch {
      return an < bn ? -1 : an > bn ? 1 : 0;
    }
  });
  return rows;
});

const submitDisabled = computed(() => {
  if (!params.model) return true;
  if (!modelReady.value) return true;
  if (!params.prompt.trim()) return true;
  if (generating.value) return true;
  return false;
});

const supportsCoverAction = computed(() => {
  return supportsAction(currentModelConfig.value?.actions, 'cover');
});

const coverSubmitDisabled = computed(() => {
  if (!params.model) return true;
  if (!modelReady.value) return true;
  if (!supportsCoverAction.value) return true;
  if (!coverSourceFile.value && !coverSourceSrc.value) return true;
  if (generating.value) return true;
  return false;
});

// ---- Composer adapters ----

const audioComposerParams = computed(() => ({
  steps: params.steps,
  guidance: params.guidance,
  seed: params.seed ?? null,
  negative_prompt: params.negative_prompt,
  lyrics: params.lyrics,
  instrumental: params.instrumental,
  bpm: params.bpm,
  key_scale: params.key_scale,
  time_signature: params.time_signature,
  vocal_language: params.vocal_language,
  vocal_type: params.vocal_type,
  source_fidelity: coverParams.source_fidelity,
}));

function applyAudioComposerParams(val: Record<string, unknown>) {
  params.steps = val.steps as number;
  params.guidance = val.guidance as number;
  params.seed = val.seed as number | null;
  params.negative_prompt = val.negative_prompt as string;
  params.lyrics = val.lyrics as string;
  params.instrumental = val.instrumental as boolean;
  params.bpm = val.bpm as number | null;
  params.key_scale = val.key_scale as string;
  params.time_signature = val.time_signature as string;
  params.vocal_language = val.vocal_language as string;
  params.vocal_type = val.vocal_type as string;
  coverParams.source_fidelity = (val.source_fidelity as number) ?? 1.0;
}

const audioModelSelectOptions = computed(() =>
  filteredModelPickerVersions.value.map((v) => ({
    label: String(v.name || ''),
    value: `${v.modelKey}|${v.versionKey}`,
    disabled: !v.ready,
    commercialUseAllowed: v.commercialUseAllowed === true,
  }))
);

const coverReferenceMedia = computed(() => {
  if (audioWorkTab.value !== 'cover') return null;
  if (!coverSourceSrc.value) return null;
  return {
    type: 'audio',
    previewUrl: coverSourceSrc.value,
    label: coverSourceName.value || $tt('audio.coverSource'),
  };
});

function onAudioComposerGenerate() {
  if (audioWorkTab.value === 'create') {
    startGeneration();
  } else {
    startCoverGeneration();
  }
}

// ---- Methods ----
async function loadModelRegistry() {
  modelsLoading.value = true;
  try {
    const [registryData, detailedStatusData] = await Promise.all([
      registryStore.load(),
      api.settings.getModelsDetailedStatus(),
    ]);
    modelRegistry.value = registryData?.models || {};
    if (detailedStatusData && typeof detailedStatusData === 'object') {
      modelsDetailedStatus.value = detailedStatusData as Record<string, unknown>;
    }
    // 恢复当前模式的模型
    const saved = getItem(getAudioModeStorageKey(audioWorkTab.value));
    if (saved) {
      const parsed = parseModelVersion(saved);
      const mk = parsed.modelKey || parsed.model || '';
      const vk = parsed.version || '';
      const key = vk ? `${mk}|${vk}` : mk;
      const stillValid = filteredModelPickerVersions.value.some(
        (r) => `${r.modelKey}|${r.versionKey}` === key && r.ready
      );
      if (stillValid) {
        selectedModelVersion.value = key;
        onModelChange(key);
      }
    }
  } catch (e) {
    console.error('Failed to load audio model registry:', e);
  } finally {
    modelsLoading.value = false;
  }
}

function applyDefaults(modelConfig: any) {
  if (!modelConfig || !modelConfig.parameters) return;
  const p = modelConfig.parameters;
  if (p.steps && p.steps.default != null) params.steps = p.steps.default;
  if (p.guidance && p.guidance.default != null) params.guidance = p.guidance.default;
  if (p.duration) {
    if (p.duration.default != null) params.duration = p.duration.default;
    const dmin = p.duration.min ?? 10;
    const dmax = p.duration.max ?? 600;
    if (params.duration < dmin) params.duration = dmin;
    if (params.duration > dmax) params.duration = dmax;
  }
  if (p.audio_formats && Array.isArray(p.audio_formats) && p.audio_formats.length > 0) {
    const fmt = String(p.audio_formats[0]).toLowerCase();
    params.audio_format = fmt;
  }
  if (!audioFormats.value.includes(String(params.audio_format || '').toLowerCase())) {
    params.audio_format = audioFormats.value[0] || 'wav';
  }
  params.negative_prompt = '';
  params.lyrics = '';
  params.bpm = null;
  params.key_scale = '';
  params.time_signature = '';
  params.vocal_language = '';
  params.vocal_type = '';
}

function getAudioModeStorageKey(mode: string): StorageKey {
  const map: Record<string, StorageKey> = {
    create: DQ_STORAGE.AUDIO_MODEL_CREATE,
    cover: DQ_STORAGE.AUDIO_MODEL_COVER,
  };
  return map[mode] || DQ_STORAGE.AUDIO_MODEL_CREATE;
}

function onModelChange(val: string) {
  const parsed = parseModelVersion(val);
  const mk = parsed.modelKey || parsed.model || '';
  const vk = parsed.version || '';
  params.model = vk ? mk + ':' + vk : mk;
  const mc = modelRegistry.value[mk];
  applyDefaults(mc);
  loadPromptDraft();
  // 保存到当前模式
  setItem(getAudioModeStorageKey(audioWorkTab.value), val);
}

function randomizeSeed() {
  params.seed = Math.floor(Math.random() * 2147483647);
}

function restoreDefaults() {
  const mc = currentModelConfig.value;
  applyDefaults(mc);
  params.n = 1;
  params.seed = null;
}

function goToDownload() {
  router.push({ name: 'models' });
}

function loadPromptDraft() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) params.prompt = raw;
  } catch (e) { /* ignore */ }
}

function savePromptDraft() {
  try {
    localStorage.setItem(STORAGE_KEY, params.prompt || '');
  } catch (e) { /* ignore */ }
}

function formatTime(seconds: number) {
  const s = Math.floor(seconds || 0);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m + ':' + String(sec).padStart(2, '0');
}

function onPickCoverSource() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'audio/*';
  input.onchange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      onCoverFilePicked(target.files[0]);
    }
  };
  input.click();
}

function onCoverFilePicked(file: File) {
  if (coverSourceSrc.value && coverSourceSrc.value.startsWith('blob:')) {
    URL.revokeObjectURL(coverSourceSrc.value);
  }
  coverSourceFile.value = file;
  coverSourceName.value = file.name;
  coverSourceSrc.value = URL.createObjectURL(file);
}

function clearCoverSource() {
  if (coverSourceSrc.value && coverSourceSrc.value.startsWith('blob:')) {
    URL.revokeObjectURL(coverSourceSrc.value);
  }
  coverSourceFile.value = null;
  coverSourceName.value = '';
  coverSourceSrc.value = '';
}

function attachTaskStream(tid: string) {
  if (!api.gen?.streamMediaTask) {
    generating.value = false;
    return;
  }

  tasksStore.registerPageOwnedStream(tid);
  tasksStore.clearTaskLogs(tid);
  tasksStore.appendTaskLog(tid, $tt('studio.startingGen'), 'info');

  const assetFileUrl = (aid: string) =>
    api.gallery?.getImageUrl
      ? api.gallery.getImageUrl('asset:' + aid)
      : '/api/assets/' + aid + '/file';

  const applyResultMeta = (meta: Record<string, unknown> | undefined) => {
    const m = meta || {};
    const eff = String(m.lyrics_effective || '').trim();
    previewLyrics.value = eff;
    previewLyricsDownload.value = eff;
  };

  const addRecentFromAsset = (aid: string, meta?: Record<string, unknown>, promptLabel?: string) => {
    const url = assetFileUrl(aid);
    const eff = meta ? String(meta.lyrics_effective || '').trim() : '';
    recentAudio.unshift({
      id: aid,
      url,
      title: String(params.title || '').trim(),
      prompt: promptLabel || params.prompt,
      lyrics: eff,
      duration: 0,
      playing: false,
    });
  };

  const setPreviewFromAsset = (aid: string, meta?: Record<string, unknown>, promptLabel?: string) => {
    const url = assetFileUrl(aid);
    const changed = previewAudioSrc.value !== url;
    previewAudioSrc.value = url;
    previewPrompt.value = previewDisplayCaption(String(params.title || ''), promptLabel || params.prompt);
    applyResultMeta(meta);
    previewIsPlaying.value = false;
    if (changed) previewAudioKey.value += 1;
  };

  api.gen.streamMediaTask(tid, {
    onLog: (logData: unknown) => {
      tasksStore.ingestTaskLog(tid, logData);
    },
    onStatus: (statusData: unknown) => {
      const row = statusData as Record<string, unknown>;
      if (row.status) currentTask.status = row.status as string;
      if (row.progress != null) currentTask.progress = row.progress as number;
    },
    onProgress: (progressData: unknown) => {
      const row = progressData as Record<string, unknown>;
      tasksStore.ingestTaskProgressLog(tid, row);
      tasksStore.patchLiveTaskProgress(tid, {
        progress: row.progress as number | undefined,
        step: row.step as number | undefined,
        total: row.total as number | undefined,
        eta_seconds: row.eta_seconds as number | undefined,
        progressMessage: (row.message ?? row.phase) as string | undefined,
      });
      if (row.progress != null) currentTask.progress = row.progress as number;
      if (row.step != null) currentTask.step = row.step as number;
      if (row.total != null) currentTask.total = row.total as number;
    },
    onResult: (resultData: unknown) => {
      const row = resultData as Record<string, unknown>;
      const meta = row.metadata as Record<string, unknown> | undefined;
      applyResultMeta(meta);
      const ids = row.asset_ids as string[] | undefined;
      if (ids?.length) {
        ids.forEach((aid) => addRecentFromAsset(aid, meta));
        setPreviewFromAsset(ids[0], meta);
      }
    },
    onDone: (doneData: unknown) => {
      const row = doneData as Record<string, unknown>;
      tasksStore.unregisterPageOwnedStream(tid);
      if (row.status === 'completed') {
        tasksStore.appendTaskLog(tid, $tt('studio.genComplete'), 'success');
        const res = row.result as Record<string, unknown> | undefined;
        const meta = res?.metadata as Record<string, unknown> | undefined;
        applyResultMeta(meta);
        const pid = res?.primary_asset_id;
        if (pid) setPreviewFromAsset(String(pid), meta);
      } else if (row.status === 'failed') {
        tasksStore.appendTaskLog(tid, $tt('studio.genFailed', { msg: String(row.error || '') }), 'error');
        toast.error($tt('studio.genFailed', { msg: String(row.error || '') }));
      }
      generating.value = false;
    },
    onError: () => {
      tasksStore.unregisterPageOwnedStream(tid);
      tasksStore.appendTaskLog(tid, $tt('studio.connectionLost'), 'warning');
      toast.error($tt('studio.connectionLost'));
      generating.value = false;
    },
  });
}

async function startCoverGeneration() {
  if (generating.value) return;
  if (!params.model) { toast.warning('Please select a model'); return; }
  if (!modelReady.value) { toast.warning('Model is not ready'); return; }
  if (!supportsCoverAction.value) {
    toast.warning('Selected model does not support cover');
    return;
  }
  if (!coverSourceFile.value) {
    toast.warning($tt('audio.coverNeedSource'));
    return;
  }

  generating.value = true;
  currentTask.id = '';
  currentTask.progress = 0;
  currentTask.step = null;
  currentTask.total = null;
  currentTask.status = '';
  previewLyrics.value = '';
  previewLyricsDownload.value = '';

  try {
    const up = await api.gen.uploadAsset(coverSourceFile.value);
    const source_asset_id = (up as any)?.id;
    if (!source_asset_id) {
      toast.error($tt('studio.error', { msg: 'upload failed' }));
      generating.value = false;
      return;
    }

    const body = {
      model: params.model,
      operation: 'cover',
      source_asset_id,
      prompt: String(params.prompt || '').trim(),
      source_fidelity: coverParams.source_fidelity ?? 1.0,
      seed: coverParams.seed ?? params.seed ?? null,
      n: 1,
      audio_format: params.audio_format || 'wav',
    };

    const resp = await api.audios.createEdit(body);
    const tid = taskIdFromSubmitResponse(resp);
    if (!tid) {
      toast.error($tt('studio.error', { msg: 'missing task id in submit response' }));
      generating.value = false;
      return;
    }
    currentTask.id = tid;
    attachTaskStream(tid);
  } catch (e: any) {
    toast.error((e.response && e.response.data && e.response.data.detail) || e.message || 'Cover failed');
    generating.value = false;
  }
}

async function startGeneration() {
  if (generating.value) return;
  if (!params.model) { toast.warning('Please select a model'); return; }
  if (!modelReady.value) { toast.warning('Model is not ready'); return; }
  if (!params.prompt.trim()) { toast.warning('Please enter a prompt'); return; }

  const parsed = parseModelVersion(selectedModelVersion.value || '');
  const mk = parsed.modelKey || parsed.model || currentModelKey.value;
  const vk = parsed.version || '';
  const verCfg =
    mk && currentModelConfig.value?.versions
      ? currentModelConfig.value.versions[vk]
      : null;
  const sizeHuman = verCfg?.size ? String(verCfg.size) : '';
  const minMemRaw = currentModelConfig.value?.parameters?.min_unified_memory_gb;
  const minUnifiedMemoryGb =
    minMemRaw != null && Number(minMemRaw) > 0 ? Number(minMemRaw) : null;
  warnIfRiskyMemory({
    systemInfo: unref(systemInfo),
    versionSizeHuman: sizeHuman,
    minUnifiedMemoryGb,
    $tt,
  });

  generating.value = true;
  currentTask.id = '';
  currentTask.progress = 0;
  currentTask.step = null;
  currentTask.total = null;
  currentTask.status = '';
  previewLyrics.value = '';
  previewLyricsDownload.value = '';

  try {
    const body = {
      model: params.model,
      title: String(params.title || '').trim(),
      prompt: params.prompt.trim(),
      negative_prompt: params.negative_prompt || '',
      duration: params.duration ?? null,
      instrumental: !!params.instrumental,
      // Empty lyrics + prompt → backend auto-inspires lyrics via 5Hz LM when LM is enabled.
      lyrics: params.lyrics?.trim() || '',
      vocal_language: params.vocal_language?.trim() || '',
      vocal_type: params.vocal_type?.trim() || '',
      bpm: params.bpm ?? null,
      key_scale: params.key_scale || '',
      time_signature: params.time_signature || '',
      steps: params.steps ?? null,
      guidance: params.guidance ?? null,
      seed: params.seed ?? null,
      n: params.n ?? 1,
      audio_format: params.audio_format || 'wav',
    };

    const resp = await api.audios.createGeneration(body);
    const tid = taskIdFromSubmitResponse(resp);
    if (!tid) {
      toast.error($tt('studio.error', { msg: 'missing task id in submit response' }));
      generating.value = false;
      return;
    }
    currentTask.id = tid;
    attachTaskStream(tid);
  } catch (e: any) {
    toast.error((e.response && e.response.data && e.response.data.detail) || e.message || 'Generation failed');
    generating.value = false;
  }
}

function recentAudioLabel(item: { title?: string; prompt?: string; name?: string }) {
  return assetDisplayLabel(item, 'Audio');
}

function isRecentActive(ra: any) {
  return !!ra?.url && ra.url === previewAudioSrc.value;
}

function selectRecent(item: any) {
  if (!item?.url) return;
  const same = isRecentActive(item);
  previewAudio(item);
}

function previewAudio(item: any) {
  const nextUrl = String(item?.url || '');
  const changed = previewAudioSrc.value !== nextUrl;
  previewAudioSrc.value = nextUrl;
  previewPrompt.value = recentAudioLabel(item);
  previewLyrics.value = item.lyrics || '';
  previewLyricsDownload.value = item.lyrics || '';
  previewDurationSec.value = item.duration || 0;
  previewIsPlaying.value = false;
  if (changed) previewAudioKey.value += 1;
}

function toggleRecentPlay(ra: any) {
  if (!ra?.url) return;
  if (!isRecentActive(ra)) {
    previewAudio(ra);
    return;
  }
  previewPlayerRef.value?.togglePlay?.();
}

function downloadPreviewAudio() {
  downloadAudioUrl(previewAudioSrc.value);
}

function downloadAudioUrl(url: string) {
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = 'audio.' + (params.audio_format || 'wav');
  a.click();
}

function downloadLyricsText() {
  const text = previewLyricsDownload.value || previewLyrics.value;
  if (!text) return;
  const blob = new Blob([text + '\n'], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'lyrics.txt';
  a.click();
  URL.revokeObjectURL(url);
}

async function loadRecentAudio() {
  try {
    const data = await api.gen.listAssets('audio', 20, 0);
    const items = (data && data.items) || [];
    recentAudio.length = 0;
    items.forEach((item) => {
      const aid = String(item.id || '');
      if (!aid) return;
      const url = api.gallery.getImageUrl(`asset:${aid}`);
      const meta = (item.metadata as Record<string, unknown>) || {};
      recentAudio.push({
        id: aid,
        url,
        title: String(meta.title || ''),
        prompt: String(meta.prompt || ''),
        lyrics: String(meta.lyrics_effective || '').trim(),
        name: String(item.name || ''),
        duration:
          item.duration_seconds != null
            ? Number(item.duration_seconds)
            : meta.duration_seconds != null
              ? Number(meta.duration_seconds)
              : 0,
      });
    });
  } catch (e) {
    console.error('loadRecentAudio', e);
    toast.error($tt('studio.error', { msg: $tt('gallery.loadFailed') }));
  }
}

// ---- Lifecycle ----
onMounted(async () => {
  await loadModelRegistry();
  if (!params.model) {
    const versions = filteredModelPickerVersions.value;
    if (versions.length > 0) {
      const rec =
        versions.find((v) => v.ready && v.recommended) ||
        versions.find((v) => v.ready) ||
        versions[0];
      selectedModelVersion.value = `${rec.modelKey}|${rec.versionKey}`;
      onModelChange(selectedModelVersion.value);
    }
  }
  if (!params.model) {
    loadPromptDraft();
  }
  loadRecentAudio();
  loadGallery();
  loadPresets();
});

let _draftTimer: ReturnType<typeof setTimeout> | null = null;
watch(() => params.prompt, () => {
  if (_draftTimer) clearTimeout(_draftTimer);
  _draftTimer = setTimeout(savePromptDraft, 500);
});

// Refresh model picker when tab changes
watch(audioWorkTab, (newMode, oldMode) => {
  // 保存旧模式模型
  if (oldMode && selectedModelVersion.value) {
    setItem(getAudioModeStorageKey(oldMode), selectedModelVersion.value);
  }
  // 恢复新模式模型
  const saved = getItem(getAudioModeStorageKey(newMode));
  if (saved) {
    const parsed = parseModelVersion(saved);
    const mk = parsed.modelKey || parsed.model || '';
    const vk = parsed.version || '';
    const key = vk ? `${mk}|${vk}` : mk;
    const stillValid = filteredModelPickerVersions.value.some(
      (r) => `${r.modelKey}|${r.versionKey}` === key && r.ready
    );
    if (stillValid) {
      selectedModelVersion.value = key;
      onModelChange(key);
      return;
    }
  }
  // fallback: 检查当前模型是否仍可用
  if (selectedModelVersion.value) {
    const currentKey = selectedModelVersion.value;
    const stillValid = filteredModelPickerVersions.value.some(
      (r) => `${r.modelKey}|${r.versionKey}` === currentKey && r.ready
    );
    if (stillValid) {
      onModelChange(currentKey);
      return;
    }
  }
  // fallback: 选择第一个可用模型
  const pick = filteredModelPickerVersions.value.find((r) => r.ready) || filteredModelPickerVersions.value[0];
  if (pick) {
    selectedModelVersion.value = `${pick.modelKey}|${pick.versionKey}`;
    onModelChange(selectedModelVersion.value);
  }
});

watch(modelFilterCommercialOnly, () => {
  const rows = filteredModelPickerVersions.value;
  const key = selectedModelVersion.value;
  if (key && rows.some((r) => `${r.modelKey}|${r.versionKey}` === key)) {
    return;
  }
  const pick = rows.find((r) => r.ready) || rows[0];
  if (!pick) {
    return;
  }
  selectedModelVersion.value = `${pick.modelKey}|${pick.versionKey}`;
  onModelChange(selectedModelVersion.value);
});

// ---- Gallery / Studio Canvas ----
const {
  galleryItems,
  galleryLoading,
  galleryHasMore,
  filterTime,
  filterModels,
  selectionMode,
  selectedPaths,
  allLoadedSelected,
  timeOptions,
  allModelOptions,
  hasActiveFilters,
  loadGallery,
  refreshGallery,
  onCanvasScroll,
  resetGalleryFilters,
  toggleSelect,
  toggleSelectionMode,
  selectAllLoaded,
  deleteItem,
  batchDeleteSelected,
  clearSelection,
} = useStudioGallery('audio');

const activeAudioTasks = computed(() => {
  const running = tasksStore.queueState.running.filter((t: Task) =>
    String(t.kind || '').startsWith('audio.')
  );
  const queued = tasksStore.queueState.queued.filter((t: Task) =>
    String(t.kind || '').startsWith('audio.')
  );
  return [...running, ...queued].map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
});

// ---- Gallery preview dialog ----
const audioPreviewVisible = ref(false);
const selectedAudioIndex = ref(0);

function onGallerySelect(item: any) {
  const idx = galleryItems.value.findIndex((it) => it.path === item.path);
  selectedAudioIndex.value = idx >= 0 ? idx : 0;
  audioPreviewVisible.value = true;
}

function onCardAction({ action, item }: { action: string; item: any }) {
  if (action === 'delete') {
    deleteItem(item);
  }
}

const presets = ref<Record<string, any>>({});

const filteredPresets = computed(() => {
  const want = audioWorkTab.value === 'cover' ? new Set(['cover']) : new Set(['create']);
  function planPresetShapeOk(preset: any) {
    return (
      Array.isArray(preset.applies_to) &&
      preset.applies_to.length > 0 &&
      preset.media_scope === 'audio'
    );
  }
  function matches(preset: any) {
    if (!planPresetShapeOk(preset)) return false;
    return preset.applies_to.some((k: string) => want.has(k));
  }
  const entries = Object.entries(presets.value)
    .filter(([, preset]) => matches(preset))
    .sort((a, b) => a[0].localeCompare(b[0], 'zh'));
  const result: Record<string, any> = {};
  for (const [name, preset] of entries) {
    result[name] = preset;
  }
  return result;
});

const durationOptions = computed(() => {
  const cfg = currentModelConfig.value;
  const dmin = cfg?.parameters?.duration?.min ?? 10;
  const dmax = cfg?.parameters?.duration?.max ?? 600;
  const ddef = cfg?.parameters?.duration?.default ?? 30;
  const opts: { label: string; value: number }[] = [];
  [10, 15, 30, 45, 60, 90, 120, 180, 300].forEach((sec) => {
    if (sec >= dmin && sec <= dmax) {
      opts.push({ label: `${sec}s`, value: sec });
    }
  });
  if (!opts.find((o) => o.value === ddef)) {
    opts.push({ label: `${ddef}s`, value: ddef });
  }
  opts.sort((a, b) => a.value - b.value);
  return opts;
});

const generateLabel = computed(() => {
  if (audioWorkTab.value === 'cover') return $tt('audio.coverGenerate');
  return $tt('audio.generate');
});

async function loadPresets() {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as any) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
    presets.value = {};
  }
}
</script>

<style scoped>
/* AudioCreateView styles moved to AudioComposer.vue */
</style>
