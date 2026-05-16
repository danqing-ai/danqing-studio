<!-- @ts-nocheck -->
<template>
  <div class="create-page">
    <el-row :gutter="24">
      <!-- Left panel: creation area -->
      <el-col :xs="24" :md="16" :lg="17" :xl="18">
        <div class="creation-panel">

          <!-- Tab bar -->
          <el-segmented
            class="dq-work-segmented dq-work-segmented--sm"
            v-model="audioWorkTab"
            :options="audioWorkSegmentOptions"
            block
          />

          <!-- Model selector -->
          <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb studio-ep-model-card">
            <template #header>
              <div class="card-title">
                <el-icon><cpu /></el-icon>
                {{ $t('create.modelSelectTitle') }}
              </div>
            </template>
            <div class="studio-ep-model-toolbar">
              <el-select
                v-model="selectedModelVersion"
                size="large"
                filterable
                @change="onModelChange"
                :placeholder="$t('studio.selectModel')"
              >
                <el-option
                  v-for="mv in filteredModelPickerVersions"
                  :key="mv.key"
                  :label="mv.label"
                  :value="mv.key"
                  :disabled="!mv.ready"
                >
                  <div class="studio-ep-picker-option">
                    <span class="studio-ep-picker-option__name" :class="{ 'is-disabled': !mv.ready }">{{ mv.label }}</span>
                    <el-tag v-if="mv.isRec" size="small" type="success">{{ $t('studio.recommended') }}</el-tag>
                    <el-tag v-if="mv.ready" size="small" type="success">{{ $t('studio.ready') }}</el-tag>
                    <el-tag v-else size="small" type="warning">{{ $t('studio.notDownloaded') }}</el-tag>
                    <span v-if="mv.size" class="studio-ep-picker-option__meta">{{ mv.size }}</span>
                  </div>
                </el-option>
              </el-select>
            </div>
            <el-alert
              v-if="selectedModelNotReady"
              :title="$t('studio.modelNotReady', { name: currentModelDisplayName })"
              type="warning"
              :closable="false"
              class="studio-ep-alert-mt"
            >
              <template #default>
                <span>{{ $t('studio.notDownloadedMsg') }}</span>
                <el-button size="small" type="primary" class="studio-ep-alert-inline-btn" @click="goToDownload">
                  {{ $t('studio.goDownload') }}
                </el-button>
              </template>
            </el-alert>
          </el-card>
          <template v-if="audioWorkTab === 'create'">

            <!-- Prompt input -->
            <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
              <template #header>
                <div class="card-title">
                  <el-icon><edit-pen /></el-icon>
                  {{ $t('studio.prompt') }}
                </div>
              </template>
              <el-input
                v-model="params.prompt"
                type="textarea"
                :rows="4"
                :placeholder="$t('audio.promptPlaceholder')"
                resize="none"
                @keydown.meta.enter.prevent="startGeneration"
                @keydown.ctrl.enter.prevent="startGeneration"
              />
              <el-collapse v-if="supportsNegativePrompt" class="studio-ep-collapse-plain">
                <el-collapse-item :title="$t('audio.negativePrompt')" name="negative">
                  <el-input
                    v-model="params.negative_prompt"
                    type="textarea"
                    :rows="2"
                    :placeholder="$t('studio.optional')"
                  />
                </el-collapse-item>
              </el-collapse>
            </el-card>

            <!-- Lyrics -->
            <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb studio-ep-audio-lyrics-card">
              <template #header>
                <div class="card-title">
                  <el-icon><document /></el-icon>
                  {{ $t('audio.lyrics') }}
                </div>
              </template>
              <el-input
                v-model="params.lyrics"
                type="textarea"
                :rows="4"
                :placeholder="$t('audio.lyricsPlaceholder')"
                resize="none"
              />
              <p class="studio-ep-field-footnote">{{ $t('studio.optional') }}</p>
              <div class="studio-ep-audio-meta-row">
                <div class="studio-ep-audio-meta-item studio-ep-audio-meta-item--switch">
                  <span class="studio-ep-audio-meta-label">{{ $t('audio.instrumental') }}</span>
                  <el-switch v-model="params.instrumental" size="small" />
                </div>
                <div class="studio-ep-audio-meta-item studio-ep-audio-meta-item--vocal">
                  <span class="studio-ep-audio-meta-label">{{ $t('audio.vocalLanguage') }}</span>
                  <el-select
                    v-model="params.vocal_language"
                    size="small"
                    class="studio-ep-audio-vocal-lang-select"
                    clearable
                    :placeholder="$t('audio.vocalLanguageAuto')"
                  >
                    <el-option v-for="l in vocalLanguages" :key="l.value" :label="l.label" :value="l.value" />
                  </el-select>
                </div>
              </div>
            </el-card>

            <!-- Music params -->
            <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
              <template #header>
                <div class="card-title">
                  <el-icon><setting /></el-icon>
                  {{ $t('audio.musicParams') }}
                </div>
              </template>
              <el-form label-position="top" size="small">
                <el-form-item :label="$t('audio.duration')">
                  <el-radio-group v-model="params.duration" size="small">
                    <el-radio-button :value="30">30s</el-radio-button>
                    <el-radio-button :value="60">60s</el-radio-button>
                    <el-radio-button :value="90">90s</el-radio-button>
                    <el-radio-button :value="120">120s</el-radio-button>
                  </el-radio-group>
                </el-form-item>
                <el-row :gutter="12">
                  <el-col :span="8">
                    <el-form-item :label="$t('audio.bpm')">
                      <el-input-number v-model="params.bpm" :min="30" :max="300" controls-position="right" class="studio-ep-w-full" :placeholder="$t('audio.bpmAuto')" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item :label="$t('audio.keyScale')">
                      <el-select v-model="params.key_scale" class="studio-ep-w-full" clearable :placeholder="$t('audio.keyScaleAuto')">
                        <el-option v-for="k in musicalKeys" :key="k" :label="k" :value="k" />
                      </el-select>
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item :label="$t('audio.timeSignature')">
                      <el-select v-model="params.time_signature" class="studio-ep-w-full" clearable :placeholder="$t('audio.timeSignatureAuto')">
                        <el-option v-for="ts in timeSignatures" :key="ts.value" :label="ts.label" :value="ts.value" />
                      </el-select>
                    </el-form-item>
                  </el-col>
                </el-row>
              </el-form>
            </el-card>

            <!-- Advanced params -->
            <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
              <el-collapse v-model="advancedOpen" class="studio-ep-collapse-plain">
                <el-collapse-item name="advanced">
                  <template #title>
                    <div class="studio-ep-collapse-title-row">
                      <el-icon><setting /></el-icon>
                      <span>{{ $t('studio.advancedParams') }}</span>
                      <el-tag v-if="hasCustomParams" size="small" type="warning">{{ $t('studio.hasCustom') }}</el-tag>
                    </div>
                  </template>
                  <el-form label-position="top" size="small" class="studio-ep-form-pt">
                    <el-form-item v-if="currentModelConfig?.parameters?.steps" :label="$t('audio.sampleQuality')">
                      <div class="param-control-row">
                        <div class="param-slider">
                          <el-slider
                            v-model="params.steps"
                            :min="currentModelConfig.parameters.steps.min"
                            :max="currentModelConfig.parameters.steps.max"
                          />
                        </div>
                        <el-input-number v-model="params.steps" :min="1" :max="200" class="param-input-number" />
                      </div>
                    </el-form-item>
                    <el-form-item v-if="currentModelConfig?.parameters?.guidance" :label="$t('audio.guidance')">
                      <div class="param-control-row">
                        <div class="param-slider">
                          <el-slider
                            v-model="params.guidance"
                            :min="currentModelConfig.parameters.guidance.min"
                            :max="currentModelConfig.parameters.guidance.max"
                            :step="0.5"
                          />
                        </div>
                        <el-input-number v-model="params.guidance" :step="0.5" class="param-input-number" />
                      </div>
                    </el-form-item>
                    <el-form-item :label="$t('audio.seed')">
                      <div class="studio-ep-seed-row">
                        <el-input v-model="params.seed" :placeholder="$t('audio.randomSeed')" />
                        <el-button @click="randomizeSeed">
                          <el-icon><refresh /></el-icon>
                        </el-button>
                      </div>
                    </el-form-item>
                    <el-form-item :label="$t('audio.batchCount')">
                      <el-input-number v-model="params.n" :min="1" :max="8" controls-position="right" class="studio-ep-w-full" />
                    </el-form-item>
                    <el-form-item :label="$t('audio.audioFormat')">
                      <el-select v-model="params.audio_format" class="studio-ep-w-full">
                        <el-option v-for="f in audioFormats" :key="f" :label="f" :value="f" />
                      </el-select>
                    </el-form-item>
                    <el-form-item>
                      <el-button text type="primary" @click="restoreDefaults" size="small">
                        <el-icon><refresh /></el-icon>
                        {{ $t('studio.restoreDefaults') }}
                      </el-button>
                    </el-form-item>
                  </el-form>
                </el-collapse-item>
              </el-collapse>
            </el-card>
          </template>

          <!-- 同声翻唱 占位 -->
          <template v-if="audioWorkTab === 'cover'">
            <el-card shadow="never" class="studio-ep-surface-card studio-ep-cover-placeholder">
              <el-icon class="studio-ep-cover-placeholder-icon" :size="40"><clock /></el-icon>
              <p>{{ $t('audio.coverComingSoon') }}</p>
            </el-card>
          </template>

          <!-- Generate button -->
          <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <el-button
              v-if="audioWorkTab === 'create'"
              type="primary"
              size="large"
              class="studio-ep-primary-cta studio-ep-primary-cta--simple"
              :disabled="submitDisabled"
              @click="startGeneration"
            >
              <el-icon size="20"><headset /></el-icon>
              <span class="studio-ep-cta-gap">
                {{ generating ? $t('audio.generating') : $t('audio.generate') }}
              </span>
            </el-button>
            <div class="studio-ep-micro-hint">
              {{ $sendShortcutHint() }}
            </div>

            <!-- Progress display -->
            <div v-if="currentTask.id" class="studio-ep-task-wrap">
              <el-progress
                :percentage="Math.round(currentTask.progress * 100)"
                :status="currentTask.status === 'failed' ? 'exception' : ''"
              />
              <div class="studio-ep-task-status">
                <template v-if="currentTask.total > 0 && currentTask.status === 'running'">
                  Step {{ currentTask.step }}/{{ currentTask.total }} &nbsp;
                </template>
                <el-tag :type="TSU.tagType(currentTask.status)" size="small">
                  {{ TSU.statusText(currentTask.status) }}
                </el-tag>
              </div>
            </div>
          </el-card>

          <!-- Logs -->
          <el-card shadow="never" class="studio-ep-surface-card">
            <template #header>
              <div class="card-title card-title--split">
                <span>
                  <el-icon><document /></el-icon>
                  {{ $t('studio.logs') }}
                </span>
                <el-button size="small" text @click="clearLogs">
                  <el-icon><delete /></el-icon>
                </el-button>
              </div>
            </template>
            <div class="log-container studio-ep-log-container--sm" ref="logContainer">
              <div v-if="logs.length === 0" class="studio-ep-log-empty">
                {{ $t('studio.logsEmpty') }}
              </div>
              <div v-for="(log, index) in logs" :key="index" class="log-line">
                <span class="log-timestamp">{{ log.time }}</span>
                <span :class="'log-' + log.level">{{ log.message }}</span>
              </div>
            </div>
          </el-card>

        </div>
      </el-col>

      <!-- Right panel: preview + recent -->
      <el-col :xs="24" :md="8" :lg="7" :xl="6">
        <div class="preview-panel">

          <!-- Current generation preview -->
          <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <template #header>
              <div class="card-title">
                <el-icon><headset /></el-icon>
                {{ $t('studio.currentPreview') }}
              </div>
            </template>
            <div v-if="previewAudioSrc" class="dq-audio-create-cover">
              <el-icon :size="34"><headset /></el-icon>
              <span class="dq-audio-create-cover-title">{{ $t('audio.previewListen') }}</span>
              <audio
                :key="previewAudioKey"
                ref="previewAudioEl"
                :src="previewAudioSrc"
                controls
                playsinline
                preload="metadata"
              ></audio>
              <div class="studio-ep-audio-caption">
                {{ previewPrompt }}
              </div>
            </div>
            <el-empty v-else :description="$t('studio.noPreview')" />
          </el-card>

          <!-- Recent generations -->
          <el-card shadow="never" class="studio-ep-surface-card">
            <template #header>
              <div class="card-title card-title--split">
                <span>
                  <el-icon><clock /></el-icon>
                  {{ $t('audio.recentTitle') }}
                </span>
                <el-button size="small" text @click="loadRecentAudio">
                  <el-icon><refresh /></el-icon>
                </el-button>
              </div>
            </template>
            <el-empty v-if="recentAudio.length === 0" :description="$t('gallery.empty')" />
            <div v-else>
              <div
                v-for="ra in recentAudio"
                :key="ra.id"
                class="gallery-card dq-audio-recent-card studio-ep-recent-card-mb"
                @click="previewAudio(ra)"
              >
                <div class="studio-ep-audio-recent-row">
                  <div class="dq-audio-recent-cover" @click.stop="previewAudio(ra)">
                    <el-icon :size="26"><headset /></el-icon>
                  </div>
                  <div class="studio-ep-audio-recent-main">
                    <div class="studio-ep-audio-recent-title">
                      {{ ra.prompt || ra.name || 'Audio' }}
                    </div>
                    <div class="studio-ep-audio-recent-meta">
                      {{ ra.duration ? formatTime(ra.duration) : '' }}
                    </div>
                  </div>
                  <div class="studio-ep-audio-recent-actions">
                    <el-button circle size="small" @click.stop="toggleRecentPlay(ra)">
                      <span class="studio-ep-play-icon">{{ ra.playing ? '⏸' : '▶' }}</span>
                    </el-button>
                    <el-button size="small" text @click.stop="downloadAudioUrl(ra.url)">
                      <el-icon><download /></el-icon>
                    </el-button>
                  </div>
                </div>
                <audio
                  :ref="(el: any) => setRecentAudioRef(el, ra.id)"
                  :src="ra.url"
                  preload="metadata"
                  @ended="ra.playing = false"
                  @loadedmetadata="onRecentMeta(ra, $event)"
                  class="studio-ep-hidden-audio"
                />
              </div>
            </div>
          </el-card>

        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, watch, onMounted, nextTick } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { formatGenLogMessage, isDuplicateDenoiseStepLog } from '@/utils/genTaskLog';
import { $tt } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';

const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const router = useRouter();

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

function getModelName(c: any, mk: string) {
  return c?.name?.zh || c?.name?.en || mk;
}

function getModelVersionName(mid: string, vk: string, vd: any) {
  return vd?.name ? `${mid} ${vd.name}` : `${mid} ${vk}`;
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
const advancedOpen = ref<string[]>(['advanced']);
const logContainer = ref<HTMLElement | null>(null);
const previewAudioEl = ref<HTMLAudioElement | null>(null);

const modelRegistry = ref<Record<string, any>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});

const params = reactive({
  model: '',
  prompt: '',
  negative_prompt: '',
  duration: 60,
  instrumental: false,
  lyrics: '',
  vocal_language: '',
  bpm: null as number | null,
  key_scale: '',
  time_signature: '',
  steps: 8,
  guidance: 7.0,
  seed: null as number | null,
  n: 2,
  audio_format: 'wav',
});

const currentTask = reactive({ id: '', progress: 0, step: null as number | null, total: null as number | null, status: '' });
const logs = reactive<Array<{ level: string; message: string; time: string }>>([]);
const previewAudioSrc = ref('');
const previewPrompt = ref('');
const previewAudioKey = ref(0);
const recentAudio = reactive<Array<any>>([]);
const recentAudioElById: Record<string, HTMLAudioElement> = {};

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
  return getModelName(c, mk) || params.model || '';
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
  return params.steps !== 8 || params.guidance !== 7.0 || params.seed !== null
    || params.n !== 2 || params.audio_format !== 'wav';
});

// ---- Audio formats ----
const audioFormats = ['mp3', 'flac', 'wav', 'opus', 'aac'];
const vocalLanguages = [
  { label: 'English', value: 'en' }, { label: '中文', value: 'zh' }, { label: '日本語', value: 'ja' },
  { label: '한국어', value: 'ko' }, { label: 'Français', value: 'fr' }, { label: 'Deutsch', value: 'de' },
  { label: 'Español', value: 'es' }, { label: 'Português', value: 'pt' },
];
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

const filteredModelPickerVersions = computed(() => {
  const rows: Array<any> = [];
  for (const [mid, config] of Object.entries(modelRegistry.value)) {
    if (!isAudioModel(config)) continue;
    const act = audioWorkTab.value === 'cover' ? 'cover' : 'create';
    if (act === 'cover' && !supportsAction(config.actions, 'cover')) continue;
    if (act === 'create' && !supportsAction(config.actions, 'create')) continue;
    const versions = config.versions || {};
    const verKeys = Object.keys(versions);
    const isRec = config.recommended === true;
    const ds = modelsDetailedStatus.value[mid] || {};
    const versionStatuses = ds.versions || {};
    for (const vk of verKeys) {
      const vd = versions[vk];
      const vst = versionStatuses[vk] || {};
      const ready = vst.ready === true || (ds.status === 'ready' && vst.ready !== false);
      const label = getModelVersionName(mid, vk, vd);
      rows.push({ key: mid + '|' + vk, label, ready, isRec, mid, vk, size: vd.size || '' });
    }
  }
  rows.sort((a, b) => {
    if (a.isRec !== b.isRec) return a.isRec ? -1 : 1;
    if (a.ready !== b.ready) return a.ready ? -1 : 1;
    return a.label.localeCompare(b.label);
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

// ---- Methods ----
async function loadModelRegistry() {
  modelsLoading.value = true;
  try {
    const data = await registryStore.getRegistry();
    modelRegistry.value = data?.models || {};
  } catch (e) { /* ignore */ }
  try {
    const statusData = await registryStore.getModelsDetailedStatus?.() || await api.settings.getModelsDetailedStatus();
    if (statusData && typeof statusData === 'object') {
      modelsDetailedStatus.value = statusData;
    }
  } catch (e) { /* ignore */ }
  modelsLoading.value = false;
}

function applyDefaults(modelConfig: any) {
  if (!modelConfig || !modelConfig.parameters) return;
  const p = modelConfig.parameters;
  if (p.steps && p.steps.default != null) params.steps = p.steps.default;
  if (p.guidance && p.guidance.default != null) params.guidance = p.guidance.default;
  if (p.duration && p.duration.default != null) params.duration = p.duration.default;
  if (p.audio_formats && Array.isArray(p.audio_formats) && p.audio_formats.length > 0) {
    params.audio_format = p.audio_formats[0];
  }
  params.negative_prompt = '';
  params.lyrics = '';
  params.bpm = null;
  params.key_scale = '';
  params.time_signature = '';
  params.vocal_language = '';
}

function onModelChange(val: string) {
  const parsed = parseModelVersion(val);
  const mk = parsed.modelKey || parsed.model || '';
  const vk = parsed.version || '';
  params.model = vk ? mk + ':' + vk : mk;
  const mc = modelRegistry.value[mk];
  applyDefaults(mc);
  loadPromptDraft();
}

function randomizeSeed() {
  params.seed = Math.floor(Math.random() * 2147483647);
}

function restoreDefaults() {
  const mc = currentModelConfig.value;
  applyDefaults(mc);
  params.guidance = 7.0;
  params.n = 2;
  params.audio_format = 'wav';
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

function clearLogs() {
  logs.length = 0;
}

async function startGeneration() {
  if (generating.value) return;
  if (!params.model) { ElMessage.warning('Please select a model'); return; }
  if (!modelReady.value) { ElMessage.warning('Model is not ready'); return; }
  if (!params.prompt.trim()) { ElMessage.warning('Please enter a prompt'); return; }

  generating.value = true;
  currentTask.id = '';
  currentTask.progress = 0;
  currentTask.step = null;
  currentTask.total = null;
  currentTask.status = '';
  logs.length = 0;

  try {
    const body = {
      model: params.model,
      prompt: params.prompt,
      negative_prompt: params.negative_prompt || '',
      duration: params.duration || null,
      instrumental: params.instrumental || false,
      lyrics: params.lyrics || '',
      vocal_language: params.vocal_language || '',
      bpm: params.bpm || null,
      key_scale: params.key_scale || '',
      time_signature: params.time_signature || '',
      steps: params.steps || null,
      guidance: params.guidance || null,
      seed: params.seed || null,
      n: params.n || 2,
      audio_format: params.audio_format || 'wav',
    };

    const resp = await api.audios.createGeneration(body);
    const tid = taskIdFromSubmitResponse(resp);
    if (!tid) {
      ElMessage.error($tt('studio.error', { msg: 'missing task id in submit response' }));
      generating.value = false;
      return;
    }
    currentTask.id = tid;
    if (!api.gen?.streamMediaTask) {
      generating.value = false;
      return;
    }

    const pushPreviewFromAsset = (aid: string) => {
      const url = api.gallery?.getImageUrl
        ? api.gallery.getImageUrl('asset:' + aid)
        : '/api/assets/' + aid + '/file';
      recentAudio.unshift({ id: aid, url, prompt: params.prompt, duration: 0, playing: false });
      previewAudioSrc.value = url;
      previewPrompt.value = params.prompt;
      previewAudioKey.value += 1;
      nextTick(() => {
        try {
          previewAudioEl.value?.load?.();
        } catch {
          /* ignore */
        }
      });
    };

    api.gen.streamMediaTask(tid, {
      onLog: (logData: unknown) => {
        const row = logData as Record<string, unknown>;
        const raw = (row.message as string) || '';
        const lvl = (row.level as string) || 'info';
        if (isDuplicateDenoiseStepLog(logs, raw)) {
          return;
        }
        const now = new Date();
        const time =
          String(now.getHours()).padStart(2, '0') +
          ':' +
          String(now.getMinutes()).padStart(2, '0') +
          ':' +
          String(now.getSeconds()).padStart(2, '0');
        logs.push({
          level: lvl,
          message: formatGenLogMessage(raw),
          time,
        });
        nextTick(() => {
          if (logContainer.value) {
            logContainer.value.scrollTop = logContainer.value.scrollHeight;
          }
        });
      },
      onStatus: (statusData: unknown) => {
        const row = statusData as Record<string, unknown>;
        if (row.status) currentTask.status = row.status as string;
        if (row.progress != null) currentTask.progress = row.progress as number;
      },
      onProgress: (progressData: unknown) => {
        const row = progressData as Record<string, unknown>;
        if (row.progress != null) currentTask.progress = row.progress as number;
        if (row.step != null) currentTask.step = row.step as number;
        if (row.total != null) currentTask.total = row.total as number;
      },
      onResult: (resultData: unknown) => {
        const row = resultData as Record<string, unknown>;
        const ids = row.asset_ids as string[] | undefined;
        if (ids?.length) {
          ids.forEach((aid) => pushPreviewFromAsset(aid));
        }
      },
      onDone: (doneData: unknown) => {
        const row = doneData as Record<string, unknown>;
        if (row.status === 'completed') {
          const pid =
            row.result &&
            (row.result as Record<string, unknown>).primary_asset_id;
          if (pid) pushPreviewFromAsset(String(pid));
        } else if (row.status === 'failed') {
          ElMessage.error($tt('studio.genFailed', { msg: String(row.error || '') }));
        }
        generating.value = false;
      },
      onError: () => {
        ElMessage.error($tt('studio.connectionLost'));
        generating.value = false;
      },
    });
  } catch (e: any) {
    ElMessage.error((e.response && e.response.data && e.response.data.detail) || e.message || 'Generation failed');
    generating.value = false;
  }
}

function setRecentAudioRef(el: any, id: string) {
  if (!id) return;
  if (el) recentAudioElById[id] = el;
  else delete recentAudioElById[id];
}

function previewAudio(item: any) {
  previewAudioSrc.value = item.url;
  previewPrompt.value = item.prompt || '';
  previewAudioKey.value += 1;
  nextTick(() => {
    try {
      previewAudioEl.value?.load?.();
    } catch (e) { /* ignore */ }
  });
}

function toggleRecentPlay(ra: any) {
  const el = recentAudioElById[ra.id];
  if (!el) return;
  if (ra.playing) {
    el.pause();
    ra.playing = false;
  } else {
    recentAudio.forEach((x: any) => {
      if (x.id !== ra.id && x.playing) {
        const o = recentAudioElById[x.id];
        if (o) {
          try { o.pause(); } catch (e) { /* ignore */ }
        }
        x.playing = false;
      }
    });
    el.play().catch(() => {});
    ra.playing = true;
  }
}

function onRecentMeta(ra: any, event: Event) {
  const el = event.target as HTMLAudioElement;
  if (el && el.duration && Number.isFinite(el.duration)) {
    ra.duration = el.duration;
  }
}

function downloadAudioUrl(url: string) {
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = 'audio.' + (params.audio_format || 'wav');
  a.click();
}

function loadRecentAudio() {
  api.gen.listAssets('audio', 20, 0).then((data: any) => {
    const items = (data && data.items) || [];
    recentAudio.length = 0;
    Object.keys(recentAudioElById).forEach((k) => delete recentAudioElById[k]);
    items.forEach((item: any) => {
      const aid = item.id || '';
      if (!aid) return;
      const url = api.gallery?.getImageUrl ? api.gallery.getImageUrl('asset:' + aid) : ('/api/assets/' + aid + '/file');
      const meta = item.metadata || {};
      recentAudio.push({
        id: aid,
        url,
        prompt: String(meta.prompt || ''),
        name: item.name || '',
        duration: item.duration_seconds != null ? Number(item.duration_seconds) : (meta.duration_seconds != null ? Number(meta.duration_seconds) : 0),
        playing: false,
      });
    });
  }).catch(() => {});
}

// ---- Lifecycle ----
onMounted(async () => {
  await loadModelRegistry();
  const versions = filteredModelPickerVersions.value;
  if (versions.length > 0) {
    const rec = versions.find((v: any) => v.ready && v.isRec) || versions.find((v: any) => v.ready) || versions[0];
    selectedModelVersion.value = rec.key;
    onModelChange(rec.key);
  }
  if (!params.model) {
    loadPromptDraft();
  }
  loadRecentAudio();
});

let _draftTimer: ReturnType<typeof setTimeout> | null = null;
watch(() => params.prompt, () => {
  if (_draftTimer) clearTimeout(_draftTimer);
  _draftTimer = setTimeout(savePromptDraft, 500);
});

// Refresh model picker when tab changes
watch(audioWorkTab, () => {
  if (selectedModelVersion.value) {
    onModelChange(selectedModelVersion.value);
  }
});
</script>
