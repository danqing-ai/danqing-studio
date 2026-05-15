<!-- @ts-nocheck -->
<template>
  <div class="create-page">
    <el-row :gutter="24">
      <!-- Left panel: creation area -->
      <el-col :xs="24" :md="16" :lg="14">
        <div class="creation-panel">

          <!-- Tab bar -->
          <div class="mode-segment" style="margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 4px;">
            <div
              class="mode-segment-item"
              :class="{ active: audioWorkTab === 'create' }"
              @click="setAudioWorkMode('create')"
            >
              <el-icon><headset /></el-icon>
              <span>{{ $t('action.audio.create') }}</span>
            </div>
            <div
              class="mode-segment-item"
              :class="{ active: audioWorkTab === 'cover' }"
              @click="setAudioWorkMode('cover')"
            >
              <el-icon><switch /></el-icon>
              <span>{{ $t('action.audio.cover') }}</span>
            </div>
          </div>

          <!-- Model selector -->
          <div class="card" style="margin-bottom: 16px;">
            <div class="card-title">
              <el-icon><cpu /></el-icon>
              {{ $t('create.modelSelectTitle') }}
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
              <el-select
                v-model="selectedModelVersion"
                style="flex: 1;"
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
                  <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span :style="!mv.ready ? 'opacity: 0.5;' : ''">{{ mv.label }}</span>
                    <el-tag v-if="mv.isRec" size="small" type="success">{{ $t('studio.recommended') }}</el-tag>
                    <el-tag v-if="mv.ready" size="small" type="success">{{ $t('studio.ready') }}</el-tag>
                    <el-tag v-else size="small" type="warning">{{ $t('studio.notDownloaded') }}</el-tag>
                    <span v-if="mv.size" style="color: var(--text-muted); font-size: 12px; margin-left: auto;">{{ mv.size }}</span>
                  </div>
                </el-option>
              </el-select>
            </div>
            <el-alert
              v-if="selectedModelNotReady"
              :title="$t('studio.modelNotReady', { name: currentModelDisplayName })"
              type="warning"
              :closable="false"
              style="margin-top: 12px;"
            >
              <template #default>
                <span>{{ $t('studio.notDownloadedMsg') }}</span>
                <el-button size="small" type="primary" @click="goToDownload" style="margin-left: 12px;">
                  {{ $t('studio.goDownload') }}
                </el-button>
              </template>
            </el-alert>
          </div>

          <!-- 文生音乐 Tab -->
          <template v-if="audioWorkTab === 'create'">

            <!-- Prompt input -->
            <div class="card" style="margin-bottom: 16px;">
              <div class="card-title">
                <el-icon><edit-pen /></el-icon>
                {{ $t('studio.prompt') }}
              </div>
              <el-input
                v-model="params.prompt"
                type="textarea"
                :rows="4"
                :placeholder="$t('audio.promptPlaceholder')"
                resize="none"
                @keydown.meta.enter.prevent="startGeneration"
                @keydown.ctrl.enter.prevent="startGeneration"
              />
              <el-collapse v-if="supportsNegativePrompt" style="margin-top: 12px; border: none;">
                <el-collapse-item :title="$t('audio.negativePrompt')" name="negative">
                  <el-input
                    v-model="params.negative_prompt"
                    type="textarea"
                    :rows="2"
                    :placeholder="$t('studio.optional')"
                  />
                </el-collapse-item>
              </el-collapse>
            </div>

            <!-- Lyrics -->
            <div class="card" style="margin-bottom: 16px;">
              <div class="card-title">
                <el-icon><document /></el-icon>
                {{ $t('audio.lyrics') }}
                <span style="color: var(--text-muted); font-size: 12px; font-weight: 400; margin-left: 4px;">{{ $t('studio.optional') }}</span>
              </div>
              <el-input
                v-model="params.lyrics"
                type="textarea"
                :rows="4"
                :placeholder="$t('audio.lyricsPlaceholder')"
                resize="none"
              />
              <div style="display: flex; align-items: center; gap: 20px; margin-top: 12px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span style="font-size: 13px; color: var(--text-secondary);">{{ $t('audio.instrumental') }}</span>
                  <el-switch v-model="params.instrumental" size="small" />
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <span style="font-size: 13px; color: var(--text-secondary);">{{ $t('audio.vocalLanguage') }}</span>
                  <el-select v-model="params.vocal_language" size="small" style="width: 140px;" clearable :placeholder="$t('audio.vocalLanguageAuto')">
                    <el-option v-for="l in vocalLanguages" :key="l.value" :label="l.label" :value="l.value" />
                  </el-select>
                </div>
              </div>
            </div>

            <!-- Music params -->
            <div class="card" style="margin-bottom: 16px;">
              <div class="card-title">
                <el-icon><setting /></el-icon>
                {{ $t('audio.musicParams') }}
              </div>
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
                      <el-input-number v-model="params.bpm" :min="30" :max="300" controls-position="right" style="width: 100%;" :placeholder="$t('audio.bpmAuto')" />
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item :label="$t('audio.keyScale')">
                      <el-select v-model="params.key_scale" style="width: 100%;" clearable :placeholder="$t('audio.keyScaleAuto')">
                        <el-option v-for="k in musicalKeys" :key="k" :label="k" :value="k" />
                      </el-select>
                    </el-form-item>
                  </el-col>
                  <el-col :span="8">
                    <el-form-item :label="$t('audio.timeSignature')">
                      <el-select v-model="params.time_signature" style="width: 100%;" clearable :placeholder="$t('audio.timeSignatureAuto')">
                        <el-option v-for="ts in timeSignatures" :key="ts.value" :label="ts.label" :value="ts.value" />
                      </el-select>
                    </el-form-item>
                  </el-col>
                </el-row>
              </el-form>
            </div>

            <!-- Advanced params -->
            <div class="card" style="margin-bottom: 16px;">
              <el-collapse v-model="advancedOpen" style="border: none;">
                <el-collapse-item name="advanced">
                  <template #title>
                    <div style="display: flex; align-items: center; gap: 8px; font-weight: 500;">
                      <el-icon><setting /></el-icon>
                      <span>{{ $t('studio.advancedParams') }}</span>
                      <el-tag v-if="hasCustomParams" size="small" type="warning">{{ $t('studio.hasCustom') }}</el-tag>
                    </div>
                  </template>
                  <el-form label-position="top" size="small" style="padding-top: 12px;">
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
                      <div style="display: flex; gap: 8px;">
                        <el-input v-model="params.seed" :placeholder="$t('audio.randomSeed')" style="flex: 1;" />
                        <el-button @click="randomizeSeed">
                          <el-icon><refresh /></el-icon>
                        </el-button>
                      </div>
                    </el-form-item>
                    <el-form-item :label="$t('audio.batchCount')">
                      <el-input-number v-model="params.n" :min="1" :max="8" controls-position="right" style="width: 100%;" />
                    </el-form-item>
                    <el-form-item :label="$t('audio.audioFormat')">
                      <el-select v-model="params.audio_format" style="width: 100%;">
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
            </div>
          </template>

          <!-- 同声翻唱 占位 -->
          <template v-if="audioWorkTab === 'cover'">
            <div class="card" style="padding: 40px 20px; text-align: center;">
              <el-icon :size="40" color="var(--text-muted)"><clock /></el-icon>
              <p style="margin-top: 16px; color: var(--text-muted); font-size: 14px;">{{ $t('audio.coverComingSoon') }}</p>
            </div>
          </template>

          <!-- Generate button -->
          <div class="card" style="margin-bottom: 16px;">
            <el-button
              v-if="audioWorkTab === 'create'"
              type="primary"
              size="large"
              style="width: 100%; height: 50px; font-size: 16px;"
              :disabled="submitDisabled"
              @click="startGeneration"
            >
              <el-icon size="20"><headset /></el-icon>
              <span style="margin-left: 8px;">
                {{ generating ? $t('audio.generating') : $t('audio.generate') }}
              </span>
            </el-button>
            <div style="margin-top: 8px; font-size: 11px; color: var(--text-muted);">
              {{ $t('studio.sendShortcutHint') }}
            </div>

            <!-- Progress display -->
            <div v-if="currentTask.id" style="margin-top: 16px;">
              <el-progress
                :percentage="Math.round(currentTask.progress * 100)"
                :status="currentTask.status === 'failed' ? 'exception' : ''"
              />
              <div style="margin-top: 8px; text-align: center; color: var(--text-muted); font-size: 13px;">
                <template v-if="currentTask.total > 0 && currentTask.status === 'running'">
                  Step {{ currentTask.step }}/{{ currentTask.total }} &nbsp;
                </template>
                <el-tag :type="TSU.tagType(currentTask.status)" size="small">
                  {{ TSU.statusText(currentTask.status) }}
                </el-tag>
              </div>
            </div>
          </div>

          <!-- Logs -->
          <div class="card">
            <div class="card-title" style="justify-content: space-between;">
              <span>
                <el-icon><document /></el-icon>
                {{ $t('studio.logs') }}
              </span>
              <el-button size="small" text @click="clearLogs">
                <el-icon><delete /></el-icon>
              </el-button>
            </div>
            <div class="log-container" ref="logContainer" style="max-height: 200px;">
              <div v-if="logs.length === 0" style="text-align: center; color: var(--text-muted); padding: 20px;">
                {{ $t('studio.logsEmpty') }}
              </div>
              <div v-for="(log, index) in logs" :key="index" class="log-line">
                <span class="log-timestamp">{{ log.time }}</span>
                <span :class="'log-' + log.level">{{ log.message }}</span>
              </div>
            </div>
          </div>

        </div>
      </el-col>

      <!-- Right panel: preview + recent -->
      <el-col :xs="24" :md="8" :lg="10">
        <div class="preview-panel">

          <!-- Current generation preview -->
          <div class="card" style="margin-bottom: 16px;">
            <div class="card-title">
              <el-icon><headset /></el-icon>
              {{ $t('studio.currentPreview') }}
            </div>
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
              <div style="margin-top: 4px; font-size: 12px; color: rgba(255,255,255,0.55); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; width: 100%; text-align: center;">
                {{ previewPrompt }}
              </div>
            </div>
            <el-empty v-else :description="$t('studio.noPreview')" />
          </div>

          <!-- Recent generations -->
          <div class="card">
            <div class="card-title" style="justify-content: space-between;">
              <span>
                <el-icon><clock /></el-icon>
                {{ $t('audio.recentTitle') }}
              </span>
              <el-button size="small" text @click="loadRecentAudio">
                <el-icon><refresh /></el-icon>
              </el-button>
            </div>
            <el-empty v-if="recentAudio.length === 0" :description="$t('gallery.empty')" />
            <div v-else>
              <div
                v-for="ra in recentAudio"
                :key="ra.id"
                class="gallery-card dq-audio-recent-card"
                style="margin-bottom: 10px; cursor: pointer;"
                @click="previewAudio(ra)"
              >
                <div style="display: flex; align-items: stretch; gap: 10px; padding: 10px;">
                  <div class="dq-audio-recent-cover" @click.stop="previewAudio(ra)">
                    <el-icon :size="26"><headset /></el-icon>
                  </div>
                  <div style="flex: 1; min-width: 0; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                      {{ ra.prompt || ra.name || 'Audio' }}
                    </div>
                    <div style="font-size: 11px; color: var(--text-muted); margin-top: 4px;">
                      {{ ra.duration ? formatTime(ra.duration) : '' }}
                    </div>
                  </div>
                  <div style="display: flex; flex-direction: column; justify-content: center; gap: 6px;">
                    <el-button circle size="small" @click.stop="toggleRecentPlay(ra)">
                      <span style="font-size: 13px;">{{ ra.playing ? '⏸' : '▶' }}</span>
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
                  style="display: none;"
                />
              </div>
            </div>
          </div>

        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, watch, onMounted, nextTick } from 'vue';
import { ElMessage } from 'element-plus';
import api from '@/utils/api';
import { $tt } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';

const tasksStore = useTasksStore();
const registryStore = useRegistryStore();

// ---- Helpers (migrated from window globals) ----
const STORAGE_KEY = 'dq-studio.audio-create-prompt-draft.v3';

function parseModelVersion(s: string) {
  const [m, v] = (s || '').split('|');
  return { model: m || '', version: v || '', modelKey: m || '' };
}

function isAudioModel(config: any) {
  return config && config.media === 'audio';
}

function supportsAction(actions: string[], action: string) {
  return Array.isArray(actions) && actions.includes(action);
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

function setAudioWorkMode(mode: string) {
  audioWorkTab.value = mode;
}

// ---- State ----
const generating = ref(false);
const modelsLoading = ref(false);
const advancedOpen = ref<string[]>([]);
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
    const ready = ds.status === 'ready';
    for (const vk of verKeys) {
      const vd = versions[vk];
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
  (window as any).DQStudioNav?.goModels?.();
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

    currentTask.id = (resp && resp.task && resp.task.id) || '';
    if (currentTask.id && api.gen?.streamMediaTask) {
      await api.gen.streamMediaTask(
        currentTask.id,
        (logData: any) => {
          const now = new Date();
          const time = String(now.getHours()).padStart(2, '0') + ':' + String(now.getMinutes()).padStart(2, '0') + ':' + String(now.getSeconds()).padStart(2, '0');
          logs.push({ level: logData.level || 'info', message: logData.message || '', time });
          nextTick(() => {
            if (logContainer.value) {
              logContainer.value.scrollTop = logContainer.value.scrollHeight;
            }
          });
        },
        (statusData: any) => {
          if (statusData.status) currentTask.status = statusData.status;
          if (statusData.progress != null) currentTask.progress = statusData.progress;
        },
        (doneData: any) => {
          if (doneData.status === 'completed') {
            const pid = doneData.result && doneData.result.primary_asset_id;
            if (pid) {
              const url = api.gallery?.getImageUrl ? api.gallery.getImageUrl('asset:' + pid) : ('/api/assets/' + pid + '/file');
              recentAudio.unshift({ id: pid, url, prompt: params.prompt, duration: 0, playing: false });
              previewAudioSrc.value = url;
              previewPrompt.value = params.prompt;
              previewAudioKey.value += 1;
              nextTick(() => {
                try {
                  previewAudioEl.value?.load?.();
                } catch (e) { /* ignore */ }
              });
            }
          }
          generating.value = false;
        },
        () => {
          ElMessage.error('Generation failed');
          generating.value = false;
        },
        (progressData: any) => {
          if (progressData.progress != null) currentTask.progress = progressData.progress;
          if (progressData.step != null) currentTask.step = progressData.step;
          if (progressData.total != null) currentTask.total = progressData.total;
        },
        (resultData: any) => {
          if (resultData && resultData.asset_ids) {
            resultData.asset_ids.forEach((aid: string) => {
              const url = api.gallery?.getImageUrl ? api.gallery.getImageUrl('asset:' + aid) : ('/api/assets/' + aid + '/file');
              recentAudio.unshift({ id: aid, url, prompt: params.prompt, duration: 0, playing: false });
              previewAudioSrc.value = url;
              previewPrompt.value = params.prompt;
              previewAudioKey.value += 1;
              nextTick(() => {
                try {
                  previewAudioEl.value?.load?.();
                } catch (e) { /* ignore */ }
              });
            });
          }
        },
      );
    } else {
      generating.value = false;
    }
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
