<!-- @ts-nocheck -->
<template>
  <div class="create-page">
    <DqRow :gutter="24">
      <!-- Left panel: creation area -->
      <DqCol :xs="24" :md="16" :lg="17" :xl="18">
        <div class="creation-panel">

          <!-- Tab bar -->
          <DqSegmented
            class="dq-work-segmented dq-work-segmented--sm"
            v-model="audioWorkTab"
            :options="audioWorkSegmentOptions"
            block
          />

          <!-- Model selector -->
          <DqSurfaceCard class="studio-surface-card studio-card-mb studio-model-card">
            <template #header>
              <div class="card-title">
                <DqIcon><cpu /></DqIcon>
                {{ $t('create.modelSelectTitle') }}
              </div>
            </template>
            <div class="studio-model-toolbar">
              <DqSelect
                v-model="selectedModelVersion"
                filterable
                @change="onModelChange"
                :placeholder="$t('studio.selectModel')"
              >
                <DqOption
                  v-for="mv in filteredModelPickerVersions"
                  :key="mv.key"
                  :label="mv.label"
                  :value="mv.key"
                  :disabled="!mv.ready"
                >
                  <div class="studio-picker-option">
                    <span class="studio-picker-option__name" :class="{ 'is-disabled': !mv.ready }">{{ mv.label }}</span>
                    <ModelLicenseBadges
                      :recommended="mv.isRec"
                      :commercial-use-allowed="mv.commercialUseAllowed"
                      effect="plain"
                    />
                    <DqTag v-if="mv.ready" size="small" type="success">{{ $t('studio.ready') }}</DqTag>
                    <DqTag v-else size="small" type="warning">{{ $t('studio.notDownloaded') }}</DqTag>
                    <span v-if="mv.size" class="studio-picker-option__meta">{{ mv.size }}</span>
                  </div>
                </DqOption>
              </DqSelect>
              <ModelPickerFilters
                v-model:commercial-only="modelFilterCommercialOnly"
                :show-installed-filter="false"
              />
            </div>
            <DqAlert
              v-if="selectedModelNotReady"
              :title="$t('studio.modelNotReady', { name: currentModelDisplayName })"
              type="warning"
              :closable="false"
              class="studio-alert-mt"
            >
              <template #default>
                <span>{{ $t('studio.notDownloadedMsg') }}</span>
                <DqButton type="primary" size="sm" class="studio-alert-inline-btn" @click="goToDownload">
                  {{ $t('studio.goDownload') }}
                </DqButton>
              </template>
            </DqAlert>
          </DqSurfaceCard>
          <template v-if="audioWorkTab === 'create'">

            <!-- Prompt input -->
            <DqSurfaceCard class="studio-surface-card studio-card-mb">
              <template #header>
                <div class="card-title">
                  <DqIcon><edit-pen /></DqIcon>
                  {{ $t('studio.prompt') }}
                </div>
              </template>
              <DqInput
                v-model="params.prompt"
                type="textarea"
                :rows="4"
                :placeholder="$t('audio.promptPlaceholder')"
                resize="none"
                @keydown.meta.enter.prevent="startGeneration"
                @keydown.ctrl.enter.prevent="startGeneration"
              />
              <DqCollapse v-if="supportsNegativePrompt" class="studio-collapse-plain">
                <DqCollapseItem :title="$t('audio.negativePrompt')" name="negative">
                  <DqInput
                    v-model="params.negative_prompt"
                    type="textarea"
                    :rows="2"
                    :placeholder="$t('studio.optional')"
                  />
                </DqCollapseItem>
              </DqCollapse>
            </DqSurfaceCard>

            <!-- Lyrics -->
            <DqSurfaceCard class="studio-surface-card studio-card-mb studio-audio-lyrics-card">
              <template #header>
                <div class="card-title card-title--split">
                  <span>
                    <DqIcon><document /></DqIcon>
                    {{ $t('audio.lyrics') }}
                  </span>
                  <DqTag size="small" type="info">{{ $t('studio.optional') }}</DqTag>
                </div>
              </template>
              <DqInput
                v-model="params.lyrics"
                type="textarea"
                class="studio-audio-lyrics-input"
                :rows="10"
                :placeholder="params.instrumental ? $t('audio.lyricsPlaceholderInstrumental') : $t('audio.lyricsPlaceholder')"
                resize="vertical"
                :disabled="params.instrumental"
              />
              <p class="studio-field-footnote">{{ $t('audio.lyricsHint') }}</p>
              <div class="studio-audio-meta-row">
                <div class="studio-audio-meta-item studio-audio-meta-item--switch">
                  <span class="studio-audio-meta-label">{{ $t('audio.instrumental') }}</span>
                  <DqSwitch v-model="params.instrumental" size="small" />
                </div>
                <div
                  v-if="supportsVocalType"
                  class="studio-audio-meta-item studio-audio-meta-item--vocal"
                >
                  <span class="studio-audio-meta-label">{{ $t('audio.vocalType') }}</span>
                  <DqSelect
                    v-model="params.vocal_type"
                    size="small"
                    class="studio-audio-vocal-type-select"
                    clearable
                    :disabled="params.instrumental"
                    :placeholder="$t('audio.vocalTypeAuto')"
                  >
                    <DqOption
                      v-for="vt in vocalTypes"
                      :key="vt.value"
                      :label="vt.label"
                      :value="vt.value"
                    />
                  </DqSelect>
                </div>
                <div class="studio-audio-meta-item studio-audio-meta-item--vocal">
                  <span class="studio-audio-meta-label">{{ $t('audio.vocalLanguage') }}</span>
                  <DqSelect
                    v-model="params.vocal_language"
                    size="small"
                    class="studio-audio-vocal-lang-select"
                    clearable
                    :disabled="params.instrumental"
                    :placeholder="$t('audio.vocalLanguageAuto')"
                  >
                    <DqOption v-for="l in vocalLanguages" :key="l.value" :label="l.label" :value="l.value" />
                  </DqSelect>
                </div>
              </div>
              <p v-if="supportsVocalType && !params.instrumental" class="studio-field-footnote">
                {{ $t('audio.vocalTypeHint') }}
              </p>
            </DqSurfaceCard>

            <!-- Music params -->
            <DqSurfaceCard class="studio-surface-card studio-card-mb">
              <template #header>
                <div class="card-title">
                  <DqIcon><setting /></DqIcon>
                  {{ $t('audio.musicParams') }}
                </div>
              </template>
              <AudioCreateMusicParams
                :params="params"
                :musical-keys="musicalKeys"
                :time-signatures="timeSignatures"
                :duration-min="currentModelConfig?.parameters?.duration?.min ?? 10"
                :duration-max="currentModelConfig?.parameters?.duration?.max ?? 600"
              />
            </DqSurfaceCard>

            <!-- Advanced params -->
            <DqSurfaceCard class="studio-surface-card studio-card-mb">
              <DqCollapse v-model="advancedOpen" class="studio-collapse-plain">
                <DqCollapseItem name="advanced">
                  <template #title>
                    <div class="studio-collapse-title-row">
                      <DqIcon><setting /></DqIcon>
                      <span>{{ $t('studio.advancedParams') }}</span>
                      <DqTag v-if="hasCustomParams" size="small" type="warning">{{ $t('studio.hasCustom') }}</DqTag>
                    </div>
                  </template>
                  <AudioCreateAdvancedParams
                    :params="params"
                    :current-model-config="currentModelConfig"
                    :audio-formats="audioFormats"
                    @restore-defaults="restoreDefaults"
                    @randomize-seed="randomizeSeed"
                  />
                </DqCollapseItem>
              </DqCollapse>
            </DqSurfaceCard>
          </template>

          <!-- 同声翻唱 占位 -->
          <template v-if="audioWorkTab === 'cover'">
            <DqSurfaceCard class="studio-surface-card studio-cover-placeholder">
              <DqIcon class="studio-cover-placeholder-icon" :size="40"><clock /></DqIcon>
              <p>{{ $t('audio.coverComingSoon') }}</p>
            </DqSurfaceCard>
          </template>

          <!-- Generate button -->
          <DqSurfaceCard class="studio-surface-card studio-card-mb">
            <DqButton
              v-if="audioWorkTab === 'create'"
              type="primary"
              class="studio-primary-cta studio-primary-cta--simple dq-btn--cta"
              :disabled="submitDisabled"
              @click="startGeneration"
            >
              <DqIcon size="20"><headset /></DqIcon>
              <span class="studio-cta-gap">
                {{ generating ? $t('audio.generating') : $t('audio.generate') }}
              </span>
            </DqButton>
            <div class="studio-micro-hint">
              {{ $sendShortcutHint() }}
            </div>

            <!-- Progress display -->
            <div v-if="currentTask.id" class="studio-task-wrap">
              <DqProgress
                :percentage="Math.round(currentTask.progress * 100)"
                :status="currentTask.status === 'failed' ? 'exception' : ''"
              />
              <div class="studio-task-status">
                <template v-if="currentTask.total > 0 && currentTask.status === 'running'">
                  Step {{ currentTask.step }}/{{ currentTask.total }} &nbsp;
                </template>
                <DqTag :type="TSU.tagType(currentTask.status)" size="small">
                  {{ TSU.statusText(currentTask.status) }}
                </DqTag>
              </div>
            </div>
          </DqSurfaceCard>

          <!-- Logs -->
          <DqSurfaceCard class="studio-surface-card">
            <template #header>
              <div class="card-title card-title--split">
                <span>
                  <DqIcon><document /></DqIcon>
                  {{ $t('studio.logs') }}
                </span>
                <DqIconButton type="text" size="sm" :label="$t('common.delete')" @click="clearLogs">
                  <DqIcon><delete /></DqIcon>
                </DqIconButton>
              </div>
            </template>
            <div class="log-container studio-log-container--sm" ref="logContainer">
              <div v-if="logs.length === 0" class="studio-log-empty">
                {{ $t('studio.logsEmpty') }}
              </div>
              <div v-for="(log, index) in logs" :key="index" class="log-line">
                <span class="log-timestamp">{{ log.time }}</span>
                <span :class="'log-' + log.level">{{ log.message }}</span>
              </div>
            </div>
          </DqSurfaceCard>

        </div>
      </DqCol>

      <!-- Right panel: preview + recent -->
      <DqCol :xs="24" :md="8" :lg="7" :xl="6">
        <div class="preview-panel preview-panel--flat">

          <StudioPreviewPane :title="$t('studio.currentPreview')" icon="headset">
            <AudioMusicPlayer
              v-if="previewAudioSrc"
              :key="previewAudioKey"
              ref="previewPlayerRef"
              layout="featured"
              :src="previewAudioSrc"
              :title="previewPrompt"
              :subtitle="previewPlayerSubtitle"
              :hue="previewArtHue"
              @download="downloadPreviewAudio"
              @play="previewIsPlaying = true"
              @pause="previewIsPlaying = false"
              @duration="previewDurationSec = $event"
            >
              <div v-if="previewLyrics" class="studio-audio-effective-lyrics">
                <div class="studio-audio-effective-lyrics__head">
                  <span class="studio-audio-effective-lyrics__label">{{ $t('audio.effectiveLyrics') }}</span>
                  <DqButton
                    v-if="previewLyricsDownload"
                    type="text"
                    size="sm"
                    @click="downloadLyricsText"
                  >
                    {{ $t('audio.downloadLyrics') }}
                  </DqButton>
                </div>
                <pre class="studio-audio-effective-lyrics__body">{{ previewLyrics }}</pre>
                <p class="studio-field-footnote studio-audio-effective-lyrics__hint">
                  {{ $t('audio.effectiveLyricsHint') }}
                </p>
              </div>
            </AudioMusicPlayer>
            <DqEmpty v-else class="studio-preview-pane__empty" :description="$t('studio.noPreview')" />
          </StudioPreviewPane>

          <StudioPreviewPane :title="$t('audio.recentTitle')" icon="clock" split-head recent>
            <template #actions>
              <DqIconButton type="text" size="sm" :label="$t('gallery.refresh')" @click="loadRecentAudio">
                <DqIcon><refresh /></DqIcon>
              </DqIconButton>
            </template>
            <DqEmpty v-if="recentAudio.length === 0" :description="$t('gallery.empty')" />
            <ul v-else class="studio-audio-recent-list">
              <li
                v-for="ra in recentAudio"
                :key="ra.id"
                class="studio-audio-recent-item"
                :class="{ 'is-active': isRecentActive(ra) }"
              >
                <button
                  type="button"
                  class="studio-audio-recent-item__main"
                  @click="selectRecent(ra)"
                >
                  <span
                    class="studio-audio-recent-item__art"
                    :style="{ '--dq-music-hue': String(artHueForPrompt(ra.prompt)) }"
                  >
                    <DqIcon :size="18"><Headset /></DqIcon>
                  </span>
                  <span class="studio-audio-recent-item__text">
                    <span class="studio-audio-recent-title">{{ ra.prompt || ra.name || 'Audio' }}</span>
                    <span class="studio-audio-recent-meta">
                      <template v-if="isRecentActive(ra) && previewIsPlaying">{{ $t('audio.nowPlaying') }}</template>
                      <template v-else-if="ra.duration">{{ formatTime(ra.duration) }}</template>
                    </span>
                  </span>
                </button>
                <div class="studio-audio-recent-item__actions">
                  <button
                    type="button"
                    class="dq-music-player__btn-play dq-music-player__btn-play--sm"
                    :aria-label="isRecentActive(ra) && previewIsPlaying ? $t('audio.pause') : $t('audio.play')"
                    @click="toggleRecentPlay(ra)"
                  >
                    <DqIcon :size="16">
                      <pause v-if="isRecentActive(ra) && previewIsPlaying" />
                      <play v-else />
                    </DqIcon>
                  </button>
                  <DqIconButton type="text" size="sm" :label="$t('gallery.download')" @click="downloadAudioUrl(ra.url)">
                    <DqIcon><Download /></DqIcon>
                  </DqIconButton>
                </div>
              </li>
            </ul>
          </StudioPreviewPane>

        </div>
      </DqCol>
    </DqRow>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, watch, onMounted, nextTick } from 'vue';
import { useRouter } from 'vue-router';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { formatGenLogMessage, isDuplicateDenoiseStepLog } from '@/utils/genTaskLog';
import { $tt } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import { pickDefaultVersionKey, resolveDefaultModelRegistryKey } from '@/utils/defaultModelSettings';
import ModelLicenseBadges from '@/components/model/ModelLicenseBadges.vue';
import ModelPickerFilters from '@/components/model/ModelPickerFilters.vue';
import { useModelRegistryFilters } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import AudioCreateMusicParams from '@/components/audio/AudioCreateMusicParams.vue';
import AudioCreateAdvancedParams from '@/components/audio/AudioCreateAdvancedParams.vue';
import AudioMusicPlayer from '@/components/audio/AudioMusicPlayer.vue';
import StudioPreviewPane from '@/components/create/StudioPreviewPane.vue';
import { Download } from '@danqing/dq-shell';

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
const previewPlayerRef = ref(null);
const previewIsPlaying = ref(false);
const previewDurationSec = ref(0);

const modelRegistry = ref<Record<string, any>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});

const params = reactive({
  model: '',
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

const currentTask = reactive({ id: '', progress: 0, step: null as number | null, total: null as number | null, status: '' });
const logs = reactive<Array<{ level: string; message: string; time: string }>>([]);
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
  const p = currentModelConfig.value?.parameters || {};
  const stepsDef = p.steps?.default ?? 8;
  const guidanceDef = p.guidance?.default ?? 3.0;
  const durationDef = p.duration?.default ?? 30;
  const formatDef = (p.audio_formats && p.audio_formats[0]) || 'wav';
  return params.steps !== stepsDef || params.guidance !== guidanceDef || params.seed !== null
    || params.n !== 1 || params.duration !== durationDef || params.audio_format !== formatDef;
});

// ---- Audio formats ----
const audioFormats = ['mp3', 'flac', 'wav', 'opus', 'aac'];
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
      rows.push({
        key: mid + '|' + vk,
        label,
        ready,
        isRec,
        commercialUseAllowed: config.commercial_use_allowed === true,
        mid,
        vk,
        size: vd.size || '',
      });
    }
  }
  rows.sort((a, b) => {
    if (a.isRec !== b.isRec) return a.isRec ? -1 : 1;
    if (a.ready !== b.ready) return a.ready ? -1 : 1;
    return a.label.localeCompare(b.label);
  });
  return applyModelVersionFilters(rows, {
    installedOnly: true,
    commercialOnly: modelFilterCommercialOnly.value,
  });
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
    const [registryData, detailedStatusData] = await Promise.all([
      registryStore.load(),
      api.settings.getModelsDetailedStatus(),
    ]);
    modelRegistry.value = registryData?.models || {};
    if (detailedStatusData && typeof detailedStatusData === 'object') {
      modelsDetailedStatus.value = detailedStatusData as Record<string, unknown>;
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
    params.audio_format = p.audio_formats[0];
  }
  params.negative_prompt = '';
  params.lyrics = '';
  params.bpm = null;
  params.key_scale = '';
  params.time_signature = '';
  params.vocal_language = '';
  params.vocal_type = '';
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

function clearLogs() {
  logs.length = 0;
}

async function startGeneration() {
  if (generating.value) return;
  if (!params.model) { toast.warning('Please select a model'); return; }
  if (!modelReady.value) { toast.warning('Model is not ready'); return; }
  if (!params.prompt.trim()) { toast.warning('Please enter a prompt'); return; }

  generating.value = true;
  currentTask.id = '';
  currentTask.progress = 0;
  currentTask.step = null;
  currentTask.total = null;
  currentTask.status = '';
  logs.length = 0;
  previewLyrics.value = '';
  previewLyricsDownload.value = '';

  try {
    const body = {
      model: params.model,
      prompt: params.prompt.trim(),
      negative_prompt: params.negative_prompt || '',
      duration: params.duration ?? null,
      instrumental: !!params.instrumental,
      // Empty lyrics → backend uses [Instrumental] (same as ACE-Step handler text2music default).
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
    if (!api.gen?.streamMediaTask) {
      generating.value = false;
      return;
    }

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

    const addRecentFromAsset = (aid: string, meta?: Record<string, unknown>) => {
      const url = assetFileUrl(aid);
      const eff = meta ? String(meta.lyrics_effective || '').trim() : '';
      recentAudio.unshift({
        id: aid,
        url,
        prompt: params.prompt,
        lyrics: eff,
        duration: 0,
        playing: false,
      });
    };

    const setPreviewFromAsset = (aid: string, meta?: Record<string, unknown>) => {
      const url = assetFileUrl(aid);
      const changed = previewAudioSrc.value !== url;
      previewAudioSrc.value = url;
      previewPrompt.value = params.prompt;
      applyResultMeta(meta);
      previewIsPlaying.value = false;
      if (changed) previewAudioKey.value += 1;
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
        if (row.status === 'completed') {
          const res = row.result as Record<string, unknown> | undefined;
          const meta = res?.metadata as Record<string, unknown> | undefined;
          applyResultMeta(meta);
          const pid = res?.primary_asset_id;
          if (pid) setPreviewFromAsset(String(pid), meta);
        } else if (row.status === 'failed') {
          toast.error($tt('studio.genFailed', { msg: String(row.error || '') }));
        }
        generating.value = false;
      },
      onError: () => {
        toast.error($tt('studio.connectionLost'));
        generating.value = false;
      },
    });
  } catch (e: any) {
    toast.error((e.response && e.response.data && e.response.data.detail) || e.message || 'Generation failed');
    generating.value = false;
  }
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
  previewPrompt.value = item.prompt || '';
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

function loadRecentAudio() {
  api.gen.listAssets('audio', 20, 0).then((data: any) => {
    const items = (data && data.items) || [];
    recentAudio.length = 0;
    items.forEach((item: any) => {
      const aid = item.id || '';
      if (!aid) return;
      const url = api.gallery?.getImageUrl ? api.gallery.getImageUrl('asset:' + aid) : ('/api/assets/' + aid + '/file');
      const meta = item.metadata || {};
      recentAudio.push({
        id: aid,
        url,
        prompt: String(meta.prompt || ''),
        lyrics: String(meta.lyrics_effective || '').trim(),
        name: item.name || '',
        duration: item.duration_seconds != null ? Number(item.duration_seconds) : (meta.duration_seconds != null ? Number(meta.duration_seconds) : 0),
      });
    });
  }).catch(() => {});
}

async function applyAudioAppSettingsDefaults() {
  try {
    const st = await api.settings.getSettings() as {
      default_model_audio?: string;
      default_model?: string;
    };
    const dm = String(st.default_model_audio || st.default_model || '').trim();
    const mk = resolveDefaultModelRegistryKey(dm, modelRegistry.value, 'audio');
    if (!mk || !modelRegistry.value[mk]) return;
    const detailed = (modelsDetailedStatus.value[mk] || {}) as { versions?: Record<string, { ready?: boolean }> };
    const vers = detailed.versions || {};
    const defaultVK = pickDefaultVersionKey(mk, modelRegistry.value, vers);
    if (!defaultVK) return;
    selectedModelVersion.value = mk + '|' + defaultVK;
    onModelChange(selectedModelVersion.value);
  } catch (_) {
    /* ignore */
  }
}

// ---- Lifecycle ----
onMounted(async () => {
  await loadModelRegistry();
  await applyAudioAppSettingsDefaults();
  if (!params.model) {
    const versions = filteredModelPickerVersions.value;
    if (versions.length > 0) {
      const rec = versions.find((v: any) => v.ready && v.isRec) || versions.find((v: any) => v.ready) || versions[0];
      selectedModelVersion.value = rec.key;
      onModelChange(rec.key);
    }
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

watch(modelFilterCommercialOnly, () => {
  const rows = filteredModelPickerVersions.value;
  const key = selectedModelVersion.value;
  if (key && rows.some((r) => r.key === key)) {
    return;
  }
  const pick = rows.find((r) => r.ready) || rows[0];
  if (!pick) {
    return;
  }
  selectedModelVersion.value = pick.key;
  onModelChange(pick.key);
});
</script>
