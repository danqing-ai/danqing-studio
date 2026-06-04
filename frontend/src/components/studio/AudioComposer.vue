<template>
  <div class="audio-composer studio-composer-shell dq-glass--panel">
    <!-- Title -->
    <DqInput
      v-model="localTitle"
      size="small"
      :placeholder="$tt('studio.workTitlePlaceholder')"
      class="audio-composer__title"
    />

    <!-- Prompt -->
    <div class="audio-composer__prompt-wrap">
      <DqInput
        v-model="localPrompt"
        type="textarea"
        :rows="3"
        :placeholder="promptPlaceholder"
        resize="none"
        class="audio-composer__prompt"
        @keydown="onPromptKeydown"
      />
      <!-- Reference / cover source inside textarea -->
      <div class="audio-composer__ref-area">
        <div v-if="referenceMedia" class="audio-composer__ref-pill">
          <span class="audio-composer__ref-label">{{ referenceMedia.label }}</span>
          <DqIconButton type="text" size="xs" :label="$tt('common.delete')" @click="$emit('remove-reference')">
            <DqIcon :size="10"><Close /></DqIcon>
          </DqIconButton>
        </div>
        <DqIconButton
          v-else-if="workMode === 'cover'"
          type="text"
          size="xs"
          :label="$tt('audio.pickCoverSource')"
          class="audio-composer__ref-add"
          @click="$emit('pick-reference')"
        >
          <DqIcon :size="14"><Picture /></DqIcon>
        </DqIconButton>
      </div>

      <!-- Preset / Style (prompt corner, not toolbar — avoids confusion with advanced params) -->
      <div v-if="styles && Object.keys(styles).length > 0" class="audio-composer__preset-area">
        <DqDropdown trigger="click" size="small" @command="onStyleCommand">
          <DqIconButton
            type="text"
            size="xs"
            :label="$tt('create.preset')"
            class="audio-composer__preset-btn"
          >
            <DqIcon :size="14"><DocumentCopy /></DqIcon>
          </DqIconButton>
          <template #dropdown>
            <DqDropdownMenu>
              <DqDropdownItem
                v-for="(preset, name) in styles"
                :key="name"
                :command="name"
              >
                {{ presetLabel(name, preset) }}
              </DqDropdownItem>
            </DqDropdownMenu>
          </template>
        </DqDropdown>
      </div>
    </div>

    <!-- Lyrics: primary creative input (create mode only, not advanced params) -->
    <div v-if="showLyrics && workMode === 'create'" class="audio-composer__lyrics-wrap">
      <div class="audio-composer__lyrics-head">
        <span class="audio-composer__lyrics-label">{{ $tt('audio.lyrics') }}</span>
        <div class="audio-composer__inline-switch">
          <span>{{ $tt('audio.instrumental') }}</span>
          <DqSwitch v-model="localParams.instrumental" size="small" />
        </div>
      </div>
      <DqInput
        v-if="!localParams.instrumental"
        v-model="localParams.lyrics"
        type="textarea"
        :rows="4"
        :placeholder="$tt('audio.lyricsPlaceholder')"
        resize="none"
        class="audio-composer__lyrics"
      />
      <p class="audio-composer__lyrics-hint">
        {{ localParams.instrumental ? $tt('audio.lyricsPlaceholderInstrumental') : $tt('audio.lyricsHint') }}
      </p>
    </div>

    <!-- Toolbar -->
    <div class="audio-composer__toolbar">
      <div class="audio-composer__toolbar-left">
        <!-- Work mode -->
        <DqSegmented
          :model-value="localWorkMode"
          size="small"
          :options="workModeOptions"
          @update:model-value="localWorkMode = $event"
        />

        <!-- Model selector -->
        <DqSelect
          v-model="localModel"
          size="small"
          class="audio-composer__select audio-composer__select--model"
          style="min-width: 140px; max-width: 200px"
          @change="(val: string) => emit('model-change', val)"
        >
          <DqOption
            v-for="item in modelOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
            :disabled="item.disabled"
          >
            <DqTag
              v-if="item.commercialUseAllowed"
              size="mini"
              type="success"
              class="audio-composer__model-badge"
            >
              {{ $tt('download.commercialUseBadge') }}
            </DqTag>
          </DqOption>
        </DqSelect>

        <!-- Duration -->
        <DqSelect
          v-model="localDuration"
          size="small"
          class="audio-composer__select audio-composer__select--duration"
        >
          <DqOption
            v-for="opt in durationOptions"
            :key="opt.value"
            :label="opt.label"
            :value="opt.value"
          />
        </DqSelect>

        <!-- Advanced toggle -->
        <DqButton
          type="text"
          size="sm"
          class="audio-composer__adv-btn"
          @click="advancedOpen = !advancedOpen"
        >
          <DqIcon :size="14"><Tools /></DqIcon>
          <span>{{ $tt('studio.advancedParams') }}</span>
          <span v-if="hasCustomParams" class="audio-composer__dot" />
        </DqButton>
      </div>

      <div class="audio-composer__toolbar-right">
        <!-- Batch count -->
        <DqSelect
          v-model="localBatchCount"
          size="small"
          class="audio-composer__select audio-composer__select--batch"
        >
          <DqOption
            v-for="n in 4"
            :key="n"
            :label="`x${n}`"
            :value="n"
          />
        </DqSelect>

        <!-- Generate button -->
        <DqButton
          type="primary"
          size="sm"
          class="audio-composer__generate"
          :disabled="generating || !canGenerate"
          @click="$emit('generate')"
        >
          <DqIcon :size="16"><MagicStick /></DqIcon>
          <span>{{ generating ? $tt('audio.generating') : generateLabel }}</span>
        </DqButton>
      </div>
    </div>

    <StudioComposerAdvancedDrawer
      v-model:open="advancedOpen"
      :reset-label="$tt('studio.resetDefaults')"
      @reset-defaults="$emit('reset-defaults')"
    >
      <div class="audio-composer__advanced-inner">
              <!-- Steps & Guidance -->
              <div v-if="stepsDef || guidanceDef" class="audio-composer__advanced-row">
                <div v-if="stepsDef" class="audio-composer__field">
                  <label>{{ $tt('create.steps') }}</label>
                  <DqSlider
                    v-model="localParams.steps"
                    :min="stepsDef.min"
                    :max="stepsDef.max"
                    :step="stepsDef.step"
                  />
                  <span class="audio-composer__field-val">{{ localParams.steps }}</span>
                </div>
                <div v-if="guidanceDef" class="audio-composer__field">
                  <label>{{ $tt('create.guidance') }}</label>
                  <DqSlider
                    v-model="localParams.guidance"
                    :min="guidanceDef.min"
                    :max="guidanceDef.max"
                    :step="guidanceDef.step"
                  />
                  <span class="audio-composer__field-val">{{ localParams.guidance }}</span>
                </div>
              </div>

              <!-- Temperature, Top-K & Seed -->
              <div class="audio-composer__advanced-row">
                <div v-if="temperatureDef" class="audio-composer__field">
                  <label>{{ $tt('create.temperature') }}</label>
                  <DqSlider
                    v-model="localParams.temperature"
                    :min="temperatureDef.min"
                    :max="temperatureDef.max"
                    :step="temperatureDef.step"
                  />
                  <span class="audio-composer__field-val">{{ localParams.temperature }}</span>
                </div>
                <div v-if="topKDef" class="audio-composer__field">
                  <label>{{ $tt('audio.topK') }}</label>
                  <DqSlider
                    v-model="localParams.top_k"
                    :min="topKDef.min"
                    :max="topKDef.max"
                    :step="1"
                  />
                  <span class="audio-composer__field-val">{{ localParams.top_k }}</span>
                </div>
                <div class="audio-composer__field">
                  <label>{{ $tt('create.seed') }}</label>
                  <div class="audio-composer__seed-wrap">
                    <DqInput
                      v-model="seedInput"
                      size="small"
                      :placeholder="$tt('studio.seedPlaceholder')"
                      style="width: 100px"
                    />
                    <DqIconButton
                      type="text"
                      size="xs"
                      :label="$tt('create.randomSeed')"
                      @click="randomizeSeed"
                    >
                      <DqIcon :size="12"><Refresh /></DqIcon>
                    </DqIconButton>
                  </div>
                </div>
              </div>

              <!-- Codec controls (HeartMuLa) -->
              <div v-if="showCodecControls" class="audio-composer__advanced-row">
                <div class="audio-composer__field">
                  <label>{{ $tt('audio.codecSteps') }}</label>
                  <DqSlider
                    v-model="localParams.codec_steps"
                    :min="codecStepsDef?.min ?? 4"
                    :max="codecStepsDef?.max ?? 24"
                    :step="1"
                  />
                  <span class="audio-composer__field-val">{{ localParams.codec_steps }}</span>
                </div>
                <div class="audio-composer__field">
                  <label>{{ $tt('audio.codecGuidance') }}</label>
                  <DqSlider
                    v-model="localParams.codec_guidance"
                    :min="codecGuidanceDef?.min ?? 1.0"
                    :max="codecGuidanceDef?.max ?? 2.0"
                    :step="codecGuidanceDef?.step ?? 0.05"
                  />
                  <span class="audio-composer__field-val">{{ localParams.codec_guidance }}</span>
                </div>
              </div>

              <!-- Cover source fidelity -->
              <div v-if="workMode === 'cover' && showCoverFidelity" class="audio-composer__advanced-row">
                <div class="audio-composer__field">
                  <label>{{ $tt('audio.sourceFidelity') }}</label>
                  <DqSlider
                    v-model="localParams.source_fidelity"
                    :min="0"
                    :max="1"
                    :step="0.05"
                  />
                  <span class="audio-composer__field-val">{{ localParams.source_fidelity }}</span>
                </div>
              </div>

              <!-- Negative prompt -->
              <div v-if="showNegativePrompt" class="audio-composer__advanced-row">
                <div class="audio-composer__field audio-composer__field--full">
                  <label>{{ $tt('create.negativePrompt') }}</label>
                  <DqInput
                    v-model="localParams.negative_prompt"
                    type="textarea"
                    :rows="2"
                    :placeholder="$tt('create.negativePlaceholder')"
                    resize="none"
                  />
                </div>
              </div>

              <!-- Music params: BPM, Key, Time Signature -->
              <div v-if="workMode === 'create' && (supportsBpm || supportsKeyScale || supportsTimeSignature)" class="audio-composer__advanced-row">
                <div v-if="supportsBpm" class="audio-composer__field">
                  <label>{{ $tt('audio.bpm') }}</label>
                  <DqInput
                    v-model="bpmInput"
                    size="small"
                    inputmode="numeric"
                    :placeholder="$tt('audio.bpmAuto')"
                    class="audio-composer__optional-input"
                  />
                </div>
                <div v-if="supportsKeyScale" class="audio-composer__field">
                  <label>{{ $tt('audio.keyScale') }}</label>
                  <DqSelect v-model="localParams.key_scale" size="small" clearable :placeholder="$tt('audio.keyScaleAuto')">
                    <DqOption v-for="k in musicalKeys" :key="k" :label="k" :value="k" />
                  </DqSelect>
                </div>
                <div v-if="supportsTimeSignature" class="audio-composer__field">
                  <label>{{ $tt('audio.timeSignature') }}</label>
                  <DqSelect v-model="localParams.time_signature" size="small" clearable :placeholder="$tt('audio.timeSignatureAuto')">
                    <DqOption
                      v-for="ts in timeSignatures"
                      :key="ts.value"
                      :label="ts.label"
                      :value="ts.value"
                    />
                  </DqSelect>
                </div>
              </div>

              <!-- Vocal type / language -->
              <div v-if="workMode === 'create' && (supportsVocalType || supportsVocalLanguage)" class="audio-composer__advanced-row">
                <div v-if="supportsVocalLanguage" class="audio-composer__field">
                  <label>{{ $tt('audio.vocalLanguage') }}</label>
                  <DqSelect v-model="localParams.vocal_language" size="small" clearable :placeholder="$tt('audio.vocalLanguageAuto')">
                    <DqOption v-for="l in vocalLanguages" :key="l.value" :label="l.label" :value="l.value" />
                  </DqSelect>
                </div>
                <div v-if="supportsVocalType" class="audio-composer__field">
                  <label>{{ $tt('audio.vocalType') }}</label>
                  <DqSelect v-model="localParams.vocal_type" size="small" clearable :placeholder="$tt('audio.vocalTypeAuto')">
                    <DqOption v-for="vt in vocalTypes" :key="vt.value" :label="vt.label" :value="vt.value" />
                  </DqSelect>
                </div>
              </div>
      </div>
    </StudioComposerAdvancedDrawer>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import StudioComposerAdvancedDrawer from './StudioComposerAdvancedDrawer.vue';
import { useI18n } from 'vue-i18n';
import {
  Close,
  DocumentCopy,
  MagicStick,
  Picture,
  Refresh,
  Tools,
} from '@danqing/dq-shell';
import { $tt } from '@/utils/i18n';

const props = defineProps<{
  modelValue: string;
  title?: string;
  workMode: string;
  model: string;
  duration: number;
  batchCount?: number;
  generating: boolean;
  canGenerate: boolean;
  generateLabel: string;
  modelOptions: Array<{ label: string; value: string; disabled?: boolean; commercialUseAllowed?: boolean }>;
  durationOptions: Array<{ label: string; value: number }>;
  styles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; media_scope?: string }>;
  params: Record<string, any>;
  hasCustomParams: boolean;
  showNegativePrompt: boolean;
  showLyrics?: boolean;
  showCodecControls?: boolean;
  showCoverFidelity?: boolean;
  referenceMedia?: { type: string; previewUrl: string; label: string } | null;
  currentModelConfig?: Record<string, any> | null;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
  (e: 'update:title', value: string): void;
  (e: 'update:workMode', value: string): void;
  (e: 'update:model', value: string): void;
  (e: 'update:duration', value: number): void;
  (e: 'update:batchCount', value: number): void;
  (e: 'update:params', value: Record<string, any>): void;
  (e: 'generate'): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'model-change', value: string): void;
  (e: 'reset-defaults'): void;
}>();

const { t: $t } = useI18n();

const advancedOpen = ref(false);
const seedInput = ref('');
const bpmInput = ref('');

function patchParams(patch: Record<string, unknown>) {
  emit('update:params', { ...props.params, ...patch });
}

const localTitle = computed({
  get: () => props.title || '',
  set: (v) => emit('update:title', v),
});

const localPrompt = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v),
});

const localWorkMode = computed({
  get: () => props.workMode,
  set: (v) => emit('update:workMode', v),
});

const workModeOptions = computed(() => [
  { label: $tt('action.audio.create'), value: 'create' },
  { label: $tt('action.audio.cover'), value: 'cover' },
]);

const localModel = computed({
  get: () => props.model,
  set: (v) => emit('update:model', v),
});

const localDuration = computed({
  get: () => props.duration,
  set: (v) => emit('update:duration', v),
});

const localBatchCount = computed({
  get: () => props.batchCount ?? 1,
  set: (v) => emit('update:batchCount', v),
});

const localParams = computed({
  get: () => props.params,
  set: (v) => emit('update:params', v),
});

const promptPlaceholder = computed(() => {
  if (props.workMode === 'cover') {
    return $tt('audio.coverPromptPlaceholder');
  }
  if (props.currentModelConfig?.family === 'heartmula') {
    return $tt('audio.promptPlaceholderHeartMuLa');
  }
  return $tt('audio.promptPlaceholder');
});

// Seed sync
watch(() => props.params.seed, (v) => {
  seedInput.value = v != null ? String(v) : '';
}, { immediate: true });

watch(seedInput, (v) => {
  const n = parseInt(v, 10);
  patchParams({ seed: !v || isNaN(n) ? null : n });
});

watch(() => props.params.bpm, (v) => {
  bpmInput.value = v != null ? String(v) : '';
}, { immediate: true });

watch(bpmInput, (v) => {
  const trimmed = v.trim();
  if (!trimmed) {
    if (props.params.bpm != null) patchParams({ bpm: null });
    return;
  }
  const n = parseInt(trimmed, 10);
  if (isNaN(n)) return;
  const clamped = Math.min(300, Math.max(30, n));
  if (props.params.bpm !== clamped) patchParams({ bpm: clamped });
  if (String(clamped) !== trimmed) bpmInput.value = String(clamped);
});

function randomizeSeed() {
  patchParams({ seed: Math.floor(Math.random() * 2147483647) });
}

// Style / Preset
function presetLabel(name: string, preset: Record<string, unknown>): string {
  const a = (preset.applies_to as string[]) || [];
  const hasC = a.includes('create');
  const hasCover = a.includes('cover');
  let tag = '';
  if (hasC && !hasCover) tag = $tt('audio.presetTagCreate');
  else if (hasCover && !hasC) tag = $tt('audio.presetTagCover');
  const display = (preset as any).name?.en || (preset as any).name?.zh || name;
  return tag ? `${tag} ${display}` : display;
}

function onStyleCommand(command: string) {
  const preset = props.styles[command];
  if (!preset) return;
  if (preset.positive) {
    localPrompt.value = localPrompt.value
      ? localPrompt.value + '\nStyle boost: ' + preset.positive
      : preset.positive;
  }
  if (preset.negative && props.showNegativePrompt) {
    patchParams({
      negative_prompt: localParams.value.negative_prompt
        ? localParams.value.negative_prompt + '\n' + preset.negative
        : preset.negative,
    });
  }
}

// Capability flags from registry
const paramDefs = computed(() => props.currentModelConfig?.parameters || {});

const stepsDef = computed(() => {
  const p = paramDefs.value.steps;
  if (!p) return null;
  return { min: p.min ?? 1, max: p.max ?? 100, step: p.step ?? 1, default: p.default ?? 8 };
});

const guidanceDef = computed(() => {
  const p = paramDefs.value.guidance;
  if (!p) return null;
  return { min: p.min ?? 1, max: p.max ?? 20, step: p.step ?? 0.1, default: p.default ?? 3.0 };
});

const temperatureDef = computed(() => {
  const p = paramDefs.value.temperature;
  if (!p) return null;
  return { min: p.min ?? 0.1, max: p.max ?? 2, step: p.step ?? 0.1, default: p.default ?? 1.0 };
});

const topKDef = computed(() => {
  const p = paramDefs.value.top_k;
  if (!p) return null;
  return { min: p.min ?? 10, max: p.max ?? 100, default: p.default ?? 50 };
});

const supportsBpm = computed(() => paramDefs.value.supports_bpm === true);
const supportsKeyScale = computed(() => props.currentModelConfig?.parameters?.supports_key_scale === true);
const supportsTimeSignature = computed(() => props.currentModelConfig?.parameters?.supports_time_signature === true);
const supportsVocalType = computed(() => props.currentModelConfig?.parameters?.supports_vocal_type === true);
const supportsVocalLanguage = computed(() => props.currentModelConfig?.parameters?.supports_vocal_language === true);

const codecStepsDef = computed(() => {
  const p = props.currentModelConfig?.parameters?.codec_steps;
  if (!p) return null;
  return { min: p.min ?? 4, max: p.max ?? 24, default: p.default ?? 10 };
});

const codecGuidanceDef = computed(() => {
  const p = props.currentModelConfig?.parameters?.codec_guidance;
  if (!p) return null;
  return { min: p.min ?? 1.0, max: p.max ?? 2.0, step: p.step ?? 0.05, default: p.default ?? 1.25 };
});

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

const vocalLanguages = [
  { label: 'English', value: 'en' }, { label: '中文', value: 'zh' }, { label: '日本語', value: 'ja' },
  { label: '한국어', value: 'ko' }, { label: 'Français', value: 'fr' }, { label: 'Deutsch', value: 'de' },
  { label: 'Español', value: 'es' }, { label: 'Português', value: 'pt' },
];

const vocalTypes = computed(() => [
  { label: $tt('audio.vocalTypeMale'), value: 'male' },
  { label: $tt('audio.vocalTypeFemale'), value: 'female' },
  { label: $tt('audio.vocalTypeChorus'), value: 'chorus' },
  { label: $tt('audio.vocalTypeDuet'), value: 'duet' },
]);

function onPromptKeydown(e: KeyboardEvent) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!props.generating && props.canGenerate) {
      emit('generate');
    }
  }
}
</script>

<style scoped>
.audio-composer {
  border-radius: var(--dq-radius-group);
  padding: 18px 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.audio-composer__title {
  margin: 0;
}

.audio-composer__title :deep(.dq-input) {
  font-size: 13px;
  border-radius: var(--dq-radius-input);
}

.audio-composer__prompt-wrap {
  position: relative;
  margin: 0;
}

.audio-composer__prompt :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: 14px;
  line-height: 1.5;
  padding: 10px 12px 32px;
  min-height: 4.5rem;
  max-height: 4.5rem;
  resize: none;
  overflow-y: auto;
  box-shadow: 0 0 0 3px transparent;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.audio-composer__prompt :deep(.dq-input--textarea:focus) {
  box-shadow: var(--dq-focus-ring);
}

/* Reference area inside textarea */
.audio-composer__ref-area {
  position: absolute;
  bottom: 6px;
  left: 8px;
  display: flex;
  align-items: center;
}

.audio-composer__ref-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 6px;
  background: var(--dq-fill-secondary);
  border-radius: 6px;
  border: 1px solid var(--dq-border-subtle);
}

.audio-composer__ref-label {
  font-size: 11px;
  color: var(--dq-label-secondary);
}

.audio-composer__ref-add {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.audio-composer__ref-add:hover {
  opacity: 1;
}

.audio-composer__preset-area {
  position: absolute;
  bottom: 6px;
  right: 8px;
  display: flex;
  align-items: center;
}

/* Toolbar */
.audio-composer__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 4px;
  flex-wrap: nowrap;
}

.audio-composer__toolbar-left,
.audio-composer__toolbar-right {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: nowrap;
}

.audio-composer__toolbar-left {
  overflow-x: auto;
  scrollbar-width: none;
}

.audio-composer__toolbar-left::-webkit-scrollbar {
  display: none;
}

.audio-composer__select {
  width: auto;
}

.audio-composer__select--model {
  min-width: 140px;
  max-width: 200px;
}

.audio-composer__select--duration {
  min-width: 70px;
}

.audio-composer__select--batch {
  min-width: 50px;
}

.audio-composer__model-label {
  flex: 1;
}

.audio-composer__model-badge {
  font-size: 10px;
  padding: 0 4px;
  height: 16px;
  line-height: 16px;
}

.audio-composer__preset-btn {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.audio-composer__preset-btn:hover {
  opacity: 1;
}

/* Lyrics block (main composer, below style prompt) */
.audio-composer__lyrics-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.audio-composer__lyrics-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.audio-composer__lyrics-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.audio-composer__lyrics :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: 14px;
  line-height: 1.5;
  padding: 10px 12px;
  min-height: 5.5rem;
  max-height: 9rem;
  resize: none;
  overflow-y: auto;
  box-shadow: 0 0 0 3px transparent;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.audio-composer__lyrics :deep(.dq-input--textarea:focus) {
  box-shadow: var(--dq-focus-ring);
}

.audio-composer__lyrics-hint {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.audio-composer__adv-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  position: relative;
}

.audio-composer__dot {
  position: absolute;
  top: 1px;
  right: 1px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-warning);
  border: 1.5px solid var(--dq-surface);
}

.audio-composer__generate {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 14px;
  font-weight: 600;
  font-size: 13px;
}

/* Advanced panel */
.audio-composer__advanced-inner {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.audio-composer__advanced-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.audio-composer__field {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 260px;
  overflow: hidden;
}

.audio-composer__field--full {
  flex: 1 1 100%;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.audio-composer__field label {
  font-size: 12px;
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  width: 90px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.audio-composer__field :deep(.dq-slider) {
  flex: 1;
}

.audio-composer__field-val {
  font-size: 12px;
  color: var(--dq-label-secondary);
  min-width: 32px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.audio-composer__seed-wrap {
  display: flex;
  align-items: center;
  gap: 4px;
}

.audio-composer__optional-input {
  flex: 1;
  min-width: 0;
}

.audio-composer__optional-input :deep(.dq-input) {
  width: 100%;
}

.audio-composer__inline-switch {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--dq-label-secondary);
}

</style>