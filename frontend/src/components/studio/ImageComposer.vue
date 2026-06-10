<template>
  <div
    class="image-composer studio-composer-shell dq-glass--panel"
    :class="{ 'image-composer--collapsed': collapsed }"
  >
    <!-- Top: Title -->
    <DqInput
      v-if="!collapsed"
      v-model="localTitle"
      size="small"
      :placeholder="$t('studio.workTitlePlaceholder')"
      class="image-composer__title"
    />

    <!-- Middle: Prompt input -->
    <div v-if="!collapsed" class="image-composer__prompt-block">
    <div class="image-composer__prompt-wrap">
      <DqInput
        v-model="localPrompt"
        type="textarea"
        :rows="3"
        :placeholder="$t('create.promptPlaceholder')"
        resize="none"
        class="image-composer__prompt"
        @keydown.meta.enter.prevent="$emit('generate')"
        @keydown.ctrl.enter.prevent="$emit('generate')"
      />

      <!-- Bottom-left: Reference image -->
      <div class="image-composer__ref-area">
        <div v-if="referenceImage" class="image-composer__ref-pill">
          <img :src="referenceImage.previewUrl" alt="ref" />
          <ComposerIconTip
            v-if="referenceAssetId"
            :content="$t('create.composerTip.reversePrompt')"
          >
            <DqIconButton
              type="text"
              size="xs"
              :disabled="reversing"
              :aria-label="$t('create.reversePrompt')"
              @click="$emit('reverse-prompt')"
            >
              <DqIcon :size="10"><Refresh /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
          <ComposerIconTip :content="$t('create.composerTip.removeRef')">
            <DqIconButton
              type="text"
              size="xs"
              :aria-label="$t('common.delete')"
              @click="$emit('remove-reference')"
            >
              <DqIcon :size="10"><Close /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
        </div>
        <ComposerIconTip v-else :content="$t('create.composerTip.refImage')">
          <DqIconButton
            type="text"
            size="xs"
            class="image-composer__ref-add"
            :aria-label="$t('create.refImage')"
            @click="$emit('pick-reference')"
          >
            <DqIcon :size="14"><Picture /></DqIcon>
          </DqIconButton>
        </ComposerIconTip>
      </div>

      <!-- Bottom-right: Preset / Style picker -->
      <div class="image-composer__preset-area">
        <ComposerIconTip
          :content="localPrompt.trim() ? $t('create.composerTip.enhance') : $t('create.composerTip.enhanceEmpty')"
        >
          <DqIconButton
            class="image-composer__preset-area__enhance-btn"
            type="text"
            size="xs"
            :disabled="enhancing || !localPrompt.trim()"
            :aria-label="$t('create.enhance')"
            @click="onEnhanceClick"
          >
            <DqIcon :size="12"><MagicStick /></DqIcon>
          </DqIconButton>
        </ComposerIconTip>
        <DqDropdown
          v-if="styles && Object.keys(styles).length > 0"
          trigger="click"
          size="small"
          @command="onStyleCommand"
        >
          <DqIconButton
            type="text"
            size="xs"
            class="image-composer__preset-btn"
            :label="$t('create.composerTip.preset')"
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
    <ComposerPromptApplyStrip
      v-if="promptApplyPreview"
      :preview="promptApplyPreview"
      @replace="$emit('prompt-apply-replace')"
      @append="$emit('prompt-apply-append')"
      @dismiss="$emit('prompt-apply-dismiss')"
    />
    </div>

    <!-- Toolbar -->
    <div class="image-composer__toolbar">
      <div class="image-composer__toolbar-left">
        <!-- Mode selector (optional) -->
        <DqSegmented
          v-if="modeOptions && modeOptions.length > 0"
          v-model="localMode"
          size="small"
          :options="modeOptions"
        />

        <!-- Model selector -->
        <DqSelect
          v-model="localModel"
          size="small"
          class="image-composer__select image-composer__select--model"
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
              class="image-composer__model-badge"
            >
              {{ $t('download.commercialUseBadge') }}
            </DqTag>
          </DqOption>
        </DqSelect>

        <!-- Size selector (txt2img only; img2img 输出尺寸跟随源图) -->
        <DqSelect
          v-if="!isImg2imgMode"
          v-model="localSize"
          size="small"
          class="image-composer__select image-composer__select--size"
        >
          <DqOption
            v-for="opt in sizeOptions"
            :key="opt.value"
            :value="opt.value"
            :label="formatResolutionOptionLabel(opt)"
          />
        </DqSelect>

        <!-- Advanced params toggle -->
        <DqButton
          type="text"
          size="sm"
          class="image-composer__adv-btn"
          @click="advancedOpen = !advancedOpen"
        >
          <DqIcon :size="14"><Tools /></DqIcon>
          <span>{{ $t('studio.advancedParams') }}</span>
        </DqButton>
      </div>

      <div class="image-composer__toolbar-right">
        <!-- Batch count -->
        <DqSelect
          v-model="localBatchCount"
          size="small"
          class="image-composer__select image-composer__select--batch"
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
          class="image-composer__generate"
          :disabled="generating || !canGenerate"
          @click="$emit('generate')"
        >
          <DqIcon size="16"><MagicStick /></DqIcon>
          <span>{{ generating ? $t('create.generating') : $t('studio.generate') }}</span>
        </DqButton>
      </div>
    </div>

    <StudioComposerAdvancedDrawer
      v-model:open="advancedOpen"
      :reset-label="$t('create.restoreDefaults')"
      @reset-defaults="$emit('reset-defaults')"
    >
      <div class="image-composer__advanced-inner">
              <div v-if="paramSchema.steps || showGuidanceSlider" class="image-composer__advanced-row">
                <div v-if="paramSchema.steps" class="image-composer__field">
                  <label>{{ $t('create.stepsLabel') }}</label>
                  <DqSlider
                    v-model="localParams.steps"
                    :min="paramSchema.steps.min ?? 1"
                    :max="paramSchema.steps.max ?? 50"
                    :step="paramSchema.steps.step ?? 1"
                  />
                  <span class="image-composer__field-val">{{ localParams.steps }}</span>
                </div>

                <div v-if="showGuidanceSlider" class="image-composer__field">
                  <label>{{ $t('create.guidanceLabel') }}</label>
                  <DqSlider
                    v-model="localParams.guidance"
                    :min="paramSchema.guidance?.min ?? 0"
                    :max="paramSchema.guidance?.max ?? 20"
                    :step="paramSchema.guidance?.step ?? 0.5"
                  />
                  <span class="image-composer__field-val">{{ localParams.guidance }}</span>
                </div>
              </div>

              <div class="image-composer__advanced-row">
                <div class="image-composer__field">
                  <label>{{ $t('studio.seed') }}</label>
                  <div class="image-composer__seed-wrap">
                    <DqInput
                      v-model="localParams.seed"
                      size="small"
                      :placeholder="$t('studio.seedPlaceholder')"
                      style="width: 100px"
                    />
                    <DqIconButton
                      type="text"
                      size="xs"
                      :label="$t('create.randomSeed')"
                      @click="randomizeSeed"
                    >
                      <DqIcon :size="12"><Refresh /></DqIcon>
                    </DqIconButton>
                  </div>
                </div>

                <div v-if="showStrengthSlider" class="image-composer__field">
                  <label>{{ $t('create.strengthLabel') }}</label>
                  <DqSlider
                    v-model="localParams.strength"
                    :min="paramSchema.strength?.min ?? 0"
                    :max="paramSchema.strength?.max ?? 1"
                    :step="paramSchema.strength?.step ?? 0.05"
                  />
                  <span class="image-composer__field-val">{{ localParams.strength }}</span>
                </div>
              </div>

              <!-- Scheduler -->
          <div v-if="currentModelConfig?.parameters?.scheduler?.options" class="image-composer__advanced-row">
            <div class="image-composer__field">
              <label>{{ currentModelConfig.parameters.scheduler.label || $t('create.schedulerLabel') }}</label>
              <DqSelect v-model="localParams.scheduler" size="small" style="width: 220px">
                    <DqOption
                      v-for="opt in currentModelConfig.parameters.scheduler.options"
                      :key="String(opt)"
                      :label="String(opt)"
                      :value="opt"
                    />
                  </DqSelect>
                </div>
              </div>

              <!-- LoRA -->
              <div v-if="currentModelConfig?.parameters?.lora_support && compatibleLoras?.length" class="image-composer__advanced-row">
                <div class="image-composer__field image-composer__field--full" style="flex-direction: column; align-items: flex-start; gap: 8px;">
                  <div style="display: flex; align-items: center; gap: 10px; width: 100%;">
                    <label>{{ $t('studio.loraLabel') }}</label>
                    <DqSelect v-model="localParams.lora" size="small" clearable :placeholder="$t('studio.noLora')" style="flex: 1; max-width: 300px;">
                      <DqOption
                        v-for="l in compatibleLoras"
                        :key="String(l.id)"
                        :label="loraOptionLabel(l)"
                        :value="l.id"
                      />
                    </DqSelect>
                  </div>
                  <div v-if="localParams.lora" style="display: flex; align-items: center; gap: 10px; width: 100%;">
                    <label style="min-width: 60px;">{{ $t('create.loraScale') }}</label>
                    <DqSlider v-model="localParams.lora_scale" :min="0" :max="2" :step="0.05" style="flex: 1;" />
                    <span class="image-composer__field-val">{{ localParams.lora_scale }}</span>
                  </div>
                </div>
              </div>

              <!-- ControlNet -->
              <div v-if="compatibleControlNets?.length" class="image-composer__advanced-row">
                <p v-if="!controlNetRuntimeAvailable" class="image-composer__control-hint">
                  {{ $t('studio.controlnetMlxOnly') }}
                </p>
                <div class="image-composer__field image-composer__field--full" style="flex-direction: column; align-items: flex-start; gap: 8px;">
                  <div style="display: flex; align-items: center; gap: 10px; width: 100%;">
                    <label>{{ $t('studio.controlNet') }}</label>
                    <DqSelect
                      v-model="localParams.controlnet"
                      size="small"
                      clearable
                      :disabled="!controlNetRuntimeAvailable"
                      :placeholder="$t('studio.noControlNet')"
                      style="flex: 1; max-width: 300px;"
                    >
                      <DqOption
                        v-for="n in compatibleControlNets"
                        :key="String(n.key)"
                        :label="controlNetOptionLabel(n)"
                        :value="String(n.key)"
                        :disabled="!controlNetReady(n)"
                      />
                    </DqSelect>
                  </div>
                  <p v-if="localParams.controlnet" class="image-composer__control-hint">
                    {{ controlNetGuideHint }}
                  </p>
                  <p
                    v-if="localParams.controlnet && showCompanionLoraHint"
                    class="image-composer__control-hint image-composer__control-hint--sub"
                  >
                    {{ $t('studio.controlnetCompanionLoraHint') }}
                  </p>
                  <div v-if="localParams.controlnet" style="display: flex; align-items: center; gap: 10px; width: 100%;">
                    <label style="min-width: 60px;">{{ controlNetStrengthLabel }}</label>
                    <DqSlider v-model="localParams.controlnet_strength" :min="0" :max="2" :step="0.05" style="flex: 1;" />
                    <span class="image-composer__field-val">{{ localParams.controlnet_strength }}</span>
                  </div>
                  <div v-if="localParams.controlnet" class="image-composer__control-image-row">
                    <label>{{ $t('canvas.controlImage') }}</label>
                    <div v-if="controlImage" class="image-composer__ref-pill image-composer__ref-pill--compact">
                      <img :src="controlImage.previewUrl" alt="control" />
                      <DqIconButton
                        type="text"
                        size="xs"
                        :label="$t('common.delete')"
                        @click="$emit('remove-control')"
                      >
                        <DqIcon :size="10"><Close /></DqIcon>
                      </DqIconButton>
                    </div>
                    <DqButton v-else size="xs" type="secondary" @click="$emit('pick-control')">
                      {{ $t('canvas.pickControlImage') }}
                    </DqButton>
                  </div>
                </div>
              </div>

              <!-- Negative prompt -->
              <div v-if="showNegativePrompt" class="image-composer__advanced-row">
                <div class="image-composer__field image-composer__field--full">
                  <label>{{ $t('studio.negativePrompt') }}</label>
                  <DqInput
                    v-model="localParams.negative_prompt"
                    type="textarea"
                    :rows="2"
                    :placeholder="$t('create.negativePlaceholder')"
                    resize="none"
                  />
                </div>
              </div>
      </div>
    </StudioComposerAdvancedDrawer>

    <!-- Inline shortcut hint -->
    <div v-if="!collapsed" class="image-composer__hint">
      {{ shortcutHint }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import StudioComposerAdvancedDrawer from './StudioComposerAdvancedDrawer.vue';
import ComposerPromptApplyStrip from './ComposerPromptApplyStrip.vue';
import ComposerIconTip from './ComposerIconTip.vue';
import { useI18n } from 'vue-i18n';
import {
  applyControlNetRegistryDefaults,
  controlNetDisplayName,
  controlNetReady,
  isCannyOrDepthControlNet,
  isReduxControlNet,
} from '@/composables/useStructuralGuide';
import {
  Close,
  DocumentCopy,
  MagicStick,
  Picture,
  Refresh,
  Tools,
} from '@danqing/dq-shell';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { $tt } from '@/utils/i18n';
import { formatResolutionOptionLabel, img2imgUsesStrength, normalizeParamsDef } from '@/utils/registryParamSchema';

const props = defineProps<{
  modelValue: string;
  title?: string;
  model: string;
  size: string;
  batchCount: number;
  generating: boolean;
  canGenerate: boolean;
  modelOptions: Array<{ label: string; value: string; disabled?: boolean; commercialUseAllowed?: boolean }>;
  sizeOptions: Array<{ label: string; value: string; pixelLabel?: string }>;
  styles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; trigger_words?: string; media_scope?: string }>;
  params: {
    steps: number;
    guidance: number;
    seed: string;
    strength: number;
    negative_prompt: string;
    scheduler?: string;
    lora?: string;
    lora_scale?: number;
    controlnet?: string;
    controlnet_strength?: number;
  };
  hasCustomParams: boolean;
  showNegativePrompt: boolean;
  referenceImage: { previewUrl: string; path: string } | null;
  controlImage?: { previewUrl: string; path: string } | null;
  /**
   * Optional mode selector. If provided, a segmented control is shown in the
   * toolbar for switching between text2img / img2img modes.
   */
  mode?: string;
  modeOptions?: Array<{ label: string; value: string }>;
  currentModelConfig?: Record<string, any> | null;
  compatibleLoras?: Record<string, unknown>[];
  compatibleControlNets?: Record<string, unknown>[];
  controlNetRuntimeAvailable?: boolean;
  enhancing?: boolean;
  reversing?: boolean;
  collapsed?: boolean;
  promptApplyPreview?: string | null;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
  (e: 'update:title', value: string): void;
  (e: 'update:model', value: string): void;
  (e: 'update:size', value: string): void;
  (e: 'update:batchCount', value: number): void;
  (e: 'update:params', value: typeof props.params): void;
  (e: 'update:mode', value: string): void;
  (e: 'generate'): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'pick-control'): void;
  (e: 'remove-control'): void;
  (e: 'model-change', value: string): void;
  (e: 'reset-defaults'): void;
  (e: 'enhance', ctx?: { stylePositive?: string }): void;
  (e: 'reverse-prompt'): void;
  (e: 'prompt-apply-replace'): void;
  (e: 'prompt-apply-append'): void;
  (e: 'prompt-apply-dismiss'): void;
}>();

const { t: $t } = useI18n();

const referenceAssetId = computed(() => {
  if (!props.referenceImage?.path) return null;
  return assetIdFromGalleryPath(props.referenceImage.path);
});

const localParams = computed({
  get: () => props.params,
  set: (v) => emit('update:params', v),
});

function controlNetOptionLabel(n: Record<string, unknown>): string {
  const name = controlNetDisplayName(n);
  return controlNetReady(n) ? name : `${name} (${$t('studio.controlnetNotInstalled')})`;
}

function loraOptionLabel(l: Record<string, unknown>): string {
  const base = String(l.name || l.id || '');
  if (l.source === 'user_trained') {
    return `${base} (${$t('studio.myLoraTag')})`;
  }
  return base;
}

const controlNetGuideHint = computed(() => {
  const key = String(localParams.value.controlnet || '').toLowerCase();
  if (key.includes('fill')) return $t('studio.controlnetFillHint');
  if (key.includes('depth')) return $t('studio.controlnetDepthHint');
  if (key.includes('redux')) return $t('studio.controlnetReduxHint');
  return $t('studio.controlnetBundleHint');
});

const controlNetStrengthLabel = computed(() => {
  if (isReduxControlNet(String(localParams.value.controlnet || ''))) {
    return $t('create.reduxStrengthLabel');
  }
  return $t('create.controlNetStrengthLabel');
});

const controlNetRuntimeAvailable = computed(
  () => props.controlNetRuntimeAvailable !== false,
);

const showCompanionLoraHint = computed(() =>
  isCannyOrDepthControlNet(String(localParams.value.controlnet || '')),
);

watch(
  () => localParams.value.controlnet,
  (key, prev) => {
    if (!key || key === prev) return;
    applyControlNetRegistryDefaults(String(key), props.compatibleControlNets, localParams.value);
  },
);

const advancedOpen = ref(false);
const selectedStyle = ref('');
const lastStylePositive = ref('');

const paramSchema = computed(() => normalizeParamsDef(props.currentModelConfig?.parameters));

const isImg2imgMode = computed(
  () => localMode.value === 'img2img' || props.referenceImage != null,
);

const showStrengthSlider = computed(
  () => isImg2imgMode.value && img2imgUsesStrength(props.currentModelConfig?.parameters),
);

const showGuidanceSlider = computed(() => {
  const g = paramSchema.value.guidance;
  if (!g) return false;
  if (g.fixed === true) return false;
  const min = typeof g.min === 'number' ? g.min : 0;
  const max = typeof g.max === 'number' ? g.max : 20;
  return min !== max;
});

const localTitle = computed({
  get: () => props.title || '',
  set: (v) => emit('update:title', v),
});

const localMode = computed({
  get: () => props.mode || 'text2img',
  set: (v) => emit('update:mode', v),
});

const localPrompt = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v),
});

const localModel = computed({
  get: () => props.model,
  set: (v) => emit('update:model', v),
});

const localSize = computed({
  get: () => props.size,
  set: (v) => emit('update:size', v),
});

const localBatchCount = computed({
  get: () => props.batchCount,
  set: (v) => emit('update:batchCount', v),
});

function patchParams(patch: Partial<typeof props.params>) {
  Object.assign(props.params, patch);
}

function randomizeSeed() {
  patchParams({ seed: String(Math.floor(Math.random() * 1_000_000)) });
}

const shortcutHint = computed(() => {
  const isMac = navigator.platform.toLowerCase().includes('mac');
  return isMac ? '⌘ + Enter ' + $t('create.sendShortcutHintMac') : 'Ctrl + Enter ' + $t('create.sendShortcutHintWin');
});

function onEnhanceClick() {
  const style = lastStylePositive.value.trim();
  emit('enhance', style ? { stylePositive: style } : undefined);
}

function onStyleChange(name: string) {
  if (!name || !props.styles[name]) return;
  const preset = props.styles[name];
  if (preset.positive) {
    lastStylePositive.value = String(preset.positive);
    const current = localPrompt.value || '';
    localPrompt.value = current
      ? current + '\nStyle boost: ' + preset.positive
      : preset.positive;
  }
  if (preset.trigger_words) {
    const tw = String(preset.trigger_words).trim();
    if (tw) {
      localPrompt.value = localPrompt.value
        ? localPrompt.value + '\n' + tw
        : tw;
    }
  }
  if (preset.negative && props.showNegativePrompt) {
    const current = localParams.value.negative_prompt || '';
    patchParams({
      negative_prompt: current
        ? current + '\n' + preset.negative
        : preset.negative,
    });
  }
  selectedStyle.value = '';
}

function presetLabel(name: string, preset: Record<string, unknown>): string {
  const a = (preset.applies_to as string[]) || [];
  const hasC = a.includes('create');
  const hasEdit = a.some((x: string) => ['rewrite', 'retouch', 'extend'].includes(x));
  let tag = '';
  if (hasC && !hasEdit) tag = $tt('create.presetTagT2I');
  else if (hasEdit && !hasC) tag = $tt('create.presetTagI2I');
  return tag ? `${tag} ${name}` : name;
}

function onLoadPreset() {
  if (!selectedStyle.value || !props.styles[selectedStyle.value]) return;
  onStyleChange(selectedStyle.value);
  selectedStyle.value = '';
}

function onStyleCommand(command: string) {
  onStyleChange(command);
}
</script>

<style scoped>
.image-composer {
  border-radius: var(--dq-radius-group);
  padding: 18px 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.image-composer--collapsed {
  padding: 10px 16px 12px;
  gap: 0;
}

/* Title */
.image-composer__title {
  margin: 0;
}

.image-composer__title :deep(.dq-input) {
  font-size: 13px;
  border-radius: var(--dq-radius-input);
}

/* Prompt */
.image-composer__prompt-block {
  display: flex;
  flex-direction: column;
}

.image-composer__prompt-wrap {
  position: relative;
  margin: 0;
}

.image-composer__prompt :deep(.dq-input--textarea) {
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

.image-composer__prompt :deep(.dq-input--textarea:focus) {
  box-shadow: var(--dq-focus-ring);
}

/* Reference image area inside textarea */
.image-composer__ref-area {
  position: absolute;
  bottom: 6px;
  left: 8px;
  z-index: 1;
  display: flex;
  align-items: center;
}

.image-composer__ref-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 2px;
  background: var(--dq-fill-secondary);
  border-radius: 6px;
  border: 1px solid var(--dq-border-subtle);
}

.image-composer__ref-pill img {
  width: 20px;
  height: 20px;
  object-fit: cover;
  border-radius: 3px;
}

.image-composer__ref-add {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.image-composer__ref-add:hover {
  opacity: 1;
}

.image-composer__control-image-row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.image-composer__ref-pill--compact img {
  width: 32px;
  height: 32px;
}

.image-composer__control-hint {
  margin: 0;
  font-size: 10px;
  line-height: 1.4;
  color: var(--dq-color-text-tertiary);
}

.image-composer__control-hint--sub {
  opacity: 0.85;
}

/* Preset picker inside textarea (bottom-right) */
.image-composer__preset-area {
  position: absolute;
  bottom: 6px;
  right: 8px;
  z-index: 1;
  display: flex;
  align-items: center;
}

.image-composer__preset-btn {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.image-composer__preset-btn:hover {
  opacity: 1;
}

/* Toolbar - single row AI agent style */
.image-composer__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 4px;
  flex-wrap: nowrap;
}

.image-composer__toolbar-left,
.image-composer__toolbar-right {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: nowrap;
}

.image-composer__toolbar-left {
  overflow-x: auto;
  scrollbar-width: none;
}

.image-composer__toolbar-left::-webkit-scrollbar {
  display: none;
}

.image-composer__select {
  width: auto;
}

.image-composer__select--size {
  min-width: 70px;
}

.image-composer__select--batch {
  min-width: 50px;
}

/* Model dropdown item */
.image-composer__model-option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.image-composer__model-label {
  flex: 1;
}

.image-composer__model-badge {
  font-size: 10px;
  padding: 0 4px;
  height: 16px;
  line-height: 16px;
}

/* Size option with ratio + pixel */
.image-composer__size-ratio {
  font-weight: 500;
}

.image-composer__size-pixel {
  margin-left: 6px;
  font-size: 11px;
  color: var(--dq-label-tertiary);
  opacity: 0.8;
}

.image-composer__adv-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  position: relative;
}

.image-composer__dot {
  position: absolute;
  top: 1px;
  right: 1px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-warning);
  border: 1.5px solid var(--dq-surface);
}

/* Generate button */
.image-composer__generate {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 6px 14px;
  font-weight: 600;
  font-size: 13px;
}

/* Advanced panel (inside drawer) */
.image-composer__advanced-inner {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.image-composer__advanced-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.image-composer__field {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 260px;
  overflow: hidden;
}

.image-composer__field--full {
  flex: 1 1 100%;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

 .image-composer__field label {
  font-size: 12px;
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  width: 90px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.image-composer__field :deep(.dq-slider) {
  flex: 1;
}

.image-composer__field-val {
  font-size: 12px;
  color: var(--dq-label-secondary);
  min-width: 32px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.image-composer__seed-wrap {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* Hint - compact inline */
.image-composer__hint {
  text-align: right;
  margin-top: 2px;
  font-size: 10px;
  color: var(--dq-label-tertiary);
  opacity: 0;
  transition: opacity 0.2s;
  height: 0;
  overflow: hidden;
}

.image-composer__prompt-wrap:focus-within + .image-composer__hint,
.image-composer__prompt-wrap:hover + .image-composer__hint {
  opacity: 1;
  height: auto;
}
</style>
