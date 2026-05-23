<!-- @ts-nocheck -->
<template>
  <div class="create-page">
    <DqRow :gutter="24">
      <!-- Left panel: creation area -->
      <DqCol :xs="24" :md="16" :lg="17" :xl="18">
        <div class="creation-panel">

          <!-- Plan §2.1: top-level tabs (create / rewrite by reference / rewrite by instruction / retouch / extend / upscale) -->
          <DqSegmented
            class="dq-work-segmented"
            :model-value="imageWorkTab"
            :options="imageWorkSegmentOptions"
            block
            @update:model-value="setImageWorkMode"
          />
          <div v-if="editingSubModeDesc" class="studio-work-desc">
            {{ editingSubModeDesc }}
          </div>

          <!-- Model selector: single-level dropdown, recommended items first -->
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
                @change="onModelVersionChange"
                :placeholder="$t('studio.selectModel')"
              >
                <template v-if="selectedModelPickerItem" #value>
                  <div class="studio-picker-option studio-picker-option--value">
                    <span class="studio-picker-option__name">{{ selectedModelPickerItem.name }}</span>
                    <ModelVersionPickerExtras
                      :recommended="selectedModelPickerItem.recommended"
                      :commercial-use-allowed="selectedModelPickerItem.commercialUseAllowed"
                      :status="String(selectedModelPickerItem.status || '')"
                      :size="String(selectedModelPickerItem.size || '')"
                    />
                  </div>
                </template>
                <DqOption
                  v-for="item in filteredModelPickerVersions"
                  :key="item.modelKey + '|' + item.versionKey"
                  :label="String(item.name)"
                  :value="item.modelKey + '|' + item.versionKey"
                  :disabled="!item.ready"
                >
                  <ModelVersionPickerExtras
                    :recommended="item.recommended"
                    :commercial-use-allowed="item.commercialUseAllowed"
                    :status="String(item.status || '')"
                    :size="String(item.size || '')"
                  />
                </DqOption>
              </DqSelect>
              <DqButton
                class="studio-model-settings-btn"
                @click="goToSettings"
                :title="$t('studio.modelSettings')"
              >
                <DqIcon :size="14"><setting /></DqIcon>
              </DqButton>
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

          <!-- Prompt (not needed for upscale) -->
          <DqSurfaceCard v-if="editMode !== 'image_upscale'"
            class="studio-surface-card studio-card-mb"
          >
            <template #header>
              <div class="card-title">
                <DqIcon><edit-pen /></DqIcon>
                {{ $t('studio.prompt') }}
              </div>
            </template>

            <!-- Preset quick pick -->
            <DqRow :gutter="8" class="studio-presets-row">
              <DqCol :span="18" class="studio-presets-row__select">
                <DqSelect
                  v-model="selectedPreset"
                  :placeholder="$t('create.preset')"
                  class="studio-presets-row__control"
                  clearable
                >
                  <DqOption
                    v-for="(preset, name) in filteredPresets"
                    :key="name"
                    :label="presetSelectLabel(name, preset)"
                    :value="name"
                  />
                </DqSelect>
              </DqCol>
              <DqCol :span="6" class="studio-presets-row__action">
                <DqButton class="studio-presets-row__control" @click="loadPreset">
                  {{ $t('create.loadPreset') }}
                </DqButton>
              </DqCol>
            </DqRow>

            <DqInput
              v-model="params.prompt"
              type="textarea"
              :rows="5"
              :placeholder="$t('create.promptPlaceholder')"
              resize="none"
              @keydown.meta.enter.prevent="startGeneration"
              @keydown.ctrl.enter.prevent="startGeneration"
            />

            <!-- Negative prompt (only shown for models that support it) -->
            <DqCollapse v-if="currentModelConfig?.parameters?.negative_prompt_support" class="studio-collapse-plain">
              <DqCollapseItem :title="$t('studio.negativePrompt')" name="negative">
                <DqInput
                  v-model="params.negative_prompt"
                  type="textarea"
                  :rows="2"
                  :placeholder="$t('create.negativePlaceholder')"
                />
              </DqCollapseItem>
            </DqCollapse>
            <div v-if="editMode === 'image_editing' && editingSubMode === 'outpainting'" class="studio-extend-panel">
              <div class="studio-extend-panel-title">{{ $t('create.extendPanelTitle') }}</div>
              <CreateExtendParams :params="params" />
            </div>
          </DqSurfaceCard>

          <!-- Upscale params (plan §6.3 /image/upscales) -->
          <DqSurfaceCard v-if="editMode === 'image_upscale'" class="studio-surface-card studio-card-mb">
            <template #header>
              <div class="card-title">
                <DqIcon><zoom-in /></DqIcon>
                {{ $t('action.image.upscale') }}
              </div>
            </template>
            <CreateUpscaleParams :params="params" media="image" />
          </DqSurfaceCard>

          <!-- Advanced params (collapsible) -->
          <DqSurfaceCard v-if="editMode !== 'image_upscale'"
            class="studio-surface-card studio-card-mb"
          >
            <DqCollapse v-model="advancedParamsOpen" class="studio-collapse-plain">
              <DqCollapseItem name="advanced">
                <template #title>
                  <div class="studio-collapse-title-row">
                    <DqIcon><setting /></DqIcon>
                    <span>{{ $t('studio.advancedParams') }}</span>
                    <DqTag v-if="hasCustomParams" size="small" type="warning">{{ $t('studio.hasCustom') }}</DqTag>
                  </div>
                </template>

                <ImageCreateAdvancedParams
                  :params="params"
                  :current-model-config="currentModelConfig"
                  :edit-mode="editMode"
                  :compatible-loras="compatibleLoras"
                  :compatible-control-nets="compatibleControlNets"
                  :control-image-src="controlImageSrc"
                  :recent-images="recentImages"
                  @reset-to-defaults="resetToDefaults"
                  @control-asset-pick="onControlAssetPick"
                  @remove-control-image="removeControlImage"
                />
              </DqCollapseItem>
            </DqCollapse>
          </DqSurfaceCard>

          <!-- Main action (plan §2.3: primary button + queue hint) -->
          <DqSurfaceCard class="studio-surface-card studio-card-mb">
            <DqButton
              type="primary"
              class="studio-primary-cta dq-btn--cta"
              :disabled="submitDisabled"
              @click="startGeneration"
            >
              <DqIcon size="20"><magic-stick /></DqIcon>
              <span class="studio-cta-gap">{{ primaryCtaLabel }}</span>
            </DqButton>
            <div class="studio-micro-hint">
              {{ $sendShortcutHint() }}
            </div>

            <!-- Progress display -->
            <div v-if="currentTask" class="studio-task-wrap">
              <div v-if="currentTask.status === 'submitting'" class="studio-task-submitting">
                <DqIcon class="studio-spin-icon" size="16"><loading /></DqIcon>
                {{ $tt('studio.submitting') }}
              </div>
              <template v-else>
                <DqProgress
                  :percentage="Math.round(currentTask.progress * 100)"
                  :status="currentTask.status === 'failed' ? 'exception' : ''"
                />
                <div class="studio-task-status">
                  <template v-if="currentTask.total > 0 && currentTask.status === 'running'">
                    Step {{ currentTask.step }}/{{ currentTask.total }} &nbsp;
                  </template>
                  <DqTag :type="getStatusType(currentTask.status)" size="small">
                    {{ getStatusText(currentTask.status) }}
                  </DqTag>
                </div>
              </template>
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

      <!-- Right panel -->
      <DqCol :xs="24" :md="8" :lg="7" :xl="6">
        <div class="preview-panel preview-panel--flat">

          <StudioPreviewPane
            v-if="editMode === 'image_editing'"
            class="studio-preview-pane--source"
            :title="$t('create.imageInput')"
            icon="picture-filled"
            split-head
          >
            <template #actions>
              <asset-picker
                accept-kind="image"
                :recent-gallery="recentImages"
                @pick="onEditAssetPick"
              />
            </template>
            <image-editor
              ref="imageEditorRef"
              :src="editImageSrc"
              :recent-gallery="recentImages"
              mode="inpainting"
              @pick-edit-source="onEditAssetPick"
            />
          </StudioPreviewPane>

          <StudioPreviewPane
            v-else-if="editMode === 'image_upscale'"
            class="studio-preview-pane--source"
            :title="$t('create.imageInput')"
            icon="picture-filled"
            split-head
          >
            <template #actions>
              <asset-picker
                accept-kind="image"
                :recent-gallery="recentImages"
                @pick="onEditAssetPick"
              />
            </template>
            <div v-if="editImageSrc" class="image-preview studio-preview-sq">
              <img class="studio-preview-media" :src="editImageSrc" alt="upscale source" />
            </div>
            <DqEmpty v-else :description="$t('studio.uploadEditImage')" />
          </StudioPreviewPane>

          <StudioPreviewPane :title="$t('studio.currentPreview')" icon="picture-filled">
            <CreateImagePreview
              v-if="previewImage"
              :src="previewImage"
              :hue="previewImageHue"
              :alt="previewCaption"
              @download="downloadPreviewImage"
              @expand="openCurrentPreviewDialog"
            />
            <DqEmpty v-else class="studio-preview-pane__empty" :description="$t('studio.noPreview')" />
            <p v-if="previewImage && previewCaption" class="studio-preview-pane__caption" :title="previewCaption">
              {{ previewCaption }}
            </p>
          </StudioPreviewPane>

          <StudioPreviewPane :title="$t('studio.recent')" icon="clock" split-head recent>
            <template #actions>
              <DqIconButton type="text" size="sm" :label="$t('gallery.refresh')" @click="loadRecentImages">
                <DqIcon><refresh /></DqIcon>
              </DqIconButton>
            </template>
            <DqEmpty v-if="recentImages.length === 0" :description="$t('gallery.empty')" />
            <div v-else class="studio-recent-grid">
              <div
                v-for="image in recentImages"
                :key="image.path"
                class="studio-recent-grid__item gallery-card"
              >
                <div class="gallery-image-wrapper studio-preview-sq" @click="showPreview(image)">
                  <img
                    v-if="!recentGalleryThumbFailed[String(image.path)]"
                    :src="getImageUrl(image)"
                    :alt="String(image.name || '')"
                    loading="lazy"
                    @error="markRecentGalleryThumbFailed(image)"
                  />
                  <div v-else class="gallery-thumb-fallback studio-recent-thumb-fallback">
                    <DqIcon :size="44"><Picture /></DqIcon>
                  </div>
                </div>
                <div class="recent-actions">
                  <DqButton class="action-btn rewrite-btn" size="sm" @click.stop="quickFromGallery(image, 'rewrite')">
                    <DqIcon :size="12"><brush /></DqIcon>
                    <span>{{ $t('studio.quickRewrite') }}</span>
                  </DqButton>
                  <DqButton class="action-btn upscale-btn" size="sm" @click.stop="quickFromGallery(image, 'upscale')">
                    <DqIcon :size="12"><zoom-in /></DqIcon>
                    <span>{{ $t('studio.quickUpscale') }}</span>
                  </DqButton>
                </div>
              </div>
            </div>
          </StudioPreviewPane>
        </div>
      </DqCol>
    </DqRow>

    <!-- Image preview dialog -->
    <DqDialog v-model:open="previewVisible" :title="selectedImage?.name ?? ''" width="70%" center>
      <div v-if="selectedImage" class="studio-dialog-center">
        <img class="studio-dialog-img" :src="getImageUrl(selectedImage)" />
      </div>
    </DqDialog>
    <DqDialog v-model:open="currentPreviewDialogOpen" :title="$t('studio.currentPreview')" width="78%" center>
      <div v-if="previewImage" class="studio-dialog-center">
        <img class="studio-dialog-img-tall" :src="previewImage" :alt="previewCaption" />
      </div>
    </DqDialog>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, watch, onMounted, inject, nextTick, unref } from 'vue';
import type { Ref } from 'vue';
import { useRouter } from 'vue-router';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt, $mn, $mvn, $pn } from '@/utils/i18n';
import { DQ_STORAGE } from '@/utils/storage';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import type { SystemInfo } from '@/types';
import { applyDefaults, hasDeviation } from '@/utils/registryParamSchema';
import { pickDefaultVersionKey, resolveDefaultModelRegistryKey } from '@/utils/defaultModelSettings';
import { warnIfRiskyMemory } from '@/composables/memoryHint';
import { formatGenLogMessage, isDuplicateDenoiseStepLog } from '@/utils/genTaskLog';
import ModelLicenseBadges from '@/components/model/ModelLicenseBadges.vue';
import ModelPickerFilters from '@/components/model/ModelPickerFilters.vue';
import ModelVersionPickerExtras from '@/components/model/ModelVersionPickerExtras.vue';
import { useModelRegistryFilters, reconcileVersionPickerSelection } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import ImageEditor from '@/components/image/ImageEditor.vue';
import ImageCreateAdvancedParams from '@/components/create/ImageCreateAdvancedParams.vue';
import CreateUpscaleParams from '@/components/create/CreateUpscaleParams.vue';
import CreateExtendParams from '@/components/create/CreateExtendParams.vue';
import StudioPreviewPane from '@/components/create/StudioPreviewPane.vue';
import CreateImagePreview from '@/components/create/CreateImagePreview.vue';
import { Picture } from '@danqing/dq-shell';

/* ------------------------------------------------------------------ */
/*  Injected / External                                                */
/* ------------------------------------------------------------------ */

const systemInfo = inject<Ref<SystemInfo>>('systemInfo');

const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const router = useRouter();

/* ------------------------------------------------------------------ */
/*  RegistryActions helpers (inlined from legacy registry_actions.js)  */
/* ------------------------------------------------------------------ */

function hasAction(actions: Record<string, unknown>, key: string): boolean {
  if (!actions || typeof actions !== 'object') return false;
  return Object.prototype.hasOwnProperty.call(actions, key) && actions[key] != null;
}
function imageSupportsCreate(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'create');
}
function imageSupportsUpscale(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'upscale');
}
function imageEditingMatches(actions: Record<string, unknown>, subMode: string): boolean {
  if (subMode === 'inpainting') {
    return hasAction(actions, 'retouch') || hasAction(actions, 'rewrite');
  }
  if (subMode === 'outpainting') {
    return hasAction(actions, 'extend') || hasAction(actions, 'retouch');
  }
  return hasAction(actions, 'rewrite');
}
function imageModelRow(config: Record<string, unknown>): boolean {
  return config && config.media === 'image' && config.category !== 'loras';
}

/* ------------------------------------------------------------------ */
/*  DQTaskStatusUi helpers (inlined from legacy task_status_ui.js)     */
/* ------------------------------------------------------------------ */

function getStatusType(status: string): string {
  const map: Record<string, string> = {
    pending: 'info',
    queued: 'info',
    submitting: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info',
  };
  return map[status] || 'info';
}
function getStatusText(status: string): string {
  const map: Record<string, string> = {
    pending: 'studio.pending',
    queued: 'studio.queued',
    submitting: 'studio.submitting',
    running: 'studio.running',
    completed: 'studio.completed',
    failed: 'studio.failed',
    cancelled: 'studio.cancelled',
  };
  const key = map[status] || `studio.${status}`;
  return $tt(key);
}

/* ------------------------------------------------------------------ */
/*  DQModelVersionValue helper (inlined from legacy model_version_value.js) */
/* ------------------------------------------------------------------ */

function parseModelVersionValue(value: string): { modelKey: string; versionKey: string } | null {
  if (!value || typeof value !== 'string') return null;
  const parts = value.split('|');
  if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
  return { modelKey: parts[0], versionKey: parts[1] };
}

/* ------------------------------------------------------------------ */
/*  Params (including advanced params)                                 */
/* ------------------------------------------------------------------ */

const params = reactive<Record<string, unknown>>({
  prompt: '',
  negative_prompt: '',
  model: '',
  version: '',
  steps: 4,
  guidance: 3.5,
  width: 1024,
  height: 1024,
  lora: '',
  lora_scale: 0.8,
  seed: '',
  strength: 0.4,
  img2img: false,
  controlnet: '',
  controlnet_strength: 0.8,
  scheduler: 'flow_match_euler_discrete',
  upscale_scale: 2,
  upscale_denoise: 0.3,
  upscale_tile: 1024,
  extend_directions: ['right'],
  extend_pixels: 256,
});

// Selected model+version combo (format: "modelKey|versionKey")
const selectedModelVersion = ref('');

// State
const generating = ref(false);
const currentTask = ref<Record<string, unknown> | null>(null);
const logs = ref<Array<{ time: string; message: string; level: string }>>([]);
const logContainer = ref<HTMLElement | null>(null);
/** Last denoise step mirrored into the log card from SSE progress (avoids empty panel until DB catches up) */
const genLogLastStep = ref(0);
/** 'denoise' | 'post' — log post-phase line once when SSE reports message post */
const genLogLastPhase = ref('');
const previewImage = ref('');
const currentPreviewDialogOpen = ref(false);

const previewCaption = computed(() => (params.prompt || '').trim());

const previewImageHue = computed(() => {
  let h = 0;
  const s = previewCaption.value || 'image';
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) % 360;
  }
  return h;
});

function openCurrentPreviewDialog() {
  if (!previewImage.value) return;
  currentPreviewDialogOpen.value = true;
}

function downloadPreviewImage() {
  if (!previewImage.value) return;
  const a = document.createElement('a');
  a.href = previewImage.value;
  a.download = 'image.png';
  a.click();
}
const recentImages = ref<Array<Record<string, unknown>>>([]);
/** 最近生成缩略图加载失败时避免浏览器默认裂图，改为统一占位 */
const recentGalleryThumbFailed = ref<Record<string, boolean>>({});

function isDefinitelyNonRasterRecent(v: Record<string, unknown>): boolean {
  const meta = v.metadata as Record<string, unknown> | undefined;
  if (meta?.asset_kind === 'video' || meta?.asset_kind === 'audio') return true;
  const ext = String(v.name || '').split('.').pop()?.toLowerCase() || '';
  return ['mp4', 'mov', 'avi', 'mkv', 'webm', 'wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
}

const markRecentGalleryThumbFailed = (image: Record<string, unknown>) => {
  const p = String(image.path || '');
  if (!p) return;
  recentGalleryThumbFailed.value = { ...recentGalleryThumbFailed.value, [p]: true };
};
const advancedParamsOpen = ref<string[]>(['advanced']);
const compatibleLoras = ref<Array<Record<string, unknown>>>([]);
const compatibleControlNets = ref<Array<Record<string, unknown>>>([]);
const controlImageSrc = ref('');
const controlImagePath = ref('');

// Mode: top-level tab and engine sub-mode (rewrite split into reference / instruct)
const editMode = ref('image_generation'); // image_generation | image_editing | image_upscale
const imageWorkTab = ref('create'); // create | rewrite_reference | rewrite_instruct | retouch | extend | upscale
const imageWorkSegmentOptions = computed(() => [
  { label: $tt('action.image.create'), value: 'create' },
  { label: $tt('create.rewriteDriveReference'), value: 'rewrite_reference' },
  { label: $tt('create.rewriteDriveInstruct'), value: 'rewrite_instruct' },
  { label: $tt('action.image.retouch'), value: 'retouch' },
  { label: $tt('action.image.extend'), value: 'extend' },
  { label: $tt('action.image.upscale'), value: 'upscale' },
]);
const editingSubMode = ref('inpainting'); // inpainting | text_editing | outpainting
/** Aligned with API rewrite_mode; driven by imageWorkTab */
const rewriteDriveMode = ref('reference');

const setImageWorkMode = (mode: string) => {
  if (mode === 'create') {
    editMode.value = 'image_generation';
    imageWorkTab.value = 'create';
  } else if (mode === 'upscale') {
    editMode.value = 'image_upscale';
    imageWorkTab.value = 'upscale';
  } else if (mode === 'rewrite' || mode === 'rewrite_reference') {
    editMode.value = 'image_editing';
    imageWorkTab.value = 'rewrite_reference';
    editingSubMode.value = 'text_editing';
    rewriteDriveMode.value = 'reference';
  } else if (mode === 'rewrite_instruct') {
    editMode.value = 'image_editing';
    imageWorkTab.value = 'rewrite_instruct';
    editingSubMode.value = 'text_editing';
    rewriteDriveMode.value = 'instruct';
  } else if (mode === 'retouch') {
    editMode.value = 'image_editing';
    imageWorkTab.value = 'retouch';
    editingSubMode.value = 'inpainting';
  } else if (mode === 'extend') {
    editMode.value = 'image_editing';
    imageWorkTab.value = 'extend';
    editingSubMode.value = 'outpainting';
  }
};

// Local redraw: image editor
const editImageSrc = ref('');
const editImagePath = ref('');
const imageEditorRef = ref<Record<string, unknown> | null>(null);

// Presets
const presets = ref<Record<string, Record<string, unknown>>>({});
const selectedPreset = ref('');

const presetActionFilter = computed(() => {
  if (editMode.value === 'image_upscale') {
    return new Set(['upscale']);
  }
  if (editMode.value === 'image_generation') {
    return new Set(['create']);
  }
  if (editingSubMode.value === 'inpainting') {
    return new Set(['retouch', 'rewrite']);
  }
  if (editingSubMode.value === 'outpainting') {
    return new Set(['extend', 'retouch']);
  }
  return new Set(['rewrite']);
});

const filteredPresets = computed(() => {
  const want = presetActionFilter.value;

  function planPresetShapeOk(preset: Record<string, unknown>) {
    return (
      Array.isArray(preset.applies_to) &&
      (preset.applies_to as unknown[]).length > 0 &&
      (preset.media_scope === 'image' || preset.media_scope === 'video')
    );
  }

  function matchesMediaScope(preset: Record<string, unknown>) {
    return preset.media_scope === 'image';
  }

  function matches(preset: Record<string, unknown>) {
    if (!planPresetShapeOk(preset)) return false;
    if (!matchesMediaScope(preset)) return false;
    return (preset.applies_to as string[]).some((k: string) => want.has(k));
  }

  const entries = Object.entries(presets.value)
    .filter(([, preset]) => matches(preset))
    .sort((a, b) => {
      const aCreate = (a[1].applies_to as string[]).includes('create');
      const bCreate = (b[1].applies_to as string[]).includes('create');
      if (aCreate !== bCreate) {
        return aCreate ? -1 : 1;
      }
      return a[0].localeCompare(b[0], 'zh');
    });
  const result: Record<string, Record<string, unknown>> = {};
  for (const [name, preset] of entries) {
    result[name] = preset;
  }
  return result;
});

// Model registry
const modelRegistry = ref<Record<string, Record<string, unknown>>>({});

// Model readiness status
const modelsStatus = ref<Record<string, unknown>>({});
const modelsDetailedStatus = ref<Record<string, { versions?: Record<string, { ready?: boolean; status?: string }> }>>({});

// All model versions (flattened list)
const allVersions = computed(() => {
  const result: Array<Record<string, unknown>> = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!imageModelRow(config)) {
      continue;
    }
    const actions = { ...(config.actions as Record<string, unknown> || {}) };
    const engine = config.engine || '';
    const versions = config.versions || { default: { name: 'Default', size: '', default: true } };
    const detailed = modelsDetailedStatus.value[modelKey] || {};
    const versionStatuses = detailed.versions || {};

    for (const [versionKey, versionConfig] of Object.entries(versions as Record<string, Record<string, unknown>>)) {
      const status = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
      const size = (versionConfig as Record<string, unknown>).size || '';

      result.push({
        modelKey,
        versionKey,
        name: $mvn(modelKey, config as { name?: string | { zh?: string; en?: string }; name_en?: string }, versionConfig as { name?: string | { zh?: string; en?: string } }),
        size,
        status: status.status,
        ready: status.ready,
        recommended: config.recommended && (versionConfig as Record<string, unknown>).default,
        commercialUseAllowed: config.commercial_use_allowed === true,
        actions,
        engine,
      });
    }
  }
  return result;
});

// Recommended versions
const recommendedVersions = computed(() => {
  return allVersions.value.filter((v) => v.recommended);
});

// Filter models by mode
const filteredAllVersions = computed(() => {
  if (editMode.value === 'image_editing') {
    return allVersions.value.filter((v) => {
      const acts = v.actions as Record<string, unknown> || {};
      if (imageWorkTab.value === 'rewrite_instruct') {
        return imageSupportsCreate(acts) && v.modelKey === 'flux1-kontext';
      }
      return imageEditingMatches(acts, editingSubMode.value);
    });
  }
  if (editMode.value === 'image_upscale') {
    return allVersions.value.filter((v) => {
      const acts = v.actions as Record<string, unknown> || {};
      return imageSupportsUpscale(acts);
    });
  }
  return allVersions.value.filter((v) => {
    const acts = v.actions as Record<string, unknown> || {};
    return imageSupportsCreate(acts);
  });
});

const filteredRecommendedVersions = computed(() => {
  return filteredAllVersions.value.filter((v) => v.recommended);
});

const selectedModelPickerItem = computed(() => {
  const key = selectedModelVersion.value;
  if (!key) return null;
  return allVersions.value.find((item) => `${item.modelKey}|${item.versionKey}` === key) ?? null;
});

/** Model dropdown: single-layer list, shows only ready models, recommended versions first */
const { commercialOnly: modelFilterCommercialOnly } = useModelRegistryFilters();

const filteredModelPickerVersions = computed(() => {
  const rows = applyModelVersionFilters(filteredAllVersions.value, {
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

// Current model config
const currentModelConfig = computed(() => modelRegistry.value[params.model as string] || null);

const currentModelDisplayName = computed(() => {
  const c = currentModelConfig.value;
  if (c) {
    return $mn(c as { name?: string | { zh?: string; en?: string }; name_en?: string }, params.model as string);
  }
  return params.model || '';
});

// Whether current selected version is ready
const selectedModelNotReady = computed(() => {
  if (!params.model || !params.version) return false;
  const detailed = modelsDetailedStatus.value[params.model as string];
  if (!detailed || !detailed.versions) return true;
  const versionStatus = detailed.versions[params.version as string];
  return !versionStatus || !versionStatus.ready;
});

// Edit sub-type description
const editingSubModeDesc = computed(() => {
  if (editMode.value === 'image_upscale') {
    return $tt('action.image.upscaleDesc');
  }
  if (editMode.value === 'image_generation') {
    return $tt('action.image.createDesc');
  }
  if (editMode.value === 'image_editing' && editingSubMode.value === 'text_editing') {
    return rewriteDriveMode.value === 'instruct'
      ? $tt('create.rewriteDriveInstructDesc')
      : $tt('create.rewriteDriveReferenceDesc');
  }
  const descMap: Record<string, string> = {
    inpainting: $tt('action.image.retouchDesc'),
    outpainting: $tt('action.image.extendDesc'),
  };
  return descMap[editingSubMode.value] || '';
});

const submitDisabled = computed(() => {
  if (selectedModelNotReady.value) return true;
  if (editMode.value === 'image_upscale') {
    return !editImageSrc.value;
  }
  return !String(params.prompt || '').trim();
});

const primaryCtaLabel = computed(() => {
  if (editMode.value === 'image_generation') {
    return $tt('studio.generate');
  }
  if (editMode.value === 'image_upscale') {
    return $tt('action.image.upscale');
  }
  if (editingSubMode.value === 'text_editing') {
    return rewriteDriveMode.value === 'instruct'
      ? $tt('create.rewriteDriveInstruct')
      : $tt('create.rewriteDriveReference');
  }
  if (editingSubMode.value === 'inpainting') {
    return $tt('action.image.retouch');
  }
  if (editingSubMode.value === 'outpainting') {
    return $tt('action.image.extend');
  }
  return $tt('studio.generate');
});

// Load model registry and status
const loadModelRegistry = async () => {
  try {
    const RS = registryStore;
    const regPromise = RS && RS.load
      ? RS.load()
      : api.settings.getModelRegistry().then((r: Record<string, unknown>) => ({ models: (r as Record<string, unknown>).models }));
    const [registryData, statusData, detailedStatusData] = await Promise.all([
      regPromise,
      api.settings.getModelsStatus(),
      api.settings.getModelsDetailedStatus(),
    ]);

    modelRegistry.value = (registryData as Record<string, unknown>).models || {};
    modelsStatus.value = statusData || {};
    modelsDetailedStatus.value = detailedStatusData || {};

    // Set default model+version (prefer ready recommended version's default)
    if (!selectedModelVersion.value) {
      let found = false;
      for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
        if (config.recommended) {
          const detailed = (detailedStatusData as Record<string, Record<string, unknown>>)[modelKey] || {};
          const versions = detailed.versions || {};
          const defaultVersionKey = Object.keys(config.versions || {}).find((k) => (config.versions as Record<string, Record<string, unknown>>)[k]?.default) || Object.keys(config.versions || {})[0];

          if (defaultVersionKey && versions[defaultVersionKey]?.ready) {
            params.model = modelKey;
            params.version = defaultVersionKey;
            selectedModelVersion.value = modelKey + '|' + defaultVersionKey;
            found = true;
            break;
          }
        }
      }

      if (!found) {
        for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
          const detailed = (detailedStatusData as Record<string, Record<string, unknown>>)[modelKey] || {};
          const versions = detailed.versions || {};
          for (const versionKey of Object.keys(config.versions || {})) {
            if (versions[versionKey]?.ready) {
              params.model = modelKey;
              params.version = versionKey;
              selectedModelVersion.value = modelKey + '|' + versionKey;
              found = true;
              break;
            }
          }
          if (found) break;
        }
      }

      if (!found) {
        const firstModel = Object.keys(modelRegistry.value)[0];
        if (firstModel) {
          const firstVersion = Object.keys(modelRegistry.value[firstModel].versions || {})[0] || 'default';
          params.model = firstModel;
          params.version = firstVersion;
          selectedModelVersion.value = firstModel + '|' + firstVersion;
        }
      }
    }

    loadModelDefaults();
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
};

// Load model default config (registry schema driven)
const loadModelDefaults = () => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return;
  applyDefaults(config.parameters as Record<string, unknown>, params);
  controlImageSrc.value = '';
  controlImagePath.value = '';
  loadCompatibleLoras();
  loadCompatibleControlNets();
};

// Load LoRAs compatible with current model
const loadCompatibleLoras = async () => {
  if (!params.model) return;
  try {
    const loras = await api.settings.getCompatibleLoras(params.model as string);
    compatibleLoras.value = (loras as Array<Record<string, unknown>>) || [];
  } catch (e) {
    console.error('Failed to load compatible loras:', e);
    compatibleLoras.value = [];
  }
};

// Load ControlNets compatible with current model
const loadCompatibleControlNets = async () => {
  if (!params.model) return;
  try {
    const nets = await api.settings.getCompatibleControlNets(params.model as string);
    compatibleControlNets.value = (nets as Array<Record<string, unknown>>) || [];
    // If current selected ControlNet is not in the returned list (incompatible or deleted), clear it
    if (params.controlnet && !(nets as Array<Record<string, unknown>>).some((n: Record<string, unknown>) => n.key === params.controlnet)) {
      params.controlnet = '';
      controlImageSrc.value = '';
      controlImagePath.value = '';
    }
  } catch (e) {
    console.error('Failed to load compatible controlnets:', e);
    compatibleControlNets.value = [];
  }
};

// Reset to default config (reload from registry)
const resetToDefaults = () => {
  loadModelDefaults();
  toast.success($tt('studio.restoredDefaults'));
};

const hasCustomParams = computed(() => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return false;
  return hasDeviation(config.parameters as Record<string, unknown>, params);
});

const presetSelectLabel = (name: string, preset: Record<string, unknown>) => {
  const a = preset.applies_to as string[];
  const hasC = a.includes('create');
  const hasEdit = a.some((x: string) => ['rewrite', 'retouch', 'extend'].includes(x));
  let tag = '';
  if (hasC && !hasEdit) tag = $tt('create.presetTagT2I');
  else if (hasEdit && !hasC) tag = $tt('create.presetTagI2I');
  const display = $pn(preset as { name_en?: string }, name);
  return tag ? `${tag} ${display}` : display;
};

// Load presets
const loadPresets = async () => {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as Record<string, Record<string, unknown>>) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
    presets.value = {};
  }
};

// Load preset into params (append to new line)
const loadPreset = () => {
  if (!selectedPreset.value || !presets.value[selectedPreset.value]) return;

  const preset = presets.value[selectedPreset.value];

  if (preset.positive) {
    params.prompt = params.prompt
      ? params.prompt + '\nStyle boost: ' + preset.positive
      : String(preset.positive);
  }
  if (preset.negative) {
    params.negative_prompt = params.negative_prompt
      ? params.negative_prompt + '\n' + preset.negative
      : String(preset.negative);
  }

  // Edit-only presets (without create): prompt to switch tab and select source image if on T2I tab
  const app = preset.applies_to as string[];
  const needsEditSource =
    !app.includes('create') &&
    app.some((x: string) => ['rewrite', 'retouch', 'extend'].includes(x));
  if (needsEditSource && editMode.value === 'image_generation') {
    toast.warning($tt('create.presetNeedsEditTab'));
  }
};

// Add log
const addLog = (message: string, level = 'info') => {
  const now = new Date();
  const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
  logs.value.push({ time, message, level });

  if (logs.value.length > 500) {
    logs.value = logs.value.slice(-500);
  }

  nextTick(() => {
    const container = logContainer.value;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  });
};

function ingestServerLog(logData: Record<string, unknown>) {
  const raw = (logData.message || '') as string;
  const lvl = (logData.level || 'info') as string;
  if (isDuplicateDenoiseStepLog(logs.value, raw)) {
    return;
  }
  addLog(formatGenLogMessage(raw), lvl);
}

// Clear logs
const clearLogs = () => {
  logs.value = [];
};

// Truncate text
const truncate = (text: string, length: number) => {
  if (!text) return '';
  return text.length > length ? text.substring(0, length) + '...' : text;
};

// Start generation
const startGeneration = async () => {
  if (editMode.value === 'image_upscale') {
    if (!editImageSrc.value) {
      toast.warning($tt('studio.uploadEditImage'));
      return;
    }
  } else if (!params.prompt) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }

  // ControlNet requires an uploaded control image
  if (editMode.value !== 'image_upscale' && params.controlnet && !controlImageSrc.value) {
    toast.warning($tt('studio.needControlImage'));
    return;
  }

  const detailed = modelsDetailedStatus.value[params.model as string];
  const versionStatus = detailed?.versions?.[params.version as string];
  if (!versionStatus?.ready) {
    toast.warning(
      $tt('studio.modelNotReadyDesc', { name: currentModelDisplayName.value, version: params.version as string }),
    );
    return;
  }

  const verCfg = (currentModelConfig.value && currentModelConfig.value.versions && (currentModelConfig.value.versions as Record<string, Record<string, unknown>>)[params.version as string]) || null;
  const sizeHuman = verCfg && verCfg.size ? String(verCfg.size) : '';
  const minMemRaw = currentModelConfig.value?.parameters?.min_unified_memory_gb;
  const minUnifiedMemoryGb =
    minMemRaw != null && Number(minMemRaw) > 0 ? Number(minMemRaw) : null;
  warnIfRiskyMemory({
    systemInfo: unref(systemInfo),
    versionSizeHuman: sizeHuman,
    minUnifiedMemoryGb,
    $tt,
  });

  // Immediately show "submitting" progress to give visual feedback
  generating.value = true;
  currentTask.value = {
    id: '',
    progress: 0,
    step: 0,
    total: 0,
    status: 'submitting',
  };

  const modelStr = params.version ? `${params.model}:${params.version}` : params.model;
  const adapters = params.lora ? [{ id: params.lora, weight: params.lora_scale || 0.8 }] : [];
  const _seedParsed =
    params.seed != null && params.seed !== '' ? parseInt(String(params.seed), 10) : null;
  const seedNum = Number.isFinite(_seedParsed) ? _seedParsed : null;
  const meta: Record<string, unknown> = {};
  if (params.scheduler) {
    meta.scheduler = params.scheduler;
  }

  const attachStreamFromSubmit = (submitRes: unknown) => {
    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) {
      addLog($tt('studio.error', { msg: 'missing task id in submit response' }), 'error');
      return;
    }
    attachStream(tid);
  };

  const attachStream = (tid: string) => {
    genLogLastStep.value = 0;
    genLogLastPhase.value = '';
    currentTask.value = {
      id: tid,
      progress: 0,
      step: 0,
      total: 0,
      status: 'queued',
      params: {
        model: modelStr,
        prompt: editMode.value === 'image_upscale' ? '' : params.prompt,
      },
    };
    api.gen.streamMediaTask(tid, {
      onLog: (logData: unknown) => ingestServerLog(logData as Record<string, unknown>),
      onStatus: (statusData: unknown) => {
        if (currentTask.value) {
          currentTask.value.progress = (statusData as Record<string, unknown>).progress ?? 0;
          currentTask.value.status = (statusData as Record<string, unknown>).status;
        }
      },
      onDone: async (doneData: unknown) => {
        const data = doneData as Record<string, unknown>;
        if (data.status === 'completed') {
          addLog($tt('studio.genComplete'), 'success');
          const updated = await api.gen.getMediaTask(tid);
          currentTask.value = updated as Record<string, unknown>;
          const pid = (updated as Record<string, unknown>).result && ((updated as Record<string, unknown>).result as Record<string, unknown>).primary_asset_id;
          if (pid) {
            previewImage.value = api.gallery.getImageUrl(`asset:${pid}`);
            addLog($tt('studio.outputFile', { name: String(pid) }), 'info');
          }
          loadRecentImages();
        } else if (data.status === 'failed') {
          const updated = await api.gen.getMediaTask(tid);
          currentTask.value = updated as Record<string, unknown>;
          addLog($tt('studio.genFailed', { msg: String(((updated as Record<string, unknown>).error_message || '')) }), 'error');
        }
      },
      onError: () => addLog($tt('studio.connectionLost'), 'warning'),
      onProgress: (progressData: unknown) => {
        const data = progressData as Record<string, unknown>;
        if (!currentTask.value) return;
        if (typeof data.progress === 'number') {
          currentTask.value.progress = data.progress;
        }
        const nextStep =
          data.step != null ? data.step : currentTask.value.step;
        const nextTotal =
          data.total != null ? data.total : currentTask.value.total;
        currentTask.value.step = nextStep;
        currentTask.value.total = nextTotal;
        if (data.message === 'post') {
          if (genLogLastPhase.value !== 'post') {
            genLogLastPhase.value = 'post';
            addLog($tt('studio.queuePostProcessHint'), 'info');
          }
        } else if (data.message === 'denoise') {
          genLogLastPhase.value = 'denoise';
        }
        if (nextTotal > 0 && nextStep > 0) {
          genLogLastStep.value = nextStep as number;
        }
      },
    });
  };

  addLog($tt('studio.startingGen'), 'info');
  try {
    if (editMode.value === 'image_upscale') {
      const ep = editImagePath.value;
      let source_asset_id: string;
      if (typeof ep === 'string' && ep.startsWith('asset:')) {
        source_asset_id = ep.slice('asset:'.length);
      } else {
        const srcBlob = await api.gen.urlToBlob(editImageSrc.value);
        source_asset_id = (
          await api.gen.uploadAsset(
            new File([srcBlob], 'upscale-src.png', { type: srcBlob.type || 'image/png' })
          )
        ).id as string;
      }
      const sc = Number(params.upscale_scale) === 4 ? 4 : 2;
      const submitRes = await api.gen.createImageUpscale({
        model: modelStr,
        source_asset_id,
        scale: sc,
        denoise: Number(params.upscale_denoise) || 0.3,
        tile_size: Number(params.upscale_tile) || 1024,
        priority: 'normal',
        metadata: {},
      });
      attachStreamFromSubmit(submitRes);
      return;
    }

    if (editMode.value === 'image_editing') {
      if (!editImageSrc.value) {
        toast.warning($tt('studio.uploadEditImage'));
        return;
      }
      const maskBlob = imageEditorRef.value ? await (imageEditorRef.value as { getMaskBlob: () => Promise<Blob> }).getMaskBlob() : null;
      if (!maskBlob && editingSubMode.value !== 'text_editing' && editingSubMode.value !== 'outpainting') {
        toast.warning($tt('studio.drawMask'));
        return;
      }
      const ep = editImagePath.value;
      let source_asset_id: string;
      if (typeof ep === 'string' && ep.startsWith('asset:')) {
        source_asset_id = ep.slice('asset:'.length);
      } else {
        const srcBlob = await api.gen.urlToBlob(editImageSrc.value);
        source_asset_id = (
          await api.gen.uploadAsset(
            new File([srcBlob], 'source.png', { type: srcBlob.type || 'image/png' })
          )
        ).id as string;
      }
      let mask_asset_id = null;
      if (maskBlob) {
        mask_asset_id = (
          await api.gen.uploadAsset(new File([maskBlob], 'mask.png', { type: 'image/png' }))
        ).id;
      }

      let operation = 'rewrite';
      if (editingSubMode.value === 'inpainting') {
        operation = 'retouch';
      } else if (editingSubMode.value === 'outpainting') {
        operation = 'extend';
      } else if (editingSubMode.value === 'text_editing') {
        operation = 'rewrite';
      }

      let extendSpec = undefined;
      if (operation === 'extend') {
        const dirs = Array.isArray(params.extend_directions)
          ? (params.extend_directions as string[]).filter((d: string) => ['top', 'bottom', 'left', 'right'].includes(d))
          : [];
        if (!dirs.length) {
          toast.warning($tt('create.extendNeedDirection'));
          return;
        }
        const px = Math.min(2048, Math.max(64, Number(params.extend_pixels) || 256));
        extendSpec = { directions: dirs, pixels: px };
      }

      const editBody: Record<string, unknown> = {
        model: modelStr,
        operation,
        source_asset_id,
        mask_asset_id,
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        source_fidelity: Math.min(0.95, Math.max(0.05, 1 - (params.strength ?? 0.4))),
        extend: extendSpec,
        n: 1,
        steps: params.steps,
        seed: seedNum,
        adapters,
        metadata: { ...meta },
        priority: 'normal',
      };
      if (operation === 'rewrite') {
        editBody.rewrite_mode = rewriteDriveMode.value;
      }
      const submitRes = await api.gen.createImageEdit(editBody);
      attachStreamFromSubmit(submitRes);
      return;
    }

    let control_asset_id = null;
    if (params.controlnet && controlImageSrc.value) {
      const cp = controlImagePath.value;
      if (typeof cp === 'string' && cp.startsWith('asset:')) {
        control_asset_id = cp.slice('asset:'.length);
      } else {
        const cblob = await api.gen.urlToBlob(controlImageSrc.value);
        control_asset_id = (
          await api.gen.uploadAsset(
            new File([cblob], 'control.png', { type: cblob.type || 'image/png' })
          )
        ).id as string;
      }
    }

    let submitRes: unknown;
    if (params.controlnet && control_asset_id) {
      submitRes = await api.gen.createImageGeneration({
        model: modelStr,
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        size: `${params.width}x${params.height}`,
        n: 1,
        steps: params.steps,
        guidance: params.guidance,
        seed: seedNum,
        adapters,
        structural_guide: {
          asset_id: control_asset_id,
          type: 'canny',
          weight: params.controlnet_strength ?? 1,
        },
        metadata: { ...meta, controlnet: params.controlnet },
        priority: 'normal',
      });
    } else {
      submitRes = await api.gen.createImageGeneration({
        model: modelStr,
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        size: `${params.width}x${params.height}`,
        n: 1,
        steps: params.steps,
        guidance: params.guidance,
        seed: seedNum,
        adapters,
        metadata: { ...meta },
        priority: 'normal',
      });
    }
    attachStreamFromSubmit(submitRes);
    tasksStore.pollQueueOnce();
  } catch (e) {
    generating.value = false;
    currentTask.value = null;
    addLog($tt('studio.error', { msg: (e as Error).message || String(e) }), 'error');
  }
};

// Load recent images
const loadRecentImages = async () => {
  try {
    const images = await api.gallery.listImages(24, 0);
    recentGalleryThumbFailed.value = {};
    recentImages.value = (images as Array<Record<string, unknown>>)
      .filter((v: Record<string, unknown>) => !isDefinitelyNonRasterRecent(v))
      .slice(0, 4);
  } catch (e) {
    console.error('Failed to load recent images:', e);
  }
};

// Get image URL
const getImageUrl = (image: Record<string, unknown>) => {
  return api.gallery.getImageUrl(String(image.path || ''));
};

// Image preview
const previewVisible = ref(false);
const selectedImage = ref<Record<string, unknown> | null>(null);

const showPreview = (image: Record<string, unknown>) => {
  selectedImage.value = image;
  previewVisible.value = true;
};

const quickFromGallery = async (image: Record<string, unknown>, mode: string) => {
  editImagePath.value = String(image.path || '');
  editImageSrc.value = getImageUrl(image);
  if (mode === 'upscale') {
    setImageWorkMode('upscale');
  } else {
    setImageWorkMode('rewrite_reference');
  }
  await loadRecentImages();
};

// Local redraw: edit file change
const onEditAssetPick = ({ path, previewUrl }: { path: string; previewUrl: string }) => {
  editImagePath.value = path;
  editImageSrc.value = previewUrl;
  addLog($tt('create.imageLoaded', { name: path }), 'info');
  void loadRecentImages();
};

const onControlAssetPick = ({ path, previewUrl }: { path: string; previewUrl: string }) => {
  controlImageSrc.value = previewUrl;
  controlImagePath.value = path;
};
const removeControlImage = () => {
  controlImageSrc.value = '';
  controlImagePath.value = '';
};

// Navigate to settings / models (Vue Router)
const goToSettings = () => {
  router.push({ name: 'settings' });
};
const goToDownload = () => {
  router.push({ name: 'models' });
};

const onModelVersionChange = (value: string) => {
  const parsed = parseModelVersionValue(value);
  if (!parsed) return;
  params.model = parsed.modelKey;
  params.version = parsed.versionKey;
  addLog($tt('studio.switchModel', { name: currentModelDisplayName.value, version: params.version as string }), 'info');
  loadModelDefaults();
};

const imageAutoSaveDraft = ref(false);
let _imgPromptSaveT: ReturnType<typeof setTimeout> | null = null;
watch(
  () => params.prompt,
  (v) => {
    if (!imageAutoSaveDraft.value) return;
    if (!DQ_STORAGE.IMAGE_CREATE_PROMPT_DRAFT) return;
    if (_imgPromptSaveT) clearTimeout(_imgPromptSaveT);
    _imgPromptSaveT = setTimeout(() => {
      try {
        localStorage.setItem(DQ_STORAGE.IMAGE_CREATE_PROMPT_DRAFT, String(v || ''));
      } catch (_) {}
    }, 500);
  },
);

const applyAppSettingsDefaults = async () => {
  try {
    const st = await api.settings.getSettings();
    imageAutoSaveDraft.value = !!st.auto_save_prompts;
    if (st.auto_save_prompts && DQ_STORAGE.IMAGE_CREATE_PROMPT_DRAFT) {
      const draft = localStorage.getItem(DQ_STORAGE.IMAGE_CREATE_PROMPT_DRAFT);
      if (draft) params.prompt = draft;
    }
    const dm = String(st.default_model_image || st.default_model || '').trim();
    const mk = resolveDefaultModelRegistryKey(dm, modelRegistry.value, 'image');
    if (!mk || !modelRegistry.value[mk]) return;
    const detailed = modelsDetailedStatus.value[mk] || {};
    const vers = detailed.versions || {};
    const defaultVK = pickDefaultVersionKey(mk, modelRegistry.value, vers);
    if (!defaultVK) return;
    params.model = mk;
    params.version = defaultVK;
    selectedModelVersion.value = mk + '|' + defaultVK;
    loadModelDefaults();
  } catch (_) {}
};

onMounted(async () => {
  await loadModelRegistry();
  await applyAppSettingsDefaults();
  loadPresets();
  loadRecentImages();
  const fromGal = localStorage.getItem(DQ_STORAGE.IMG2IMG_REF);
  if (fromGal) {
    setImageWorkMode('rewrite_reference');
    editImagePath.value = fromGal;
    editImageSrc.value = api.gallery.getImageUrl(fromGal);
    localStorage.removeItem(DQ_STORAGE.IMG2IMG_REF);
    toast.success($tt('create.img2imgFromGallery'));
  }
});

// Watch edit mode switch: auto-select a supported model for image editing
watch(editMode, (newMode) => {
  if (newMode === 'image_editing') {
    params.strength = 0.99;
    const config = currentModelConfig.value;
    const acts = (config && config.actions) ? config.actions as Record<string, unknown> : {};
    const hasCap = imageEditingMatches(acts, editingSubMode.value);
    if (!hasCap) {
      const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
      if (firstMatch) {
        params.model = firstMatch.modelKey;
        params.version = firstMatch.versionKey;
        selectedModelVersion.value = String(firstMatch.modelKey) + '|' + String(firstMatch.versionKey);
        loadModelDefaults();
      }
    }
  } else if (newMode === 'image_upscale') {
    const config = currentModelConfig.value;
    const acts = (config && config.actions) ? config.actions as Record<string, unknown> : {};
    const hasCap = imageSupportsUpscale(acts);
    if (!hasCap) {
      const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
      if (firstMatch) {
        params.model = firstMatch.modelKey;
        params.version = firstMatch.versionKey;
        selectedModelVersion.value = String(firstMatch.modelKey) + '|' + String(firstMatch.versionKey);
        loadModelDefaults();
      }
    }
  }
});

// Watch sub-type switch: re-filter models
watch(editingSubMode, () => {
  if (editMode.value !== 'image_editing') return;
  const config = currentModelConfig.value;
  const acts = (config && config.actions) ? config.actions as Record<string, unknown> : {};
  const hasCap = imageEditingMatches(acts, editingSubMode.value);
  if (!hasCap) {
    const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
    if (firstMatch) {
      params.model = firstMatch.modelKey;
      params.version = firstMatch.versionKey;
      selectedModelVersion.value = String(firstMatch.modelKey) + '|' + String(firstMatch.versionKey);
      loadModelDefaults();
    }
  }
});

watch(imageWorkTab, (t) => {
  if (t !== 'rewrite_reference' && t !== 'rewrite_instruct') return;
  const okInList = filteredAllVersions.value.some(
    (v) => v.modelKey === params.model && v.versionKey === params.version,
  );
  if (!okInList) {
    const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
    if (firstMatch) {
      params.model = firstMatch.modelKey;
      params.version = firstMatch.versionKey;
      selectedModelVersion.value = String(firstMatch.modelKey) + '|' + String(firstMatch.versionKey);
      loadModelDefaults();
    }
  }
});

watch(modelFilterCommercialOnly, () => {
  if (
    reconcileVersionPickerSelection(filteredModelPickerVersions.value, params, selectedModelVersion)
  ) {
    loadModelDefaults();
  }
});
</script>
