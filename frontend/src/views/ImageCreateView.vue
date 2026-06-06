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
        :active-tasks="activeImageTasks"
        :loading="galleryLoading"
        :has-more="galleryHasMore"
        media="image"
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
      <ImageComposer
        v-model="params.prompt"
        v-model:title="params.title"
        v-model:model="selectedModelVersion"
        v-model:size="selectedSize"
        v-model:batch-count="batchCount"
        :generating="generating"
        :can-generate="canGenerate"
        :model-options="modelSelectOptions"
        :size-options="sizeOptions"
        :styles="filteredPresets"
        :params="params"
        :has-custom-params="hasCustomParams"
        :show-negative-prompt="!!currentModelConfig?.parameters?.negative_prompt_support"
        :reference-image="referenceImage"
        :mode="imageMode"
        :mode-options="imageModeOptions"
        :current-model-config="currentModelConfig"
        :compatible-loras="compatibleLoras"
        :compatible-control-nets="compatibleControlNets"
        @update:mode="onModeChange"
        @generate="startGeneration"
        @pick-reference="showAssetPicker = true"
        @remove-reference="removeReferenceImage"
        @model-change="onModelVersionChange"
        @reset-defaults="resetToDefaults"
      />
    </template>
  </StudioLayout>

  <!-- Asset picker for reference image -->
  <DqDialog v-model:open="showAssetPicker" :title="$t('assetPicker.dialogTitle')" width="70%">
    <AssetPicker
      accept-kind="image"
      :recent-gallery="recentImages"
      @pick="onReferencePick"
    />
  </DqDialog>

  <!-- Preview dialog -->
  <GalleryPreviewDialog
    v-model:visible="previewVisible"
    v-model:index="selectedImageIndex"
    :items="galleryItems"
    media="image"
  />

  <!-- Unified Editor Drawer: retouch / extend / upscale -->
  <DqDrawer
    v-model:open="showEditorDrawer"
    :title="editorDrawerTitle"
    direction="rtl"
    size="520px"
    class="studio-image-editor-drawer"
  >
    <div v-if="editDrawerItem" class="studio-editor-drawer">
      <div v-if="editorMode === 'retouch'" class="studio-retouch-panel">
        <DqPrefPane class="studio-create-pref-pane">
          <DqPrefRow :label="$t('studio.model')">
            <DqSelect v-model="retouchModelVersion" size="small" style="width: 100%" :placeholder="$t('studio.selectModel')">
              <DqOption
                v-for="item in retouchModelOptions"
                :key="item.value"
                :label="item.label"
                :value="item.value"
                :disabled="item.disabled"
              >
                <DqTag
                  v-if="item.commercialUseAllowed"
                  size="mini"
                  type="success"
                  class="studio-drawer-model-badge"
                >
                  {{ $t('download.commercialUseBadge') }}
                </DqTag>
              </DqOption>
            </DqSelect>
          </DqPrefRow>
        </DqPrefPane>

        <div class="studio-retouch-editor-wrap">
          <ImageEditor
            ref="imageEditorRef"
            :src="getImageUrl(editDrawerItem)"
            mode="inpainting"
            :show-submit-button="false"
          />
        </div>
        <DqButton type="primary" block class="studio-drawer-submit" @click="onEditorSubmit">
          {{ $t('action.image.retouch') }}
        </DqButton>
      </div>
      <div v-else-if="editorMode === 'extend'" class="studio-extend-panel">
        <DqPrefPane class="studio-create-pref-pane">
          <DqPrefRow :label="$t('studio.model')">
            <DqSelect v-model="extendModelVersion" size="small" style="width: 100%" :placeholder="$t('studio.selectModel')">
              <DqOption
                v-for="item in extendModelOptions"
                :key="item.value"
                :label="item.label"
                :value="item.value"
                :disabled="item.disabled"
              >
                <DqTag
                  v-if="item.commercialUseAllowed"
                  size="mini"
                  type="success"
                  class="studio-drawer-model-badge"
                >
                  {{ $t('download.commercialUseBadge') }}
                </DqTag>
              </DqOption>
            </DqSelect>
          </DqPrefRow>
        </DqPrefPane>
        <CreateExtendParams :params="extendParams" />
        <DqButton type="primary" block class="studio-drawer-submit" @click="onExtendSubmit">
          {{ $t('action.image.extend') }}
        </DqButton>
      </div>
      <div v-else-if="editorMode === 'upscale'" class="studio-upscale-panel">
        <DqPrefPane class="studio-create-pref-pane">
          <DqPrefRow :label="$t('studio.model')">
            <DqSelect v-model="upscaleModelVersion" size="small" style="width: 100%" :placeholder="$t('studio.selectModel')">
              <DqOption
                v-for="item in upscaleModelOptions"
                :key="item.value"
                :label="item.label"
                :value="item.value"
                :disabled="item.disabled"
              >
                <DqTag
                  v-if="item.commercialUseAllowed"
                  size="mini"
                  type="success"
                  class="studio-drawer-model-badge"
                >
                  {{ $t('download.commercialUseBadge') }}
                </DqTag>
              </DqOption>
            </DqSelect>
          </DqPrefRow>
        </DqPrefPane>
        <CreateUpscaleParams :params="upscaleParams" media="image" />
        <DqButton type="primary" block class="studio-drawer-submit" @click="onUpscaleSubmit">
          {{ $t('action.image.upscale') }}
        </DqButton>
      </div>
    </div>
  </DqDrawer>
</template>

<script setup lang="ts">
// @ts-nocheck — legacy create view; narrow types in a follow-up pass
import { ref, reactive, computed, watch, onMounted, onUnmounted, inject, nextTick, unref } from 'vue';
import type { Ref } from 'vue';
import { useRouter } from 'vue-router';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt, $mn, $md, $mvn, $pn } from '@/utils/i18n';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import type { SystemInfo, GalleryItem, Task } from '@/types';
import { applyDefaults, hasDeviation, strengthDefaultFromRegistry, strengthToSourceFidelity } from '@/utils/registryParamSchema';

import { warnIfRiskyMemory } from '@/composables/memoryHint';
import { reconcileVersionPickerSelection } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import { previewDisplayCaption, truncateDisplayLabel } from '@/utils/assetDisplay';
// Studio components
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvas from '@/components/studio/StudioCanvas.vue';
import StudioGalleryFilters from '@/components/studio/StudioGalleryFilters.vue';
import ImageComposer from '@/components/studio/ImageComposer.vue';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import ImageEditor from '@/components/image/ImageEditor.vue';
import CreateExtendParams from '@/components/create/CreateExtendParams.vue';
import CreateUpscaleParams from '@/components/create/CreateUpscaleParams.vue';
import GalleryPreviewDialog from '@/components/gallery/GalleryPreviewDialog.vue';
import { useStudioGallery } from '@/composables/useStudioGallery';

/* ------------------------------------------------------------------ */
/*  Injected / External                                                */
/* ------------------------------------------------------------------ */

const systemInfo = inject<Ref<SystemInfo>>('systemInfo');
const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const router = useRouter();

/* ------------------------------------------------------------------ */
/*  RegistryActions helpers                                            */
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
function imageSupportsExtend(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'extend');
}
function imageSupportsRetouch(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'retouch');
}
function imageModelRow(config: Record<string, unknown>): boolean {
  return config && config.media === 'image' && config.category !== 'loras';
}

/* ------------------------------------------------------------------ */
/*  Params                                                             */
/* ------------------------------------------------------------------ */

const params = reactive<Record<string, unknown>>({
  title: '',
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

const selectedModelVersion = ref('');
const selectedSize = ref(getItem(DQ_STORAGE.IMAGE_LAST_SIZE) || '1024x1024');
const batchCount = ref(1);
const generating = ref(false);

/* ------------------------------------------------------------------ */
/*  Reference Image (new: replaces old edit mode tabs)                 */
/* ------------------------------------------------------------------ */

const referenceImage = ref<{ previewUrl: string; path: string; assetId?: string } | null>(null);
const showAssetPicker = ref(false);

function onReferencePick({ path, previewUrl }: { path: string; previewUrl: string }) {
  referenceImage.value = { path, previewUrl };
  showAssetPicker.value = false;
  imageMode.value = 'img2img';
}

function removeReferenceImage() {
  referenceImage.value = null;
}

/* ------------------------------------------------------------------ */
/*  LoRAs / ControlNets                                                */
/* ------------------------------------------------------------------ */

const compatibleLoras = ref<Record<string, unknown>[]>([]);
const compatibleControlNets = ref<Record<string, unknown>[]>([]);

async function loadCompatibleAdapters(modelKey: string) {
  if (!modelKey) {
    compatibleLoras.value = [];
    compatibleControlNets.value = [];
    return;
  }
  try {
    const [loras, controlNets] = await Promise.all([
      api.settings.getCompatibleLoras(modelKey),
      api.settings.getCompatibleControlNets(modelKey),
    ]);
    compatibleLoras.value = (loras as Record<string, unknown>[]) || [];
    compatibleControlNets.value = (controlNets as Record<string, unknown>[]) || [];
  } catch (e) {
    console.error('Failed to load compatible adapters:', e);
    compatibleLoras.value = [];
    compatibleControlNets.value = [];
    toast.error($tt('studio.error', { msg: $tt('studio.adapterLoadFailed') }));
  }
}

/* ------------------------------------------------------------------ */
/*  Gallery / Studio Canvas                                            */
/* ------------------------------------------------------------------ */

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
} = useStudioGallery('image');

watch(() => params.model, (modelKey) => {
  if (modelKey) loadCompatibleAdapters(String(modelKey));
});

/* ------------------------------------------------------------------ */
/*  Active Tasks (generating placeholders)                             */
/* ------------------------------------------------------------------ */

const activeImageTasks = computed(() => {
  const running = tasksStore.queueState.running.filter((t: Task) =>
    String(t.kind || '').startsWith('image.')
  );
  const queued = tasksStore.queueState.queued.filter((t: Task) =>
    String(t.kind || '').startsWith('image.')
  );
  return [...running, ...queued].map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
});

/* ------------------------------------------------------------------ */
/*  Task status helpers                                                */
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
/*  Model registry                                                     */
/* ------------------------------------------------------------------ */

const modelRegistry = ref<Record<string, Record<string, unknown>>>({});
const modelsDetailedStatus = ref<Record<string, { versions?: Record<string, { ready?: boolean; status?: string }> }>>({});

const allVersions = computed(() => {
  const result: Array<Record<string, unknown>> = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!imageModelRow(config)) continue;
    const actions = { ...(config.actions as Record<string, unknown> || {}) };
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
        description: $md(config as { description?: string | { zh?: string; en?: string }; description_en?: string }),
        size,
        status: status.status,
        ready: status.ready,
        recommended: config.recommended && (versionConfig as Record<string, unknown>).default,
        commercialUseAllowed: config.commercial_use_allowed === true,
        actions,
      });
    }
  }
  return result;
});

const filteredAllVersions = computed(() => {
  return allVersions.value.filter((v) => {
    const acts = v.actions as Record<string, unknown> || {};
    if (imageMode.value === 'img2img') {
      return hasAction(acts, 'rewrite') || hasAction(acts, 'retouch');
    }
    return imageSupportsCreate(acts);
  });
});

const filteredModelPickerVersions = computed(() => {
  const rows = applyModelVersionFilters(filteredAllVersions.value, {
    installedOnly: true,
    commercialOnly: false, // show all; badge indicates commercial status
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

const modelSelectOptions = computed(() => {
  return filteredModelPickerVersions.value.map((v) => ({
    label: String(v.name || ''),
    value: `${v.modelKey}|${v.versionKey}`,
    disabled: !v.ready,
    commercialUseAllowed: v.commercialUseAllowed as boolean,
  }));
});

const currentModelConfig = computed(() => modelRegistry.value[params.model as string] || null);
const currentModelDisplayName = computed(() => {
  const c = currentModelConfig.value;
  if (c) {
    return $mn(c as { name?: string | { zh?: string; en?: string }; name_en?: string }, params.model as string);
  }
  return params.model || '';
});

const selectedModelNotReady = computed(() => {
  if (!params.model || !params.version) return false;
  const detailed = modelsDetailedStatus.value[params.model as string];
  if (!detailed || !detailed.versions) return true;
  const versionStatus = detailed.versions[params.version as string];
  return !versionStatus || !versionStatus.ready;
});

const canGenerate = computed(() => {
  if (selectedModelNotReady.value) return false;
  if (!String(params.prompt || '').trim()) return false;
  if (imageMode.value === 'img2img' && !referenceImage.value) return false;
  return true;
});

function parseModelVersionValue(value: string): { modelKey: string; versionKey: string } | null {
  if (!value || typeof value !== 'string') return null;
  const parts = value.split('|');
  if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
  return { modelKey: parts[0], versionKey: parts[1] };
}

function getImageModeStorageKey(mode: string): StorageKey {
  const map: Record<string, StorageKey> = {
    text2img: DQ_STORAGE.IMAGE_MODEL_TEXT2IMG,
    img2img: DQ_STORAGE.IMAGE_MODEL_IMG2IMG,
    retouch: DQ_STORAGE.IMAGE_MODEL_RETOUCH,
    extend: DQ_STORAGE.IMAGE_MODEL_EXTEND,
    upscale: DQ_STORAGE.IMAGE_MODEL_UPSCALE,
  };
  return map[mode] || DQ_STORAGE.IMAGE_MODEL_TEXT2IMG;
}

function isModelAvailable(modelKey: string, versionKey: string): boolean {
  return filteredModelPickerVersions.value.some(
    (v) => v.modelKey === modelKey && v.versionKey === versionKey && v.ready
  );
}

function restoreModelForMode(mode: string) {
  const saved = getItem(getImageModeStorageKey(mode));
  if (saved) {
    const parsed = parseModelVersionValue(saved);
    if (parsed && isModelAvailable(parsed.modelKey, parsed.versionKey)) {
      selectedModelVersion.value = saved;
      params.model = parsed.modelKey;
      params.version = parsed.versionKey;
      loadModelDefaults();
      loadCompatibleAdapters(parsed.modelKey);
      return;
    }
  }
  // fallback: 尝试当前模型是否支持该 mode
  const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
  const currentParsed = parseModelVersionValue(currentKey);
  if (currentParsed && isModelAvailable(currentParsed.modelKey, currentParsed.versionKey)) {
    return;
  }
  // fallback: 选第一个可用模型
  if (filteredModelPickerVersions.value.length > 0) {
    const first = filteredModelPickerVersions.value[0];
    selectedModelVersion.value = `${first.modelKey}|${first.versionKey}`;
    params.model = first.modelKey;
    params.version = first.versionKey;
    loadModelDefaults();
    loadCompatibleAdapters(first.modelKey);
  }
}

const showEditorDrawer = ref(false);
const editorMode = ref<'retouch' | 'extend' | 'upscale'>('retouch');
const editDrawerItem = ref<GalleryItem | null>(null);
const editorDrawerTitle = computed(() => {
  const map: Record<string, string> = {
    retouch: $tt('action.image.retouch'),
    extend: $tt('action.image.extend'),
    upscale: $tt('action.image.upscale'),
  };
  return map[editorMode.value] || '';
});

function onModelVersionChange(value: string) {
  const parsed = parseModelVersionValue(value);
  if (!parsed) return;
  params.model = parsed.modelKey;
  params.version = parsed.versionKey;
  const activeMode = showEditorDrawer.value ? editorMode.value : imageMode.value;
  setItem(getImageModeStorageKey(activeMode), value);
  loadModelDefaults();
  loadCompatibleAdapters(parsed.modelKey);
}

/* ------------------------------------------------------------------ */
/*  Size options                                                       */
/* ------------------------------------------------------------------ */

const sizeOptions = computed(() => [
  { label: '1:1', value: '512x512', pixelLabel: '512×512' },
  { label: '1:1 HD', value: '1024x1024', pixelLabel: '1024×1024' },
  { label: '2:3', value: '1024x1536', pixelLabel: '1024×1536' },
  { label: '3:2', value: '1536x1024', pixelLabel: '1536×1024' },
  { label: '16:9 (1080p)', value: '1920x1080', pixelLabel: '1920×1080' },
  { label: '9:16 (1080p)', value: '1080x1920', pixelLabel: '1080×1920' },
  { label: '16:9 (720p)', value: '1344x768', pixelLabel: '1344×768' },
  { label: '9:16 (720p)', value: '768x1344', pixelLabel: '768×1344' },
]);

const imageMode = ref('text2img');

const imageModeOptions = computed(() => [
  { label: $tt('action.image.text2img'), value: 'text2img' },
  { label: $tt('action.image.img2img'), value: 'img2img' },
]);

function onModeChange(mode: string) {
  imageMode.value = mode;
  if (mode === 'text2img') {
    removeReferenceImage();
  }
}

watch(imageMode, (newMode, oldMode) => {
  // 保存旧模式模型
  if (oldMode && selectedModelVersion.value) {
    setItem(getImageModeStorageKey(oldMode), selectedModelVersion.value);
  }
  // 恢复新模式模型
  const saved = getItem(getImageModeStorageKey(newMode));
  if (saved) {
    const parsed = parseModelVersionValue(saved);
    if (parsed && isModelAvailable(parsed.modelKey, parsed.versionKey)) {
      selectedModelVersion.value = saved;
      params.model = parsed.modelKey;
      params.version = parsed.versionKey;
      loadModelDefaults();
      loadCompatibleAdapters(parsed.modelKey);
      return;
    }
  }
  // fallback
  if (
    reconcileVersionPickerSelection(filteredModelPickerVersions.value, params, selectedModelVersion)
  ) {
    loadModelDefaults();
    loadCompatibleAdapters(String(params.model || ''));
  }
});

watch(selectedSize, (val) => {
  const [w, h] = val.split('x').map(Number);
  if (w && h) {
    params.width = w;
    params.height = h;
  }
  setItem(DQ_STORAGE.IMAGE_LAST_SIZE, val);
});

// Drawer 关闭时保存 editor 模型并恢复 composer 模型
watch(showEditorDrawer, (isOpen, wasOpen) => {
  if (wasOpen && !isOpen) {
    // drawer 关闭
    if (selectedModelVersion.value) {
      setItem(getImageModeStorageKey(editorMode.value), selectedModelVersion.value);
    }
    // 恢复 composer 模型
    restoreModelForMode(imageMode.value);
  }
});

const hasCustomParams = computed(() => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return false;
  return hasDeviation(config.parameters as Record<string, unknown>, params);
});

/* ------------------------------------------------------------------ */
/*  Presets / Styles                                                   */
/* ------------------------------------------------------------------ */

const presets = ref<Record<string, Record<string, unknown>>>({});

const presetActionFilter = computed(() => new Set(['create']));

const filteredPresets = computed(() => {
  const want = presetActionFilter.value;

  function planPresetShapeOk(preset: Record<string, unknown>) {
    return (
      Array.isArray(preset.applies_to) &&
      (preset.applies_to as unknown[]).length > 0 &&
      preset.media_scope === 'image'
    );
  }

  function matches(preset: Record<string, unknown>) {
    if (!planPresetShapeOk(preset)) return false;
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

const loadPresets = async () => {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as Record<string, Record<string, unknown>>) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
    presets.value = {};
  }
};

/* ------------------------------------------------------------------ */
/*  Generation logic (simplified: text-to-image + image-to-image)      */
/* ------------------------------------------------------------------ */

let activeGenStream: EventSource | null = null;
const currentTask = ref<Record<string, unknown> | null>(null);

function closeGenStream() {
  if (activeGenStream) {
    activeGenStream.close();
    activeGenStream = null;
  }
}

function attachStreamFromSubmit(submitRes: unknown) {
  const tid = taskIdFromSubmitResponse(submitRes);
  if (!tid) {
    generating.value = false;
    return;
  }
  tasksStore.clearTaskLogs(tid);
  tasksStore.appendTaskLog(tid, $tt('studio.startingGen'), 'info');
  tasksStore.registerPageOwnedStream(tid);
  currentTask.value = {
    id: tid,
    progress: 0,
    step: 0,
    total: 0,
    status: 'queued',
    params: { model: params.model, title: String(params.title || '').trim(), prompt: params.prompt },
  };

  activeGenStream = api.gen.streamMediaTask(tid, {
    onLog: (logData: any) => {
      tasksStore.ingestTaskLog(tid, logData);
    },
    onStatus: (statusData: any) => {
      if (currentTask.value) {
        currentTask.value = { ...currentTask.value, ...statusData };
      }
    },
    onProgress: (progressData: any) => {
      tasksStore.ingestTaskProgressLog(tid, progressData);
      tasksStore.patchLiveTaskProgress(tid, {
        progress: progressData.progress,
        step: progressData.step,
        total: progressData.total,
        eta_seconds: progressData.eta_seconds,
        progressMessage: progressData.message ?? progressData.phase,
      });
      if (currentTask.value) {
        currentTask.value = {
          ...currentTask.value,
          progress: progressData.progress ?? currentTask.value.progress,
          step: progressData.step ?? currentTask.value.step,
          total: progressData.total ?? currentTask.value.total,
        };
      }
    },
    onDone: async (doneData: any) => {
      generating.value = false;
      tasksStore.unregisterPageOwnedStream(tid);
      if (doneData.status === 'completed') {
        tasksStore.appendTaskLog(tid, $tt('studio.genComplete'), 'success');
        toast.success($tt('studio.genComplete'));
        setTimeout(() => loadGallery(true), 1000);
      } else if (doneData.status === 'failed') {
        const updated = await api.gen.getMediaTask(tid) as any;
        tasksStore.appendTaskLog(
          tid,
          $tt('studio.genFailed', { msg: updated.error || updated.error_message || '' }),
          'error'
        );
        toast.error($tt('studio.genFailed', { msg: updated.error || updated.error_message || '' }));
      }
      currentTask.value = null;
      closeGenStream();
    },
    onError: () => {
      generating.value = false;
      tasksStore.unregisterPageOwnedStream(tid);
      tasksStore.appendTaskLog(tid, $tt('studio.connectionLost'), 'warning');
      currentTask.value = null;
      closeGenStream();
    },
  });
}

const startGeneration = async () => {
  if (generating.value) return;
  if (!String(params.prompt || '').trim()) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }

  const detailed = modelsDetailedStatus.value[params.model as string];
  const versionStatus = detailed?.versions?.[params.version as string];
  if (!versionStatus?.ready) {
    toast.warning($tt('studio.modelNotReadyDesc', { name: currentModelDisplayName.value, version: params.version as string }));
    return;
  }

  const verCfg =
    (currentModelConfig.value &&
      currentModelConfig.value.versions &&
      (currentModelConfig.value.versions as Record<string, Record<string, unknown>>)[params.version as string]) ||
    null;
  const sizeHuman = verCfg && verCfg.size ? String(verCfg.size) : '';
  const minMemRaw = currentModelConfig.value?.parameters?.min_unified_memory_gb;
  const minUnifiedMemoryGb = minMemRaw != null && Number(minMemRaw) > 0 ? Number(minMemRaw) : null;
  warnIfRiskyMemory({
    systemInfo: unref(systemInfo),
    versionSizeHuman: sizeHuman,
    minUnifiedMemoryGb,
    $tt,
  });

  generating.value = true;

  try {
    const modelStr = params.version ? `${params.model}:${params.version}` : params.model;
    const seedNum = params.seed ? parseInt(String(params.seed), 10) : null;
    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });
    const meta: Record<string, unknown> = {};
    if (params.scheduler) meta.scheduler = params.scheduler;

    let control_asset_id: string | null = null;
    if (params.controlnet) {
      // controlnet image is not yet managed in this simplified composer; skip if not set
      control_asset_id = null;
    }

    let submitRes: unknown;
    const hasRef = referenceImage.value != null;

    if (hasRef) {
      // Image-to-image via edits endpoint
      let source_asset_id: string;
      const rp = referenceImage.value!.path;
      if (typeof rp === 'string' && rp.startsWith('asset:')) {
        source_asset_id = rp.slice('asset:'.length);
      } else {
        const blob = await api.gen.urlToBlob(referenceImage.value!.previewUrl);
        const up = await api.gen.uploadAsset(
          new File([blob], 'ref.png', { type: blob.type || 'image/png' })
        );
        source_asset_id = (up as any).id;
      }

      const editBody: Record<string, unknown> = {
        model: modelStr,
        operation: 'rewrite',
        source_asset_id,
        title: String(params.title || '').trim(),
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        n: 1,
        steps: params.steps,
        guidance: params.guidance,
        seed: seedNum,
        adapters,
        source_fidelity: strengthToSourceFidelity(
          params.strength,
          strengthDefaultFromRegistry(currentModelConfig.value?.parameters as Record<string, unknown> | undefined),
        ),
        metadata: { ...meta },
        priority: 'normal',
      };
      submitRes = await api.gen.createImageEdit(editBody);
    } else {
      // Text-to-image
      const genBody: Record<string, unknown> = {
        model: modelStr,
        title: String(params.title || '').trim(),
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        size: `${params.width}x${params.height}`,
        n: batchCount.value,
        steps: params.steps,
        guidance: params.guidance,
        seed: seedNum,
        adapters,
        metadata: { ...meta },
        priority: 'normal',
      };
      if (control_asset_id) {
        genBody.structural_guide = { asset_id: control_asset_id, strength: Number(params.controlnet_strength) || 0.8 };
      }
      submitRes = await api.gen.createImageGeneration(genBody);
    }

    attachStreamFromSubmit(submitRes);
    tasksStore.pollQueueOnce();
  } catch (e) {
    generating.value = false;
    closeGenStream();
    currentTask.value = null;
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
};

/* ------------------------------------------------------------------ */
/*  Load model registry                                                */
/* ------------------------------------------------------------------ */

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
    modelsDetailedStatus.value = (detailedStatusData as any) || {};

    if (!selectedModelVersion.value) {
      // 尝试从本地存储恢复上次选择的模型
      const lastModel = getItem(DQ_STORAGE.IMAGE_LAST_MODEL);
      if (lastModel) {
        const parsed = parseModelVersionValue(lastModel);
        if (parsed) {
          const detailed = (detailedStatusData as Record<string, Record<string, unknown>>)[parsed.modelKey] || {};
          const versions = detailed.versions || {};
          if (versions[parsed.versionKey]?.ready) {
            params.model = parsed.modelKey;
            params.version = parsed.versionKey;
            selectedModelVersion.value = lastModel;
            loadModelDefaults();
            restoreSavedSize();
            loadCompatibleAdapters(parsed.modelKey);
            return;
          }
        }
      }

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
    restoreSavedSize();
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
};

const loadModelDefaults = () => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return;
  const prevWidth = params.width;
  const prevHeight = params.height;
  applyDefaults(config.parameters as Record<string, unknown>, params);
  params.width = prevWidth;
  params.height = prevHeight;
};

function restoreSavedSize() {
  const savedSize = getItem(DQ_STORAGE.IMAGE_LAST_SIZE);
  if (savedSize && sizeOptions.value.some((o) => o.value === savedSize)) {
    selectedSize.value = savedSize;
    const [w, h] = savedSize.split('x').map(Number);
    if (w && h) {
      params.width = w;
      params.height = h;
    }
  }
}

const resetToDefaults = () => {
  loadModelDefaults();
  toast.success($tt('studio.restoredDefaults'));
};

/* ------------------------------------------------------------------ */
/*  Gallery interactions                                               */
/* ------------------------------------------------------------------ */

const previewVisible = ref(false);
const selectedImageIndex = ref(0);

function getImageUrl(item: GalleryItem) {
  return api.gallery.getImageUrl(item.path);
}

function onGallerySelect(item: GalleryItem) {
  const idx = galleryItems.value.findIndex((it) => it.path === item.path);
  selectedImageIndex.value = idx >= 0 ? idx : 0;
  previewVisible.value = true;
}

/* ------------------------------------------------------------------ */
/*  Card actions: retouch / extend / upscale / download / delete       */
/* ------------------------------------------------------------------ */

const upscaleParams = reactive({
  upscale_scale: 2,
  upscale_denoise: 0.3,
  upscale_tile: 1024,
});
const upscaleModelVersion = ref('');
const upscaleModelOptions = computed(() => {
  return allVersions.value
    .filter((v) => {
      const acts = (v.actions as Record<string, unknown>) || {};
      return imageSupportsUpscale(acts);
    })
    .map((v) => ({
      label: String(v.name || ''),
      value: `${v.modelKey}|${v.versionKey}`,
      disabled: !v.ready,
      commercialUseAllowed: v.commercialUseAllowed as boolean,
    }));
});
const extendModelVersion = ref('');
const extendModelOptions = computed(() => {
  return allVersions.value
    .filter((v) => {
      const acts = (v.actions as Record<string, unknown>) || {};
      return imageSupportsExtend(acts);
    })
    .map((v) => ({
      label: String(v.name || '') + (!v.ready ? ` (${$tt('common.notReady')})` : ''),
      value: `${v.modelKey}|${v.versionKey}`,
      disabled: !v.ready,
      commercialUseAllowed: v.commercialUseAllowed as boolean,
    }));
});
const extendParams = reactive({
  extend_directions: ['right'],
  extend_pixels: 256,
});
const retouchModelVersion = ref('');
const retouchModelOptions = computed(() => {
  return allVersions.value
    .filter((v) => {
      const acts = (v.actions as Record<string, unknown>) || {};
      return imageSupportsRetouch(acts);
    })
    .map((v) => ({
      label: String(v.name || '') + (!v.ready ? ` (${$tt('common.notReady')})` : ''),
      value: `${v.modelKey}|${v.versionKey}`,
      disabled: !v.ready,
      commercialUseAllowed: v.commercialUseAllowed as boolean,
    }));
});
const imageEditorRef = ref<any>(null);

function onCardAction({ action, item }: { action: string; item: GalleryItem }) {
  switch (action) {
    case 'retouch':
    case 'extend':
    case 'upscale':
      // 保存当前 composer 模型到对应 imageMode key
      if (selectedModelVersion.value) {
        setItem(getImageModeStorageKey(imageMode.value), selectedModelVersion.value);
      }
      editDrawerItem.value = item;
      editorMode.value = action as 'retouch' | 'extend' | 'upscale';
      // 恢复该 action 上次使用的模型
      restoreModelForMode(action);
      // 同步 drawer 内模型选择器
      if (action === 'upscale') {
        const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
        const currentSupportsUpscale = allVersions.value.some((v) => {
          const acts = (v.actions as Record<string, unknown>) || {};
          return `${v.modelKey}|${v.versionKey}` === currentKey && imageSupportsUpscale(acts);
        });
        if (currentSupportsUpscale) {
          upscaleModelVersion.value = currentKey;
        } else {
          const first = upscaleModelOptions.value.find((o) => !o.disabled);
          upscaleModelVersion.value = first ? first.value : '';
        }
      } else if (action === 'extend') {
        const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
        const currentSupportsExtend = allVersions.value.some((v) => {
          const acts = (v.actions as Record<string, unknown>) || {};
          return `${v.modelKey}|${v.versionKey}` === currentKey && imageSupportsExtend(acts);
        });
        if (currentSupportsExtend) {
          extendModelVersion.value = currentKey;
        } else {
          const first = extendModelOptions.value.find((o) => !o.disabled);
          extendModelVersion.value = first ? first.value : '';
        }
      } else if (action === 'retouch') {
        const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
        const currentSupportsRetouch = allVersions.value.some((v) => {
          const acts = (v.actions as Record<string, unknown>) || {};
          return `${v.modelKey}|${v.versionKey}` === currentKey && imageSupportsRetouch(acts);
        });
        if (currentSupportsRetouch) {
          retouchModelVersion.value = currentKey;
        } else {
          const first = retouchModelOptions.value.find((o) => !o.disabled);
          retouchModelVersion.value = first ? first.value : '';
        }
      }
      showEditorDrawer.value = true;
      break;
    case 'download':
      downloadItem(item);
      break;
    case 'delete':
      deleteItem(item);
      break;
  }
}

function downloadItem(item: GalleryItem) {
  const url = getImageUrl(item);
  const a = document.createElement('a');
  a.href = url;
  a.download = item.name;
  a.click();
  toast.success($tt('gallery.startDownload'));
}

async function onEditorSubmit() {
  if (!editDrawerItem.value || !imageEditorRef.value) return;
  try {
    const maskBlob = await imageEditorRef.value.getMaskBlob();
    if (!maskBlob) {
      toast.warning($tt('studio.drawMask'));
      return;
    }

    const path = editDrawerItem.value.path;
    let source_asset_id: string;
    if (path.startsWith('asset:')) {
      source_asset_id = path.slice('asset:'.length);
    } else {
      const blob = await api.gen.urlToBlob(getImageUrl(editDrawerItem.value));
      const up = await api.gen.uploadAsset(
        new File([blob], 'source.png', { type: blob.type || 'image/png' })
      );
      source_asset_id = (up as any).id;
    }

    const mask_asset_id = (
      await api.gen.uploadAsset(new File([maskBlob], 'mask.png', { type: 'image/png' }))
    ).id as string;

    if (!retouchModelVersion.value) {
      toast.warning($tt('studio.selectModel'));
      return;
    }
    const [modelKey, versionKey] = retouchModelVersion.value.split('|');
    const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
    const seedNum = params.seed ? parseInt(String(params.seed), 10) : null;
    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });

    const submitRes = await api.gen.createImageEdit({
      model: modelStr,
      operation: 'retouch',
      source_asset_id,
      mask_asset_id,
      title: String(params.title || '').trim(),
      prompt: params.prompt,
      negative_prompt: params.negative_prompt || '',
      source_fidelity: strengthToSourceFidelity(
        params.strength,
        strengthDefaultFromRegistry(currentModelConfig.value?.parameters as Record<string, unknown> | undefined),
      ),
      n: 1,
      steps: params.steps,
      seed: seedNum,
      adapters,
      metadata: {},
      priority: 'normal',
    });
    attachStreamFromSubmit(submitRes);
    showEditorDrawer.value = false;
    tasksStore.pollQueueOnce();
  } catch (e) {
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
}

async function onExtendSubmit() {
  if (!editDrawerItem.value) return;
  try {
    const dirs = Array.isArray(extendParams.extend_directions)
      ? extendParams.extend_directions.filter((d: string) => ['top', 'bottom', 'left', 'right'].includes(d))
      : [];
    if (!dirs.length) {
      toast.warning($tt('create.extendNeedDirection'));
      return;
    }
    const px = Math.min(2048, Math.max(64, Number(extendParams.extend_pixels) || 256));

    const path = editDrawerItem.value.path;
    let source_asset_id: string;
    if (path.startsWith('asset:')) {
      source_asset_id = path.slice('asset:'.length);
    } else {
      const blob = await api.gen.urlToBlob(getImageUrl(editDrawerItem.value));
      const up = await api.gen.uploadAsset(
        new File([blob], 'source.png', { type: blob.type || 'image/png' })
      );
      source_asset_id = (up as any).id;
    }

    if (!extendModelVersion.value) {
      toast.warning($tt('studio.selectModel'));
      return;
    }
    const [modelKey, versionKey] = extendModelVersion.value.split('|');
    const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
    const seedNum = params.seed ? parseInt(String(params.seed), 10) : null;
    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });

    const submitRes = await api.gen.createImageEdit({
      model: modelStr,
      operation: 'extend',
      source_asset_id,
      title: String(params.title || '').trim(),
      prompt: params.prompt,
      negative_prompt: params.negative_prompt || '',
      extend: { directions: dirs, pixels: px },
      n: 1,
      steps: params.steps,
      seed: seedNum,
      adapters,
      metadata: {},
      priority: 'normal',
    });
    attachStreamFromSubmit(submitRes);
    showEditorDrawer.value = false;
    tasksStore.pollQueueOnce();
  } catch (e) {
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
}

async function onUpscaleSubmit() {
  if (!editDrawerItem.value) return;
  if (!upscaleModelVersion.value) {
    toast.warning($tt('studio.selectModel'));
    return;
  }
  try {
    const path = editDrawerItem.value.path;
    let source_asset_id: string;
    if (path.startsWith('asset:')) {
      source_asset_id = path.slice('asset:'.length);
    } else {
      const blob = await api.gen.urlToBlob(getImageUrl(editDrawerItem.value));
      const up = await api.gen.uploadAsset(
        new File([blob], 'upscale.png', { type: blob.type || 'image/png' })
      );
      source_asset_id = (up as any).id;
    }

    const [modelKey, versionKey] = upscaleModelVersion.value.split('|');
    const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
    const submitRes = await api.gen.createImageUpscale({
      model: modelStr,
      source_asset_id,
      scale: upscaleParams.upscale_scale,
      denoise: upscaleParams.upscale_denoise,
      metadata: {},
      priority: 'normal',
    });
    attachStreamFromSubmit(submitRes);
    showEditorDrawer.value = false;
    tasksStore.pollQueueOnce();
  } catch (e) {
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
}

/* ------------------------------------------------------------------ */
/*  Recent images (for asset picker)                                   */
/* ------------------------------------------------------------------ */

const recentImages = ref<Array<Record<string, unknown>>>([]);
const recentGalleryThumbFailed = ref<Record<string, boolean>>({});

const loadRecentImages = async () => {
  try {
    const images = await api.gallery.listImages(24, 0);
    recentGalleryThumbFailed.value = {};
    recentImages.value = (images as Array<Record<string, unknown>>)
      .filter((v: Record<string, unknown>) => {
        const meta = v.metadata as Record<string, unknown> | undefined;
        if (meta?.asset_kind === 'video' || meta?.asset_kind === 'audio') return false;
        const ext = String(v.name || '').split('.').pop()?.toLowerCase() || '';
        return !['mp4', 'mov', 'avi', 'mkv', 'webm', 'wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
      })
      .slice(0, 4);
  } catch (e) {
    console.error('Failed to load recent images:', e);
  }
};

/* ------------------------------------------------------------------ */
/*  App settings defaults                                              */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

onMounted(async () => {
  await loadModelRegistry();
  loadPresets();
  loadRecentImages();
  loadGallery(true);
  tasksStore.ensureQueuePoller();
});

onUnmounted(() => {
  closeGenStream();
  tasksStore.releaseQueuePoller();
});

</script>

<style scoped>
.studio-dialog-center {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.studio-dialog-img {
  max-width: 100%;
  max-height: 70vh;
  border-radius: 8px;
  object-fit: contain;
}

.studio-editor-drawer {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.studio-extend-panel,
.studio-upscale-panel,
.studio-retouch-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  margin-top: 0;
  padding-top: 0;
  border-top: none;
}

.studio-retouch-editor-wrap {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.studio-drawer-submit {
  margin-top: auto;
}
</style>

<style>
.studio-image-editor-drawer .dq-drawer-body {
  display: flex;
  flex-direction: column;
  padding: 16px 18px 20px;
  overflow: hidden;
}

.studio-image-editor-drawer .studio-create-pref-pane.dq-pref-pane {
  border: 0.5px solid var(--dq-glass-border);
  border-radius: var(--dq-radius-group);
  background: var(--dq-glass-grouped-bg);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  backdrop-filter: var(--dq-glass-blur-light);
}

.studio-image-editor-drawer .studio-create-pref-pane .dq-pref-row:first-child {
  padding-top: 10px;
}
</style>

<style>
.studio-drawer-model-label {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-drawer-model-badge {
  flex-shrink: 0;
  margin-left: 6px;
}
</style>
