<template>
  <StudioLayout
    class="studio-create-page"
    :freeform="viewMode === 'canvas'"
    :collapsible="viewMode === 'canvas'"
    v-model:composer-collapsed="composerCollapsed"
    @scroll="onCanvasScroll"
  >
    <template #filters>
      <StudioGalleryFilters
        :filter-time="filterTime"
        :filter-models="filterModels"
        :time-options="timeOptions"
        :model-options="allModelOptions"
        :selection-mode="selectionMode"
        :selected-count="selectedPaths.size"
        :all-selected="allLoadedSelected"
        :view-mode="viewMode"
        :supports-canvas="true"
        canvas-media="video"
        @update:filter-time="filterTime = $event"
        @update:filter-models="filterModels = $event"
        @refresh="refreshGallery"
        @toggle-selection-mode="toggleSelectionMode"
        @select-all="selectAllLoaded"
        @batch-delete="batchDeleteSelected"
        @clear-selection="clearSelection"
        @update:view-mode="onViewModeChange"
        @composer-restore="onCanvasComposerRestore"
      />
    </template>

    <template #canvas>
      <StudioCanvas
        v-if="viewMode === 'grid'"
        :items="galleryItems"
        :active-tasks="activeVideoTasks"
        :loading="galleryLoading"
        :has-more="galleryHasMore"
        media="video"
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
      <InfiniteCanvas
        v-else
        ref="infiniteCanvasRef"
        :items="galleryItems"
        media="video"
        @card-action="onCardAction"
        @download="downloadVideoItem"
        @toggle-grid-view="() => onViewModeChange('grid')"
        @node-selected="onCanvasNodeSelected"
        @session-ready="onCanvasSessionReady"
        @open-preview="onCanvasOpenPreview"
        @composer-restore="onCanvasComposerRestore"
        @overlay-cleared="onCanvasOverlayCleared"
        @use-as-start-frame="onCanvasUseAsStartFrame"
        @use-as-tail-frame="onCanvasUseAsTailFrame"
        @use-as-video-source="onCanvasUseAsVideoSource"
      />
    </template>

    <template #composer>
      <VideoComposer
        v-model="params.prompt"
        v-model:title="params.title"
        v-model:work-mode="videoWorkMode"
        v-model:model="selectedModelVersion"
        v-model:size="selectedSize"
        v-model:duration="selectedDurationSec"
        v-model:batch-count="batchCount"
        :generating="generating"
        :can-generate="!submitDisabled"
        :generate-label="primaryCtaLabel"
        :model-options="videoModelSelectOptions"
        :size-options="sizeOptions"
        :duration-options="durationOptions"
        :styles="filteredPresets"
        :params="params"
        :has-custom-params="hasCustomParams"
        :show-negative-prompt="!!currentModelConfig?.parameters?.negative_prompt_support"
        :show-lora="!!currentModelConfig?.parameters?.lora_support"
        :show-batch-count="true"
        :reference-media="referenceMedia"
        :reference-asset-path="startImagePath || null"
        :enhancing="isEnhancing"
        :reversing="isReversing"
        :tail-reference-media="tailReferenceMedia"
        :current-model-config="currentModelConfig"
        :compatible-loras="compatibleLoras"
        :model-not-ready="selectedModelNotReady"
        :model-not-ready-name="currentModelDisplayName"
        :work-mode-options="videoWorkSegmentOptions"
        @generate="startGeneration"
        @pick-reference="showAssetPicker = true"
        @remove-reference="removeStartImage"
        @pick-tail-reference="showTailAssetPicker = true"
        @remove-tail-reference="removeTailImage"
        @model-change="onModelVersionChange"
        @reset-defaults="resetToDefaults"
        @go-download="goToDownload"
        @enhance="onEnhancePrompt"
        @reverse-prompt="onReversePromptFromReference"
        :prompt-apply-preview="promptApplyPreview"
        @prompt-apply-replace="onPromptApplyReplace"
        @prompt-apply-append="onPromptApplyAppend"
        @prompt-apply-dismiss="promptApply.clear()"
        :collapsed="viewMode === 'canvas' && composerCollapsed"
        :storyboard-expanding="isStoryboardExpanding"
        @storyboard-expand="onStoryboardExpand"
      />
    </template>
  </StudioLayout>

  <!-- Start image asset picker -->
  <DqDialog v-model:open="showAssetPicker" :title="assetPickerTitle" width="70%">
    <AssetPicker
      :accept-kind="videoWorkMode === 'upscale' ? 'video' : 'image'"
      :recent-gallery="recentStartImages"
      @pick="onStartAssetPick"
    />
  </DqDialog>

  <!-- Tail image asset picker -->
  <DqDialog v-model:open="showTailAssetPicker" :title="$t('video.tailFrameTitle')" width="70%">
    <AssetPicker
      accept-kind="image"
      :recent-gallery="recentStartImages"
      @pick="onTailAssetPick"
    />
  </DqDialog>

  <!-- Start image preview dialog -->
  <DqDialog v-model:open="startImageViewerVisible" :title="$t('action.video.startImage')" width="70%" center>
    <div v-if="startImageSrc" class="studio-dialog-center">
      <img class="studio-dialog-img-tall" :src="startImageSrc" />
    </div>
  </DqDialog>

  <!-- Tail image preview dialog -->
  <DqDialog v-model:open="tailImageViewerVisible" :title="$t('video.tailFrameTitle')" width="70%" center>
    <div v-if="tailImageSrc" class="studio-dialog-center">
      <img class="studio-dialog-img-tall" :src="tailImageSrc" />
    </div>
  </DqDialog>

  <!-- Video preview dialog -->
  <GalleryPreviewDialog
    v-model:visible="videoPreviewVisible"
    v-model:index="selectedVideoIndex"
    :items="galleryItems"
    media="video"
  />

  <StudioLineagePanel
    v-model:modelValue="showLineageDrawer"
    :asset-id="lineageTargetAssetId"
    :on-canvas-ids="canvasAssetIdsOnCanvas"
    @focus-asset="onLineageFocusAsset"
  />
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, onMounted, inject, nextTick } from 'vue';
import type { Ref } from 'vue';
import { useRouter } from 'vue-router';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt, $mn, $md, $mvn, $pn } from '@/utils/i18n';
import { useRegistryStore } from '@/stores/registry';
import { useTasksStore } from '@/stores/tasks';
import type { SystemInfo, GalleryItem, Task } from '@/types';
import { warnIfRiskyMemory } from '@/composables/memoryHint';
import { pickDefaultVersionKey, resolveDefaultModelRegistryKey } from '@/utils/defaultModelSettings';
import { useModelRegistryFilters, reconcileVersionPickerSelection } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import GalleryPreviewDialog from '@/components/gallery/GalleryPreviewDialog.vue';
import { previewDisplayCaption, truncateDisplayLabel } from '@/utils/assetDisplay';

// Studio components
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvas from '@/components/studio/StudioCanvas.vue';
import StudioGalleryFilters from '@/components/studio/StudioGalleryFilters.vue';
import VideoComposer from '@/components/studio/VideoComposer.vue';
import InfiniteCanvas from '@/components/studio/InfiniteCanvas.vue';
import StudioLineagePanel from '@/components/studio/StudioLineagePanel.vue';
import { canvasAutoAddEnabled, useCanvasStore } from '@/composables/useCanvasStore';
import { useComposerCollapse } from '@/composables/useComposerCollapse';
import {
  activateCanvasViewForResults,
  maybeShowCanvasWorkspaceHint,
} from '@/utils/canvasWorkspaceHint';
import { previewUrlForGalleryItem } from '@/utils/canvasAssets';
import { useStudioGallery } from '@/composables/useStudioGallery';
import { useComposerLlm } from '@/composables/useComposerLlm';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { applyPromptDraft, consumePromptDraft } from '@/utils/promptApply';
import { usePromptApplyOffer } from '@/composables/usePromptApplyOffer';
import {
  buildResolutionSizeOptions,
  loadImageNaturalSize,
  parseSizeValue,
  pickClosestResolutionPreset,
  pickResolutionForModel,
} from '@/utils/registryParamSchema';
import {
  getVideoSizeForModel,
  migrateLegacyVideoLastSize,
  setVideoSizeForModel,
} from '@/utils/videoSizeStorage';
import {
  formatStoryboardScript,
  isLongVideoTargetDuration,
  parseStoryboardPrompt,
} from '@/utils/videoStoryboardPrompt';

const router = useRouter();
const registryStore = useRegistryStore();
const tasksStore = useTasksStore();
const systemInfo = inject<Ref<SystemInfo>>('systemInfo');

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

function hasAction(actions: Record<string, unknown>, key: string) {
  if (!actions || typeof actions !== 'object') return false;
  return Object.prototype.hasOwnProperty.call(actions, key) && actions[key] != null;
}
function videoModelRow(config: Record<string, unknown>) {
  return config && config.media === 'video';
}
function videoSupportsAnimate(actions: Record<string, unknown>) {
  return hasAction(actions, 'animate');
}
function videoSupportsUpscale(actions: Record<string, unknown>) {
  return hasAction(actions, 'upscale');
}
function videoSupportsCreate(actions: Record<string, unknown>) {
  return hasAction(actions, 'create');
}

function tagType(status: string) {
  const map: Record<string, string> = {
    pending: 'info',
    queued: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info',
  };
  return map[status] || 'info';
}
function statusText(status: string) {
  const suffix: Record<string, string> = {
    pending: 'pending',
    queued: 'queued',
    running: 'running',
    completed: 'completed',
    failed: 'failed',
    cancelled: 'cancelled',
  };
  const suf = suffix[status] || status;
  return $tt('studio.' + suf);
}

function parseModelVersionValue(value: string) {
  if (!value || typeof value !== 'string') return null;
  const parts = value.split('|');
  if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
  return { modelKey: parts[0], versionKey: parts[1] };
}

/* ------------------------------------------------------------------ */
/*  Params                                                             */
/* ------------------------------------------------------------------ */

const params = reactive({
  title: '',
  prompt: '',
  negative_prompt: '',
  model: '',
  version: '',
  width: 720,
  height: 480,
  num_frames: 49,
  fps: 8,
  steps: 50,
  guide_scale: 5.0,
  shift: 5.0,
  seed: '',
  image_path: '',
  upscale_scale: 4,
  upscale_denoise: 0.3,
  upscale_max_frames: 300,
  lora: '',
  lora_scale: 1.0,
});

const selectedModelVersion = ref('');
const selectedSize = ref('720x1280');
const selectedDurationSec = ref(5);
const batchCount = ref(1);

// State
const currentTask = ref<any>(null);
const generating = ref(false);
const infiniteCanvasRef = ref<InstanceType<typeof InfiniteCanvas> | null>(null);
const pendingCanvasAssetIds = ref<string[]>([]);
const videoCanvas = useCanvasStore('video');
const canvasSelectedItem = ref<GalleryItem | null>(null);
const { collapsed: composerCollapsed, setCollapsed: setComposerCollapsed } = useComposerCollapse('video');

const savedViewMode = getItem(DQ_STORAGE.VIDEO_VIEW_MODE);
const viewMode = ref<'grid' | 'canvas'>(
  savedViewMode === 'canvas' || savedViewMode === 'grid' ? savedViewMode : 'grid'
);

watch(viewMode, (mode) => {
  setItem(DQ_STORAGE.VIDEO_VIEW_MODE, mode);
});

function onViewModeChange(mode: 'grid' | 'canvas') {
  if (viewMode.value === mode) return;
  const batchPaths =
    mode === 'canvas' && selectedPaths.value.size > 0
      ? Array.from(selectedPaths.value)
      : [];
  viewMode.value = mode;
  if (mode === 'canvas') {
    clearSelection();
    maybeShowCanvasWorkspaceHint();
    const batchCount = batchPaths.length;
    nextTick(() => {
      syncCompositorOverlaysOnCanvasEnter();
      persistComposerSnapshot();
      if (batchCount > 0) {
        addAssetPathsToCanvas(batchPaths, {
          fit: true,
          placement: 'center',
          focusLast: false,
        });
        toast.success($tt('canvas.batchAdded', { n: batchCount }));
      } else if (Object.keys(videoCanvas.items).length > 0) {
        infiniteCanvasRef.value?.fitAll();
      }
    });
  } else {
    clearSelection();
  }
}

function shouldAutoAddToCanvas(): boolean {
  return viewMode.value === 'canvas' || canvasAutoAddEnabled('video');
}

function addAssetPathsToCanvas(
  paths: string[],
  opts?: {
    fit?: boolean;
    placement?: 'staging' | 'center';
    focusLast?: boolean;
  }
) {
  if (!paths.length) return;
  if (viewMode.value === 'canvas' && infiniteCanvasRef.value) {
    infiniteCanvasRef.value.addPathsToCanvas(paths, {
      fit: opts?.fit,
      placement: opts?.placement ?? 'staging',
      focusLast: opts?.focusLast ?? true,
    });
    return;
  }
  videoCanvas.addPathsFromGallery(paths, galleryItems.value, {
    placement: opts?.placement ?? 'center',
  });
}

function bindVideoAssetFromPath(path: string): { path: string; previewUrl: string } {
  const item = galleryItems.value.find((g) => g.path === path);
  const previewUrl = item
    ? previewUrlForGalleryItem(item)
    : path.startsWith('asset:')
      ? `/api/assets/${path.slice('asset:'.length)}/file`
      : path;
  return { path, previewUrl };
}

function onCanvasSessionReady(_payload: { sessionId: string }) {
  if (viewMode.value === 'canvas') {
    nextTick(() => {
      syncCompositorOverlaysOnCanvasEnter();
      if (Object.keys(videoCanvas.items).length > 0) {
        infiniteCanvasRef.value?.fitAll();
      }
    });
  }
}

function syncCanvasOverlay(
  kind: import('@/types').CanvasOverlayKind,
  path: string | null
) {
  if (!path) {
    videoCanvas.clearOverlay(kind);
    return;
  }
  const item = galleryItems.value.find((g) => g.path === path) ?? undefined;
  videoCanvas.setOverlay(kind, path, item);
}

function restoreComposerFromCanvasOverlays() {
  const sf = videoCanvas.overlays.start_frame;
  if (!startImagePath.value && sf?.path) {
    const p = bindVideoAssetFromPath(sf.path);
    startImagePath.value = p.path;
    startImageSrc.value = p.previewUrl;
    videoWorkMode.value = 'animate';
  }
  const tf = videoCanvas.overlays.tail_frame;
  if (!tailImagePath.value && tf?.path) {
    const p = bindVideoAssetFromPath(tf.path);
    tailImagePath.value = p.path;
    tailImageSrc.value = p.previewUrl;
    videoWorkMode.value = 'animate';
  }
  const vs = videoCanvas.overlays.video_source;
  if (!sourceVideoPath.value && vs?.path) {
    const p = bindVideoAssetFromPath(vs.path);
    sourceVideoPath.value = p.path;
    sourceVideoSrc.value = p.previewUrl;
    videoWorkMode.value = 'upscale';
  }
}

function syncCompositorOverlaysOnCanvasEnter() {
  restoreComposerFromCanvasOverlays();
  syncCanvasOverlay('start_frame', startImagePath.value || null);
  syncCanvasOverlay('tail_frame', tailImagePath.value || null);
  syncCanvasOverlay('video_source', sourceVideoPath.value || null);
}

function onCanvasOverlayCleared(kind: import('@/types').CanvasOverlayKind) {
  if (kind === 'start_frame') {
    startImageSrc.value = '';
    startImagePath.value = '';
  } else if (kind === 'tail_frame') {
    tailImageSrc.value = '';
    tailImagePath.value = '';
  } else if (kind === 'video_source') {
    sourceVideoSrc.value = '';
    sourceVideoPath.value = '';
  }
}

function onCanvasNodeSelected(item: GalleryItem | null) {
  canvasSelectedItem.value = item;
  if (!item) {
    persistComposerSnapshot();
    return;
  }
  setComposerCollapsed(false);
  if (item.prompt) params.prompt = item.prompt;
  if (item.title) params.title = item.title;
  if (item.model) {
    const raw = String(item.model);
    if (raw.includes(':')) {
      const [modelKey, versionKey] = raw.split(':', 2);
      params.model = modelKey;
      params.version = versionKey;
      selectedModelVersion.value = `${modelKey}|${versionKey}`;
    } else {
      params.model = raw;
      selectedModelVersion.value = raw;
    }
  }
  persistComposerSnapshot();
}

function persistComposerSnapshot() {
  videoCanvas.setComposerSnapshot({
    prompt: String(params.prompt || ''),
    title: String(params.title || ''),
    model: String(params.model || ''),
    version: String(params.version || ''),
    negative_prompt: String(params.negative_prompt || ''),
    seed: params.seed != null ? String(params.seed) : undefined,
    mode: videoWorkMode.value,
    start_image_path: startImagePath.value || undefined,
    tail_image_path: tailImagePath.value || undefined,
    source_video_path: sourceVideoPath.value || undefined,
  });
}

function onCanvasComposerRestore(snapshot: {
  prompt?: string;
  title?: string;
  model?: string;
  version?: string;
  negative_prompt?: string;
  seed?: string;
  mode?: string;
  start_image_path?: string;
  tail_image_path?: string;
  source_video_path?: string;
}) {
  if (!snapshot || typeof snapshot !== 'object') return;
  if (snapshot.prompt != null) params.prompt = snapshot.prompt;
  if (snapshot.title != null) params.title = snapshot.title;
  if (snapshot.negative_prompt != null) params.negative_prompt = snapshot.negative_prompt;
  if (snapshot.seed != null) params.seed = snapshot.seed;
  if (snapshot.mode === 'create' || snapshot.mode === 'animate' || snapshot.mode === 'upscale') {
    videoWorkMode.value = snapshot.mode;
  }
  if (snapshot.model) {
    params.model = snapshot.model;
    if (snapshot.version) {
      params.version = snapshot.version;
      selectedModelVersion.value = `${snapshot.model}|${snapshot.version}`;
    } else {
      selectedModelVersion.value = snapshot.model;
    }
  }
  if (snapshot.start_image_path) {
    const p = bindVideoAssetFromPath(snapshot.start_image_path);
    startImagePath.value = p.path;
    startImageSrc.value = p.previewUrl;
  }
  if (snapshot.tail_image_path) {
    const p = bindVideoAssetFromPath(snapshot.tail_image_path);
    tailImagePath.value = p.path;
    tailImageSrc.value = p.previewUrl;
  }
  if (snapshot.source_video_path) {
    const p = bindVideoAssetFromPath(snapshot.source_video_path);
    sourceVideoPath.value = p.path;
    sourceVideoSrc.value = p.previewUrl;
  }
  if (viewMode.value === 'canvas') {
    nextTick(() => syncCompositorOverlaysOnCanvasEnter());
  }
}

function buildCanvasMeta(extra: Record<string, unknown> = {}): Record<string, unknown> {
  const meta: Record<string, unknown> = { ...extra };
  const sid = videoCanvas.sessionId.value;
  if (sid) meta.canvas_session_id = sid;
  const parentPath =
    canvasSelectedItem.value?.path || videoCanvas.activeAssetPath.value || '';
  if (parentPath.startsWith('asset:')) {
    meta.parent_asset_id = parentPath.slice('asset:'.length);
    if (videoWorkMode.value === 'animate') meta.relation_type = 'animate';
    else if (videoWorkMode.value === 'upscale') meta.relation_type = 'upscale';
    else meta.relation_type = 'create';
  }
  return meta;
}

function onCanvasOpenPreview(item: GalleryItem) {
  const idx = galleryItems.value.findIndex((g) => g.path === item.path);
  if (idx >= 0) {
    selectedVideoIndex.value = idx;
    videoPreviewVisible.value = true;
  }
}

function downloadVideoItem(item: GalleryItem) {
  const a = document.createElement('a');
  a.href = getVideoUrl(item);
  a.download = item.name || 'video.mp4';
  a.click();
  toast.success($tt('gallery.startDownload'));
}
const previewVideo = ref('');
const previewVideoKey = ref(0);
const previewVideoDurationSec = ref(0);

const previewCaption = computed(() =>
  previewDisplayCaption(String(params.title || ''), String(params.prompt || '')),
);

function formatPreviewClock(sec: number) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  return m + ':' + String(s % 60).padStart(2, '0');
}

const previewVideoSubtitle = computed(() => {
  const parts: string[] = [];
  if (currentModelDisplayName.value) parts.push(currentModelDisplayName.value);
  if (previewVideoDurationSec.value > 0) {
    parts.push(formatPreviewClock(previewVideoDurationSec.value));
  } else if (params.num_frames > 0 && params.fps > 0) {
    parts.push(formatPreviewClock((params.num_frames - 1) / params.fps));
  }
  return parts.join(' · ');
});

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
} = useStudioGallery('video');

const activeVideoTasks = computed(() => {
  const running = tasksStore.queueState.running.filter((t: Task) =>
    String(t.kind || '').startsWith('video.')
  );
  const queued = tasksStore.queueState.queued.filter((t: Task) =>
    String(t.kind || '').startsWith('video.')
  );
  return [...running, ...queued].map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
});

const showLineageDrawer = ref(false);
const lineageTargetAssetId = ref('');

const canvasAssetIdsOnCanvas = computed(() =>
  Object.keys(videoCanvas.items)
    .filter((p) => p.startsWith('asset:'))
    .map((p) => p.slice('asset:'.length))
);

async function onLineageFocusAsset(assetId: string) {
  if (viewMode.value !== 'canvas') {
    await activateCanvasViewForResults(viewMode, syncCompositorOverlaysOnCanvasEnter);
  }
  infiniteCanvasRef.value?.focusLineageAsset(assetId);
}

async function addGalleryItemToCanvas(
  item: GalleryItem,
  opts?: { switchView?: boolean; fit?: boolean }
) {
  if (opts?.switchView !== false && viewMode.value !== 'canvas') {
    onViewModeChange('canvas');
    await nextTick();
    await nextTick();
  }
  addAssetPathsToCanvas([item.path], {
    placement: 'center',
    focusLast: true,
    fit: opts?.fit,
  });
  toast.success($tt('canvas.addedToCanvas'));
}

function onCardAction({ action, item }: { action: string; item: GalleryItem }) {
  if (action === 'add-to-canvas') {
    void addGalleryItemToCanvas(item, { fit: true });
  } else if (action === 'delete') {
    deleteItem(item);
  } else if (action === 'download') {
    downloadVideoItem(item);
  } else if (action === 'lineage') {
    lineageTargetAssetId.value = item.path.startsWith('asset:')
      ? item.path.slice('asset:'.length)
      : '';
    showLineageDrawer.value = true;
  }
}

/* ------------------------------------------------------------------ */
/*  Reference Media                                                    */
/* ------------------------------------------------------------------ */

const startImageSrc = ref('');
const startImagePath = ref('');
const startImageViewerVisible = ref(false);
const tailImageSrc = ref('');
const tailImagePath = ref('');
const tailImageViewerVisible = ref(false);
const sourceVideoSrc = ref('');
const sourceVideoPath = ref('');

watch(
  [startImagePath, tailImagePath, sourceVideoPath],
  () => {
    persistComposerSnapshot();
    if (viewMode.value === 'canvas') {
      syncCanvasOverlay('start_frame', startImagePath.value || null);
      syncCanvasOverlay('tail_frame', tailImagePath.value || null);
      syncCanvasOverlay('video_source', sourceVideoPath.value || null);
    }
  }
);

const showAssetPicker = ref(false);
const showTailAssetPicker = ref(false);

function removeStartImage() {
  if (videoWorkMode.value === 'upscale') {
    sourceVideoSrc.value = '';
    sourceVideoPath.value = '';
    return;
  }
  startImageSrc.value = '';
  startImagePath.value = '';
}

function removeTailImage() {
  tailImageSrc.value = '';
  tailImagePath.value = '';
}

function onModelVersionChange(val: string) {
  const parsed = parseModelVersionValue(val);
  if (!parsed) return;
  params.model = parsed.modelKey;
  params.version = parsed.versionKey;
  loadModelDefaults();
  syncResolutionForModel(parsed.modelKey);
  loadCompatibleLoras();
}

const { isEnhancing, isReversing, enhance: doEnhance, reversePrompt, storyboardLongVideo, isStoryboardExpanding } = useComposerLlm();
const promptApply = usePromptApplyOffer();
const promptApplyPreview = computed(() => promptApply.pending.value?.result ?? null);

async function onEnhancePrompt(ctx?: { stylePositive?: string }) {
  const prompt = String(params.prompt || '').trim();
  if (!prompt) return;
  const enhanced = await doEnhance(
    prompt,
    ctx?.stylePositive,
    'video_create',
    params.model,
    { quietSuccess: true },
  );
  if (enhanced) {
    promptApply.offer(params.prompt, enhanced, (text) => { params.prompt = text; });
  }
}

async function onStoryboardExpand() {
  const prompt = String(params.prompt || '').trim();
  if (!prompt) return;
  const p = currentModelConfig.value?.parameters;
  const extendSec = Number(p?.long_video_segment_extend_sec?.default ?? 7);
  const refSec = Number(p?.long_video_reference_duration_sec?.default ?? 3);
  const result = await storyboardLongVideo({
    prompt,
    target_duration_sec: selectedDurationSec.value,
    initial_duration_sec: 8,
    segment_extend_sec: extendSec,
    reference_duration_sec: refSec,
  });
  if (!result) return;
  params.prompt = formatStoryboardScript(result.opening_prompt, result.segment_prompts || []);
}

async function onReversePromptFromReference() {
  const assetId = assetIdFromGalleryPath(startImagePath.value || '');
  if (!assetId) {
    toast.warning($tt('create.reverseNeedAsset'));
    return;
  }
  const prompt = await reversePrompt(assetId, { quietSuccess: true });
  if (prompt) {
    promptApply.offer(params.prompt, prompt, (text) => { params.prompt = text; });
  }
}

function onPromptApplyReplace() {
  promptApply.applyReplace((text) => { params.prompt = text; });
}

function onPromptApplyAppend() {
  promptApply.applyAppend(() => params.prompt, (text) => { params.prompt = text; });
}

function goToDownload() {
  router.push({ name: 'models' });
}

function onStartAssetPick(payload: { path: string; previewUrl: string }) {
  if (videoWorkMode.value === 'upscale') {
    sourceVideoSrc.value = payload.previewUrl || '';
    sourceVideoPath.value = payload.path;
    if (viewMode.value === 'canvas') {
      syncCanvasOverlay('video_source', payload.path);
    }
  } else {
    startImageSrc.value = payload.previewUrl || '';
    startImagePath.value = payload.path;
    if (viewMode.value === 'canvas') {
      syncCanvasOverlay('start_frame', payload.path);
    }
    void syncResolutionForStartImage(payload.previewUrl || '');
  }
  showAssetPicker.value = false;
}

function onCanvasUseAsStartFrame(payload: { path: string; previewUrl: string; quiet?: boolean }) {
  videoWorkMode.value = 'animate';
  startImageSrc.value = payload.previewUrl || '';
  startImagePath.value = payload.path;
  syncCanvasOverlay('start_frame', payload.path);
  void syncResolutionForStartImage(payload.previewUrl || '');
  if (!payload.quiet) toast.success($tt('canvas.startFrameBound'));
}

function onCanvasUseAsTailFrame(payload: { path: string; previewUrl: string; quiet?: boolean }) {
  videoWorkMode.value = 'animate';
  tailImageSrc.value = payload.previewUrl || '';
  tailImagePath.value = payload.path;
  syncCanvasOverlay('tail_frame', payload.path);
  if (!payload.quiet) toast.success($tt('canvas.tailFrameBound'));
}

function onCanvasUseAsVideoSource(payload: { path: string; previewUrl: string; quiet?: boolean }) {
  videoWorkMode.value = 'upscale';
  sourceVideoSrc.value = payload.previewUrl || '';
  sourceVideoPath.value = payload.path;
  syncCanvasOverlay('video_source', payload.path);
  if (!payload.quiet) toast.success($tt('canvas.videoSourceBound'));
}

function onTailAssetPick(payload: { path: string; previewUrl: string }) {
  tailImageSrc.value = payload.previewUrl || '';
  tailImagePath.value = payload.path;
  if (viewMode.value === 'canvas') {
    syncCanvasOverlay('tail_frame', payload.path);
  }
  showTailAssetPicker.value = false;
}

const assetPickerTitle = computed(() => {
  if (videoWorkMode.value === 'animate') return $tt('action.video.startImage');
  if (videoWorkMode.value === 'upscale') return $tt('video.videoSourceTitle');
  return $tt('create.refImage');
});

const referenceMedia = computed(() => {
  if (videoWorkMode.value === 'animate' && startImageSrc.value) {
    return {
      type: 'image' as const,
      previewUrl: startImageSrc.value,
      label: $tt('action.video.startImage'),
    };
  }
  if (videoWorkMode.value === 'upscale' && sourceVideoSrc.value) {
    return {
      type: 'video' as const,
      previewUrl: sourceVideoSrc.value,
      label: $tt('video.videoSourceTitle'),
    };
  }
  return null;
});

const tailReferenceMedia = computed(() => {
  if (videoWorkMode.value === 'animate' && tailImageSrc.value) {
    return {
      type: 'image' as const,
      previewUrl: tailImageSrc.value,
      label: $tt('video.tailFrameTitle'),
    };
  }
  return undefined;
});

const recentStartImages = ref<GalleryItem[]>([]);

/* ------------------------------------------------------------------ */
/*  Work Mode                                                          */
/* ------------------------------------------------------------------ */

const videoWorkMode = ref('create');

function longVideoSupported(): boolean {
  return Boolean(currentModelConfig.value?.parameters?.long_video_support);
}

function isLongVideoCreate(): boolean {
  return videoWorkMode.value === 'create'
    && longVideoSupported()
    && isLongVideoTargetDuration(selectedDurationSec.value, true);
}

const videoWorkSegmentOptions = computed(() => {
  const acts = currentModelConfig.value?.actions || {};
  const opts: { label: string; value: string }[] = [];
  if (videoSupportsCreate(acts)) {
    opts.push({ label: $tt('action.video.create'), value: 'create' });
  }
  if (videoSupportsAnimate(acts)) {
    opts.push({ label: $tt('action.video.animate'), value: 'animate' });
  }
  if (videoSupportsUpscale(acts)) {
    opts.push({ label: $tt('action.video.upscale'), value: 'upscale' });
  }
  if (opts.length === 0) {
    return [
      { label: $tt('action.video.create'), value: 'create' },
      { label: $tt('action.video.animate'), value: 'animate' },
      { label: $tt('action.video.upscale'), value: 'upscale' },
    ];
  }
  return opts;
});

/* ------------------------------------------------------------------ */
/*  Model Registry                                                     */
/* ------------------------------------------------------------------ */

const modelRegistry = ref<Record<string, any>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});

const compatibleLoras = ref<{ id: string; name?: string; parameters?: any }[]>([]);

function buildVideoAdapters() {
  const adapters: { id: string; weight: number }[] = [];
  if (params.lora) {
    adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 1.0 });
  }
  return adapters;
}

const loadCompatibleLoras = async () => {
  if (!params.model) {
    compatibleLoras.value = [];
    return;
  }
  try {
    const loras = await api.settings.getCompatibleLoras(params.model);
    compatibleLoras.value = (loras as any[]) || [];
  } catch (e) {
    console.error('Failed to load compatible loras:', e);
    compatibleLoras.value = [];
  }
};

const allVersions = computed(() => {
  const result: any[] = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!videoModelRow(config)) {
      continue;
    }
    const actions = { ...(config.actions || {}) };
    const versions = config.versions || {};
    const detailed = modelsDetailedStatus.value[modelKey] || {};
    const versionStatuses = detailed.versions || {};

    for (const [versionKey, versionConfig] of Object.entries(versions)) {
      const status = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
      result.push({
        modelKey,
        versionKey,
        name: $mvn(modelKey, config, versionConfig as any),
        description: $md(config as { description?: string | { zh?: string; en?: string }; description_en?: string }),
        size: (versionConfig as any).size || '',
        status: status.status,
        ready: status.ready,
        recommended: config.recommended && (versionConfig as any).default,
        commercialUseAllowed: config.commercial_use_allowed === true,
        actions,
      });
    }
  }
  return result;
});

const videoVersionsForMode = computed(() => {
  const filtered = allVersions.value.filter((v) => {
    const acts = v.actions || {};
    if (videoWorkMode.value === 'animate') {
      return videoSupportsAnimate(acts);
    }
    if (videoWorkMode.value === 'upscale') {
      return videoSupportsUpscale(acts);
    }
    return videoSupportsCreate(acts);
  });
  if (videoWorkMode.value === 'upscale' || videoWorkMode.value === 'animate') {
    return filtered;
  }
  return filtered.length ? filtered : allVersions.value;
});

const videoRecommendedForMode = computed(() => {
  return videoVersionsForMode.value.filter((v) => v.recommended);
});

const selectedModelPickerItem = computed(() => {
  const key = selectedModelVersion.value;
  if (!key) return null;
  return allVersions.value.find((item) => `${item.modelKey}|${item.versionKey}` === key) ?? null;
});

const { commercialOnly: modelFilterCommercialOnly } = useModelRegistryFilters();

const videoModelPickerVersions = computed(() => {
  const rows = applyModelVersionFilters(videoVersionsForMode.value, {
    installedOnly: true,
    commercialOnly: modelFilterCommercialOnly.value,
  });
  rows.sort((a, b) => {
    const ar = a.recommended ? 1 : 0;
    const br = b.recommended ? 1 : 0;
    if (ar !== br) return br - ar;
    const an = a.name || '';
    const bn = b.name || '';
    try {
      return an.localeCompare(bn, 'zh');
    } catch {
      return an < bn ? -1 : an > bn ? 1 : 0;
    }
  });
  return rows;
});

const videoModelSelectOptions = computed(() => {
  return videoModelPickerVersions.value.map((v) => ({
    label: String(v.name || ''),
    value: `${v.modelKey}|${v.versionKey}`,
    disabled: !v.ready,
    commercialUseAllowed: v.commercialUseAllowed as boolean,
  }));
});

const currentModelConfig = computed(() => modelRegistry.value[params.model] || null);

const currentModelDisplayName = computed(() => {
  const c = currentModelConfig.value;
  if (c) {
    return $mn(c, params.model);
  }
  return params.model || '';
});

const selectedModelNotReady = computed(() => {
  if (!params.model || !params.version) return false;
  const detailed = modelsDetailedStatus.value[params.model];
  if (!detailed || !detailed.versions) return true;
  const versionStatus = detailed.versions[params.version];
  return !versionStatus || !versionStatus.ready;
});

const submitDisabled = computed(() => {
  if (generating.value) return true;
  if (selectedModelNotReady.value) return true;
  if (videoWorkMode.value === 'upscale') {
    return !sourceVideoSrc.value;
  }
  if (!String(params.prompt || '').trim()) return true;
  if (videoWorkMode.value === 'animate' && !startImageSrc.value) return true;
  return false;
});

const primaryCtaLabel = computed(() => {
  if (videoWorkMode.value === 'animate') return $tt('action.video.animate');
  if (videoWorkMode.value === 'upscale') return $tt('action.video.upscale');
  return $tt('action.video.create');
});

const outputClipSecRounded = computed(() => {
  const fps = Math.max(1, Number(params.fps) || 1);
  const nf = Math.max(1, Number(params.num_frames) || 1);
  return Math.round(((nf - 1) / fps) * 10) / 10;
});

const currentVersionDiskSize = computed(() => {
  const cfg = currentModelConfig.value;
  if (!cfg || !params.version) return '';
  const v = (cfg.versions || {})[params.version];
  return v && v.size ? String(v.size) : '';
});

/* ------------------------------------------------------------------ */
/*  Size & Duration Options (registry-driven)                          */
/* ------------------------------------------------------------------ */

function alignNumFrames(frames: number, schema?: { min?: number; max?: number; step?: number }) {
  let n = Math.round(frames);
  const min = schema?.min ?? 1;
  const max = schema?.max ?? 200;
  const step = schema?.step ?? 1;
  if (step > 1) {
    n = Math.round((n - 1) / step) * step + 1;
  }
  return Math.min(max, Math.max(min, n));
}

function durationSecFromFrames(frames: number, fps: number): number {
  const rate = Math.max(1, Number(fps) || 1);
  const nf = Math.max(1, Number(frames) || 1);
  return Math.max(0, (nf - 1) / rate);
}

function snapDurationSecToOptions(
  sec: number,
  options: Array<{ value: number }>,
): number {
  if (!options.length) return Math.max(1, Math.round(sec));
  let best = options[0].value;
  let bestDiff = Math.abs(sec - best);
  for (const opt of options) {
    const diff = Math.abs(sec - opt.value);
    if (diff < bestDiff) {
      bestDiff = diff;
      best = opt.value;
    }
  }
  return best;
}

function framesFromDurationSec(
  sec: number,
  schema?: { min?: number; max?: number; step?: number },
  fps?: number,
) {
  const rate = Math.max(1, Number(fps) || 1);
  const durationSec = Math.max(0, Number(sec) || 0);
  return alignNumFrames(durationSec * rate + 1, schema);
}

function syncNumFramesFromDuration() {
  const schema = currentModelConfig.value?.parameters?.num_frames;
  params.num_frames = framesFromDurationSec(selectedDurationSec.value, schema, params.fps);
}

const sizeOptions = computed(() =>
  buildResolutionSizeOptions(currentModelConfig.value?.parameters as Record<string, unknown> | undefined),
);

function applySelectedSize(val: string) {
  const parsed = parseSizeValue(val);
  if (!parsed) return;
  params.width = parsed.width;
  params.height = parsed.height;
}

async function syncResolutionForStartImage(previewUrl: string) {
  if (videoWorkMode.value !== 'animate' || !previewUrl) return;
  const options = sizeOptions.value;
  if (!options.length) return;
  try {
    const { width, height } = await loadImageNaturalSize(previewUrl);
    const pick = pickClosestResolutionPreset(options, width, height);
    if (!pick) return;
    if (selectedSize.value !== pick) selectedSize.value = pick;
    else applySelectedSize(pick);
  } catch {
    // keep current resolution when preview cannot be measured
  }
}

function syncResolutionForModel(modelId?: string, opts?: { ignoreSaved?: boolean }) {
  if (videoWorkMode.value === 'upscale') return;
  const mid = modelId || String(params.model || '');
  if (!mid) return;
  migrateLegacyVideoLastSize(mid);
  const saved = opts?.ignoreSaved ? null : getVideoSizeForModel(mid);
  const pick = pickResolutionForModel(
    currentModelConfig.value?.parameters as Record<string, unknown> | undefined,
    saved,
  );
  if (!pick) return;
  if (selectedSize.value !== pick) selectedSize.value = pick;
  else applySelectedSize(pick);
}

const durationOptions = computed(() => {
  const p = currentModelConfig.value?.parameters;
  const nf = p?.num_frames;
  const fps = p?.fps?.default ?? params.fps ?? 16;
  let secs = [1, 2, 3, 4, 5, 8, 10];
  if (nf?.default && fps > 0) {
    const regSec = Math.max(1, Math.round((Number(nf.default) - 1) / fps));
    if (!secs.includes(regSec)) {
      secs = [...secs, regSec].sort((a, b) => a - b);
    }
  }
  if (p?.long_video_support) {
    const longOpts = (p.long_video_target_duration_sec?.options as number[]) || [30, 60, 90];
    for (const s of longOpts) {
      const n = Number(s);
      if (n > 0 && !secs.includes(n)) secs.push(n);
    }
    secs.sort((a, b) => a - b);
  }
  return secs.map((sec) => ({ label: `${sec}s`, value: sec }));
});

/* ------------------------------------------------------------------ */
/*  Model Loading                                                      */
/* ------------------------------------------------------------------ */

const loadModelRegistry = async () => {
  try {
    const regPromise = registryStore.load(true);
    const [registryData, detailedStatusData] = await Promise.all([
      regPromise,
      api.settings.getModelsDetailedStatus(),
    ]);

    modelRegistry.value = (registryData && (registryData as any).models) || {};
    modelsDetailedStatus.value = (detailedStatusData as any) || {};

    // Set default model
    if (!selectedModelVersion.value) {
      let found = false;
      for (const item of videoRecommendedForMode.value) {
        if (item.ready) {
          params.model = item.modelKey;
          params.version = item.versionKey;
          selectedModelVersion.value = item.modelKey + '|' + item.versionKey;
          found = true;
          break;
        }
      }
      if (!found) {
        for (const item of videoVersionsForMode.value) {
          if (item.ready) {
            params.model = item.modelKey;
            params.version = item.versionKey;
            selectedModelVersion.value = item.modelKey + '|' + item.versionKey;
            found = true;
            break;
          }
        }
      }
      if (!found && videoVersionsForMode.value.length > 0) {
        const first = videoVersionsForMode.value[0];
        params.model = first.modelKey;
        params.version = first.versionKey;
        selectedModelVersion.value = first.modelKey + '|' + first.versionKey;
      }
    }

    loadModelDefaults();
    syncResolutionForModel(String(params.model || ''));
    loadCompatibleLoras();
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
};

function syncComposerPickersFromParams() {
  selectedDurationSec.value = snapDurationSecToOptions(
    durationSecFromFrames(params.num_frames, params.fps),
    durationOptions.value,
  );
  syncNumFramesFromDuration();
}

const loadModelDefaults = () => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return;

  const p = config.parameters;
  if (videoWorkMode.value === 'upscale') {
    if (p.scale_factor && p.scale_factor.default != null) {
      params.upscale_scale = p.scale_factor.default;
    }
    if (p.max_frames && p.max_frames.default != null) {
      params.upscale_max_frames = p.max_frames.default;
    }
    if (p.fps) params.fps = p.fps.default;
    params.seed = '';
    params.lora = '';
    return;
  }
  if (p.steps) params.steps = p.steps.default;
  if (p.guide_scale) params.guide_scale = p.guide_scale.default;
  if (p.shift) params.shift = p.shift.default;
  if (p.num_frames) params.num_frames = p.num_frames.default;
  if (p.fps) params.fps = p.fps.default;
  if (p.lora_scale?.default != null) params.lora_scale = p.lora_scale.default;
  params.seed = '';
  params.lora = '';
  syncComposerPickersFromParams();
};

const resetToDefaults = () => {
  loadModelDefaults();
  syncResolutionForModel(String(params.model || ''), { ignoreSaved: true });
  toast.success($tt('studio.restoredDefaults'));
};

const hasCustomParams = computed(() => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return false;
  const p = config.parameters;
  if (videoWorkMode.value === 'upscale') {
    if (p.scale_factor && params.upscale_scale !== p.scale_factor.default) return true;
    if (p.max_frames && params.upscale_max_frames !== p.max_frames.default) return true;
    if (params.seed) return true;
    return false;
  }
  if (p.steps && params.steps !== p.steps.default) return true;
  if (p.guide_scale && params.guide_scale !== p.guide_scale.default) return true;
  if (p.shift && params.shift !== p.shift.default) return true;
  if (p.width && params.width !== p.width.default) return true;
  if (p.height && params.height !== p.height.default) return true;
  if (p.num_frames && p.fps) {
    const defaultSec = snapDurationSecToOptions(
      durationSecFromFrames(p.num_frames.default, p.fps.default),
      durationOptions.value,
    );
    if (selectedDurationSec.value !== defaultSec) return true;
  }
  if (p.fps && params.fps !== p.fps.default) return true;
  if (params.seed) return true;
  if (params.lora) return true;
  if (p.lora_scale && params.lora_scale !== p.lora_scale.default) return true;
  return false;
});

/* ------------------------------------------------------------------ */
/*  Presets                                                            */
/* ------------------------------------------------------------------ */

const presets = ref<Record<string, any>>({});
const selectedPreset = ref('');

const presetActionFilter = computed(() => {
  if (videoWorkMode.value === 'animate') {
    return new Set(['animate']);
  }
  if (videoWorkMode.value === 'upscale') {
    return new Set(['upscale']);
  }
  return new Set(['create']);
});

const filteredPresets = computed(() => {
  const want = presetActionFilter.value;

  function planPresetShapeOk(preset: any) {
    return (
      Array.isArray(preset.applies_to) &&
      preset.applies_to.length > 0 &&
      (preset.media_scope === 'image' || preset.media_scope === 'video')
    );
  }

  function matchesMediaScope(preset: any) {
    return preset.media_scope === 'video';
  }

  function matches(preset: any) {
    if (!planPresetShapeOk(preset)) return false;
    if (!matchesMediaScope(preset)) return false;
    return preset.applies_to.some((k: string) => want.has(k));
  }
  const entries = Object.entries(presets.value)
    .filter(([, preset]) => matches(preset))
    .sort((a: [string, any], b: [string, any]) => {
      const ac = a[1].applies_to.includes('create');
      const bc = b[1].applies_to.includes('create');
      if (ac !== bc) {
        return ac ? -1 : 1;
      }
      return a[0].localeCompare(b[0], 'zh');
    });
  const result: Record<string, any> = {};
  for (const [name, preset] of entries) {
    result[name] = preset;
  }
  return result;
});

const presetSelectLabel = (name: string, preset: any) => {
  const a = preset.applies_to;
  const hasC = a.includes('create');
  const hasA = a.includes('animate');
  const hasU = a.includes('upscale');
  let tag = '';
  if (hasC && hasA) {
    tag = $tt('video.presetTagHybrid');
  } else if (hasC && !hasA) {
    tag = $tt('video.presetTagT2V');
  } else if (hasA && !hasC) {
    tag = $tt('video.presetTagI2V');
  } else if (hasU && !hasC && !hasA) {
    tag = $tt('video.presetTagUpscale');
  }
  const display = $pn(preset, name);
  return tag ? `${tag} ${display}` : display;
};

const loadPresets = async () => {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as any) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
    presets.value = {};
  }
};

const loadPreset = () => {
  if (!selectedPreset.value || !presets.value[selectedPreset.value]) return;
  const preset = presets.value[selectedPreset.value];
  const app = preset.applies_to;
  const animateOnly = app.includes('animate') && !app.includes('create');
  if (animateOnly && (videoWorkMode.value === 'create' || !startImageSrc.value)) {
    toast.warning($tt('video.presetNeedsStartImage'));
  }
  if (preset.positive) {
    params.prompt = params.prompt
      ? params.prompt + '\nStyle boost: ' + preset.positive
      : preset.positive;
  }
  if (preset.negative) {
    params.negative_prompt = params.negative_prompt
      ? params.negative_prompt + '\n' + preset.negative
      : preset.negative;
  }
};

/* ------------------------------------------------------------------ */
/*  Generation                                                         */
/* ------------------------------------------------------------------ */

const startGeneration = async () => {
  if (videoWorkMode.value !== 'upscale' && !String(params.prompt || '').trim()) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }

  const detailed = modelsDetailedStatus.value[params.model];
  const versionStatus = detailed?.versions?.[params.version];
  if (!versionStatus?.ready) {
    toast.warning(
      $tt('studio.modelNotReadyDesc', {
        name: currentModelConfig.value?.name || params.model,
        version: params.version,
      })
    );
    return;
  }

  const verCfg =
    (currentModelConfig.value &&
      currentModelConfig.value.versions &&
      currentModelConfig.value.versions[params.version]) ||
    null;
  const sizeHuman = verCfg && verCfg.size ? String(verCfg.size) : '';
  const minMemRaw = currentModelConfig.value?.parameters?.min_unified_memory_gb;
  const minUnifiedMemoryGb =
    minMemRaw != null && Number(minMemRaw) > 0 ? Number(minMemRaw) : null;
  warnIfRiskyMemory({
    systemInfo: systemInfo?.value,
    versionSizeHuman: sizeHuman,
    minUnifiedMemoryGb,
    $tt,
  });

  if (videoWorkMode.value === 'animate' && !startImageSrc.value) {
    toast.warning($tt('video.needStartImage'));
    return;
  }
  if (videoWorkMode.value === 'upscale' && !sourceVideoSrc.value) {
    toast.warning($tt('video.upscaleNeedSource'));
    return;
  }

  generating.value = true;
  currentTask.value = {
    id: '',
    progress: 0,
    step: 0,
    total: 0,
    status: 'submitting',
  };

  try {
    persistComposerSnapshot();
    const meta = buildCanvasMeta();
    const modelStr = params.version ? `${params.model}:${params.version}` : params.model;
    let submitRes: any;
    if (videoWorkMode.value === 'animate') {
      let source_asset_id: string;
      const sp = startImagePath.value;
      if (typeof sp === 'string' && sp.startsWith('asset:')) {
        source_asset_id = sp.slice('asset:'.length);
      } else {
        const blob = await api.gen.urlToBlob(startImageSrc.value);
        const up = await api.gen.uploadAsset(
          new File([blob], 'start.png', { type: blob.type || 'image/png' })
        );
        source_asset_id = (up as any).id;
      }
      let tail_asset_id: string | undefined;
      if (tailImageSrc.value) {
        const tp = tailImagePath.value;
        if (typeof tp === 'string' && tp.startsWith('asset:')) {
          tail_asset_id = tp.slice('asset:'.length);
        } else {
          const tblob = await api.gen.urlToBlob(tailImageSrc.value);
          const tup = await api.gen.uploadAsset(
            new File([tblob], 'tail.png', { type: tblob.type || 'image/png' })
          );
          tail_asset_id = (tup as any).id;
        }
      }
      const animateBody: Record<string, unknown> = {
        model: modelStr,
        operation: 'animate',
        source_asset_id,
        title: String(params.title || '').trim(),
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        size: `${params.width}x${params.height}`,
        num_frames: params.num_frames,
        fps: params.fps || 16,
        steps: params.steps,
        guidance: params.guide_scale,
        shift: params.shift || undefined,
        seed: params.seed ? parseInt(params.seed, 10) : null,
        metadata: { ...meta },
        priority: 'normal',
      };
      if (tail_asset_id) {
        animateBody.tail_asset_id = tail_asset_id;
      }
      const adapters = buildVideoAdapters();
      if (adapters.length > 0) {
        animateBody.adapters = adapters;
      }
      submitRes = await api.gen.createVideoEdit(animateBody);
    } else if (videoWorkMode.value === 'upscale') {
      let source_asset_id: string;
      const vp = sourceVideoPath.value;
      if (typeof vp === 'string' && vp.startsWith('asset:')) {
        source_asset_id = vp.slice('asset:'.length);
      } else {
        const blob = await api.gen.urlToBlob(sourceVideoSrc.value);
        const ext =
          (blob.type && blob.type.includes('webm') && 'webm') ||
          (blob.type && blob.type.includes('quicktime') && 'mov') ||
          'mp4';
        const up = await api.gen.uploadAsset(
          new File([blob], `upscale-src.${ext}`, { type: blob.type || 'video/mp4' })
        );
        source_asset_id = (up as any).id;
      }
      const upscaleBody: Record<string, unknown> = {
        model: modelStr,
        source_asset_id,
        scale: Number(params.upscale_scale) === 4 ? 4 : 2,
        denoise: Number(params.upscale_denoise) || 0.3,
        max_frames: Math.min(
          4000,
          Math.max(1, parseInt(String(params.upscale_max_frames), 10) || 300)
        ),
        metadata: { ...meta },
        priority: 'normal',
      };
      const sd = params.seed ? parseInt(String(params.seed), 10) : null;
      if (sd != null && !Number.isNaN(sd)) {
        (upscaleBody.metadata as Record<string, unknown>).seed = sd;
      }
      submitRes = await api.gen.createVideoUpscale(upscaleBody);
    } else {
      const body: Record<string, unknown> = {
        model: modelStr,
        title: String(params.title || '').trim(),
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        size: `${params.width}x${params.height}`,
        num_frames: params.num_frames,
        fps: params.fps || 16,
        steps: params.steps,
        guidance: params.guide_scale,
        shift: params.shift || undefined,
        seed: params.seed ? parseInt(params.seed, 10) : null,
        metadata: { ...meta },
        priority: 'normal',
      };
      const adapters = buildVideoAdapters();
      if (adapters.length > 0) {
        body.adapters = adapters;
      }
      if (isLongVideoCreate()) {
        const p = currentModelConfig.value?.parameters;
        const extendSec = Number(p?.long_video_segment_extend_sec?.default ?? 7);
        const refSec = Number(p?.long_video_reference_duration_sec?.default ?? 3);
        const parsed = parseStoryboardPrompt(params.prompt);
        body.fps = params.fps || 24;
        body.long_video = {
          target_duration_sec: selectedDurationSec.value,
          initial_duration_sec: 8,
          segment_extend_sec: extendSec,
          reference_duration_sec: refSec,
          overlap_blend_frames: 4,
          opening_prompt: parsed.opening || params.prompt,
          segment_prompts: parsed.segmentPrompts.length > 0 ? parsed.segmentPrompts : undefined,
        };
      }
      submitRes = await api.gen.createVideoGeneration(body);
    }
    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) {
      throw new Error('missing task id in submit response');
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
      progressMessage: null,
      params: { model: modelStr, title: String(params.title || '').trim(), prompt: params.prompt },
    };
    api.gen.streamMediaTask(tid, {
      onLog: (logData: any) => {
        tasksStore.ingestTaskLog(tid, logData);
      },
      onTrace: (traceData: unknown) => {
        tasksStore.ingestTaskPipelineTrace(tid, traceData);
      },
      onResult: (resultData: any) => {
        const ids = (resultData?.asset_ids as string[] | undefined) || [];
        if (ids.length > 0) pendingCanvasAssetIds.value = ids;
      },
      onStatus: (statusData: any) => {
        if (currentTask.value) {
          currentTask.value.progress = statusData.progress ?? 0;
          currentTask.value.status = statusData.status;
        }
      },
      onDone: async (doneData: any) => {
        generating.value = false;
        tasksStore.unregisterPageOwnedStream(tid);
        if (doneData.status === 'completed') {
          tasksStore.appendTaskLog(tid, $tt('studio.genComplete'), 'success');
          const updated = await api.gen.getMediaTask(tid) as any;
          currentTask.value = updated;
          const pid = updated.result && updated.result.primary_asset_id;
          const ids = [...pendingCanvasAssetIds.value];
          const willAutoAdd = shouldAutoAddToCanvas() && ids.length > 0;
          pendingCanvasAssetIds.value = [];
          if (pid) {
            previewVideo.value = api.gallery.getImageUrl(`asset:${pid}`);
            previewVideoKey.value += 1;
            previewVideoDurationSec.value = 0;
          }
          if (!willAutoAdd) {
            toast.success($tt('studio.genComplete'));
          }
          await loadGallery(true);
          if (willAutoAdd) {
            await activateCanvasViewForResults(viewMode, syncCompositorOverlaysOnCanvasEnter);
            addAssetPathsToCanvas(ids.map((id) => `asset:${id}`), { placement: 'staging' });
          }
        } else if (doneData.status === 'failed') {
          const updated = await api.gen.getMediaTask(tid) as any;
          currentTask.value = updated;
          tasksStore.appendTaskLog(
            tid,
            $tt('studio.genFailed', { msg: updated.error || updated.error_message || '' }),
            'error'
          );
          toast.error($tt('studio.genFailed', { msg: updated.error || updated.error_message || '' }));
        }
      },
      onError: () => {
        tasksStore.unregisterPageOwnedStream(tid);
        tasksStore.appendTaskLog(tid, $tt('studio.connectionLost'), 'warning');
        toast.warning($tt('studio.connectionLost'));
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
        if (!currentTask.value) return;
        if (typeof progressData.progress === 'number') {
          currentTask.value.progress = progressData.progress;
        }
        const nextStep =
          progressData.step != null
            ? progressData.step
            : currentTask.value.step;
        const nextTotal =
          progressData.total != null
            ? progressData.total
            : currentTask.value.total;
        currentTask.value.step = nextStep;
        currentTask.value.total = nextTotal;
        if (progressData.message != null) {
          currentTask.value.progressMessage = progressData.message;
        }
      },
    });
  } catch (e: any) {
    generating.value = false;
    currentTask.value = null;
    toast.error($tt('studio.error', { msg: e.message || String(e) }));
  }
};

/* ------------------------------------------------------------------ */
/*  Recent / Reference Media                                           */
/* ------------------------------------------------------------------ */

const loadRecentStartImages = async () => {
  try {
    const images = await api.gallery.listImages(24, 0);
    recentStartImages.value = images
      .filter((v) => {
        if (v.metadata && v.metadata.asset_kind === 'video') {
          return false;
        }
        const ext = v.name?.split('.').pop()?.toLowerCase();
        return !['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext || '');
      })
      .slice(0, 8);
  } catch (e) {
    console.error('Failed to load recent start images:', e);
  }
};

const getVideoUrl = (video: GalleryItem) => {
  return api.gallery.getImageUrl(video.path);
};

// Video preview
const videoPreviewVisible = ref(false);
const selectedVideoIndex = ref(0);

function onGallerySelect(item: GalleryItem) {
  const idx = galleryItems.value.findIndex((it) => it.path === item.path);
  selectedVideoIndex.value = idx >= 0 ? idx : 0;
  videoPreviewVisible.value = true;
}

watch(videoWorkMode, () => {
  const cfg = currentModelConfig.value;
  const acts = cfg && cfg.actions ? cfg.actions : {};
  let ok = true;
  if (videoWorkMode.value === 'animate') {
    ok = videoSupportsAnimate(acts);
  } else if (videoWorkMode.value === 'upscale') {
    ok = videoSupportsUpscale(acts);
  } else {
    ok = videoSupportsCreate(acts);
  }
  if (!ok) {
    const first = videoRecommendedForMode.value[0] || videoVersionsForMode.value[0];
    if (first) {
      params.model = first.modelKey;
      params.version = first.versionKey;
      selectedModelVersion.value = first.modelKey + '|' + first.versionKey;
      loadModelDefaults();
      syncResolutionForModel(first.modelKey);
    }
  } else if (videoWorkMode.value !== 'upscale') {
    syncResolutionForModel();
  }
});

watch(
  () => currentModelConfig.value?.actions,
  () => {
    const values = videoWorkSegmentOptions.value.map((o) => o.value);
    if (values.length > 0 && !values.includes(videoWorkMode.value)) {
      videoWorkMode.value = values[0];
    }
  },
  { deep: true },
);

watch(modelFilterCommercialOnly, () => {
  if (
    reconcileVersionPickerSelection(videoModelPickerVersions.value, params, selectedModelVersion)
  ) {
    loadModelDefaults();
    syncResolutionForModel(String(params.model || ''));
  }
});

// Watch size and duration changes to update params
watch(selectedSize, (val) => {
  applySelectedSize(val);
  const mid = String(params.model || '');
  if (mid) setVideoSizeForModel(mid, val);
});

watch(sizeOptions, () => {
  syncResolutionForModel();
});

watch(selectedDurationSec, () => {
  syncNumFramesFromDuration();
});

watch(
  () => params.fps,
  () => {
    syncNumFramesFromDuration();
  },
);

watch(durationOptions, (opts) => {
  if (!opts.length) return;
  selectedDurationSec.value = snapDurationSecToOptions(selectedDurationSec.value, opts);
  if (!longVideoSupported() && selectedDurationSec.value >= 30) {
    selectedDurationSec.value = snapDurationSecToOptions(8, opts);
  }
  syncNumFramesFromDuration();
});

watch(selectedModelVersion, (val) => {
  onModelVersionChange(val);
});

onMounted(async () => {
  await loadModelRegistry();
  loadPresets();
  loadRecentStartImages();
  loadGallery(true);
  tasksStore.ensureQueuePoller();
  const promptDraft = consumePromptDraft(DQ_STORAGE.VIDEO_CREATE_PROMPT_DRAFT);
  if (promptDraft) {
    params.prompt = applyPromptDraft(params.prompt, promptDraft);
  }
});
</script>
