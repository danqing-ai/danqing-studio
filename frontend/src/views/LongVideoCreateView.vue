<template>
  <div class="lv-studio studio-create-page lv-studio--full">
    <header class="lv-studio__toolbar dq-glass--bar">
      <div class="lv-studio__toolbar-inner">
        <h1 class="lv-studio__toolbar-title">{{ $tt('video.longVideoPageTitle') }}</h1>
        <div class="lv-studio__toolbar-spacer" />
        <div class="lv-studio__toolbar-actions">
          <DqButton
            size="sm"
            type="text"
            :loading="savingProject"
            :disabled="savingProject || !project"
            @click="saveProject"
          >
            {{ $tt('video.longVideoSaveProject') }}
          </DqButton>
          <DqButton
            type="primary"
            size="sm"
            :loading="generating"
            :disabled="generating || mergeDisabled"
            @click="mergeLongVideo"
          >
            {{ $tt('video.longVideoMergeFilm') }}
          </DqButton>
        </div>
      </div>
    </header>

    <DqAlert
      v-if="!registrySupported"
      type="warning"
      :closable="false"
      class="lv-studio__alert"
      :title="$tt('video.longVideoModelUnsupported')"
    />

    <div class="lv-studio__body">
      <aside class="lv-studio__sidebar">
        <LongVideoProjectSidebar
          :projects="savedProjects"
          :active-project-id="project?.project_id"
          :loading="projectsLoading"
          @open="loadSavedProject"
          @new-project="createNewProject"
          @delete="deleteSavedProject"
        />
      </aside>

      <div class="lv-studio__editor">
        <LongVideoSettingsPanel
          inline
          :title="projectTitle"
          :keyframe-model="project?.keyframe_model ?? defaultKeyframeModel"
          :segment-model="project?.segment_video_model ?? defaultSegmentModel"
          :output-size="projectOutputSize"
          :output-size-options="segmentResolutionOptions"
          :keyframe-model-options="keyframeModelOptions"
          :segment-model-options="segmentModelOptions"
          :model-label="modelLabel"
          :overlap-frames="project?.overlap_frames ?? 4"
          @update:title="patchProjectField('title', $event)"
          @update:keyframe-model="onKeyframeModelChange"
          @update:segment-model="onSegmentModelChange"
          @update:output-size="patchProjectField('output_size', $event)"
          @update:overlap-frames="patchProjectField('overlap_frames', $event)"
        />

        <LongVideoBriefPanel
          :source-mode="project?.source_mode ?? 'brief'"
          :brief="project?.brief ?? ''"
          :chapter-text="project?.chapter_text ?? ''"
          :chapter-title="project?.chapter_title ?? ''"
          :chapter-analysis="project?.chapter_analysis ?? null"
          :target-duration-sec="project?.target_duration_sec ?? 60"
          :segment-duration-sec="project?.segment_duration_sec ?? 5"
          :expanding="isStoryboardExpanding"
          :analyzing="isChapterAnalyzing"
          @update:source-mode="onSourceModeChange"
          @update:brief="patchProjectField('brief', $event)"
          @update:chapter-text="patchProjectField('chapter_text', $event)"
          @update:chapter-title="patchProjectField('chapter_title', $event)"
          @update:chapter-analysis="patchProjectField('chapter_analysis', $event)"
          @update:target-duration-sec="patchProjectField('target_duration_sec', $event)"
          @analyze-chapter="onChapterAnalyze"
          @expand="onStoryboardExpand"
        />

        <LongVideoCastPanel
          :characters="project?.characters ?? []"
          :style-anchor="project?.style_anchor ?? ''"
          :character-anchor="project?.character_anchor ?? ''"
          @update:characters="onUpdateCharacters"
          @update:style-anchor="onUpdateStyleAnchor"
        />

          <LongVideoStoryboardRail
            :shots="shots"
            :selection="selection"
            :keyframe-generating-index="keyframeGeneratingIndex"
            :segment-generating-index="segmentGeneratingIndex"
            @select-node="onSelectNode"
            @select-edge="onSelectEdge"
            @add-keyframe="onAddKeyframe"
          />
      </div>
    </div>

    <LongVideoInspectorDrawer
      :open="Boolean(selection)"
      :shots="shots"
      :selection="selection"
      :segment-duration-options="segmentDurationOptions"
      :keyframe-generating="keyframeGenerating"
      :segment-generating="segmentGeneratingIndex != null"
      :visual-polishing="visualPolishing"
      :motion-polishing="motionPolishing"
      :output-size-label="outputSizeLabel"
      :segment-model-label="segmentModelLabel"
      :segment-compose-params="segmentComposeParams"
      :segment-param-schema="segmentParamSchema"
      :segment-compatible-loras="segmentCompatibleLoras"
      :segment-show-negative-prompt="segmentShowNegativePrompt"
      :segment-show-seed-field="segmentShowSeedField"
      :segment-show-lora="segmentShowLora"
      :default-chain-mode="project?.chain_mode ?? 'keyframe_only'"
      :compose-model="keyframeModelId"
      :compose-params="keyframeComposeParams"
      :compose-styles="composeStyles"
      :compose-show-negative-prompt="composeShowNegativePrompt"
      :compose-mode="composeMode"
      :compose-model-config="keyframeModelConfig"
      :compatible-loras="keyframeCompatibleLoras"
      :compatible-control-nets="keyframeCompatibleControlNets"
      :control-net-runtime-available="controlNetRuntimeAvailable"
      :reference-image="referenceImage"
      :control-image="keyframeControlImage"
      :inpaint-source-image="keyframeInpaintSourceImage"
      :inpaint-mask-image="keyframeInpaintMaskImage"
      :can-generate-keyframe="canGenerateKeyframe"
      :characters="project?.characters ?? []"
      @update:open="onInspectorOpenChange"
      @update-visual="onUpdateVisual"
      @update-cast-looks="onUpdateCastLooks"
      @update-motion="onUpdateMotion"
      @update-duration="onUpdateDuration"
      @update-chain-mode="onUpdateChainMode"
      @update-compose-mode="composeMode = $event"
      @reset-compose-defaults="resetKeyframeComposeDefaults()"
      @reset-segment-defaults="resetSegmentComposeDefaults()"
      @generate-keyframe="onGenerateKeyframe"
      @generate-segment="onGenerateSegment"
      @pick-keyframe-gallery="onPickKeyframeGallery"
      @clear-keyframe="onClearKeyframe"
      @clear-segment="onClearSegment"
      @insert-keyframe-before="onInsertKeyframeBefore"
      @insert-keyframe-after="onInsertKeyframeAfter"
      @remove-keyframe="onRemoveKeyframe"
      @select-node="onSelectNode"
      @polish-visual="onPolishVisual"
      @polish-motion="onPolishMotion"
      @pick-reference="openAssetPicker('reference')"
      @remove-reference="onRemoveReference"
      @pick-control="openAssetPicker('control')"
      @remove-control="onRemoveControl"
      @pick-inpaint-source="openAssetPicker('inpaint_source')"
      @remove-inpaint-source="onRemoveInpaintSource"
      @pick-inpaint-mask="openAssetPicker('inpaint_mask')"
      @remove-inpaint-mask="onRemoveInpaintMask"
    />

    <DqDialog v-model:open="showAssetPicker" :title="$tt('assetPicker.dialogTitle')" width="70%">
      <AssetPicker media="image" @pick="onAssetPickerPick" />
    </DqDialog>

    <LongVideoOutputStrip
      :asset-id="project?.final_asset_id"
      @open-gallery="openFinalInGallery"
    />

    <LongVideoGalleryDrawer
      v-model:open="showKeyframePicker"
      :recent-gallery="recentImages"
      @pick="onKeyframeAssetPick"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch, inject, type Ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast, confirm } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import type { LongVideoChainMode, LongVideoCharacter, LongVideoChapterAnalysis, LongVideoProjectState, LongVideoProjectSummary, LongVideoShotCastLook, LongVideoShotState, SystemInfo } from '@/types';
import LongVideoBriefPanel from '@/components/long-video/LongVideoBriefPanel.vue';
import LongVideoCastPanel from '@/components/long-video/LongVideoCastPanel.vue';
import LongVideoSettingsPanel from '@/components/long-video/LongVideoSettingsPanel.vue';
import LongVideoStoryboardRail from '@/components/long-video/LongVideoStoryboardRail.vue';
import LongVideoInspectorDrawer from '@/components/long-video/LongVideoInspectorDrawer.vue';
import LongVideoGalleryDrawer from '@/components/long-video/LongVideoGalleryDrawer.vue';
import LongVideoOutputStrip from '@/components/long-video/LongVideoOutputStrip.vue';
import LongVideoProjectSidebar from '@/components/long-video/LongVideoProjectSidebar.vue';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import { useLongVideoProject } from '@/composables/useLongVideoProject';
import { useLongVideoRegistry } from '@/composables/useLongVideoRegistry';
import {
  useLongVideoKeyframeCompose,
  type KeyframeComposeParams,
} from '@/composables/useLongVideoKeyframeCompose';
import { useLongVideoSegmentCompose } from '@/composables/useLongVideoSegmentCompose';
import { useComposerLlm } from '@/composables/useComposerLlm';
import { useStudioGallery } from '@/composables/useStudioGallery';
import {
  appendZImageEnhancementFields,
  isControlNetHostRuntimeAvailable,
  resolveControlAssetId,
} from '@/composables/useStructuralGuide';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import router from '@/router';
import {
  formatResolutionOptionLabel,
  normalizeParamsDef,
  parseSizeValue,
  strengthDefaultFromRegistry,
  strengthToSourceFidelity,
} from '@/utils/registryParamSchema';
import {
  applyStoryboardShots,
  createEmptyShot,
  defaultLongVideoProject,
  effectiveShotChainMode,
  insertKeyframeAfter,
  insertKeyframeBefore,
  invalidateSegmentAsset,
  keyframeGenerationPrompt,
  shotSceneText,
  syncRosterToCharacterAnchor,
  hydrateCharacterRoster,
  MIN_LONG_VIDEO_KEYFRAMES,
  nextShotId,
  numFramesForDurationSec,
  projectStateForServer,
  removeKeyframeAt,
  removeKeyframeNeedsConfirm,
  shotDurationSec,
} from '@/utils/longVideoProject';
import '@/styles/long-video.css';

const tasksStore = useTasksStore();
const systemInfo = inject<Ref<SystemInfo>>('systemInfo', ref({} as SystemInfo));
const longVideoProject = useLongVideoProject();
const {
  profile,
  supported: registrySupported,
  modelLabel,
  modelParameters,
  resolutionOptionsForModel,
  pickOutputSizeForModel,
  loadRegistry,
} = useLongVideoRegistry();
const { enhance, isEnhancing, storyboardLongVideo, isStoryboardExpanding, analyzeLongVideoChapter, isChapterAnalyzing } = useComposerLlm();
const { locale: uiLocale } = useI18n();
const { galleryItems, loadGallery } = useStudioGallery('image');

const generating = ref(false);
const savingProject = ref(false);
const keyframeGeneratingIndex = ref<number | null>(null);
const segmentGeneratingIndex = ref<number | null>(null);
const visualPolishing = ref(false);
const motionPolishing = ref(false);
const showKeyframePicker = ref(false);
const keyframePickIndex = ref<number | null>(null);
const showAssetPicker = ref(false);
const assetPickerMode = ref<'reference' | 'control' | 'inpaint_source' | 'inpaint_mask'>('reference');
const composeMode = ref('text2img');
const referenceImage = ref<{ previewUrl: string; path: string; assetId?: string } | null>(null);
const savedProjects = ref<LongVideoProjectSummary[]>([]);
const projectsLoading = ref(false);

const project = computed(() => longVideoProject.project.value);
const shots = computed(() => project.value?.shots ?? []);
const selection = computed(() => project.value?.selection ?? null);

const defaultKeyframeModel = computed(() => profile.value?.keyframe_models[0] ?? 'z-image-turbo');
const defaultSegmentModel = computed(() => profile.value?.segment_models[0] ?? 'wan-2.2-i2v-14b');
const keyframeModelOptions = computed(() => profile.value?.keyframe_models ?? ['z-image-turbo', 'z-image']);
const segmentModelOptions = computed(() => profile.value?.segment_models ?? ['wan-2.2-i2v-14b']);
const segmentDurationOptions = [3, 5, 8];

const keyframeGenerating = computed(() => keyframeGeneratingIndex.value != null);

const composeModelId = ref('z-image-turbo');
const composeOutputSize = ref('1280x704');
const keyframeCompose = useLongVideoKeyframeCompose(composeModelId, composeOutputSize);
const {
  params: keyframeComposeParams,
  compatibleLoras: keyframeCompatibleLoras,
  compatibleControlNets: keyframeCompatibleControlNets,
  controlImage: keyframeControlImage,
  inpaintSourceImage: keyframeInpaintSourceImage,
  inpaintMaskImage: keyframeInpaintMaskImage,
  currentModelConfig: keyframeModelConfig,
  filteredPresets: composeStyles,
  showNegativePrompt: composeShowNegativePrompt,
  resetToDefaults: resetKeyframeComposeDefaults,
} = keyframeCompose;

const keyframeModelId = computed(() => project.value?.keyframe_model ?? defaultKeyframeModel.value);

const controlNetRuntimeAvailable = computed(() =>
  isControlNetHostRuntimeAvailable(
    systemInfo.value?.controlnet_runtime_available,
    [],
    systemInfo.value,
  ),
);

const selectedNodeIndex = computed(() =>
  selection.value?.kind === 'node' ? selection.value.index : null,
);

const canGenerateKeyframe = computed(() => {
  const idx = selectedNodeIndex.value;
  if (idx == null) return false;
  const shot = shots.value[idx];
  if (!shot?.visual_prompt.trim()) return false;
  if (composeMode.value === 'img2img' && !referenceImage.value) return false;
  return true;
});

function bindReferenceFromShot(shot: LongVideoShotState | undefined) {
  const refId = shot?.reference_asset_id;
  if (refId) {
    referenceImage.value = {
      path: `asset:${refId}`,
      previewUrl: api.gallery.getImageUrl(`asset:${refId}`),
      assetId: refId,
    };
    composeMode.value = 'img2img';
    return;
  }
  referenceImage.value = null;
  if (composeMode.value === 'img2img') composeMode.value = 'text2img';
}

watch(
  () => (selection.value?.kind === 'node' ? shots.value[selection.value.index] : null),
  (shot) => bindReferenceFromShot(shot ?? undefined),
  { immediate: true },
);

function openAssetPicker(mode: 'reference' | 'control' | 'inpaint_source' | 'inpaint_mask') {
  assetPickerMode.value = mode;
  showAssetPicker.value = true;
}

function assetIdFromPickPath(path: string): string | undefined {
  return path.startsWith('asset:') ? path.slice('asset:'.length) : undefined;
}

function resolveInpaintAssetId(ref: { path: string; assetId?: string } | null): string {
  if (!ref) return '';
  if (ref.assetId) return ref.assetId;
  if (ref.path.startsWith('asset:')) return ref.path.slice('asset:'.length);
  return '';
}

function patchShotReference(index: number, assetId: string | undefined) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) =>
    i === index ? { ...s, reference_asset_id: assetId } : s,
  );
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onRemoveReference() {
  referenceImage.value = null;
  composeMode.value = 'text2img';
  const idx = selectedNodeIndex.value;
  if (idx != null) patchShotReference(idx, undefined);
}

function onAssetPickerPick(payload: { path: string; previewUrl: string }) {
  const { path, previewUrl } = payload;
  const assetId = assetIdFromPickPath(path);
  showAssetPicker.value = false;

  if (assetPickerMode.value === 'control') {
    keyframeControlImage.value = { path, previewUrl };
    return;
  }
  if (assetPickerMode.value === 'inpaint_source') {
    if (!path.startsWith('asset:')) {
      toast.warning($tt('canvas.controlImageAssetRequired'));
      return;
    }
    keyframeInpaintSourceImage.value = { path, previewUrl };
    return;
  }
  if (assetPickerMode.value === 'inpaint_mask') {
    if (!path.startsWith('asset:')) {
      toast.warning($tt('canvas.controlImageAssetRequired'));
      return;
    }
    keyframeInpaintMaskImage.value = { path, previewUrl };
    return;
  }

  referenceImage.value = { path, previewUrl, assetId };
  composeMode.value = 'img2img';
  const idx = selectedNodeIndex.value;
  if (idx != null && assetId) patchShotReference(idx, assetId);
}

const projectTitle = computed(() => project.value?.title ?? '');

function effectiveTargetDurationSec(lv: LongVideoProjectState): number {
  if (!lv.shots.length) return lv.target_duration_sec ?? 60;
  const total = lv.shots.reduce((sum, s) => sum + shotDurationSec(s), 0);
  return Math.max(lv.target_duration_sec ?? 60, total);
}

const segmentModelId = computed(() => project.value?.segment_video_model ?? defaultSegmentModel.value);

const segmentModelLabel = computed(() => modelLabel(segmentModelId.value));

const segmentCompose = useLongVideoSegmentCompose(segmentModelId);
const {
  params: segmentComposeParams,
  compatibleLoras: segmentCompatibleLoras,
  paramSchema: segmentParamSchema,
  showNegativePrompt: segmentShowNegativePrompt,
  showSeedField: segmentShowSeedField,
  showLora: segmentShowLora,
  resetToDefaults: resetSegmentComposeDefaults,
} = segmentCompose;

const segmentResolutionOptions = computed(() => resolutionOptionsForModel(segmentModelId.value));

const projectOutputSize = computed(() => {
  const saved = project.value?.output_size;
  const picked = pickOutputSizeForModel(segmentModelId.value, saved);
  return picked ?? segmentResolutionOptions.value[0]?.value ?? '1280x704';
});

const outputSizePixels = computed(() => {
  const parsed = parseSizeValue(projectOutputSize.value);
  return parsed ?? { width: 1280, height: 704 };
});

const outputSizeLabel = computed(() => {
  const opt = segmentResolutionOptions.value.find((o) => o.value === projectOutputSize.value);
  if (opt) return formatResolutionOptionLabel(opt);
  return projectOutputSize.value.replace('x', '×');
});

function scalarDefault(modelId: string, key: string, fallback: number): number {
  const spec = normalizeParamsDef(modelParameters(modelId))[key];
  const def = spec?.default;
  return typeof def === 'number' ? def : fallback;
}

function syncOutputSizeForSegmentModel(modelId: string) {
  const next = pickOutputSizeForModel(modelId, project.value?.output_size);
  if (next && next !== project.value?.output_size) {
    patchProjectField('output_size', next);
  }
}

function onSegmentModelChange(modelId: string) {
  patchProjectField('segment_video_model', modelId);
  syncOutputSizeForSegmentModel(modelId);
}

function onKeyframeModelChange(modelId: string) {
  patchProjectField('keyframe_model', modelId);
}

watch(
  keyframeModelId,
  (id) => {
    composeModelId.value = id;
  },
  { immediate: true },
);

watch(
  projectOutputSize,
  (size) => {
    composeOutputSize.value = size;
  },
  { immediate: true },
);

function onRemoveControl() {
  keyframeControlImage.value = null;
}

function onRemoveInpaintSource() {
  keyframeInpaintSourceImage.value = null;
}

function onRemoveInpaintMask() {
  keyframeInpaintMaskImage.value = null;
}

function onInspectorOpenChange(open: boolean) {
  if (!open) clearSelection();
}

const mergeDisabled = computed(() => {
  if (!shots.value.length || generating.value || segmentGeneratingIndex.value != null) return true;
  return !shots.value.every((s) => s.status === 'segment_ready' && s.segment_asset_id);
});

const recentImages = computed(() => galleryItems.value.slice(0, 40) as Array<Record<string, unknown>>);

function patchProjectField<K extends keyof LongVideoProjectState>(key: K, value: LongVideoProjectState[K]) {
  longVideoProject.patchProject({ [key]: value } as Partial<LongVideoProjectState>);
}

function storyboardShotsFromResponse(
  apiShots: Array<{
    id?: string;
    order?: number;
    visual_prompt?: string;
    motion_prompt?: string;
    scene_prompt?: string;
    cast_looks?: LongVideoShotCastLook[];
  }>,
  segmentDurationSec: number,
): LongVideoShotState[] {
  return apiShots.map((s, i) => ({
    id: s.id || `shot_${String(i).padStart(2, '0')}`,
    order: i,
    visual_prompt: s.visual_prompt || '',
    motion_prompt: s.motion_prompt || '',
    scene_prompt: s.scene_prompt || '',
    cast_looks: s.cast_looks ?? [],
    duration_sec: segmentDurationSec,
    status: 'draft' as const,
  }));
}

async function onChapterAnalyze() {
  const lv = project.value;
  if (!lv) return;
  const text = (lv.chapter_text || '').trim();
  if (!text) {
    toast.warning($tt('video.longVideoChapterNeedText'));
    return;
  }

  const result = await analyzeLongVideoChapter(
    {
      chapter_text: text,
      chapter_title: (lv.chapter_title || '').trim(),
      locale: uiLocale.value,
    },
    { quietSuccess: true },
  );
  if (!result?.scene_beats?.length) return;

  const scene_beats = result.scene_beats.map((s) => ({
    order: s.order,
    title: s.title || '',
    beat: s.beat || '',
  }));
  const segmentDurationSec = lv.segment_duration_sec ?? 5;
  const targetDuration = scene_beats.length * segmentDurationSec;
  const rosterPatch = hydrateCharacterRoster(
    {
      characters: (result.characters ?? []) as LongVideoCharacter[],
      character_anchor: result.character_anchor || '',
      style_anchor: result.style_anchor || '',
    },
    uiLocale.value.startsWith('zh') ? 'zh' : 'en',
  );
  const analysis: LongVideoChapterAnalysis = {
    synopsis: result.synopsis || '',
    scene_beats,
    character_anchor: rosterPatch.character_anchor,
    style_anchor: rosterPatch.style_anchor,
    characters: rosterPatch.characters,
  };
  longVideoProject.patchProject({
    chapter_analysis: analysis,
    target_duration_sec: targetDuration,
    character_anchor: rosterPatch.character_anchor,
    characters: rosterPatch.characters,
    style_anchor: rosterPatch.style_anchor,
    brief: result.synopsis || lv.brief,
  });
  toast.info($tt('video.longVideoChapterAnalyzeReady', { n: scene_beats.length }));
}

function onSourceModeChange(mode: 'brief' | 'chapter') {
  patchProjectField('source_mode', mode);
}

async function onStoryboardExpand() {
  const lv = project.value;
  if (!lv) return;
  const sourceMode = lv.source_mode ?? 'brief';
  const segmentDurationSec = lv.segment_duration_sec ?? 5;

  if (sourceMode === 'chapter') {
    const analysis = lv.chapter_analysis;
    const beats = (analysis?.scene_beats ?? []).map((s) => s.beat.trim()).filter(Boolean);
    if (beats.length < 2) {
      toast.warning($tt('video.longVideoChapterNeedAnalysis'));
      return;
    }

    const hasExistingWork = lv.shots.some(
      (s) => s.visual_prompt?.trim() || s.motion_prompt?.trim() || s.keyframe_asset_id || s.segment_asset_id,
    );
    if (hasExistingWork) {
      try {
        await confirm(
          $tt('video.longVideoStoryboardOverwriteConfirm'),
          $tt('video.storyboardExpand'),
          { type: 'warning' },
        );
      } catch {
        return;
      }
    }

    const targetDuration = beats.length * segmentDurationSec;
    const result = await storyboardLongVideo(
      {
        prompt: analysis?.synopsis || beats[0],
        target_duration_sec: targetDuration,
        segment_duration_sec: segmentDurationSec,
        use_shot_plan: true,
        locale: uiLocale.value,
        source_mode: 'chapter',
        scene_beats: beats,
        prebuilt_character_anchor: analysis?.character_anchor || lv.character_anchor || '',
        prebuilt_style_anchor: analysis?.style_anchor || lv.style_anchor || '',
      },
      { quietSuccess: true },
    );
    if (!result?.shots?.length) return;

    const shotsNext = storyboardShotsFromResponse(result.shots, segmentDurationSec);
    const rosterPatch = hydrateCharacterRoster(
      {
        characters: (result.characters ?? analysis?.characters ?? []) as LongVideoCharacter[],
        character_anchor: result.character_anchor || analysis?.character_anchor || '',
        style_anchor: result.style_anchor || analysis?.style_anchor || '',
      },
      uiLocale.value.startsWith('zh') ? 'zh' : 'en',
    );
    longVideoProject.setProject(
      applyStoryboardShots(
        {
          ...lv,
          target_duration_sec: targetDuration,
          character_anchor: rosterPatch.character_anchor,
          characters: rosterPatch.characters,
          style_anchor: rosterPatch.style_anchor,
        },
        shotsNext,
      ),
    );
    toast.info($tt('video.longVideoStoryboardReady'));
    return;
  }

  const brief = (lv.brief || '').trim();
  if (!brief) {
    toast.warning($tt('video.longVideoNeedBrief'));
    return;
  }

  const hasExistingWork = lv.shots.some(
    (s) => s.visual_prompt?.trim() || s.motion_prompt?.trim() || s.keyframe_asset_id || s.segment_asset_id,
  );
  if (hasExistingWork) {
    try {
      await confirm(
        $tt('video.longVideoStoryboardOverwriteConfirm'),
        $tt('video.storyboardExpand'),
        { type: 'warning' },
      );
    } catch {
      return;
    }
  }

  const result = await storyboardLongVideo(
    {
      prompt: brief,
      target_duration_sec: lv.target_duration_sec ?? 60,
      segment_duration_sec: segmentDurationSec,
      use_shot_plan: true,
      locale: uiLocale.value,
    },
    { quietSuccess: true },
  );
  if (!result?.shots?.length) return;

  const shotsNext = storyboardShotsFromResponse(result.shots, segmentDurationSec);
  const rosterPatch = hydrateCharacterRoster(
    {
      characters: (result.characters ?? []) as LongVideoCharacter[],
      character_anchor: result.character_anchor || '',
      style_anchor: result.style_anchor || '',
    },
    uiLocale.value.startsWith('zh') ? 'zh' : 'en',
  );
  longVideoProject.setProject(
    applyStoryboardShots(
      {
        ...lv,
        character_anchor: rosterPatch.character_anchor,
        characters: rosterPatch.characters,
        style_anchor: rosterPatch.style_anchor,
      },
      shotsNext,
    ),
  );
  toast.info($tt('video.longVideoStoryboardReady'));
}

function clearSelection() {
  longVideoProject.setSelection(null);
}

function onSelectNode(index: number) {
  longVideoProject.setSelection({ kind: 'node', index });
}

function onSelectEdge(index: number) {
  longVideoProject.setSelection({ kind: 'edge', index });
}

function onUpdateCharacters(characters: LongVideoCharacter[]) {
  const lv = project.value;
  if (!lv) return;
  longVideoProject.setProject({
    ...lv,
    characters,
    character_anchor: syncRosterToCharacterAnchor(characters, lv.style_anchor ?? ''),
  });
}

function onUpdateStyleAnchor(styleAnchor: string) {
  const lv = project.value;
  if (!lv) return;
  longVideoProject.setProject({
    ...lv,
    style_anchor: styleAnchor,
    character_anchor: syncRosterToCharacterAnchor(lv.characters ?? [], styleAnchor),
  });
}

function onUpdateVisual(index: number, value: string) {
  const lv = project.value;
  if (!lv) return;
  const scene = value.trim();
  const shotsNext = lv.shots.map((s, i) =>
    i === index ? { ...s, visual_prompt: scene, scene_prompt: scene } : s,
  );
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onUpdateCastLooks(index: number, castLooks: LongVideoShotCastLook[]) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) => (i === index ? { ...s, cast_looks: castLooks } : s));
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onUpdateMotion(index: number, value: string) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) => (i === index ? { ...s, motion_prompt: value } : s));
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onUpdateDuration(index: number, value: number) {
  const lv = project.value;
  if (!lv) return;
  const sec = Math.max(1, Number(value) || 5);
  const shotsNext = lv.shots.map((s, i) => (i === index ? { ...s, duration_sec: sec } : s));
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onUpdateChainMode(index: number, value: LongVideoChainMode) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) => {
    if (i !== index) return s;
    const stored = value === lv.chain_mode ? undefined : value;
    if (stored === s.chain_mode) return s;
    return invalidateSegmentAsset({ ...s, chain_mode: stored });
  });
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

async function onPolishMotion(index: number) {
  const lv = project.value;
  if (!lv) return;
  const shot = lv.shots[index];
  const nextShot = lv.shots[index + 1];
  if (!shot) return;

  const left = shot.visual_prompt.trim();
  const right = nextShot?.visual_prompt.trim() ?? '';
  const motion = shot.motion_prompt.trim();
  const seed = motion || [left, right].filter(Boolean).join('\n\n');
  if (!seed) {
    toast.warning($tt('video.longVideoPolishNeedContext'));
    return;
  }

  const contextParts = [
    left ? `${$tt('video.longVideoPolishFrom')}: ${left}` : '',
    right ? `${$tt('video.longVideoPolishTo')}: ${right}` : '',
    motion ? `${$tt('video.longVideoMotionPrompt')}: ${motion}` : '',
  ].filter(Boolean);

  const prompt = contextParts.length > 1 ? contextParts.join('\n') : seed;
  motionPolishing.value = true;
  try {
    const enhanced = await enhance(prompt, undefined, 'video_create', lv.segment_video_model);
    if (enhanced) onUpdateMotion(index, enhanced);
  } finally {
    motionPolishing.value = false;
  }
}

async function onPolishVisual(index: number) {
  const lv = project.value;
  if (!lv) return;
  const shot = lv.shots[index];
  const prompt = shot?.visual_prompt.trim();
  if (!prompt) return;

  visualPolishing.value = true;
  try {
    const enhanced = await enhance(prompt, undefined, 'image_create', lv.keyframe_model);
    if (enhanced) onUpdateVisual(index, enhanced);
  } finally {
    visualPolishing.value = false;
  }
}

function openFinalInGallery() {
  const id = project.value?.final_asset_id;
  if (!id) return;
  void router.push({ name: 'video_create', query: { asset: id } });
}

function onAddKeyframe() {
  const lv =
    project.value ||
    defaultLongVideoProject({
      keyframe_model: defaultKeyframeModel.value,
      segment_video_model: defaultSegmentModel.value,
      title: projectTitle.value,
    });
  const id = nextShotId(lv.shots);
  const shotsNext = [...lv.shots, createEmptyShot(lv.shots.length, id)];
  longVideoProject.setProject({
    ...lv,
    shots: shotsNext,
    selection: { kind: 'node', index: shotsNext.length - 1 },
  });
}

function onInsertKeyframeBefore(index: number) {
  const lv = project.value;
  if (!lv) return;
  const id = nextShotId(lv.shots);
  const { shots: shotsNext, newIndex } = insertKeyframeBefore(lv.shots, index, id);
  longVideoProject.setProject({
    ...lv,
    shots: shotsNext,
    selection: { kind: 'node', index: newIndex },
  });
}

function onInsertKeyframeAfter(index: number) {
  const lv = project.value;
  if (!lv) return;
  const id = nextShotId(lv.shots);
  const { shots: shotsNext, newIndex } = insertKeyframeAfter(lv.shots, index, id);
  longVideoProject.setProject({
    ...lv,
    shots: shotsNext,
    selection: { kind: 'node', index: newIndex },
  });
}

async function onRemoveKeyframe(index: number) {
  const lv = project.value;
  if (!lv) return;
  if (lv.shots.length <= MIN_LONG_VIDEO_KEYFRAMES) {
    toast.warning($tt('video.longVideoRemoveKeyframeMin'));
    return;
  }

  const n = index + 1;
  const isFirst = index === 0;
  const isLast = index === lv.shots.length - 1;

  if (removeKeyframeNeedsConfirm(lv.shots, index)) {
    let body: string;
    if (isFirst) {
      body = $tt('video.longVideoRemoveKeyframeConfirmFirst', { n });
    } else if (isLast) {
      body = $tt('video.longVideoRemoveKeyframeConfirmLast', { n });
    } else {
      body = $tt('video.longVideoRemoveKeyframeConfirmMiddle', {
        n,
        from: index,
        to: index + 2,
      });
    }
    try {
      await confirm(body, $tt('video.longVideoRemoveKeyframeConfirmTitle'), { type: 'warning' });
    } catch {
      return;
    }
  }

  const result = removeKeyframeAt(lv.shots, index);
  if (!result) return;
  longVideoProject.setProject({
    ...lv,
    shots: result.shots,
    selection: result.selection,
  });
}

function onClearKeyframe(index: number) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) =>
    i === index ? { ...s, keyframe_asset_id: undefined, status: 'draft' as const } : s,
  );
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onClearSegment(index: number) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) => {
    if (i !== index) return s;
    const next = { ...s, segment_asset_id: undefined };
    if (next.status === 'segment_ready') {
      next.status = next.keyframe_asset_id ? ('keyframe_ready' as const) : ('draft' as const);
    }
    return next;
  });
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onPickKeyframeGallery(index: number) {
  keyframePickIndex.value = index;
  showKeyframePicker.value = true;
}

function onKeyframeAssetPick(payload: { path: string }) {
  const index = keyframePickIndex.value;
  showKeyframePicker.value = false;
  keyframePickIndex.value = null;
  if (index == null) return;
  const assetId = assetIdFromGalleryPath(payload.path);
  if (!assetId) return;
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) =>
    i === index ? { ...s, keyframe_asset_id: assetId, status: 'keyframe_ready' as const } : s,
  );
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function waitForImageTask(tid: string): Promise<{ assetId: string | null }> {
  return new Promise((resolve, reject) => {
    let assetId: string | null = null;
    api.gen.streamMediaTask(tid, {
      onResult: (resultData: unknown) => {
        const data = resultData as { asset_ids?: string[] };
        const ids = data?.asset_ids || [];
        if (ids.length > 0) assetId = ids[0];
      },
      onDone: (doneData: unknown) => {
        tasksStore.unregisterPageOwnedStream(tid);
        const data = doneData as { status: string };
        if (data.status === 'completed') resolve({ assetId });
        else reject(new Error('keyframe generation failed'));
      },
      onError: () => {
        tasksStore.unregisterPageOwnedStream(tid);
        reject(new Error('connection lost'));
      },
    });
  });
}

async function onGenerateKeyframe(index: number) {
  const lv = project.value;
  if (!lv) return;
  const shot = lv.shots[index];
  if (!shot?.visual_prompt.trim()) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  if (composeMode.value === 'img2img' && !referenceImage.value) {
    toast.warning($tt('create.refImageNeeded'));
    return;
  }

  const p = keyframeComposeParams;
  const modelStr = keyframeModelId.value;
  const adapters: Array<{ id: string; weight: number }> = [];
  if (p.lora) adapters.push({ id: String(p.lora), weight: Number(p.lora_scale) || 0.8 });

  let control_asset_id: string | null = null;
  if (p.controlnet) {
    try {
      control_asset_id = resolveControlAssetId(keyframeControlImage.value, {
        assetRequired: $tt('canvas.controlImageAssetRequired'),
        required: $tt('canvas.controlImageRequired'),
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      toast.warning(msg);
      return;
    }
  }

  const hasRef = referenceImage.value != null && composeMode.value === 'img2img';
  let source_asset_id: string | null = null;
  if (hasRef && referenceImage.value) {
    const rp = referenceImage.value.path;
    if (typeof rp === 'string' && rp.startsWith('asset:')) {
      source_asset_id = rp.slice('asset:'.length);
    } else {
      try {
        const blob = await api.gen.urlToBlob(referenceImage.value.previewUrl);
        const up = await api.gen.uploadAsset(
          new File([blob], 'ref.png', { type: blob.type || 'image/png' }),
        );
        source_asset_id = (up as { id: string }).id;
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        toast.error($tt('studio.error', { msg }));
        return;
      }
    }
  }

  const meta: Record<string, unknown> = {
    long_video_shot_id: shot.id,
    long_video_phase: 'keyframe',
  };
  if (p.scheduler) meta.scheduler = p.scheduler;

  const inpSrc = resolveInpaintAssetId(keyframeInpaintSourceImage.value);
  const inpMsk = resolveInpaintAssetId(keyframeInpaintMaskImage.value);
  const enhCommon = {
    controlnet: String(p.controlnet || ''),
    controlAssetId: control_asset_id,
    controlnetStrength: Number(p.controlnet_strength) || 0.8,
    inpaintSourceId: inpSrc,
    inpaintMaskId: inpMsk,
    lemicaMode: String(p.lemica_mode || 'none'),
    latentRefineScale: Number(p.latent_refine_scale),
    latentRefineDenoise: Number(p.latent_refine_denoise),
  };

  const seedNum = p.seed ? parseInt(String(p.seed), 10) : null;
  const { width, height } = outputSizePixels.value;
  const scene = shotSceneText(shot);
  const t2iPrompt = keyframeGenerationPrompt(scene, {
    characterAnchor: lv.character_anchor ?? '',
    characters: lv.characters,
    styleAnchor: lv.style_anchor,
    castLooks: shot.cast_looks,
  });

  keyframeGeneratingIndex.value = index;
  try {
    let submitRes: unknown;
    if (hasRef && source_asset_id) {
      const editBody: Record<string, unknown> = {
        model: modelStr,
        operation: 'rewrite',
        source_asset_id,
        prompt: t2iPrompt,
        negative_prompt: p.negative_prompt || '',
        n: 1,
        steps: p.steps,
        guidance: p.guidance,
        seed: seedNum,
        adapters,
        source_fidelity: strengthToSourceFidelity(
          p.strength,
          strengthDefaultFromRegistry(
            keyframeModelConfig.value?.parameters as Record<string, unknown> | undefined,
          ),
        ),
        metadata: meta,
        priority: 'normal',
      };
      const editEnh = appendZImageEnhancementFields(editBody, enhCommon);
      if (!editEnh.ok) throw new Error($tt('create.inpaintPairRequired'));
      submitRes = await api.gen.createImageEdit(editBody);
    } else {
      const genBody: Record<string, unknown> = {
        model: modelStr,
        prompt: t2iPrompt,
        negative_prompt: p.negative_prompt || '',
        size: `${width}x${height}`,
        n: 1,
        steps: p.steps,
        guidance: p.guidance,
        seed: seedNum,
        adapters,
        metadata: meta,
        priority: 'normal',
      };
      const genEnh = appendZImageEnhancementFields(genBody, enhCommon);
      if (!genEnh.ok) throw new Error($tt('create.inpaintPairRequired'));
      submitRes = await api.gen.createImageGeneration(genBody);
    }

    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) throw new Error('no task id');
    tasksStore.registerPageOwnedStream(tid);
    const { assetId } = await waitForImageTask(tid);
    if (!assetId) throw new Error('no asset');
    const latest = project.value ?? lv;
    const shotsNext = latest.shots.map((s, i) =>
      i === index ? { ...s, keyframe_asset_id: assetId, status: 'keyframe_ready' as const } : s,
    );
    longVideoProject.setProject({ ...latest, shots: shotsNext });
    await loadGallery(true);
    toast.success($tt('video.longVideoKeyframeDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.genFailed', { msg }));
  } finally {
    if (keyframeGeneratingIndex.value === index) {
      keyframeGeneratingIndex.value = null;
    }
  }
}

async function onGenerateSegment(index: number) {
  const lv = project.value;
  if (!lv) return;
  const shot = lv.shots[index];
  if (!shot?.keyframe_asset_id) {
    toast.warning($tt('video.longVideoNeedKeyframeForSegment'));
    return;
  }
  const motion = (shot.motion_prompt || shot.visual_prompt || '').trim();
  if (!motion) {
    toast.warning($tt('video.longVideoNeedMotionForSegment'));
    return;
  }

  const chainMode = effectiveShotChainMode(shot, lv.chain_mode);
  const prevSegId = index > 0 ? lv.shots[index - 1]?.segment_asset_id : undefined;
  if (chainMode === 'last_frame' && index > 0 && !prevSegId) {
    toast.warning($tt('video.longVideoSegmentChainNeedPrevSegment'));
    return;
  }

  segmentGeneratingIndex.value = index;
  try {
    const segModel = lv.segment_video_model;
    const p = segmentComposeParams;
    const fps = p.fps || scalarDefault(segModel, 'fps', 16);
    const nfSchema = normalizeParamsDef(modelParameters(segModel)).num_frames as
      | { min?: number; max?: number; step?: number }
      | undefined;
    const numFrames = numFramesForDurationSec(shotDurationSec(shot), fps, nfSchema);
    const { width, height } = outputSizePixels.value;
    const body: Record<string, unknown> = {
      model: segModel,
      operation: 'animate',
      source_asset_id: shot.keyframe_asset_id,
      prompt: motion,
      negative_prompt: p.negative_prompt || '',
      size: `${width}x${height}`,
      num_frames: numFrames,
      fps,
      steps: p.steps || scalarDefault(segModel, 'steps', 40),
      guidance: p.guide_scale ?? scalarDefault(segModel, 'guide_scale', 3.0),
      shift: p.shift || scalarDefault(segModel, 'shift', 12.0) || undefined,
      seed: shot.seed ?? (p.seed ? parseInt(p.seed, 10) : null),
      metadata: {
        long_video_shot_id: shot.id,
        long_video_phase: 'segment',
        long_video_chain_mode: chainMode,
        ...(chainMode === 'last_frame' && prevSegId
          ? { long_video_prev_segment_asset_id: prevSegId }
          : {}),
      },
    };
    const adapters = buildSegmentAdapters();
    if (adapters.length > 0) {
      body.adapters = adapters;
    }
    const submitRes = await api.gen.createVideoEdit(body);
    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) throw new Error('no task id');
    tasksStore.registerPageOwnedStream(tid);
    const { assetId } = await waitForImageTask(tid);
    if (!assetId) throw new Error('no asset');
    const latest = project.value ?? lv;
    const shotsNext = latest.shots.map((s, i) =>
      i === index ? { ...s, segment_asset_id: assetId, status: 'segment_ready' as const, error: undefined } : s,
    );
    longVideoProject.setProject({ ...latest, shots: shotsNext });
    toast.success($tt('video.longVideoSegmentDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.genFailed', { msg }));
  } finally {
    if (segmentGeneratingIndex.value === index) {
      segmentGeneratingIndex.value = null;
    }
  }
}

async function loadProjectList() {
  projectsLoading.value = true;
  try {
    savedProjects.value = await api.longVideo.listProjects(100);
  } catch {
    savedProjects.value = [];
  } finally {
    projectsLoading.value = false;
  }
}

function applyServerProject(
  detail: { id: string; title: string; state: Partial<LongVideoProjectState> },
  opts?: { silent?: boolean },
) {
  longVideoProject.setProject({
    ...hydrateCharacterRoster(
      defaultLongVideoProject(detail.state),
      uiLocale.value.startsWith('zh') ? 'zh' : 'en',
    ),
    project_id: detail.id,
    title: detail.title || detail.state.title,
    selection: null,
  });
  syncOutputSizeForSegmentModel(detail.state.segment_video_model ?? defaultSegmentModel.value);
  if (!opts?.silent) {
    toast.success($tt('video.longVideoProjectLoaded'));
  }
}

async function loadSavedProject(projectId: string, opts?: { silent?: boolean }) {
  try {
    const detail = await api.longVideo.getProject(projectId);
    applyServerProject(detail, opts);
  } catch (e: unknown) {
    if (!opts?.silent) {
      toast.warning($tt('video.longVideoProjectLoadFailed'));
    }
    const msg = e instanceof Error ? e.message : String(e);
    if (!opts?.silent) toast.error($tt('studio.error', { msg }));
    throw e;
  }
}

async function createNewProject() {
  const draft = defaultLongVideoProject({
    keyframe_model: defaultKeyframeModel.value,
    segment_video_model: defaultSegmentModel.value,
  });
  try {
    const created = await api.longVideo.createProject({
      title: draft.title?.trim() || $tt('video.longVideoPageTitle'),
      state: projectStateForServer(draft),
    });
    applyServerProject(created, { silent: true });
    await loadProjectList();
    toast.success($tt('video.longVideoNewProjectDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  }
}

async function deleteSavedProject(projectId: string) {
  try {
    await api.longVideo.deleteProject(projectId);
    if (project.value?.project_id === projectId) {
      await createNewProject();
    }
    await loadProjectList();
    toast.success($tt('video.longVideoProjectDeleted'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  }
}

async function saveProject() {
  const lv = project.value;
  if (!lv) return;
  savingProject.value = true;
  try {
    const title = lv.title?.trim() || $tt('video.longVideoPageTitle');
    const state = projectStateForServer({ ...lv, title });
    let saved;
    if (lv.project_id) {
      saved = await api.longVideo.updateProject(lv.project_id, { title, state });
    } else {
      saved = await api.longVideo.createProject({ title, state });
      patchProjectField('project_id', saved.id);
    }
    patchProjectField('title', saved.title);
    longVideoProject.persistNow();
    await loadProjectList();
    toast.success($tt('video.longVideoSaveProjectOk'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  } finally {
    savingProject.value = false;
  }
}

function buildSegmentAdapters() {
  const adapters: { id: string; weight: number }[] = [];
  if (segmentComposeParams.lora) {
    adapters.push({
      id: String(segmentComposeParams.lora),
      weight: Number(segmentComposeParams.lora_scale) || 1.0,
    });
  }
  return adapters;
}

async function runGenerationTask(submitRes: unknown) {
  const tid = taskIdFromSubmitResponse(submitRes);
  if (!tid) throw new Error('missing task id');
  tasksStore.registerPageOwnedStream(tid);
  generating.value = true;
  return new Promise<void>((resolve, reject) => {
    api.gen.streamMediaTask(tid, {
      onDone: async (doneData: unknown) => {
        generating.value = false;
        tasksStore.unregisterPageOwnedStream(tid);
        const data = doneData as { status: string };
        if (data.status === 'completed') {
          const updated = (await api.gen.getMediaTask(tid)) as {
            result?: { metadata?: { shots?: LongVideoProjectState['shots']; primary_asset_id?: string } };
            metadata?: { long_video?: { shots?: LongVideoProjectState['shots'] } };
          };
          const shotsMeta =
            updated?.result?.metadata?.shots ?? updated?.metadata?.long_video?.shots;
          const primaryId = updated?.result?.metadata?.primary_asset_id;
          if (longVideoProject.project.value) {
            const patch: Partial<LongVideoProjectState> = {};
            if (Array.isArray(shotsMeta)) patch.shots = shotsMeta;
            if (primaryId) patch.final_asset_id = primaryId;
            longVideoProject.patchProject(patch);
          }
          toast.success($tt('studio.genComplete'));
          resolve();
        } else {
          toast.error($tt('studio.genFailed', { msg: '' }));
          reject(new Error('generation failed'));
        }
      },
      onError: () => {
        generating.value = false;
        tasksStore.unregisterPageOwnedStream(tid);
        reject(new Error('connection lost'));
      },
    });
  });
}

async function mergeLongVideo() {
  const lv =
    project.value ||
    defaultLongVideoProject({
      keyframe_model: defaultKeyframeModel.value,
      segment_video_model: defaultSegmentModel.value,
      target_duration_sec: 60,
      title: projectTitle.value,
    });

  if (!lv.shots.length) {
    toast.warning($tt('video.longVideoNeedKeyframes'));
    return;
  }

  generating.value = true;
  try {
    const meta: Record<string, unknown> = {
      source: 'long_video_create',
      long_video_phase: 'assemble_only',
    };
    const prompt = lv.shots[0]?.visual_prompt?.trim() || 'long video';
    const { width, height } = outputSizePixels.value;
    const segModel = lv.segment_video_model;
    const p = segmentComposeParams;
    const longBody: Record<string, unknown> = {
      model: segModel,
      title: projectTitle.value.trim(),
      prompt,
      negative_prompt: p.negative_prompt || '',
      size: `${width}x${height}`,
      fps: p.fps || scalarDefault(segModel, 'fps', 16),
      steps: p.steps || scalarDefault(segModel, 'steps', 40),
      guidance: p.guide_scale ?? scalarDefault(segModel, 'guide_scale', 3.0),
      shift: p.shift || scalarDefault(segModel, 'shift', 12.0) || undefined,
      seed: p.seed ? parseInt(p.seed, 10) : null,
      metadata: { ...meta },
      priority: 'normal',
      long_video: {
        strategy: 'segmented_i2v',
        target_duration_sec: effectiveTargetDurationSec(lv),
        keyframe_model: lv.keyframe_model,
        segment_video_model: lv.segment_video_model,
        segment_duration_sec: Math.max(...lv.shots.map((s) => shotDurationSec(s)), 5),
        overlap_frames: lv.overlap_frames,
        chain_mode: lv.chain_mode,
        character_lora_id: lv.character_lora_id || undefined,
        keyframe_adapters: buildSegmentAdapters(),
        shots: lv.shots.map((s) => ({ ...s, duration_sec: shotDurationSec(s) })),
      },
    };
    const submitRes = await api.gen.createVideoLongGeneration(longBody);
    await runGenerationTask(submitRes);
    openGlobalTaskQueue();
  } catch (e: unknown) {
    generating.value = false;
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  }
}

onMounted(async () => {
  await loadRegistry();
  await loadProjectList();
  const pid = project.value?.project_id;
  if (pid) {
    try {
      await loadSavedProject(pid, { silent: true });
    } catch {
      longVideoProject.initProject({
        keyframe_model: defaultKeyframeModel.value,
        segment_video_model: defaultSegmentModel.value,
        title: project.value?.title ?? '',
      });
    }
  } else {
    longVideoProject.initProject({
      keyframe_model: defaultKeyframeModel.value,
      segment_video_model: defaultSegmentModel.value,
      title: project.value?.title ?? '',
    });
  }
  syncOutputSizeForSegmentModel(segmentModelId.value);
  await loadGallery(true);
});

watch(segmentModelId, (modelId) => {
  syncOutputSizeForSegmentModel(modelId);
});
</script>
