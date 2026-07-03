<template>
  <div class="lv-studio studio-create-page lv-studio--full">
    <header class="lv-studio__toolbar dq-glass--bar">
      <div class="lv-studio__toolbar-inner">
        <h1 class="lv-studio__toolbar-title">{{ $tt('video.longVideoPageTitle') }}</h1>
        <div class="lv-studio__toolbar-spacer" />
        <span v-if="saveStatusLabel" class="lv-studio__save-status" :class="saveStatusClass">
          {{ saveStatusLabel }}
        </span>
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
          :project-id="project?.project_id ?? ''"
          @update:title="patchProjectField('title', $event)"
          @update:keyframe-model="onKeyframeModelChange"
          @update:segment-model="onSegmentModelChange"
          @update:output-size="patchProjectField('output_size', $event)"
          @update:overlap-frames="patchProjectField('overlap_frames', $event)"
        />

        <LongVideoEditorTabs
          :model-value="editorTab"
          :cast-count="project?.characters?.length ?? 0"
          :scene-entity-count="project?.scenes?.length ?? 0"
          :shot-count="shots.length"
          :script-done="scriptStepDone"
          :cast-done="castStepDone"
          :scenes-done="scenesStepDone"
          :storyboard-done="storyboardStepDone"
          @update:model-value="onEditorTabChange"
        />

        <div v-show="editorTab === 'script'" class="lv-editor-pane lv-editor-pane--script">
          <LongVideoBriefPanel
            :script-text="scriptText"
            :chapter-title="project?.chapter_title ?? ''"
            :chapter-analysis="project?.chapter_analysis ?? null"
            :characters="project?.characters ?? []"
            :script-parsed="scriptStepDone"
            :style-anchor="project?.style_anchor ?? ''"
            :target-duration-sec="project?.target_duration_sec ?? 60"
            :segment-duration-sec="project?.segment_duration_sec ?? 5"
            :script-parse-llm-model="project?.script_parse_llm_model ?? ''"
            :parsing="isChapterAnalyzing"
            :parse-progress-phase="scriptParseProgressPhase"
            :expanding="isScriptExpanding"
            :parse-error="scriptParseError"
            :project-id="project?.project_id ?? ''"
            :parsed-shot-count="shots.length"
            @update:script-text="onScriptTextChange"
            @update:chapter-title="patchProjectField('chapter_title', $event)"
            @update:chapter-analysis="onChapterAnalysisChange"
            @update:target-duration-sec="onTargetDurationChange"
            @update:segment-duration-sec="onSegmentDurationChange"
            @update:script-parse-llm-model="patchProjectField('script_parse_llm_model', $event)"
            @update:style-anchor="onUpdateStyleAnchor"
            @expand="onScriptExpand"
            @parse="onScriptParse"
            @go-to-cast="onEditorTabChange('cast')"
        />
        </div>

        <div v-show="editorTab === 'cast'" class="lv-editor-pane lv-editor-pane--cast">
        <LongVideoCastPanel
          :characters="project?.characters ?? []"
          :character-anchor="project?.character_anchor ?? ''"
          :script-synopsis="projectSynopsis"
          :script-parsed="scriptStepDone"
          :project-id="project?.project_id ?? ''"
          :compatible-loras="keyframeCompatibleLoras"
          :portrait-generating-key="portraitGeneratingKey"
          :vision-backfill-key="visionBackfillKey"
          :batch-generating="batchPortraitGenerating"
          @update:characters="onUpdateCharacters"
          @generate-portrait="onGeneratePortrait"
          @pick-portrait-gallery="onPickPortraitGallery"
          @clear-portrait="onClearPortrait"
          @vision-backfill="onVisionBackfillLook"
          @batch-generate-portraits="onBatchGeneratePortraits"
          @import-style-anchor="onUpdateStyleAnchor"
          @go-to-script="onEditorTabChange('script')"
          @go-to-storyboard="onEditorTabChange('storyboard')"
        />
        </div>

        <div v-show="editorTab === 'scenes'" class="lv-editor-pane lv-editor-pane--scenes">
        <LongVideoScenePanel
          :scenes="project?.scenes ?? []"
          :script-parsed="scriptStepDone"
          :project-id="project?.project_id ?? ''"
          :ref-generating-key="sceneRefGeneratingKey"
          :vision-backfill-key="sceneVisionBackfillKey"
          :batch-generating="batchSceneRefGenerating"
          @update:scenes="onUpdateScenes"
          @generate-ref="onGenerateSceneRef"
          @pick-ref-gallery="onPickSceneRefGallery"
          @clear-ref="onClearSceneRef"
          @vision-backfill="onVisionBackfillSceneLook"
          @batch-generate-refs="onBatchGenerateSceneRefs"
          @go-to-storyboard="onEditorTabChange('storyboard')"
        />
        </div>

        <div v-show="editorTab === 'storyboard'" class="lv-editor-pane lv-editor-pane--storyboard">
        <LongVideoGroupToolbar
          :shots="shots"
          :selection="selection"
          :chain-mode="project?.chain_mode ?? 'keyframe_only'"
          :generating="batchGroupGenerating"
          @generate-group="onBatchGenerateCurrentGroup"
          @generate-all-anchors="onBatchGenerateAllAnchors"
          @generate-all-segments="onBatchGenerateAllSegments"
        />
        <LongVideoGroupProgressBar
          :shots="shots"
          @select-group="onSelectBeatGroup"
        />
        <LongVideoBeatGroupRail
          :shots="shots"
          :selection="selection"
          :keyframe-generating-index="keyframeGeneratingIndex"
          :segment-generating-indices="segmentGeneratingIndicesList"
          :output-width="outputSizePixels.width"
          :output-height="outputSizePixels.height"
          @select-segment="onSelectSegment"
          @insert-anchor="onInsertFaceAnchor"
          @resplit-beat="onResplitBeatGroup"
        />
        </div>
      </div>
    </div>

    <LongVideoInspectorDrawer
      :open="Boolean(selection)"
      :shots="shots"
      :selection="selection"
      :segment-duration-options="segmentDurationOptions"
      :keyframe-generating="keyframeGenerating"
      :segment-generating="segmentGeneratingForSelection"
      :segment-model-supports-r2v="segmentModelSupportsR2v"
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
      :scenes="project?.scenes ?? []"
      :character-anchor="project?.character_anchor ?? ''"
      :style-anchor="project?.style_anchor ?? ''"
      :project-id="project?.project_id ?? ''"
      :parse-run-id="project?.chapter_analysis?.parse_run_id ?? ''"
      @update:open="onInspectorOpenChange"
      @update-visual="onUpdateVisual"
      @update-cast-looks="onUpdateCastLooks"
      @update-scene-look="onUpdateSceneLook"
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
      @select-segment="onSelectSegment"
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

    <LongVideoParseStrategyDialog
      v-model:open="parseStrategyOpen"
      @choose="onParseStrategyChoose"
      @cancel="onParseStrategyCancel"
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
      @pick="onGalleryDrawerPick"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch, nextTick, inject, type Ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast, confirm } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse, type LongVideoChapterAnalyzeShot } from '@/utils/api';
import {
  analyzeReferenceViaChat,
  checkKeyframeConsistencyViaChat,
  expandScriptViaChat,
} from '@/utils/llmMessages';
import { $tt } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import type { LongVideoChainMode, LongVideoCharacter, LongVideoChapterAnalysis, LongVideoEditorTab, LongVideoProjectState, LongVideoProjectSummary, LongVideoScene, LongVideoShotCastLook, LongVideoShotSceneLook, LongVideoShotState, SystemInfo } from '@/types';
import LongVideoParseStrategyDialog, {
  type LongVideoParseStrategy,
} from '@/components/long-video/LongVideoParseStrategyDialog.vue';
import LongVideoBriefPanel from '@/components/long-video/LongVideoBriefPanel.vue';
import LongVideoCastPanel from '@/components/long-video/LongVideoCastPanel.vue';
import LongVideoScenePanel from '@/components/long-video/LongVideoScenePanel.vue';
import LongVideoEditorTabs from '@/components/long-video/LongVideoEditorTabs.vue';
import LongVideoSettingsPanel from '@/components/long-video/LongVideoSettingsPanel.vue';
import LongVideoGroupToolbar from '@/components/long-video/LongVideoGroupToolbar.vue';
import LongVideoBeatGroupRail from '@/components/long-video/LongVideoBeatGroupRail.vue';
import LongVideoGroupProgressBar from '@/components/long-video/LongVideoGroupProgressBar.vue';
import LongVideoInspectorDrawer from '@/components/long-video/LongVideoInspectorDrawer.vue';
import LongVideoGalleryDrawer from '@/components/long-video/LongVideoGalleryDrawer.vue';
import LongVideoOutputStrip from '@/components/long-video/LongVideoOutputStrip.vue';
import LongVideoProjectSidebar from '@/components/long-video/LongVideoProjectSidebar.vue';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import { useLongVideoProject } from '@/composables/useLongVideoProject';
import { useLongVideoAutoSave } from '@/composables/useLongVideoAutoSave';
import { useLongVideoRegistry } from '@/composables/useLongVideoRegistry';
import {
  useLongVideoKeyframeCompose,
  type KeyframeComposeParams,
} from '@/composables/useLongVideoKeyframeCompose';
import { useLongVideoSegmentCompose } from '@/composables/useLongVideoSegmentCompose';
import { useComposerLlm } from '@/composables/useComposerLlm';
import { useStudioGallery } from '@/composables/useStudioGallery';
import {
  appendImageInferenceFields,
  appendZImageEnhancementFields,
  isControlNetHostRuntimeAvailable,
  resolveControlAssetId,
} from '@/composables/useStructuralGuide';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import router from '@/router';
import {
  appendActiveEnumFields,
  formatResolutionOptionLabel,
  normalizeParamsDef,
  parseSizeValue,
  strengthDefaultFromRegistry,
  strengthToSourceFidelity,
  VIDEO_INFERENCE_ENUM_KEYS,
} from '@/utils/registryParamSchema';
import {
  allocateShotDurations,
  buildPortraitPrompt,
  buildCastVisionBackfillQuestion,
  buildSceneVisionBackfillQuestion,
  buildParseProvenanceByShot,
  buildSceneEnvironmentPrompt,
  resolveLongVideoLocale,
  collectCastReferenceAssetIdsForShot,
  createEmptyShot,
  defaultLongVideoProject,
  effectiveShotChainMode,
  enrichShotsWithSceneLooks,
  mergeParsedShotsWithPrevious,
  insertKeyframeAfter,
  insertKeyframeBefore,
  invalidateSegmentAsset,
  resolveScriptText,
  longVideoHasPersistableContent,
  PORTRAIT_REFERENCE_SIZE,
  PORTRAIT_NEGATIVE_PROMPT_ZH,
  PORTRAIT_NEGATIVE_PROMPT_EN,
  SCENE_REFERENCE_SIZE,
  SCENE_ENV_NEGATIVE_ZH,
  SCENE_ENV_NEGATIVE_EN,
  keyframeGenerationPrompt,
  keyframePromptContextForShot,
  KEYFRAME_CAST_NEGATIVE_ZH,
  KEYFRAME_CAST_NEGATIVE_EN,
  looksMissingPortrait,
  looksMissingSceneReference,
  normalizeCharacterLookLabels,
  mergeCharacterRosters,
  mergeSceneRosters,
  mergeKeyframeLoraAdapters,
  resolvePrimaryCastPortraitForShot,
  shotCastMatchText,
  shotSceneText,
  shotKeyframeText,
  shotNeedsKeyframe,
  shotVideoPrompt,
  canGenerateSegmentShot,
  segmentI2vSourceAssetId,
  findFaceAnchorShot,
  groupShotsByBeat,
  planGroupGeneration,
  allPendingAnchorKeyframeIndices,
  allPendingSegmentIndices,
  selectedBeatGroupId,
  insertFaceAnchorIntoGroup,
  resplitBeatGroupRule,
  groupHasFaceAnchor,
  syncChapterAnalysisFields,
  syncRosterToCharacterAnchor,
  hydrateCharacterRoster,
  MIN_LONG_VIDEO_KEYFRAMES,
  nextShotId,
  numFramesForDurationSec,
  projectStateForServer,
  removeKeyframeAt,
  removeKeyframeNeedsConfirm,
  shotDurationSec,
  buildKeyframeGroundingMetadata,
  resolveSceneForShot,
} from '@/utils/longVideoProject';
import { berniniMaxReferenceImages } from '@/utils/videoEditSource';
import '@/styles/long-video.css';

const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const systemInfo = inject<Ref<SystemInfo>>('systemInfo', ref({} as SystemInfo));
const longVideoProject = useLongVideoProject();
const {
  profile,
  supported: registrySupported,
  modelLabel,
  modelParameters,
  modelHasBerniniRenderer,
  resolutionOptionsForModel,
  pickOutputSizeForModel,
  loadRegistry,
} = useLongVideoRegistry();
const { enhance, isEnhancing, analyzeLongVideoChapter, isChapterAnalyzing } = useComposerLlm();
const { locale: uiLocale } = useI18n();
const { galleryItems, loadGallery } = useStudioGallery('image', { defaultGroupMode: false });

const generating = ref(false);
const savingProject = ref(false);
const keyframeGeneratingIndex = ref<number | null>(null);
const segmentGeneratingIndices = ref<Set<number>>(new Set());
const segmentGeneratingIndicesList = computed(() => [...segmentGeneratingIndices.value]);
const segmentGeneratingForSelection = computed(() => {
  const sel = selection.value;
  if (sel?.kind !== 'segment' && sel?.kind !== 'clip') return false;
  return segmentGeneratingIndices.value.has(sel.index);
});

const segmentModelSupportsR2v = computed(() =>
  modelHasBerniniRenderer(project.value?.segment_video_model ?? defaultSegmentModel.value),
);
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
const portraitGeneratingKey = ref<string | null>(null);
const visionBackfillKey = ref<string | null>(null);
const sceneVisionBackfillKey = ref<string | null>(null);
const batchPortraitGenerating = ref(false);
const sceneRefGeneratingKey = ref<string | null>(null);
const batchSceneRefGenerating = ref(false);
const batchGroupGenerating = ref(false);
const galleryPickMode = ref<'keyframe' | 'portrait' | 'scene_ref'>('keyframe');
const galleryPickPortrait = ref<{ ci: number; li: number } | null>(null);
const galleryPickSceneRef = ref<{ si: number; li: number } | null>(null);
const consistencyWarnings = ref<Record<number, string>>({});
const consistencyChecking = ref(false);
const suppressAutoSave = ref(0);
const parseStrategyOpen = ref(false);
const scriptParseError = ref('');
const scriptParseProgressPhase = ref('');
const isScriptExpanding = ref(false);

type ChapterAnalyzeApiResult = NonNullable<Awaited<ReturnType<typeof analyzeLongVideoChapter>>>;

const project = computed(() => longVideoProject.project.value);
const shots = computed(() => project.value?.shots ?? []);
const selection = computed(() => project.value?.selection ?? null);
const editorTab = computed((): LongVideoEditorTab => project.value?.editor_tab ?? 'script');

function withProjectMetadata(meta: Record<string, unknown> = {}): Record<string, unknown> {
  const pid = project.value?.project_id;
  if (!pid) return meta;
  return {
    ...meta,
    group_id: pid,
    long_video_project_id: pid,
    long_video_project_title: project.value?.title ?? '',
  };
}

/** Best-effort save so generated assets get a gallery group via project_id. */
async function ensureProjectSavedForGeneration(): Promise<void> {
  const lv = project.value;
  if (!lv || lv.project_id) return;
  if (!longVideoHasPersistableContent(lv)) return;
  await persistProjectToServer({ silent: true });
}

const scriptText = computed(() => {
  const lv = project.value;
  return lv ? resolveScriptText(lv) : '';
});

const projectSynopsis = computed(() => {
  const lv = project.value;
  if (!lv) return '';
  return lv.chapter_analysis?.synopsis?.trim() || '';
});

const scriptStepDone = computed(() => {
  const lv = project.value;
  if (!lv) return false;
  return (lv.chapter_analysis?.scene_beats?.length ?? 0) >= 2;
});

const castStepDone = computed(() => {
  const chars = project.value?.characters ?? [];
  if (!chars.length) return false;
  return looksMissingPortrait(chars).length === 0;
});

const scenesStepDone = computed(() => {
  const scenes = project.value?.scenes ?? [];
  if (!scenes.length) return true;
  return looksMissingSceneReference(scenes).length === 0;
});

const storyboardStepDone = computed(() => shots.value.length >= MIN_LONG_VIDEO_KEYFRAMES);

const defaultKeyframeModel = computed(() => profile.value?.keyframe_models[0] ?? 'z-image-turbo');
const defaultSegmentModel = computed(() => profile.value?.segment_models[0] ?? 'wan-2.2-i2v-14b');
const keyframeModelOptions = computed(() => profile.value?.keyframe_models ?? []);
const segmentModelOptions = computed(() => profile.value?.segment_models ?? []);
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
  selection.value?.kind === 'segment' ? selection.value.index : null,
);

const canGenerateKeyframe = computed(() => {
  const idx = selectedNodeIndex.value;
  if (idx == null) return false;
  const shot = shots.value[idx];
  if (!shotNeedsKeyframe(shot)) return false;
  if (!shotKeyframeText(shot)) return false;
  if (composeMode.value === 'img2img' && !referenceImage.value) return false;
  return true;
});

const consistencyWarningForSelection = computed(() => {
  const idx = selectedNodeIndex.value;
  if (idx == null) return null;
  return consistencyWarnings.value[idx] ?? null;
});

function onEditorTabChange(tab: LongVideoEditorTab) {
  patchProjectField('editor_tab', tab);
}

function portraitKey(ci: number, li: number): string {
  return `${ci}-${li}`;
}

function patchLookPortrait(
  ci: number,
  li: number,
  patch: { reference_asset_id?: string; portrait_prompt?: string; body?: string; vision_description?: string },
) {
  const lv = project.value;
  if (!lv?.characters?.[ci]?.looks[li]) return;
  const characters = lv.characters.map((ch, i) => {
    if (i !== ci) return ch;
    return {
      ...ch,
      looks: ch.looks.map((lk, j) => {
        if (j !== li) return lk;
        const next = { ...lk, ...patch };
        if ('portrait_prompt' in patch && patch.portrait_prompt === undefined) {
          delete next.portrait_prompt;
        }
        if ('vision_description' in patch && patch.vision_description === undefined) {
          delete next.vision_description;
        }
        return next;
      }),
    };
  });
  onUpdateCharacters(characters);
}

function sceneRefKey(si: number, li: number): string {
  return `${si}-${li}`;
}

function patchSceneLookRef(
  si: number,
  li: number,
  patch: { reference_asset_id?: string; environment_prompt?: string; body?: string; vision_description?: string },
) {
  const lv = project.value;
  if (!lv?.scenes?.[si]?.looks[li]) return;
  const scenes = lv.scenes.map((sc, i) => {
    if (i !== si) return sc;
    return {
      ...sc,
      looks: sc.looks.map((lk, j) => {
        if (j !== li) return lk;
        const next = { ...lk, ...patch };
        if ('environment_prompt' in patch && patch.environment_prompt === undefined) {
          delete next.environment_prompt;
        }
        if ('vision_description' in patch && patch.vision_description === undefined) {
          delete next.vision_description;
        }
        return next;
      }),
    };
  });
  onUpdateScenes(scenes);
}

function onUpdateScenes(scenes: LongVideoScene[]) {
  longVideoProject.patchProject({ scenes });
}

function patchSceneGrounding(
  si: number,
  patch: { grounding_panorama_asset_id?: string; grounding_depth_asset_id?: string },
) {
  const lv = project.value;
  if (!lv?.scenes?.[si]) return;
  const scenes = lv.scenes.map((sc, i) => (i === si ? { ...sc, ...patch } : sc));
  onUpdateScenes(scenes);
}

async function onGenerateSceneRef(si: number, li: number) {
  const lv = project.value;
  if (!lv) return;
  const sc = lv.scenes?.[si];
  const look = sc?.looks[li];
  if (!sc || !look) return;

  const locale = uiLocale.value.startsWith('zh') ? 'zh' : 'en';
  const prompt = buildSceneEnvironmentPrompt(sc, look, lv.style_anchor ?? '', locale, {
    useCache: false,
  });
  const model = lv.keyframe_model ?? defaultKeyframeModel.value;
  const { width, height } = SCENE_REFERENCE_SIZE;
  const key = sceneRefKey(si, li);
  const modelParameters = (registryStore.registry?.models?.[model] as Record<string, unknown> | undefined)
    ?.parameters as Record<string, unknown> | undefined;

  sceneRefGeneratingKey.value = key;
  try {
    await ensureProjectSavedForGeneration();
    const genBody: Record<string, unknown> = {
      model,
      prompt,
      negative_prompt: locale === 'zh' ? SCENE_ENV_NEGATIVE_ZH : SCENE_ENV_NEGATIVE_EN,
      size: `${width}x${height}`,
      n: 1,
      steps: keyframeComposeParams.steps,
      guidance: keyframeComposeParams.guidance,
      metadata: withProjectMetadata({ long_video_phase: 'scene_ref', scene_id: sc.id, scene_look_id: look.id }),
      priority: 'normal',
    };
    appendImageInferenceFields(
      genBody,
      keyframeComposeParams as Record<string, unknown>,
      modelParameters,
    );
    const submitRes = await api.gen.createImageGeneration(genBody);
    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) throw new Error('no task id');
    tasksStore.registerPageOwnedStream(tid);
    const { assetId } = await waitForImageTask(tid);
    if (!assetId) throw new Error('no asset');
    patchSceneLookRef(si, li, { reference_asset_id: assetId, environment_prompt: prompt });
    patchSceneGrounding(si, { grounding_panorama_asset_id: assetId });
    try {
      const depth = await api.longVideo.sceneGroundingDepthFromAsset({
        source_asset_id: assetId,
        width,
        height,
      });
      if (depth.depth_asset_id) {
        patchSceneGrounding(si, {
          grounding_panorama_asset_id: assetId,
          grounding_depth_asset_id: depth.depth_asset_id,
        });
      }
    } catch {
      /* depth-pro optional; panorama still usable */
    }
    try {
      await loadGallery(true);
    } catch {
      /* gallery refresh must not fail the generation flow */
    }
    toast.success($tt('video.longVideoSceneRefDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.genFailed', { msg }));
  } finally {
    if (sceneRefGeneratingKey.value === key) sceneRefGeneratingKey.value = null;
  }
}

function onPickSceneRefGallery(si: number, li: number) {
  galleryPickMode.value = 'scene_ref';
  galleryPickSceneRef.value = { si, li };
  showKeyframePicker.value = true;
}

function onClearSceneRef(si: number, li: number) {
  patchSceneLookRef(si, li, {
    reference_asset_id: undefined,
    environment_prompt: undefined,
    vision_description: undefined,
  });
}

async function onVisionBackfillSceneLook(si: number, li: number) {
  const lv = project.value;
  const sc = lv?.scenes?.[si];
  const look = sc?.looks[li];
  const assetId = look?.reference_asset_id;
  if (!sc || !look || !assetId) return;
  const key = sceneRefKey(si, li);
  sceneVisionBackfillKey.value = key;
  const locale = resolveLongVideoLocale(uiLocale.value);
  try {
    const visionInfo = await api.gen.getVisionModelInfo();
    if (!visionInfo.available) {
      toast.warning($tt('video.longVideoSceneVisionUnavailable'));
      return;
    }
    const question = buildSceneVisionBackfillQuestion(sc, look, locale);
    const text = await analyzeReferenceViaChat(assetId, question, { locale });
    if (text) {
      patchSceneLookRef(si, li, { vision_description: text });
      toast.success($tt('video.longVideoSceneVisionBackfillDone'));
    }
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  } finally {
    if (sceneVisionBackfillKey.value === key) sceneVisionBackfillKey.value = null;
  }
}

async function onBatchGenerateSceneRefs() {
  const lv = project.value;
  if (!lv || batchSceneRefGenerating.value) return;
  const missing = looksMissingSceneReference(lv.scenes ?? []);
  if (!missing.length) return;
  batchSceneRefGenerating.value = true;
  try {
    for (const { sceneIndex, lookIndex } of missing) {
      await onGenerateSceneRef(sceneIndex, lookIndex);
    }
  } finally {
    batchSceneRefGenerating.value = false;
  }
}

async function onGeneratePortrait(ci: number, li: number) {
  const lv = project.value;
  if (!lv) return;
  const ch = lv.characters?.[ci];
  const look = ch?.looks[li];
  if (!ch || !look) return;

  const locale = resolveLongVideoLocale(uiLocale.value);
  const otherNames = (lv.characters ?? [])
    .filter((_, i) => i !== ci)
    .map((c) => c.name)
    .filter(Boolean);
  const prompt = buildPortraitPrompt(ch, look, lv.style_anchor ?? '', locale, {
    otherCharacterNames: otherNames,
    useCache: false,
  });
  const model = lv.portrait_model ?? lv.keyframe_model ?? defaultKeyframeModel.value;
  const { width, height } = PORTRAIT_REFERENCE_SIZE;
  const key = portraitKey(ci, li);
  const modelParameters = (registryStore.registry?.models?.[model] as Record<string, unknown> | undefined)
    ?.parameters as Record<string, unknown> | undefined;

  portraitGeneratingKey.value = key;
  try {
    await ensureProjectSavedForGeneration();
    const genBody: Record<string, unknown> = {
      model,
      prompt,
      negative_prompt: locale === 'zh' ? PORTRAIT_NEGATIVE_PROMPT_ZH : PORTRAIT_NEGATIVE_PROMPT_EN,
      size: `${width}x${height}`,
      n: 1,
      steps: keyframeComposeParams.steps,
      guidance: keyframeComposeParams.guidance,
      metadata: withProjectMetadata({ long_video_phase: 'cast_portrait', cast_character_id: ch.id, cast_look_id: look.id }),
      priority: 'normal',
    };
    appendImageInferenceFields(
      genBody,
      keyframeComposeParams as Record<string, unknown>,
      modelParameters,
    );
    const submitRes = await api.gen.createImageGeneration(genBody);
    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) throw new Error('no task id');
    tasksStore.registerPageOwnedStream(tid);
    const { assetId } = await waitForImageTask(tid);
    if (!assetId) throw new Error('no asset');
    patchLookPortrait(ci, li, { reference_asset_id: assetId, portrait_prompt: prompt });
    try {
      await loadGallery(true);
    } catch {
      /* gallery refresh must not fail the generation flow */
    }
    toast.success($tt('video.longVideoPortraitDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.genFailed', { msg }));
  } finally {
    if (portraitGeneratingKey.value === key) portraitGeneratingKey.value = null;
  }
}

function onPickPortraitGallery(ci: number, li: number) {
  galleryPickMode.value = 'portrait';
  galleryPickPortrait.value = { ci, li };
  showKeyframePicker.value = true;
}

function onClearPortrait(ci: number, li: number) {
  patchLookPortrait(ci, li, {
    reference_asset_id: undefined,
    portrait_prompt: undefined,
    vision_description: undefined,
  });
}

async function onVisionBackfillLook(ci: number, li: number) {
  const lv = project.value;
  const ch = lv?.characters?.[ci];
  const look = ch?.looks[li];
  const assetId = look?.reference_asset_id;
  if (!ch || !look || !assetId) return;
  const key = portraitKey(ci, li);
  visionBackfillKey.value = key;
  const locale = resolveLongVideoLocale(uiLocale.value);
  const otherNames = (lv.characters ?? [])
    .filter((_, i) => i !== ci)
    .map((c) => c.name.trim())
    .filter(Boolean);
  try {
    const visionInfo = await api.gen.getVisionModelInfo();
    if (!visionInfo.available) {
      toast.warning($tt('video.longVideoCastVisionUnavailable'));
      return;
    }
    const question = buildCastVisionBackfillQuestion(ch, otherNames, locale);
    const text = await analyzeReferenceViaChat(assetId, question, { locale });
    if (text) {
      patchLookPortrait(ci, li, { vision_description: text });
      toast.success($tt('video.longVideoCastVisionBackfillDone'));
    }
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  } finally {
    if (visionBackfillKey.value === key) visionBackfillKey.value = null;
  }
}

async function onBatchGeneratePortraits() {
  const lv = project.value;
  if (!lv || batchPortraitGenerating.value) return;
  const missing = looksMissingPortrait(lv.characters ?? []);
  if (!missing.length) return;
  batchPortraitGenerating.value = true;
  try {
    for (const { characterIndex, lookIndex } of missing) {
      await onGeneratePortrait(characterIndex, lookIndex);
    }
  } finally {
    batchPortraitGenerating.value = false;
  }
}

async function onCheckConsistency(index: number) {
  const lv = project.value;
  const shot = lv?.shots[index];
  if (!lv || !shot?.keyframe_asset_id) return;
  const scene = shotCastMatchText(shot);
  const portrait = resolvePrimaryCastPortraitForShot(lv.characters ?? [], shot.cast_looks ?? [], scene);
  if (!portrait?.reference_asset_id) return;

  consistencyChecking.value = true;
  try {
    const visionInfo = await api.gen.getVisionModelInfo();
    if (!visionInfo.available) {
      toast.warning($tt('video.longVideoCastVisionUnavailable'));
      return;
    }
    const locale = resolveLongVideoLocale(uiLocale.value);
    const answer = await checkKeyframeConsistencyViaChat(
      portrait.reference_asset_id,
      shot.keyframe_asset_id,
      locale,
    );
    const ok = /一致|consistent/i.test(answer);
    if (ok) {
      const next = { ...consistencyWarnings.value };
      delete next[index];
      consistencyWarnings.value = next;
    } else {
      consistencyWarnings.value = {
        ...consistencyWarnings.value,
        [index]: answer || $tt('video.longVideoConsistencyWarning'),
      };
    }
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  } finally {
    consistencyChecking.value = false;
  }
}

function onGalleryDrawerPick(payload: { path: string }) {
  if (galleryPickMode.value === 'portrait' && galleryPickPortrait.value) {
    const assetId = assetIdFromGalleryPath(payload.path);
    if (!assetId) {
      toast.warning($tt('canvas.controlImageAssetRequired'));
      return;
    }
    const { ci, li } = galleryPickPortrait.value;
    patchLookPortrait(ci, li, { reference_asset_id: assetId });
    galleryPickPortrait.value = null;
    galleryPickMode.value = 'keyframe';
    showKeyframePicker.value = false;
    return;
  }
  if (galleryPickMode.value === 'scene_ref' && galleryPickSceneRef.value) {
    const assetId = assetIdFromGalleryPath(payload.path);
    if (!assetId) {
      toast.warning($tt('canvas.controlImageAssetRequired'));
      return;
    }
    const { si, li } = galleryPickSceneRef.value;
    patchSceneLookRef(si, li, { reference_asset_id: assetId });
    galleryPickSceneRef.value = null;
    galleryPickMode.value = 'keyframe';
    showKeyframePicker.value = false;
    return;
  }
  onKeyframeAssetPick(payload);
}

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
  () => (selection.value?.kind === 'segment' ? shots.value[selection.value.index] : null),
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
  if (!shots.value.length || generating.value || segmentGeneratingIndices.value.size > 0) return true;
  return !shots.value.every((s) => s.status === 'segment_ready' && s.segment_asset_id);
});

const recentImages = computed(() => galleryItems.value.slice(0, 40) as Array<Record<string, unknown>>);

function patchProjectField<K extends keyof LongVideoProjectState>(key: K, value: LongVideoProjectState[K]) {
  longVideoProject.patchProject({ [key]: value } as Partial<LongVideoProjectState>);
}

function hasCastPortraitWork(): boolean {
  const chars = project.value?.characters ?? [];
  return chars.some((ch) => ch.looks.some((lk) => Boolean(lk.reference_asset_id)));
}

function hasStoryboardGeneratedWork(): boolean {
  const lv = project.value;
  if (!lv) return false;
  return lv.shots.some(
    (s) => s.visual_prompt?.trim() || s.motion_prompt?.trim() || s.keyframe_asset_id || s.segment_asset_id,
  );
}

function hasExistingParseWork(): boolean {
  const lv = project.value;
  if (!lv) return false;
  if (scriptStepDone.value) return true;
  if ((lv.characters?.length ?? 0) > 0) return true;
  if ((lv.shots?.length ?? 0) > 0) return true;
  if (hasCastPortraitWork()) return true;
  return hasStoryboardGeneratedWork();
}

function deriveNewProjectTitle(): string {
  const lv = project.value;
  const fromChapter = lv?.chapter_title?.trim();
  if (fromChapter) return fromChapter;
  const text = scriptText.value.trim();
  if (!text) return $tt('video.longVideoPageTitle');
  return text.length > 28 ? `${text.slice(0, 28)}…` : text;
}

async function bootstrapNewProjectForParse(): Promise<boolean> {
  const lv = project.value;
  if (!lv) return false;
  const draft = defaultLongVideoProject({
    keyframe_model: lv.keyframe_model,
    segment_video_model: lv.segment_video_model,
    segment_duration_sec: lv.segment_duration_sec ?? 5,
    target_duration_sec: lv.target_duration_sec ?? 60,
    output_size: lv.output_size,
    overlap_frames: lv.overlap_frames,
    chain_mode: lv.chain_mode,
    title: deriveNewProjectTitle(),
    script_text: scriptText.value,
    chapter_title: lv.chapter_title,
  });
  try {
    const created = await api.longVideo.createProject({
      title: draft.title?.trim() || $tt('video.longVideoPageTitle'),
      state: projectStateForServer(draft),
    });
    applyServerProject(created, { silent: true });
    await loadProjectList();
    toast.info($tt('video.longVideoParseStrategyNewProjectStarted'));
    return true;
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
    return false;
  }
}

function setProjectWithChapterSync(next: LongVideoProjectState) {
  const locale: 'zh' | 'en' = uiLocale.value.startsWith('zh') ? 'zh' : 'en';
  const beatTexts = (next.chapter_analysis?.scene_beats ?? []).map((b) => b.beat);
  const characters = normalizeCharacterLookLabels(next.characters ?? [], locale, beatTexts);
  const style = (next.style_anchor ?? '').trim();
  const characterAnchor =
    characters.length > 0
      ? syncRosterToCharacterAnchor(characters, style)
      : next.character_anchor;
  let patched: LongVideoProjectState = {
    ...next,
    characters,
    character_anchor: characterAnchor,
  };
  const analysis = patched.chapter_analysis;
  if (analysis) {
    patched = {
      ...patched,
      chapter_analysis: syncChapterAnalysisFields(analysis, {
        characters,
        character_anchor: characterAnchor,
        style_anchor: patched.style_anchor ?? analysis.style_anchor,
      }),
    };
  }
  longVideoProject.setProject(patched);
}

function onScriptTextChange(text: string) {
  const lv = project.value;
  if (!lv) return;
  longVideoProject.patchProject({ script_text: text });
}

async function onScriptExpand() {
  const lv = project.value;
  if (!lv) return;
  const text = scriptText.value.trim();
  if (!text) {
    toast.warning($tt('video.longVideoScriptNeedText'));
    return;
  }
  isScriptExpanding.value = true;
  scriptParseError.value = '';
  try {
    const seg = Math.max(1, lv.segment_duration_sec ?? 5);
    const target = Math.max(seg, lv.target_duration_sec ?? 60);
    const expanded = await expandScriptViaChat(text, {
      locale: uiLocale.value,
      targetShotCount: Math.max(2, Math.round(target / seg)),
      narrativeBudget: 'standard',
    });
    const next = expanded.trim();
    if (!next) {
      toast.error($tt('video.longVideoScriptExpandFailed', { msg: $tt('video.longVideoChapterAnalyzeEmpty') }));
      return;
    }
    longVideoProject.patchProject({ script_text: next });
    toast.success($tt('video.longVideoScriptExpandDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('video.longVideoScriptExpandFailed', { msg }));
  } finally {
    isScriptExpanding.value = false;
  }
}

async function onScriptParse() {
  if (hasExistingParseWork()) {
    parseStrategyOpen.value = true;
    return;
  }
  await runScriptParsePipeline('replace');
}

async function onParseStrategyChoose(strategy: LongVideoParseStrategy) {
  await runScriptParsePipeline(strategy);
}

function onParseStrategyCancel() {
  parseStrategyOpen.value = false;
}

function onChapterAnalysisChange(value: LongVideoChapterAnalysis | undefined) {
  const lv = project.value;
  if (!lv) return;
  setProjectWithChapterSync({
    ...lv,
    chapter_analysis: value,
    ...(value
      ? {
          style_anchor: value.style_anchor ?? lv.style_anchor,
          character_anchor: value.character_anchor ?? lv.character_anchor,
          characters: value.characters?.length ? value.characters : lv.characters,
          scenes: value.scenes?.length ? value.scenes : lv.scenes,
        }
      : {}),
  });
}

function onTargetDurationChange(value: number) {
  patchProjectField('target_duration_sec', value);
}

function onSegmentDurationChange(value: number) {
  const sec = Math.max(1, Number(value) || 5);
  longVideoProject.patchProject({ segment_duration_sec: sec });
}

function storyboardShotsFromResponse(
  apiShots: LongVideoChapterAnalyzeShot[],
  opts: {
    defaultSegmentSec: number;
    targetDurationSec: number;
    beatTexts?: string[];
  },
): LongVideoShotState[] {
  const fallbackDurations = allocateShotDurations({
    sceneCount: apiShots.length,
    targetDurationSec: opts.targetDurationSec,
    defaultSegmentSec: opts.defaultSegmentSec,
    beatTexts: opts.beatTexts,
  });
  return apiShots.map((s, i) => {
    const videoPrompt = (s.video_prompt || s.motion_prompt || '').trim();
    const startVisual = (s.start_visual_prompt || s.visual_prompt || '').trim();
    const chainMode =
      s.chain_mode === 'first_last' || s.flf_mode === 'first_last'
        ? ('keyframe_only' as const)
        : s.chain_mode ??
          (s.start_frame_mode === 'prev_segment_tail'
            ? ('last_frame' as const)
            : ('keyframe_only' as const));
    return {
      id: s.id || `shot_${String(i).padStart(2, '0')}`,
      order: i,
      visual_prompt: startVisual,
      motion_prompt: videoPrompt,
      video_prompt: videoPrompt,
      start_visual_prompt: startVisual || undefined,
      end_visual_prompt: s.end_visual_prompt?.trim() || undefined,
      anchor_visual_prompt: s.anchor_visual_prompt?.trim() || undefined,
      segment_role: s.segment_role ?? 'keyframe',
      start_frame_mode: s.start_frame_mode ?? 'keyframe',
      segment_group_id: s.segment_group_id,
      segment_group_index: s.segment_group_index,
      face_anchor_shot_id: s.face_anchor_shot_id,
      flf_mode: s.flf_mode ?? 'none',
      end_frame_sync_anchor: s.end_frame_sync_anchor,
      chain_mode: chainMode,
      scene_prompt: s.scene_prompt || '',
      cast_looks: s.cast_looks ?? [],
      scene_look: s.scene_look,
      first_frame_visibility: s.first_frame_visibility,
      end_visibility: s.end_visibility,
      characters_on_screen: s.characters_on_screen ?? [],
      clip_start_state: s.clip_start_state,
      clip_end_state: s.clip_end_state,
      first_frame_requirement: s.first_frame_requirement,
      camera_zone_id: s.camera_zone_id,
      first_frame_strategy: s.first_frame_strategy,
      location: s.location?.trim() || undefined,
      narrative_beat_index:
        typeof s.narrative_beat_index === 'number' ? s.narrative_beat_index : undefined,
      shot_size: s.shot_size?.trim() || undefined,
      duration_sec:
        typeof s.duration_sec === 'number' && s.duration_sec > 0
          ? s.duration_sec
          : fallbackDurations[i] ?? opts.defaultSegmentSec,
      status: 'draft' as const,
    };
  });
}

function applyScriptAnalyzeResult(
  result: ChapterAnalyzeApiResult,
  opts: { mergeCharacters?: boolean } = {},
): number {
  const lv = project.value;
  if (!lv || !result?.scene_beats?.length) return 0;

  const scene_beats = result.scene_beats.map((s) => ({
    order: s.order,
    title: s.title || '',
    beat: s.beat || '',
  }));
  const mergeCharacters = opts.mergeCharacters !== false;
  const incomingChars = (result.characters ?? []) as LongVideoCharacter[];
  const mergedChars = mergeCharacters
    ? mergeCharacterRosters(lv.characters ?? [], incomingChars)
    : incomingChars;
  const rosterPatch = hydrateCharacterRoster(
    {
      characters: mergedChars,
      character_anchor: result.character_anchor || lv.character_anchor || '',
      style_anchor: result.style_anchor || lv.style_anchor || '',
    },
    uiLocale.value.startsWith('zh') ? 'zh' : 'en',
  );
  const incomingScenes = (result.scenes ?? []) as LongVideoScene[];
  const mergedScenes = mergeCharacters
    ? mergeSceneRosters(lv.scenes ?? [], incomingScenes)
    : incomingScenes;
  const projectForProvenance = {
    character_anchor: rosterPatch.character_anchor,
    characters: rosterPatch.characters,
    scenes: mergedScenes,
    style_anchor: rosterPatch.style_anchor,
  };
  const parsedShots = result.shots?.length
    ? enrichShotsWithSceneLooks(
        storyboardShotsFromResponse(result.shots, {
          defaultSegmentSec: lv.segment_duration_sec ?? 5,
          targetDurationSec: lv.target_duration_sec ?? 60,
          beatTexts: scene_beats.map((s) => s.beat),
        }),
        mergedScenes,
      )
    : [];
  const provenanceByShot = parsedShots.length
    ? buildParseProvenanceByShot(parsedShots, projectForProvenance)
    : {};
  const parseRunId = result.parse_run_id || lv.chapter_analysis?.parse_run_id || '';
  const parseAt = parseRunId ? new Date().toISOString() : lv.chapter_analysis?.last_parse_at;
  const parseHistory = [...(lv.chapter_analysis?.parse_history ?? [])];
  if (parseRunId && parsedShots.length) {
    parseHistory.push({
      parse_run_id: parseRunId,
      at: parseAt || new Date().toISOString(),
      shot_count: parsedShots.length,
      provenance_by_shot_id: provenanceByShot,
    });
    while (parseHistory.length > 5) parseHistory.shift();
  }
  const analysis: LongVideoChapterAnalysis = {
    synopsis: result.synopsis || '',
    mood: result.mood || '',
    scene_beats,
    character_anchor: rosterPatch.character_anchor,
    style_anchor: rosterPatch.style_anchor,
    characters: rosterPatch.characters,
    scenes: mergedScenes,
    quality_warnings: result.quality_warnings ?? [],
    quality_issues: result.quality_issues ?? [],
    parse_run_id: parseRunId,
    last_parse_at: parseAt,
    parse_phases: result.parse_phases ?? lv.chapter_analysis?.parse_phases,
    shot_t2i_provenance: Object.keys(provenanceByShot).length ? provenanceByShot : lv.chapter_analysis?.shot_t2i_provenance,
    parse_history: parseHistory.length ? parseHistory : lv.chapter_analysis?.parse_history,
  };
  setProjectWithChapterSync({
    ...lv,
    chapter_analysis: analysis,
    character_anchor: rosterPatch.character_anchor,
    characters: rosterPatch.characters,
    scenes: mergedScenes,
    style_anchor: rosterPatch.style_anchor,
    chapter_title: result.chapter_title?.trim() || lv.chapter_title,
    shots:
      parsedShots.length
        ? mergeParsedShotsWithPrevious(lv.shots ?? [], parsedShots)
        : [],
  });
  return scene_beats.length;
}

async function runScriptParsePipeline(strategy: LongVideoParseStrategy) {
  parseStrategyOpen.value = false;
  scriptParseError.value = '';
  scriptParseProgressPhase.value = 'plan';

  if (strategy === 'new_project') {
    if (!(await bootstrapNewProjectForParse())) return;
  }

  const lv = project.value;
  if (!lv) return;
  const text = scriptText.value.trim();
  if (!text) {
    toast.warning($tt('video.longVideoScriptNeedText'));
    return;
  }

  const segmentDurationSec = lv.segment_duration_sec ?? 5;
  const mergeCharacters = strategy === 'replace';

  await ensureProjectSavedForGeneration();
  const projectId = project.value?.project_id ?? lv.project_id ?? '';
  const parseModel = lv.script_parse_llm_model?.trim() || 'qwen3.6-27b';

  const analyzeResult = await analyzeLongVideoChapter(
    {
      chapter_text: text,
      chapter_title: (lv.chapter_title || '').trim(),
      locale: uiLocale.value,
      target_duration_sec: lv.target_duration_sec ?? 60,
      segment_duration_sec: segmentDurationSec,
      max_clip_sec: 10,
      long_video_project_id: projectId,
      model: parseModel,
    },
    {
      quietSuccess: true,
      onProgress: (phase) => {
        scriptParseProgressPhase.value = phase;
      },
    },
  );
  scriptParseProgressPhase.value = '';
  if (!analyzeResult?.scene_beats?.length) {
    if (!analyzeResult) {
      scriptParseError.value = $tt('video.longVideoChapterAnalyzeFailedGeneric');
    } else {
      scriptParseError.value = $tt('video.longVideoChapterAnalyzeEmpty');
    }
    return;
  }

  const sceneCount = applyScriptAnalyzeResult(analyzeResult, { mergeCharacters });
  if (!sceneCount) return;

  const lvAfter = project.value;
  const hasParsedShots = (analyzeResult.shots?.length ?? lvAfter?.shots?.length ?? 0) > 0;
  if (!hasParsedShots) {
    scriptParseError.value = $tt('video.longVideoChapterAnalyzeNoShots');
    return;
  }

  const castCount = analyzeResult.characters?.length ?? lvAfter?.characters?.length ?? 0;
  const shotCount = lvAfter?.shots?.length ?? 0;
  toast.success(
    $tt('video.longVideoScriptPipelineReady', {
      scenes: sceneCount,
      cast: castCount,
      shots: shotCount,
    }),
  );
  const qWarn = analyzeResult.quality_warnings?.length ?? 0;
  if (qWarn > 0) {
    toast.warning(
      $tt('video.longVideoParseQualityWarnings', {
        count: qWarn,
        sample: analyzeResult.quality_warnings?.[0] ?? '',
      }),
    );
  }
  onEditorTabChange('storyboard');
}

function clearSelection() {
  longVideoProject.setSelection(null);
}

function onSelectSegment(index: number) {
  longVideoProject.setSelection({ kind: 'segment', index });
}

function onSelectBeatGroup(groupId: string) {
  longVideoProject.setSelection({ kind: 'beat_group', groupId });
  const idx = shots.value.findIndex((s) => s.segment_group_id === groupId);
  if (idx >= 0) onSelectSegment(idx);
  requestAnimationFrame(() => {
    const el = document.querySelector(`[data-group-id="${CSS.escape(groupId)}"]`);
    el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
}

function onUpdateCharacters(characters: LongVideoCharacter[]) {
  const lv = project.value;
  if (!lv) return;
  const character_anchor = syncRosterToCharacterAnchor(characters, lv.style_anchor ?? '');
  setProjectWithChapterSync({
    ...lv,
    characters,
    character_anchor,
  });
}

function onUpdateStyleAnchor(styleAnchor: string) {
  const lv = project.value;
  if (!lv) return;
  const character_anchor = syncRosterToCharacterAnchor(lv.characters ?? [], styleAnchor);
  setProjectWithChapterSync({
    ...lv,
    style_anchor: styleAnchor,
    character_anchor,
  });
}

function onUpdateVisual(index: number, value: string) {
  const lv = project.value;
  if (!lv) return;
  const scene = value.trim();
  const shotsNext = lv.shots.map((s, i) => {
    if (i !== index) return s;
    const next = { ...s, visual_prompt: scene };
    if (s.segment_role === 'face_anchor') {
      next.anchor_visual_prompt = scene;
    } else {
      next.start_visual_prompt = scene;
    }
    return next;
  });
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onUpdateMotion(index: number, value: string) {
  const lv = project.value;
  if (!lv) return;
  const motion = value;
  const shotsNext = lv.shots.map((s, i) =>
    i === index ? { ...s, motion_prompt: motion, video_prompt: motion } : s,
  );
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onUpdateCastLooks(index: number, castLooks: LongVideoShotCastLook[]) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) => (i === index ? { ...s, cast_looks: castLooks } : s));
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function onUpdateSceneLook(index: number, sceneLook: LongVideoShotSceneLook | undefined) {
  const lv = project.value;
  if (!lv) return;
  const shotsNext = lv.shots.map((s, i) => {
    if (i !== index) return s;
    const next = { ...s };
    if (sceneLook) next.scene_look = sceneLook;
    else delete next.scene_look;
    return next;
  });
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

  const left = shotKeyframeText(shot);
  const right = nextShot ? shotKeyframeText(nextShot) : '';
  const motion = shotVideoPrompt(shot);
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
  const prompt = shot ? shotKeyframeText(shot) : '';
  if (!prompt.trim()) return;

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
    selection: { kind: 'segment', index: shotsNext.length - 1 },
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
    selection: { kind: 'segment', index: newIndex },
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
    selection: { kind: 'segment', index: newIndex },
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
  const keyframeText = shotKeyframeText(shot);
  if (!shotNeedsKeyframe(shot)) {
    toast.warning($tt('video.longVideoSegmentTailFrameNoKeyframe'));
    return;
  }
  if (!keyframeText.trim()) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }

  const scene = shotCastMatchText(shot);
  const castRefIds = collectCastReferenceAssetIdsForShot(
    lv.characters ?? [],
    shot.cast_looks ?? [],
    scene,
  );

  const p = keyframeComposeParams;
  const modelStr = keyframeModelId.value;
  const adapters = mergeKeyframeLoraAdapters(
    p.lora ? [{ id: String(p.lora), weight: Number(p.lora_scale) || 0.8 }] : [],
    lv.characters ?? [],
    shot.cast_looks ?? [],
    scene,
    lv.character_lora_id,
    Number(p.lora_scale) || 0.8,
  );

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

  const hasManualRef = referenceImage.value != null && composeMode.value === 'img2img';
  let source_asset_id: string | null = null;
  if (hasManualRef && referenceImage.value) {
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

  const useImg2img = Boolean(source_asset_id);
  if (composeMode.value === 'img2img' && !source_asset_id) {
    toast.warning($tt('create.refImageNeeded'));
    return;
  }

  const locale = uiLocale.value.startsWith('zh') ? 'zh' : 'en';
  const castNegative = locale === 'zh' ? KEYFRAME_CAST_NEGATIVE_ZH : KEYFRAME_CAST_NEGATIVE_EN;
  const negativePrompt = [p.negative_prompt, castRefIds.length ? castNegative : '']
    .filter(Boolean)
    .join(', ');

  const meta = withProjectMetadata({
    long_video_shot_id: shot.id,
    long_video_phase: 'keyframe',
  });
  if (shot.first_frame_strategy === 't2i_from_grounding' || shot.first_frame_strategy === 'scene_composite') {
    const scene = resolveSceneForShot(lv, shot);
    Object.assign(meta, buildKeyframeGroundingMetadata(shot, scene));
  }
  if (p.scheduler) meta.scheduler = p.scheduler;

  const inpSrc = resolveInpaintAssetId(keyframeInpaintSourceImage.value);
  const inpMsk = resolveInpaintAssetId(keyframeInpaintMaskImage.value);
  const enhCommon = {
    parameters: keyframeModelConfig.value?.parameters as Record<string, unknown> | undefined,
    params: p,
    controlnet: String(p.controlnet || ''),
    controlAssetId: control_asset_id,
    controlnetStrength: Number(p.controlnet_strength) || 0.8,
    inpaintSourceId: inpSrc,
    inpaintMaskId: inpMsk,
  };

  const seedNum = p.seed ? parseInt(String(p.seed), 10) : null;
  const { width, height } = outputSizePixels.value;
  const t2iPrompt = keyframeGenerationPrompt(keyframeText, keyframePromptContextForShot(shot, lv));

  keyframeGeneratingIndex.value = index;
  try {
    await ensureProjectSavedForGeneration();
    let submitRes: unknown;
    if (useImg2img && source_asset_id) {
      const editBody: Record<string, unknown> = {
        model: modelStr,
        operation: 'rewrite',
        source_asset_id,
        prompt: t2iPrompt,
        negative_prompt: negativePrompt || '',
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
        negative_prompt: negativePrompt || '',
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
    const nextWarnings = { ...consistencyWarnings.value };
    delete nextWarnings[index];
    consistencyWarnings.value = nextWarnings;
    try {
      await loadGallery(true);
    } catch {
      /* gallery refresh must not fail the generation flow */
    }
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

function markSegmentPending(index: number) {
  const next = new Set(segmentGeneratingIndices.value);
  next.add(index);
  segmentGeneratingIndices.value = next;
}

function unmarkSegmentPending(index: number) {
  const next = new Set(segmentGeneratingIndices.value);
  next.delete(index);
  segmentGeneratingIndices.value = next;
}

function applySegmentGenerationResult(index: number, shotId: string, assetId: string) {
  const lv = project.value;
  if (!lv?.shots[index] || lv.shots[index].id !== shotId) return;
  const shotsNext = lv.shots.map((s, i) =>
    i === index
      ? { ...s, segment_asset_id: assetId, status: 'segment_ready' as const, error: undefined }
      : s,
  );
  longVideoProject.setProject({ ...lv, shots: shotsNext });
}

function watchSegmentTaskCompletion(taskId: string, index: number, shotId: string) {
  let assetId: string | null = null;
  tasksStore.openTaskLogStream(taskId, {
    onResult: (resultData: unknown) => {
      const ids = (resultData as { asset_ids?: string[] })?.asset_ids || [];
      if (ids.length > 0) {
        assetId = ids[0];
        applySegmentGenerationResult(index, shotId, assetId);
      }
    },
    onDone: async (doneData: unknown) => {
      unmarkSegmentPending(index);
      const data = doneData as { status: string };
      if (data.status === 'completed') {
        if (!assetId) {
          try {
            const task = (await api.gen.getMediaTask(taskId)) as { asset_ids?: string[] };
            const ids = task?.asset_ids || [];
            if (ids[0]) applySegmentGenerationResult(index, shotId, ids[0]);
          } catch {
            /* optional fallback */
          }
        }
        try {
          await loadGallery(true);
        } catch {
          /* gallery refresh must not fail the generation flow */
        }
        toast.success($tt('video.longVideoSegmentDone'));
      } else {
        toast.error($tt('studio.genFailed', { msg: '' }));
      }
    },
    onError: () => {
      unmarkSegmentPending(index);
      toast.error($tt('studio.genFailed', { msg: '' }));
    },
  });
}

function buildSegmentSubmitBody(
  lv: LongVideoProjectState,
  index: number,
): Record<string, unknown> | null {
  const shot = lv.shots[index];
  if (!shot || !canGenerateSegmentShot(lv, index)) return null;
  const sourceAssetId = segmentI2vSourceAssetId(lv, index);
  if (!sourceAssetId) return null;
  const motion = shotVideoPrompt(shot);
  if (!motion) return null;

  const chainMode = effectiveShotChainMode(shot, lv.chain_mode);
  const prevSegId = index > 0 ? lv.shots[index - 1]?.segment_asset_id : undefined;
  const segModel = lv.segment_video_model;
  const p = segmentComposeParams;
  const fps = p.fps || scalarDefault(segModel, 'fps', 16);
  const nfSchema = normalizeParamsDef(modelParameters(segModel)).num_frames as
    | { min?: number; max?: number; step?: number }
    | undefined;
  const cappedDurationSec = Math.min(shotDurationSec(shot), 10);
  const numFrames = numFramesForDurationSec(cappedDurationSec, fps, nfSchema);
  const { width, height } = outputSizePixels.value;
  const body: Record<string, unknown> = {
    model: segModel,
    operation: 'animate',
    source_asset_id: sourceAssetId,
    prompt: motion,
    negative_prompt: p.negative_prompt || '',
    size: `${width}x${height}`,
    num_frames: numFrames,
    fps,
    steps: p.steps || scalarDefault(segModel, 'steps', 40),
    guidance: p.guide_scale ?? scalarDefault(segModel, 'guide_scale', 3.0),
    shift: p.shift || scalarDefault(segModel, 'shift', 12.0) || undefined,
    seed: shot.seed ?? (p.seed ? parseInt(p.seed, 10) : null),
    priority: 'normal',
    metadata: withProjectMetadata({
      long_video_shot_id: shot.id,
      long_video_phase: 'segment',
      long_video_chain_mode: chainMode,
      ...(chainMode === 'last_frame' && prevSegId
        ? { long_video_prev_segment_asset_id: prevSegId }
        : {}),
    }),
  };
  const adapters = buildSegmentAdapters();
  if (adapters.length > 0) {
    body.adapters = adapters;
  }
  if (modelHasBerniniRenderer(segModel) || chainMode === 'reference_r2v') {
    const refIds = collectCastReferenceAssetIdsForShot(
      lv.characters ?? [],
      shot.cast_looks ?? [],
      shotCastMatchText(shot),
    );
    if (refIds.length) {
      body.reference_asset_ids = refIds.slice(0, berniniMaxReferenceImages());
    } else if (chainMode === 'reference_r2v') {
      return null;
    }
  }
  appendActiveEnumFields(
    body,
    p as Record<string, unknown>,
    VIDEO_INFERENCE_ENUM_KEYS,
    normalizeParamsDef(modelParameters(segModel)) as Record<string, unknown> | undefined,
  );
  return body;
}

async function onGenerateSegment(index: number) {
  const lv = project.value;
  if (!lv) return;
  const shot = lv.shots[index];
  if (!shot) return;

  const chainMode = effectiveShotChainMode(shot, lv.chain_mode);
  const prevSegId = index > 0 ? lv.shots[index - 1]?.segment_asset_id : undefined;
  if (chainMode === 'last_frame' && index > 0 && !prevSegId) {
    toast.warning($tt('video.longVideoSegmentChainNeedPrevSegment'));
    return;
  }
  if (!canGenerateSegmentShot(lv, index)) {
    if (!shotVideoPrompt(shot)) {
      toast.warning($tt('video.longVideoNeedMotionForSegment'));
    } else if (shot.start_frame_mode === 'anchor_link' || shot.segment_role === 'post_anchor') {
      toast.warning($tt('video.longVideoNeedAnchorForSegment'));
    } else {
      toast.warning($tt('video.longVideoNeedKeyframeForSegment'));
    }
    return;
  }
  if (segmentGeneratingIndices.value.has(index)) {
    toast.warning($tt('video.longVideoSegmentAlreadyQueued'));
    return;
  }

  await ensureProjectSavedForGeneration();
  const lvAfterSave = project.value ?? lv;
  const body = buildSegmentSubmitBody(lvAfterSave, index);
  if (!body) return;

  const shotId = shot.id;
  markSegmentPending(index);
  try {
    const submitRes = await api.gen.createVideoEdit(body);
    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) throw new Error('no task id');
    watchSegmentTaskCompletion(tid, index, shotId);
    tasksStore.pollQueueOnce();
    openGlobalTaskQueue();
    toast.success($tt('video.longVideoSegmentQueued'));
  } catch (e: unknown) {
    unmarkSegmentPending(index);
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.genFailed', { msg }));
  }
}

function waitForVideoTask(tid: string): Promise<{ assetId: string | null }> {
  return new Promise((resolve, reject) => {
    let assetId: string | null = null;
    api.gen.streamMediaTask(tid, {
      onResult: (resultData: unknown) => {
        const ids = (resultData as { asset_ids?: string[] })?.asset_ids || [];
        if (ids.length > 0) assetId = ids[0];
      },
      onDone: async (doneData: unknown) => {
        tasksStore.unregisterPageOwnedStream(tid);
        const data = doneData as { status: string };
        if (data.status === 'completed') {
          if (!assetId) {
            try {
              const task = (await api.gen.getMediaTask(tid)) as { asset_ids?: string[] };
              if (task?.asset_ids?.[0]) assetId = task.asset_ids[0];
            } catch {
              /* optional fallback */
            }
          }
          resolve({ assetId });
        } else {
          reject(new Error('segment generation failed'));
        }
      },
      onError: () => {
        tasksStore.unregisterPageOwnedStream(tid);
        reject(new Error('connection lost'));
      },
    });
  });
}

async function generateSegmentAndWait(
  index: number,
  opts: { openQueue?: boolean; toastOnDone?: boolean } = {},
): Promise<boolean> {
  const lv = project.value;
  if (!lv) return false;
  const shot = lv.shots[index];
  if (!shot) return false;

  const chainMode = effectiveShotChainMode(shot, lv.chain_mode);
  const prevSegId = index > 0 ? lv.shots[index - 1]?.segment_asset_id : undefined;
  if (chainMode === 'last_frame' && index > 0 && !prevSegId) {
    toast.warning($tt('video.longVideoSegmentChainNeedPrevSegment'));
    return false;
  }
  if (!canGenerateSegmentShot(lv, index)) {
    if (!shotVideoPrompt(shot)) {
      toast.warning($tt('video.longVideoNeedMotionForSegment'));
    } else if (
      shot.start_frame_mode === 'anchor_link' ||
      shot.segment_role === 'post_anchor'
    ) {
      toast.warning($tt('video.longVideoNeedAnchorForSegment'));
    } else {
      toast.warning($tt('video.longVideoNeedKeyframeForSegment'));
    }
    return false;
  }
  if (segmentGeneratingIndices.value.has(index)) {
    toast.warning($tt('video.longVideoSegmentAlreadyQueued'));
    return false;
  }

  await ensureProjectSavedForGeneration();
  const lvAfterSave = project.value ?? lv;
  const body = buildSegmentSubmitBody(lvAfterSave, index);
  if (!body) return false;

  const shotId = shot.id;
  markSegmentPending(index);
  try {
    const submitRes = await api.gen.createVideoEdit(body);
    const tid = taskIdFromSubmitResponse(submitRes);
    if (!tid) throw new Error('no task id');
    tasksStore.registerPageOwnedStream(tid);
    if (opts.openQueue) {
      tasksStore.pollQueueOnce();
      openGlobalTaskQueue();
    }
    const { assetId } = await waitForVideoTask(tid);
    unmarkSegmentPending(index);
    if (assetId) {
      applySegmentGenerationResult(index, shotId, assetId);
      try {
        await loadGallery(true);
      } catch {
        /* gallery refresh must not fail the generation flow */
      }
      if (opts.toastOnDone) toast.success($tt('video.longVideoSegmentDone'));
      return true;
    }
    toast.error($tt('studio.genFailed', { msg: '' }));
    return false;
  } catch (e: unknown) {
    unmarkSegmentPending(index);
    const msg = e instanceof Error ? e.message : String(e);
    if (opts.toastOnDone || opts.openQueue) toast.error($tt('studio.genFailed', { msg }));
    return false;
  }
}

async function runBatchPlan(keyframeIndices: number[], segmentIndices: number[]) {
  if (batchGroupGenerating.value) return;
  if (!keyframeIndices.length && !segmentIndices.length) {
    toast.info($tt('video.longVideoBatchNothingToDo'));
    return;
  }
  batchGroupGenerating.value = true;
  openGlobalTaskQueue();
  let ok = 0;
  let fail = 0;
  try {
    for (const idx of keyframeIndices) {
      await onGenerateKeyframe(idx);
      if (project.value?.shots[idx]?.keyframe_asset_id) ok += 1;
      else fail += 1;
    }
    for (const idx of segmentIndices) {
      const success = await generateSegmentAndWait(idx, { openQueue: false, toastOnDone: false });
      if (success) ok += 1;
      else fail += 1;
    }
    if (fail === 0) {
      toast.success($tt('video.batchSubmitted', { count: ok }));
    } else {
      toast.warning($tt('video.longVideoBatchPartial', { ok, fail }));
    }
  } finally {
    batchGroupGenerating.value = false;
    await loadGallery(true);
  }
}

async function onBatchGenerateCurrentGroup() {
  const lv = project.value;
  if (!lv) return;
  const groupId = selectedBeatGroupId(lv.shots, selection.value);
  if (!groupId) {
    toast.warning($tt('video.longVideoBatchSelectGroup'));
    return;
  }
  const group = groupShotsByBeat(lv.shots).find((g) => g.groupId === groupId);
  if (!group) return;
  const plan = planGroupGeneration(lv, group);
  await runBatchPlan(plan.keyframeIndices, plan.segmentIndices);
}

async function onBatchGenerateAllAnchors() {
  const lv = project.value;
  if (!lv) return;
  const indices = allPendingAnchorKeyframeIndices(lv.shots);
  await runBatchPlan(indices, []);
}

async function onBatchGenerateAllSegments() {
  const lv = project.value;
  if (!lv) return;
  const indices = allPendingSegmentIndices(lv);
  await runBatchPlan([], indices);
}

function onInsertFaceAnchor(groupId: string) {
  const lv = project.value;
  if (!lv) return;
  if (groupHasFaceAnchor(lv.shots, groupId)) {
    toast.warning($tt('video.longVideoAnchorAlreadyExists'));
    return;
  }
  const shotsNext = insertFaceAnchorIntoGroup(lv.shots, groupId);
  if (shotsNext === lv.shots) {
    toast.warning($tt('video.longVideoInsertAnchorFailed'));
    return;
  }
  const firstIdx = shotsNext.findIndex((s) => s.segment_group_id === groupId);
  longVideoProject.setProject({
    ...lv,
    shots: shotsNext,
    selection: firstIdx >= 0 ? { kind: 'segment', index: firstIdx } : lv.selection,
  });
  toast.success($tt('video.longVideoInsertAnchorDone'));
}

async function onResplitBeatGroup(groupId: string) {
  const lv = project.value;
  if (!lv) return;
  try {
    await confirm($tt('video.longVideoResplitBeatConfirm'), $tt('video.longVideoResplitBeat'), {
      type: 'warning',
    });
  } catch {
    return;
  }
  const shotsNext = resplitBeatGroupRule(lv.shots, groupId);
  const firstIdx = shotsNext.findIndex((s) => s.segment_group_id === groupId);
  longVideoProject.setProject({
    ...lv,
    shots: shotsNext,
    selection: firstIdx >= 0 ? { kind: 'segment', index: firstIdx } : null,
  });
  toast.success($tt('video.longVideoResplitBeatDone'));
}

async function loadProjectList() {
  projectsLoading.value = true;
  try {
    savedProjects.value = await api.longVideo.listProjects(100);
  } catch {
    savedProjects.value = [];
    toast.warning($tt('video.longVideoProjectListFailed'));
  } finally {
    projectsLoading.value = false;
  }
}

function applyServerProject(
  detail: { id: string; title: string; state: Partial<LongVideoProjectState> },
  opts?: { silent?: boolean },
) {
  runWithAutoSaveSuppressed(() => {
    const base = hydrateCharacterRoster(
      defaultLongVideoProject(detail.state),
      uiLocale.value.startsWith('zh') ? 'zh' : 'en',
    );
    setProjectWithChapterSync({
      ...base,
      project_id: detail.id,
      title: detail.title || detail.state.title,
      selection: null,
    });
  });
  syncOutputSizeForSegmentModel(detail.state.segment_video_model ?? defaultSegmentModel.value);
  resetKeyframeComposeDefaults();
  resetSegmentComposeDefaults();
  referenceImage.value = null;
  keyframeControlImage.value = null;
  keyframeInpaintSourceImage.value = null;
  keyframeInpaintMaskImage.value = null;
  consistencyWarnings.value = {};
  if (!opts?.silent) {
    toast.success($tt('video.longVideoProjectLoaded'));
  }
}

function runWithAutoSaveSuppressed(fn: () => void) {
  suppressAutoSave.value += 1;
  fn();
  void nextTick(() => {
    suppressAutoSave.value = Math.max(0, suppressAutoSave.value - 1);
  });
}

async function persistProjectToServer(opts: { silent?: boolean } = {}): Promise<boolean> {
  const lv = project.value;
  if (!lv) return false;
  if (!lv.project_id && !longVideoHasPersistableContent(lv)) {
    if (!opts.silent) {
      toast.warning($tt('video.longVideoSaveNothing'));
    }
    return false;
  }
  if (savingProject.value) return false;
  savingProject.value = true;
  try {
    const title = lv.title?.trim() || $tt('video.longVideoPageTitle');
    const state = projectStateForServer({ ...lv, title });
    if (lv.project_id) {
      const saved = await api.longVideo.updateProject(lv.project_id, { title, state });
      runWithAutoSaveSuppressed(() => {
        patchProjectField('title', saved.title);
      });
    } else {
      const created = await api.longVideo.createProject({ title, state });
      runWithAutoSaveSuppressed(() => {
        longVideoProject.patchProject({ project_id: created.id, title: created.title });
      });
      if (!opts.silent) {
        toast.success($tt('video.longVideoProjectCreated'));
      }
    }
    longVideoProject.persistNow();
    await loadProjectList();
    if (!opts.silent && lv.project_id) {
      toast.success($tt('video.longVideoSaveProjectOk'));
    }
    return true;
  } catch (e: unknown) {
    if (!opts.silent) {
      const msg = e instanceof Error ? e.message : String(e);
      toast.error($tt('studio.error', { msg }));
    }
    return false;
  } finally {
    savingProject.value = false;
  }
}

const { status: autoSaveStatus, cancelPending: cancelAutoSavePending } = useLongVideoAutoSave({
  project,
  suppress: suppressAutoSave,
  hasPersistableContent: longVideoHasPersistableContent,
  save: persistProjectToServer,
});

const saveStatusLabel = computed(() => {
  const lv = project.value;
  if (!lv) return '';
  if (!lv.project_id) {
    if (autoSaveStatus.value === 'pending' || autoSaveStatus.value === 'saving') {
      return $tt('video.longVideoAutoSaveCreating');
    }
    if (longVideoHasPersistableContent(lv)) {
      return $tt('video.longVideoAutoSaveWillCreate');
    }
    return $tt('video.longVideoLocalDraftHint');
  }
  switch (autoSaveStatus.value) {
    case 'pending':
      return $tt('video.longVideoAutoSavePending');
    case 'saving':
      return $tt('video.longVideoAutoSaveSaving');
    case 'saved':
      return $tt('video.longVideoAutoSaveSaved');
    case 'error':
      return $tt('video.longVideoAutoSaveError');
    default:
      return '';
  }
});

const saveStatusClass = computed(() => {
  if (!project.value?.project_id) return 'is-local';
  if (autoSaveStatus.value === 'error') return 'is-error';
  if (autoSaveStatus.value === 'saved') return 'is-saved';
  return '';
});

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

function resetToLocalDraft() {
  runWithAutoSaveSuppressed(() => {
    longVideoProject.setProject(
      defaultLongVideoProject({
        keyframe_model: defaultKeyframeModel.value,
        segment_video_model: defaultSegmentModel.value,
      }),
    );
  });
  syncOutputSizeForSegmentModel(defaultSegmentModel.value);
  resetKeyframeComposeDefaults();
  resetSegmentComposeDefaults();
  referenceImage.value = null;
  keyframeControlImage.value = null;
  keyframeInpaintSourceImage.value = null;
  keyframeInpaintMaskImage.value = null;
  consistencyWarnings.value = {};
}

async function deleteSavedProject(projectId: string) {
  try {
    await confirm(
      $tt('video.longVideoProjectDeleteConfirm'),
      $tt('common.delete'),
      { type: 'warning' },
    );
  } catch {
    return;
  }
  cancelAutoSavePending();
  const isActive = project.value?.project_id === projectId;
  try {
    await api.longVideo.deleteProject(projectId);
    if (isActive) {
      resetToLocalDraft();
    }
    await loadProjectList();
    toast.success($tt('video.longVideoProjectDeleted'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
  }
}

async function saveProject() {
  cancelAutoSavePending();
  await persistProjectToServer({ silent: false });
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
    const meta = withProjectMetadata({
      source: 'long_video_create',
      long_video_phase: 'assemble_only',
    });
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
      metadata: meta,
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
      patchProjectField('project_id', undefined);
    }
  } else {
    runWithAutoSaveSuppressed(() => {
      longVideoProject.initProject({
        keyframe_model: defaultKeyframeModel.value,
        segment_video_model: defaultSegmentModel.value,
        title: project.value?.title ?? '',
      });
    });
  }
  syncOutputSizeForSegmentModel(segmentModelId.value);
  await loadGallery(true);
});

watch(segmentModelId, (modelId) => {
  syncOutputSizeForSegmentModel(modelId);
});
</script>
