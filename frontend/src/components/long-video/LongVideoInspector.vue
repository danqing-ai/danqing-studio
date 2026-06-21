<template>
  <div class="lv-inspector">
    <!-- Node -->
    <div v-if="selection?.kind === 'node' && selectedShot" class="lv-inspector__body">
      <section class="lv-inspector__canvas lv-inspector__canvas--sticky">
        <div class="lv-inspector__media-head">
          <div v-if="outputSizeLabel" class="lv-inspector__meta">
            <span class="lv-inspector__meta-size">{{ outputSizeLabel }}</span>
          </div>
          <div class="lv-inspector__tools">
            <button
              type="button"
              class="lv-inspector__tool"
              :disabled="keyframeGenerating"
              :title="$tt('video.longVideoImportFromGallery')"
              @click="$emit('pick-keyframe-gallery', selection.index)"
            >
              <DqIcon :size="14"><PictureFilled /></DqIcon>
              <span>{{ $tt('video.longVideoGalleryShort') }}</span>
            </button>
            <button
              v-if="selectedShot.keyframe_asset_id"
              type="button"
              class="lv-inspector__tool lv-inspector__tool--danger"
              :disabled="keyframeGenerating"
              :title="$tt('video.longVideoClearKeyframe')"
              @click="$emit('clear-keyframe', selection.index)"
            >
              <DqIcon :size="14"><Delete /></DqIcon>
              <span>{{ $tt('video.longVideoClearShort') }}</span>
            </button>
          </div>
        </div>

        <div
          v-if="previewUrl || keyframeGenerating"
          class="lv-inspector__preview"
          :class="{ 'is-generating': keyframeGenerating }"
        >
          <img v-if="previewUrl" class="lv-inspector__img" :src="previewUrl" alt="" />
          <div v-if="keyframeGenerating" class="lv-inspector__loading" aria-live="polite">
            <DqIcon :size="24"><Loading /></DqIcon>
          </div>
        </div>
      </section>

      <LongVideoKeyframeComposePanel
        :prompt="shotSceneText"
        :model="composeModel"
        :mode="composeMode ?? 'text2img'"
        :params="composeParams"
        :generating="keyframeGenerating"
        :can-generate="canGenerateKeyframe"
        :styles="composeStyles"
        :show-negative-prompt="composeShowNegativePrompt"
        :reference-image="referenceImage"
        :control-image="controlImage"
        :inpaint-source-image="inpaintSourceImage"
        :inpaint-mask-image="inpaintMaskImage"
        :current-model-config="composeModelConfig"
        :compatible-loras="compatibleLoras"
        :compatible-control-nets="compatibleControlNets"
        :control-net-runtime-available="controlNetRuntimeAvailable"
        :enhancing="visualPolishing"
        @update:prompt="$emit('update-visual', selection.index, $event)"
        @update:mode="$emit('update-compose-mode', $event)"
        @generate="$emit('generate-keyframe', selection.index)"
        @pick-reference="$emit('pick-reference')"
        @remove-reference="$emit('remove-reference')"
        @pick-control="$emit('pick-control')"
        @remove-control="$emit('remove-control')"
        @pick-inpaint-source="$emit('pick-inpaint-source')"
        @remove-inpaint-source="$emit('remove-inpaint-source')"
        @pick-inpaint-mask="$emit('pick-inpaint-mask')"
        @remove-inpaint-mask="$emit('remove-inpaint-mask')"
        @reset-defaults="$emit('reset-compose-defaults')"
        @enhance="$emit('polish-visual', selection.index)"
      >
        <template #after-prompt>
          <LongVideoCastLookPanel
            v-if="characters.length"
            :characters="characters"
            :cast-looks="selectedShot.cast_looks ?? []"
            :scene-text="shotSceneText"
            :select-id-prefix="`lv-cast-${selection.index}`"
            @update:cast-looks="$emit('update-cast-looks', selection.index, $event)"
          />
          <p v-else class="lv-inspector__cast-empty">{{ $tt('video.longVideoCastLooksNeedRoster') }}</p>
        </template>
      </LongVideoKeyframeComposePanel>

      <footer class="lv-inspector__footer lv-inspector__footer--node">
        <p v-if="selection.index > 0" class="lv-inspector__footer-hint">
          {{ $tt('video.longVideoInsertKeyframeHint') }}
        </p>
        <DqButton
          v-if="selection.index === 0"
          type="default"
          block
          @click="$emit('insert-keyframe-before', selection.index)"
        >
          {{ $tt('video.longVideoInsertKeyframeBeforeFirst') }}
        </DqButton>
        <DqButton
          type="text"
          block
          class="lv-inspector__danger-btn"
          :disabled="!canRemoveKeyframe"
          :title="canRemoveKeyframe ? '' : $tt('video.longVideoRemoveKeyframeMin')"
          @click="$emit('remove-keyframe', selection.index)"
        >
          {{ $tt('video.longVideoRemoveKeyframe') }}
        </DqButton>
      </footer>
    </div>

    <!-- Edge (segment between keyframes) -->
    <div v-else-if="selection?.kind === 'edge' && edgeShot" class="lv-inspector__body">
      <section class="lv-inspector__canvas lv-inspector__canvas--sticky">
        <div class="lv-inspector__media-head">
          <div class="lv-inspector__meta">
            <span class="lv-inspector__meta-range">{{ edgeRangeLabel }}</span>
            <span class="lv-inspector__meta-dot" aria-hidden="true">·</span>
            <span class="lv-inspector__meta-size">{{ edgeDurationLabel }}</span>
            <template v-if="segmentModelLabel">
              <span class="lv-inspector__meta-dot" aria-hidden="true">·</span>
              <span class="lv-inspector__meta-size">{{ segmentModelLabel }}</span>
            </template>
          </div>
          <div class="lv-inspector__tools">
            <button
              v-if="edgeShot.segment_asset_id"
              type="button"
              class="lv-inspector__tool lv-inspector__tool--danger"
              :disabled="segmentGenerating"
              :title="$tt('video.longVideoClearSegment')"
              @click="$emit('clear-segment', selection.index)"
            >
              <DqIcon :size="14"><Delete /></DqIcon>
              <span>{{ $tt('video.longVideoClearShort') }}</span>
            </button>
          </div>
        </div>

        <div
          class="lv-inspector__preview"
          :class="{ 'is-generating': segmentGenerating, 'is-empty': !edgeSegmentVideoUrl && !segmentGenerating }"
        >
          <video
            v-if="edgeSegmentVideoUrl"
            class="lv-inspector__segment-video"
            :src="edgeSegmentVideoUrl"
            controls
            playsinline
            preload="metadata"
          />
          <div v-else-if="!segmentGenerating" class="lv-inspector__edge-preview">
            <button
              type="button"
              class="lv-inspector__edge-thumb"
              :title="$tt('video.longVideoKeyframeEdit', { n: selection.index + 1 })"
              @click="$emit('select-node', selection.index)"
            >
              <img v-if="edgeFromUrl" :src="edgeFromUrl" alt="" />
              <span v-else class="lv-inspector__edge-ph">#{{ selection.index + 1 }}</span>
            </button>
            <span class="lv-inspector__edge-arrow" aria-hidden="true">→</span>
            <button
              type="button"
              class="lv-inspector__edge-thumb"
              :title="$tt('video.longVideoKeyframeEdit', { n: selection.index + 2 })"
              @click="$emit('select-node', selection.index + 1)"
            >
              <img v-if="edgeToUrl" :src="edgeToUrl" alt="" />
              <span v-else class="lv-inspector__edge-ph">#{{ selection.index + 2 }}</span>
            </button>
          </div>
          <div v-if="segmentGenerating" class="lv-inspector__loading" aria-live="polite">
            <DqIcon :size="24"><Loading /></DqIcon>
          </div>
        </div>
      </section>

      <LongVideoSegmentComposePanel
        :motion-prompt="edgeShot.motion_prompt"
        :duration-sec="edgeDurationSec"
        :duration-options="segmentDurationOptions"
        :chain-mode="edgeChainMode"
        :can-use-last-frame="edgeCanUseLastFrame"
        :params="segmentComposeParams"
        :param-schema="segmentParamSchema"
        :num-frames="edgeNumFrames"
        :generating="segmentGenerating"
        :can-generate="canGenerateEdgeSegment"
        :can-polish="canPolishMotion"
        :polishing="motionPolishing"
        :missing-keyframe="!edgeShot.keyframe_asset_id"
        :show-negative-prompt="segmentShowNegativePrompt"
        :show-seed-field="segmentShowSeedField"
        :show-lora="segmentShowLora"
        :compatible-loras="segmentCompatibleLoras"
        @update:motion="$emit('update-motion', selection.index, $event)"
        @update:duration="$emit('update-duration', selection.index, $event)"
        @update:chain-mode="$emit('update-chain-mode', selection.index, $event)"
        @generate="$emit('generate-segment', selection.index)"
        @polish="$emit('polish-motion', selection.index)"
        @reset-defaults="$emit('reset-segment-defaults')"
      />

      <footer class="lv-inspector__footer lv-inspector__footer--edge">
        <DqButton
          type="text"
          block
          class="lv-inspector__secondary-btn"
          @click="$emit('insert-keyframe-after', selection.index)"
        >
          {{ $tt('video.longVideoInsertKeyframeHere', { from: selection.index + 1, to: selection.index + 2 }) }}
        </DqButton>
      </footer>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { Delete, Loading, PictureFilled } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import { keyframeThumbnailUrl, MIN_LONG_VIDEO_KEYFRAMES, numFramesForDurationSec, shotDurationSec, effectiveShotChainMode, extractKeyframeShotScene } from '@/utils/longVideoProject';
import LongVideoCastLookPanel from './LongVideoCastLookPanel.vue';
import LongVideoKeyframeComposePanel from './LongVideoKeyframeComposePanel.vue';
import LongVideoSegmentComposePanel from './LongVideoSegmentComposePanel.vue';
import type { KeyframeComposeParams } from '@/composables/useLongVideoKeyframeCompose';
import type { SegmentComposeParams } from '@/composables/useLongVideoSegmentCompose';
import type { LongVideoChainMode, LongVideoCharacter, LongVideoSelection, LongVideoShotCastLook, LongVideoShotState } from '@/types';
import type { NormalizedParamSpec } from '@/utils/registryParamSchema';

const props = defineProps<{
  shots: LongVideoShotState[];
  selection: LongVideoSelection;
  segmentDurationOptions: number[];
  segmentModelLabel?: string;
  segmentComposeParams: SegmentComposeParams;
  segmentParamSchema: Record<string, NormalizedParamSpec>;
  segmentCompatibleLoras?: Record<string, unknown>[];
  segmentShowNegativePrompt?: boolean;
  segmentShowSeedField?: boolean;
  segmentShowLora?: boolean;
  defaultChainMode: LongVideoChainMode;
  keyframeGenerating?: boolean;
  segmentGenerating?: boolean;
  visualPolishing?: boolean;
  motionPolishing?: boolean;
  outputSizeLabel?: string;
  composeModel: string;
  composeParams: KeyframeComposeParams;
  composeStyles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; trigger_words?: string; media_scope?: string }>;
  composeShowNegativePrompt?: boolean;
  composeMode?: string;
  composeModelConfig?: Record<string, unknown> | null;
  compatibleLoras?: Record<string, unknown>[];
  compatibleControlNets?: Record<string, unknown>[];
  controlNetRuntimeAvailable?: boolean;
  referenceImage: { previewUrl: string; path: string } | null;
  controlImage?: { previewUrl: string; path: string } | null;
  inpaintSourceImage?: { previewUrl: string; path: string } | null;
  inpaintMaskImage?: { previewUrl: string; path: string } | null;
  canGenerateKeyframe?: boolean;
  characters?: LongVideoCharacter[];
}>();

defineEmits<{
  (e: 'update-visual', index: number, value: string): void;
  (e: 'update-cast-looks', index: number, value: LongVideoShotCastLook[]): void;
  (e: 'update-motion', index: number, value: string): void;
  (e: 'update-duration', index: number, value: number): void;
  (e: 'update-chain-mode', index: number, value: LongVideoChainMode): void;
  (e: 'update-compose-mode', value: string): void;
  (e: 'reset-compose-defaults'): void;
  (e: 'reset-segment-defaults'): void;
  (e: 'generate-keyframe', index: number): void;
  (e: 'generate-segment', index: number): void;
  (e: 'pick-keyframe-gallery', index: number): void;
  (e: 'clear-keyframe', index: number): void;
  (e: 'clear-segment', index: number): void;
  (e: 'insert-keyframe-before', index: number): void;
  (e: 'insert-keyframe-after', index: number): void;
  (e: 'remove-keyframe', index: number): void;
  (e: 'select-node', index: number): void;
  (e: 'polish-visual', index: number): void;
  (e: 'polish-motion', index: number): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'pick-control'): void;
  (e: 'remove-control'): void;
  (e: 'pick-inpaint-source'): void;
  (e: 'remove-inpaint-source'): void;
  (e: 'pick-inpaint-mask'): void;
  (e: 'remove-inpaint-mask'): void;
}>();

const { t: $tt } = useI18n();

const canRemoveKeyframe = computed(() => props.shots.length > MIN_LONG_VIDEO_KEYFRAMES);

const selectedShot = computed(() => {
  if (props.selection?.kind !== 'node') return null;
  return props.shots[props.selection.index] ?? null;
});

const characters = computed(() => props.characters ?? []);

const shotSceneText = computed(() => {
  if (!selectedShot.value) return '';
  return (
    selectedShot.value.scene_prompt ||
    extractKeyframeShotScene(selectedShot.value.visual_prompt)
  ).trim();
});

const edgeShot = computed(() => {
  if (props.selection?.kind !== 'edge') return null;
  return props.shots[props.selection.index] ?? null;
});

const edgeDurationSec = computed(() => shotDurationSec(edgeShot.value ?? undefined));

const edgeChainMode = computed(() =>
  effectiveShotChainMode(edgeShot.value ?? undefined, props.defaultChainMode),
);

const edgeCanUseLastFrame = computed(() => {
  if (props.selection?.kind !== 'edge') return false;
  const idx = props.selection.index;
  return idx > 0 && Boolean(props.shots[idx - 1]?.segment_asset_id);
});

const edgeNumFrames = computed(() => {
  if (props.selection?.kind !== 'edge') return 0;
  const schema = props.segmentParamSchema.num_frames;
  return numFramesForDurationSec(
    edgeDurationSec.value,
    props.segmentComposeParams.fps,
    schema as { min?: number; max?: number; step?: number } | undefined,
  );
});

const edgeDurationLabel = computed(() =>
  $tt('video.longVideoSegmentDurationSec', { sec: edgeDurationSec.value }),
);

const edgeRangeLabel = computed(() => {
  if (props.selection?.kind !== 'edge') return '';
  const idx = props.selection.index;
  return `#${idx + 1} → #${idx + 2}`;
});

const edgeSegmentVideoUrl = computed(() => {
  const id = edgeShot.value?.segment_asset_id;
  return id ? `/api/assets/${id}/file` : '';
});

const canPolishMotion = computed(() => {
  if (props.selection?.kind !== 'edge') return false;
  const idx = props.selection.index;
  const left = props.shots[idx]?.visual_prompt?.trim();
  const right = props.shots[idx + 1]?.visual_prompt?.trim();
  const motion = edgeShot.value?.motion_prompt?.trim();
  return Boolean(motion || left || right);
});

const canGenerateEdgeSegment = computed(() => {
  if (props.selection?.kind !== 'edge') return false;
  const idx = props.selection.index;
  const shot = props.shots[idx];
  if (!shot?.keyframe_asset_id) return false;
  const motion = (shot.motion_prompt || shot.visual_prompt || '').trim();
  return Boolean(motion);
});

function thumbForAsset(assetId: string | undefined) {
  if (!assetId) return '';
  return keyframeThumbnailUrl(assetId, (p) => api.gallery.getImageUrl(p));
}

const edgeFromUrl = computed(() => {
  if (props.selection?.kind !== 'edge') return '';
  return thumbForAsset(props.shots[props.selection.index]?.keyframe_asset_id);
});

const edgeToUrl = computed(() => {
  if (props.selection?.kind !== 'edge') return '';
  return thumbForAsset(props.shots[props.selection.index + 1]?.keyframe_asset_id);
});

const previewUrl = computed(() => {
  if (!selectedShot.value?.keyframe_asset_id) return '';
  return keyframeThumbnailUrl(selectedShot.value.keyframe_asset_id, (p) => api.gallery.getImageUrl(p));
});
</script>

<style scoped>
.lv-inspector {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.lv-inspector__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 4px 2px 12px;
}

.lv-inspector__cast-empty {
  margin: -4px 0 0;
  font-size: 11px;
  line-height: 1.4;
  color: var(--dq-label-tertiary);
}

.lv-inspector__canvas {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.lv-inspector__canvas--sticky {
  position: sticky;
  top: 0;
  z-index: 2;
  padding-bottom: 8px;
  background: var(--dq-surface-base, var(--dq-bg));
}

.lv-inspector__media-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.lv-inspector__preview {
  position: relative;
  aspect-ratio: 16 / 9;
  border-radius: 10px;
  overflow: hidden;
  background: #000;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  box-shadow: inset 0 1px 0 color-mix(in srgb, white 4%, transparent);
}

.lv-inspector__preview.is-empty {
  background: color-mix(in srgb, var(--dq-surface-elevated) 40%, #000);
}

.lv-inspector__img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  border: none;
  outline: none;
}

.lv-inspector__segment-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.lv-inspector__edge-preview {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  height: 100%;
  padding: 14px;
  box-sizing: border-box;
}

.lv-inspector__edge-thumb {
  flex: 1;
  max-width: 42%;
  aspect-ratio: 16 / 9;
  border-radius: 8px;
  overflow: hidden;
  padding: 0;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-surface-elevated) 50%, transparent);
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
}

.lv-inspector__edge-thumb:hover {
  border-color: color-mix(in srgb, var(--dq-accent) 45%, var(--dq-border-subtle));
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 12%, transparent);
}

.lv-inspector__edge-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-inspector__edge-ph {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  font-size: 11px;
  font-weight: 600;
  color: var(--dq-label-tertiary);
}

.lv-inspector__edge-arrow {
  flex-shrink: 0;
  font-size: 14px;
  font-weight: 700;
  color: var(--dq-label-tertiary);
}

.lv-inspector__tools {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.lv-inspector__tool {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 28px;
  padding: 0 10px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  border-radius: 7px;
  background: color-mix(in srgb, var(--dq-glass-floating-bar-bg, var(--dq-surface-elevated)) 85%, transparent);
  color: var(--dq-label-secondary);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}

.lv-inspector__tool:hover:not(:disabled) {
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-surface-elevated));
  color: var(--dq-accent);
}

.lv-inspector__tool:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.lv-inspector__tool--danger:hover:not(:disabled) {
  background: color-mix(in srgb, var(--dq-danger) 10%, var(--dq-surface-elevated));
  color: var(--dq-danger);
}

.lv-inspector__loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--dq-surface-base) 62%, transparent);
  backdrop-filter: var(--dq-glass-blur-light);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  color: var(--dq-accent);
}

.lv-inspector__preview.is-generating .lv-inspector__segment-video,
.lv-inspector__preview.is-generating .lv-inspector__edge-preview {
  opacity: 0.35;
}

.lv-inspector__meta {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  flex-wrap: wrap;
}

.lv-inspector__meta-range {
  font-size: 11px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--dq-label-secondary);
}

.lv-inspector__meta-dot {
  font-size: 11px;
  color: var(--dq-label-tertiary);
}

.lv-inspector__meta-size {
  font-size: 11px;
  color: var(--dq-label-tertiary);
}

.lv-inspector__footer {
  margin-top: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 12px;
  border-top: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
}

.lv-inspector__footer-hint {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--dq-label-tertiary);
  text-align: center;
}

.lv-inspector__danger-btn,
.lv-inspector__secondary-btn {
  color: var(--dq-label-tertiary) !important;
}

.lv-inspector__danger-btn:hover:not(:disabled) {
  color: var(--dq-danger) !important;
  background: color-mix(in srgb, var(--dq-danger) 8%, transparent) !important;
}

.lv-inspector__secondary-btn:hover:not(:disabled) {
  color: var(--dq-accent) !important;
  background: color-mix(in srgb, var(--dq-accent) 8%, transparent) !important;
}

.lv-inspector__danger-btn:disabled {
  opacity: 0.4;
}
</style>
