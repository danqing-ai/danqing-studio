<template>
  <DqDrawer
    :open="open"
    :title="panelTitle"
    direction="rtl"
    size="440px"
    class="lv-inspector-drawer"
    @update:open="$emit('update:open', $event)"
  >
    <LongVideoInspector
      :shots="shots"
      :selection="selection"
      :segment-duration-options="segmentDurationOptions"
      :keyframe-generating="keyframeGenerating"
      :segment-generating="segmentGenerating"
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
      :default-chain-mode="defaultChainMode"
      :compose-model="composeModel"
      :compose-params="composeParams"
      :compose-styles="composeStyles"
      :compose-show-negative-prompt="composeShowNegativePrompt"
      :compose-mode="composeMode"
      :compose-model-config="composeModelConfig"
      :compatible-loras="compatibleLoras"
      :compatible-control-nets="compatibleControlNets"
      :control-net-runtime-available="controlNetRuntimeAvailable"
      :reference-image="referenceImage"
      :control-image="controlImage"
      :inpaint-source-image="inpaintSourceImage"
      :inpaint-mask-image="inpaintMaskImage"
      :can-generate-keyframe="canGenerateKeyframe"
      :characters="characters"
      :scenes="scenes"
      :character-anchor="characterAnchor"
      :style-anchor="styleAnchor"
      :project-id="projectId"
      :parse-run-id="parseRunId"
      @update-visual="(i, v) => $emit('update-visual', i, v)"
      @update-cast-looks="(i, v) => $emit('update-cast-looks', i, v)"
      @update-scene-look="(i, v) => $emit('update-scene-look', i, v)"
      @update-motion="(i, v) => $emit('update-motion', i, v)"
      @update-duration="(i, v) => $emit('update-duration', i, v)"
      @update-chain-mode="(i, v) => $emit('update-chain-mode', i, v)"
      @update-compose-mode="$emit('update-compose-mode', $event)"
      @reset-compose-defaults="$emit('reset-compose-defaults')"
      @reset-segment-defaults="$emit('reset-segment-defaults')"
      @generate-keyframe="$emit('generate-keyframe', $event)"
      @generate-segment="$emit('generate-segment', $event)"
      @pick-keyframe-gallery="$emit('pick-keyframe-gallery', $event)"
      @clear-keyframe="$emit('clear-keyframe', $event)"
      @clear-segment="$emit('clear-segment', $event)"
      @select-segment="$emit('select-segment', $event)"
      @polish-visual="$emit('polish-visual', $event)"
      @polish-motion="$emit('polish-motion', $event)"
      @pick-reference="$emit('pick-reference')"
      @remove-reference="$emit('remove-reference')"
      @pick-control="$emit('pick-control')"
      @remove-control="$emit('remove-control')"
      @pick-inpaint-source="$emit('pick-inpaint-source')"
      @remove-inpaint-source="$emit('remove-inpaint-source')"
      @pick-inpaint-mask="$emit('pick-inpaint-mask')"
      @remove-inpaint-mask="$emit('remove-inpaint-mask')"
    />
  </DqDrawer>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import LongVideoInspector from './LongVideoInspector.vue';
import type { KeyframeComposeParams } from '@/composables/useLongVideoKeyframeCompose';
import type { SegmentComposeParams } from '@/composables/useLongVideoSegmentCompose';
import type { LongVideoChainMode, LongVideoCharacter, LongVideoScene, LongVideoSelection, LongVideoShotCastLook, LongVideoShotSceneLook, LongVideoShotState } from '@/types';
import type { NormalizedParamSpec } from '@/utils/registryParamSchema';

const props = defineProps<{
  open: boolean;
  shots: LongVideoShotState[];
  selection: LongVideoSelection;
  segmentDurationOptions: number[];
  keyframeGenerating?: boolean;
  segmentGenerating?: boolean;
  segmentModelSupportsR2v?: boolean;
  visualPolishing?: boolean;
  motionPolishing?: boolean;
  outputSizeLabel?: string;
  segmentModelLabel?: string;
  segmentComposeParams: SegmentComposeParams;
  segmentParamSchema: Record<string, NormalizedParamSpec>;
  segmentCompatibleLoras?: Record<string, unknown>[];
  segmentShowNegativePrompt?: boolean;
  segmentShowSeedField?: boolean;
  segmentShowLora?: boolean;
  defaultChainMode: LongVideoChainMode;
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
  scenes?: LongVideoScene[];
  characterAnchor?: string;
  styleAnchor?: string;
  projectId?: string;
  parseRunId?: string;
}>();

defineEmits<{
  (e: 'update:open', value: boolean): void;
  (e: 'update-visual', index: number, value: string): void;
  (e: 'update-cast-looks', index: number, value: LongVideoShotCastLook[]): void;
  (e: 'update-scene-look', index: number, value: LongVideoShotSceneLook | undefined): void;
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
  (e: 'select-segment', index: number): void;
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

const panelTitle = computed(() => {
  const sel = props.selection;
  if (!sel || (sel.kind !== 'segment' && sel.kind !== 'clip')) {
    return $tt('video.longVideoInspectorTitle');
  }
  return $tt('video.longVideoKeyframeEdit', { n: sel.index + 1 });
});
</script>

<style scoped>
.lv-inspector-drawer :deep(.lv-inspector) {
  height: 100%;
}
</style>
