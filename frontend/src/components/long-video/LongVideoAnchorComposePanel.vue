<template>
  <section class="lv-anchor-compose">
    <DqAlert
      type="info"
      :closable="false"
      class="lv-anchor-compose__hint"
      :title="$tt('video.longVideoAnchorHint')"
    />
    <LongVideoKeyframeComposePanel
      :prompt="prompt"
      :model="model"
      :mode="mode ?? 'text2img'"
      :params="params"
      :generating="generating"
      :can-generate="canGenerate"
      :styles="styles"
      :show-negative-prompt="showNegativePrompt"
      :reference-image="referenceImage"
      :control-image="controlImage"
      :inpaint-source-image="inpaintSourceImage"
      :inpaint-mask-image="inpaintMaskImage"
      :current-model-config="currentModelConfig"
      :compatible-loras="compatibleLoras"
      :compatible-control-nets="compatibleControlNets"
      :control-net-runtime-available="controlNetRuntimeAvailable"
      :enhancing="enhancing"
      @update:prompt="$emit('update:prompt', $event)"
      @update:mode="$emit('update:mode', $event)"
      @generate="$emit('generate')"
      @pick-reference="$emit('pick-reference')"
      @remove-reference="$emit('remove-reference')"
      @pick-control="$emit('pick-control')"
      @remove-control="$emit('remove-control')"
      @pick-inpaint-source="$emit('pick-inpaint-source')"
      @remove-inpaint-source="$emit('remove-inpaint-source')"
      @pick-inpaint-mask="$emit('pick-inpaint-mask')"
      @remove-inpaint-mask="$emit('remove-inpaint-mask')"
      @reset-defaults="$emit('reset-defaults')"
      @enhance="$emit('enhance')"
    >
      <template #after-prompt>
        <slot name="after-prompt" />
      </template>
    </LongVideoKeyframeComposePanel>
  </section>
</template>

<script setup lang="ts">
import LongVideoKeyframeComposePanel from './LongVideoKeyframeComposePanel.vue';
import type { KeyframeComposeParams } from '@/composables/useLongVideoKeyframeCompose';

defineProps<{
  prompt: string;
  model: string;
  mode?: string;
  params: KeyframeComposeParams;
  generating?: boolean;
  canGenerate?: boolean;
  styles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; trigger_words?: string; media_scope?: string }>;
  showNegativePrompt?: boolean;
  referenceImage: { previewUrl: string; path: string } | null;
  controlImage?: { previewUrl: string; path: string } | null;
  inpaintSourceImage?: { previewUrl: string; path: string } | null;
  inpaintMaskImage?: { previewUrl: string; path: string } | null;
  currentModelConfig?: Record<string, unknown> | null;
  compatibleLoras?: Record<string, unknown>[];
  compatibleControlNets?: Record<string, unknown>[];
  controlNetRuntimeAvailable?: boolean;
  enhancing?: boolean;
}>();

defineEmits<{
  (e: 'update:prompt', value: string): void;
  (e: 'update:mode', value: string): void;
  (e: 'generate'): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'pick-control'): void;
  (e: 'remove-control'): void;
  (e: 'pick-inpaint-source'): void;
  (e: 'remove-inpaint-source'): void;
  (e: 'pick-inpaint-mask'): void;
  (e: 'remove-inpaint-mask'): void;
  (e: 'reset-defaults'): void;
  (e: 'enhance'): void;
}>();
</script>

<style scoped>
.lv-anchor-compose {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.lv-anchor-compose__hint {
  margin: 0;
}
</style>
