<script setup lang="ts">
// @ts-nocheck
defineProps<{
  params: Record<string, unknown>;
  sourcePreviewSrc: string;
  sourceFileName: string;
}>();

const emit = defineEmits<{
  pickFile: [file: File];
  clearSource: [];
}>();

function onFileChange(ev: Event) {
  const input = ev.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) emit('pickFile', file);
  input.value = '';
}
</script>

<template>
  <DqSurfaceCard class="studio-surface-card studio-card-mb">
    <template #header>
      <div class="card-title">
        <DqIcon><headset /></DqIcon>
        {{ $t('audio.coverSource') }}
      </div>
    </template>
    <p class="studio-field-footnote studio-field-footnote--mb">{{ $t('audio.coverSourceHint') }}</p>
    <div class="studio-cover-source-row">
      <label class="studio-cover-upload-btn">
        <input type="file" accept="audio/*,.wav,.mp3,.flac,.m4a" class="studio-cover-upload-input" @change="onFileChange" />
        <DqButton type="secondary" size="sm">{{ $t('audio.coverPickFile') }}</DqButton>
      </label>
      <span v-if="sourceFileName" class="studio-cover-file-name">{{ sourceFileName }}</span>
      <DqButton v-if="sourcePreviewSrc" type="text" size="sm" @click="emit('clearSource')">
        {{ $t('common.delete') }}
      </DqButton>
    </div>
    <audio v-if="sourcePreviewSrc" class="studio-cover-audio-preview" controls :src="sourcePreviewSrc" />
  </DqSurfaceCard>

  <DqSurfaceCard class="studio-surface-card studio-card-mb">
    <template #header>
      <div class="card-title">
        <DqIcon><edit /></DqIcon>
        {{ $t('audio.coverPrompt') }}
      </div>
    </template>
    <DqInput
      v-model="params.prompt"
      type="textarea"
      :rows="3"
      :placeholder="$t('audio.coverPromptPlaceholder')"
    />
    <p class="studio-field-footnote">{{ $t('audio.coverPromptHint') }}</p>
  </DqSurfaceCard>

  <DqSurfaceCard class="studio-surface-card studio-card-mb">
    <template #header>
      <div class="card-title">
        <DqIcon><setting /></DqIcon>
        {{ $t('audio.coverFidelity') }}
      </div>
    </template>
    <DqPrefPane class="studio-create-pref-pane">
      <DqPrefRow :label="$t('audio.coverFidelity')" class="settings-pref-row--slider">
        <div class="param-control-row settings-pref-slider-row">
          <div class="param-slider">
            <DqSlider v-model="params.source_fidelity" :min="0" :max="1" :step="0.05" />
          </div>
          <DqInputNumber
            v-model="params.source_fidelity"
            :min="0"
            :max="1"
            :step="0.05"
            controls-position="right"
            class="param-input-number"
          />
        </div>
      </DqPrefRow>
      <p class="studio-field-footnote">{{ $t('audio.coverFidelityHint') }}</p>
    </DqPrefPane>
  </DqSurfaceCard>
</template>

<style scoped>
.studio-cover-source-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}
.studio-cover-upload-input {
  position: absolute;
  width: 0;
  height: 0;
  opacity: 0;
  pointer-events: none;
}
.studio-cover-upload-btn {
  position: relative;
  display: inline-flex;
}
.studio-cover-file-name {
  font-size: 0.875rem;
  color: var(--dq-text-secondary);
  max-width: 16rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.studio-cover-audio-preview {
  width: 100%;
  margin-top: 0.5rem;
}
.studio-field-footnote--mb {
  margin-bottom: 0.75rem;
}
</style>
