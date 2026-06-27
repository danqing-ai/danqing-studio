<template>
  <section class="lv-brief lv-panel lv-section lv-section--compact">
    <div class="lv-brief__tabs">
      <button
        type="button"
        class="lv-brief__tab"
        :class="{ 'lv-brief__tab--active': sourceMode === 'brief' }"
        @click="emit('update:sourceMode', 'brief')"
      >
        {{ $tt('video.longVideoSourceBrief') }}
      </button>
      <button
        type="button"
        class="lv-brief__tab"
        :class="{ 'lv-brief__tab--active': sourceMode === 'chapter' }"
        @click="emit('update:sourceMode', 'chapter')"
      >
        {{ $tt('video.longVideoSourceChapter') }}
      </button>
    </div>

    <div class="lv-brief__head">
      <span class="lv-section__title">
        {{ sourceMode === 'chapter' ? $tt('video.longVideoChapterLabel') : $tt('video.longVideoBriefLabel') }}
      </span>
      <div class="lv-brief__head-actions">
        <label v-if="sourceMode === 'brief'" class="lv-brief__duration">
          <span class="lv-brief__duration-label">{{ $tt('video.longVideoTargetDuration') }}</span>
          <DqSelect
            :model-value="targetDurationSec"
            size="small"
            class="lv-brief__duration-select"
            @update:model-value="onDurationChange"
          >
            <DqOption
              v-for="sec in durationChoices"
              :key="sec"
              :label="$tt('video.longVideoTargetDurationSec', { sec })"
              :value="sec"
            />
          </DqSelect>
        </label>
        <span v-if="sourceMode === 'brief' && estimatedShots > 0" class="lv-brief__meta">
          {{ $tt('video.longVideoStoryboardShotEstimate', { n: estimatedShots }) }}
        </span>
        <span v-if="sourceMode === 'chapter' && chapterSceneCount > 0" class="lv-brief__meta">
          {{ $tt('video.longVideoChapterDurationEstimate', {
            shots: chapterSceneCount,
            sec: chapterEstimatedDuration,
          }) }}
        </span>

        <DqButton
          v-if="sourceMode === 'chapter'"
          size="sm"
          :loading="analyzing"
          :disabled="analyzing || !chapterText.trim()"
          @click="emit('analyze-chapter')"
        >
          {{ analyzing ? $tt('video.longVideoChapterAnalyzing') : $tt('video.longVideoChapterAnalyze') }}
        </DqButton>
        <DqButton
          type="primary"
          size="sm"
          :loading="expanding"
          :disabled="expandDisabled"
          @click="emit('expand')"
        >
          {{ expanding ? $tt('video.storyboardExpanding') : $tt('video.storyboardExpand') }}
        </DqButton>
      </div>
    </div>

    <template v-if="sourceMode === 'brief'">
      <textarea
        class="lv-brief__textarea dq-input dq-input--textarea"
        :value="brief"
        :placeholder="$tt('video.longVideoBriefPlaceholder')"
        rows="3"
        @input="onBriefInput"
      />
    </template>

    <template v-else>
      <input
        class="lv-brief__title dq-input"
        type="text"
        :value="chapterTitle"
        :placeholder="$tt('video.longVideoChapterTitlePh')"
        @input="onChapterTitleInput"
      />
      <div class="lv-brief__chapter-toolbar">
        <DqButton size="sm" type="text" @click="triggerFileUpload">
          {{ $tt('video.longVideoChapterUploadTxt') }}
        </DqButton>
        <input
          ref="fileInputRef"
          type="file"
          accept=".txt,text/plain"
          class="lv-brief__file-input"
          @change="onFileSelected"
        />
      </div>
      <textarea
        class="lv-brief__textarea lv-brief__textarea--chapter dq-input dq-input--textarea"
        :value="chapterText"
        :placeholder="$tt('video.longVideoChapterPlaceholder')"
        rows="8"
        @input="onChapterTextInput"
      />

      <details v-if="chapterAnalysis?.scene_beats?.length" class="lv-brief__preview" open>
        <summary>{{ $tt('video.longVideoChapterPreview') }}</summary>
        <p v-if="chapterAnalysis.synopsis" class="lv-brief__synopsis">{{ chapterAnalysis.synopsis }}</p>
        <ol class="lv-brief__scene-list">
          <li v-for="(scene, idx) in chapterAnalysis.scene_beats" :key="scene.order ?? idx">
            <textarea
              class="lv-brief__scene-beat dq-input dq-input--textarea"
              :value="scene.beat"
              rows="2"
              @input="onSceneBeatInput(idx, $event)"
            />
          </li>
        </ol>
      </details>
    </template>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast } from '@/utils/feedback';
import type { LongVideoChapterAnalysis } from '@/types';

const MAX_TXT_BYTES = 512 * 1024;

const props = defineProps<{
  sourceMode: 'brief' | 'chapter';
  brief: string;
  chapterText: string;
  chapterTitle: string;
  chapterAnalysis?: LongVideoChapterAnalysis | null;
  targetDurationSec: number;
  segmentDurationSec: number;
  expanding?: boolean;
  analyzing?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:sourceMode', value: 'brief' | 'chapter'): void;
  (e: 'update:brief', value: string): void;
  (e: 'update:chapterText', value: string): void;
  (e: 'update:chapterTitle', value: string): void;
  (e: 'update:chapterAnalysis', value: LongVideoChapterAnalysis | undefined): void;
  (e: 'update:targetDurationSec', value: number): void;
  (e: 'expand'): void;
  (e: 'analyze-chapter'): void;
}>();

const { t: $tt } = useI18n();
const fileInputRef = ref<HTMLInputElement | null>(null);

const durationChoices = [30, 45, 60, 90, 120] as const;

const estimatedShots = computed(() => {
  const seg = Math.max(1, props.segmentDurationSec || 5);
  const target = Math.max(seg, props.targetDurationSec || 60);
  return Math.max(2, Math.ceil(target / seg));
});

const chapterSceneCount = computed(() => props.chapterAnalysis?.scene_beats?.length ?? 0);

const chapterEstimatedDuration = computed(() => {
  const seg = Math.max(1, props.segmentDurationSec || 5);
  return chapterSceneCount.value * seg;
});

const expandDisabled = computed(() => {
  if (props.expanding) return true;
  if (props.sourceMode === 'brief') return !props.brief.trim();
  return chapterSceneCount.value < 2;
});

function onBriefInput(event: Event) {
  emit('update:brief', (event.target as HTMLTextAreaElement).value);
}

function onChapterTextInput(event: Event) {
  emit('update:chapterText', (event.target as HTMLTextAreaElement).value);
  emit('update:chapterAnalysis', undefined);
}

function onChapterTitleInput(event: Event) {
  emit('update:chapterTitle', (event.target as HTMLInputElement).value);
}

function onDurationChange(value: number | string) {
  const sec = typeof value === 'number' ? value : Number.parseInt(value, 10);
  if (Number.isFinite(sec) && sec > 0) {
    emit('update:targetDurationSec', sec);
  }
}

function onSceneBeatInput(index: number, event: Event) {
  const beat = (event.target as HTMLTextAreaElement).value;
  const analysis = props.chapterAnalysis;
  if (!analysis?.scene_beats?.length) return;
  const scene_beats = analysis.scene_beats.map((s, i) =>
    i === index ? { ...s, beat } : s,
  );
  emit('update:chapterAnalysis', { ...analysis, scene_beats });
}

function triggerFileUpload() {
  fileInputRef.value?.click();
}

async function onFileSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  if (file.size > MAX_TXT_BYTES) {
    toast.warning($tt('video.longVideoChapterFileTooLarge'));
    return;
  }
  try {
    const text = await file.text();
    emit('update:chapterText', text);
    emit('update:chapterAnalysis', undefined);
    if (!props.chapterTitle.trim() && file.name) {
      emit('update:chapterTitle', file.name.replace(/\.txt$/i, ''));
    }
  } catch {
    toast.error($tt('video.longVideoChapterFileReadFailed'));
  }
}
</script>

<style scoped>
.lv-brief__tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 8px;
}

.lv-brief__tab {
  padding: 4px 10px;
  border: 1px solid var(--dq-border-subtle, rgba(255, 255, 255, 0.12));
  border-radius: 6px;
  background: transparent;
  color: var(--dq-text-secondary, #aaa);
  cursor: pointer;
  font-size: 12px;
}

.lv-brief__tab--active {
  color: var(--dq-text-primary, #fff);
  border-color: var(--dq-accent, #6ea8fe);
  background: rgba(110, 168, 254, 0.08);
}

.lv-brief__head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.lv-brief__head-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.lv-brief__duration {
  display: flex;
  align-items: center;
  gap: 6px;
}

.lv-brief__duration-label {
  font-size: 12px;
  color: var(--dq-text-secondary, #aaa);
}

.lv-brief__duration-select {
  min-width: 88px;
}

.lv-brief__meta {
  font-size: 12px;
  color: var(--dq-text-secondary, #aaa);
}

.lv-brief__textarea {
  width: 100%;
  min-height: 72px;
  resize: vertical;
}

.lv-brief__textarea--chapter {
  min-height: 160px;
}

.lv-brief__title {
  width: 100%;
  margin-bottom: 8px;
}

.lv-brief__chapter-toolbar {
  margin-bottom: 6px;
}

.lv-brief__file-input {
  display: none;
}

.lv-brief__preview {
  margin-top: 12px;
  font-size: 13px;
}

.lv-brief__synopsis {
  margin: 8px 0;
  color: var(--dq-text-secondary, #aaa);
  line-height: 1.5;
}

.lv-brief__scene-list {
  margin: 0;
  padding-left: 1.2rem;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-brief__scene-beat {
  width: 100%;
  min-height: 48px;
  resize: vertical;
  font-size: 13px;
}
</style>
