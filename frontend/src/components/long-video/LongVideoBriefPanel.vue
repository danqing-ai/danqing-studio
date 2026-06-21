<template>
  <section class="lv-brief lv-panel lv-section lv-section--compact">
    <div class="lv-brief__head">
      <span class="lv-section__title">{{ $tt('video.longVideoBriefLabel') }}</span>
      <div class="lv-brief__head-actions">
        <label class="lv-brief__duration">
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
        <span v-if="estimatedShots > 0" class="lv-brief__meta">
          {{ $tt('video.longVideoStoryboardShotEstimate', { n: estimatedShots }) }}
        </span>
        <DqButton
          type="primary"
          size="sm"
          :loading="expanding"
          :disabled="expanding || !brief.trim()"
          @click="$emit('expand')"
        >
          {{ expanding ? $tt('video.storyboardExpanding') : $tt('video.storyboardExpand') }}
        </DqButton>
      </div>
    </div>

    <textarea
      class="lv-brief__textarea dq-input dq-input--textarea"
      :value="brief"
      :placeholder="$tt('video.longVideoBriefPlaceholder')"
      rows="3"
      @input="onBriefInput"
    />
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  brief: string;
  targetDurationSec: number;
  segmentDurationSec: number;
  expanding?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:brief', value: string): void;
  (e: 'update:targetDurationSec', value: number): void;
  (e: 'expand'): void;
}>();

const { t: $tt } = useI18n();

const durationChoices = [30, 45, 60, 90, 120] as const;

const estimatedShots = computed(() => {
  const seg = Math.max(1, props.segmentDurationSec || 5);
  const target = Math.max(seg, props.targetDurationSec || 60);
  return Math.max(2, Math.ceil(target / seg));
});

function onBriefInput(event: Event) {
  emit('update:brief', (event.target as HTMLTextAreaElement).value);
}

function onDurationChange(value: number | string) {
  const sec = typeof value === 'number' ? value : Number.parseInt(value, 10);
  if (Number.isFinite(sec) && sec > 0) {
    emit('update:targetDurationSec', sec);
  }
}
</script>
