<template>
  <div class="lv-rail lv-beat-rail" :style="railAspectStyle">
    <div class="lv-panel lv-section lv-section--rail">
      <div class="lv-section__head">
        <span class="lv-section__title">{{ $tt('video.longVideoTimeline') }}</span>
      </div>

      <div class="lv-beat-rail__track">
        <div v-if="!shots.length" class="lv-rail__empty">
          <p class="lv-rail__empty-title">{{ $tt('video.longVideoEmptyRailTitle') }}</p>
        </div>
        <div v-else class="lv-beat-rail__groups">
          <LongVideoBeatGroupCard
            v-for="group in groups"
            :key="group.groupId"
            :group="group"
            :shots="shots"
            :selection="selection"
            :keyframe-generating-index="keyframeGeneratingIndex"
            :segment-generating-indices="segmentGeneratingIndices"
            @select-segment="$emit('select-segment', $event)"
            @insert-anchor="$emit('insert-anchor', $event)"
            @resplit-beat="$emit('resplit-beat', $event)"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { groupShotsByBeat } from '@/utils/longVideoProject';
import LongVideoBeatGroupCard from './LongVideoBeatGroupCard.vue';
import type { LongVideoSelection, LongVideoShotState } from '@/types';

const props = defineProps<{
  shots: LongVideoShotState[];
  selection: LongVideoSelection;
  keyframeGeneratingIndex?: number | null;
  segmentGeneratingIndices?: number[];
  outputWidth?: number;
  outputHeight?: number;
}>();

defineEmits<{
  (e: 'select-segment', index: number): void;
  (e: 'insert-anchor', groupId: string): void;
  (e: 'resplit-beat', groupId: string): void;
}>();

const { t: $tt } = useI18n();

const groups = computed(() => groupShotsByBeat(props.shots));

const railAspectStyle = computed(() => {
  const w = props.outputWidth ?? 1280;
  const h = props.outputHeight ?? 704;
  return {
    '--lv-rail-thumb-aspect': `${w} / ${h}`,
    '--lv-rail-thumb-h-ratio': String(h / w),
  };
});
</script>
