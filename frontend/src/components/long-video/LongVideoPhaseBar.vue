<template>
  <div class="lv-phase-bar" :class="{ 'lv-phase-bar--compact': compact }" role="list">
    <div
      v-for="(phase, i) in phases"
      :key="phase.id"
      class="lv-phase-bar__item"
      :class="{
        'is-active': activePhase === phase.id,
        'is-done': phaseDone(phase.id),
      }"
      role="listitem"
    >
      <span class="lv-phase-bar__dot" aria-hidden="true">
        <span v-if="phaseDone(phase.id)">✓</span>
        <span v-else>{{ i + 1 }}</span>
      </span>
      <span class="lv-phase-bar__label">{{ phase.label }}</span>
      <span v-if="i < phases.length - 1" class="lv-phase-bar__line" aria-hidden="true" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import type { LongVideoShotState } from '@/types';

export type LongVideoPhaseId = 'keyframes' | 'segments' | 'assembly';

const props = defineProps<{
  shots: LongVideoShotState[];
  activePhase?: LongVideoPhaseId;
  finalAssetId?: string;
  compact?: boolean;
}>();

const { t: $tt } = useI18n();

const phases = computed(() => [
  { id: 'keyframes' as const, label: $tt('video.longVideoPhaseKeyframes') },
  { id: 'segments' as const, label: $tt('video.longVideoPhaseSegments') },
  { id: 'assembly' as const, label: $tt('video.longVideoPhaseAssembly') },
]);

function phaseDone(id: LongVideoPhaseId): boolean {
  if (id === 'keyframes') {
    return props.shots.length > 0 && props.shots.every((s) => Boolean(s.keyframe_asset_id));
  }
  if (id === 'segments') {
    return props.shots.length > 0 && props.shots.every((s) => s.status === 'segment_ready');
  }
  if (id === 'assembly') return Boolean(props.finalAssetId);
  return false;
}
</script>
