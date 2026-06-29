<template>
  <div v-if="shots.length" class="lv-group-toolbar">
    <DqButton
      type="primary"
      size="sm"
      :loading="generating"
      :disabled="generating || !canGenerateGroup"
      @click="$emit('generate-group')"
    >
      {{ $tt('video.longVideoBatchGenerateGroup') }}
    </DqButton>
    <DqButton
      type="default"
      size="sm"
      :loading="generating"
      :disabled="generating || pendingAnchorCount === 0"
      @click="$emit('generate-all-anchors')"
    >
      {{ $tt('video.longVideoBatchGenerateAnchors', { n: pendingAnchorCount }) }}
    </DqButton>
    <DqButton
      type="default"
      size="sm"
      :loading="generating"
      :disabled="generating || pendingSegmentCount === 0"
      @click="$emit('generate-all-segments')"
    >
      {{ $tt('video.longVideoBatchGenerateSegments', { n: pendingSegmentCount }) }}
    </DqButton>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  allPendingAnchorKeyframeIndices,
  allPendingSegmentIndices,
  groupShotsByBeat,
  planGroupGeneration,
  selectedBeatGroupId,
} from '@/utils/longVideoProject';
import type { LongVideoSelection, LongVideoShotState } from '@/types';

const props = defineProps<{
  shots: LongVideoShotState[];
  selection: LongVideoSelection | null;
  chainMode?: import('@/types').LongVideoChainMode;
  generating?: boolean;
}>();

defineEmits<{
  (e: 'generate-group'): void;
  (e: 'generate-all-anchors'): void;
  (e: 'generate-all-segments'): void;
}>();

const { t: $tt } = useI18n();

const pendingAnchorCount = computed(() => allPendingAnchorKeyframeIndices(props.shots).length);

const pendingSegmentCount = computed(() =>
  allPendingSegmentIndices({ shots: props.shots, chain_mode: props.chainMode }).length,
);

const canGenerateGroup = computed(() => {
  const groupId = selectedBeatGroupId(props.shots, props.selection);
  if (!groupId) return false;
  const group = groupShotsByBeat(props.shots).find((g) => g.groupId === groupId);
  if (!group) return false;
  const plan = planGroupGeneration({ shots: props.shots, chain_mode: props.chainMode }, group);
  return plan.keyframeIndices.length > 0 || plan.segmentIndices.length > 0;
});
</script>

<style scoped>
.lv-group-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 0 0 10px;
}
</style>
