<template>
  <div v-if="groups.length" class="lv-group-progress" role="list">
    <button
      v-for="group in groups"
      :key="group.groupId"
      type="button"
      class="lv-group-progress__pill"
      :class="pillClass(group)"
      role="listitem"
      @click="$emit('select-group', group.groupId)"
    >
      {{ groupLabel(group) }}
    </button>
    <span class="lv-group-progress__summary">{{ summaryLabel }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { beatGroupProgress, groupShotsByBeat } from '@/utils/longVideoProject';
import type { LongVideoBeatGroup, LongVideoShotState } from '@/types';

const props = defineProps<{
  shots: LongVideoShotState[];
}>();

defineEmits<{
  (e: 'select-group', groupId: string): void;
}>();

const { t: $tt } = useI18n();

const groups = computed(() => groupShotsByBeat(props.shots));

const readyCount = computed(() =>
  groups.value.filter((g) => beatGroupProgress(props.shots, g) === 'group_ready').length,
);

const summaryLabel = computed(() =>
  $tt('video.longVideoGroupProgressSummary', {
    ready: readyCount.value,
    total: groups.value.length,
  }),
);

function pillClass(group: LongVideoBeatGroup) {
  const st = beatGroupProgress(props.shots, group);
  return {
    'is-ready': st === 'group_ready',
    'is-anchor': st === 'anchor_ready',
    'is-pending': st === 'needs_anchor',
  };
}

function groupLabel(group: LongVideoBeatGroup) {
  const st = beatGroupProgress(props.shots, group);
  const prefix =
    st === 'group_ready' ? '✓' : st === 'anchor_ready' ? '◐' : '○';
  return `${prefix} ${group.beatIndex + 1}`;
}
</script>
