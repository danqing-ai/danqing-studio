<template>
  <button
    type="button"
    class="lv-rail__segment"
    :class="[
      `lv-rail__segment--${variant}`,
      stateClass,
      { 'lv-rail__segment--selected': selected, 'lv-rail__segment--rtl': rtl && variant === 'horizontal' },
    ]"
    :title="title"
    :aria-label="ariaLabel"
    @click.stop="$emit('select', idx)"
  >
    <template v-if="variant === 'horizontal'">
      <span class="lv-rail__segment-arrow lv-rail__segment-arrow--in" aria-hidden="true" :title="inHint">{{ flowArrow }}</span>
      <span class="lv-rail__segment-body">
        <span class="lv-rail__segment-kind">{{ $tt('video.longVideoEdgeShort') }}</span>
        <span class="lv-rail__segment-range">{{ idx + 1 }}→{{ idx + 2 }}</span>
        <span class="lv-rail__segment-meta">{{ metaLabel }}</span>
      </span>
      <span class="lv-rail__segment-arrow lv-rail__segment-arrow--out" aria-hidden="true" :title="outHint">{{ flowArrow }}</span>
    </template>
    <template v-else>
      <span class="lv-rail__segment-arrow lv-rail__segment-arrow--in lv-rail__segment-arrow--down" aria-hidden="true" :title="inHint">↓</span>
      <span class="lv-rail__segment-body lv-rail__segment-body--turn">
        <span class="lv-rail__segment-kind">{{ $tt('video.longVideoEdgeShort') }}</span>
        <span class="lv-rail__segment-range">{{ idx + 1 }}→{{ idx + 2 }}</span>
        <span class="lv-rail__segment-meta">{{ metaLabel }}</span>
      </span>
      <span class="lv-rail__segment-arrow lv-rail__segment-arrow--out lv-rail__segment-arrow--down" aria-hidden="true" :title="outHint">↓</span>
    </template>
  </button>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { shotDurationSec } from '@/utils/longVideoProject';
import type { LongVideoShotState } from '@/types';

const props = defineProps<{
  idx: number;
  shot: LongVideoShotState;
  selected?: boolean;
  generating?: boolean;
  variant: 'horizontal' | 'turn';
  rtl?: boolean;
}>();

defineEmits<{
  (e: 'select', idx: number): void;
}>();

const { t: $tt } = useI18n();

type EdgeState = 'empty' | 'pending' | 'ready' | 'generating';

const state = computed<EdgeState>(() => {
  if (props.generating) return 'generating';
  if (props.shot.segment_asset_id) return 'ready';
  if (props.shot.motion_prompt.trim()) return 'pending';
  return 'empty';
});

const stateClass = computed(() => ({
  'lv-rail__segment--empty': state.value === 'empty',
  'lv-rail__segment--pending': state.value === 'pending',
  'lv-rail__segment--ready': state.value === 'ready',
  'lv-rail__segment--generating': state.value === 'generating',
}));

const flowArrow = computed(() => (props.rtl ? '←' : '→'));

const metaLabel = computed(() => {
  if (state.value === 'generating') return $tt('create.generating');
  if (state.value === 'ready') return $tt('video.longVideoEdgePreview');
  return $tt('video.longVideoSegmentDurationSec', { sec: shotDurationSec(props.shot) });
});

const title = computed(() => {
  const from = props.idx + 1;
  const to = props.idx + 2;
  const sec = shotDurationSec(props.shot);
  if (props.shot.segment_asset_id) {
    return $tt('video.longVideoSegmentEdgeHintReady', { from, to, sec });
  }
  return $tt('video.longVideoSegmentEdgeHint', { from, to, sec });
});

const ariaLabel = computed(() => {
  const motion = (props.shot.motion_prompt || '').trim();
  const base = title.value;
  if (motion) return `${base} — ${motion.slice(0, 80)}`;
  return `${base} — ${$tt('video.longVideoTransitionEmpty')}`;
});

const inHint = computed(() =>
  $tt('video.longVideoRailArrowIn', { from: props.idx + 1, to: props.idx + 2 }),
);

const outHint = computed(() =>
  $tt('video.longVideoRailArrowOut', { from: props.idx + 1, to: props.idx + 2 }),
);
</script>
