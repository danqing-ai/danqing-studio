<template>
  <div class="lv-rail">
    <div class="lv-panel lv-section lv-section--rail">
      <div class="lv-section__head">
        <span class="lv-section__title">{{ $tt('video.longVideoTimeline') }}</span>
        <DqButton type="text" size="sm" @click="$emit('add-keyframe')">
          + {{ $tt('video.longVideoAddKeyframe') }}
        </DqButton>
      </div>

      <div class="lv-rail__track">
        <div v-if="!shots.length" class="lv-rail__empty">
          <p class="lv-rail__empty-title">{{ $tt('video.longVideoEmptyRailTitle') }}</p>
          <DqButton type="primary" @click="$emit('add-keyframe')">
            + {{ $tt('video.longVideoAddFirstKeyframe') }}
          </DqButton>
        </div>
        <div v-else ref="scrollEl" class="lv-rail__snake">
          <div v-for="(row, ri) in railRows" :key="row.rowIndex" class="lv-rail__row-wrap">
            <div class="lv-rail__row">
              <div
                v-if="ri > 0"
                class="lv-rail__row-lead-spacer"
                :style="{ width: `${rowLeadSpacerPx(row, railRows[ri - 1])}px` }"
                aria-hidden="true"
              />
              <template v-for="(shotIdx, pos) in row.shotIndices" :key="shots[shotIdx].id">
                <div
                  class="lv-rail__unit"
                  :data-shot-index="shotIdx"
                >
                  <button
                    type="button"
                    class="lv-rail__node"
                    :class="nodeClasses(shotIdx)"
                    @click="$emit('select-node', shotIdx)"
                  >
                    <div class="lv-rail__node-thumb">
                      <img
                        v-if="thumbUrl(shots[shotIdx].keyframe_asset_id)"
                        :src="thumbUrl(shots[shotIdx].keyframe_asset_id)"
                        alt=""
                      />
                      <div v-else class="lv-rail__node-empty">
                        <span class="lv-rail__node-empty-glyph" aria-hidden="true" />
                        <span>{{ $tt('video.longVideoKeyframePending') }}</span>
                      </div>
                      <span
                        v-if="keyframeGeneratingIndex === shotIdx"
                        class="lv-rail__generating"
                        aria-hidden="true"
                      />
                      <span class="lv-rail__badge" :class="statusClass(shots[shotIdx])">
                        {{ statusShort(shots[shotIdx]) }}
                      </span>
                    </div>
                    <div class="lv-rail__node-foot">
                      <span class="lv-rail__node-index">#{{ shotIdx + 1 }}</span>
                    </div>
                  </button>
                </div>

                <LongVideoRailSegment
                  v-if="pos < row.shotIndices.length - 1"
                  :idx="horizontalEdgeIndex(row, pos)"
                  :shot="shots[horizontalEdgeIndex(row, pos)]"
                  :selected="isEdgeSelected(horizontalEdgeIndex(row, pos))"
                  :generating="segmentGeneratingIndex === horizontalEdgeIndex(row, pos)"
                  variant="horizontal"
                  :rtl="row.reversed"
                  @select="$emit('select-edge', $event)"
                />
              </template>
            </div>

            <div v-if="ri < railRows.length - 1" class="lv-rail__turn-lane">
              <div
                class="lv-rail__turn-spacer"
                :style="{ width: `${turnLaneSpacerPx(row)}px` }"
                aria-hidden="true"
              />
              <LongVideoRailSegment
                :idx="rowTurnEdgeIndex(row)"
                :shot="shots[rowTurnEdgeIndex(row)]"
                :selected="isEdgeSelected(rowTurnEdgeIndex(row))"
                :generating="segmentGeneratingIndex === rowTurnEdgeIndex(row)"
                variant="turn"
                @select="$emit('select-edge', $event)"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue';
import { useI18n } from 'vue-i18n';
import LongVideoRailSegment from './LongVideoRailSegment.vue';
import {
  horizontalEdgeIndex,
  rowLeadSpacerPx,
  rowTurnEdgeIndex,
  turnLaneSpacerPx,
  useLongVideoRailSnakeLayout,
} from '@/composables/useLongVideoRailSnakeLayout';
import type { LongVideoSelection, LongVideoShotState } from '@/types';

const props = defineProps<{
  shots: LongVideoShotState[];
  selection: LongVideoSelection;
  keyframeGeneratingIndex?: number | null;
  segmentGeneratingIndex?: number | null;
}>();

defineEmits<{
  (e: 'select-node', index: number): void;
  (e: 'select-edge', index: number): void;
  (e: 'add-keyframe'): void;
}>();

const { t: $tt } = useI18n();

const scrollEl = ref<HTMLElement | null>(null);
const shotCount = computed(() => props.shots.length);
const { railRows, updateItemsPerRow } = useLongVideoRailSnakeLayout(shotCount, scrollEl);

let resizeObserver: ResizeObserver | null = null;

onMounted(() => {
  updateItemsPerRow();
  if (!scrollEl.value) return;
  resizeObserver = new ResizeObserver(() => updateItemsPerRow());
  resizeObserver.observe(scrollEl.value);
});

onUnmounted(() => {
  resizeObserver?.disconnect();
  resizeObserver = null;
});

watch(
  () => props.selection,
  async (sel) => {
    if (!sel || !scrollEl.value) return;
    await nextTick();
    const idx = sel.index;
    scrollEl.value.querySelector(`[data-shot-index="${idx}"]`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'nearest',
      inline: 'nearest',
    });
  },
);

function isNodeSelected(idx: number) {
  return props.selection?.kind === 'node' && props.selection.index === idx;
}

function isEdgeSelected(idx: number) {
  return props.selection?.kind === 'edge' && props.selection.index === idx;
}

function nodeClasses(idx: number) {
  const shot = props.shots[idx];
  return {
    'lv-rail__node--selected': isNodeSelected(idx),
    'lv-rail__node--ready': Boolean(shot?.keyframe_asset_id),
    'lv-rail__node--failed': shot?.status === 'failed',
    'lv-rail__node--generating': props.keyframeGeneratingIndex === idx,
  };
}

function thumbUrl(assetId: string | undefined) {
  if (!assetId) return '';
  return `/api/assets/${assetId}/thumbnail`;
}

function statusShort(shot: LongVideoShotState) {
  if (shot.status === 'failed') return '!';
  if (shot.status === 'segment_ready') return 'V';
  if (shot.keyframe_asset_id || shot.status === 'keyframe_ready') return 'K';
  return '·';
}

function statusClass(shot: LongVideoShotState) {
  if (shot.status === 'failed') return 'is-failed';
  if (shot.status === 'segment_ready') return 'is-segment';
  if (shot.keyframe_asset_id) return 'is-keyframe';
  return 'is-draft';
}
</script>
