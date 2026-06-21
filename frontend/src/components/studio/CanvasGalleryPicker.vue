<template>
  <DqDialog
    v-model:open="openModel"
    :title="$t('canvas.importWorksTitle')"
    width="min(720px, 92vw)"
    destroy-on-close
    @update:open="onOpenChange"
  >
    <div class="canvas-gallery-picker">
      <p class="canvas-gallery-picker__hint">{{ $t('canvas.importWorksHint') }}</p>

      <div class="canvas-gallery-picker__toolbar">
        <DqButton size="sm" type="text" @click="selectAllVisible">
          {{ $t('canvas.importSelectAll') }}
        </DqButton>
        <DqButton size="sm" type="text" :disabled="selected.size === 0" @click="clearSelection">
          {{ $t('canvas.importClearSelection') }}
        </DqButton>
        <span class="canvas-gallery-picker__count">
          {{ $t('canvas.importSelected', { n: selected.size }) }}
        </span>
      </div>

      <DqEmpty
        v-if="visibleItems.length === 0"
        :description="$t('canvas.importWorksEmpty')"
      />

      <div v-else class="canvas-gallery-picker__grid">
        <button
          v-for="item in visibleItems"
          :key="item.path"
          type="button"
          class="canvas-gallery-picker__cell"
          :class="{
            'canvas-gallery-picker__cell--selected': selected.has(item.path),
            'canvas-gallery-picker__cell--on-canvas': onCanvas.has(item.path),
          }"
          @click="toggle(item.path)"
          @dblclick.prevent="importOne(item.path)"
        >
          <span
            class="canvas-gallery-picker__check dq-gallery-check"
            :class="{ 'is-checked': selected.has(item.path) }"
            role="checkbox"
            :aria-checked="selected.has(item.path)"
          />
          <div class="canvas-gallery-picker__thumb">
            <template v-if="media === 'image'">
              <img
                v-if="!thumbFailed[item.path]"
                :src="thumbUrl(item)"
                :alt="item.name"
                loading="lazy"
                decoding="sync"
                @error="onThumbError(item)"
              />
              <div v-else class="canvas-gallery-picker__fallback">
                <DqIcon :size="24"><Picture /></DqIcon>
              </div>
            </template>
            <template v-else-if="media === 'video'">
              <video :src="fileUrl(item)" muted preload="metadata" />
            </template>
            <template v-else>
              <div class="canvas-gallery-picker__audio">
                <DqIcon :size="22"><Headset /></DqIcon>
              </div>
            </template>
          </div>
          <span v-if="onCanvas.has(item.path)" class="canvas-gallery-picker__badge">
            {{ $t('canvas.onCanvasBadge') }}
          </span>
          <span class="canvas-gallery-picker__caption" :title="item.prompt || item.name">
            {{ caption(item) }}
          </span>
        </button>
      </div>
    </div>

    <template #footer>
      <DqButton size="sm" @click="openModel = false">
        {{ $t('common.cancel') }}
      </DqButton>
      <DqButton
        type="primary"
        size="sm"
        :disabled="selected.size === 0"
        @click="confirmImport"
      >
        {{ $t('canvas.importConfirm', { n: selected.size }) }}
      </DqButton>
    </template>
  </DqDialog>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue';
import { Headset, Picture } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import {
  galleryThumbnailUrl,
  isAudioGalleryItem,
  isImageGalleryItem,
  isVideoGalleryItem,
} from '@/utils/canvasAssets';
import type { GalleryItem } from '@/types';

const props = defineProps<{
  open: boolean;
  items: GalleryItem[];
  media: 'image' | 'video' | 'audio';
  onCanvasPaths?: string[];
}>();

const emit = defineEmits<{
  (e: 'update:open', value: boolean): void;
  (e: 'import', paths: string[]): void;
}>();

const openModel = computed({
  get: () => props.open,
  set: (v: boolean) => emit('update:open', v),
});

const selected = ref<Set<string>>(new Set());
const thumbFailed = reactive<Record<string, boolean>>({});
/** After thumbnail 404, retry once with full image file. */
const thumbRetryFile = reactive<Record<string, boolean>>({});

const onCanvas = computed(() => new Set(props.onCanvasPaths ?? []));

const visibleItems = computed(() =>
  props.items.filter((item) => {
    if (props.media === 'image') return isImageGalleryItem(item);
    if (props.media === 'video') return isVideoGalleryItem(item);
    return isAudioGalleryItem(item);
  }),
);

function onOpenChange(open: boolean) {
  if (!open) clearSelection();
}

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) clearSelection();
  },
);

function thumbUrl(item: GalleryItem) {
  if (thumbRetryFile[item.path]) {
    return api.gallery.getImageUrl(item.path);
  }
  return galleryThumbnailUrl(item);
}

function onThumbError(item: GalleryItem) {
  const path = item.path;
  if (!path) return;
  if (!thumbRetryFile[path] && path.startsWith('asset:')) {
    thumbRetryFile[path] = true;
    return;
  }
  thumbFailed[path] = true;
}

function fileUrl(item: GalleryItem) {
  return api.gallery.getImageUrl(item.path);
}

function caption(item: GalleryItem) {
  const text = item.prompt || item.name || '';
  return text.length > 28 ? `${text.slice(0, 28)}…` : text;
}

function toggle(path: string) {
  const next = new Set(selected.value);
  if (next.has(path)) next.delete(path);
  else next.add(path);
  selected.value = next;
}

function selectAllVisible() {
  selected.value = new Set(visibleItems.value.map((i) => i.path));
}

function clearSelection() {
  selected.value = new Set();
  for (const key of Object.keys(thumbRetryFile)) delete thumbRetryFile[key];
  for (const key of Object.keys(thumbFailed)) delete thumbFailed[key];
}

function confirmImport() {
  const paths = Array.from(selected.value);
  if (paths.length === 0) return;
  emit('import', paths);
  openModel.value = false;
}

function importOne(path: string) {
  emit('import', [path]);
  openModel.value = false;
}
</script>

<style scoped>
.canvas-gallery-picker {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 200px;
}

.canvas-gallery-picker__hint {
  margin: 0;
  font-size: 12px;
  color: var(--dq-label-secondary);
  line-height: 1.45;
}

.canvas-gallery-picker__toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.canvas-gallery-picker__count {
  margin-left: auto;
  font-size: 12px;
  color: var(--dq-label-tertiary);
  font-variant-numeric: tabular-nums;
}

.canvas-gallery-picker__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(108px, 1fr));
  grid-auto-rows: auto;
  align-items: start;
  gap: 10px;
  max-height: min(52vh, 420px);
  overflow-y: auto;
  padding: 2px;
}

.canvas-gallery-picker__cell {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 6px;
  padding: 0;
  border: 1px solid var(--dq-border-subtle);
  border-radius: 8px;
  background: var(--dq-surface-inset);
  cursor: pointer;
  text-align: left;
  overflow: hidden;
  appearance: none;
  -webkit-appearance: none;
  font: inherit;
  color: inherit;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.canvas-gallery-picker__cell:hover {
  border-color: var(--dq-border-strong);
}

.canvas-gallery-picker__cell--selected {
  border-color: var(--dq-accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 24%, transparent);
}

.canvas-gallery-picker__cell--on-canvas:not(.canvas-gallery-picker__cell--selected) {
  opacity: 0.72;
}

.canvas-gallery-picker__thumb {
  position: relative;
  width: 100%;
  flex: 0 0 auto;
  height: 0;
  padding-bottom: 100%;
  overflow: hidden;
  background: var(--dq-fill-secondary);
}

.canvas-gallery-picker__thumb img,
.canvas-gallery-picker__thumb video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.canvas-gallery-picker__fallback,
.canvas-gallery-picker__audio {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-label-tertiary);
  background: var(--dq-fill-secondary);
}

.canvas-gallery-picker__check {
  position: absolute;
  top: 6px;
  left: 6px;
  z-index: 2;
}

.canvas-gallery-picker__badge {
  position: absolute;
  top: 6px;
  right: 6px;
  z-index: 2;
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--dq-bg-page) 82%, transparent);
  color: var(--dq-label-secondary);
  border: 1px solid var(--dq-border-subtle);
}

.canvas-gallery-picker__caption {
  flex: 0 0 auto;
  padding: 0 8px 8px;
  font-size: 11px;
  color: var(--dq-label-secondary);
  line-height: 1.3;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
