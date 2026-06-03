<template>
  <div
    class="studio-card"
    :class="{
      'studio-card--video': isVideo,
      'studio-card--audio': isAudio,
      'studio-card--selected': selected,
      'studio-card--selection-mode': selectionMode,
    }"
    @mouseenter="isHovered = true"
    @mouseleave="isHovered = false"
    @click="$emit('click', $event)"
  >
    <!-- Media container (always square) -->
    <div class="studio-card__media">
      <div
        v-if="selectionMode"
        class="studio-card__checkbox"
        @click.stop="$emit('toggle-select')"
      >
        <span
          class="dq-gallery-check"
          :class="{ 'is-checked': selected }"
          role="checkbox"
          :aria-checked="selected"
        />
      </div>
      <!-- Image -->
      <template v-if="isImage">
        <img
          v-if="!thumbFailed"
          :src="thumbUrl"
          :alt="item.name"
          loading="lazy"
          @error="thumbFailed = true"
        />
        <div v-else class="studio-card__fallback">
          <DqIcon :size="36"><Picture /></DqIcon>
        </div>
      </template>

      <!-- Video -->
      <template v-else-if="isVideo">
        <video
          :src="fileUrl"
          muted
          loop
          preload="metadata"
          @mouseenter="handleVideoEnter"
          @mouseleave="handleVideoLeave"
        />
        <div class="studio-card__play-overlay">
          <DqIcon size="28"><VideoPlay /></DqIcon>
        </div>
      </template>

      <!-- Audio -->
      <template v-else-if="isAudio">
        <div class="studio-card__audio-tile" :style="audioTileStyle">
          <div class="studio-card__audio-visualizer">
            <span
              v-for="i in 8"
              :key="i"
              class="studio-card__audio-bar"
              :style="{ '--bar-i': i - 1 }"
            />
          </div>
          <DqIcon class="studio-card__audio-icon" :size="32"><Headset /></DqIcon>
          <span class="studio-card__audio-label">{{ $t('gallery.audioLabel') }}</span>
          <span v-if="durationLabel" class="studio-card__audio-dur">{{ durationLabel }}</span>
        </div>
      </template>

      <!-- Hover overlay -->
      <transition name="studio-fade">
        <div v-if="isHovered && !selectionMode" class="studio-card__overlay">
          <div class="studio-card__overlay-bg" />
          <div class="studio-card__overlay-actions">
            <!-- Image-specific actions -->
            <template v-if="isImage">
              <DqIconButton
                type="text"
                size="sm"
                :label="$t('action.image.retouch')"
                @click.stop="emitAction('retouch')"
              >
                <DqIcon :size="14"><Brush /></DqIcon>
              </DqIconButton>
              <DqIconButton
                type="text"
                size="sm"
                :label="$t('action.image.extend')"
                @click.stop="emitAction('extend')"
              >
                <DqIcon :size="14"><Grid /></DqIcon>
              </DqIconButton>
              <DqIconButton
                type="text"
                size="sm"
                :label="$t('action.image.upscale')"
                @click.stop="emitAction('upscale')"
              >
                <DqIcon :size="14"><ZoomIn /></DqIcon>
              </DqIconButton>
            </template>

            <!-- Common actions -->
            <DqIconButton
              type="text"
              size="sm"
              :label="$t('gallery.download')"
              @click.stop="emitAction('download')"
            >
              <DqIcon :size="14"><Download /></DqIcon>
            </DqIconButton>

            <DqIconButton
              type="text"
              size="sm"
              :label="$t('common.delete')"
              @click.stop="emitAction('delete')"
            >
              <DqIcon :size="14"><Delete /></DqIcon>
            </DqIconButton>
          </div>

          <div class="studio-card__overlay-info">
            <span class="studio-card__overlay-res">{{ resolutionLabel }}</span>
            <span v-if="item.model" class="studio-card__overlay-model">{{ item.model }}</span>
          </div>
        </div>
      </transition>
    </div>

    <!-- Footer (always visible) -->
    <div class="studio-card__footer">
      <span v-if="item.prompt" class="studio-card__prompt" :title="item.prompt">
        {{ truncate(item.prompt, 40) }}
      </span>
      <span v-else class="studio-card__prompt studio-card__prompt--empty">
        {{ item.name }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import {
  Brush,
  Delete,
  Download,
  Grid,
  Headset,
  Picture,
  VideoPlay,
  ZoomIn,
} from '@danqing/dq-shell';
import { api } from '@/utils/api';
import type { GalleryItem } from '@/types';

const props = defineProps<{
  item: GalleryItem;
  media: 'image' | 'video' | 'audio';
  selectionMode?: boolean;
  selected?: boolean;
}>();

const emit = defineEmits<{
  (e: 'click', event: MouseEvent): void;
  (e: 'toggle-select'): void;
  (e: 'action', payload: { action: string; item: GalleryItem }): void;
}>();

const isHovered = ref(false);
const thumbFailed = ref(false);

const isImage = computed(() => {
  if (props.item.metadata?.asset_kind === 'video' || props.item.metadata?.asset_kind === 'audio') return false;
  const ext = props.item.name?.split('.').pop()?.toLowerCase() || '';
  return ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext);
});

const isVideo = computed(() => {
  if (props.item.metadata?.asset_kind === 'audio') return false;
  if (props.item.metadata?.asset_kind === 'video') return true;
  const ext = props.item.name?.split('.').pop()?.toLowerCase() || '';
  return ['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext);
});

const isAudio = computed(() => {
  if (props.item.metadata?.asset_kind === 'audio') return true;
  const ext = props.item.name?.split('.').pop()?.toLowerCase() || '';
  return ['wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
});

const fileUrl = computed(() => {
  return api.gallery.getImageUrl(props.item.path);
});

const thumbUrl = computed(() => {
  return props.item.thumbnail || fileUrl.value;
});

const durationLabel = computed(() => {
  const raw = props.item.duration_seconds ?? (props.item.metadata?.duration_seconds as number | undefined);
  if (raw == null) return '';
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return '';
  const sec = Math.round(n * 10) / 10;
  return `${sec}s`;
});

const resolutionLabel = computed(() => {
  if (isAudio.value) return durationLabel.value || '—';
  return `${props.item.width || 0}×${props.item.height || 0}`;
});

const audioTileStyle = computed(() => {
  const text = props.item.prompt || props.item.name || '';
  let h = 0;
  for (let i = 0; i < text.length; i += 1) {
    h = (h * 31 + text.charCodeAt(i)) % 360;
  }
  return { '--dq-audio-hue': String(h) } as Record<string, string>;
});

function emitAction(action: string) {
  emit('action', { action, item: props.item });
}

function handleVideoEnter(e: Event) {
  const video = e.target as HTMLVideoElement;
  video.play().catch(() => {});
}

function handleVideoLeave(e: Event) {
  const video = e.target as HTMLVideoElement;
  video.pause();
  video.currentTime = 0;
}

function truncate(text: string, length: number): string {
  if (!text) return '';
  return text.length > length ? text.substring(0, length) + '…' : text;
}
</script>

<style scoped>
.studio-card {
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 8px;
  transition: transform 0.18s ease;
}

.studio-card:hover {
  transform: translateY(-1px);
}

.studio-card--selected .studio-card__media {
  border-color: var(--dq-accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.studio-card__checkbox {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 12;
}

.studio-card__media {
  position: relative;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  border-radius: var(--dq-radius-group);
  border: 0.5px solid var(--dq-glass-border);
  background: var(--dq-surface-inset);
  box-shadow: var(--dq-shadow-sm);
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

.studio-card:hover .studio-card__media {
  border-color: var(--dq-glass-border-strong);
  box-shadow: var(--dq-shadow-md);
}

.studio-card__media img,
.studio-card__media video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.studio-card__fallback {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-label-tertiary);
}

/* Audio tile */
.studio-card__audio-tile {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--dq-label-secondary);
  background:
    radial-gradient(ellipse 90% 70% at 50% 110%, hsl(var(--dq-audio-hue, 220) 55% 42% / 0.28) 0%, transparent 68%),
    radial-gradient(ellipse 50% 40% at 18% 16%, hsl(var(--dq-audio-hue, 220) 45% 38% / 0.12) 0%, transparent 62%),
    linear-gradient(165deg, var(--dq-surface-inset-hover) 0%, var(--dq-bg-elevated) 100%);
  position: relative;
  overflow: hidden;
}

.studio-card__audio-tile::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 3px,
      hsl(var(--dq-audio-hue, 220) 50% 50% / 0.04) 3px,
      hsl(var(--dq-audio-hue, 220) 50% 50% / 0.04) 4px
    );
  pointer-events: none;
}

.studio-card__audio-visualizer {
  position: relative;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 3px;
  height: 32px;
  z-index: 1;
}

.studio-card__audio-bar {
  width: 3px;
  border-radius: 2px;
  background: hsl(var(--dq-audio-hue, 220) 60% 55% / 0.55);
  /* Static symmetric waveform — no motion (avoids implying playback) */
  height: calc(8px + min(var(--bar-i), 7 - var(--bar-i)) * 6px);
  opacity: 0.75;
}

.studio-card__audio-icon {
  opacity: 0.7;
  color: hsl(var(--dq-audio-hue, 220) 50% 60%);
  z-index: 1;
}

.studio-card__audio-label {
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  z-index: 1;
}

.studio-card__audio-dur {
  font-size: 11px;
  font-weight: 500;
  opacity: 0.8;
  color: var(--dq-label-tertiary);
  z-index: 1;
}

/* Play overlay for video */
.studio-card__play-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--dq-overlay-light);
  color: white;
  opacity: 1;
  transition: opacity 0.2s ease;
  pointer-events: none;
}

.studio-card:hover .studio-card__play-overlay {
  opacity: 0;
}

/* Hover overlay */
.studio-card__overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 10px;
}

.studio-card__overlay-bg {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    transparent 0%,
    transparent 50%,
    var(--dq-overlay-gradient-end) 100%
  );
  pointer-events: none;
}

.studio-card__overlay-actions {
  position: relative;
  display: flex;
  gap: 4px;
  justify-content: flex-end;
  opacity: 0;
  transform: translateY(-4px);
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.studio-card:hover .studio-card__overlay-actions {
  opacity: 1;
  transform: translateY(0);
}

.studio-card__overlay-actions :deep(.dq-icon-btn) {
  background: var(--dq-overlay-card);
  border: none;
  color: var(--dq-label-on-media);
  backdrop-filter: var(--dq-glass-blur-light);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
}

.studio-card__overlay-actions :deep(.dq-icon-btn:hover) {
  background: var(--dq-overlay-deep);
}

.studio-card__overlay-info {
  position: relative;
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 8px;
}

.studio-card__overlay-res {
  font-size: 11px;
  color: var(--dq-label-on-media);
  font-weight: 500;
}

.studio-card__overlay-model {
  font-size: 11px;
  color: var(--dq-label-secondary);
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* Footer */
.studio-card__footer {
  padding: 0 2px;
  min-height: 0;
  display: flex;
  align-items: center;
}

.studio-card__prompt {
  font-size: 12px;
  color: var(--dq-label-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.4;
  letter-spacing: -0.01em;
}

.studio-card:hover .studio-card__prompt {
  color: var(--dq-label-primary);
}

.studio-card__prompt--empty {
  color: var(--dq-label-tertiary);
}

/* Transition */
.studio-fade-enter-active,
.studio-fade-leave-active {
  transition: opacity 0.2s ease;
}

.studio-fade-enter-from,
.studio-fade-leave-to {
  opacity: 0;
}
</style>
