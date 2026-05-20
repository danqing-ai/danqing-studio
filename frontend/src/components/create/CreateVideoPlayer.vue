<!-- @ts-nocheck -->
<template>
  <div
    class="dq-music-player dq-media-player dq-media-player--video"
    :class="{ 'is-playing': isPlaying, 'is-loading': isLoading }"
    :style="{ '--dq-music-hue': String(hue) }"
  >
    <video
      ref="videoEl"
      class="dq-media-player__video"
      :src="src"
      preload="metadata"
      playsinline
      @loadedmetadata="onLoadedMetadata"
      @timeupdate="onTimeUpdate"
      @ended="onEnded"
      @play="onPlay"
      @pause="onPause"
      @waiting="isLoading = true"
      @canplay="isLoading = false"
    />

    <div class="dq-media-player__screen">
      <div class="dq-media-player__screen-vignette" />
      <button
        type="button"
        class="dq-media-player__screen-play"
        :disabled="!src"
        :aria-label="isPlaying ? $t('audio.pause') : $t('audio.play')"
        @click="togglePlay"
      >
        <DqIcon :size="36">
          <pause v-if="isPlaying" />
          <play v-else />
        </DqIcon>
      </button>
    </div>

    <div class="dq-media-player__meta dq-media-player__meta--below">
      <p class="dq-music-player__eyebrow">{{ $t('studio.previewNow') }}</p>
      <h3 class="dq-music-player__title" :title="title">{{ displayTitle }}</h3>
      <p v-if="subtitle" class="dq-music-player__subtitle">{{ subtitle }}</p>
    </div>

    <div
      class="dq-music-player__progress"
      role="slider"
      :aria-valuenow="Math.round(progressPct)"
      aria-valuemin="0"
      aria-valuemax="100"
      :aria-label="$t('audio.seek')"
      @click="onSeek"
    >
      <div class="dq-music-player__progress-track">
        <div class="dq-music-player__progress-buffer" :style="{ width: bufferPct + '%' }" />
        <div class="dq-music-player__progress-fill" :style="{ width: progressPct + '%' }" />
      </div>
    </div>

    <div class="dq-music-player__transport">
      <span class="dq-music-player__time">{{ formatClock(currentTime) }}</span>
      <div class="dq-music-player__controls">
        <button
          type="button"
          class="dq-music-player__btn-play"
          :disabled="!src"
          :aria-label="isPlaying ? $t('audio.pause') : $t('audio.play')"
          @click="togglePlay"
        >
          <DqIcon :size="26">
            <pause v-if="isPlaying" />
            <play v-else />
          </DqIcon>
        </button>
        <DqIconButton
          v-if="showDownload && src"
          type="text"
          size="sm"
          class="dq-music-player__btn-dl"
          :label="$t('gallery.download')"
          @click="emit('download')"
        >
          <DqIcon><download /></DqIcon>
        </DqIconButton>
      </div>
      <span class="dq-music-player__time">{{ formatClock(duration) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';

const props = defineProps({
  src: { type: String, default: '' },
  title: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  showDownload: { type: Boolean, default: true },
  hue: { type: Number, default: 268 },
});

const emit = defineEmits(['download', 'play', 'pause', 'duration']);

const videoEl = ref<HTMLVideoElement | null>(null);
const isPlaying = ref(false);
const isLoading = ref(false);
const currentTime = ref(0);
const duration = ref(0);
const bufferEnd = ref(0);

const displayTitle = computed(() => (props.title || '').trim() || 'Untitled');

const progressPct = computed(() => {
  if (!duration.value) return 0;
  return Math.min(100, (currentTime.value / duration.value) * 100);
});

const bufferPct = computed(() => {
  if (!duration.value) return 0;
  return Math.min(100, (bufferEnd.value / duration.value) * 100);
});

function formatClock(sec: number) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  return m + ':' + String(s % 60).padStart(2, '0');
}

function onLoadedMetadata() {
  const el = videoEl.value;
  if (!el) return;
  duration.value = Number.isFinite(el.duration) ? el.duration : 0;
  isLoading.value = false;
  if (duration.value > 0) emit('duration', duration.value);
}

function onTimeUpdate() {
  const el = videoEl.value;
  if (!el) return;
  currentTime.value = el.currentTime;
  if (el.buffered.length > 0) {
    bufferEnd.value = el.buffered.end(el.buffered.length - 1);
  }
}

function onEnded() {
  isPlaying.value = false;
  currentTime.value = 0;
  emit('pause');
}

function onPlay() {
  isPlaying.value = true;
  emit('play');
}

function onPause() {
  isPlaying.value = false;
  emit('pause');
}

function onSeek(ev: MouseEvent) {
  const el = videoEl.value;
  const track = ev.currentTarget as HTMLElement;
  if (!el || !duration.value || !track) return;
  const rect = track.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
  el.currentTime = ratio * duration.value;
  currentTime.value = el.currentTime;
}

function togglePlay() {
  const el = videoEl.value;
  if (!el || !props.src) return;
  if (isPlaying.value) el.pause();
  else el.play().catch(() => {});
}

function load() {
  const el = videoEl.value;
  if (!el) return;
  currentTime.value = 0;
  duration.value = 0;
  isPlaying.value = false;
  try {
    el.load();
  } catch {
    /* ignore */
  }
}

watch(() => props.src, () => load());

defineExpose({ load, togglePlay, pause: () => videoEl.value?.pause() });
</script>
