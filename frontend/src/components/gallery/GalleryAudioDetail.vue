<!-- @ts-nocheck -->
<template>
  <div
    class="gallery-audio-detail"
    :class="`gallery-audio-detail--${variant}`"
  >
    <AudioMusicPlayer
      :key="item.path"
      :src="src"
      :title="displayTitle"
      :subtitle="displaySubtitle"
      :hue="artHue"
      :duration-seconds="durationSeconds"
      :layout="variant === 'lightbox' ? 'featured' : 'inline'"
      :autoplay="false"
      :show-download="variant === 'lightbox'"
      @download="emit('download')"
    />

    <div class="gallery-audio-detail__lyrics-block studio-audio-effective-lyrics">
      <div class="studio-audio-effective-lyrics__head">
        <span class="studio-audio-effective-lyrics__label">{{ $t('audio.effectiveLyrics') }}</span>
        <div v-if="lyricsText" class="gallery-audio-detail__lyrics-actions">
          <DqButton type="text" size="sm" @click="copyLyrics">
            {{ $t('gallery.copy') }}
          </DqButton>
          <DqButton type="text" size="sm" @click="downloadLyrics">
            {{ $t('audio.downloadLyrics') }}
          </DqButton>
        </div>
      </div>
      <pre v-if="lyricsText" class="studio-audio-effective-lyrics__body">{{ lyricsText }}</pre>
      <p v-else-if="isInstrumental" class="gallery-audio-detail__lyrics-empty">
        {{ $t('gallery.lyricsInstrumental') }}
      </p>
      <p v-else class="gallery-audio-detail__lyrics-empty">
        {{ $t('gallery.lyricsUnavailable') }}
      </p>
      <p v-if="lyricsText" class="studio-field-footnote studio-audio-effective-lyrics__hint">
        {{ $t('audio.effectiveLyricsHint') }}
      </p>
    </div>

    <template v-if="variant === 'lightbox'">
      <GalleryAssetDetailMeta :item="item" show-prompt />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import AudioMusicPlayer from '@/components/audio/AudioMusicPlayer.vue';
import GalleryAssetDetailMeta from '@/components/gallery/GalleryAssetDetailMeta.vue';
import type { GalleryItem } from '@/types';
import { assetDisplayLabel } from '@/utils/assetDisplay';
import { toast } from '@/utils/feedback';
import { $tt } from '@/utils/i18n';

const props = defineProps({
  item: { type: Object as () => GalleryItem, required: true },
  src: { type: String, required: true },
  variant: { type: String, default: 'sidebar' },
  durationLabel: { type: String, default: '' },

});

const emit = defineEmits(['download']);

const { t: $t } = useI18n();

const displayTitle = computed(() => assetDisplayLabel(props.item));

const displaySubtitle = computed(() => {
  const parts: string[] = [];
  if (props.item.model) parts.push(props.item.model);
  if (props.durationLabel) parts.push(props.durationLabel);
  return parts.join(' · ');
});

const durationSeconds = computed(() => {
  const raw =
    props.item.duration_seconds ??
    (props.item.metadata?.duration_seconds as number | undefined);
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 0;
});

const lyricsText = computed(() => {
  const meta = props.item.metadata || {};
  const eff = String(meta.lyrics_effective || '').trim();
  if (eff) return eff;
  return String(meta.lyrics_input || '').trim();
});

const isInstrumental = computed(
  () => props.item.metadata?.lyrics_alignment === 'instrumental',
);

const artHue = computed(() => artHueForPrompt(displayTitle.value));

function artHueForPrompt(text: string) {
  let h = 0;
  const s = String(text || 'audio');
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) % 360;
  }
  return h;
}

async function copyLyrics() {
  if (!lyricsText.value) return;
  try {
    await navigator.clipboard.writeText(lyricsText.value);
    toast.success($tt('gallery.copied'));
  } catch {
    const ta = document.createElement('textarea');
    ta.value = lyricsText.value;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    toast.success($tt('gallery.copied'));
  }
}

function downloadLyrics() {
  if (!lyricsText.value) return;
  const blob = new Blob([lyricsText.value + '\n'], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const base = (props.item.name || 'lyrics').replace(/\.[^.]+$/, '');
  a.href = url;
  a.download = base + '_lyrics.txt';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
</script>

<style scoped>
/* Lightbox overrides for dark theme readability */
.gallery-audio-detail--lightbox .gallery-audio-detail__lyrics-block {
  background: var(--dq-surface-inset, var(--dq-fill-tertiary));
  border: 0.5px solid var(--dq-border-subtle);
}

.gallery-audio-detail--lightbox .studio-audio-effective-lyrics__body {
  color: var(--dq-label-primary);
}
</style>
