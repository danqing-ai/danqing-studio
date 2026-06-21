<template>
  <footer v-if="assetId" class="lv-output dq-glass--bar">
    <div class="lv-output__inner">
      <div class="lv-output__head">
        <span class="lv-output__title">{{ $tt('video.longVideoFinalPreview') }}</span>
        <DqButton size="sm" type="text" @click="$emit('open-gallery')">
          {{ $tt('video.longVideoOpenInGallery') }}
        </DqButton>
      </div>
      <video
        class="lv-output__video"
        :src="videoUrl"
        controls
        playsinline
        preload="metadata"
      />
    </div>
  </footer>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  assetId?: string;
}>();

defineEmits<{
  (e: 'open-gallery'): void;
}>();

const { t: $tt } = useI18n();

const videoUrl = computed(() => (props.assetId ? `/api/assets/${props.assetId}/file` : ''));
</script>
