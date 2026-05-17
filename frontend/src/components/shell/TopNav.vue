<template>
  <div class="dq-top-nav-bar">
    <div class="header-brand">
      <DqIcon class="dq-top-nav-brand-icon" :size="28"><MagicStick /></DqIcon>
      <span class="brand-title">DanQing Studio</span>
      <span class="brand-subtitle">v4</span>
    </div>

    <nav class="nav-menu dq-top-nav-menu" role="navigation" :aria-label="$t('nav.main')">
      <button
        v-for="item in navItems"
        :key="item.id"
        type="button"
        class="dq-top-nav-menu__item"
        :class="{ 'is-active': activePage === item.id }"
        @click="onNavSelect(item.id)"
      >
        <DqIcon><component :is="item.icon" /></DqIcon>
        <span>{{ item.label }}</span>
      </button>
    </nav>

    <div class="header-actions">
      <DqCountBadge :value="queueCount" :hidden="queueCount === 0" class="queue-badge">
        <DqButton class="dq-top-nav-queue-btn" @click="openQueue" :title="$tt('studio.taskQueue')">
          <DqIcon><DocumentCopy /></DqIcon>
          <span class="dq-top-nav-queue-label">{{ $tt('studio.taskQueue') }}</span>
        </DqButton>
      </DqCountBadge>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  Brush,
  DocumentCopy,
  Download,
  MagicStick,
  Microphone,
  PictureFilled,
  Setting,
  VideoCamera,
} from '@danqing/dq-shell';

defineProps<{
  activePage: string;
  queueCount: number;
}>();

const emit = defineEmits<{
  (e: 'navigate', page: string): void;
  (e: 'open-queue'): void;
}>();

const { t } = useI18n();

const navItems = computed(() => [
  { id: 'image_create', icon: Brush, label: t('nav.image_create') },
  { id: 'video_create', icon: VideoCamera, label: t('nav.video_create') },
  { id: 'audio_create', icon: Microphone, label: t('nav.audio_create') },
  { id: 'gallery', icon: PictureFilled, label: t('nav.gallery') },
  { id: 'models', icon: Download, label: t('nav.models') },
  { id: 'settings', icon: Setting, label: t('nav.settings') },
]);

function onNavSelect(index: string) {
  emit('navigate', index);
}

function openQueue() {
  emit('open-queue');
}
</script>
