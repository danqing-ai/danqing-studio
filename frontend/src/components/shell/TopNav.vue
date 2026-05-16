<template>
  <div class="dq-top-nav-bar">
    <div class="header-brand">
      <el-icon class="dq-top-nav-brand-icon" :size="28"><magic-stick /></el-icon>
      <span class="brand-title">DanQing Studio</span>
      <span class="brand-subtitle">v4</span>
    </div>

    <el-menu
      ref="navMenuRef"
      :default-active="activePage"
      mode="horizontal"
      class="nav-menu"
      @select="onNavSelect"
    >
      <el-menu-item index="image_create">
        <el-icon><brush /></el-icon>
        <span>{{ $t('nav.image_create') }}</span>
      </el-menu-item>
      <el-menu-item index="video_create">
        <el-icon><video-camera /></el-icon>
        <span>{{ $t('nav.video_create') }}</span>
      </el-menu-item>
      <el-menu-item index="audio_create">
        <el-icon><microphone /></el-icon>
        <span>{{ $t('nav.audio_create') }}</span>
      </el-menu-item>
      <el-menu-item index="gallery">
        <el-icon><picture-filled /></el-icon>
        <span>{{ $t('nav.gallery') }}</span>
      </el-menu-item>
      <el-menu-item index="models">
        <el-icon><download /></el-icon>
        <span>{{ $t('nav.models') }}</span>
      </el-menu-item>
      <el-menu-item index="settings">
        <el-icon><setting /></el-icon>
        <span>{{ $t('nav.settings') }}</span>
      </el-menu-item>
    </el-menu>

    <div class="header-actions">
      <el-badge :value="queueCount" :hidden="queueCount === 0" class="queue-badge">
        <el-button class="dq-top-nav-queue-btn" @click="openQueue" :title="$tt('studio.taskQueue')">
          <el-icon><document-copy /></el-icon>
          <span class="dq-top-nav-queue-label">{{ $tt('studio.taskQueue') }}</span>
        </el-button>
      </el-badge>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue';

const props = defineProps<{
  activePage: string;
  queueCount: number;
}>();

const emit = defineEmits<{
  (e: 'navigate', page: string): void;
  (e: 'open-queue'): void;
}>();

const navMenuRef = ref<{ activeIndex: string } | null>(null);

function onNavSelect(index: string) {
  emit('navigate', index);
}

function openQueue() {
  emit('open-queue');
}

watch(
  () => props.activePage,
  (newVal) => {
    nextTick(() => {
      if (navMenuRef.value) {
        navMenuRef.value.activeIndex = newVal;
      }
    });
  }
);
</script>