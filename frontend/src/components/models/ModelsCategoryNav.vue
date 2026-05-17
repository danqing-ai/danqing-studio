<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  Aim,
  Download,
  FolderChecked,
  Grid,
  Headset,
  MagicStick,
  PictureFilled,
  Tools,
  VideoCamera,
  ZoomIn,
} from '@danqing/dq-shell';

const props = defineProps<{
  activeCategory: string;
  totalModelCount: number;
  activeDownloadCount: number;
}>();

const emit = defineEmits<{
  select: [category: string];
}>();

const { t } = useI18n();

type NavEntry =
  | { kind: 'divider' }
  | {
      kind: 'item';
      id: string;
      icon: object;
      label: string;
      count?: number;
      badge?: number;
      badgeType?: 'info' | 'primary';
    };

const entries = computed<NavEntry[]>(() => [
  {
    kind: 'item',
    id: 'all',
    icon: Grid,
    label: t('download.allModels'),
    count: props.totalModelCount,
    badgeType: 'info',
  },
  { kind: 'item', id: 'image_models', icon: PictureFilled, label: t('download.imageModels') },
  { kind: 'item', id: 'video_models', icon: VideoCamera, label: t('download.videoModels') },
  { kind: 'item', id: 'music_models', icon: Headset, label: t('download.audioModels') },
  { kind: 'item', id: 'controlnets', icon: Aim, label: t('download.controlNet') },
  { kind: 'item', id: 'upscalers', icon: ZoomIn, label: t('download.upscalers') },
  { kind: 'item', id: 'tools', icon: Tools, label: t('download.tools') },
  { kind: 'item', id: 'loras', icon: MagicStick, label: t('download.loraModels') },
  { kind: 'divider' },
  {
    kind: 'item',
    id: 'downloading',
    icon: Download,
    label: t('download.downloadingTab'),
    badge: props.activeDownloadCount,
    badgeType: 'primary',
  },
  { kind: 'item', id: 'installed', icon: FolderChecked, label: t('download.installed') },
]);
</script>

<template>
  <nav class="dq-download-menu models-page__menu" role="navigation" :aria-label="$t('download.downloadCenter')">
    <template v-for="(entry, idx) in entries" :key="entry.kind === 'divider' ? `div-${idx}` : entry.id">
      <hr v-if="entry.kind === 'divider'" class="models-page__menu-divider" />
      <button
        v-else
        type="button"
        class="dq-download-menu__item"
        :class="{ 'is-active': activeCategory === entry.id }"
        @click="emit('select', entry.id)"
      >
        <DqIcon class="dq-download-menu__icon"><component :is="entry.icon" /></DqIcon>
        <span class="dq-download-menu__label">{{ entry.label }}</span>
        <DqTag
          v-if="entry.count !== undefined"
          size="small"
          :type="entry.badgeType ?? 'info'"
          class="dq-menu-end-tag"
        >
          {{ entry.count }}
        </DqTag>
        <DqTag
          v-else-if="entry.badge && entry.badge > 0"
          size="small"
          :type="entry.badgeType ?? 'primary'"
          class="dq-menu-end-tag"
        >
          {{ entry.badge }}
        </DqTag>
      </button>
    </template>
  </nav>
</template>
