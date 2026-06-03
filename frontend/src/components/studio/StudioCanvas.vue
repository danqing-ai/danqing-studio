<template>
  <div class="studio-canvas" :class="{ 'studio-canvas--batch': selectedCount > 0 }">
    <!-- Empty state -->
    <div v-if="items.length === 0 && !loading" class="studio-canvas__empty">
      <DqEmpty :description="emptyMessage" />
      <DqButton v-if="hasActiveFilters" type="primary" @click="$emit('reset-filters')">
        {{ $t('gallery.clearFilters') }}
      </DqButton>
    </div>

    <!-- Active tasks (generating placeholders) -->
    <div v-if="activeTasks.length > 0" class="studio-canvas__section">
      <div class="studio-canvas__section-header">
        <span class="studio-canvas__section-dot studio-canvas__section-dot--active" />
        {{ $tt('studio.running') }}
      </div>
      <div class="studio-canvas__grid">
        <ActiveTaskCard
          v-for="task in activeTasks"
          :key="task.id"
          :task="task"
          :media="media"
        />
      </div>
    </div>

    <!-- Completed items grouped by time -->
    <div
      v-for="group in groupedItems"
      :key="group.label"
      class="studio-canvas__section"
    >
      <div class="studio-canvas__section-header">
        {{ group.label }}
      </div>
      <div class="studio-canvas__grid">
        <StudioCard
          v-for="item in group.items"
          :key="item.path"
          :item="item"
          :media="media"
          :selection-mode="selectionMode"
          :selected="isItemSelected(item)"
          @click="handleCardClick(item, $event)"
          @toggle-select="$emit('toggle-select', item)"
          @action="$emit('card-action', $event)"
        />
      </div>
    </div>

    <!-- Loading indicator -->
    <div v-if="loading" class="studio-canvas__loading">
      <DqIcon class="is-loading" size="24"><Loading /></DqIcon>
    </div>

    <!-- Load more -->
    <div v-else-if="hasMore && items.length > 0" class="studio-canvas__load-more">
      <DqButton size="sm" @click="$emit('load-more')">
        {{ $t('common.loadMore') }}
      </DqButton>
    </div>

    <!-- End of list -->
    <div v-if="!hasMore && items.length > 0" class="studio-canvas__end">
      {{ $t('gallery.noMore') }}
    </div>

    <!-- Batch action bar -->
    <teleport to="body">
      <div
        v-if="selectedCount > 0"
        class="gallery-batch-bar"
        role="toolbar"
        :aria-label="$tt('gallery.selectedCount', { count: selectedCount })"
      >
        <div class="gallery-batch-bar-content">
          <div class="gallery-batch-bar__lead">
            <span class="gallery-batch-count">
              {{ $tt('gallery.selectedCount', { count: selectedCount }) }}
            </span>
            <DqButton class="gallery-batch-bar__select-all" type="text" size="sm" @click="$emit('select-all')">
              {{ allSelected ? $t('gallery.deselectAll') : $t('gallery.selectAll') }}
            </DqButton>
          </div>
          <div class="gallery-batch-bar__actions">
            <DqButton type="danger" size="sm" round @click="$emit('batch-delete')">
              <DqIcon><Delete /></DqIcon>
              {{ $t('gallery.batchDelete') }}
            </DqButton>
          </div>
          <DqButton
            class="gallery-batch-bar__dismiss dq-btn--icon-circle"
            type="text"
            :aria-label="$t('common.close')"
            @click="$emit('clear-selection')"
          >
            <DqIcon><Close /></DqIcon>
          </DqButton>
        </div>
      </div>
    </teleport>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, Delete, Loading } from '@danqing/dq-shell';
import { $tt } from '@/utils/i18n';
import type { GalleryItem, Task } from '@/types';
import StudioCard from './StudioCard.vue';
import ActiveTaskCard from './ActiveTaskCard.vue';

const props = defineProps<{
  items: GalleryItem[];
  activeTasks: Task[];
  loading: boolean;
  hasMore: boolean;
  media: 'image' | 'video' | 'audio';
  hasActiveFilters?: boolean;
  selectionMode?: boolean;
  selectedPaths?: Set<string>;
  allSelected?: boolean;
}>();

const emit = defineEmits<{
  (e: 'select', item: GalleryItem): void;
  (e: 'card-action', payload: { action: string; item: GalleryItem }): void;
  (e: 'reset-filters'): void;
  (e: 'load-more'): void;
  (e: 'toggle-select', item: GalleryItem): void;
  (e: 'select-all'): void;
  (e: 'batch-delete'): void;
  (e: 'clear-selection'): void;
}>();

const { t: $t } = useI18n();

const selectedCount = computed(() => props.selectedPaths?.size ?? 0);

const emptyMessage = computed(() => {
  if (props.hasActiveFilters) {
    return $t('gallery.emptyFiltered');
  }
  return $t('gallery.empty');
});

interface Group {
  label: string;
  items: GalleryItem[];
}

const groupedItems = computed(() => {
  const groups: Group[] = [];
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

  const todayItems: GalleryItem[] = [];
  const yesterdayItems: GalleryItem[] = [];
  const weekItems: GalleryItem[] = [];
  const monthItems: GalleryItem[] = [];
  const earlierItems: GalleryItem[] = [];

  props.items.forEach((item) => {
    const date = new Date(item.created_at);
    if (date >= today) {
      todayItems.push(item);
    } else if (date >= yesterday) {
      yesterdayItems.push(item);
    } else if (date >= weekAgo) {
      weekItems.push(item);
    } else if (date >= monthAgo) {
      monthItems.push(item);
    } else {
      earlierItems.push(item);
    }
  });

  if (todayItems.length) groups.push({ label: $t('gallery.groupToday'), items: todayItems });
  if (yesterdayItems.length) groups.push({ label: $t('gallery.groupYesterday'), items: yesterdayItems });
  if (weekItems.length) groups.push({ label: $t('gallery.groupThisWeek'), items: weekItems });
  if (monthItems.length) groups.push({ label: $t('gallery.groupThisMonth'), items: monthItems });
  if (earlierItems.length) groups.push({ label: $t('gallery.groupEarlier'), items: earlierItems });

  return groups;
});

function isItemSelected(item: GalleryItem) {
  return props.selectedPaths?.has(item.path) ?? false;
}

function handleCardClick(item: GalleryItem, event: MouseEvent) {
  if (props.selectionMode) {
    emit('toggle-select', item);
    return;
  }
  emit('select', item);
}
</script>

<style scoped>
.studio-canvas {
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
}

.studio-canvas--batch {
  padding-bottom: 88px;
}

.studio-canvas__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  gap: 16px;
}

.studio-canvas__section {
  margin-bottom: 28px;
}

.studio-canvas__section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  margin-bottom: 14px;
  padding-left: 2px;
}

.studio-canvas__section-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-accent);
  animation: studio-pulse 2s ease-in-out infinite;
}

.studio-canvas__section-dot--active {
  background: var(--dq-accent);
}

@keyframes studio-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.studio-canvas__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 14px 16px;
}

@media (min-width: 768px) {
  .studio-canvas__grid {
    grid-template-columns: repeat(auto-fill, minmax(188px, 1fr));
    gap: 18px 20px;
  }
}

@media (min-width: 1200px) {
  .studio-canvas__grid {
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  }
}

.studio-canvas__loading,
.studio-canvas__load-more {
  display: flex;
  justify-content: center;
  padding: 32px;
}

.studio-canvas__loading {
  color: var(--dq-label-secondary);
}

.studio-canvas__end {
  text-align: center;
  padding: 24px;
  font-size: 13px;
  color: var(--dq-label-tertiary);
}

.is-loading {
  animation: studio-spin 1s linear infinite;
}

@keyframes studio-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>
