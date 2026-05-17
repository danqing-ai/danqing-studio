<!-- @ts-nocheck -->
<template>
  <div
    class="gallery-page"
    :class="{ 'gallery-page--batch': selectedItems.length > 0 }"
  >
    <!-- Main content area -->
    <div class="gallery-page__main">
      <!-- Top toolbar -->
      <div class="gallery-page__toolbar">
        <div class="gallery-page__toolbar-inner">
          <!-- View toggle -->
          <div
            class="gallery-view-segmented"
            role="radiogroup"
            :aria-label="$t('gallery.viewMode')"
          >
            <button
              type="button"
              class="gallery-view-segmented__btn"
              :class="{ 'is-active': viewMode === 'grid' }"
              role="radio"
              :aria-checked="viewMode === 'grid'"
              @click="viewMode = 'grid'"
            >
              <DqIcon><Menu /></DqIcon>
            </button>
            <button
              type="button"
              class="gallery-view-segmented__btn"
              :class="{ 'is-active': viewMode === 'list' }"
              role="radio"
              :aria-checked="viewMode === 'list'"
              @click="viewMode = 'list'"
            >
              <DqIcon><Document /></DqIcon>
            </button>
          </div>

          <span class="dq-vdivider dq-vdivider--zero" aria-hidden="true" />

          <!-- Type filter pills -->
          <div class="gallery-filter-pills">
            <button
              v-for="opt in typeOptions"
              :key="opt.value"
              class="gallery-pill"
              :class="{ active: filterType === opt.value }"
              @click="filterType = opt.value"
            >
              {{ opt.label }}
            </button>
          </div>

          <span class="dq-vdivider dq-vdivider--zero" aria-hidden="true" />

          <!-- Time filter pills -->
          <div class="gallery-filter-pills">
            <button
              v-for="opt in timeOptions"
              :key="opt.value"
              class="gallery-pill"
              :class="{ active: filterTime === opt.value }"
              @click="filterTime = opt.value"
            >
              {{ opt.label }}
            </button>
          </div>

          <!-- 右侧：与左侧筛选同排；窄屏时整组换行，避免 spacer 单独占一行把模型筛选挤到下一行 -->
          <div class="gallery-page__toolbar-tail">
            <DqSelect
              v-model="filterModels"
              size="small"
              multiple
              collapse-tags
              :placeholder="$t('gallery.filterModel')"
              class="gallery-page__model-filter"
            >
              <DqOption v-for="m in allModelOptions" :key="m" :label="m" :value="m" />
            </DqSelect>

            <DqButton size="sm" @click="showAdvancedFilter = true" :title="$t('gallery.advancedFilter')">
              <DqIcon><Filter /></DqIcon>
            </DqButton>

            <DqButton type="primary" size="sm" @click="refresh">
              <DqIcon><Refresh /></DqIcon>
            </DqButton>
          </div>
        </div>
      </div>

      <!-- Content area -->
      <div class="gallery-scroll-area gallery-page__scroll" @scroll="onScroll">
        <!-- Empty state -->
        <div v-if="items.length === 0 && !loading" class="gallery-page__empty">
          <DqEmpty :description="emptyMessage" />
          <DqButton v-if="hasActiveFilters" type="primary" @click="resetFilters">
            {{ $t('gallery.clearFilters') }}
          </DqButton>
        </div>

        <!-- Grid view -->
        <template v-else-if="viewMode === 'grid'">
          <div v-for="group in groupedItems" :key="group.label" class="gallery-group">
            <div class="gallery-group-header">{{ group.label }}</div>
            <div class="gallery-grid">
              <div
                v-for="item in group.items"
                :key="item.path"
                class="gallery-card"
                :class="{ 'is-selected': isSelected(item) }"
                @click="handleCardClick(item, $event)"
              >
                <!-- Selection checkbox -->
                <div class="gallery-checkbox" @click.stop="toggleSelect(item)">
                  <span
                    class="dq-gallery-check"
                    :class="{ 'is-checked': isSelected(item) }"
                    role="checkbox"
                    :aria-checked="isSelected(item)"
                  />
                </div>

                <!-- Media container -->
                <div class="gallery-media-wrapper" @click.stop="showPreview(item)">
                  <template v-if="isImage(item)">
                    <img
                      v-if="!galleryImageLoadFailed[item.path]"
                      :src="getImageUrl(item)"
                      :alt="item.name"
                      loading="lazy"
                      @error="markGalleryImageFailed(item.path)"
                    />
                    <div v-else class="gallery-thumb-fallback">
                      <DqIcon :size="48"><Picture /></DqIcon>
                    </div>
                  </template>
                  <template v-else-if="isAudio(item)">
                    <div class="gallery-audio-tile">
                      <DqIcon class="gallery-audio-tile-icon" :size="42"><Headset /></DqIcon>
                      <span class="gallery-audio-tile-label">{{ $t('gallery.audioLabel') }}</span>
                      <span v-if="formatVideoDuration(item)" class="gallery-audio-tile-dur">{{ formatVideoDuration(item) }}</span>
                    </div>
                  </template>
                  <template v-else-if="isVideo(item)">
                    <video
                      :src="getImageUrl(item)"
                      muted
                      loop
                      preload="metadata"
                      @mouseenter="handleVideoEnter"
                      @mouseleave="handleVideoLeave"
                    ></video>
                    <div class="video-overlay">
                      <DqIcon size="28" color="var(--dq-label-primary)"><VideoPlay /></DqIcon>
                    </div>
                  </template>
                  <template v-else>
                    <div class="gallery-audio-tile">
                      <DqIcon class="gallery-audio-tile-icon" :size="36"><Document /></DqIcon>
                    </div>
                  </template>
                </div>

                <!-- Hover overlay：底部渐变 + 分辨率/模型；点击打开预览 -->
                <div class="gallery-card-overlay" @click.stop="showPreview(item)">
                  <div class="gallery-card-overlay-gradient">
                    <div class="gallery-card-overlay-info">
                      <span class="gallery-card-resolution" v-if="isAudio(item)">{{ formatVideoDuration(item) || '—' }}</span>
                      <span class="gallery-card-resolution" v-else>{{ item.width }}×{{ item.height }}</span>
                      <span v-if="item.model" class="gallery-card-model">{{ item.model }}</span>
                    </div>
                  </div>
                </div>

                <!-- 操作入口：右上角胶囊（el-dropdown 自带 relative，须外包一层做 absolute） -->
                <div class="gallery-card-more" @click.stop>
                  <DqDropdown
                    trigger="click"
                    @command="handleCommand($event, item)"
                  >
                    <DqIconButton
                      type="text"
                      size="sm"
                      class="gallery-card-more-btn"
                      :label="$t('gallery.moreActions')"
                    >
                      <DqIcon><MoreFilled /></DqIcon>
                    </DqIconButton>
                    <template #dropdown>
                      <DqDropdownMenu>
                        <DqDropdownItem command="download">
                          <DqIcon><Download /></DqIcon>
                          {{ $t('gallery.download') }}
                        </DqDropdownItem>
                        <DqDropdownItem command="copyPrompt" v-if="item.prompt">
                          <DqIcon><CopyDocument /></DqIcon>
                          {{ $t('gallery.copyPrompt') }}
                        </DqDropdownItem>
                        <DqDropdownItem command="useForImg2Img" v-if="isImage(item)">
                          <DqIcon><Brush /></DqIcon>
                          {{ $t('gallery.useForImg2Img') }}
                        </DqDropdownItem>
                        <DqDropdownItem command="delete" divided>
                          <DqIcon color="var(--dq-danger)"><Delete /></DqIcon>
                          <DqText type="danger" size="small">{{ $t('gallery.delete') }}</DqText>
                        </DqDropdownItem>
                      </DqDropdownMenu>
                    </template>
                  </DqDropdown>
                </div>

                <!-- Footer info (only shown when not hovering) -->
                <div class="gallery-card-footer">
                  <span v-if="(isVideo(item) || isAudio(item)) && formatVideoDuration(item)" class="gallery-card-duration">
                    {{ formatVideoDuration(item) }}
                  </span>
                  <span class="gallery-card-type-icon" :class="{ 'is-video': isVideo(item), 'is-audio': isAudio(item) }">
                    <DqIcon v-if="isVideo(item)" size="12"><VideoPlay /></DqIcon>
                    <DqIcon v-else-if="isAudio(item)" size="12"><Headset /></DqIcon>
                    <DqIcon v-else size="12"><Picture /></DqIcon>
                  </span>
                </div>
              </div>
            </div>
          </div>
        </template>

        <!-- List view -->
        <div v-else class="gallery-list-view">
          <ul class="gallery-ios-list" role="list">
            <li
              v-for="row in flatItems"
              :key="row.path"
              class="gallery-ios-row"
              :class="{
                'is-selected': isSelected(row),
                'is-active': detailItem?.path === row.path,
              }"
              role="listitem"
              @click="showDetail(row)"
            >
              <button
                type="button"
                class="gallery-ios-row__check"
                @click.stop="toggleSelect(row)"
              >
                <span
                  class="dq-gallery-check"
                  :class="{ 'is-checked': isSelected(row) }"
                  role="checkbox"
                  :aria-checked="isSelected(row)"
                />
              </button>
              <button
                type="button"
                class="gallery-ios-row__thumb"
                @click.stop="showPreview(row)"
              >
                <img
                  v-if="isImage(row) && !galleryImageLoadFailed[row.path]"
                  :src="getImageUrl(row)"
                  alt=""
                  @error="markGalleryImageFailed(row.path)"
                />
                <div
                  v-else-if="isImage(row)"
                  class="gallery-list-thumb-fallback"
                >
                  <DqIcon size="20"><Picture /></DqIcon>
                </div>
                <div v-else-if="isAudio(row)" class="gallery-audio-thumb-sm">
                  <DqIcon size="20"><Headset /></DqIcon>
                </div>
                <div v-else class="gallery-list-thumb-center">
                  <DqIcon size="20"><VideoPlay /></DqIcon>
                </div>
              </button>
              <div class="gallery-ios-row__body">
                <p class="gallery-ios-row__title">{{ truncatePrompt(row.prompt) }}</p>
                <div class="gallery-ios-row__meta">
                  <span v-if="row.model" class="gallery-list-tag">{{ row.model }}</span>
                  <span v-if="row.width && row.height" class="gallery-list-meta">
                    {{ row.width }}×{{ row.height }}
                  </span>
                  <span v-if="(isVideo(row) || isAudio(row)) && formatVideoDuration(row)" class="gallery-list-meta">
                    {{ formatVideoDuration(row) }}
                  </span>
                  <span class="gallery-list-meta">{{ formatRelativeTime(row.created_at) }}</span>
                </div>
              </div>
              <div class="gallery-ios-row__actions" @click.stop>
                <DqIconButton type="text" size="sm" :label="$t('gallery.download')" @click="downloadImage(row)">
                  <DqIcon><Download /></DqIcon>
                </DqIconButton>
                <DqIconButton type="danger" size="sm" :label="$t('common.delete')" @click="deleteImage(row)">
                  <DqIcon><Delete /></DqIcon>
                </DqIconButton>
              </div>
            </li>
          </ul>
        </div>

        <!-- Loading -->
        <div v-if="loading" class="gallery-page__loading">
          <DqIcon class="is-loading" size="28"><Loading /></DqIcon>
        </div>

        <!-- End-of-list hint -->
        <div v-if="!hasMore && items.length > 0" class="gallery-page__end-hint">
          {{ $t('gallery.noMore') }}
        </div>
      </div>
    </div>

    <!-- Right side detail panel -->
    <div v-if="detailItem" class="gallery-detail-panel">
      <div class="gallery-detail-panel__header">
        <span class="gallery-detail-panel__title">{{ $t('gallery.details') }}</span>
        <DqIconButton type="text" size="sm" :label="$t('gallery.close')" @click="detailItem = null">
          <DqIcon><Close /></DqIcon>
        </DqIconButton>
      </div>

      <div class="gallery-detail-panel__body">
        <!-- Large preview -->
        <div class="gallery-detail-panel__preview">
          <template v-if="isImage(detailItem)">
            <img
              v-if="!galleryImageLoadFailed[detailItem.path]"
              :src="getImageUrl(detailItem)"
              @click="showPreview(detailItem)"
              @error="markGalleryImageFailed(detailItem.path)"
            />
            <div v-else class="gallery-detail-panel__preview-fallback" @click="showPreview(detailItem)">
              <DqIcon :size="48"><Picture /></DqIcon>
            </div>
          </template>
          <video v-else :src="getImageUrl(detailItem)" controls></video>
        </div>

        <!-- Action buttons -->
        <div class="gallery-detail-actions">
          <DqButton size="sm" @click="downloadImage(detailItem)">
            <DqIcon><Download /></DqIcon>
            {{ $t('gallery.download') }}
          </DqButton>
          <DqButton v-if="isImage(detailItem)" size="sm" @click="useForImg2Img(detailItem)">
            <DqIcon><Brush /></DqIcon>
            {{ $t('gallery.useForImg2Img') }}
          </DqButton>
        </div>

        <!-- Prompt -->
        <div class="gallery-detail-prompt-box">
          <div class="gallery-detail-prompt-head">
            <span class="gallery-detail-prompt-label">{{ $t('gallery.prompt') }}</span>
            <DqIconButton
              v-if="detailItem.prompt"
              type="text"
              size="sm"
              :label="$t('gallery.copy')"
              @click="copyText(detailItem.prompt)"
            >
              <DqIcon><CopyDocument /></DqIcon>
            </DqIconButton>
          </div>
          <div class="gallery-detail-prompt-text">
            {{ detailItem.prompt || $t('gallery.noPrompt') }}
          </div>
        </div>

        <!-- Param list -->
        <div class="gallery-detail-params">
          <div class="detail-param">
            <span class="detail-param-label">{{ $t('gallery.model') }}</span>
            <span class="detail-param-value">{{ detailItem.model || 'N/A' }}</span>
          </div>
          <div class="detail-param">
            <span class="detail-param-label">{{ $t('gallery.resolution') }}</span>
            <span class="detail-param-value">{{ detailItem.width }}×{{ detailItem.height }}</span>
          </div>
          <div class="detail-param">
            <span class="detail-param-label">{{ $t('gallery.createdAt') }}</span>
            <span class="detail-param-value">{{ formatDate(detailItem.created_at) }}</span>
          </div>
          <div v-if="detailItem.metadata?.steps" class="detail-param">
            <span class="detail-param-label">{{ $t('gallery.steps') }}</span>
            <span class="detail-param-value">{{ detailItem.metadata.steps }}</span>
          </div>
          <div v-if="detailItem.metadata?.guidance" class="detail-param">
            <span class="detail-param-label">{{ $t('gallery.cfg') }}</span>
            <span class="detail-param-value">{{ detailItem.metadata.guidance }}</span>
          </div>
          <div v-if="detailItem.metadata?.seed" class="detail-param">
            <span class="detail-param-label">{{ $t('gallery.seed') }}</span>
            <span class="detail-param-value is-mono">{{ detailItem.metadata.seed }}</span>
          </div>
          <div v-if="isVideo(detailItem) && formatVideoDuration(detailItem)" class="detail-param">
            <span class="detail-param-label">{{ $t('gallery.durationLabel') }}</span>
            <span class="detail-param-value">{{ formatVideoDuration(detailItem) }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Fixed bottom batch action bar -->
    <teleport to="body">
      <div
        v-if="selectedItems.length > 0"
        class="gallery-batch-bar"
        role="toolbar"
        :aria-label="$tt('gallery.selectedCount', { count: selectedItems.length })"
      >
        <div class="gallery-batch-bar-content">
          <div class="gallery-batch-bar__lead">
            <span class="gallery-batch-count">
              {{ $tt('gallery.selectedCount', { count: selectedItems.length }) }}
            </span>
            <DqButton class="gallery-batch-bar__select-all" type="text" size="sm" @click="selectAllPage">
              {{ allPageSelected ? $t('gallery.deselectAll') : $t('gallery.selectAll') }}
            </DqButton>
          </div>
          <div class="gallery-batch-bar__actions">
            <DqButton type="primary" size="sm" round @click="batchDownload">
              <DqIcon><Download /></DqIcon>
              {{ $t('gallery.batchDownload') }}
            </DqButton>
            <DqButton type="danger" size="sm" round @click="batchDelete">
              <DqIcon><Delete /></DqIcon>
              {{ $t('gallery.batchDelete') }}
            </DqButton>
          </div>
          <DqButton
            class="gallery-batch-bar__dismiss dq-btn--icon-circle"
            type="text"
            :aria-label="$t('common.close')"
            @click="selectedItems = []"
          >
            <DqIcon><Close /></DqIcon>
          </DqButton>
        </div>
      </div>
    </teleport>

    <!-- Lightbox full-screen preview -->
    <teleport to="body">
      <div v-if="previewVisible" class="gallery-lightbox" @click="previewVisible = false">
        <div class="lightbox-content" @click.stop>
          <DqIconButton
            class="lightbox-close"
            type="text"
            size="lg"
            :label="$t('gallery.close')"
            @click="previewVisible = false"
          >
            <DqIcon size="24"><Close /></DqIcon>
          </DqIconButton>

          <DqIconButton
            class="lightbox-nav lightbox-prev"
            type="text"
            size="lg"
            :label="$t('gallery.prev')"
            :disabled="selectedIndex <= 0"
            @click="showPrev"
          >
            <DqIcon size="24"><ArrowLeft /></DqIcon>
          </DqIconButton>

          <DqIconButton
            class="lightbox-nav lightbox-next"
            type="text"
            size="lg"
            :label="$t('gallery.next')"
            :disabled="selectedIndex >= flatItems.length - 1"
            @click="showNext"
          >
            <DqIcon size="24"><ArrowRight /></DqIcon>
          </DqIconButton>

          <div class="lightbox-media-wrapper">
            <img
              v-if="selectedItem && isImage(selectedItem) && !galleryImageLoadFailed[selectedItem.path]"
              :src="getImageUrl(selectedItem)"
              :alt="selectedItem.name"
              @error="markGalleryImageFailed(selectedItem.path)"
            />
            <div
              v-else-if="selectedItem && isImage(selectedItem)"
              class="gallery-lightbox-image-fallback"
            >
              <DqIcon :size="64"><Picture /></DqIcon>
            </div>
            <audio
              v-else-if="selectedItem && isAudio(selectedItem)"
              :key="selectedItem.path"
              :src="getImageUrl(selectedItem)"
              controls
              autoplay
              playsinline
              class="gallery-lightbox-audio"
            ></audio>
            <video
              v-else-if="selectedItem"
              :src="getImageUrl(selectedItem)"
              controls
              autoplay
              playsinline
            ></video>
          </div>

          <div class="lightbox-info" v-if="selectedItem">
            <span>{{ selectedItem.name }}</span>
            <span class="gallery-lightbox-muted">
              <template v-if="isAudio(selectedItem)">{{ formatVideoDuration(selectedItem) || '—' }}</template>
              <template v-else>{{ selectedItem.width }}×{{ selectedItem.height }}</template>
              · {{ selectedIndex + 1 }} / {{ flatItems.length }}
            </span>
          </div>
        </div>
      </div>
    </teleport>

    <GalleryAdvancedFilterDrawer
      v-model:visible="showAdvancedFilter"
      v-model:filter-date-range="filterDateRange"
      v-model:filter-min-width="filterMinWidth"
      v-model:filter-actions="filterActions"
      @reset="resetAdvancedFilters"
      @apply="applyAdvancedFilters"
    />
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, computed, onMounted, watch, nextTick } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast, confirm } from '@/utils/feedback';
import {
  Menu,
  Document,
  Filter,
  Refresh,
  Headset,
  VideoPlay,
  Picture,
  MoreFilled,
  Download,
  CopyDocument,
  Brush,
  Delete,
  Close,
  ArrowLeft,
  ArrowRight,
  Loading,
} from '@danqing/dq-shell';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import { DQ_STORAGE } from '@/utils/storage';
import type { GalleryItem } from '@/types';
import { useDocumentEvent } from '@/composables/useDocumentEvent';
import GalleryAdvancedFilterDrawer from '@/components/gallery/GalleryAdvancedFilterDrawer.vue';

const { t: $t } = useI18n();

// Data
const items = ref<GalleryItem[]>([]);
/** 图片加载失败时用占位 UI，避免脚本写内联 style */
const galleryImageLoadFailed = ref<Record<string, boolean>>({});
const loading = ref(false);
const hasMore = ref(true);
const offset = ref(0);
const limit = 40;

// View and filters
const viewMode = ref('grid');
const filterType = ref('all');
const filterModels = ref<string[]>([]);
const filterTime = ref('all');
const filterDateRange = ref<Date[] | null>(null);
const filterMinWidth = ref(256);
const filterActions = ref<string[]>([]);
const showAdvancedFilter = ref(false);

// Selection
const selectedItems = ref<GalleryItem[]>([]);

// Preview and details
const previewVisible = ref(false);
const selectedItem = ref<GalleryItem | null>(null);
const selectedIndex = ref(-1);
const detailItem = ref<GalleryItem | null>(null);

// Options
const typeOptions = computed(() => [
  { value: 'all', label: $t('gallery.filterAll') },
  { value: 'image', label: $t('gallery.filterImage') },
  { value: 'video', label: $t('gallery.filterVideo') },
  { value: 'audio', label: $t('gallery.filterAudio') },
]);

const timeOptions = computed(() => [
  { value: 'all', label: $t('gallery.dateAll') },
  { value: 'today', label: $t('gallery.dateToday') },
  { value: 'week', label: $t('gallery.dateWeek') },
  { value: 'month', label: $t('gallery.dateMonth') },
]);

const allModelOptions = computed(() => {
  const set = new Set<string>();
  items.value.forEach((it) => {
    if (it.model) set.add(it.model);
  });
  return Array.from(set).sort();
});

const flatItems = computed(() => {
  const result: GalleryItem[] = [];
  groupedItems.value.forEach(g => result.push(...g.items));
  return result;
});

const allPageSelected = computed(() => {
  return flatItems.value.length > 0 && flatItems.value.every((it) => isSelected(it));
});

const hasActiveFilters = computed(() => {
  return filterType.value !== 'all' ||
    filterModels.value.length > 0 ||
    filterTime.value !== 'all' ||
    filterDateRange.value !== null ||
    filterMinWidth.value > 256 ||
    filterActions.value.length > 0;
});

const emptyMessage = computed(() => {
  if (hasActiveFilters.value) {
    return $t('gallery.emptyFiltered');
  }
  return $t('gallery.empty');
});

// Date grouping
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

  items.value.forEach(item => {
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

// Build filter params
const buildFilterParams = () => {
  const params: Record<string, unknown> = {};
  if (filterType.value !== 'all') {
    params.kind = filterType.value;
  }
  if (filterModels.value.length > 0) {
    params.model = filterModels.value[0]; // Backend doesn't support multi-select yet, use first
  }
  if (filterTime.value !== 'all') {
    const now = new Date();
    let date: Date | undefined;
    switch (filterTime.value) {
      case 'today':
        date = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        break;
      case 'week':
        date = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        break;
      case 'month':
        date = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
        break;
    }
    if (date) {
      params.created_after = date.toISOString();
    }
  }
  if (filterDateRange.value && filterDateRange.value.length === 2) {
    params.created_after = filterDateRange.value[0].toISOString();
    params.created_before = filterDateRange.value[1].toISOString();
  }
  if (filterMinWidth.value > 256) {
    // Backend doesn't support yet, filter on frontend
  }
  if (filterActions.value.length > 0) {
    // Backend doesn't support yet, filter on frontend
  }
  return params;
};

// Load media list
const loadImages = async (reset = false) => {
  if (loading.value) return;
  if (reset) {
    items.value = [];
    galleryImageLoadFailed.value = {};
    offset.value = 0;
    hasMore.value = true;
    selectedItems.value = [];
    detailItem.value = null;
  }
  if (!hasMore.value && !reset) return;

  loading.value = true;
  try {
    const params = buildFilterParams();
    const rows = await api.gallery.listImages(limit, offset.value, params);
    console.log('GalleryPage loaded rows:', rows.length, rows[0]);
    if (reset) {
      items.value = rows || [];
    } else {
      items.value.push(...(rows || []));
    }

    if ((rows || []).length < limit) {
      hasMore.value = false;
    } else {
      offset.value += limit;
    }
  } catch (e) {
    console.error('Failed to load gallery:', e);
    toast.error($tt('gallery.loadFailed'));
  } finally {
    loading.value = false;
    // Auto-load next page if content doesn't fill the scroll area
    nextTick(() => {
      const scrollArea = document.querySelector('.gallery-scroll-area');
      if (scrollArea && scrollArea.scrollHeight <= scrollArea.clientHeight && hasMore.value && !loading.value) {
        loadImages(false);
      }
    });
  }
};

const refresh = () => {
  loadImages(true);
};

// Infinite scroll
const onScroll = (e: Event) => {
  const el = e.target as HTMLElement;
  const bottom = el.scrollHeight - el.scrollTop - el.clientHeight;
  if (bottom < 200 && !loading.value && hasMore.value) {
    loadImages(false);
  }
};

// Watch filter changes
watch([filterType, filterTime, filterModels], () => {
  loadImages(true);
});

// Selection helpers
const isSelected = (item: GalleryItem) => {
  return selectedItems.value.some((it) => it.path === item.path);
};

const toggleSelect = (item: GalleryItem) => {
  const idx = selectedItems.value.findIndex((it) => it.path === item.path);
  if (idx >= 0) {
    selectedItems.value.splice(idx, 1);
  } else {
    selectedItems.value.push(item);
  }
};

const selectAllPage = () => {
  if (allPageSelected.value) {
    selectedItems.value = [];
  } else {
    selectedItems.value = [...flatItems.value];
  }
};

// Card click handler
const handleCardClick = (item: GalleryItem, event: MouseEvent) => {
  if (event.ctrlKey || event.metaKey || event.shiftKey) {
    event.preventDefault();
    toggleSelect(item);
  }
};

// Batch delete
const batchDelete = async () => {
  if (selectedItems.value.length === 0) return;
  try {
    await confirm(
      $tt('gallery.batchDeleteConfirm', { count: selectedItems.value.length }),
      $tt('gallery.confirmDeleteTitle'),
      { type: 'warning' }
    );

    const paths = selectedItems.value.map((it) => it.path);
    await api.gallery.batchDeleteImages(paths);
    toast.success($tt('gallery.batchDeleted', { count: selectedItems.value.length }));
    selectedItems.value = [];
    loadImages(true);
  } catch (e) {
    if (e !== 'cancel') {
      console.error('Batch delete failed:', e);
      toast.error($tt('gallery.batchDeleteFailed'));
    }
  }
};

// Batch download
const batchDownload = () => {
  selectedItems.value.forEach((item) => {
    downloadImage(item);
  });
};

// Single item operations
const getImageUrl = (item: GalleryItem) => {
  if (!item || !item.path) {
    console.error('GalleryPage: item or item.path is missing', item);
    return '';
  }
  try {
    return api.gallery.getImageUrl(item.path);
  } catch (e) {
    console.error('GalleryPage: getImageUrl failed for path:', item.path, e);
    return '';
  }
};

const thumbUrl = (item: GalleryItem) => {
  if (item.thumbnail) return item.thumbnail;
  return getImageUrl(item);
};

const isImage = (item: GalleryItem | null) => {
  if (!item) return false;
  if (item.metadata?.asset_kind === 'video' || item.metadata?.asset_kind === 'audio') return false;
  const ext = item.name?.split('.').pop()?.toLowerCase();
  return ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext || '');
};

const isAudio = (item: GalleryItem | null) => {
  if (!item) return false;
  if (item.metadata?.asset_kind === 'audio') return true;
  const ext = item.name?.split('.').pop()?.toLowerCase();
  return ['wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext || '');
};

const isVideo = (item: GalleryItem | null) => {
  if (!item) return false;
  if (item.metadata?.asset_kind === 'audio') return false;
  if (item.metadata?.asset_kind === 'video') return true;
  const ext = item.name?.split('.').pop()?.toLowerCase();
  return ['mp4', 'mov', 'avi', 'mkv'].includes(ext || '');
};

const showPreview = (item: GalleryItem) => {
  const idx = flatItems.value.findIndex((it) => it.path === item.path);
  selectedIndex.value = idx >= 0 ? idx : -1;
  selectedItem.value = item;
  previewVisible.value = true;
};

const showDetail = (item: GalleryItem) => {
  detailItem.value = item;
};

const showPrev = () => {
  const idx = selectedIndex.value;
  if (idx <= 0) return;
  selectedIndex.value = idx - 1;
  selectedItem.value = flatItems.value[idx - 1];
};

const showNext = () => {
  const idx = selectedIndex.value;
  if (idx < 0 || idx >= flatItems.value.length - 1) return;
  selectedIndex.value = idx + 1;
  selectedItem.value = flatItems.value[idx + 1];
};

const deleteImage = async (item: GalleryItem) => {
  try {
    await confirm(
      $tt('gallery.confirmDelete'),
      $tt('gallery.confirmDeleteTitle'),
      { type: 'warning' }
    );

    await api.gallery.deleteImage(item.path);
    toast.success($tt('gallery.deleted'));

    const idx = items.value.findIndex((it) => it.path === item.path);
    if (idx >= 0) {
      items.value.splice(idx, 1);
    }
    if (detailItem.value?.path === item.path) {
      detailItem.value = null;
    }
  } catch (e) {
    if (e !== 'cancel') {
      console.error('Failed to delete item:', e);
    }
  }
};

const useForImg2Img = (item: GalleryItem) => {
  localStorage.setItem(DQ_STORAGE.IMG2IMG_REF, item.path);
  toast.success($tt('gallery.img2imgSet'));
};

const downloadImage = (item: GalleryItem) => {
  const url = getImageUrl(item);
  const link = document.createElement('a');
  link.href = url;
  link.download = item.name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  toast.success($tt('gallery.startDownload'));
};

const handleCommand = (command: string, item: GalleryItem) => {
  switch (command) {
    case 'download':
      downloadImage(item);
      break;
    case 'copyPrompt':
      copyText(item.prompt);
      break;
    case 'useForImg2Img':
      useForImg2Img(item);
      break;
    case 'delete':
      deleteImage(item);
      break;
  }
};

// Video interaction
const handleVideoEnter = (e: Event) => {
  const video = e.target as HTMLVideoElement;
  video.play().catch(() => {});
};

const handleVideoLeave = (e: Event) => {
  const video = e.target as HTMLVideoElement;
  video.pause();
  video.currentTime = 0;
};

function markGalleryImageFailed(path: string) {
  if (!path) return;
  galleryImageLoadFailed.value = {
    ...galleryImageLoadFailed.value,
    [path]: true,
  };
}

// Formatting
const formatDate = (dateStr: string) => {
  if (!dateStr) return 'N/A';
  const date = new Date(dateStr);
  return date.toLocaleString();
};

const formatRelativeTime = (dateStr: string) => {
  if (!dateStr) return '';
  const date = new Date(dateStr);
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diff < 60) return $t('gallery.justNow');
  if (diff < 3600) return $tt('gallery.minutesAgo', { m: Math.floor(diff / 60) });
  if (diff < 86400) return $tt('gallery.hoursAgo', { h: Math.floor(diff / 3600) });
  if (diff < 604800) return $tt('gallery.daysAgo', { d: Math.floor(diff / 86400) });
  return formatDate(dateStr);
};

const formatVideoDuration = (item: GalleryItem | null) => {
  if (!item) return '';
  const raw = item.duration_seconds ?? (item.metadata?.duration_seconds as number | undefined);
  if (raw == null || raw === '') return '';
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return '';
  const sec = Math.round(n * 10) / 10;
  return $tt('gallery.durationSecs', { sec });
};

const truncatePrompt = (prompt: string | undefined) => {
  if (!prompt) return '—';
  if (prompt.length <= 60) return prompt;
  return prompt.substring(0, 60) + '...';
};

const copyText = async (text: string | undefined) => {
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    toast.success($tt('gallery.copied'));
  } catch (e) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    toast.success($tt('gallery.copied'));
  }
};

const resetFilters = () => {
  filterType.value = 'all';
  filterModels.value = [];
  filterTime.value = 'all';
  filterDateRange.value = null;
  filterMinWidth.value = 256;
  filterActions.value = [];
};

const resetAdvancedFilters = () => {
  filterDateRange.value = null;
  filterMinWidth.value = 256;
  filterActions.value = [];
};

const applyAdvancedFilters = () => {
  showAdvancedFilter.value = false;
  loadImages(true);
};

// Keyboard navigation
const onKeydown = (e: KeyboardEvent) => {
  if (previewVisible.value) {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      showPrev();
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      showNext();
    } else if (e.key === 'Escape') {
      previewVisible.value = false;
    } else if (e.key === 'd' || e.key === 'D') {
      if (selectedItem.value) {
        downloadImage(selectedItem.value);
      }
    } else if (e.key === 'Delete') {
      if (selectedItem.value) {
        deleteImage(selectedItem.value);
      }
    }
  } else if (e.key === 'Escape') {
    selectedItems.value = [];
  }
};

useDocumentEvent('keydown', onKeydown);

onMounted(() => {
  loadImages(true);
});
</script>
