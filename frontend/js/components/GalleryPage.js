/**
 * Gallery page component v3 — Original aspect ratio compact grid, date grouping, Midjourney-style hover interaction
 */

const GalleryPage = {
    template: `
        <div class="gallery-page" style="display: flex; height: 100%; gap: 0;">
            <!-- Main content area -->
            <div style="flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden;">
                <!-- Top toolbar -->
                <div style="padding: 12px 16px 8px; border-bottom: 1px solid var(--border-color); flex-shrink: 0;">
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <!-- View toggle -->
                        <el-radio-group v-model="viewMode" size="small">
                            <el-radio-button label="grid">
                                <el-icon><Menu /></el-icon>
                            </el-radio-button>
                            <el-radio-button label="list">
                                <el-icon><Document /></el-icon>
                            </el-radio-button>
                        </el-radio-group>

                        <el-divider direction="vertical" style="margin: 0;" />

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

                        <el-divider direction="vertical" style="margin: 0;" />

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

                        <div style="flex: 1;"></div>

                        <!-- Model filter -->
                        <el-select 
                            v-model="filterModels" 
                            size="small" 
                            multiple 
                            collapse-tags
                            :placeholder="$t('gallery.filterModel')" 
                            style="width: 160px;"
                        >
                            <el-option v-for="m in allModelOptions" :key="m" :label="m" :value="m" />
                        </el-select>

                        <!-- More filters -->
                        <el-button size="small" @click="showAdvancedFilter = true">
                            <el-icon><filter /></el-icon>
                        </el-button>

                        <el-button @click="refresh" type="primary" plain size="small">
                            <el-icon><refresh /></el-icon>
                        </el-button>
                    </div>
                </div>

                <!-- Content area -->
                <div class="gallery-scroll-area" style="flex: 1; overflow-y: auto; padding: 12px 16px;" @scroll="onScroll">
                    <!-- Empty state -->
                    <div v-if="items.length === 0 && !loading" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 16px;">
                        <el-empty :description="emptyMessage" />
                        <el-button v-if="hasActiveFilters" @click="resetFilters" type="primary">
                            {{ $t('gallery.clearFilters') }}
                        </el-button>
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
                                        <el-checkbox :model-value="isSelected(item)" size="small" />
                                    </div>

                                    <!-- Media container -->
                                    <div class="gallery-media-wrapper" @click.stop="showPreview(item)">
                                        <template v-if="isImage(item)">
                                            <img :src="getImageUrl(item)" :alt="item.name" loading="lazy" @error="handleImageError" />
                                        </template>
                                        <template v-else>
                                            <video
                                                :src="getImageUrl(item)"
                                                muted
                                                loop
                                                preload="metadata"
                                                @mouseenter="handleVideoEnter"
                                                @mouseleave="handleVideoLeave"
                                            ></video>
                                            <div class="video-overlay">
                                                <el-icon size="28" color="white"><video-play /></el-icon>
                                            </div>
                                        </template>
                                    </div>
                                    
                                    <!-- Hover overlay -->
                                    <div class="gallery-card-overlay">
                                        <div class="gallery-card-overlay-content">
                                            <el-button 
                                                type="primary" 
                                                size="large" 
                                                circle
                                                @click.stop="showPreview(item)"
                                            >
                                                <el-icon size="20"><zoom-in /></el-icon>
                                            </el-button>
                                            <div class="gallery-card-overlay-info">
                                                <span class="gallery-card-resolution">{{ item.width }}×{{ item.height }}</span>
                                                <span v-if="item.model" class="gallery-card-model">{{ item.model }}</span>
                                            </div>
                                        </div>
                                        <!-- More actions dropdown -->
                                        <el-dropdown 
                                            trigger="click" 
                                            @command="handleCommand($event, item)"
                                            @click.stop
                                            class="gallery-card-more"
                                        >
                                            <el-button text size="small" circle>
                                                <el-icon color="white"><more-filled /></el-icon>
                                            </el-button>
                                            <template #dropdown>
                                                <el-dropdown-menu>
                                                    <el-dropdown-item command="download">
                                                        <el-icon><download /></el-icon>
                                                        {{ $t('gallery.download') }}
                                                    </el-dropdown-item>
                                                    <el-dropdown-item command="copyPrompt" v-if="item.prompt">
                                                        <el-icon><copy-document /></el-icon>
                                                        {{ $t('gallery.copyPrompt') }}
                                                    </el-dropdown-item>
                                                    <el-dropdown-item command="useForImg2Img" v-if="isImage(item)">
                                                        <el-icon><brush /></el-icon>
                                                        {{ $t('gallery.useForImg2Img') }}
                                                    </el-dropdown-item>
                                                    <el-dropdown-item command="delete" divided>
                                                        <el-icon color="#f56c6c"><delete /></el-icon>
                                                        <span style="color: #f56c6c;">{{ $t('gallery.delete') }}</span>
                                                    </el-dropdown-item>
                                                </el-dropdown-menu>
                                            </template>
                                        </el-dropdown>
                                    </div>

                                    <!-- Footer info (only shown when not hovering) -->
                                    <div class="gallery-card-footer">
                                        <span v-if="isVideo(item) && formatVideoDuration(item)" class="gallery-card-duration">
                                            {{ formatVideoDuration(item) }}
                                        </span>
                                        <span class="gallery-card-type-icon" :class="{ 'is-video': isVideo(item) }">
                                            <el-icon v-if="isVideo(item)" size="12"><video-play /></el-icon>
                                            <el-icon v-else size="12"><picture /></el-icon>
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </template>

                    <!-- List view -->
                    <div v-else class="gallery-list-view">
                        <el-table
                            :data="flatItems"
                            style="width: 100%;"
                            size="small"
                            @row-click="showDetail"
                            highlight-current-row
                        >
                            <el-table-column type="selection" width="40">
                                <template #default="{ row }">
                                    <el-checkbox :model-value="isSelected(row)" @click.stop="toggleSelect(row)" />
                                </template>
                            </el-table-column>
                            <el-table-column width="60">
                                <template #default="{ row }">
                                    <div style="width: 40px; height: 40px; border-radius: 4px; overflow: hidden; background: var(--bg-tertiary);">
                                        <img v-if="isImage(row)" :src="getImageUrl(row)" style="width: 100%; height: 100%; object-fit: cover;" @click.stop="showPreview(row)" @error="handleImageError" />
                                        <div v-else style="width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">
                                            <el-icon size="18"><video-play /></el-icon>
                                        </div>
                                    </div>
                                </template>
                            </el-table-column>
                            <el-table-column :label="$t('gallery.prompt')" min-width="200">
                                <template #default="{ row }">
                                    <el-tooltip :content="row.prompt || $t('gallery.noPrompt')" placement="top" :show-after="500">
                                        <span class="gallery-list-prompt">{{ truncatePrompt(row.prompt) }}</span>
                                    </el-tooltip>
                                </template>
                            </el-table-column>
                            <el-table-column :label="$t('gallery.model')" width="120">
                                <template #default="{ row }">
                                    <el-tag size="small" effect="plain" v-if="row.model">{{ row.model }}</el-tag>
                                    <span v-else style="color: var(--text-muted);">—</span>
                                </template>
                            </el-table-column>
                            <el-table-column :label="$t('gallery.resolution')" width="90">
                                <template #default="{ row }">
                                    <span style="font-size: 12px; color: var(--text-muted);">{{ row.width }}×{{ row.height }}</span>
                                </template>
                            </el-table-column>
                            <el-table-column width="50">
                                <template #default="{ row }">
                                    <el-icon v-if="isVideo(row)" size="16"><video-play /></el-icon>
                                    <el-icon v-else size="16"><picture /></el-icon>
                                </template>
                            </el-table-column>
                            <el-table-column width="80">
                                <template #default="{ row }">
                                    <span v-if="isVideo(row) && formatVideoDuration(row)" style="font-size: 12px; color: var(--text-muted);">
                                        {{ formatVideoDuration(row) }}
                                    </span>
                                </template>
                            </el-table-column>
                            <el-table-column width="100">
                                <template #default="{ row }">
                                    <span style="font-size: 12px; color: var(--text-muted);">{{ formatRelativeTime(row.created_at) }}</span>
                                </template>
                            </el-table-column>
                            <el-table-column width="80" fixed="right">
                                <template #default="{ row }">
                                    <el-button size="small" text @click.stop="downloadImage(row)">
                                        <el-icon><download /></el-icon>
                                    </el-button>
                                    <el-button size="small" text type="danger" @click.stop="deleteImage(row)">
                                        <el-icon><delete /></el-icon>
                                    </el-button>
                                </template>
                            </el-table-column>
                        </el-table>
                    </div>

                    <!-- Loading -->
                    <div v-if="loading" style="text-align: center; padding: 32px;">
                        <el-icon class="is-loading" size="28"><loading /></el-icon>
                    </div>

                    <!-- End-of-list hint -->
                    <div v-if="!hasMore && items.length > 0" style="text-align: center; padding: 24px; color: var(--text-muted); font-size: 13px;">
                        {{ $t('gallery.noMore') }}
                    </div>
                </div>
            </div>

            <!-- Right side detail panel -->
            <div v-if="detailItem" class="gallery-detail-panel" style="width: 320px; flex-shrink: 0; border-left: 1px solid var(--border-color); background: var(--bg-secondary); display: flex; flex-direction: column;">
                <div style="padding: 16px; border-bottom: 1px solid var(--border-color); display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-weight: 600; font-size: 14px;">{{ $t('gallery.details') }}</span>
                    <el-button text size="small" @click="detailItem = null">
                        <el-icon><close /></el-icon>
                    </el-button>
                </div>
                
                <div style="flex: 1; overflow-y: auto; padding: 16px;">
                    <!-- Large preview -->
                    <div style="margin-bottom: 16px; border-radius: 8px; overflow: hidden; background: var(--bg-tertiary);">
                        <img v-if="isImage(detailItem)" :src="getImageUrl(detailItem)" style="width: 100%; display: block; cursor: pointer;" @click="showPreview(detailItem)" />
                        <video v-else :src="getImageUrl(detailItem)" controls style="width: 100%; display: block;"></video>
                    </div>

                    <!-- Action buttons -->
                    <div style="display: flex; gap: 8px; margin-bottom: 16px;">
                        <el-button size="small" @click="downloadImage(detailItem)" style="flex: 1;">
                            <el-icon><download /></el-icon>
                            {{ $t('gallery.download') }}
                        </el-button>
                        <el-button size="small" @click="useForImg2Img(detailItem)" v-if="isImage(detailItem)" style="flex: 1;">
                            <el-icon><brush /></el-icon>
                            {{ $t('gallery.useForImg2Img') }}
                        </el-button>
                    </div>

                    <!-- Prompt -->
                    <div class="gallery-detail-prompt-box">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-size: 12px; color: var(--text-muted);">{{ $t('gallery.prompt') }}</span>
                            <el-button v-if="detailItem.prompt" size="small" text type="primary" @click="copyText(detailItem.prompt)">
                                <el-icon><copy-document /></el-icon>
                            </el-button>
                        </div>
                        <div class="gallery-detail-prompt-text">
                            {{ detailItem.prompt || $t('gallery.noPrompt') }}
                        </div>
                    </div>

                    <!-- Param list -->
                    <div style="display: grid; gap: 8px;">
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
                            <span class="detail-param-value" style="font-family: monospace;">{{ detailItem.metadata.seed }}</span>
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
                <div v-if="selectedItems.length > 0" class="gallery-batch-bar">
                    <div class="gallery-batch-bar-content">
                        <span class="gallery-batch-count">
                            {{ $tt('gallery.selectedCount', { count: selectedItems.length }) }}
                        </span>
                        <div class="gallery-batch-actions">
                            <el-button size="small" @click="selectAllPage">
                                {{ allPageSelected ? $t('gallery.deselectAll') : $t('gallery.selectAll') }}
                            </el-button>
                            <el-button size="small" type="primary" @click="batchDownload">
                                <el-icon><download /></el-icon>
                                {{ $t('gallery.batchDownload') }}
                            </el-button>
                            <el-button size="small" type="danger" @click="batchDelete">
                                <el-icon><delete /></el-icon>
                                {{ $t('gallery.batchDelete') }}
                            </el-button>
                            <el-button size="small" text @click="selectedItems = []">
                                <el-icon><close /></el-icon>
                            </el-button>
                        </div>
                    </div>
                </div>
            </teleport>

            <!-- Lightbox full-screen preview -->
            <teleport to="body">
                <div v-if="previewVisible" class="gallery-lightbox" @click="previewVisible = false">
                    <div class="lightbox-content" @click.stop>
                        <el-button class="lightbox-close" circle text size="large" @click="previewVisible = false">
                            <el-icon size="24"><close /></el-icon>
                        </el-button>

                        <el-button
                            class="lightbox-nav lightbox-prev"
                            circle
                            size="large"
                            :disabled="selectedIndex <= 0"
                            @click="showPrev"
                        >
                            <el-icon size="24"><arrow-left /></el-icon>
                        </el-button>

                        <el-button
                            class="lightbox-nav lightbox-next"
                            circle
                            size="large"
                            :disabled="selectedIndex >= flatItems.length - 1"
                            @click="showNext"
                        >
                            <el-icon size="24"><arrow-right /></el-icon>
                        </el-button>

                        <div class="lightbox-media-wrapper">
                            <img v-if="isImage(selectedItem)" :src="getImageUrl(selectedItem)" :alt="selectedItem.name" />
                            <video v-else :src="getImageUrl(selectedItem)" controls autoplay></video>
                        </div>

                        <div class="lightbox-info">
                            <span>{{ selectedItem.name }}</span>
                            <span style="color: var(--text-muted);">
                                {{ selectedItem.width }}×{{ selectedItem.height }}
                                · {{ selectedIndex + 1 }} / {{ flatItems.length }}
                            </span>
                        </div>
                    </div>
                </div>
            </teleport>

            <!-- Advanced filter drawer -->
            <el-drawer
                v-model="showAdvancedFilter"
                :title="$t('gallery.advancedFilter')"
                direction="rtl"
                size="320px"
            >
                <div style="display: flex; flex-direction: column; gap: 20px;">
                    <div>
                        <div style="font-weight: 600; margin-bottom: 10px; font-size: 13px;">{{ $t('gallery.dateRange') }}</div>
                        <el-date-picker
                            v-model="filterDateRange"
                            type="daterange"
                            size="small"
                            style="width: 100%;"
                            :start-placeholder="$t('gallery.startDate')"
                            :end-placeholder="$t('gallery.endDate')"
                        />
                    </div>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 10px; font-size: 13px;">{{ $t('gallery.minResolution') }}</div>
                        <el-slider v-model="filterMinWidth" :min="256" :max="2048" :step="64" show-stops />
                        <div style="text-align: center; font-size: 12px; color: var(--text-muted);">≥ {{ filterMinWidth }}px</div>
                    </div>
                    <div>
                        <div style="font-weight: 600; margin-bottom: 10px; font-size: 13px;">{{ $t('gallery.actionType') }}</div>
                        <el-checkbox-group v-model="filterActions">
                            <el-checkbox label="create">{{ $t('gallery.actionCreate') }}</el-checkbox>
                            <el-checkbox label="rewrite">{{ $t('gallery.actionRewrite') }}</el-checkbox>
                            <el-checkbox label="upscale">{{ $t('gallery.actionUpscale') }}</el-checkbox>
                        </el-checkbox-group>
                    </div>
                </div>
                <template #footer>
                    <div style="display: flex; gap: 10px;">
                        <el-button @click="resetAdvancedFilters" size="small">{{ $t('gallery.resetFilters') }}</el-button>
                        <el-button type="primary" @click="applyAdvancedFilters" size="small">{{ $t('gallery.apply') }}</el-button>
                    </div>
                </template>
            </el-drawer>
        </div>
    `,
    
    setup() {
        const { ref, computed, onMounted, onUnmounted, watch, getCurrentInstance, nextTick } = Vue;
        const { Menu, Document } = ElementPlusIconsVue;

        // Get i18n translation functions
        const instance = getCurrentInstance();
        const $t = instance ? instance.proxy.$t : (key) => key;
        const $tt = window.$tt || ((key, params) => key);

        // Data
        const items = ref([]);
        const loading = ref(false);
        const hasMore = ref(true);
        const offset = ref(0);
        const limit = 40;

        // View and filters
        const viewMode = ref('grid');
        const filterType = ref('all');
        const filterModels = ref([]);
        const filterTime = ref('all');
        const filterDateRange = ref(null);
        const filterMinWidth = ref(256);
        const filterActions = ref([]);
        const showAdvancedFilter = ref(false);

        // Selection
        const selectedItems = ref([]);

        // Preview and details
        const previewVisible = ref(false);
        const selectedItem = ref(null);
        const selectedIndex = ref(-1);
        const detailItem = ref(null);

        // Options
        const typeOptions = computed(() => [
            { value: 'all', label: $t('gallery.filterAll') },
            { value: 'image', label: $t('gallery.filterImage') },
            { value: 'video', label: $t('gallery.filterVideo') },
        ]);

        const timeOptions = computed(() => [
            { value: 'all', label: $t('gallery.dateAll') },
            { value: 'today', label: $t('gallery.dateToday') },
            { value: 'week', label: $t('gallery.dateWeek') },
            { value: 'month', label: $t('gallery.dateMonth') },
        ]);

        const allModelOptions = computed(() => {
            const set = new Set();
            items.value.forEach((it) => {
                if (it.model) set.add(it.model);
            });
            return Array.from(set).sort();
        });

        const flatItems = computed(() => {
            const result = [];
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
                   filterDateRange.value ||
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
        const groupedItems = computed(() => {
            const groups = [];
            const now = new Date();
            const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
            const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
            const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

            const todayItems = [];
            const yesterdayItems = [];
            const weekItems = [];
            const monthItems = [];
            const earlierItems = [];

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
            const params = {};
            if (filterType.value !== 'all') {
                params.kind = filterType.value;
            }
            if (filterModels.value.length > 0) {
                params.model = filterModels.value[0]; // Backend doesn't support multi-select yet, use first
            }
            if (filterTime.value !== 'all') {
                const now = new Date();
                let date;
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
                ElementPlus.ElMessage.error($tt('gallery.loadFailed'));
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
        const onScroll = (e) => {
            const el = e.target;
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
        const isSelected = (item) => {
            return selectedItems.value.some((it) => it.path === item.path);
        };

        const toggleSelect = (item) => {
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
        const handleCardClick = (item, event) => {
            if (event.ctrlKey || event.metaKey || event.shiftKey) {
                event.preventDefault();
                toggleSelect(item);
            }
        };

        // Batch delete
        const batchDelete = async () => {
            if (selectedItems.value.length === 0) return;
            try {
                await ElementPlus.ElMessageBox.confirm(
                    $tt('gallery.batchDeleteConfirm', { count: selectedItems.value.length }),
                    $tt('gallery.confirmDeleteTitle'),
                    { type: 'warning' }
                );
                
                const paths = selectedItems.value.map((it) => it.path);
                await api.gallery.batchDeleteImages(paths);
                ElementPlus.ElMessage.success($tt('gallery.batchDeleted', { count: selectedItems.value.length }));
                selectedItems.value = [];
                loadImages(true);
            } catch (e) {
                if (e !== 'cancel') {
                    console.error('Batch delete failed:', e);
                    ElementPlus.ElMessage.error($tt('gallery.batchDeleteFailed'));
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
        const getImageUrl = (item) => {
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

        const thumbUrl = (item) => {
            if (item.thumbnail) return item.thumbnail;
            return getImageUrl(item);
        };
        
        const isImage = (item) => {
            if (item.metadata?.asset_kind === 'video') return false;
            const ext = item.name?.split('.').pop()?.toLowerCase();
            return ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext);
        };

        const isVideo = (item) => {
            if (item.metadata?.asset_kind === 'video') return true;
            const ext = item.name?.split('.').pop()?.toLowerCase();
            return ['mp4', 'mov', 'avi', 'mkv'].includes(ext);
        };

        const showPreview = (item) => {
            const idx = flatItems.value.findIndex((it) => it.path === item.path);
            selectedIndex.value = idx >= 0 ? idx : -1;
            selectedItem.value = item;
            previewVisible.value = true;
        };

        const showDetail = (item) => {
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

        const deleteImage = async (item) => {
            try {
                await ElementPlus.ElMessageBox.confirm(
                    $tt('gallery.confirmDelete'),
                    $tt('gallery.confirmDeleteTitle'),
                    { type: 'warning' }
                );
                
                await api.gallery.deleteImage(item.path);
                ElementPlus.ElMessage.success($tt('gallery.deleted'));
                
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

        const useForImg2Img = (item) => {
            const SK = window.DQ_STORAGE || {};
            if (SK.IMG2IMG_REF) localStorage.setItem(SK.IMG2IMG_REF, item.path);
            ElementPlus.ElMessage.success($tt('gallery.img2imgSet'));
        };

        const downloadImage = (item) => {
            const url = getImageUrl(item);
            const link = document.createElement('a');
            link.href = url;
            link.download = item.name;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            ElementPlus.ElMessage.success($tt('gallery.startDownload'));
        };

        const handleCommand = (command, item) => {
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
        const handleVideoEnter = (e) => {
            const video = e.target;
            video.play().catch(() => {});
        };

        const handleVideoLeave = (e) => {
            const video = e.target;
            video.pause();
            video.currentTime = 0;
        };

        const handleImageError = (e) => {
            const img = e.target;
            img.style.display = 'none';
            const parent = img.parentElement;
            if (parent) {
                parent.style.display = 'flex';
                parent.style.alignItems = 'center';
                parent.style.justifyContent = 'center';
                const icon = document.createElement('span');
                icon.innerHTML = '❓';
                icon.style.fontSize = '24px';
                icon.style.opacity = '0.5';
                parent.appendChild(icon);
            }
        };

        // Formatting
        const formatDate = (dateStr) => {
            if (!dateStr) return 'N/A';
            const date = new Date(dateStr);
            return date.toLocaleString();
        };

        const formatRelativeTime = (dateStr) => {
            if (!dateStr) return '';
            const date = new Date(dateStr);
            const now = new Date();
            const diff = Math.floor((now - date) / 1000);
            
            if (diff < 60) return $t('gallery.justNow');
            if (diff < 3600) return $tt('gallery.minutesAgo', { m: Math.floor(diff / 60) });
            if (diff < 86400) return $tt('gallery.hoursAgo', { h: Math.floor(diff / 3600) });
            if (diff < 604800) return $tt('gallery.daysAgo', { d: Math.floor(diff / 86400) });
            return formatDate(dateStr);
        };

        const formatVideoDuration = (item) => {
            if (!item) return '';
            const raw = item.duration_seconds ?? item.metadata?.duration_seconds;
            if (raw == null || raw === '') return '';
            const n = Number(raw);
            if (!Number.isFinite(n) || n <= 0) return '';
            const sec = Math.round(n * 10) / 10;
            return typeof $tt === 'function' ? $tt('gallery.durationSecs', { sec }) : `${sec}s`;
        };

        const truncatePrompt = (prompt) => {
            if (!prompt) return '—';
            if (prompt.length <= 60) return prompt;
            return prompt.substring(0, 60) + '...';
        };

        const copyText = async (text) => {
            if (!text) return;
            try {
                await navigator.clipboard.writeText(text);
                ElementPlus.ElMessage.success($tt('gallery.copied'));
            } catch (e) {
                const ta = document.createElement('textarea');
                ta.value = text;
                document.body.appendChild(ta);
                ta.select();
                document.execCommand('copy');
                document.body.removeChild(ta);
                ElementPlus.ElMessage.success($tt('gallery.copied'));
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
        const onKeydown = (e) => {
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

        onMounted(() => {
            loadImages(true);
            window.addEventListener('keydown', onKeydown);
        });

        onUnmounted(() => {
            window.removeEventListener('keydown', onKeydown);
        });

        return {
            items,
            loading,
            hasMore,
            viewMode,
            filterType,
            filterModels,
            filterTime,
            filterDateRange,
            filterMinWidth,
            filterActions,
            showAdvancedFilter,
            typeOptions,
            timeOptions,
            allModelOptions,
            flatItems,
            groupedItems,
            selectedItems,
            allPageSelected,
            hasActiveFilters,
            emptyMessage,
            previewVisible,
            selectedItem,
            selectedIndex,
            detailItem,
            loadImages,
            refresh,
            onScroll,
            isSelected,
            toggleSelect,
            selectAllPage,
            handleCardClick,
            batchDelete,
            batchDownload,
            getImageUrl,
            thumbUrl,
            isImage,
            isVideo,
            showPreview,
            showDetail,
            showPrev,
            showNext,
            deleteImage,
            useForImg2Img,
            downloadImage,
            handleCommand,
            handleVideoEnter,
            handleVideoLeave,
            handleImageError,
            formatDate,
            formatRelativeTime,
            formatVideoDuration,
            truncatePrompt,
            copyText,
            resetFilters,
            resetAdvancedFilters,
            applyAdvancedFilters,
            Menu,
            Document,
        };
    }
};
