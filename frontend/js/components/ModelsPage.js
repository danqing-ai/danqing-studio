/**
 * Models page — Registry model download, import, and queue (main nav "Models")
 */

const ModelsPage = {
    template: `
        <div class="models-page" style="display: flex; gap: 20px; height: calc(100vh - 120px);">
            <!-- Left sidebar: category navigation -->
            <div class="download-sidebar" style="width: 220px; flex-shrink: 0;">
                <div class="card" style="height: 100%; display: flex; flex-direction: column;">
                    <div class="card-title" style="font-size: 16px;">
                        <el-icon><box /></el-icon>
                        {{ $t('download.downloadCenter') }}
                    </div>
                    <div style="font-size: 12px; color: var(--text-muted); margin: 6px 0 10px 0; line-height: 1.4;">
                        {{ $t('models.pageSubtitle') }}
                    </div>
                    
                    <el-menu
                        :default-active="activeCategory"
                        @select="handleCategorySelect"
                        style="border: none; flex: 1; background: transparent;"
                    >
                        <el-menu-item index="all">
                            <el-icon><grid /></el-icon>
                            <span>{{ $t('download.allModels') }}</span>
                            <el-tag size="small" type="info" style="margin-left: auto;">{{ totalModelCount }}</el-tag>
                        </el-menu-item>
                        
                        <el-menu-item index="image_models">
                            <el-icon><picture-filled /></el-icon>
                            <span>{{ $t('download.imageModels') }}</span>
                        </el-menu-item>

                        <el-menu-item index="video_models">
                            <el-icon><video-camera /></el-icon>
                            <span>{{ $t('download.videoModels') }}</span>
                        </el-menu-item>

                        <el-menu-item index="music_models">
                            <el-icon><headset /></el-icon>
                            <span>{{ $t('download.audioModels') }}</span>
                        </el-menu-item>
                        
                        <el-menu-item index="controlnets">
                            <el-icon><aim /></el-icon>
                            <span>{{ $t('download.controlNet') }}</span>
                        </el-menu-item>
                        
                        <el-menu-item index="upscalers">
                            <el-icon><zoom-in /></el-icon>
                            <span>{{ $t('download.upscalers') }}</span>
                        </el-menu-item>
                        
                        <el-menu-item index="tools">
                            <el-icon><tools /></el-icon>
                            <span>{{ $t('download.tools') }}</span>
                        </el-menu-item>
                        
                        <el-menu-item index="loras">
                            <el-icon><magic-stick /></el-icon>
                            <span>{{ $t('download.loraModels') }}</span>
                        </el-menu-item>
                        
                        <el-divider />
                        
                        <el-menu-item index="downloading">
                            <el-icon><download /></el-icon>
                            <span>{{ $t('download.downloadingTab') }}</span>
                            <el-tag v-if="activeDownloadCount > 0" size="small" type="primary" style="margin-left: auto;">{{ activeDownloadCount }}</el-tag>
                        </el-menu-item>
                        
                        <el-menu-item index="installed">
                            <el-icon><folder-checked /></el-icon>
                            <span>{{ $t('download.installed') }}</span>
                        </el-menu-item>
                    </el-menu>
                    
                    <!-- Disk space -->
                    <div v-if="diskSpace" class="disk-space-panel">
                        <div class="disk-space-title">
                            <el-icon><monitor /></el-icon>
                            {{ $t('download.diskSpace') }}
                        </div>
                        <div class="disk-space-item">
                            <div class="disk-space-label">
                                <span>{{ $t('download.modelLabel') }}</span>
                                <span class="disk-space-value">{{ diskSpace.models?.size_human }}</span>
                            </div>
                            <el-progress :percentage="getDiskPercent('models')" :show-text="false" :stroke-width="4" />
                        </div>
                        <div class="disk-space-item">
                            <div class="disk-space-label">
                                <span>{{ $t('download.loraLabel') }}</span>
                                <span class="disk-space-value">{{ diskSpace.loras?.size_human }}</span>
                            </div>
                            <el-progress :percentage="getDiskPercent('loras')" :show-text="false" :stroke-width="4" color="#67c23a" />
                        </div>
                        <div class="disk-space-footer">
                            {{ $t('download.free') }}: {{ diskSpace.models?.free_human }}
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Right content area -->
            <div style="flex: 1; overflow-y: auto;">
                <!-- Model grid (category browsing) -->
                <div v-if="['all', 'image_models', 'video_models', 'music_models', 'controlnets', 'upscalers', 'tools', 'loras'].includes(activeCategory)">
                    <!-- Quick-start workflow -->
                    <div v-if="activeCategory === 'all' || activeCategory === 'image_models'" 
                         style="background: var(--bg-card); 
                                border: 1px solid var(--border-color);
                                padding: 16px 20px; border-radius: 12px; margin-bottom: 20px; 
                                display: flex; justify-content: space-between; align-items: center;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <div style="width: 40px; height: 40px; border-radius: 10px; background: rgba(233, 69, 96, 0.1); border: 1px solid rgba(233, 69, 96, 0.2); display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: var(--primary); font-size: 18px; font-weight: 700;">
                                <el-icon><promotion /></el-icon>
                            </div>
                            <div>
                                <div style="font-weight: 600; font-size: 14px; margin-bottom: 2px; color: var(--text-primary);">{{ $t('download.quickStart') }}</div>
                                <div style="font-size: 12px; color: var(--text-muted);">{{ $t('download.quickStartDesc') }}</div>
                            </div>
                        </div>
                        <el-button type="primary" 
                                   @click="downloadRecommendedSet"
                                   :loading="downloadingRecommended"
                                   :disabled="downloadingRecommended">
                            <el-icon><download /></el-icon>
                            {{ $t('download.oneClickInstall') }}
                        </el-button>
                    </div>

                    <!-- Category title -->
                    <div class="page-header">
                        <h2 class="page-title">{{ categoryTitle }}</h2>
                        <div class="page-actions">
                            <el-button @click="showImportDialog" v-if="activeCategory !== 'loras' && activeCategory !== 'installed'" size="small">
                                <el-icon><upload /></el-icon>
                                {{ $t('download.importLocal') }}
                            </el-button>
                            <el-input 
                                v-model="filterQuery" 
                                :placeholder="$t('download.searchModel')" 
                                style="width: 220px;"
                                size="small"
                                clearable
                            >
                                <template #prefix>
                                    <el-icon><search /></el-icon>
                                </template>
                            </el-input>
                            <el-button @click="refreshStatus" :loading="refreshing" size="small" circle>
                                <el-icon><refresh /></el-icon>
                            </el-button>
                        </div>
                    </div>

                    <!-- Import local model dialog -->
                    <el-dialog v-model="importDialogVisible" :title="$t('download.importTitle')" width="500px">
                        <el-form label-position="top">
                            <el-form-item :label="$t('download.modelName')">
                                <el-input v-model="importModelName" :placeholder="$t('download.modelNamePlaceholder')" />
                            </el-form-item>
                            <el-form-item :label="$t('download.modelPath')">
                                <el-input v-model="importModelPath" :placeholder="$t('download.modelPathPlaceholder')" />
                            </el-form-item>
                            <el-form-item :label="$t('download.modelType')">
                                <el-select v-model="importModelType" style="width: 100%">
                                    <el-option :label="$t('download.baseModel')" value="base" />
                                    <el-option :label="$t('download.loraType')" value="lora" />
                                    <el-option :label="$t('download.controlnetType')" value="controlnet" />
                                </el-select>
                            </el-form-item>
                        </el-form>
                        <template #footer>
                            <el-button @click="importDialogVisible = false">{{ $t('download.cancel') }}</el-button>
                            <el-button type="primary" @click="importLocalModel" :loading="importing">
                                {{ $t('download.import_') }}
                            </el-button>
                        </template>
                    </el-dialog>
                    
                    <!-- Model card grid -->
                    <el-row :gutter="16" class="model-grid">
                        <el-col 
                            :xs="24" :sm="12" :md="8" :lg="6" 
                            v-for="model in filteredModels" 
                            :key="model.id"
                            style="margin-bottom: 16px;"
                        >
                            <el-card 
                                :body-style="{ padding: '0' }" 
                                class="model-card"
                                :class="{ 'model-ready': model.ready }"
                            >
                                <!-- Card header: icon/preview + status -->
                                <div class="model-card-header">
                                    <div class="model-icon">
                                        {{ getModelInitials(model) }}
                                    </div>
                                    <div class="model-status">
                                        <el-tag v-if="modelsDetailedStatus[model.id]?.status === 'ready'" size="small" type="success">{{ $t('download.readyTag') }}</el-tag>
                                        <el-tag v-else-if="modelsDetailedStatus[model.id]?.status === 'incomplete'" size="small" type="danger">{{ $t('download.incompleteTag') }}</el-tag>
                                        <el-tag v-else size="small" type="warning">{{ $t('download.notDownloadedTag') }}</el-tag>
                                    </div>
                                    <el-tag v-if="model.recommended" size="small" class="recommended-badge" type="success">{{ $t('download.recommendedBadge') }}</el-tag>
                                </div>
                                
                                <!-- Card content -->
                                <div class="model-card-content">
                                    <div class="model-card-name">
                                        {{ $mn(model) }}
                                    </div>
                                    <el-tooltip :content="$md(model)" placement="top" effect="dark">
                                        <div class="model-card-desc">
                                            {{ $md(model) }}
                                        </div>
                                    </el-tooltip>
                                    
                                    <!-- Meta info -->
                                    <div class="model-card-meta">
                                        <el-tag v-if="model.size" size="small" type="info" effect="plain">{{ model.size }}</el-tag>
                                        <el-tag v-if="model.source === 'huggingface'" size="small" type="primary" effect="plain">HF</el-tag>
                                        <el-tag v-else-if="model.source === 'modelscope'" size="small" type="danger" effect="plain">ModelScope</el-tag>
                                        <el-tag v-else-if="model.source === 'civitai'" size="small" type="warning" effect="plain">CivitAI</el-tag>
                                        <el-tag v-if="model.base_model" size="small" type="success" effect="plain">{{ model.base_model }}</el-tag>
                                    </div>
                                    
                                <!-- Version list (registry entries must have versions per check_consistency) -->
                                <div v-if="model.versions" style="margin-bottom: 12px;">
                                    <div v-for="(ver, verKey) in model.versions" :key="verKey" 
                                         style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-color);">
                                        <div style="flex: 1; min-width: 0;">
                                            <div style="display: flex; align-items: center; gap: 6px;">
                                                <span style="font-size: 13px; color: var(--text-primary);">{{ ver.name }}</span>
                                                <el-tag size="small" type="info" effect="plain">{{ ver.size }}</el-tag>
                                                <el-tag v-if="ver.source_type === 'derived'" size="small" type="warning" effect="plain">{{ $t('download.derivedTag') }}</el-tag>
                                                <el-tag v-else-if="ver.source_type === 'prequantized'" size="small" type="primary" effect="plain">{{ $t('download.prequantized') }}</el-tag>
                                            </div>
                                            <div v-if="ver.source_type === 'derived'" style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                                                {{ $t('download.basedOn', { name: model.versions[ver.from_version]?.name || ver.from_version }) }}
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 6px; flex-shrink: 0; flex-wrap: wrap; justify-content: flex-end;">
                                            <!-- Show different buttons based on version status -->
                                            <template v-if="getVersionStatus(model.id, verKey) === 'ready'">
                                                <el-button type="warning" size="small" @click="downloadVersion(model, verKey)">
                                                    <el-icon><download /></el-icon> {{ $t('download.forceDownload') }}
                                                </el-button>
                                                <el-button type="danger" size="small" @click="deleteVersion(model, verKey)">
                                                    <el-icon><delete /></el-icon>
                                                </el-button>
                                            </template>
                                            <template v-else-if="getVersionStatus(model.id, verKey) === 'parent_missing'">
                                                <el-tooltip 
                                                    v-if="!canDownload(model)"
                                                    :content="getDependencyHint(model)"
                                                    placement="top"
                                                >
                                                    <span>
                                                        <el-button 
                                                            type="primary" 
                                                            size="small"
                                                            :disabled="true"
                                                        >
                                                            <el-icon><download /></el-icon>
                                                            {{ $t('download.downloadVersion') }}
                                                        </el-button>
                                                    </span>
                                                </el-tooltip>
                                                <el-button 
                                                    v-else
                                                    type="primary" 
                                                    size="small"
                                                    @click="downloadVersion(model, ver.from_version, { uiLoadingKey: model.id + '-' + verKey })"
                                                    :loading="downloadingModels[model.id + '-' + verKey]"
                                                >
                                                    <el-icon><download /></el-icon>
                                                    {{ $t('download.downloadVersion') }}
                                                </el-button>
                                            </template>
                                            <template v-else-if="getVersionStatus(model.id, verKey) === 'quantize'">
                                                <el-tooltip 
                                                    v-if="!canDownload(model)"
                                                    :content="getDependencyHint(model)"
                                                    placement="top"
                                                >
                                                    <span>
                                                        <el-button 
                                                            type="primary" 
                                                            size="small"
                                                            :disabled="true"
                                                        >
                                                            {{ $t('download.quantizeVersion') }}
                                                        </el-button>
                                                    </span>
                                                </el-tooltip>
                                                <el-button 
                                                    v-else
                                                    type="primary" 
                                                    size="small"
                                                    @click="quantizeVersion(model, verKey)"
                                                    :loading="downloadingModels[model.id + '-' + verKey]"
                                                >
                                                    {{ $t('download.quantizeVersion') }}
                                                </el-button>
                                            </template>
                                            <template v-else>
                                                <el-tooltip 
                                                    v-if="!canDownload(model)"
                                                    :content="getDependencyHint(model)"
                                                    placement="top"
                                                >
                                                    <span>
                                                        <el-button 
                                                            type="primary" 
                                                            size="small"
                                                            :disabled="true"
                                                        >
                                                            <el-icon><download /></el-icon>
                                                            {{ $t('download.downloadVersion') }}
                                                        </el-button>
                                                    </span>
                                                </el-tooltip>
                                                <el-button 
                                                    v-else
                                                    type="primary" 
                                                    size="small"
                                                    @click="downloadVersion(model, verKey)"
                                                    :loading="downloadingModels[model.id + '-' + verKey]"
                                                >
                                                    <el-icon><download /></el-icon>
                                                    {{ $t('download.downloadVersion') }}
                                                </el-button>
                                            </template>
                                        </div>
                                    </div>
                                </div>
                                </div>
                            </el-card>
                        </el-col>
                    </el-row>
                    
                    <el-empty v-if="filteredModels.length === 0 && activeCategory !== 'loras'" :description="$t('download.noModelsInCategory')" />
                    
                    <!-- LoRA search -->
                    <div v-if="activeCategory === 'loras'" style="margin-top: 32px; border-top: 1px solid var(--border-color); padding-top: 24px;">
                        <div class="page-header">
                            <h2 class="page-title">{{ $t('download.civitaiSearch') }}</h2>
                        </div>
                    
                    <div class="card" style="margin-bottom: 16px;">
                        <div style="display: flex; gap: 12px;">
                            <el-input 
                                v-model="searchQuery" 
                                :placeholder="$t('download.searchCivitai')" 
                                style="flex: 1;"
                                @keyup.enter="searchCivitai"
                                clearable
                            >
                                <template #prefix>
                                    <el-icon><search /></el-icon>
                                </template>
                            </el-input>
                            <el-select v-model="searchType" style="width: 150px;">
                                <el-option label="LoRA" value="LORA" />
                                <el-option label="Checkpoint" value="Checkpoint" />
                                <el-option :label="$t('download.all')" value="LORA,Checkpoint" />
                            </el-select>
                            <el-button type="primary" @click="searchCivitai" :loading="searching">
                                <el-icon><search /></el-icon>
                                {{ $t('download.search') }}
                            </el-button>
                        </div>
                    </div>
                    
                    <el-row :gutter="16" v-if="searchResults.length > 0">
                        <el-col :xs="24" :sm="12" :md="8" v-for="model in searchResults" :key="model.id" style="margin-bottom: 16px;">
                            <el-card :body-style="{ padding: '12px' }" class="civitai-card">
                                <div style="display: flex; gap: 12px;">
                                    <div class="civitai-preview">
                                        <img v-if="model.model_versions[0]?.images[0]?.url" 
                                             :src="model.model_versions[0].images[0].url" 
                                             @error="$event.target.style.display='none'" />
                                        <div v-else class="no-preview">
                                            <el-icon><picture-filled /></el-icon>
                                        </div>
                                    </div>
                                    
                                    <div style="flex: 1; min-width: 0;">
                                        <div class="civitai-name">{{ model.name }}</div>
                                        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">
                                            {{ model.type }} | {{ model.model_versions[0]?.base_model || 'Unknown' }}
                                        </div>
                                        <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">
                                            {{ model.creator?.username || $tt('download.unknownCreator') }}
                                        </div>
                                        <div style="display: flex; gap: 8px; align-items: center;">
                                            <el-tag v-if="model.nsfw" size="small" type="danger">{{ $t('download.nsfwTag') }}</el-tag>
                                            <el-tag size="small" type="info">
                                                <el-icon><download /></el-icon>
                                                {{ formatNumber(model.stats?.downloadCount || 0) }}
                                            </el-tag>
                                        </div>
                                    </div>
                                </div>
                                
                                <div style="margin-top: 12px; display: flex; gap: 8px;">
                                    <el-select v-model="selectedVersions[model.id]" size="small" style="flex: 1;" :placeholder="$t('download.selectVersion')">
                                        <el-option 
                                            v-for="v in model.model_versions" 
                                            :key="v.id" 
                                            :label="v.name" 
                                            :value="v.id"
                                        />
                                    </el-select>
                                    <el-button type="primary" size="small" @click="downloadCivitaiModel(model)" :loading="downloadingLoras[model.id]">
                                        {{ $t('download.download_') }}
                                    </el-button>
                                </div>
                            </el-card>
                        </el-col>
                    </el-row>
                    
                    <el-empty v-else-if="!searching && hasSearched" :description="$t('download.noResults')" />
                    </div>
                </div>
                
                <!-- Installed -->
                <div v-if="activeCategory === 'installed'">
                    <div class="page-header">
                        <h2 class="page-title">{{ $t('download.installedLabel') }}</h2>
                    </div>
                    
                    <el-table :data="installedModels" style="width: 100%">
                        <el-table-column prop="name" :label="$t('download.nameCol')" />
                        <el-table-column prop="type" :label="$t('download.typeCol')" width="120">
                            <template #default="scope">
                                <el-tag size="small" :type="getModelTypeTagType(scope.row.type)">
                                    {{ scope.row.type || 'unknown' }}
                                </el-tag>
                            </template>
                        </el-table-column>
                        <el-table-column prop="size_human" :label="$t('download.sizeCol')" width="120" />
                        <el-table-column prop="path" :label="$t('download.pathCol')" />
                    </el-table>
                </div>
                
                <!-- Downloading -->
                <div v-if="activeCategory === 'downloading'">
                    <div class="page-header">
                        <h2 class="page-title">{{ $t('download.downloadingTab') }} ({{ activeDownloadCount }})</h2>
                    </div>
                    
                    <div v-if="activeDownloadCount === 0" class="card">
                        <el-empty :description="$t('download.noTasks')" />
                    </div>
                    
                    <div v-else class="card">
                        <div v-for="(item, taskId) in activeDownloads" :key="taskId" style="margin-bottom: 16px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; align-items: center;">
                                <span style="font-weight: 500;">{{ item.name }}</span>
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <span style="color: var(--text-muted); font-size: 13px;">
                                        <span v-if="item.total_size > 0">
                                            {{ Math.round(item.progress * 100) }}%
                                            <span v-if="item.speed">({{ item.speed }})</span>
                                        </span>
                                        <span v-else-if="item.downloaded_size > 0">
                                            {{ formatBytes(item.downloaded_size) }}
                                            <span v-if="item.speed">({{ item.speed }})</span>
                                        </span>
                                        <span v-else>{{ $t('download.preparing') }}</span>
                                    </span>
                                    <el-button v-if="item.status === 'paused'" type="primary" size="small" @click="resumeDownload(taskId)">
                                        {{ $t('download.resume') }}
                                    </el-button>
                                    <el-button v-else-if="item.status === 'running'" size="small" @click="cancelDownload(taskId)">
                                        {{ $t('download.cancelDownload') }}
                                    </el-button>
                                    <el-button v-else-if="item.status === 'failed'" type="danger" size="small" @click="deleteDownload(taskId)">
                                        <el-icon><delete /></el-icon>
                                        {{ $t('download.deleteTask') }}
                                    </el-button>
                                </div>
                            </div>
                            <el-progress
                                :percentage="item.total_size > 0 ? Math.round(item.progress * 100) : 0"
                                :status="item.status === 'failed' ? 'exception' : ''"
                                :stroke-width="8"
                                :show-text="item.total_size > 0"
                            />
                            <div v-if="item.error" style="color: var(--error); font-size: 12px; margin-top: 4px;">
                                {{ item.error }}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `,
    
    setup() {
        const { ref, reactive, onMounted, computed, onUnmounted } = Vue;
        
        // Category nav
        const activeCategory = ref('all');
        const categoryTitle = computed(() => {
            const titles = {
                'all': $tt('download.allModels'),
                'image_models': '🖼️ ' + $tt('download.imageModels'),
                'video_models': '🎬 ' + $tt('download.videoModels'),
                'music_models': '🎵 ' + $tt('download.audioModels'),
                'controlnets': '🎯 ' + $tt('download.controlNet'),
                'upscalers': '🔍 ' + $tt('download.upscalers'),
                'tools': '⚙️ ' + $tt('download.tools'),
                'loras': '🎭 ' + $tt('download.loraModels')
            };
            return titles[activeCategory.value] || $tt('download.title');
        });
        const categoryIcons = {
            'image_models': '🖼️',
            'video_models': '🎬',
            'music_models': '🎵',
            'controlnets': '🎯',
            'upscalers': '🔍',
            'tools': '⚙️',
            'loras': '🎭'
        };
        
        // Model data
        const modelRegistry = ref({});
        const modelsStatus = ref({});
        const modelsDetailedStatus = ref({});
        const categories = ref({});
        const filterQuery = ref('');
        const refreshing = ref(false);
        
        // Download status
        const downloadingModels = ref({});
        const downloadingLoras = ref({});
        const downloadingRecommended = ref(false);
        const activeDownloads = ref({});
        const selectedVersions = ref({});
        const sseConnections = ref({});
        
        // CivitAI search
        const searchQuery = ref('');
        const searchType = ref('LORA');
        const searching = ref(false);
        const searchResults = ref([]);
        const hasSearched = ref(false);
        
        // Installed
        const installedModels = ref([]);
        
        // Disk space
        const diskSpace = ref(null);
        
        // Import local model
        const importDialogVisible = ref(false);
        const importModelName = ref('');
        const importModelPath = ref('');
        const importModelType = ref('base');
        const importing = ref(false);

        /** Registry v2: `name` / `description` may be { zh, en } (GET /api/registry as-is) */
        const modelLabel = (m) => (typeof window.$mn === 'function' ? window.$mn(m, m.id) : (m.id || ''));
        const modelSearchBlob = (m) => {
            const n = modelLabel(m);
            const d = typeof window.$md === 'function' ? window.$md(m, '') : '';
            return `${n} ${d}`.toLowerCase();
        };
        
        // Filtered model list
        const filteredModels = computed(() => {
            let list = [];
            for (const [id, config] of Object.entries(modelRegistry.value)) {
                const model = {
                    id,
                    ...config,
                    ready: modelsStatus.value[id] || false
                };
                
                // Category filter (sidebar `image_models` ↔ registry `base_models`)
                if (activeCategory.value !== 'all') {
                    const regCat = model.category;
                    if (activeCategory.value === 'image_models') {
                        if (regCat !== 'base_models') continue;
                    } else if (activeCategory.value !== regCat) {
                        continue;
                    }
                }
                
                // Search filter
                if (filterQuery.value) {
                    const query = filterQuery.value.toLowerCase();
                    if (!modelSearchBlob(model).includes(query)) {
                        continue;
                    }
                }
                
                list.push(model);
            }
            
            // Sort: recommended first, then by name
            return list.sort((a, b) => {
                if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
                return modelLabel(a).localeCompare(modelLabel(b));
            });
        });
        
        const totalModelCount = computed(() => Object.keys(modelRegistry.value).length);
        
        // Active download task count (includes paused tasks)
        const activeDownloadCount = computed(() => {
            return Object.values(activeDownloads.value).filter(item => item.status === 'running' || item.status === 'paused' || item.status === 'failed').length;
        });
        
        // Load model registry
        const loadModelRegistry = async () => {
            try {
                const RS = window.RegistryStore;
                const regPromise =
                    RS && RS.load
                        ? RS.load()
                        : api.settings.getModelRegistry().then((r) => ({
                              models: r.models,
                              categories: r.categories,
                          }));
                const [registryData, statusData, detailedStatusData] = await Promise.all([
                    regPromise,
                    api.settings.getModelsStatus(),
                    api.settings.getModelsDetailedStatus(),
                ]);

                modelRegistry.value = registryData.models || {};
                modelsStatus.value = statusData || {};
                modelsDetailedStatus.value = detailedStatusData || {};
                categories.value = registryData.categories || {};
            } catch (e) {
                console.error('Failed to load model registry:', e);
            }
        };
        
        // Refresh status
        const refreshStatus = async () => {
            refreshing.value = true;
            try {
                const [statusData, detailedStatusData] = await Promise.all([
                    api.settings.getModelsStatus(),
                    api.settings.getModelsDetailedStatus()
                ]);
                modelsStatus.value = statusData || {};
                modelsDetailedStatus.value = detailedStatusData || {};
                await loadInstalled();
                await loadDiskSpace();
            } catch (e) {
                console.error('Refresh failed:', e);
            } finally {
                refreshing.value = false;
            }
        };
        
        // Load installed
        const loadInstalled = async () => {
            try {
                const models = await api.settings.listModels();
                installedModels.value = models || [];
            } catch (e) {
                console.error('Failed to load installed:', e);
            }
        };
        
        // Load disk space
        const loadDiskSpace = async () => {
            try {
                const data = await api.settings.getDiskSpace();
                diskSpace.value = data;
            } catch (e) {
                console.error('Failed to load disk space:', e);
            }
        };
        
        // Load active downloads (restore after page refresh)
        const loadActiveDownloads = async () => {
            try {
                const tasks = await api.download.listDownloads();
                if (!Array.isArray(tasks)) return;
                for (const task of tasks) {
                    if (task.status === 'running' || task.status === 'pending' || task.status === 'paused' || task.status === 'failed') {
                        activeDownloads.value[task.id] = {
                            name: task.filename || task.url,
                            progress: task.progress || 0,
                            status: task.status,
                            speed: '',
                            error: task.error_message || '',
                            total_size: task.total_size || 0,
                            downloaded_size: task.downloaded_size || 0
                        };
                        // Only connect SSE for truly running tasks
                        if (task.status === 'running') {
                            connectProgressSSE(task.id, task.filename || task.url);
                        }
                    }
                }
            } catch (e) {
                console.error('Failed to load active downloads:', e);
            }
        };
        
        // Disk usage percentage
        const getDiskPercent = (type) => {
            if (!diskSpace.value || !diskSpace.value[type]) return 0;
            const info = diskSpace.value[type];
            if (!info.exists || info.total === 0) return 0;
            return Math.round(info.size / info.total * 100);
        };
        
        // Format file size
        const formatBytes = (bytes) => {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        };
        
        // Check if model is ready
        const isModelReady = (modelId) => {
            return modelsStatus.value[modelId] || false;
        };
        
        // Get model name (registry entry or id)
        const getModelName = (modelId) => {
            const cfg = modelRegistry.value[modelId];
            if (!cfg) return modelId;
            const row = { id: modelId, ...cfg };
            return modelLabel(row);
        };
        
        // Check if download is allowed (dependencies met)
        const canDownload = (model) => {
            if (!model.dependencies || model.dependencies.length === 0) return true;
            return model.dependencies.every(dep => isModelReady(dep));
        };

        // Get hint when dependencies are not met
        const getDependencyHint = (model) => {
            if (!model.dependencies || model.dependencies.length === 0) return '';
            const missing = model.dependencies.filter(dep => !isModelReady(dep));
            if (missing.length === 0) return '';
            const names = missing.map(d => getModelName(d)).join('、');
            return $tt('download.dependencyMissing', { models: names });
        };
        
        // Get version status
        const getVersionStatus = (modelId, versionKey) => {
            const detail = modelsDetailedStatus.value[modelId];
            if (!detail || !detail.versions) return 'missing';

            const verStatus = detail.versions[versionKey];
            if (!verStatus) return 'missing';

            if (verStatus.ready) return 'ready';

            const model = modelRegistry.value[modelId];
            const ver = model?.versions?.[versionKey];
            if (ver && ver.source_type === 'derived' && ver.from_version) {
                const parentVer = ver.from_version;
                const parentSt = parentVer ? detail.versions[parentVer] : null;
                if (!parentSt || !parentSt.ready) {
                    return 'parent_missing';
                }
                return 'quantize';
            }

            return 'missing';
        };

        // Download a specific version (optionally show loading on another row, e.g. derived row while parent installs)
        const downloadVersion = async (model, versionKey, opts = {}) => {
            const uiKey = opts.uiLoadingKey != null ? opts.uiLoadingKey : `${model.id}-${versionKey}`;
            if (downloadingModels.value[uiKey]) return;

            downloadingModels.value[uiKey] = true;

            try {
                const version = model.versions[versionKey];
                const data = await api.models.install(model.id, { version: versionKey });
                const label = version ? `${modelLabel(model)} ${version.name}` : `${modelLabel(model)} ${versionKey}`;
                connectProgressSSE(data.task_id, label, uiKey);
            } catch (e) {
                console.error('Download failed:', e);
                ElementPlus.ElMessage.error($tt('download.downloadFailed', { msg: e.message }));
                delete downloadingModels.value[uiKey];
            }
        };

        /** Local MLX quantization (derived version); parent must already be installed. */
        const quantizeVersion = async (model, versionKey) => {
            const ver = model.versions[versionKey];
            const fromV = ver?.from_version;
            if (!fromV) {
                ElementPlus.ElMessage.error($tt('download.versionConfigError'));
                return;
            }
            const uiKey = `${model.id}-${versionKey}`;
            if (downloadingModels.value[uiKey]) return;

            downloadingModels.value[uiKey] = true;
            try {
                const data = await api.download.startConvert({
                    model_name: model.id,
                    from_version: fromV,
                    to_version: versionKey,
                });
                const taskId = data.task_id;
                const label = ver?.name ? `${modelLabel(model)} ${ver.name}` : `${modelLabel(model)} ${versionKey}`;
                connectConversionSSE(taskId, label, uiKey);
            } catch (e) {
                console.error('Quantize failed:', e);
                ElementPlus.ElMessage.error($tt('download.quantizeFailed', { msg: e.message || String(e) }));
                delete downloadingModels.value[uiKey];
            }
        };

        // Delete a specific version
        const deleteVersion = async (model, versionKey) => {
            try {
                const version = model.versions[versionKey];
                await ElementPlus.ElMessageBox.confirm(
                    $tt('download.deleteConfirm', { name: `${modelLabel(model)} ${version?.name || versionKey}` }),
                    $tt('download.deleteConfirmTitle'),
                    { confirmButtonText: $tt('download.deleteConfirmBtn'), cancelButtonText: $tt('download.deleteCancelBtn'), type: 'warning' }
                );
                const result = await api.models.deleteVersion(model.id, versionKey);
                if (result.success) {
                    ElementPlus.ElMessage.success($tt('download.deletedMsg', { name: `${modelLabel(model)} ${version?.name || versionKey}` }));
                } else {
                    ElementPlus.ElMessage.error(result.error || $tt('download.deleteFailed'));
                }
                refreshStatus();
            } catch (e) {
                if (e !== 'cancel') {
                    console.error('Delete failed:', e);
                    ElementPlus.ElMessage.error($tt('download.deleteFailed') + ': ' + (e.message || e));
                }
            }
        };
        
        // SSE progress connection; `downloadingKey` aligns with `downloadingModels` key (multi-version install: `${modelId}-${versionKey}`)
        const connectProgressSSE = (taskId, name, downloadingKey = null) => {
            const dmKey = downloadingKey != null ? downloadingKey : taskId;
            if (sseConnections.value[taskId]) {
                sseConnections.value[taskId].close();
            }
            
            const eventSource = new EventSource(api.download.installProgressStreamUrl(taskId));
            sseConnections.value[taskId] = eventSource;
            
            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    activeDownloads.value[taskId] = {
                        name: name,
                        progress: data.progress || 0,
                        status: data.status,
                        speed: data.speed || '',
                        error: data.error_message || '',
                        total_size: data.total_size,
                        downloaded_size: data.downloaded_size
                    };
                    
                    if (data.status === 'completed') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        setTimeout(() => {
                            delete activeDownloads.value[taskId];
                            delete downloadingModels.value[dmKey];
                        }, 2000);
                        ElementPlus.ElMessage.success($tt('download.downloadComplete', { name }));
                        refreshStatus();
                    } else if (data.status === 'failed') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        delete downloadingModels.value[dmKey];
                        ElementPlus.ElMessage.error($tt('download.downloadFailed', { name, msg: data.error_message }));
                    } else if (data.status === 'cancelled') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        delete activeDownloads.value[taskId];
                        delete downloadingModels.value[dmKey];
                        ElementPlus.ElMessage.info($tt('download.downloadCancelled', { name }));
                    }
                } catch (e) {
                    console.error('SSE parse error:', e);
                }
            };
            
            eventSource.onerror = (error) => {
                console.error('SSE error:', error);
                eventSource.close();
                delete sseConnections.value[taskId];
                delete downloadingModels.value[dmKey];
            };
        };

        /** Quantization job progress (same drawer as downloads; ``total_size: 1`` drives percent bar). */
        const connectConversionSSE = (taskId, name, dmKey = null) => {
            const rowKey = dmKey != null ? dmKey : taskId;
            if (sseConnections.value[taskId]) {
                try { sseConnections.value[taskId].close(); } catch (e) {}
            }

            const eventSource = new EventSource(api.download.convertProgressStreamUrl(taskId));
            sseConnections.value[taskId] = eventSource;

            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    activeDownloads.value[taskId] = {
                        kind: 'quantize',
                        name,
                        progress: typeof data.progress === 'number' ? data.progress : 0,
                        status: data.status || 'running',
                        speed: '',
                        total_size: 1,
                        downloaded_size: 0,
                        error: data.error_message || '',
                    };

                    if (data.status === 'completed') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        setTimeout(() => {
                            delete activeDownloads.value[taskId];
                            delete downloadingModels.value[rowKey];
                        }, 2000);
                        ElementPlus.ElMessage.success($tt('download.quantizeComplete', { name }));
                        refreshStatus();
                    } else if (data.status === 'failed') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        delete downloadingModels.value[rowKey];
                        ElementPlus.ElMessage.error(
                            $tt('download.quantizeFailed', { msg: data.error_message || '' }),
                        );
                    } else if (data.status === 'cancelled') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        delete activeDownloads.value[taskId];
                        delete downloadingModels.value[rowKey];
                        ElementPlus.ElMessage.info($tt('download.genCancelled', { name }));
                    }
                } catch (e) {
                    console.error('Conversion SSE parse error:', e);
                }
            };

            eventSource.onerror = () => {
                eventSource.close();
                delete sseConnections.value[taskId];
                delete downloadingModels.value[rowKey];
            };
        };

        // Batch download recommended models
        const downloadRecommendedSet = async () => {
            if (downloadingRecommended.value) return;
            downloadingRecommended.value = true;
            
            try {
                const modelsToDownload = ['flux1-schnell']; // can be expanded to more
                
                ElementPlus.ElMessage.info($tt('download.batchDownloadStart'));
                
                const data = await api.models.installBatch(modelsToDownload);
                
                for (const result of data.results) {
                    if (result.status === 'started') {
                        const modelName = getModelName(result.model_name);
                        connectProgressSSE(result.task_id, modelName);
                    } else if (result.status === 'skipped') {
                        ElementPlus.ElMessage.warning(`${result.model_name}: ${result.reason}`);
                    }
                }
                
            } catch (e) {
                console.error('Batch download failed:', e);
                ElementPlus.ElMessage.error($tt('download.batchDownloadFailed', { msg: e.message }));
            } finally {
                downloadingRecommended.value = false;
            }
        };
        
        // Cancel download
        const cancelDownload = async (taskId) => {
            const item = activeDownloads.value[taskId];
            if (item && item.kind === 'quantize') {
                try {
                    await api.download.cancelConversion(taskId);
                } catch (e) {
                    console.error('Cancel conversion failed:', e);
                }
                return;
            }
            try {
                await api.download.cancel(taskId);
            } catch (e) {
                console.error('Cancel failed:', e);
            }
        };
        
        // Resume download
        const resumeDownload = async (taskId) => {
            try {
                const item = activeDownloads.value[taskId];
                if (!item) return;
                
                await api.download.resume(taskId);
                
                // Update status to running
                item.status = 'running';
                
                // Connect SSE to monitor progress
                connectProgressSSE(taskId, item.name);
                
                ElementPlus.ElMessage.success($tt('download.resumeStart', { name: item.name }));
            } catch (e) {
                console.error('Resume failed:', e);
                ElementPlus.ElMessage.error($tt('download.resumeFailed', { msg: e.message }));
            }
        };

        const deleteDownload = async (taskId) => {
            try {
                const item = activeDownloads.value[taskId];
                if (!item || item.kind !== 'quantize') {
                    await api.download.delete(taskId);
                }
                delete activeDownloads.value[taskId];
                ElementPlus.ElMessage.success($tt('download.deleteSuccess'));
            } catch (e) {
                console.error('Delete download failed:', e);
                ElementPlus.ElMessage.error($tt('download.deleteFailed'));
            }
        };
        
        // CivitAI search
        const searchCivitai = async () => {
            if (searching.value) return;
            searching.value = true;
            hasSearched.value = true;
            
            try {
                const data = await api.download.civitaiSearch({
                    q: searchQuery.value,
                    types: searchType.value,
                    limit: '20',
                });
                const models = Array.isArray(data) ? data : (data.items || []);
                searchResults.value = models;
                
                models.forEach(model => {
                    if (model.model_versions.length > 0 && !selectedVersions.value[model.id]) {
                        selectedVersions.value[model.id] = model.model_versions[0].id;
                    }
                });
            } catch (e) {
                console.error('Search failed:', e);
                ElementPlus.ElMessage.error($tt('download.searchFailed'));
            } finally {
                searching.value = false;
            }
        };
        
        // Download CivitAI model
        const downloadCivitaiModel = async (model) => {
            const versionId = selectedVersions.value[model.id];
            if (!versionId) {
                ElementPlus.ElMessage.warning($tt('download.selectVersionWarn'));
                return;
            }
            
            const version = model.model_versions.find(v => v.id === versionId);
            if (!version || !version.files.length) {
                ElementPlus.ElMessage.error($tt('download.noDownloadableFile'));
                return;
            }
            
            const primaryFile = version.files.find(f => f.primary) || version.files[0];
            downloadingLoras.value[model.id] = true;
            
            try {
                const data = await api.download.startLoraDownload(
                    primaryFile.download_url,
                    primaryFile.name
                );
                connectProgressSSE(data.task_id, model.name);
            } catch (e) {
                console.error('Download failed:', e);
                ElementPlus.ElMessage.error($tt('download.downloadFailed', { msg: e.message }));
            } finally {
                downloadingLoras.value[model.id] = false;
            }
        };
        
        // Format number
        const formatNumber = (num) => {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toString();
        };
        
        // Import local model
        const showImportDialog = () => {
            importModelName.value = '';
            importModelPath.value = '';
            importModelType.value = 'base';
            importDialogVisible.value = true;
        };
        
        const importLocalModel = async () => {
            if (!importModelName.value || !importModelPath.value) {
                ElementPlus.ElMessage.warning($tt('download.importWarn'));
                return;
            }
            
            importing.value = true;
            try {
                // Call backend API to create symlink or copy file
                // Simplified: save to localStorage, prompt user to manually place file
                const SK = window.DQ_STORAGE || {};
                const importedModels = JSON.parse(
                    (SK.IMPORTED_MODELS && localStorage.getItem(SK.IMPORTED_MODELS)) || '[]'
                );
                importedModels.push({
                    name: importModelName.value,
                    path: importModelPath.value,
                    type: importModelType.value,
                    importedAt: new Date().toISOString()
                });
                if (SK.IMPORTED_MODELS) localStorage.setItem(SK.IMPORTED_MODELS, JSON.stringify(importedModels));
                
                ElementPlus.ElMessage.success($tt('download.importSuccess', { name: importModelName.value }));
                importDialogVisible.value = false;
            } catch (e) {
                console.error('Import failed:', e);
                ElementPlus.ElMessage.error($tt('download.importFailed'));
            } finally {
                importing.value = false;
            }
        };
        
        // Get model initials (param is full model row, as name may be i18n object)
        const getModelInitials = (model) => {
            const name = modelLabel(model);
            if (!name) return 'M';
            const dashIndex = name.indexOf('-');
            const spaceIndex = name.indexOf(' ');
            let endIndex = -1;
            if (dashIndex !== -1 && spaceIndex !== -1) {
                endIndex = Math.min(dashIndex, spaceIndex);
            } else if (dashIndex !== -1) {
                endIndex = dashIndex;
            } else if (spaceIndex !== -1) {
                endIndex = spaceIndex;
            }
            if (endIndex !== -1) {
                return name.slice(0, endIndex);
            }
            return name.slice(0, 3);
        };
        
        // Get el-tag type for model type
        const getModelTypeTagType = (type) => {
            const typeMap = {
                'diffusion': 'primary',
                'controlnet': 'warning',
                'upscaler': 'success',
                'tool': 'info',
                'lora': 'danger',
                'video': 'success'
            };
            return typeMap[type] || 'info';
        };
        
        // Category selection
        const handleCategorySelect = (index) => {
            activeCategory.value = index;
        };
        
        // Close all SSE on page unload to prevent connection leaks consuming browser concurrency slots on refresh/close
        const closeAllSSE = () => {
            Object.values(sseConnections.value).forEach(es => {
                try { es.close(); } catch (e) {}
            });
            sseConnections.value = {};
        };

        onMounted(() => {
            loadModelRegistry();
            loadInstalled();
            loadDiskSpace();
            loadActiveDownloads();
            window.addEventListener('beforeunload', closeAllSSE);
        });

        onUnmounted(() => {
            closeAllSSE();
            window.removeEventListener('beforeunload', closeAllSSE);
        });
        
        return {
            $mn: window.$mn,
            $md: window.$md,
            activeCategory,
            categoryTitle,
            categoryIcons,
            filteredModels,
            totalModelCount,
            activeDownloadCount,
            filterQuery,
            refreshing,
            modelsDetailedStatus,
            downloadingModels,
            downloadingLoras,
            downloadingRecommended,
            activeDownloads,
            searchQuery,
            searchType,
            searching,
            searchResults,
            hasSearched,
            selectedVersions,
            installedModels,
            diskSpace,
            importDialogVisible,
            importModelName,
            importModelPath,
            importModelType,
            importing,
            handleCategorySelect,
            refreshStatus,
            downloadVersion,
            quantizeVersion,
            downloadRecommendedSet,
            cancelDownload,
            resumeDownload,
            deleteDownload,
            deleteVersion,
            searchCivitai,
            downloadCivitaiModel,
            formatNumber,
            isModelReady,
            getModelName,
            getModelInitials,
            getModelTypeTagType,
            canDownload,
            getDiskPercent,
            formatBytes,
            getVersionStatus,
            getDependencyHint,
            showImportDialog,
            importLocalModel
        };
    }
};
