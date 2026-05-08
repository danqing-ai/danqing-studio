/**
 * 模型页 — 注册表模型下载、导入与队列（主导航「模型」）
 */

const ModelsPage = {
    template: `
        <div class="models-page" style="display: flex; gap: 20px; height: calc(100vh - 120px);">
            <!-- 左侧分类导航 -->
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
                        
                        <el-menu-item index="base_models">
                            <el-icon><brush /></el-icon>
                            <span>{{ $t('download.baseModels') }}</span>
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
                    
                    <!-- 磁盘空间 -->
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
            
            <!-- 右侧内容区 -->
            <div style="flex: 1; overflow-y: auto;">
                <!-- 模型网格 (分类浏览) -->
                <div v-if="['all', 'base_models', 'controlnets', 'upscalers', 'tools', 'loras'].includes(activeCategory)">
                    <!-- 快速开始工作流 -->
                    <div v-if="activeCategory === 'all' || activeCategory === 'base_models'" 
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

                    <!-- 分类标题 -->
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

                    <!-- 导入本地模型对话框 -->
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
                    
                    <!-- 模型卡片网格 -->
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
                                <!-- 卡片头部：图标/预览 + 状态 -->
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
                                
                                <!-- 卡片内容 -->
                                <div class="model-card-content">
                                    <div class="model-card-name">
                                        {{ $mn(model) }}
                                    </div>
                                    <el-tooltip :content="$md(model)" placement="top" effect="dark">
                                        <div class="model-card-desc">
                                            {{ $md(model) }}
                                        </div>
                                    </el-tooltip>
                                    
                                    <!-- 元信息 -->
                                    <div class="model-card-meta">
                                        <el-tag v-if="model.size" size="small" type="info" effect="plain">{{ model.size }}</el-tag>
                                        <el-tag v-if="model.source === 'huggingface'" size="small" type="primary" effect="plain">HF</el-tag>
                                        <el-tag v-else-if="model.source === 'modelscope'" size="small" type="danger" effect="plain">ModelScope</el-tag>
                                        <el-tag v-else-if="model.source === 'civitai'" size="small" type="warning" effect="plain">CivitAI</el-tag>
                                        <el-tag v-if="model.base_model" size="small" type="success" effect="plain">{{ model.base_model }}</el-tag>
                                    </div>
                                    
                                <!-- 版本列表（注册表项必有 versions，见 check_consistency） -->
                                <div v-if="model.versions" style="margin-bottom: 12px;">
                                    <div v-for="(ver, verKey) in model.versions" :key="verKey" 
                                         style="display: flex; align-items: center; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border-color);">
                                        <div style="flex: 1; min-width: 0;">
                                            <div style="display: flex; align-items: center; gap: 6px;">
                                                <span style="font-size: 13px; color: var(--text-primary);">{{ ver.name }}</span>
                                                <el-tag size="small" type="info" effect="plain">{{ ver.size }}</el-tag>
                                                <el-tag v-if="ver.source_type === 'derived'" size="small" type="warning" effect="plain">{{ $t('download.generated') }}</el-tag>
                                                <el-tag v-else-if="ver.source_type === 'prequantized'" size="small" type="primary" effect="plain">{{ $t('download.prequantized') }}</el-tag>
                                            </div>
                                            <div v-if="ver.source_type === 'derived'" style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                                                {{ $t('download.basedOn', { name: model.versions[ver.from_version]?.name || ver.from_version }) }}
                                            </div>
                                        </div>
                                        <div style="display: flex; gap: 6px; flex-shrink: 0;">
                                            <!-- 根据版本状态显示不同按钮 -->
                                            <template v-if="getVersionStatus(model.id, verKey) === 'ready'">
                                                <el-button type="warning" size="small" @click="downloadVersion(model, verKey)">
                                                    <el-icon><download /></el-icon> {{ $t('download.forceDownload') }}
                                                </el-button>
                                                <el-button type="danger" size="small" @click="deleteVersion(model, verKey)">
                                                    <el-icon><delete /></el-icon>
                                                </el-button>
                                            </template>
                                            <template v-else-if="getVersionStatus(model.id, verKey) === 'generatable'">
                                                <el-button 
                                                    type="primary" 
                                                    size="small"
                                                    @click="convertModel(model, verKey)"
                                                    :loading="convertingModels[model.id + '-' + verKey]"
                                                >
                                                    <el-icon><cpu /></el-icon>
                                                    {{ $t('download.genVersion') }}
                                                </el-button>
                                            </template>
                                            <template v-else-if="getVersionStatus(model.id, verKey) === 'parent_missing'">
                                                <el-tag size="small" type="info" effect="plain">{{ $t('download.waitingParent') }}</el-tag>
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
                    
                    <!-- LoRA 搜索 -->
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
                
                <!-- 已安装 -->
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
                
                <!-- 下载中 -->
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
        
        // 分类导航
        const activeCategory = ref('all');
        const categoryTitle = computed(() => {
            const titles = {
                'all': $tt('download.allModels'),
                'base_models': '🎨 ' + $tt('download.baseModels'),
                'controlnets': '🎯 ' + $tt('download.controlNet'),
                'upscalers': '🔍 ' + $tt('download.upscalers'),
                'tools': '⚙️ ' + $tt('download.tools'),
                'loras': '🎭 ' + $tt('download.loraModels')
            };
            return titles[activeCategory.value] || $tt('download.title');
        });
        const categoryIcons = {
            'base_models': '🎨',
            'controlnets': '🎯',
            'upscalers': '🔍',
            'tools': '⚙️',
            'loras': '🎭'
        };
        
        // 模型数据
        const modelRegistry = ref({});
        const modelsStatus = ref({});
        const modelsDetailedStatus = ref({});
        const categories = ref({});
        const filterQuery = ref('');
        const refreshing = ref(false);
        
        // 下载状态
        const downloadingModels = ref({});
        const downloadingLoras = ref({});
        const downloadingRecommended = ref(false);
        const activeDownloads = ref({});
        const activeConversions = ref({});
        const convertingModels = ref({});
        const selectedVersions = ref({});
        const sseConnections = ref({});
        
        // CivitAI 搜索
        const searchQuery = ref('');
        const searchType = ref('LORA');
        const searching = ref(false);
        const searchResults = ref([]);
        const hasSearched = ref(false);
        
        // 已安装
        const installedModels = ref([]);
        
        // 磁盘空间
        const diskSpace = ref(null);
        
        // 导入本地模型
        const importDialogVisible = ref(false);
        const importModelName = ref('');
        const importModelPath = ref('');
        const importModelType = ref('base');
        const importing = ref(false);

        /** 注册表 v2：`name` / `description` 可能为 { zh, en }（GET /api/registry 原样） */
        const modelLabel = (m) => (typeof window.$mn === 'function' ? window.$mn(m, m.id) : (m.id || ''));
        const modelSearchBlob = (m) => {
            const n = modelLabel(m);
            const d = typeof window.$md === 'function' ? window.$md(m, '') : '';
            return `${n} ${d}`.toLowerCase();
        };
        
        // 过滤后的模型列表
        const filteredModels = computed(() => {
            let list = [];
            for (const [id, config] of Object.entries(modelRegistry.value)) {
                const model = {
                    id,
                    ...config,
                    ready: modelsStatus.value[id] || false
                };
                
                // 分类过滤
                if (activeCategory.value !== 'all' && model.category !== activeCategory.value) {
                    continue;
                }
                
                // 搜索过滤
                if (filterQuery.value) {
                    const query = filterQuery.value.toLowerCase();
                    if (!modelSearchBlob(model).includes(query)) {
                        continue;
                    }
                }
                
                list.push(model);
            }
            
            // 排序：推荐的排前面，然后按名称排序
            return list.sort((a, b) => {
                if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
                return modelLabel(a).localeCompare(modelLabel(b));
            });
        });
        
        const totalModelCount = computed(() => Object.keys(modelRegistry.value).length);
        
        // 活跃下载任务数量（包含暂停中的任务）
        const activeDownloadCount = computed(() => {
            return Object.values(activeDownloads.value).filter(item => item.status === 'running' || item.status === 'paused' || item.status === 'failed').length;
        });
        
        // 加载模型注册表
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
        
        // 刷新状态
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
        
        // 加载已安装
        const loadInstalled = async () => {
            try {
                const models = await api.settings.listModels();
                installedModels.value = models || [];
            } catch (e) {
                console.error('Failed to load installed:', e);
            }
        };
        
        // 加载磁盘空间
        const loadDiskSpace = async () => {
            try {
                const data = await api.settings.getDiskSpace();
                diskSpace.value = data;
            } catch (e) {
                console.error('Failed to load disk space:', e);
            }
        };
        
        // 加载活跃下载任务（刷新页面后恢复）
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
                        // 只有真正在运行的任务才连接 SSE
                        if (task.status === 'running') {
                            connectProgressSSE(task.id, task.filename || task.url);
                        }
                    }
                }
            } catch (e) {
                console.error('Failed to load active downloads:', e);
            }
        };
        
        // 磁盘使用百分比
        const getDiskPercent = (type) => {
            if (!diskSpace.value || !diskSpace.value[type]) return 0;
            const info = diskSpace.value[type];
            if (!info.exists || info.total === 0) return 0;
            return Math.round(info.size / info.total * 100);
        };
        
        // 格式化文件大小
        const formatBytes = (bytes) => {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
        };
        
        // 检查模型是否就绪
        const isModelReady = (modelId) => {
            return modelsStatus.value[modelId] || false;
        };
        
        // 获取模型名称（注册表项或 id）
        const getModelName = (modelId) => {
            const cfg = modelRegistry.value[modelId];
            if (!cfg) return modelId;
            const row = { id: modelId, ...cfg };
            return modelLabel(row);
        };
        
        // 检查是否可以下载（依赖是否满足）
        const canDownload = (model) => {
            if (!model.dependencies || model.dependencies.length === 0) return true;
            return model.dependencies.every(dep => isModelReady(dep));
        };

        // 获取依赖不满足时的提示信息
        const getDependencyHint = (model) => {
            if (!model.dependencies || model.dependencies.length === 0) return '';
            const missing = model.dependencies.filter(dep => !isModelReady(dep));
            if (missing.length === 0) return '';
            const names = missing.map(d => getModelName(d)).join('、');
            return $tt('download.dependencyMissing', { models: names });
        };
        
        // 获取版本状态
        const getVersionStatus = (modelId, versionKey) => {
            const detail = modelsDetailedStatus.value[modelId];
            if (!detail || !detail.versions) return 'missing';
            
            const verStatus = detail.versions[versionKey];
            if (!verStatus) return 'missing';
            
            if (verStatus.ready) return 'ready';
            
            // 检查是否是 derived 版本
            const model = modelRegistry.value[modelId];
            if (model && model.versions && model.versions[versionKey]) {
                const ver = model.versions[versionKey];
                if (ver.source_type === 'derived') {
                    // 检查父版本是否就绪
                    const parentVer = ver.from_version;
                    if (parentVer && detail.versions[parentVer] && detail.versions[parentVer].ready) {
                        return 'generatable';
                    }
                    return 'parent_missing';
                }
            }
            
            return 'missing';
        };
        
        // SSE 转换进度连接
        const connectConversionSSE = (taskId, name) => {
            if (sseConnections.value[taskId]) {
                sseConnections.value[taskId].close();
            }
            
            const eventSource = new EventSource(api.download.convertProgressStreamUrl(taskId));
            sseConnections.value[taskId] = eventSource;
            
            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    activeConversions.value[taskId] = {
                        name: name,
                        progress: data.progress || 0,
                        status: data.status,
                        stage: data.stage,
                        error: data.error_message || ''
                    };
                    
                    if (data.status === 'completed') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        setTimeout(() => {
                            delete activeConversions.value[taskId];
                            delete convertingModels.value[`${data.model_name}-${data.to_version}`];
                        }, 2000);
                        ElementPlus.ElMessage.success($tt('download.genComplete', { name }));
                        refreshStatus();
                    } else if (data.status === 'failed') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        ElementPlus.ElMessage.error($tt('download.genFailed', { name, msg: data.error_message }));
                        delete convertingModels.value[`${data.model_name}-${data.to_version}`];
                    } else if (data.status === 'cancelled') {
                        eventSource.close();
                        delete sseConnections.value[taskId];
                        delete activeConversions.value[taskId];
                        delete convertingModels.value[`${data.model_name}-${data.to_version}`];
                        ElementPlus.ElMessage.info($tt('download.genCancelled', { name }));
                    }
                } catch (e) {
                    console.error('SSE parse error:', e);
                }
            };
            
            eventSource.onerror = (error) => {
                console.error('SSE error:', error);
                eventSource.close();
                delete sseConnections.value[taskId];
            };
        };
        
        // 转换模型（生成量化版本）
        const convertModel = async (model, versionKey) => {
            const key = `${model.id}-${versionKey}`;
            if (convertingModels.value[key]) return;
            
            const version = model.versions[versionKey];
            if (!version || !version.from_version) {
                ElementPlus.ElMessage.error($tt('download.versionConfigError'));
                return;
            }
            
            convertingModels.value[key] = true;
            
            try {
                ElementPlus.ElMessage.info($tt('download.startConvert', { name: modelLabel(model), version: version.name }));
                
                const data = await api.download.startConvert({
                    model_name: model.id,
                    from_version: version.from_version,
                    to_version: versionKey,
                });
                connectConversionSSE(data.task_id, `${modelLabel(model)} ${version.name}`);
                
            } catch (e) {
                console.error('Conversion failed:', e);
                ElementPlus.ElMessage.error($tt('download.convertFailed', { msg: e.message }));
                delete convertingModels.value[key];
            }
        };
        
        // 下载指定版本
        const downloadVersion = async (model, versionKey) => {
            const key = `${model.id}-${versionKey}`;
            if (downloadingModels.value[key]) return;
            
            downloadingModels.value[key] = true;
            
            try {
                const version = model.versions[versionKey];
                
                const data = await api.models.install(model.id, { version: versionKey });
                connectProgressSSE(data.task_id, `${modelLabel(model)} ${version?.name || ''}`, key);
                
            } catch (e) {
                console.error('Download failed:', e);
                ElementPlus.ElMessage.error($tt('download.downloadFailed', { msg: e.message }));
                delete downloadingModels.value[key];
            }
        };
        
        // 删除指定版本
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
        
        // 取消转换
        const cancelConversion = async (taskId) => {
            try {
                await api.download.cancelConvert(taskId);
            } catch (e) {
                console.error('Cancel conversion failed:', e);
            }
        };
        
        // SSE 进度连接；`downloadingKey` 与 `downloadingModels` 的键一致（多版本安装为 `${modelId}-${versionKey}`）
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
        
        // 批量下载推荐模型
        const downloadRecommendedSet = async () => {
            if (downloadingRecommended.value) return;
            downloadingRecommended.value = true;
            
            try {
                const modelsToDownload = ['flux1-schnell']; // 可以扩展为更多
                
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
        
        // 取消下载
        const cancelDownload = async (taskId) => {
            try {
                await api.download.cancel(taskId);
            } catch (e) {
                console.error('Cancel failed:', e);
            }
        };
        
        // 恢复下载
        const resumeDownload = async (taskId) => {
            try {
                const item = activeDownloads.value[taskId];
                if (!item) return;
                
                await api.download.resume(taskId);
                
                // 更新状态为 running
                item.status = 'running';
                
                // 连接 SSE 监听进度
                connectProgressSSE(taskId, item.name);
                
                ElementPlus.ElMessage.success($tt('download.resumeStart', { name: item.name }));
            } catch (e) {
                console.error('Resume failed:', e);
                ElementPlus.ElMessage.error($tt('download.resumeFailed', { msg: e.message }));
            }
        };

        const deleteDownload = async (taskId) => {
            try {
                await api.download.delete(taskId);
                delete activeDownloads.value[taskId];
                ElementPlus.ElMessage.success($tt('download.deleteSuccess'));
            } catch (e) {
                console.error('Delete download failed:', e);
                ElementPlus.ElMessage.error($tt('download.deleteFailed'));
            }
        };
        
        // CivitAI 搜索
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
        
        // 下载 CivitAI 模型
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
        
        // 格式化数字
        const formatNumber = (num) => {
            if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
            if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
            return num.toString();
        };
        
        // 导入本地模型
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
                // 这里应该调用后端 API 创建软链接或复制文件
                // 简化处理：前端记录到 localStorage，提示用户手动放置文件
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
        
        // 获取模型名称首字母缩写（入参为整行 model，因 name 可能为 i18n 对象）
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
        
        // 获取模型类型对应的标签类型
        const getModelTypeTagType = (type) => {
            const typeMap = {
                'diffusion': 'primary',
                'controlnet': 'warning',
                'upscaler': 'success',
                'tool': 'info',
                'lora': 'danger'
            };
            return typeMap[type] || 'info';
        };
        
        // 分类选择
        const handleCategorySelect = (index) => {
            activeCategory.value = index;
        };
        
        // 页面卸载时关闭所有 SSE，防止刷新/关闭时连接泄漏占用浏览器并发槽
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
            activeConversions,
            convertingModels,
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
            downloadRecommendedSet,
            cancelDownload,
            resumeDownload,
            deleteDownload,
            cancelConversion,
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
            convertModel,
            showImportDialog,
            importLocalModel
        };
    }
};
