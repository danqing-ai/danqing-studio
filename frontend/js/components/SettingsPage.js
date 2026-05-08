/**
 * 设置页面组件 - 美化版
 */

const SettingsPage = {
    template: `
        <div class="settings-page">
            <el-tabs type="border-card" v-model="activeTab">
                
                <!-- 模型配置 -->
                <el-tab-pane :label="$t('settings.modelConfig')" name="models">
                    <div class="card" style="margin-bottom: 24px;">
                        <div class="card-title">
                            <el-icon><box /></el-icon>
                            {{ $t('settings.modelConfig') }}
                            <span style="color: var(--text-muted); font-size: 13px; font-weight: normal; margin-left: 8px;">
                                {{ $t('settings.modelConfigDesc') }}
                            </span>
                        </div>
                        
                        <!-- 模型选择 -->
                        <el-select v-model="selectedModel" style="width: 100%; margin-bottom: 20px;" size="large"
                            @change="onModelSelect"
                        >
                            <el-option
                                v-for="(config, key) in modelRegistry"
                                :key="key"
                                :label="$mn(config)"
                                :value="key"
                            >
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span>{{ $mn(config) }}</span>
                                    <el-tag v-if="config.recommended" size="small" type="success">{{ $t('studio.recommended') }}</el-tag>
                                    <el-tag size="small" type="info">{{ config.engine }}</el-tag>
                                </div>
                            </el-option>
                        </el-select>
                        
                        <!-- 当前模型配置 -->
                        <div v-if="currentModelConfig">
                            <!-- 模型信息卡片 -->
                            <div class="model-info-card">
                                <div class="model-info-grid">
                                    <div class="model-info-item">
                                        <div class="model-info-label">
                                            <el-icon><cpu /></el-icon>
                                            {{ $t('settings.engine') }}
                                        </div>
                                        <div class="model-info-value">{{ currentModelConfig.engine }}</div>
                                    </div>
                                    <div class="model-info-item">
                                        <div class="model-info-label">
                                            <el-icon><document /></el-icon>
                                            {{ $t('settings.modelType') }}
                                        </div>
                                        <div class="model-info-value">{{ currentModelConfig.type }}</div>
                                    </div>
                                    <div class="model-info-item full-width">
                                        <div class="model-info-label">
                                            <el-icon><star /></el-icon>
                                            {{ $t('settings.modelActions') }}
                                        </div>
                                        <div class="model-info-tags">
                                            <el-tag 
                                                v-for="key in modelActionKeyList" 
                                                :key="key" 
                                                size="small" 
                                                effect="dark"
                                                class="action-tag"
                                            >
                                                {{ actionTagLabel(key) }}
                                            </el-tag>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- 模型参数配置（Plan C3：与创作页共用 RegistryParamsForm + 注册表 schema） -->
                            <div class="model-params-section">
                                <h4 class="section-title">
                                    <el-icon><sliders /></el-icon>
                                    {{ $t('settings.parameters') }}
                                </h4>
                                <registry-params-form
                                    v-if="currentModelConfig"
                                    :model-config="currentModelConfig"
                                    :params="modelParams"
                                    :loras="settingsLorasForForm"
                                    :controlnets="null"
                                    control-image-src=""
                                    :control-recent-gallery="[]"
                                    @restore-defaults="onSettingsModelRestoreDefaults"
                                />
                                <div class="save-button-wrapper">
                                    <el-button type="primary" @click="saveModelConfig" class="save-button">
                                        <el-icon><check /></el-icon>
                                        {{ $t('common.save') }}
                                    </el-button>
                                </div>
                            </div>
                        </div>
                    </div>
                </el-tab-pane>
                
                <!-- 提示词模板 -->
                <el-tab-pane :label="$t('settings.promptTemplates')" name="presets">
                    <div class="card" style="margin-bottom: 24px;">
                        <div class="card-title" style="justify-content: space-between;">
                            <span>
                                <el-icon><collection /></el-icon>
                                {{ $t('settings.promptTemplates') }}
                                <span style="color: var(--text-muted); font-size: 13px; font-weight: normal; margin-left: 8px;">
                                    {{ $t('settings.promptTemplatesDesc') }}
                                </span>
                            </span>
                            <el-button type="primary" size="small" @click="openPresetDialog()">
                                <el-icon><plus /></el-icon>
                                {{ $t('settings.addTemplate') }}
                            </el-button>
                        </div>
                        
                        <el-empty v-if="Object.keys(presets).length === 0" :description="$t('settings.noTemplates')" />
                        
                        <el-table v-else :data="presetList" style="width: 100%">
                            <el-table-column :label="$t('settings.templateName')" prop="name" min-width="150" />
                            <el-table-column :label="$t('settings.presetMediaScope')" width="140">
                                <template #default="{ row }">
                                    <span style="font-size: 12px;">{{ presetMediaLabel(row.preset) }}</span>
                                </template>
                            </el-table-column>
                            <el-table-column :label="$t('settings.positivePrompt')" min-width="250">
                                <template #default="{ row }">
                                    <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 300px;">
                                        {{ row.preset.positive || '-' }}
                                    </div>
                                </template>
                            </el-table-column>
                            <el-table-column :label="$t('settings.negativePrompt')" min-width="250">
                                <template #default="{ row }">
                                    <div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 300px;">
                                        {{ row.preset.negative || '-' }}
                                    </div>
                                </template>
                            </el-table-column>
                            <el-table-column :label="$t('settings.presetAppliesTo')" min-width="160">
                                <template #default="{ row }">
                                    <span style="font-size: 12px; color: var(--text-muted);">{{ presetAppliesSummary(row.preset) }}</span>
                                </template>
                            </el-table-column>
                            <el-table-column :label="$t('common.action')" width="150" fixed="right">
                                <template #default="{ row }">
                                    <el-button size="small" text type="primary" @click="openPresetDialog(row.name, row.preset)">
                                        <el-icon><edit /></el-icon>
                                    </el-button>
                                    <el-button size="small" text type="danger" @click="confirmDeletePreset(row.name)">
                                        <el-icon><delete /></el-icon>
                                    </el-button>
                                </template>
                            </el-table-column>
                        </el-table>
                    </div>
                    
                    <!-- 添加/编辑模板对话框 -->
                    <el-dialog
                        v-model="presetDialogVisible"
                        :title="editingPresetName ? $t('settings.editTemplate') : $t('settings.addTemplate')"
                        width="600px"
                    >
                        <el-form :model="presetForm" label-position="top">
                            <el-form-item :label="$t('settings.templateName')" required>
                                <el-input v-model="presetForm.name" :placeholder="$t('settings.presetNamePlaceholder')" />
                            </el-form-item>
                            <el-form-item :label="$t('settings.presetMediaScope')">
                                <el-radio-group v-model="presetForm.media_scope">
                                    <el-radio-button label="image">{{ $t('settings.presetMediaImage') }}</el-radio-button>
                                    <el-radio-button label="video">{{ $t('settings.presetMediaVideo') }}</el-radio-button>
                                </el-radio-group>
                                <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
                                    {{ $t('settings.presetMediaScopeHint') }}
                                </div>
                            </el-form-item>
                            <el-form-item :label="$t('settings.positivePrompt')">
                                <el-input
                                    v-model="presetForm.positive"
                                    type="textarea"
                                    :rows="4"
                                    :placeholder="$t('settings.positivePlaceholder')"
                                />
                            </el-form-item>
                            <el-form-item :label="$t('settings.negativePrompt')">
                                <el-input
                                    v-model="presetForm.negative"
                                    type="textarea"
                                    :rows="2"
                                    :placeholder="$t('settings.negativePlaceholder')"
                                />
                            </el-form-item>
                            <el-form-item :label="$t('settings.presetAppliesTo')">
                                <el-checkbox-group v-model="presetForm.applies_to">
                                    <el-checkbox label="create">{{ $t('action.image.create') }}</el-checkbox>
                                    <el-checkbox label="rewrite">{{ $t('action.image.rewrite') }}</el-checkbox>
                                    <el-checkbox label="retouch">{{ $t('action.image.retouch') }}</el-checkbox>
                                    <el-checkbox label="extend">{{ $t('action.image.extend') }}</el-checkbox>
                                    <el-checkbox label="upscale">{{ $t('action.image.upscale') }}</el-checkbox>
                                    <el-checkbox label="animate">{{ $t('action.video.animate') }}</el-checkbox>
                                </el-checkbox-group>
                            </el-form-item>
                        </el-form>
                        <template #footer>
                            <el-button @click="presetDialogVisible = false">{{ $t('common.cancel') }}</el-button>
                            <el-button type="primary" @click="savePreset">{{ $t('common.save') }}</el-button>
                        </template>
                    </el-dialog>
                </el-tab-pane>
                
                <!-- 系统设置 -->
                <el-tab-pane :label="$t('settings.systemConfig')" name="system">
                    <el-row :gutter="24">
                        <!-- 左侧：公共配置 + 模型列表 + LoRA 列表 -->
                        <el-col :xs="24" :md="16">
                            <!-- 公共配置 -->
                            <div class="card" style="margin-bottom: 24px;">
                                <div class="card-title">
                                    <el-icon><setting /></el-icon>
                                    {{ $t('settings.publicConfig') }}
                                    <span style="color: var(--text-muted); font-size: 13px; font-weight: normal; margin-left: 8px;">
                                        {{ $t('settings.publicConfigDesc') }}
                                    </span>
                                </div>
                                
                                <el-form :model="settings" label-position="top">
                                    <el-form-item :label="$t('settings.defaultModel')">
                                        <el-select v-model="settings.default_model" style="width: 100%" :placeholder="$t('settings.selectDefaultModel')">
                                            <el-option
                                                v-for="model in installedModels"
                                                :key="model.name"
                                                :label="model.name"
                                                :value="model.name"
                                            />
                                        </el-select>
                                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                            {{ $t('settings.defaultModelDesc') }}
                                        </div>
                                    </el-form-item>
                                    
                                    <el-row :gutter="16">
                                        <el-col :span="12">
                                            <el-form-item :label="$t('settings.language')">
                                                <el-select v-model="settings.language" @change="handleLanguageChange">
                                                    <el-option :label="$t('settings.label_zh')" value="zh" />
                                                    <el-option :label="$t('settings.label_en')" value="en" />
                                                </el-select>
                                            </el-form-item>
                                        </el-col>
                                        
                                        <el-col :span="12">
                                            <el-form-item :label="$t('settings.outputFormat')">
                                                <el-select v-model="settings.output_format">
                                                    <el-option label="PNG" value="png" />
                                                    <el-option label="JPEG" value="jpg" />
                                                    <el-option label="WebP" value="webp" />
                                                </el-select>
                                            </el-form-item>
                                        </el-col>
                                    </el-row>
                                    
                                    <el-form-item :label="$t('settings.memoryLimit')">
                                        <div class="param-control-row">
                                            <div class="param-slider">
                                                <el-slider v-model="settings.mlx_memory_limit" :min="32" :max="256" :step="8" />
                                            </div>
                                            <span style="color: var(--text-muted); white-space: nowrap;">{{ settings.mlx_memory_limit }} GB</span>
                                        </div>
                                    </el-form-item>

                                    <el-form-item :label="$t('settings.modelCacheTtl')">
                                        <div class="param-control-row">
                                            <div class="param-slider">
                                                <el-slider v-model="settings.model_cache_ttl_minutes" :min="5" :max="120" :step="5" />
                                            </div>
                                            <span style="color: var(--text-muted); white-space: nowrap;">{{ settings.model_cache_ttl_minutes }} min</span>
                                        </div>
                                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                            {{ $t('settings.modelCacheTtlDesc') }}
                                        </div>
                                    </el-form-item>

                                    <el-form-item :label="$t('settings.queueImageFirst')">
                                        <el-switch v-model="settings.queue_image_first" />
                                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                            {{ $t('settings.queueImageFirstDesc') }}
                                        </div>
                                    </el-form-item>

                                    <el-form-item :label="$t('settings.theme')">
                                        <el-radio-group v-model="settings.theme">
                                            <el-radio-button label="dark">{{ $t('settings.themeDark') }}</el-radio-button>
                                            <el-radio-button label="light">{{ $t('settings.themeLight') }}</el-radio-button>
                                        </el-radio-group>
                                    </el-form-item>

                                    <el-form-item :label="$t('settings.autoSavePrompts')">
                                        <el-switch v-model="settings.auto_save_prompts" />
                                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                            {{ $t('settings.autoSavePromptsDesc') }}
                                        </div>
                                    </el-form-item>

                                    <el-form-item :label="$t('settings.customModelsDir')">
                                        <el-input v-model="settings.custom_models_dir" clearable :placeholder="'./models'" />
                                    </el-form-item>
                                    <el-form-item :label="$t('settings.customLorasDir')">
                                        <el-input v-model="settings.custom_loras_dir" clearable :placeholder="'./loras'" />
                                    </el-form-item>
                                    <el-form-item :label="$t('settings.customOutputsDir')">
                                        <el-input v-model="settings.custom_outputs_dir" clearable :placeholder="'./outputs'" />
                                    </el-form-item>
                                    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">
                                        {{ $t('settings.customPathsHint') }}
                                    </div>
                                    
                                    <el-divider>HuggingFace</el-divider>

                                    <el-form-item :label="$t('settings.huggingfaceToken')">
                                        <el-input v-model="settings.huggingface_token" type="password" show-password
                                            :placeholder="$t('settings.huggingfaceTokenPlaceholder')">
                                            <template #prefix>
                                                <el-icon><key /></el-icon>
                                            </template>
                                        </el-input>
                                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                            {{ $t('settings.huggingfaceTokenDesc') }}
                                        </div>
                                    </el-form-item>

                                    <el-divider>CivitAI</el-divider>

                                    <el-form-item :label="$t('settings.civitaiToken')">
                                        <el-input v-model="settings.civitai_token" type="password" show-password
                                            :placeholder="$t('settings.civitaiTokenPlaceholder')">
                                            <template #prefix>
                                                <el-icon><key /></el-icon>
                                            </template>
                                        </el-input>
                                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                            {{ $t('settings.civitaiTokenDesc') }}
                                        </div>
                                    </el-form-item>

                                    <el-form-item v-if="settings.civitai_token">
                                        <el-checkbox v-model="settings.nsfw_enabled" size="large">
                                            <span style="color: var(--danger);">{{ $t('settings.nsfwContent') }}</span>
                                        </el-checkbox>
                                        <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                            {{ $t('settings.nsfwDesc') }}
                                        </div>
                                    </el-form-item>

                                    <el-form-item>
                                        <el-button type="primary" @click="saveSettings">
                                            <el-icon><check /></el-icon>
                                            {{ $t('common.save') }}
                                        </el-button>
                                    </el-form-item>
                                </el-form>
                            </div>
                            
                        </el-col>
                        
                        <!-- 右侧：系统信息 + 实时资源监控 -->
                        <el-col :xs="24" :md="8">
                            <!-- 系统信息 -->
                            <div class="card" style="margin-bottom: 24px;">
                                <div class="card-title">
                                    <el-icon><cpu /></el-icon>
                                    {{ $t('settings.systemInfo') }}
                                </div>
                                
                                <div class="system-info-grid">
                                    <div class="info-item">
                                        <div class="info-icon">
                                            <el-icon style="font-size: 20px;"><monitor /></el-icon>
                                        </div>
                                        <div class="info-content">
                                            <div class="info-label">{{ $t('settings.platform') }}</div>
                                            <div class="info-value">{{ systemInfo.platform }} {{ systemInfo.architecture }}</div>
                                        </div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-icon">
                                            <el-icon style="font-size: 20px;"><cpu /></el-icon>
                                        </div>
                                        <div class="info-content">
                                            <div class="info-label">{{ $t('settings.memory') }}</div>
                                            <div class="info-value">{{ systemInfo.memory_gb?.toFixed(1) }} GB</div>
                                        </div>
                                    </div>
                                    <div class="info-item">
                                        <div class="info-icon">
                                            <el-icon style="font-size: 20px;"><document /></el-icon>
                                        </div>
                                        <div class="info-content">
                                            <div class="info-label">{{ $t('settings.pythonVersion') }}</div>
                                            <div class="info-value">{{ systemInfo.python_version }}</div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- 依赖版本 -->
                                <div v-if="systemInfo.dependencies" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                                    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">{{ $t('settings.dependencies') }}</div>
                                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                                        <el-tag v-for="(version, name) in systemInfo.dependencies" :key="name" size="small" type="info" effect="plain">
                                            {{ name }} {{ version }}
                                        </el-tag>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- 模型缓存状态 -->
                            <div class="card" style="margin-bottom: 24px;">
                                <div class="card-title" style="justify-content: space-between; align-items: center;">
                                    <span style="display: flex; align-items: center; gap: 8px;">
                                        <el-icon style="font-size: 18px; color: var(--primary);"><collection /></el-icon>
                                        {{ $t('settings.modelCacheTitle') }}
                                    </span>
                                    <el-button size="small" text @click="refreshCacheStatus" :loading="cacheLoading">
                                        <el-icon><refresh /></el-icon>
                                    </el-button>
                                </div>
                                <div v-if="cacheError" style="color: var(--el-color-danger); font-size: 13px; margin-top: 8px;">{{ cacheError }}</div>
                                <template v-else>
                                    <div v-if="cacheStatus && cacheStatus.cache" style="font-size: 12px; color: var(--text-muted); margin: 12px 0;">
                                        {{ $tt('settings.modelCacheTotal', {
                                            count: cacheStatus.cache.cached_models,
                                            total: cacheStatus.cache.total_gb,
                                            limit: cacheStatus.cache.limit_gb
                                        }) }}
                                    </div>
                                    <div v-if="cacheStatus && cacheStatus.cache && cacheStatus.cache.models && cacheStatus.cache.models.length" class="cache-list">
                                        <div v-for="m in cacheStatus.cache.models" :key="m.key" class="cache-item">
                                            <div class="cache-item-icon">
                                                <el-icon><cpu /></el-icon>
                                            </div>
                                            <div class="cache-item-info">
                                                <div class="cache-item-name">{{ m.key }}</div>
                                                <div class="cache-item-meta">
                                                    {{ $tt('settings.modelCacheIdle', { minutes: m.idle_minutes }) }}
                                                </div>
                                            </div>
                                            <div class="cache-item-size">{{ m.size_gb }} GB</div>
                                        </div>
                                    </div>
                                    <el-empty
                                        v-else-if="!cacheStatus || !cacheStatus.cache || !cacheStatus.cache.models || !cacheStatus.cache.models.length"
                                        :description="$t('settings.modelCacheEmpty')"
                                    />
                                </template>
                            </div>
                            
                            <!-- 实时资源监控 -->
                            <div class="card system-monitor-card">
                                <div class="card-title">
                                    <el-icon><monitor /></el-icon>
                                    {{ $t('settings.resourceMonitor') }}
                                    <span style="margin-left: auto; color: var(--text-muted); font-size: 12px; font-weight: normal;">
                                        {{ $t('settings.realtime') }}
                                    </span>
                                </div>
                                
                                <!-- CPU -->
                                <div class="monitor-item">
                                    <div class="monitor-label">
                                        <el-icon><cpu /></el-icon>
                                        <span>{{ $t('settings.cpu') }}</span>
                                        <span class="monitor-value">{{ monitorData.cpu_percent }}%</span>
                                    </div>
                                    <el-progress 
                                        :percentage="monitorData.cpu_percent" 
                                        :color="getProgressColor(monitorData.cpu_percent)"
                                        :show-text="false"
                                        :stroke-width="8"
                                    />
                                </div>
                                
                                <!-- 内存 -->
                                <div class="monitor-item">
                                    <div class="monitor-label">
                                        <el-icon><menu /></el-icon>
                                        <span>{{ $t('settings.memoryLabel') }}</span>
                                        <span class="monitor-value">
                                            {{ monitorData.memory.used_gb }} / {{ monitorData.memory.total_gb }} GB
                                        </span>
                                    </div>
                                    <el-progress 
                                        :percentage="monitorData.memory.percent" 
                                        :color="getProgressColor(monitorData.memory.percent)"
                                        :show-text="false"
                                        :stroke-width="8"
                                    />
                                </div>
                                
                                <!-- GPU -->
                                <div v-if="monitorData.gpu" class="monitor-item">
                                    <div class="monitor-label">
                                        <el-icon><odometer /></el-icon>
                                        <span>{{ $t('settings.gpu') }}</span>
                                        <span v-if="monitorData.gpu.model" class="monitor-value">
                                            {{ monitorData.gpu.model }}
                                        </span>
                                    </div>
                                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px; display: flex; gap: 12px;">
                                        <span v-if="monitorData.gpu.memory_gb">{{ monitorData.gpu.memory_gb }} {{ $t('settings.unifiedMemory') }}</span>
                                        <span v-if="monitorData.gpu.note">{{ monitorData.gpu.note }}</span>
                                    </div>
                                </div>
                                
                                    <div style="text-align: center; margin-top: 12px;">
                                    <el-tag size="small" type="info" effect="plain">
                                        {{ $t('settings.refreshInterval') }}
                                    </el-tag>
                                </div>
                            </div>
                        </el-col>
                    </el-row>
                </el-tab-pane>
            </el-tabs>
        </div>
    `,

    setup() {
        const { ref, reactive, onMounted, inject, computed, watch, onUnmounted } = Vue;
        const systemInfo = inject('systemInfo');
        const SK = window.DQ_STORAGE || {};

        // 恢复标签页状态
        const activeTab = ref((SK.SETTINGS_TAB && localStorage.getItem(SK.SETTINGS_TAB)) || 'models');
        // 语言始终跟随 i18n，不从后端加载
        const settings = reactive({
            language: (typeof i18n !== 'undefined' ? i18n.global.locale.value : 'zh'),
            theme: 'dark',
            default_model: '',
            auto_save_prompts: true,
            output_format: 'png',
            mlx_memory_limit: 120,
            model_cache_ttl_minutes: 30,
            queue_image_first: false,
            civitai_token: '',
            huggingface_token: '',
            nsfw_enabled: false,
            custom_models_dir: '',
            custom_loras_dir: '',
            custom_outputs_dir: '',
        });

        // 模型配置
        const modelRegistry = ref({});
        const selectedModel = ref('');
        const modelParams = reactive({
            steps: 4,
            guidance: 3.5,
            guide_scale: 3.0,
            shift: 0.0,
            num_frames: 97,
            fps: 24,
            width: 1024,
            height: 1024,
            lora: '',
            lora_scale: 0.8,
            temperature: 0.7,
            system_prompt: '',
            scheduler: 'flow_match_euler_discrete',
            strength: 0.4,
            seed: '',
            controlnet: '',
            controlnet_strength: 0.8,
        });

        /** 设置页：与当前模型兼容的已安装 LoRA（供 AdapterPicker） */
        const settingsCompatibleLoras = ref([]);

        // 已安装的模型和LoRA
        const installedModels = ref([]);
        
        // 提示词模板
        const presets = ref({});
        const presetDialogVisible = ref(false);
        const editingPresetName = ref('');
        const presetForm = reactive({
            name: '',
            positive: '',
            negative: '',
            media_scope: 'image',
            applies_to: ['create'],
        });

        const presetList = computed(() => {
            return Object.entries(presets.value).map(([name, preset]) => ({
                name,
                preset
            }));
        });

        const presetAppliesSummary = (preset) => preset.applies_to.join(', ');

        const presetMediaLabel = (preset) =>
            preset.media_scope === 'video' ? $tt('settings.presetMediaVideo') : $tt('settings.presetMediaImage');

        // 加载提示词模板
        const loadPresets = async () => {
            try {
                const data = await api.settings.getPresets();
                presets.value = data || {};
            } catch (e) {
                console.error('Failed to load presets:', e);
            }
        };

        // 打开模板对话框
        const openPresetDialog = (name = '', preset = null) => {
            editingPresetName.value = name;
            if (name && preset) {
                presetForm.name = name;
                presetForm.positive = preset.positive || '';
                presetForm.negative = preset.negative || '';
                presetForm.applies_to = [...preset.applies_to];
                presetForm.media_scope = preset.media_scope;
            } else {
                presetForm.name = '';
                presetForm.positive = '';
                presetForm.negative = '';
                presetForm.media_scope = 'image';
                presetForm.applies_to = ['create'];
            }
            presetDialogVisible.value = true;
        };

        // 保存模板
        const savePreset = async () => {
            if (!presetForm.name.trim()) {
                ElementPlus.ElMessage.warning($tt('settings.enterTemplateName'));
                return;
            }
            try {
                if (!Array.isArray(presetForm.applies_to) || !presetForm.applies_to.length) {
                    ElementPlus.ElMessage.warning($tt('settings.presetAppliesRequired'));
                    return;
                }
                const applies = [...presetForm.applies_to];
                await api.settings.savePreset(presetForm.name.trim(), {
                    positive: presetForm.positive,
                    negative: presetForm.negative,
                    media_scope: presetForm.media_scope,
                    applies_to: applies,
                });
                ElementPlus.ElMessage.success($tt('settings.templateSaved'));
                presetDialogVisible.value = false;
                await loadPresets();
            } catch (e) {
                console.error('Failed to save preset:', e);
                ElementPlus.ElMessage.error($tt('settings.saveFailed'));
            }
        };

        // 确认删除模板
        const confirmDeletePreset = (name) => {
            ElementPlus.ElMessageBox.confirm(
                $tt('settings.deletePresetConfirm', { name }),
                $tt('settings.deletePresetTitle'),
                {
                    confirmButtonText: $tt('settings.deleteConfirm'),
                    cancelButtonText: $tt('settings.deleteCancel'),
                    type: 'warning'
                }
            ).then(() => {
                deletePreset(name);
            }).catch(() => {});
        };

        // 删除模板
        const deletePreset = async (name) => {
            try {
                await api.settings.deletePreset(name);
                ElementPlus.ElMessage.success($tt('settings.templateDeleted'));
                await loadPresets();
            } catch (e) {
                console.error('Failed to delete preset:', e);
                ElementPlus.ElMessage.error($tt('settings.saveFailed'));
            }
        };

        // 实时资源监控
        const monitorData = reactive({
            cpu_percent: 0,
            memory: {
                total_gb: 0,
                used_gb: 0,
                percent: 0
            },
            gpu: null
        });
        let monitorInterval = null;

        const cacheStatus = reactive({
            cache: null,
            mlx: {},
        });
        const cacheLoading = ref(false);
        const cacheError = ref('');

        const refreshCacheStatus = async () => {
            cacheLoading.value = true;
            cacheError.value = '';
            try {
                const data = await api.system.getCacheStatus();
                cacheStatus.cache = data.cache || null;
                cacheStatus.mlx = data.mlx || {};
            } catch (e) {
                cacheError.value = e.message || String(e);
            } finally {
                cacheLoading.value = false;
            }
        };

        const currentModelConfig = computed(() => {
            return modelRegistry.value[selectedModel.value] || null;
        });

        const settingsLorasForForm = computed(() => {
            const c = currentModelConfig.value;
            if (!c || !c.parameters || !c.parameters.lora_support) return null;
            return settingsCompatibleLoras.value;
        });

        const modelActionKeyList = computed(() => {
            const cfg = currentModelConfig.value;
            if (!cfg || !cfg.actions) return [];
            return Object.keys(cfg.actions).filter((k) => cfg.actions[k] != null);
        });

        /** Plan D：注册表动作 chip 走顶层 action.*；视频 create 与图像 create 分文案 */
        const actionTagLabel = (key) => {
            const cfg = currentModelConfig.value;
            const engine = cfg && cfg.engine ? String(cfg.engine) : '';
            if (key === 'animate') {
                return window.$tt('action.video.animate');
            }
            if (key === 'create' && engine === 'mlx-video') {
                return window.$tt('action.video.create');
            }
            const imageKeys = new Set(['create', 'rewrite', 'retouch', 'extend', 'upscale']);
            if (imageKeys.has(key)) {
                return window.$tt('action.image.' + key);
            }
            return window.$tt('settings.actionTags.' + key);
        };

        // 持久化标签页状态
        watch(activeTab, (newVal) => {
            if (SK.SETTINGS_TAB) localStorage.setItem(SK.SETTINGS_TAB, newVal);
            if (newVal === 'system') {
                refreshCacheStatus();
            }
        });

        // 同步 i18n 语言变化
        watch(() => settings.language, (newVal) => {
            if (typeof i18n !== 'undefined' && i18n.global.locale.value !== newVal) {
                i18n.global.locale.value = newVal;
                if (SK.LANG) localStorage.setItem(SK.LANG, newVal);
                document.documentElement.lang = newVal;
            }
        });

        watch(
            () => settings.theme,
            (t) => {
                if (typeof window.DQApplyTheme === 'function') {
                    window.DQApplyTheme(t || 'dark');
                }
            },
        );

        // 加载模型注册表
        const loadModelRegistry = async () => {
            try {
                const data = await api.settings.getModelRegistry();
                modelRegistry.value = data.models || {};

                // 设置默认选中模型
                if (!selectedModel.value || !modelRegistry.value[selectedModel.value]) {
                    const recommended = Object.entries(modelRegistry.value)
                        .find(([key, val]) => val.recommended);
                    if (recommended) {
                        selectedModel.value = recommended[0];
                    } else {
                        const first = Object.keys(modelRegistry.value)[0];
                        if (first) selectedModel.value = first;
                    }
                }

                onModelSelect();
            } catch (e) {
                console.error('Failed to load model registry:', e);
            }
        };

        const loadSettingsCompatibleLoras = async () => {
            const mk = selectedModel.value;
            if (!mk) {
                settingsCompatibleLoras.value = [];
                return;
            }
            try {
                const list = await api.settings.getCompatibleLoras(mk);
                settingsCompatibleLoras.value = list || [];
            } catch (e) {
                console.error('Failed to load compatible loras (settings):', e);
                settingsCompatibleLoras.value = [];
            }
        };

        // 模型选择变化：注册表 defaults → modelParams（与创作页 RegistryParamSchema 一致）
        const onModelSelect = () => {
            const config = currentModelConfig.value;
            if (!config || !config.parameters) return;
            const R = window.RegistryParamSchema;
            if (R) R.applyDefaults(config.parameters, modelParams);
            loadSettingsCompatibleLoras();
        };

        const onSettingsModelRestoreDefaults = () => {
            const config = currentModelConfig.value;
            const R = window.RegistryParamSchema;
            if (R && config && config.parameters) {
                R.applyDefaults(config.parameters, modelParams);
                ElementPlus.ElMessage.success($tt('studio.restoredDefaults'));
            }
        };

        // 保存模型配置（直接修改 models_registry.json）
        const saveModelConfig = async () => {
            const config = currentModelConfig.value;
            if (!config || !config.parameters) return;

            try {
                const params = {};
                for (const [key, spec] of Object.entries(config.parameters)) {
                    if (typeof spec !== 'object' || !Object.prototype.hasOwnProperty.call(spec, 'default')) {
                        continue;
                    }
                    if (spec.type === 'bool' && String(key).endsWith('_support')) continue;
                    if (!Object.prototype.hasOwnProperty.call(modelParams, key)) continue;
                    params[key] = modelParams[key];
                }

                const result = await api.settings.updateModelParameters(selectedModel.value, params);
                if (result.success) {
                    ElementPlus.ElMessage.success($tt('settings.modelConfigSaved'));
                    await loadModelRegistry();
                    const RS = window.RegistryStore;
                    if (RS && typeof RS.load === 'function') {
                        try {
                            await RS.load();
                        } catch (_) {}
                    }
                } else {
                    ElementPlus.ElMessage.error(result.error || $tt('settings.saveFailed'));
                }
            } catch (e) {
                console.error('Failed to save model config:', e);
                ElementPlus.ElMessage.error($tt('settings.saveFailed'));
            }
        };

        // 加载设置
        const loadSettings = async () => {
            try {
                const data = await api.settings.getSettings();
                Object.assign(settings, data);
                if (typeof i18n !== 'undefined' && data.language) {
                    i18n.global.locale.value = data.language;
                    settings.language = data.language;
                    if (SK.LANG) localStorage.setItem(SK.LANG, data.language);
                    document.documentElement.lang = data.language;
                }
                if (typeof window.DQApplyTheme === 'function') {
                    window.DQApplyTheme(settings.theme || 'dark');
                }
            } catch (e) {
                console.error('Failed to load settings:', e);
            }
        };

        // 保存设置
        const saveSettings = async () => {
            try {
                await api.settings.updateSettings({
                    language: settings.language,
                    theme: settings.theme,
                    output_format: settings.output_format,
                    mlx_memory_limit: settings.mlx_memory_limit,
                    model_cache_ttl_minutes: settings.model_cache_ttl_minutes,
                    queue_image_first: settings.queue_image_first,
                    auto_save_prompts: settings.auto_save_prompts,
                    default_model: settings.default_model,
                    civitai_token: settings.civitai_token || '',
                    huggingface_token: settings.huggingface_token || '',
                    nsfw_enabled: settings.nsfw_enabled,
                    custom_models_dir: settings.custom_models_dir || '',
                    custom_loras_dir: settings.custom_loras_dir || '',
                    custom_outputs_dir: settings.custom_outputs_dir || '',
                });
                if (typeof window.DQApplyTheme === 'function') {
                    window.DQApplyTheme(settings.theme || 'dark');
                }
                ElementPlus.ElMessage.success($tt('settings.saved'));
            } catch (e) {
                ElementPlus.ElMessage.error($tt('settings.saveFailed'));
            }
        };

        // 加载已安装模型
        const loadInstalledModels = async () => {
            try {
                const models = await api.settings.listModels();
                installedModels.value = models;
            } catch (e) {
                console.error('Failed to load models:', e);
            }
        };

        // 语言切换
        const handleLanguageChange = (lang) => {
            settings.language = lang;
            // watch 会自动同步 i18n 和 localStorage，无需刷新页面
        };
        
        // 获取进度条颜色
        const getProgressColor = (percent) => {
            if (percent < 50) return '#67c23a';
            if (percent < 80) return '#e6a23c';
            return '#f56c6c';
        };
        
        // 加载系统监控数据
        const loadMonitorData = async () => {
            try {
                const data = await api.settings.getSystemMonitor();
                Object.assign(monitorData, data);
            } catch (e) {
                console.error('Failed to load monitor data:', e);
            }
        };
        
        // 启动监控定时器
        const startMonitor = () => {
            loadMonitorData();
            monitorInterval = setInterval(loadMonitorData, 3000);
        };
        
        // 停止监控定时器
        const stopMonitor = () => {
            if (monitorInterval) {
                clearInterval(monitorInterval);
                monitorInterval = null;
            }
        };

        onMounted(() => {
            loadModelRegistry();
            loadSettings();
            loadInstalledModels();
            loadPresets();
            startMonitor();
            refreshCacheStatus();
        });
        
        onUnmounted(() => {
            stopMonitor();
        });

        return {
            $mn: window.$mn,
            activeTab,
            settings,
            modelRegistry,
            selectedModel,
            modelParams,
            installedModels,
            currentModelConfig,
            settingsLorasForForm,
            onSettingsModelRestoreDefaults,
            modelActionKeyList,
            actionTagLabel,
            systemInfo,
            monitorData,
            presets,
            presetDialogVisible,
            editingPresetName,
            presetForm,
            presetList,
            onModelSelect,
            saveModelConfig,
            saveSettings,
            handleLanguageChange,
            getProgressColor,
            cacheStatus,
            cacheLoading,
            cacheError,
            refreshCacheStatus,
            openPresetDialog,
            savePreset,
            confirmDeletePreset,
            deletePreset,
            presetAppliesSummary,
            presetMediaLabel
        };
    }
};