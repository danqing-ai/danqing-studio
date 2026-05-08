/**
 * 创作页面组件 - 简洁版
 * 对普通用户友好，只保留核心功能
 */

const ImageCreatePage = {
    template: `
        <div class="create-page">
            <el-row :gutter="24">
                <!-- 左侧面板：创作区 -->
                <el-col :xs="24" :md="16" :lg="14">
                    <div class="creation-panel">
                        
                        <!-- Plan §2.1：一级 Tab（文生图 / 参考原图 / 按描述改图 / 局部修饰 / 扩展 / 放大） -->
                        <div class="mode-segment" style="margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 4px;">
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: imageWorkTab === 'create' }"
                                    @click="setImageWorkMode('create')"
                                >
                                    <el-icon><magic-stick /></el-icon>
                                    <span>{{ $t('action.image.create') }}</span>
                                </div>
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: imageWorkTab === 'rewrite_reference' }"
                                    @click="setImageWorkMode('rewrite_reference')"
                                >
                                    <el-icon><copy-document /></el-icon>
                                    <span>{{ $t('create.rewriteDriveReference') }}</span>
                                </div>
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: imageWorkTab === 'rewrite_instruct' }"
                                    @click="setImageWorkMode('rewrite_instruct')"
                                >
                                    <el-icon><edit-pen /></el-icon>
                                    <span>{{ $t('create.rewriteDriveInstruct') }}</span>
                                </div>
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: imageWorkTab === 'retouch' }"
                                    @click="setImageWorkMode('retouch')"
                                >
                                    <el-icon><brush /></el-icon>
                                    <span>{{ $t('action.image.retouch') }}</span>
                                </div>
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: imageWorkTab === 'extend' }"
                                    @click="setImageWorkMode('extend')"
                                >
                                    <el-icon><expand /></el-icon>
                                    <span>{{ $t('action.image.extend') }}</span>
                                </div>
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: imageWorkTab === 'upscale' }"
                                    @click="setImageWorkMode('upscale')"
                                >
                                    <el-icon><zoom-in /></el-icon>
                                    <span>{{ $t('action.image.upscale') }}</span>
                                </div>
                        </div>
                        <div v-if="editingSubModeDesc" style="margin-bottom: 16px; font-size: 12px; color: var(--text-muted); line-height: 1.5;">
                            {{ editingSubModeDesc }}
                        </div>

                        <!-- 模型选择：单层下拉，推荐项靠前 -->
                        <div class="card" style="margin-bottom: 16px;">
                            <div class="card-title">
                                <el-icon><cpu /></el-icon>
                                {{ $t('create.modelSelectTitle') }}
                            </div>
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <el-select
                                    v-model="selectedModelVersion"
                                    style="flex: 1;"
                                    size="large"
                                    filterable
                                    @change="onModelVersionChange"
                                    :placeholder="$t('studio.selectModel')"
                                >
                                    <el-option
                                        v-for="item in filteredModelPickerVersions"
                                        :key="item.modelKey + '|' + item.versionKey"
                                        :label="item.name"
                                        :value="item.modelKey + '|' + item.versionKey"
                                        :disabled="!item.ready"
                                    >
                                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                            <span :style="!item.ready ? 'opacity: 0.5;' : ''">{{ item.name }}</span>
                                            <el-tag v-if="item.recommended" size="small" type="success">{{ $t('studio.recommended') }}</el-tag>
                                            <el-tag v-if="item.status === 'ready'" size="small" type="success">{{ $t('studio.ready') }}</el-tag>
                                            <el-tag v-else-if="item.status === 'incomplete'" size="small" type="danger">{{ $t('studio.incomplete') }}</el-tag>
                                            <el-tag v-else size="small" type="warning">{{ $t('studio.notDownloaded') }}</el-tag>
                                            <span v-if="item.size" style="color: var(--text-muted); font-size: 12px; margin-left: auto;">
                                                {{ item.size }}
                                            </span>
                                        </div>
                                    </el-option>
                                </el-select>
                                <el-button
                                    circle
                                    size="large"
                                    @click="goToSettings"
                                    :title="$t('studio.modelSettings')"
                                >
                                    <el-icon><setting /></el-icon>
                                </el-button>
                            </div>
                            <el-alert
                                v-if="selectedModelNotReady"
                                :title="$t('studio.modelNotReady', { name: currentModelDisplayName })"
                                type="warning"
                                :closable="false"
                                style="margin-top: 12px;"
                            >
                                <template #default>
                                    <span>{{ $t('studio.notDownloadedMsg') }}</span>
                                    <el-button size="small" type="primary" @click="goToDownload" style="margin-left: 12px;">
                                        {{ $t('studio.goDownload') }}
                                    </el-button>
                                </template>
                            </el-alert>
                        </div>
                        
                        <!-- 提示词（精修放大不需要） -->
                        <div v-if="editMode !== 'image_upscale'" class="card" style="margin-bottom: 16px;">
                            <div class="card-title">
                                <el-icon><edit-pen /></el-icon>
                                {{ $t('studio.prompt') }}
                            </div>
                            
                            <!-- 预设快速选择 -->
                            <el-row :gutter="8" style="margin-bottom: 16px;">
                                <el-col :span="18">
                                    <el-select 
                                        v-model="selectedPreset" 
                                        :placeholder="$t('create.preset')" 
                                        style="width: 100%"
                                        clearable
                                    >
                                        <el-option
                                            v-for="(preset, name) in filteredPresets"
                                            :key="name"
                                            :label="presetSelectLabel(name, preset)"
                                            :value="name"
                                        />
                                    </el-select>
                                </el-col>
                                <el-col :span="6">
                                    <el-button @click="loadPreset" style="width: 100%">
                                        {{ $t('create.loadPreset') }}
                                    </el-button>
                                </el-col>
                            </el-row>
                            
                            <el-input
                                v-model="params.prompt"
                                type="textarea"
                                :rows="5"
                                :placeholder="$t('create.promptPlaceholder')"
                                resize="none"
                                @keydown.meta.enter.prevent="startGeneration"
                                @keydown.ctrl.enter.prevent="startGeneration"
                            />
                            
                            <!-- 负面提示词（仅支持负面提示词的模型显示） -->
                            <el-collapse v-if="currentModelConfig?.parameters?.negative_prompt_support" style="margin-top: 12px; border: none;">
                                <el-collapse-item :title="$t('studio.negativePrompt')" name="negative">
                                    <el-input
                                        v-model="params.negative_prompt"
                                        type="textarea"
                                        :rows="2"
                                        :placeholder="$t('create.negativePlaceholder')"
                                    />
                                </el-collapse-item>
                            </el-collapse>
                            <div v-if="editMode === 'image_editing' && editingSubMode === 'outpainting'" style="margin-top: 16px; padding-top: 12px; border-top: 1px solid var(--border-color);">
                                <div style="font-size: 13px; font-weight: 500; margin-bottom: 8px;">{{ $t('create.extendPanelTitle') }}</div>
                                <el-form label-position="top" size="small">
                                    <el-form-item :label="$t('create.extendDirections')">
                                        <el-checkbox-group v-model="params.extend_directions">
                                            <el-checkbox label="top">{{ $t('create.extendTop') }}</el-checkbox>
                                            <el-checkbox label="bottom">{{ $t('create.extendBottom') }}</el-checkbox>
                                            <el-checkbox label="left">{{ $t('create.extendLeft') }}</el-checkbox>
                                            <el-checkbox label="right">{{ $t('create.extendRight') }}</el-checkbox>
                                        </el-checkbox-group>
                                    </el-form-item>
                                    <el-form-item :label="$t('create.extendPixels')">
                                        <el-input-number v-model="params.extend_pixels" :min="64" :max="2048" :step="64" style="width: 100%;" />
                                    </el-form-item>
                                </el-form>
                            </div>
                        </div>
                        
                        <!-- 精修放大参数（plan §6.3 /image/upscales） -->
                        <div v-if="editMode === 'image_upscale'" class="card" style="margin-bottom: 16px;">
                            <div class="card-title">
                                <el-icon><zoom-in /></el-icon>
                                {{ $t('action.image.upscale') }}
                            </div>
                            <el-form label-position="top" size="small">
                                <el-form-item :label="$t('create.upscaleScale')">
                                    <el-select v-model="params.upscale_scale" style="width: 100%;">
                                        <el-option :label="'2×'" :value="2" />
                                        <el-option :label="'4×'" :value="4" />
                                    </el-select>
                                </el-form-item>
                                <el-form-item :label="$t('create.upscaleDenoise')">
                                    <div class="param-control-row">
                                        <div class="param-slider">
                                            <el-slider v-model="params.upscale_denoise" :min="0" :max="1" :step="0.05" />
                                        </div>
                                        <el-input-number v-model="params.upscale_denoise" :min="0" :max="1" :step="0.05" class="param-input-number" />
                                    </div>
                                </el-form-item>
                                <el-form-item :label="$t('create.upscaleTile')">
                                    <el-input-number v-model="params.upscale_tile" :min="256" :max="4096" :step="128" style="width: 100%;" />
                                </el-form-item>
                            </el-form>
                        </div>
                        
                        <!-- 高级参数（可折叠） -->
                        <div v-if="editMode !== 'image_upscale'" class="card" style="margin-bottom: 16px;">
                            <el-collapse v-model="advancedParamsOpen" style="border: none;">
                                <el-collapse-item name="advanced">
                                    <template #title>
                                        <div style="display: flex; align-items: center; gap: 8px; font-weight: 500;">
                                            <el-icon><setting /></el-icon>
                                            <span>{{ $t('studio.advancedParams') }}</span>
                                            <el-tag v-if="hasCustomParams" size="small" type="warning">{{ $t('studio.hasCustom') }}</el-tag>
                                        </div>
                                    </template>
                                    
                                    <registry-params-form
                                        :model-config="currentModelConfig"
                                        :params="params"
                                        :param-visibility="advancedParamVisibility"
                                        :loras="compatibleLoras"
                                        :controlnets="compatibleControlNets"
                                        :control-image-src="controlImageSrc"
                                        :control-recent-gallery="recentImages"
                                        @control-asset-pick="onControlAssetPick"
                                        @remove-control-image="removeControlImage"
                                        @restore-defaults="resetToDefaults"
                                    />
                                </el-collapse-item>
                            </el-collapse>
                        </div>
                        
                        <!-- 主操作（plan §2.3：主按钮 + 排队提示） -->
                        <div class="card" style="margin-bottom: 16px;">
                            <el-button
                                type="primary"
                                size="large"
                                style="width: 100%; min-width: 200px; height: 50px; font-size: 16px;"
                                :disabled="submitDisabled"
                                @click="startGeneration"
                            >
                                <el-icon size="20"><magic-stick /></el-icon>
                                <span style="margin-left: 8px;">{{ primaryCtaLabel }}</span>
                            </el-button>
                            <div style="margin-top: 8px; font-size: 11px; color: var(--text-muted);">
                                {{ $t('studio.sendShortcutHint') }}
                            </div>
                            
                            <!-- 进度显示 -->
                            <div v-if="currentTask" style="margin-top: 16px;">
                                <el-progress 
                                    :percentage="Math.round(currentTask.progress * 100)" 
                                    :status="currentTask.status === 'failed' ? 'exception' : ''"
                                />
                                <div style="margin-top: 8px; text-align: center; color: var(--text-muted); font-size: 13px;">
                                    <el-tag :type="getStatusType(currentTask.status)" size="small">
                                        {{ getStatusText(currentTask.status) }}
                                    </el-tag>
                                </div>
                            </div>
                        </div>
                        
                        <!-- 日志 -->
                        <div class="card">
                            <div class="card-title" style="justify-content: space-between;">
                                <span>
                                    <el-icon><document /></el-icon>
                                    {{ $t('studio.logs') }}
                                </span>
                                <el-button size="small" text @click="clearLogs">
                                    <el-icon><delete /></el-icon>
                                </el-button>
                            </div>
                            
                            <div class="log-container" ref="logContainer" style="max-height: 200px;">
                                <div v-if="logs.length === 0" style="text-align: center; color: var(--text-muted); padding: 20px;">
                                    {{ $t('studio.logsEmpty') }}
                                </div>
                                <div v-for="(log, index) in logs" :key="index" class="log-line">
                                    <span class="log-timestamp">{{ log.time }}</span>
                                    <span :class="'log-' + log.level">{{ log.message }}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </el-col>
                
                <!-- 右侧面板 -->
                <el-col :xs="24" :md="8" :lg="10">
                    <div class="preview-panel">
                        
                        <!-- 改图 / 局部修饰 / 扩展：编辑器 -->
                        <div v-if="editMode === 'image_editing'" class="card" style="margin-bottom: 16px;">
                            <div class="source-input-card-head">
                                <div class="card-title" style="margin-bottom: 0;">
                                    <span>
                                        <el-icon><picture-filled /></el-icon>
                                        {{ $t('create.imageInput') }}
                                    </span>
                                </div>
                                <asset-picker
                                    accept-kind="image"
                                    :recent-gallery="recentImages"
                                    @pick="onEditAssetPick"
                                />
                            </div>
                            <image-editor
                                ref="imageEditorRef"
                                :src="editImageSrc"
                                :recent-gallery="recentImages"
                                mode="inpainting"
                                @pick-edit-source="onEditAssetPick"
                            />
                        </div>
                        <!-- 精修放大：仅需源图 -->
                        <div v-else-if="editMode === 'image_upscale'" class="card" style="margin-bottom: 16px;">
                            <div class="source-input-card-head">
                                <div class="card-title" style="margin-bottom: 0;">
                                    <span>
                                        <el-icon><picture-filled /></el-icon>
                                        {{ $t('create.imageInput') }}
                                    </span>
                                </div>
                                <asset-picker
                                    accept-kind="image"
                                    :recent-gallery="recentImages"
                                    @pick="onEditAssetPick"
                                />
                            </div>
                            <div v-if="editImageSrc" class="image-preview" style="aspect-ratio: 1; margin-top: 8px;">
                                <img :src="editImageSrc" alt="upscale source" style="width: 100%; height: 100%; object-fit: contain;" />
                            </div>
                            <el-empty v-else :description="$t('studio.uploadEditImage')" />
                        </div>
                        
                        <!-- 当前生成预览 -->
                        <div class="card" style="margin-bottom: 16px;">
                            <div class="card-title">
                                <el-icon><picture-filled /></el-icon>
                                {{ $t('studio.currentPreview') }}
                            </div>
                            
                            <div v-if="previewImage" class="image-preview" style="aspect-ratio: 1;">
                                <img :src="previewImage" alt="Preview" style="width: 100%; height: 100%; object-fit: contain;" />
                            </div>
                            <el-empty v-else :description="$t('studio.noPreview')" />
                        </div>
                        
                        <!-- 最近生成 -->
                        <div class="card">
                            <div class="card-title" style="justify-content: space-between;">
                                <span>
                                    <el-icon><clock /></el-icon>
                                    {{ $t('studio.recent') }}
                                </span>
                                <el-button size="small" text @click="loadRecentImages">
                                    <el-icon><refresh /></el-icon>
                                </el-button>
                            </div>
                            
                            <el-empty v-if="recentImages.length === 0" :description="$t('gallery.empty')" />
                            
                            <el-row v-else :gutter="8">
                                <el-col
                                    v-for="image in recentImages"
                                    :key="image.path"
                                    :span="12"
                                    style="margin-bottom: 8px;"
                                >
                                    <div class="gallery-card">
                                        <div class="gallery-image-wrapper" style="aspect-ratio: 1;" @click="showPreview(image)">
                                            <img :src="getImageUrl(image)" :alt="image.name" loading="lazy" />
                                        </div>
                                        <div class="recent-actions">
                                            <el-button class="action-btn rewrite-btn" size="small" @click.stop="quickFromGallery(image, 'rewrite')">
                                                <el-icon><brush /></el-icon>
                                                <span>{{ $t('studio.quickRewrite') }}</span>
                                            </el-button>
                                            <el-button class="action-btn upscale-btn" size="small" @click.stop="quickFromGallery(image, 'upscale')">
                                                <el-icon><zoom-in /></el-icon>
                                                <span>{{ $t('studio.quickUpscale') }}</span>
                                            </el-button>
                                        </div>
                                    </div>
                                </el-col>
                            </el-row>
                        </div>
                    </div>
                </el-col>
            </el-row>
            
            <!-- 图片预览对话框 -->
            <el-dialog v-model="previewVisible" :title="selectedImage?.name" width="70%" center>
                <div v-if="selectedImage" style="text-align: center;">
                    <img :src="getImageUrl(selectedImage)" style="max-width: 100%; border-radius: 8px;" />
                </div>
            </el-dialog>
        </div>
    `,
    
    setup() {
        const { ref, reactive, onMounted, inject, computed, nextTick, watch } = Vue;
        const systemInfo = inject('systemInfo');
        const RA = window.RegistryActions || {};

        // 参数（含高级参数）
        const params = reactive({
            prompt: '',
            negative_prompt: '',
            model: '',
            version: '',
            steps: 4,
            guidance: 3.5,
            width: 1024,
            height: 1024,
            lora: '',
            lora_scale: 0.8,
            seed: '',
            strength: 0.4,
            img2img: false,
            controlnet: '',
            controlnet_strength: 0.8,
            scheduler: 'flow_match_euler_discrete',
            upscale_scale: 2,
            upscale_denoise: 0.3,
            upscale_tile: 1024,
            extend_directions: ['right'],
            extend_pixels: 256,
        });
        
        // 选中的模型+版本组合（格式: "modelKey|versionKey"）
        const selectedModelVersion = ref('');
        
        // 状态
        const generating = ref(false);
        const currentTask = ref(null);
        const logs = ref([]);
        const previewImage = ref('');
        const recentImages = ref([]);
        const advancedParamsOpen = ref([]);
        const compatibleLoras = ref([]);
        const compatibleControlNets = ref([]);
        const controlImageSrc = ref('');
        const controlImagePath = ref('');
        
        // 模式：一级 Tab 与引擎子模式（rewrite 拆成 reference / instruct 两档）
        const editMode = ref('image_generation'); // image_generation | image_editing | image_upscale
        const imageWorkTab = ref('create'); // create | rewrite_reference | rewrite_instruct | retouch | extend | upscale
        const editingSubMode = ref('inpainting'); // inpainting | text_editing | outpainting
        /** 与 API rewrite_mode 对齐；由 imageWorkTab 驱动 */
        const rewriteDriveMode = ref('reference');

        const setImageWorkMode = (mode) => {
            if (mode === 'create') {
                editMode.value = 'image_generation';
                imageWorkTab.value = 'create';
            } else if (mode === 'upscale') {
                editMode.value = 'image_upscale';
                imageWorkTab.value = 'upscale';
            } else if (mode === 'rewrite' || mode === 'rewrite_reference') {
                editMode.value = 'image_editing';
                imageWorkTab.value = 'rewrite_reference';
                editingSubMode.value = 'text_editing';
                rewriteDriveMode.value = 'reference';
            } else if (mode === 'rewrite_instruct') {
                editMode.value = 'image_editing';
                imageWorkTab.value = 'rewrite_instruct';
                editingSubMode.value = 'text_editing';
                rewriteDriveMode.value = 'instruct';
            } else if (mode === 'retouch') {
                editMode.value = 'image_editing';
                imageWorkTab.value = 'retouch';
                editingSubMode.value = 'inpainting';
            } else if (mode === 'extend') {
                editMode.value = 'image_editing';
                imageWorkTab.value = 'extend';
                editingSubMode.value = 'outpainting';
            }
        };
        
        // 局部重绘：图片编辑器
        const editImageSrc = ref('');
        const editImagePath = ref('');
        const imageEditorRef = ref(null);
        
        // 预设
        const presets = ref({});
        const selectedPreset = ref('');
        
        const presetActionFilter = computed(() => {
            if (editMode.value === 'image_upscale') {
                return new Set(['upscale']);
            }
            if (editMode.value === 'image_generation') {
                return new Set(['create']);
            }
            if (editingSubMode.value === 'inpainting') {
                return new Set(['retouch', 'rewrite']);
            }
            if (editingSubMode.value === 'outpainting') {
                return new Set(['extend', 'retouch']);
            }
            return new Set(['rewrite']);
        });

        const filteredPresets = computed(() => {
            const want = presetActionFilter.value;

            function planPresetShapeOk(preset) {
                return (
                    Array.isArray(preset.applies_to) &&
                    preset.applies_to.length > 0 &&
                    (preset.media_scope === 'image' || preset.media_scope === 'video')
                );
            }

            function matchesMediaScope(preset) {
                return preset.media_scope === 'image';
            }

            function matches(preset) {
                if (!planPresetShapeOk(preset)) return false;
                if (!matchesMediaScope(preset)) return false;
                return preset.applies_to.some((k) => want.has(k));
            }

            const entries = Object.entries(presets.value)
                .filter(([, preset]) => matches(preset))
                .sort((a, b) => {
                    const aCreate = a[1].applies_to.includes('create');
                    const bCreate = b[1].applies_to.includes('create');
                    if (aCreate !== bCreate) {
                        return aCreate ? -1 : 1;
                    }
                    return a[0].localeCompare(b[0], 'zh');
                });
            const result = {};
            for (const [name, preset] of entries) {
                result[name] = preset;
            }
            return result;
        });
        
        // 模型注册表
        const modelRegistry = ref({});
        
        // 模型就绪状态
        const modelsStatus = ref({});
        const modelsDetailedStatus = ref({});
        
        // 所有模型版本（扁平化列表）
        const allVersions = computed(() => {
            const result = [];
            for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
                if (!RA.imageModelRow || !RA.imageModelRow(config)) {
                    continue;
                }
                const actions = { ...(config.actions || {}) };
                const engine = config.engine || '';
                const versions = config.versions || { default: { name: '默认版本', size: '', default: true } };
                const detailed = modelsDetailedStatus.value[modelKey] || {};
                const versionStatuses = detailed.versions || {};
                
                for (const [versionKey, versionConfig] of Object.entries(versions)) {
                    const status = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
                    const size = versionConfig.size || '';
                    
                    result.push({
                        modelKey,
                        versionKey,
                        name:
                            typeof window.$mvn === 'function'
                                ? window.$mvn(modelKey, config, versionConfig)
                                : String(modelKey),
                        size,
                        status: status.status,
                        ready: status.ready,
                        recommended: config.recommended && versionConfig.default,
                        actions,
                        engine
                    });
                }
            }
            return result;
        });
        
        // 推荐版本
        const recommendedVersions = computed(() => {
            return allVersions.value.filter(v => v.recommended);
        });
        
        // 按模式过滤模型
        const filteredAllVersions = computed(() => {
            if (editMode.value === 'image_editing') {
                return allVersions.value.filter((v) => {
                    const acts = v.actions || {};
                    if (imageWorkTab.value === 'rewrite_instruct') {
                        return RA.hasAction(acts, 'rewrite') && v.modelKey === 'flux1-kontext';
                    }
                    return RA.imageEditingMatches ? RA.imageEditingMatches(acts, editingSubMode.value) : false;
                });
            }
            if (editMode.value === 'image_upscale') {
                return allVersions.value.filter((v) => {
                    const acts = v.actions || {};
                    return RA.imageSupportsUpscale ? RA.imageSupportsUpscale(acts) : false;
                });
            }
            return allVersions.value.filter((v) => {
                const acts = v.actions || {};
                return RA.imageSupportsCreate ? RA.imageSupportsCreate(acts) : false;
            });
        });
        
        const filteredRecommendedVersions = computed(() => {
            return filteredAllVersions.value.filter(v => v.recommended);
        });

        /** 模型下拉：单层列表，推荐版本排在前面 */
        const filteredModelPickerVersions = computed(() => {
            const rows = [...filteredAllVersions.value];
            rows.sort((a, b) => {
                const ar = a.recommended ? 1 : 0;
                const br = b.recommended ? 1 : 0;
                if (ar !== br) return br - ar;
                const an = a.name || '';
                const bn = b.name || '';
                try {
                    return an.localeCompare(bn, 'zh');
                } catch {
                    return an < bn ? -1 : an > bn ? 1 : 0;
                }
            });
            return rows;
        });

        // 当前模型配置
        const currentModelConfig = computed(() => modelRegistry.value[params.model] || null);

        const currentModelDisplayName = computed(() => {
            const c = currentModelConfig.value;
            if (typeof window.$mn === 'function' && c) {
                return window.$mn(c, params.model);
            }
            return params.model || '';
        });
        
        // 当前选中版本是否就绪
        const selectedModelNotReady = computed(() => {
            if (!params.model || !params.version) return false;
            const detailed = modelsDetailedStatus.value[params.model];
            if (!detailed || !detailed.versions) return true;
            const versionStatus = detailed.versions[params.version];
            return !versionStatus || !versionStatus.ready;
        });

        // 编辑子类型描述
        const editingSubModeDesc = computed(() => {
            if (editMode.value === 'image_upscale') {
                return $tt('action.image.upscaleDesc');
            }
            if (editMode.value === 'image_generation') {
                return $tt('action.image.createDesc');
            }
            if (editMode.value === 'image_editing' && editingSubMode.value === 'text_editing') {
                return rewriteDriveMode.value === 'instruct'
                    ? $tt('create.rewriteDriveInstructDesc')
                    : $tt('create.rewriteDriveReferenceDesc');
            }
            const descMap = {
                inpainting: $tt('action.image.retouchDesc'),
                outpainting: $tt('action.image.extendDesc'),
            };
            return descMap[editingSubMode.value] || '';
        });

        const submitDisabled = computed(() => {
            if (selectedModelNotReady.value) return true;
            if (editMode.value === 'image_upscale') {
                return !editImageSrc.value;
            }
            return !String(params.prompt || '').trim();
        });

        const primaryCtaLabel = computed(() => {
            if (editMode.value === 'image_generation') {
                return $tt('studio.generate');
            }
            if (editMode.value === 'image_upscale') {
                return $tt('action.image.upscale');
            }
            if (editingSubMode.value === 'text_editing') {
                return rewriteDriveMode.value === 'instruct'
                    ? $tt('create.rewriteDriveInstruct')
                    : $tt('create.rewriteDriveReference');
            }
            if (editingSubMode.value === 'inpainting') {
                return $tt('action.image.retouch');
            }
            if (editingSubMode.value === 'outpainting') {
                return $tt('action.image.extend');
            }
            return $tt('studio.generate');
        });

        // 加载模型注册表和状态
        const loadModelRegistry = async () => {
            try {
                const RS = window.RegistryStore;
                const regPromise =
                    RS && RS.load
                        ? RS.load()
                        : api.settings.getModelRegistry().then((r) => ({ models: r.models }));
                const [registryData, statusData, detailedStatusData] = await Promise.all([
                    regPromise,
                    api.settings.getModelsStatus(),
                    api.settings.getModelsDetailedStatus(),
                ]);

                modelRegistry.value = registryData.models || {};
                modelsStatus.value = statusData || {};
                modelsDetailedStatus.value = detailedStatusData || {};
                
                // 设置默认模型+版本（优先选择已就绪的推荐版本的默认版本）
                if (!selectedModelVersion.value) {
                    let found = false;
                    for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
                        if (config.recommended) {
                            const detailed = detailedStatusData[modelKey] || {};
                            const versions = detailed.versions || {};
                            const defaultVersionKey = Object.keys(config.versions || {}).find(k => config.versions[k].default) || Object.keys(config.versions || {})[0];
                            
                            if (defaultVersionKey && versions[defaultVersionKey]?.ready) {
                                params.model = modelKey;
                                params.version = defaultVersionKey;
                                selectedModelVersion.value = modelKey + '|' + defaultVersionKey;
                                found = true;
                                break;
                            }
                        }
                    }
                    
                    if (!found) {
                        for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
                            const detailed = detailedStatusData[modelKey] || {};
                            const versions = detailed.versions || {};
                            for (const versionKey of Object.keys(config.versions || {})) {
                                if (versions[versionKey]?.ready) {
                                    params.model = modelKey;
                                    params.version = versionKey;
                                    selectedModelVersion.value = modelKey + '|' + versionKey;
                                    found = true;
                                    break;
                                }
                            }
                            if (found) break;
                        }
                    }
                    
                    if (!found) {
                        const firstModel = Object.keys(modelRegistry.value)[0];
                        if (firstModel) {
                            const firstVersion = Object.keys(modelRegistry.value[firstModel].versions || {})[0] || 'default';
                            params.model = firstModel;
                            params.version = firstVersion;
                            selectedModelVersion.value = firstModel + '|' + firstVersion;
                        }
                    }
                }
                
                loadModelDefaults();
            } catch (e) {
                console.error('Failed to load model registry:', e);
            }
        };
        
        // 加载模型的默认配置（注册表 schema 驱动）
        const loadModelDefaults = () => {
            const config = currentModelConfig.value;
            if (!config || !config.parameters) return;
            const R = window.RegistryParamSchema;
            if (R) {
                R.applyDefaults(config.parameters, params);
            }
            controlImageSrc.value = '';
            controlImagePath.value = '';
            loadCompatibleLoras();
            loadCompatibleControlNets();
        };
        
        // 加载与当前模型匹配的 LoRA
        const loadCompatibleLoras = async () => {
            if (!params.model) return;
            try {
                const loras = await api.settings.getCompatibleLoras(params.model);
                compatibleLoras.value = loras || [];
            } catch (e) {
                console.error('Failed to load compatible loras:', e);
                compatibleLoras.value = [];
            }
        };
        
        // 加载与当前模型匹配的 ControlNet
        const loadCompatibleControlNets = async () => {
            if (!params.model) return;
            try {
                const nets = await api.settings.getCompatibleControlNets(params.model);
                compatibleControlNets.value = nets || [];
                // 如果当前选的 ControlNet 不在返回列表中（不匹配或已删除），清空
                if (params.controlnet && !nets.some(n => n.key === params.controlnet)) {
                    params.controlnet = '';
                    controlImageSrc.value = '';
                    controlImagePath.value = '';
                }
            } catch (e) {
                console.error('Failed to load compatible controlnets:', e);
                compatibleControlNets.value = [];
            }
        };
        
        // 重置为默认配置（从注册表重新加载）
        const resetToDefaults = () => {
            loadModelDefaults();
            ElementPlus.ElMessage.success($tt('studio.restoredDefaults'));
        };
        
        const hasCustomParams = computed(() => {
            const config = currentModelConfig.value;
            if (!config || !config.parameters) return false;
            const R = window.RegistryParamSchema;
            if (R) return R.hasDeviation(config.parameters, params);
            return false;
        });

        const advancedParamVisibility = computed(() => ({
            width: true,
            strength: editMode.value !== 'image_upscale' && editMode.value === 'image_editing',
        }));

        const presetSelectLabel = (name, preset) => {
            const a = preset.applies_to;
            const hasC = a.includes('create');
            const hasEdit = a.some((x) => ['rewrite', 'retouch', 'extend'].includes(x));
            let tag = '';
            if (hasC && !hasEdit) tag = $tt('create.presetTagT2I');
            else if (hasEdit && !hasC) tag = $tt('create.presetTagI2I');
            const display = window.$pn ? window.$pn(preset, name) : name;
            return tag ? `${tag} ${display}` : display;
        };
        
        // 加载预设
        const loadPresets = async () => {
            try {
                const data = await api.settings.getPresets();
                presets.value = data || {};
            } catch (e) {
                console.error('Failed to load presets:', e);
                presets.value = {};
            }
        };
        
        // 加载预设到参数（追加到新行）
        const loadPreset = () => {
            if (!selectedPreset.value || !presets.value[selectedPreset.value]) return;
            
            const preset = presets.value[selectedPreset.value];
            
            if (preset.positive) {
                params.prompt = params.prompt
                    ? params.prompt + '\n风格增强: ' + preset.positive
                    : preset.positive;
            }
            if (preset.negative) {
                params.negative_prompt = params.negative_prompt
                    ? params.negative_prompt + '\n' + preset.negative
                    : preset.negative;
            }
            
            // 仅面向编辑类动作、不含文生图 create 的预设：在文生图 Tab 上提示切换到改图并选源图
            const app = preset.applies_to;
            const needsEditSource =
                !app.includes('create') &&
                app.some((x) => ['rewrite', 'retouch', 'extend'].includes(x));
            if (needsEditSource && editMode.value === 'image_generation') {
                ElementPlus.ElMessage.warning($tt('create.presetNeedsEditTab'));
            }
        };
        
        // 添加日志
        const addLog = (message, level = 'info') => {
            const now = new Date();
            const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
            logs.value.push({ time, message, level });
            
            if (logs.value.length > 500) {
                logs.value = logs.value.slice(-500);
            }
            
            nextTick(() => {
                const container = document.querySelector('.log-container');
                if (container) {
                    container.scrollTop = container.scrollHeight;
                }
            });
        };
        
        // 清空日志
        const clearLogs = () => {
            logs.value = [];
        };
        
        // 截断文本
        const truncate = (text, length) => {
            if (!text) return '';
            return text.length > length ? text.substring(0, length) + '...' : text;
        };
        
        // 开始生成
        const startGeneration = async () => {
            if (editMode.value === 'image_upscale') {
                if (!editImageSrc.value) {
                    ElementPlus.ElMessage.warning($tt('studio.uploadEditImage'));
                    return;
                }
            } else if (!params.prompt) {
                ElementPlus.ElMessage.warning($tt('studio.enterPrompt'));
                return;
            }

            // ControlNet 必须上传控制图
            if (editMode.value !== 'image_upscale' && params.controlnet && !controlImageSrc.value) {
                ElementPlus.ElMessage.warning($tt('studio.needControlImage'));
                return;
            }
            
            const detailed = modelsDetailedStatus.value[params.model];
            const versionStatus = detailed?.versions?.[params.version];
            if (!versionStatus?.ready) {
                ElementPlus.ElMessage.warning(
                    $tt('studio.modelNotReadyDesc', { name: currentModelDisplayName.value, version: params.version }),
                );
                return;
            }

            const verCfg = (currentModelConfig.value && currentModelConfig.value.versions && currentModelConfig.value.versions[params.version]) || null;
            const sizeHuman = verCfg && verCfg.size ? String(verCfg.size) : '';
            if (window.DQMemoryHint && typeof window.DQMemoryHint.warnIfRisky === 'function') {
                window.DQMemoryHint.warnIfRisky({ systemInfo, versionSizeHuman: sizeHuman, $tt });
            }

            const modelStr = params.version ? `${params.model}:${params.version}` : params.model;
            const adapters = params.lora ? [{ id: params.lora, weight: params.lora_scale || 0.8 }] : [];
            const _seedParsed =
                params.seed != null && params.seed !== '' ? parseInt(String(params.seed), 10) : null;
            const seedNum = Number.isFinite(_seedParsed) ? _seedParsed : null;
            const meta = {};
            if (params.scheduler) {
                meta.scheduler = params.scheduler;
            }

            const attachStream = (tid) => {
                currentTask.value = {
                    id: tid,
                    progress: 0,
                    status: 'queued',
                    params: {
                        model: modelStr,
                        prompt: editMode.value === 'image_upscale' ? '' : params.prompt,
                    },
                };
                api.gen.streamMediaTask(
                    tid,
                    (logData) => addLog(logData.message || '', logData.level || 'info'),
                    (statusData) => {
                        if (currentTask.value) {
                            currentTask.value.progress = statusData.progress ?? 0;
                            currentTask.value.status = statusData.status;
                        }
                    },
                    async (doneData) => {
                        if (doneData.status === 'completed') {
                            addLog($tt('studio.genComplete'), 'success');
                            const updated = await api.gen.getMediaTask(tid);
                            currentTask.value = updated;
                            const pid = updated.result && updated.result.primary_asset_id;
                            if (pid) {
                                previewImage.value = api.gallery.getImageUrl(`asset:${pid}`);
                                addLog($tt('studio.outputFile', { name: pid }), 'info');
                            }
                            loadRecentImages();
                        } else if (doneData.status === 'failed') {
                            const updated = await api.gen.getMediaTask(tid);
                            currentTask.value = updated;
                            addLog($tt('studio.genFailed', { msg: updated.error_message || '' }), 'error');
                        }
                    },
                    () => addLog($tt('studio.connectionLost'), 'warning')
                );
            };

            addLog($tt('studio.startingGen'), 'info');
            try {
                if (editMode.value === 'image_upscale') {
                    const ep = editImagePath.value;
                    let source_asset_id;
                    if (typeof ep === 'string' && ep.startsWith('asset:')) {
                        source_asset_id = ep.slice('asset:'.length);
                    } else {
                        const srcBlob = await api.gen.urlToBlob(editImageSrc.value);
                        source_asset_id = (
                            await api.gen.uploadAsset(
                                new File([srcBlob], 'upscale-src.png', { type: srcBlob.type || 'image/png' })
                            )
                        ).id;
                    }
                    const sc = Number(params.upscale_scale) === 4 ? 4 : 2;
                    const submitRes = await api.gen.createImageUpscale({
                        model: modelStr,
                        source_asset_id,
                        scale: sc,
                        denoise: Number(params.upscale_denoise) || 0.3,
                        tile_size: Number(params.upscale_tile) || 1024,
                        priority: 'normal',
                        metadata: {},
                    });
                    attachStream(submitRes.task.id);
                    return;
                }

                if (editMode.value === 'image_editing') {
                    if (!editImageSrc.value) {
                        ElementPlus.ElMessage.warning($tt('studio.uploadEditImage'));
                        return;
                    }
                    const maskBlob = imageEditorRef.value ? await imageEditorRef.value.getMaskBlob() : null;
                    if (!maskBlob && editingSubMode.value !== 'text_editing' && editingSubMode.value !== 'outpainting') {
                        ElementPlus.ElMessage.warning($tt('studio.drawMask'));
                        return;
                    }
                    const ep = editImagePath.value;
                    let source_asset_id;
                    if (typeof ep === 'string' && ep.startsWith('asset:')) {
                        source_asset_id = ep.slice('asset:'.length);
                    } else {
                        const srcBlob = await api.gen.urlToBlob(editImageSrc.value);
                        source_asset_id = (
                            await api.gen.uploadAsset(
                                new File([srcBlob], 'source.png', { type: srcBlob.type || 'image/png' })
                            )
                        ).id;
                    }
                    let mask_asset_id = null;
                    if (maskBlob) {
                        mask_asset_id = (
                            await api.gen.uploadAsset(new File([maskBlob], 'mask.png', { type: 'image/png' }))
                        ).id;
                    }

                    let operation = 'rewrite';
                    if (editingSubMode.value === 'inpainting') {
                        operation = 'retouch';
                    } else if (editingSubMode.value === 'outpainting') {
                        operation = 'extend';
                    } else if (editingSubMode.value === 'text_editing') {
                        operation = 'rewrite';
                    }

                    let extendSpec = undefined;
                    if (operation === 'extend') {
                        const dirs = Array.isArray(params.extend_directions)
                            ? params.extend_directions.filter((d) => ['top', 'bottom', 'left', 'right'].includes(d))
                            : [];
                        if (!dirs.length) {
                            ElementPlus.ElMessage.warning($tt('create.extendNeedDirection'));
                            return;
                        }
                        const px = Math.min(2048, Math.max(64, Number(params.extend_pixels) || 256));
                        extendSpec = { directions: dirs, pixels: px };
                    }

                    const editBody = {
                        model: modelStr,
                        operation,
                        source_asset_id,
                        mask_asset_id,
                        prompt: params.prompt,
                        negative_prompt: params.negative_prompt || '',
                        source_fidelity: Math.min(0.95, Math.max(0.05, 1 - (params.strength ?? 0.4))),
                        extend: extendSpec,
                        n: 1,
                        steps: params.steps,
                        seed: seedNum,
                        adapters,
                        metadata: { ...meta },
                        priority: 'normal',
                    };
                    if (operation === 'rewrite') {
                        editBody.rewrite_mode = rewriteDriveMode.value;
                    }
                    const submitRes = await api.gen.createImageEdit(editBody);
                    attachStream(submitRes.task.id);
                    return;
                }

                let control_asset_id = null;
                if (params.controlnet && controlImageSrc.value) {
                    const cp = controlImagePath.value;
                    if (typeof cp === 'string' && cp.startsWith('asset:')) {
                        control_asset_id = cp.slice('asset:'.length);
                    } else {
                        const cblob = await api.gen.urlToBlob(controlImageSrc.value);
                        control_asset_id = (
                            await api.gen.uploadAsset(
                                new File([cblob], 'control.png', { type: cblob.type || 'image/png' })
                            )
                        ).id;
                    }
                }

                let submitRes;
                if (params.controlnet && control_asset_id) {
                    submitRes = await api.gen.createImageGeneration({
                        model: modelStr,
                        prompt: params.prompt,
                        negative_prompt: params.negative_prompt || '',
                        size: `${params.width}x${params.height}`,
                        n: 1,
                        steps: params.steps,
                        guidance: params.guidance,
                        seed: seedNum,
                        adapters,
                        structural_guide: {
                            asset_id: control_asset_id,
                            type: 'canny',
                            weight: params.controlnet_strength ?? 1,
                        },
                        metadata: { ...meta, controlnet: params.controlnet },
                        priority: 'normal',
                    });
                } else {
                    submitRes = await api.gen.createImageGeneration({
                        model: modelStr,
                        prompt: params.prompt,
                        negative_prompt: params.negative_prompt || '',
                        size: `${params.width}x${params.height}`,
                        n: 1,
                        steps: params.steps,
                        guidance: params.guidance,
                        seed: seedNum,
                        adapters,
                        metadata: { ...meta },
                        priority: 'normal',
                    });
                }
                attachStream(submitRes.task.id);
            } catch (e) {
                addLog($tt('studio.error', { msg: e.message || String(e) }), 'error');
            }
        };
        
        // 加载最近图片
        const loadRecentImages = async () => {
            try {
                const images = await api.gallery.listImages(24, 0);
                recentImages.value = images.filter((v) => {
                    if (v.metadata && v.metadata.asset_kind === 'video') {
                        return false;
                    }
                    const ext = v.name?.split('.').pop()?.toLowerCase();
                    return !['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext || '');
                }).slice(0, 4);
            } catch (e) {
                console.error('Failed to load recent images:', e);
            }
        };
        
        // 获取图片URL
        const getImageUrl = (image) => {
            return api.gallery.getImageUrl(image.path);
        };
        
        // 图片预览
        const previewVisible = ref(false);
        const selectedImage = ref(null);
        
        const showPreview = (image) => {
            selectedImage.value = image;
            previewVisible.value = true;
        };

        const quickFromGallery = async (image, mode) => {
            editImagePath.value = image.path;
            editImageSrc.value = getImageUrl(image);
            if (mode === 'upscale') {
                setImageWorkMode('upscale');
            } else {
                setImageWorkMode('rewrite_reference');
            }
            await loadRecentImages();
        };
        
        // 局部重绘：编辑文件变化
        const onEditAssetPick = async ({ path, previewUrl }) => {
            editImagePath.value = path;
            editImageSrc.value = previewUrl;
            addLog($tt('create.imageLoaded', { name: path }), 'info');
            await loadRecentImages();
        };
        
        const onControlAssetPick = ({ path, previewUrl }) => {
            controlImageSrc.value = previewUrl;
            controlImagePath.value = path;
        };
        const removeControlImage = () => {
            controlImageSrc.value = '';
            controlImagePath.value = '';
        };
        
        // 跳转到设置页面
        const goToSettings = () => window.DQStudioNav.goSettings();
        const goToDownload = () => window.DQStudioNav.goModels();
        
        const TSU = window.DQTaskStatusUi;
        const getStatusType = (status) =>
            TSU && typeof TSU.tagType === 'function' ? TSU.tagType(status) : 'info';
        const getStatusText = (status) =>
            TSU && typeof TSU.statusText === 'function' ? TSU.statusText(status, $tt) : String(status);

        const onModelVersionChange = (value) => {
            const MVV = window.DQModelVersionValue;
            const parsed = MVV && typeof MVV.parse === 'function' ? MVV.parse(value) : null;
            if (!parsed) return;
            params.model = parsed.modelKey;
            params.version = parsed.versionKey;
            addLog($tt('studio.switchModel', { name: currentModelDisplayName.value, version: params.version }), 'info');
            loadModelDefaults();
        };

        const imageAutoSaveDraft = ref(false);
        let _imgPromptSaveT = null;
        watch(
            () => params.prompt,
            (v) => {
                if (!imageAutoSaveDraft.value) return;
                const SK = window.DQ_STORAGE || {};
                if (!SK.IMAGE_CREATE_PROMPT_DRAFT) return;
                if (_imgPromptSaveT) clearTimeout(_imgPromptSaveT);
                _imgPromptSaveT = setTimeout(() => {
                    try {
                        localStorage.setItem(SK.IMAGE_CREATE_PROMPT_DRAFT, String(v || ''));
                    } catch (_) {}
                }, 500);
            },
        );

        const applyAppSettingsDefaults = async () => {
            try {
                const st = await api.settings.getSettings();
                imageAutoSaveDraft.value = !!st.auto_save_prompts;
                const SK = window.DQ_STORAGE || {};
                if (st.auto_save_prompts && SK.IMAGE_CREATE_PROMPT_DRAFT) {
                    const draft = localStorage.getItem(SK.IMAGE_CREATE_PROMPT_DRAFT);
                    if (draft) params.prompt = draft;
                }
                const dm = (st.default_model || '').trim();
                if (!dm || !modelRegistry.value || !Object.keys(modelRegistry.value).length) return;
                let mk = null;
                if (modelRegistry.value[dm]) {
                    mk = dm;
                } else {
                    for (const [k, cfg] of Object.entries(modelRegistry.value)) {
                        const media = cfg && cfg.media;
                        if (media === 'video') continue;
                        const n = cfg && cfg.name;
                        if (typeof n === 'string' && n === dm) {
                            mk = k;
                            break;
                        }
                        if (n && typeof n === 'object' && (n.zh === dm || n.en === dm)) {
                            mk = k;
                            break;
                        }
                    }
                }
                if (!mk || !modelRegistry.value[mk]) return;
                const detailed = modelsDetailedStatus.value[mk] || {};
                const vers = detailed.versions || {};
                const cfg = modelRegistry.value[mk];
                const versionKeys = Object.keys(cfg.versions || {});
                const defaultVK =
                    versionKeys.find((vk) => cfg.versions[vk] && cfg.versions[vk].default) || versionKeys[0];
                if (!defaultVK) return;
                const stRow = vers[defaultVK];
                if (stRow && stRow.ready === false) return;
                params.model = mk;
                params.version = defaultVK;
                selectedModelVersion.value = mk + '|' + defaultVK;
                loadModelDefaults();
            } catch (_) {}
        };

        onMounted(async () => {
            await loadModelRegistry();
            await applyAppSettingsDefaults();
            loadPresets();
            loadRecentImages();
            const SK = window.DQ_STORAGE || {};
            const fromGal = SK.IMG2IMG_REF ? localStorage.getItem(SK.IMG2IMG_REF) : null;
            if (fromGal) {
                setImageWorkMode('rewrite_reference');
                editImagePath.value = fromGal;
                editImageSrc.value = api.gallery.getImageUrl(fromGal);
                if (SK.IMG2IMG_REF) localStorage.removeItem(SK.IMG2IMG_REF);
                ElementPlus.ElMessage.success($tt('create.img2imgFromGallery'));
            }
        });
        
        // 监听编辑模式切换：图像编辑自动选支持的模型
        watch(editMode, (newMode) => {
            if (newMode === 'image_editing') {
                params.strength = 0.99;
                const config = currentModelConfig.value;
                const acts = (config && config.actions) ? config.actions : {};
                const hasCap = RA.imageEditingMatches ? RA.imageEditingMatches(acts, editingSubMode.value) : false;
                if (!hasCap) {
                    const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
                    if (firstMatch) {
                        params.model = firstMatch.modelKey;
                        params.version = firstMatch.versionKey;
                        selectedModelVersion.value = firstMatch.modelKey + '|' + firstMatch.versionKey;
                        loadModelDefaults();
                    }
                }
            } else if (newMode === 'image_upscale') {
                const config = currentModelConfig.value;
                const acts = (config && config.actions) ? config.actions : {};
                const hasCap = RA.imageSupportsUpscale ? RA.imageSupportsUpscale(acts) : false;
                if (!hasCap) {
                    const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
                    if (firstMatch) {
                        params.model = firstMatch.modelKey;
                        params.version = firstMatch.versionKey;
                        selectedModelVersion.value = firstMatch.modelKey + '|' + firstMatch.versionKey;
                        loadModelDefaults();
                    }
                }
            }
        });

        // 监听子类型切换：重新过滤模型
        watch(editingSubMode, () => {
            if (editMode.value !== 'image_editing') return;
            const config = currentModelConfig.value;
            const acts = (config && config.actions) ? config.actions : {};
            const hasCap = RA.imageEditingMatches ? RA.imageEditingMatches(acts, editingSubMode.value) : false;
            if (!hasCap) {
                const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
                if (firstMatch) {
                    params.model = firstMatch.modelKey;
                    params.version = firstMatch.versionKey;
                    selectedModelVersion.value = firstMatch.modelKey + '|' + firstMatch.versionKey;
                    loadModelDefaults();
                }
            }
        });

        watch(imageWorkTab, (t) => {
            if (t !== 'rewrite_reference' && t !== 'rewrite_instruct') return;
            const okInList = filteredAllVersions.value.some(
                (v) => v.modelKey === params.model && v.versionKey === params.version,
            );
            if (!okInList) {
                const firstMatch = filteredRecommendedVersions.value[0] || filteredAllVersions.value[0];
                if (firstMatch) {
                    params.model = firstMatch.modelKey;
                    params.version = firstMatch.versionKey;
                    selectedModelVersion.value = firstMatch.modelKey + '|' + firstMatch.versionKey;
                    loadModelDefaults();
                }
            }
        });
        
        return {
            $pn: window.$pn,
            params,
            generating,
            currentTask,
            logs,
            previewImage,
            recentImages,
            advancedParamsOpen,
            compatibleLoras,
            compatibleControlNets,
            controlImageSrc,
            controlImagePath,
            presets,
            filteredPresets,
            selectedPreset,
            modelRegistry,
            modelsStatus,
            modelsDetailedStatus,
            selectedModelVersion,
            allVersions,
            filteredAllVersions,
            filteredRecommendedVersions,
            filteredModelPickerVersions,
            recommendedVersions,
            currentModelConfig,
            currentModelDisplayName,
            selectedModelNotReady,
            hasCustomParams,
            systemInfo,
            previewVisible,
            selectedImage,
            editMode,
            imageWorkTab,
            editingSubMode,
            rewriteDriveMode,
            setImageWorkMode,
            editingSubModeDesc,
            submitDisabled,
            primaryCtaLabel,
            editImageSrc,
            editImagePath,
            imageEditorRef,
            loadPreset,
            addLog,
            clearLogs,
            startGeneration,
            loadRecentImages,
            getImageUrl,
            showPreview,
            quickFromGallery,
            goToSettings,
            goToDownload,
            getStatusType,
            getStatusText,
            onModelVersionChange,
            onEditAssetPick,
            onControlAssetPick,
            removeControlImage,
            resetToDefaults,
            truncate,
            advancedParamVisibility,
            presetSelectLabel
        };
    }
};
