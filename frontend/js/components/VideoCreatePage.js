/**
 * 视频创作页面组件
 */

const VideoCreatePage = {
    template: `
        <div class="create-page">
            <el-row :gutter="24">
                <!-- 左侧面板：创作区 -->
                <el-col :xs="24" :md="16" :lg="14">
                    <div class="creation-panel">
                        
                        <!-- Plan §3.1：文生视频 / 图生视频 二级 Tab -->
                        <div class="mode-segment" style="margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 4px;">
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: videoWorkMode === 'create' }"
                                    @click="setVideoWorkMode('create')"
                                >
                                    <el-icon><video-camera /></el-icon>
                                    <span>{{ $t('action.video.create') }}</span>
                                </div>
                                <div
                                    class="mode-segment-item"
                                    :class="{ active: videoWorkMode === 'animate' }"
                                    @click="setVideoWorkMode('animate')"
                                >
                                    <el-icon><PictureFilled /></el-icon>
                                    <span>{{ $t('action.video.animate') }}</span>
                                </div>
                        </div>

                        <!-- 模型选择：单层下拉 -->
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
                                        v-for="item in videoModelPickerVersions"
                                        :key="item.modelKey + '|' + item.versionKey"
                                        :label="item.name"
                                        :value="item.modelKey + '|' + item.versionKey"
                                        :disabled="!item.ready"
                                    >
                                        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                                            <span :style="!item.ready ? 'opacity: 0.5;' : ''">{{ item.name }}</span>
                                            <el-tag v-if="item.recommended" size="small" type="success">{{ $t('studio.recommended') }}</el-tag>
                                            <el-tag v-if="item.status === 'ready'" size="small" type="success">{{ $t('studio.ready') }}</el-tag>
                                            <el-tag v-else size="small" type="warning">{{ $t('studio.notDownloaded') }}</el-tag>
                                            <span v-if="item.size" style="color: var(--text-muted); font-size: 12px; margin-left: auto;">
                                                {{ item.size }}
                                            </span>
                                        </div>
                                    </el-option>
                                </el-select>
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
                        
                        <el-alert
                            type="info"
                            :closable="false"
                            show-icon
                            style="margin-bottom: 16px;"
                        >
                            <template #title>{{ $t('video.runtimeCardTitle') }}</template>
                            <div style="font-size: 13px; line-height: 1.65; color: var(--text-secondary);">
                                <p style="margin: 0 0 6px 0;">{{ $tt('video.runtimeClipSecs', { sec: outputClipSecRounded }) }}</p>
                                <p style="margin: 0 0 6px 0;">{{ $t('video.runtimeGenWarning') }}</p>
                                <p v-if="currentVersionDiskSize" style="margin: 0;">{{ $tt('video.runtimeModelSize', { size: currentVersionDiskSize }) }}</p>
                            </div>
                        </el-alert>
                        
                        <!-- 动画化：起始图（必需） -->
                        <div v-if="videoWorkMode === 'animate'" class="card" style="margin-bottom: 16px;">
                            <div class="card-title" style="justify-content: space-between;">
                                <span>
                                    <el-icon><PictureFilled /></el-icon>
                                    {{ $t('action.video.startImage') }}
                                </span>
                            </div>
                            
                            <div v-if="startImageSrc" class="ref-image-thumb" @click="showStartImagePreview">
                                <img :src="startImageSrc" alt="start" />
                                <div class="ref-image-actions">
                                    <el-button size="small" circle @click.stop="showStartImagePreview" :title="$t('studio.zoomIn')">
                                        <el-icon><ZoomIn /></el-icon>
                                    </el-button>
                                    <el-button size="small" circle type="danger" @click.stop="removeStartImage" :title="$t('studio.delete')">
                                        <el-icon><Delete /></el-icon>
                                    </el-button>
                                </div>
                            </div>
                            <div v-else class="ref-image-placeholder" style="padding: 12px; cursor: default;">
                                <asset-picker
                                    accept-kind="image"
                                    :recent-gallery="recentStartImages"
                                    @pick="onStartAssetPick"
                                />
                            </div>
                        </div>

                        <div v-if="videoWorkMode === 'animate'" class="card" style="margin-bottom: 16px;">
                            <div class="card-title" style="justify-content: space-between;">
                                <span>
                                    <el-icon><PictureFilled /></el-icon>
                                    {{ $t('video.tailFrameTitle') }}
                                    <span style="color: var(--text-muted); font-size: 12px; font-weight: 400; margin-left: 4px;">{{ $t('studio.optional') }}</span>
                                </span>
                            </div>
                            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">{{ $t('video.tailFrameHint') }}</div>
                            <div v-if="tailImageSrc" class="ref-image-thumb" @click="showTailImagePreview">
                                <img :src="tailImageSrc" alt="tail" />
                                <div class="ref-image-actions">
                                    <el-button size="small" circle @click.stop="showTailImagePreview" :title="$t('studio.zoomIn')">
                                        <el-icon><ZoomIn /></el-icon>
                                    </el-button>
                                    <el-button size="small" circle type="danger" @click.stop="removeTailImage" :title="$t('studio.delete')">
                                        <el-icon><Delete /></el-icon>
                                    </el-button>
                                </div>
                            </div>
                            <div v-else class="ref-image-placeholder" style="padding: 12px; cursor: default;">
                                <asset-picker
                                    accept-kind="image"
                                    :recent-gallery="recentStartImages"
                                    @pick="onTailAssetPick"
                                />
                            </div>
                        </div>
                        
                        <!-- 提示词输入 -->
                        <div class="card" style="margin-bottom: 16px;">
                            <div class="card-title">
                                <el-icon><edit-pen /></el-icon>
                                {{ $t('studio.prompt') }}
                            </div>
                            
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
                                :placeholder="$t('video.promptPlaceholder')"
                                resize="none"
                                @keydown.meta.enter.prevent="startGeneration"
                                @keydown.ctrl.enter.prevent="startGeneration"
                            />
                            
                            <!-- 负面提示词 -->
                            <el-collapse v-if="currentModelConfig?.parameters?.negative_prompt_support" style="margin-top: 12px; border: none;">
                                <el-collapse-item :title="$t('studio.negativePrompt')" name="negative">
                                    <el-input
                                        v-model="params.negative_prompt"
                                        type="textarea"
                                        :rows="2"
                                        :placeholder="$t('video.negativePlaceholder')"
                                    />
                                </el-collapse-item>
                            </el-collapse>
                        </div>
                        
                        <!-- 高级参数 -->
                        <div class="card" style="margin-bottom: 16px;">
                            <el-collapse v-model="advancedParamsOpen" style="border: none;">
                                <el-collapse-item name="advanced">
                                    <template #title>
                                        <div style="display: flex; align-items: center; gap: 8px; font-weight: 500;">
                                            <el-icon><setting /></el-icon>
                                            <span>{{ $t('studio.advancedParams') }}</span>
                                            <el-tag v-if="hasCustomParams" size="small" type="warning">{{ $t('studio.hasCustom') }}</el-tag>
                                        </div>
                                    </template>
                                    
                                    <el-form label-position="top" size="small" style="padding-top: 12px;">
                                        <!-- 步数 -->
                                        <el-form-item v-if="currentModelConfig?.parameters?.steps" :label="$t('studio.steps')">
                                            <div class="param-control-row">
                                                <div class="param-slider">
                                                    <el-slider
                                                        v-model="params.steps"
                                                        :min="currentModelConfig.parameters.steps.min"
                                                        :max="currentModelConfig.parameters.steps.max"
                                                    />
                                                </div>
                                                <el-input-number v-model="params.steps" :min="1" :max="100" class="param-input-number" />
                                            </div>
                                        </el-form-item>

                                        <!-- CFG -->
                                        <el-form-item v-if="currentModelConfig?.parameters?.guide_scale" :label="$t('video.guideScaleLabel')">
                                            <div class="param-control-row">
                                                <div class="param-slider">
                                                    <el-slider
                                                        v-model="params.guide_scale"
                                                        :min="currentModelConfig.parameters.guide_scale.min"
                                                        :max="currentModelConfig.parameters.guide_scale.max"
                                                        :step="0.1"
                                                    />
                                                </div>
                                                <el-input-number v-model="params.guide_scale" :step="0.1" class="param-input-number" />
                                            </div>
                                        </el-form-item>

                                        <!-- Shift (Wan only) -->
                                        <el-form-item v-if="currentModelConfig?.parameters?.shift" :label="$t('video.shiftLabel')">
                                            <div class="param-control-row">
                                                <div class="param-slider">
                                                    <el-slider
                                                        v-model="params.shift"
                                                        :min="currentModelConfig.parameters.shift.min"
                                                        :max="currentModelConfig.parameters.shift.max"
                                                        :step="0.5"
                                                    />
                                                </div>
                                                <el-input-number v-model="params.shift" :step="0.5" class="param-input-number" />
                                            </div>
                                        </el-form-item>
                                        
                                        <!-- 分辨率 -->
                                        <el-form-item v-if="currentModelConfig?.parameters?.width" :label="$t('studio.resolution')">
                                            <div style="display: flex; align-items: center; gap: 8px;">
                                                <el-select v-model="params.width" style="width: 120px;">
                                                    <el-option
                                                        v-for="w in currentModelConfig.parameters.width.options"
                                                        :key="w"
                                                        :label="w"
                                                        :value="w"
                                                    />
                                                </el-select>
                                                <span style="color: var(--text-muted);">x</span>
                                                <el-select v-model="params.height" style="width: 120px;">
                                                    <el-option
                                                        v-for="h in currentModelConfig.parameters.height.options"
                                                        :key="h"
                                                        :label="h"
                                                        :value="h"
                                                    />
                                                </el-select>
                                            </div>
                                        </el-form-item>
                                        
                                        <!-- 帧数 -->
                                        <el-form-item v-if="currentModelConfig?.parameters?.num_frames" :label="$t('video.numFramesLabel')">
                                            <div class="param-control-row">
                                                <div class="param-slider">
                                                    <el-slider
                                                        v-model="params.num_frames"
                                                        :min="currentModelConfig.parameters.num_frames.min"
                                                        :max="currentModelConfig.parameters.num_frames.max"
                                                        :step="currentModelConfig.parameters.num_frames.step || 1"
                                                    />
                                                </div>
                                                <el-input-number v-model="params.num_frames" :min="1" :max="257" class="param-input-number" />
                                            </div>
                                            <div v-if="currentModelConfig.parameters.num_frames.note" style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">
                                                {{ currentModelConfig.parameters.num_frames.note }}
                                            </div>
                                        </el-form-item>
                                        
                                        <!-- FPS -->
                                        <el-form-item v-if="currentModelConfig?.parameters?.fps" :label="$t('video.fpsLabel')">
                                            <div class="param-control-row">
                                                <div class="param-slider">
                                                    <el-slider
                                                        v-model="params.fps"
                                                        :min="currentModelConfig.parameters.fps.min"
                                                        :max="currentModelConfig.parameters.fps.max"
                                                    />
                                                </div>
                                                <el-input-number v-model="params.fps" :min="1" :max="60" class="param-input-number" />
                                            </div>
                                        </el-form-item>
                                        
                                        <!-- 种子 -->
                                        <el-form-item v-if="currentModelConfig?.parameters?.seed_support" :label="$t('studio.seed')">
                                            <div style="display: flex; gap: 8px;">
                                                <el-input v-model="params.seed" :placeholder="$t('studio.seedPlaceholder')" style="flex: 1;" />
                                                <el-button @click="params.seed = Math.floor(Math.random() * 1000000)">
                                                    <el-icon><refresh /></el-icon>
                                                </el-button>
                                            </div>
                                        </el-form-item>
                                        
                                        <!-- 恢复默认 -->
                                        <el-form-item>
                                            <el-button text type="primary" @click="resetToDefaults" size="small">
                                                <el-icon><refresh /></el-icon>
                                                {{ $t('studio.restoreDefaults') }}
                                            </el-button>
                                        </el-form-item>
                                    </el-form>
                                </el-collapse-item>
                            </el-collapse>
                        </div>

                        <!-- LoRA 选择器 -->
                        <div v-if="currentModelConfig?.parameters?.lora_support" class="card" style="margin-bottom: 16px;">
                            <div style="font-weight: 500; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                <el-icon><collection-tag /></el-icon>
                                <span>{{ $t('studio.loraLabel') }}</span>
                            </div>

                            <!-- 已选 LoRA 列表 -->
                            <div v-if="selectedLoras.length > 0" style="margin-bottom: 12px;">
                                <div
                                    v-for="(lora, index) in selectedLoras"
                                    :key="lora.id"
                                    style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; padding: 8px; background: var(--bg-secondary); border-radius: 6px;"
                                >
                                    <span style="flex: 1; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                                        {{ compatibleLoras.find(c => c.id === lora.id)?.name || lora.id }}
                                    </span>
                                    <el-slider
                                        v-model="lora.weight"
                                        :min="0"
                                        :max="2"
                                        :step="0.1"
                                        style="width: 120px;"
                                    />
                                    <span style="font-size: 12px; color: var(--text-muted); width: 32px; text-align: right;">{{ lora.weight.toFixed(1) }}</span>
                                    <el-button size="small" text @click="moveLoraUp(index)" :disabled="index === 0">
                                        <el-icon><arrow-up /></el-icon>
                                    </el-button>
                                    <el-button size="small" text @click="moveLoraDown(index)" :disabled="index === selectedLoras.length - 1">
                                        <el-icon><arrow-down /></el-icon>
                                    </el-button>
                                    <el-button size="small" text type="danger" @click="removeLora(index)">
                                        <el-icon><delete /></el-icon>
                                    </el-button>
                                </div>
                            </div>

                            <!-- 添加 LoRA -->
                            <el-select
                                :model-value="''"
                                style="width: 100%;"
                                :placeholder="$t('studio.noLora')"
                                @update:model-value="addLora($event)"
                            >
                                <el-option :label="$t('studio.noLora')" value="" />
                                <el-option
                                    v-for="lora in compatibleLoras.filter(c => !selectedLoras.find(s => s.id === c.id))"
                                    :key="lora.id"
                                    :label="lora.name || lora.id"
                                    :value="lora.id"
                                />
                            </el-select>
                        </div>

                        <!-- 生成按钮 -->
                        <div class="card" style="margin-bottom: 16px;">
                            <el-button
                                type="primary"
                                size="large"
                                style="width: 100%; height: 50px; font-size: 16px;"
                                :disabled="submitDisabled || !systemInfo.env_ready"
                                @click="startGeneration"
                            >
                                <el-icon size="20"><video-camera /></el-icon>
                                <span style="margin-left: 8px;">
                                    {{ primaryCtaLabel }}
                                </span>
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
                        <!-- 当前生成预览 -->
                        <div class="card" style="margin-bottom: 16px;">
                            <div class="card-title">
                                <el-icon><video-camera /></el-icon>
                                {{ $t('studio.currentPreview') }}
                            </div>
                            
                            <div v-if="previewVideo" class="video-preview" style="aspect-ratio: 16/9;">
                                <video :src="previewVideo" controls style="width: 100%; height: 100%; object-fit: contain; border-radius: 8px;"></video>
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
                                <el-button size="small" text @click="loadRecentVideos">
                                    <el-icon><refresh /></el-icon>
                                </el-button>
                            </div>
                            
                            <el-empty v-if="recentVideos.length === 0" :description="$t('gallery.empty')" />
                            
                            <el-row v-else :gutter="8">
                                <el-col
                                    v-for="video in recentVideos"
                                    :key="video.path"
                                    :span="12"
                                    style="margin-bottom: 8px;"
                                >
                                    <div class="gallery-card" @click="showVideoPreview(video)">
                                        <div class="gallery-image-wrapper" style="aspect-ratio: 16/9;">
                                            <video :src="getVideoUrl(video)" style="width: 100%; height: 100%; object-fit: cover;" preload="metadata"></video>
                                        </div>
                                    </div>
                                </el-col>
                            </el-row>
                        </div>
                    </div>
                </el-col>
            </el-row>
            
            <!-- 起始图预览对话框 -->
            <el-dialog v-model="startImageViewerVisible" :title="$t('action.video.startImage')" width="70%" center>
                <div v-if="startImageSrc" style="text-align: center;">
                    <img :src="startImageSrc" style="max-width: 100%; max-height: 70vh; border-radius: 8px;" />
                </div>
            </el-dialog>

            <el-dialog v-model="tailImageViewerVisible" :title="$t('video.tailFrameTitle')" width="70%" center>
                <div v-if="tailImageSrc" style="text-align: center;">
                    <img :src="tailImageSrc" style="max-width: 100%; max-height: 70vh; border-radius: 8px;" />
                </div>
            </el-dialog>
            
            <!-- 视频预览对话框 -->
            <el-dialog v-model="videoPreviewVisible" :title="selectedVideo?.name" width="80%" center destroy-on-close>
                <div v-if="selectedVideo" style="text-align: center;">
                    <video :src="getVideoUrl(selectedVideo)" controls style="max-width: 100%; border-radius: 8px;"></video>
                </div>
            </el-dialog>
        </div>
    `,
    
    setup() {
        const { ref, reactive, onMounted, inject, computed, nextTick, watch } = Vue;
        const systemInfo = inject('systemInfo');
        const RA = window.RegistryActions || {};

        // 参数
        const params = reactive({
            prompt: '',
            negative_prompt: '',
            model: '',
            version: '',
            width: 768,
            height: 512,
            num_frames: 97,
            fps: 24,
            steps: 4,
            guide_scale: 3.0,
            shift: 0.0,
            seed: '',
            image_path: '',
        });
        
        const selectedModelVersion = ref('');
        
        // 状态
        const generating = ref(false);
        const currentTask = ref(null);
        const logs = ref([]);
        const previewVideo = ref('');
        const recentVideos = ref([]);
        const recentStartImages = ref([]);
        const advancedParamsOpen = ref([]);
        
        /** Plan §3.1：创建（文生视频）与 动画化（图生视频） */
        const videoWorkMode = ref('create');
        const setVideoWorkMode = (mode) => {
            videoWorkMode.value = mode === 'animate' ? 'animate' : 'create';
        };

        // 起始图片
        const startImageSrc = ref('');
        const startImagePath = ref('');
        const startImageViewerVisible = ref(false);
        const tailImageSrc = ref('');
        const tailImagePath = ref('');
        const tailImageViewerVisible = ref(false);
        
        // 视频预览
        const videoPreviewVisible = ref(false);
        const selectedVideo = ref(null);
        
        // 模型注册表
        const modelRegistry = ref({});
        const modelsDetailedStatus = ref({});

        // LoRA
        const selectedLoras = ref([]);
        const compatibleLoras = ref([]);

        const loadCompatibleLoras = async () => {
            if (!params.model) {
                compatibleLoras.value = [];
                return;
            }
            try {
                const loras = await api.settings.getCompatibleLoras(params.model);
                compatibleLoras.value = loras || [];
            } catch (e) {
                console.error('Failed to load compatible loras:', e);
                compatibleLoras.value = [];
            }
        };

        const addLora = (loraId) => {
            if (selectedLoras.value.find(l => l.id === loraId)) return;
            const lora = compatibleLoras.value.find(l => l.id === loraId);
            const defaultWeight = lora && lora.parameters && lora.parameters.lora_scale
                ? lora.parameters.lora_scale.default
                : 1.0;
            selectedLoras.value.push({ id: loraId, weight: defaultWeight });
        };

        const removeLora = (index) => {
            selectedLoras.value.splice(index, 1);
        };

        const moveLoraUp = (index) => {
            if (index <= 0) return;
            const tmp = selectedLoras.value[index];
            selectedLoras.value[index] = selectedLoras.value[index - 1];
            selectedLoras.value[index - 1] = tmp;
        };

        const moveLoraDown = (index) => {
            if (index >= selectedLoras.value.length - 1) return;
            const tmp = selectedLoras.value[index];
            selectedLoras.value[index] = selectedLoras.value[index + 1];
            selectedLoras.value[index + 1] = tmp;
        };
        
        // 所有模型版本
        const allVersions = computed(() => {
            const result = [];
            for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
                if (!RA.videoModelRow || !RA.videoModelRow(config)) {
                    continue;
                }
                const actions = { ...(config.actions || {}) };
                const versions = config.versions || {};
                const detailed = modelsDetailedStatus.value[modelKey] || {};
                const versionStatuses = detailed.versions || {};
                
                for (const [versionKey, versionConfig] of Object.entries(versions)) {
                    const status = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
                    result.push({
                        modelKey,
                        versionKey,
                        name:
                            typeof window.$mvn === 'function'
                                ? window.$mvn(modelKey, config, versionConfig)
                                : String(modelKey),
                        size: versionConfig.size || '',
                        status: status.status,
                        ready: status.ready,
                        recommended: config.recommended && versionConfig.default,
                        actions,
                    });
                }
            }
            return result;
        });
        
        const recommendedVersions = computed(() => {
            return allVersions.value.filter(v => v.recommended);
        });

        const videoVersionsForMode = computed(() => {
            const filtered = allVersions.value.filter((v) => {
                const acts = v.actions || {};
                if (videoWorkMode.value === 'animate') {
                    return RA.videoSupportsAnimate ? RA.videoSupportsAnimate(acts) : false;
                }
                return RA.videoSupportsCreate ? RA.videoSupportsCreate(acts) : true;
            });
            return filtered.length ? filtered : allVersions.value;
        });

        const videoRecommendedForMode = computed(() => {
            return videoVersionsForMode.value.filter((v) => v.recommended);
        });

        const videoModelPickerVersions = computed(() => {
            const rows = [...videoVersionsForMode.value];
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

        const submitDisabled = computed(() => {
            if (selectedModelNotReady.value) return true;
            if (!String(params.prompt || '').trim()) return true;
            if (videoWorkMode.value === 'animate' && !startImageSrc.value) return true;
            return false;
        });

        const primaryCtaLabel = computed(() =>
            videoWorkMode.value === 'animate' ? $tt('action.video.animate') : $tt('action.video.create'),
        );

        /** Plan §3.2：成片时长（秒，一位小数）由帧数÷帧率估算 */
        const outputClipSecRounded = computed(() => {
            const fps = Math.max(1, Number(params.fps) || 1);
            const nf = Math.max(1, Number(params.num_frames) || 1);
            return Math.round((nf / fps) * 10) / 10;
        });

        /** 注册表当前版本的 size 字段（如 19GB），供显存/磁盘提示 */
        const currentVersionDiskSize = computed(() => {
            const cfg = currentModelConfig.value;
            if (!cfg || !params.version) return '';
            const v = (cfg.versions || {})[params.version];
            return v && v.size ? String(v.size) : '';
        });
        
        // 加载模型注册表和状态
        const loadModelRegistry = async () => {
            try {
                const RS = window.RegistryStore;
                const regPromise =
                    RS && RS.load
                        ? RS.load()
                        : api.settings.getModelRegistry().then((r) => ({ models: r.models }));
                const [registryData, detailedStatusData] = await Promise.all([
                    regPromise,
                    api.settings.getModelsDetailedStatus(),
                ]);

                modelRegistry.value = (registryData && registryData.models) || {};
                modelsDetailedStatus.value = detailedStatusData || {};
                
                // 设置默认模型
                if (!selectedModelVersion.value) {
                    let found = false;
                    for (const item of videoRecommendedForMode.value) {
                        if (item.ready) {
                            params.model = item.modelKey;
                            params.version = item.versionKey;
                            selectedModelVersion.value = item.modelKey + '|' + item.versionKey;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        for (const item of videoVersionsForMode.value) {
                            if (item.ready) {
                                params.model = item.modelKey;
                                params.version = item.versionKey;
                                selectedModelVersion.value = item.modelKey + '|' + item.versionKey;
                                found = true;
                                break;
                            }
                        }
                    }
                    if (!found && videoVersionsForMode.value.length > 0) {
                        const first = videoVersionsForMode.value[0];
                        params.model = first.modelKey;
                        params.version = first.versionKey;
                        selectedModelVersion.value = first.modelKey + '|' + first.versionKey;
                    }
                }
                
                loadModelDefaults();
            } catch (e) {
                console.error('Failed to load model registry:', e);
            }
        };
        
        // 加载模型的默认配置
        const loadModelDefaults = () => {
            const config = currentModelConfig.value;
            if (!config || !config.parameters) return;
            
            const p = config.parameters;
            if (p.steps) params.steps = p.steps.default;
            if (p.guide_scale) params.guide_scale = p.guide_scale.default;
            if (p.shift) params.shift = p.shift.default;
            if (p.width) params.width = p.width.default;
            if (p.height) params.height = p.height.default;
            if (p.num_frames) params.num_frames = p.num_frames.default;
            if (p.fps) params.fps = p.fps.default;
            params.seed = '';
        };
        
        // 重置为默认配置
        const resetToDefaults = () => {
            loadModelDefaults();
            ElementPlus.ElMessage.success($tt('studio.restoredDefaults'));
        };
        
        // 检查是否有自定义参数
        const hasCustomParams = computed(() => {
            const config = currentModelConfig.value;
            if (!config || !config.parameters) return false;
            const p = config.parameters;
            if (p.steps && params.steps !== p.steps.default) return true;
            if (p.guide_scale && params.guide_scale !== p.guide_scale.default) return true;
            if (p.shift && params.shift !== p.shift.default) return true;
            if (p.width && params.width !== p.width.default) return true;
            if (p.height && params.height !== p.height.default) return true;
            if (p.num_frames && params.num_frames !== p.num_frames.default) return true;
            if (p.fps && params.fps !== p.fps.default) return true;
            if (params.seed) return true;
            return false;
        });

        const presets = ref({});
        const selectedPreset = ref('');

        const presetActionFilter = computed(() => {
            if (videoWorkMode.value === 'animate') {
                return new Set(['animate']);
            }
            return new Set(['create']);
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
                return preset.media_scope === 'video';
            }

            function matches(preset) {
                if (!planPresetShapeOk(preset)) return false;
                if (!matchesMediaScope(preset)) return false;
                return preset.applies_to.some((k) => want.has(k));
            }
            const entries = Object.entries(presets.value)
                .filter(([, preset]) => matches(preset))
                .sort((a, b) => {
                    const ac = a[1].applies_to.includes('create');
                    const bc = b[1].applies_to.includes('create');
                    if (ac !== bc) {
                        return ac ? -1 : 1;
                    }
                    return a[0].localeCompare(b[0], 'zh');
                });
            const result = {};
            for (const [name, preset] of entries) {
                result[name] = preset;
            }
            return result;
        });

        const presetSelectLabel = (name, preset) => {
            const a = preset.applies_to;
            const hasC = a.includes('create');
            const hasA = a.includes('animate');
            let tag = '';
            if (hasC && hasA) {
                tag = $tt('video.presetTagHybrid');
            } else if (hasC && !hasA) {
                tag = $tt('video.presetTagT2V');
            } else if (hasA && !hasC) {
                tag = $tt('video.presetTagI2V');
            }
            const display = window.$pn ? window.$pn(preset, name) : name;
            return tag ? `${tag} ${display}` : display;
        };

        const loadPresets = async () => {
            try {
                const data = await api.settings.getPresets();
                presets.value = data || {};
            } catch (e) {
                console.error('Failed to load presets:', e);
                presets.value = {};
            }
        };

        const loadPreset = () => {
            if (!selectedPreset.value || !presets.value[selectedPreset.value]) return;
            const preset = presets.value[selectedPreset.value];
            const app = preset.applies_to;
            const animateOnly = app.includes('animate') && !app.includes('create');
            if (animateOnly && (videoWorkMode.value === 'create' || !startImageSrc.value)) {
                ElementPlus.ElMessage.warning($tt('video.presetNeedsStartImage'));
            }
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
            if (!params.prompt) {
                ElementPlus.ElMessage.warning($tt('studio.enterPrompt'));
                return;
            }
            
            const detailed = modelsDetailedStatus.value[params.model];
            const versionStatus = detailed?.versions?.[params.version];
            if (!versionStatus?.ready) {
                ElementPlus.ElMessage.warning($tt('studio.modelNotReadyDesc', { name: currentModelConfig.value?.name || params.model, version: params.version }));
                return;
            }

            const verCfg = (currentModelConfig.value && currentModelConfig.value.versions && currentModelConfig.value.versions[params.version]) || null;
            const sizeHuman = verCfg && verCfg.size ? String(verCfg.size) : '';
            if (window.DQMemoryHint && typeof window.DQMemoryHint.warnIfRisky === 'function') {
                window.DQMemoryHint.warnIfRisky({ systemInfo, versionSizeHuman: sizeHuman, $tt });
            }

            addLog($tt('studio.startingGen'), 'info');
            
            try {
                const modelStr = params.version ? `${params.model}:${params.version}` : params.model;
                let submitRes;
                if (videoWorkMode.value === 'animate') {
                    if (!startImageSrc.value) {
                        ElementPlus.ElMessage.warning($tt('video.needStartImage'));
                        return;
                    }
                    let source_asset_id;
                    const sp = startImagePath.value;
                    if (typeof sp === 'string' && sp.startsWith('asset:')) {
                        source_asset_id = sp.slice('asset:'.length);
                    } else {
                        const blob = await api.gen.urlToBlob(startImageSrc.value);
                        const up = await api.gen.uploadAsset(
                            new File([blob], 'start.png', { type: blob.type || 'image/png' })
                        );
                        source_asset_id = up.id;
                    }
                    let tail_asset_id = undefined;
                    if (tailImageSrc.value) {
                        const tp = tailImagePath.value;
                        if (typeof tp === 'string' && tp.startsWith('asset:')) {
                            tail_asset_id = tp.slice('asset:'.length);
                        } else {
                            const tblob = await api.gen.urlToBlob(tailImageSrc.value);
                            const tup = await api.gen.uploadAsset(
                                new File([tblob], 'tail.png', { type: tblob.type || 'image/png' })
                            );
                            tail_asset_id = tup.id;
                        }
                    }
                    const animateBody = {
                        model: modelStr,
                        operation: 'animate',
                        source_asset_id,
                        prompt: params.prompt,
                        negative_prompt: params.negative_prompt || '',
                        size: `${params.width}x${params.height}`,
                        num_frames: params.num_frames,
                        fps: params.fps || 16,
                        steps: params.steps,
                        guidance: params.guide_scale,
                        shift: params.shift || undefined,
                        seed: params.seed ? parseInt(params.seed, 10) : null,
                        priority: 'normal',
                    };
                    if (tail_asset_id) {
                        animateBody.tail_asset_id = tail_asset_id;
                    }
                    if (selectedLoras.value.length > 0) {
                        animateBody.adapters = selectedLoras.value.map(l => ({ id: l.id, weight: l.weight }));
                    }
                    submitRes = await api.gen.createVideoEdit(animateBody);
                } else {
                    submitRes = await api.gen.createVideoGeneration({
                        model: modelStr,
                        prompt: params.prompt,
                        negative_prompt: params.negative_prompt || '',
                        size: `${params.width}x${params.height}`,
                        num_frames: params.num_frames,
                        fps: params.fps || 16,
                        steps: params.steps,
                        guidance: params.guide_scale,
                        shift: params.shift || undefined,
                        seed: params.seed ? parseInt(params.seed, 10) : null,
                        priority: 'normal',
                    });
                }
                const tid = submitRes.task.id;
                currentTask.value = { id: tid, progress: 0, status: 'queued', params: { model: modelStr } };
                api.gen.streamMediaTask(
                    tid,
                    (logData) => addLog(logData.message, logData.level),
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
                                previewVideo.value = api.gallery.getImageUrl(`asset:${pid}`);
                                addLog($tt('studio.outputFile', { name: pid }), 'info');
                            }
                            loadRecentVideos();
                        } else if (doneData.status === 'failed') {
                            const updated = await api.gen.getMediaTask(tid);
                            currentTask.value = updated;
                            addLog($tt('studio.genFailed', { msg: updated.error_message || '' }), 'error');
                        }
                    },
                    () => addLog($tt('studio.connectionLost'), 'warning')
                );
            } catch (e) {
                addLog($tt('studio.error', { msg: e.message }), 'error');
            }
        };
        
        // 加载最近视频
        const loadRecentVideos = async () => {
            try {
                const videos = await api.gallery.listImages(4, 0);
                // 过滤视频文件
                recentVideos.value = videos.filter((v) => {
                    if (v.metadata && v.metadata.asset_kind === 'video') {
                        return true;
                    }
                    const ext = v.name?.split('.').pop()?.toLowerCase();
                    return ['mp4', 'mov', 'avi', 'mkv'].includes(ext);
                });
            } catch (e) {
                console.error('Failed to load recent videos:', e);
            }
        };

        const loadRecentStartImages = async () => {
            try {
                const images = await api.gallery.listImages(24, 0);
                recentStartImages.value = images
                    .filter((v) => {
                        if (v.metadata && v.metadata.asset_kind === 'video') {
                            return false;
                        }
                        const ext = v.name?.split('.').pop()?.toLowerCase();
                        return !['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext || '');
                    })
                    .slice(0, 8);
            } catch (e) {
                console.error('Failed to load recent start images:', e);
            }
        };
        
        // 获取视频URL
        const getVideoUrl = (video) => {
            return api.gallery.getImageUrl(video.path);
        };
        
        // 显示视频预览
        const showVideoPreview = (video) => {
            selectedVideo.value = video;
            videoPreviewVisible.value = true;
        };
        
        // 起始图相关
        const onStartAssetPick = async ({ path, previewUrl }) => {
            startImagePath.value = path;
            startImageSrc.value = previewUrl;
            addLog($tt('studio.startImageAdded', { name: path.replace(/^asset:/, '') }), 'info');
            await loadRecentStartImages();
        };
        
        const removeStartImage = () => {
            startImageSrc.value = '';
            startImagePath.value = '';
        };
        
        const showStartImagePreview = () => {
            startImageViewerVisible.value = true;
        };

        const onTailAssetPick = async ({ path, previewUrl }) => {
            tailImagePath.value = path;
            tailImageSrc.value = previewUrl;
            addLog($tt('studio.startImageAdded', { name: path.replace(/^asset:/, '') }), 'info');
            await loadRecentStartImages();
        };

        const removeTailImage = () => {
            tailImageSrc.value = '';
            tailImagePath.value = '';
        };

        const showTailImagePreview = () => {
            tailImageViewerVisible.value = true;
        };
        
        // 跳转到下载页面
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
            selectedLoras.value = []; // 切换模型时清空已选 LoRA
            loadModelDefaults();
            loadCompatibleLoras();
            addLog(
                $tt('studio.switchModel', {
                    name: currentModelConfig.value?.name || params.model,
                    version: params.version,
                }),
                'info'
            );
        };

        const videoAutoSaveDraft = ref(false);
        let _vidPromptSaveT = null;
        watch(
            () => params.prompt,
            (v) => {
                if (!videoAutoSaveDraft.value) return;
                const SK = window.DQ_STORAGE || {};
                if (!SK.VIDEO_CREATE_PROMPT_DRAFT) return;
                if (_vidPromptSaveT) clearTimeout(_vidPromptSaveT);
                _vidPromptSaveT = setTimeout(() => {
                    try {
                        localStorage.setItem(SK.VIDEO_CREATE_PROMPT_DRAFT, String(v || ''));
                    } catch (_) {}
                }, 500);
            },
        );

        const applyVideoAppSettingsDefaults = async () => {
            try {
                const st = await api.settings.getSettings();
                videoAutoSaveDraft.value = !!st.auto_save_prompts;
                const SK = window.DQ_STORAGE || {};
                if (st.auto_save_prompts && SK.VIDEO_CREATE_PROMPT_DRAFT) {
                    const draft = localStorage.getItem(SK.VIDEO_CREATE_PROMPT_DRAFT);
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
                        if (media !== 'video') continue;
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
            await applyVideoAppSettingsDefaults();
            loadPresets();
            loadRecentVideos();
            loadRecentStartImages();
        });

        watch(videoWorkMode, () => {
            const cfg = currentModelConfig.value;
            const acts = (cfg && cfg.actions) ? cfg.actions : {};
            const ok =
                videoWorkMode.value === 'animate'
                    ? RA.videoSupportsAnimate && RA.videoSupportsAnimate(acts)
                    : RA.videoSupportsCreate && RA.videoSupportsCreate(acts);
            if (!ok) {
                const first = videoRecommendedForMode.value[0] || videoVersionsForMode.value[0];
                if (first) {
                    params.model = first.modelKey;
                    params.version = first.versionKey;
                    selectedModelVersion.value = first.modelKey + '|' + first.versionKey;
                    loadModelDefaults();
                }
            }
        });

        return {
            params,
            selectedModelVersion,
            generating,
            currentTask,
            logs,
            previewVideo,
            recentVideos,
            recentStartImages,
            advancedParamsOpen,
            videoWorkMode,
            setVideoWorkMode,
            videoVersionsForMode,
            videoRecommendedForMode,
            videoModelPickerVersions,
            currentModelDisplayName,
            submitDisabled,
            primaryCtaLabel,
            startImageSrc,
            startImagePath,
            startImageViewerVisible,
            tailImageSrc,
            tailImagePath,
            tailImageViewerVisible,
            onTailAssetPick,
            removeTailImage,
            showTailImagePreview,
            videoPreviewVisible,
            selectedVideo,
            currentModelConfig,
            selectedModelNotReady,
            outputClipSecRounded,
            currentVersionDiskSize,
            allVersions,
            recommendedVersions,
            hasCustomParams,
            presets,
            selectedPreset,
            filteredPresets,
            loadPreset,
            presetSelectLabel,
            systemInfo,
            loadModelDefaults,
            resetToDefaults,
            addLog,
            clearLogs,
            truncate,
            startGeneration,
            loadRecentVideos,
            loadRecentStartImages,
            getVideoUrl,
            showVideoPreview,
            onStartAssetPick,
            removeStartImage,
            showStartImagePreview,
            goToDownload,
            getStatusType,
            getStatusText,
            onModelVersionChange,
            // LoRA
            selectedLoras,
            compatibleLoras,
            addLora,
            removeLora,
            moveLoraUp,
            moveLoraDown,
        };
    }
};
