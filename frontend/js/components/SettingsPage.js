/**
 * Settings page component - Beautified version
 */

const SettingsPage = {
    template: `
        <div class="settings-page">
            <el-tabs type="border-card" v-model="activeTab">
                
                <!-- Model config (enhanced) -->
                <el-tab-pane :label="$t('settings.modelConfig')" name="models">
                    <div class="card" style="margin-bottom: 24px;">
                        <div class="card-title">
                            <el-icon><box /></el-icon>
                            {{ $t('settings.modelConfig') }}
                            <span style="color: var(--text-muted); font-size: 13px; font-weight: normal; margin-left: 8px;">
                                {{ $t('settings.modelConfigDesc') }}
                            </span>
                        </div>

                        <!-- Model selector (enhanced: version status, category, size) -->
                        <el-select
                            v-model="selectedModel"
                            style="width: 100%; margin-bottom: 20px;"
                            size="large"
                            @change="onModelSelect"
                            filterable
                        >
                            <el-option
                                v-for="(config, key) in sortedModelRegistry"
                                :key="key"
                                :label="$mn(config)"
                                :value="key"
                            >
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-weight: 500;">{{ $mn(config) }}</span>
                                    <el-tag v-if="config.recommended" size="small" type="success" effect="dark">{{ $t('studio.recommended') }}</el-tag>
                                    <el-tag size="small" type="info">{{ config.engine }}</el-tag>
                                    <el-tag v-if="config.category" size="small" type="warning">{{ categoryLabel(config.category) }}</el-tag>
                                    <span style="margin-left: auto; font-size: 12px; color: var(--text-muted);">
                                        {{ installedVersionCount(key) }}/{{ versionCount(config) }} {{ $t('settings.versionsInstalled') }}
                                    </span>
                                </div>
                            </el-option>
                        </el-select>

                        <!-- Current model config (enhanced) -->
                        <div v-if="currentModelConfig">
                            <!-- Model overview header -->
                            <div class="model-overview-header" style="display: flex; align-items: flex-start; gap: 16px; margin-bottom: 24px; padding: 16px; background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border-color);">
                                <div style="width: 56px; height: 56px; border-radius: 12px; background: rgba(233, 69, 96, 0.1); border: 1px solid rgba(233, 69, 96, 0.2); display: flex; align-items: center; justify-content: center; flex-shrink: 0; color: var(--primary); font-size: 14px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding: 0 4px; box-sizing: border-box;">
                                    {{ modelInitials(currentModelConfig) }}
                                </div>
                                <div style="flex: 1; min-width: 0;">
                                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                                        <span style="font-size: 18px; font-weight: 600; color: var(--text-primary);">{{ $mn(currentModelConfig) }}</span>
                                        <el-tag v-if="currentModelConfig.recommended" size="small" type="success" effect="dark">{{ $t('studio.recommended') }}</el-tag>
                                    </div>
                                    <div style="font-size: 13px; color: var(--text-muted); line-height: 1.5;">{{ $md(currentModelConfig) }}</div>
                                    <div style="display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap;">
                                        <el-tag size="small" type="info">{{ currentModelConfig.engine }}</el-tag>
                                        <el-tag v-if="currentModelConfig.category" size="small" type="warning">{{ categoryLabel(currentModelConfig.category) }}</el-tag>
                                        <el-tag size="small" type="primary">{{ currentModelConfig.type }}</el-tag>
                                        <el-tag
                                            v-for="key in modelActionKeyList"
                                            :key="key"
                                            size="small"
                                            effect="plain"
                                        >
                                            {{ actionTagLabel(key) }}
                                        </el-tag>
                                    </div>
                                </div>
                            </div>

                            <!-- Two-column layout: param config + model info -->
                            <el-row :gutter="24">
                                <!-- Left column: parameter config -->
                                <el-col :xs="24" :md="16">
                                    <!-- Default version selection -->
                                    <div v-if="currentModelConfig.versions" style="margin-bottom: 20px;">
                                        <div style="font-weight: 600; font-size: 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                            <el-icon><collection /></el-icon>
                                            {{ $t('settings.defaultVersion') }}
                                        </div>
                                        <el-select v-model="selectedDefaultVersion" style="width: 100%;" @change="onDefaultVersionChange">
                                            <el-option
                                                v-for="(ver, verKey) in currentModelConfig.versions"
                                                :key="verKey"
                                                :value="verKey"
                                            >
                                                <div style="display: flex; align-items: center; gap: 8px;">
                                                    <span>{{ ver.name }}</span>
                                                    <el-tag size="small" type="info">{{ ver.size }}</el-tag>
                                                    <el-tag
                                                        v-if="versionStatus(selectedModel, verKey) === 'ready'"
                                                        size="small"
                                                        type="success"
                                                    >{{ $t('settings.installed') }}</el-tag>
                                                    <el-tag
                                                        v-else-if="versionStatus(selectedModel, verKey) === 'generatable'"
                                                        size="small"
                                                        type="warning"
                                                    >{{ versionStatusLabel(selectedModel, verKey) }}</el-tag>
                                                    <el-tag
                                                        v-else
                                                        size="small"
                                                        type="info"
                                                    >{{ $t('settings.notInstalled') }}</el-tag>
                                                    <span
                                                        v-if="isRecommendedVersion(verKey)"
                                                        style="margin-left: auto; font-size: 12px; color: var(--primary);"
                                                    >{{ $t('settings.recommendedForYourHardware') }}</span>
                                                </div>
                                            </el-option>
                                        </el-select>
                                        <div v-if="hardwareAdvice" style="margin-top: 8px; padding: 10px 14px; border-radius: 8px; font-size: 13px;" :style="hardwareAdviceStyle">
                                            <div style="display: flex; align-items: center; gap: 8px;">
                                                <el-icon :size="16"><component :is="hardwareAdvice.icon" /></el-icon>
                                                <span>{{ hardwareAdvice.message }}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Parameter preset management -->
                                    <div style="margin-bottom: 20px;">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                            <div style="font-weight: 600; font-size: 14px; display: flex; align-items: center; gap: 8px;">
                                                <el-icon><magic-stick /></el-icon>
                                                {{ $t('settings.paramPresets') }}
                                            </div>
                                            <el-button type="primary" size="small" @click="openParamPresetDialog">
                                                <el-icon><plus /></el-icon>
                                                {{ $t('settings.saveAsPreset') }}
                                            </el-button>
                                        </div>
                                        <div v-if="paramPresetsForModel.length === 0" style="font-size: 13px; color: var(--text-muted); padding: 12px; background: var(--bg-card); border-radius: 8px; border: 1px dashed var(--border-color);">
                                            {{ $t('settings.noParamPresets') }}
                                        </div>
                                        <div v-else style="display: flex; flex-wrap: wrap; gap: 8px;">
                                            <el-tag
                                                v-for="preset in paramPresetsForModel"
                                                :key="preset.id"
                                                size="small"
                                                effect="plain"
                                                closable
                                                @close="deleteParamPreset(preset.id)"
                                                @click="loadParamPreset(preset)"
                                                style="cursor: pointer;"
                                                :type="preset.isDefault ? 'success' : ''"
                                            >
                                                {{ preset.name }}
                                                <span v-if="preset.isDefault" style="margin-left: 4px; font-size: 10px;">({{ $t('settings.default') }})</span>
                                            </el-tag>
                                        </div>
                                    </div>

                                    <!-- Parameter config form (enhanced: note tooltips, type icons, reset buttons) -->
                                    <div class="model-params-section">
                                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                                            <h4 class="section-title" style="margin: 0;">
                                                <el-icon><sliders /></el-icon>
                                                {{ $t('settings.parameters') }}
                                            </h4>
                                            <el-button text type="primary" size="small" @click="onSettingsModelRestoreDefaults">
                                                <el-icon><refresh /></el-icon>
                                                {{ $t('studio.restoreDefaults') }}
                                            </el-button>
                                        </div>

                                        <!-- Custom param form (replaces RegistryParamsForm, adds note display and reset buttons) -->
                                        <el-form label-position="top" size="small" v-if="currentModelConfig">
                                            <!-- Resolution -->
                                            <el-form-item v-if="resPair" :label="$t('studio.resolution')">
                                                <div style="display: flex; align-items: center; gap: 8px;">
                                                    <el-select v-model="modelParams.width" style="width: 120px;">
                                                        <el-option v-for="w in resPair.width.options" :key="w" :label="String(w)" :value="w" />
                                                    </el-select>
                                                    <span style="color: var(--text-muted);">x</span>
                                                    <el-select v-model="modelParams.height" style="width: 120px;">
                                                        <el-option v-for="h in resPair.height.options" :key="h" :label="String(h)" :value="h" />
                                                    </el-select>
                                                </div>
                                            </el-form-item>

                                            <!-- Scalar parameters -->
                                            <template v-for="key in scalarKeys" :key="key">
                                                <el-form-item v-if="specOf(key)">
                                                    <template #label>
                                                        <div style="display: flex; align-items: center; gap: 6px;">
                                                            <span>{{ paramLabel(key, specOf(key)) }}</span>
                                                            <el-tooltip v-if="specOf(key).note" :content="specOf(key).note" placement="top">
                                                                <el-icon style="color: var(--text-muted); cursor: help;"><question-filled /></el-icon>
                                                            </el-tooltip>
                                                            <el-tag v-if="isParamChanged(key)" size="small" type="warning" effect="plain" style="margin-left: auto;">{{ $t('settings.modified') }}</el-tag>
                                                        </div>
                                                    </template>
                                                    <!-- int / float slider -->
                                                    <template v-if="specOf(key).type === 'int' || specOf(key).type === 'float'">
                                                        <div class="param-control-row">
                                                            <div class="param-slider">
                                                                <el-slider
                                                                    v-model="modelParams[key]"
                                                                    :min="specOf(key).min"
                                                                    :max="specOf(key).max"
                                                                    :step="numStep(key, specOf(key))"
                                                                />
                                                            </div>
                                                            <el-input-number
                                                                v-model="modelParams[key]"
                                                                :min="specOf(key).min"
                                                                :max="specOf(key).max"
                                                                :step="numStep(key, specOf(key))"
                                                                class="param-input-number"
                                                            />
                                                            <el-button
                                                                v-if="isParamChanged(key)"
                                                                size="small"
                                                                text
                                                                @click="resetParam(key)"
                                                                :title="$t('settings.resetToDefault')"
                                                            >
                                                                <el-icon><refresh-left /></el-icon>
                                                            </el-button>
                                                        </div>
                                                    </template>
                                                    <!-- enum -->
                                                    <div v-else-if="specOf(key).type === 'enum'" style="display: flex; align-items: center; gap: 8px;">
                                                        <el-select v-model="modelParams[key]" style="flex: 1;">
                                                            <el-option v-for="opt in specOf(key).options" :key="String(opt)" :label="String(opt)" :value="opt" />
                                                        </el-select>
                                                        <el-button
                                                            v-if="isParamChanged(key)"
                                                            size="small"
                                                            text
                                                            @click="resetParam(key)"
                                                            :title="$t('settings.resetToDefault')"
                                                        >
                                                            <el-icon><refresh-left /></el-icon>
                                                        </el-button>
                                                    </div>
                                                    <!-- bool -->
                                                    <div v-else-if="specOf(key).type === 'bool'" style="display: flex; align-items: center; gap: 8px;">
                                                        <el-switch v-model="modelParams[key]" />
                                                        <el-button
                                                            v-if="isParamChanged(key)"
                                                            size="small"
                                                            text
                                                            @click="resetParam(key)"
                                                            :title="$t('settings.resetToDefault')"
                                                        >
                                                            <el-icon><refresh-left /></el-icon>
                                                        </el-button>
                                                    </div>
                                                </el-form-item>
                                            </template>

                                            <!-- LoRA -->
                                            <adapter-picker
                                                v-if="showLoraBlock"
                                                :items="adapterItems"
                                                :adapter-id="modelParams.lora"
                                                @update:adapter-id="modelParams.lora = $event"
                                                :weight="modelParams.lora_scale"
                                                @update:weight="modelParams.lora_scale = $event"
                                                :weight-spec="loraScaleSpec"
                                            />

                                            <!-- Seed -->
                                            <el-form-item v-if="seedSupport" :label="$t('studio.seed')">
                                                <div style="display: flex; gap: 8px;">
                                                    <el-input v-model="modelParams.seed" :placeholder="$t('studio.seedPlaceholder')" style="flex: 1;" />
                                                    <el-button @click="modelParams.seed = String(Math.floor(Math.random() * 1_000_000))">
                                                        <el-icon><refresh /></el-icon>
                                                    </el-button>
                                                </div>
                                            </el-form-item>
                                        </el-form>

                                        <div class="save-button-wrapper" style="margin-top: 20px;">
                                            <el-button type="primary" @click="saveModelConfig" class="save-button">
                                                <el-icon><check /></el-icon>
                                                {{ $t('common.save') }}
                                            </el-button>
                                        </div>
                                    </div>
                                </el-col>

                                <!-- Right column: model info reference -->
                                <el-col :xs="24" :md="8">
                                    <!-- Version status matrix -->
                                    <div class="card" style="margin-bottom: 20px; background: var(--bg-card);">
                                        <div style="font-weight: 600; font-size: 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                            <el-icon><collection /></el-icon>
                                            {{ $t('settings.versionStatus') }}
                                        </div>
                                        <div v-if="!currentModelConfig.versions" style="font-size: 13px; color: var(--text-muted);">
                                            {{ $t('settings.noVersions') }}
                                        </div>
                                        <div v-else style="display: flex; flex-direction: column; gap: 8px;">
                                            <div
                                                v-for="(ver, verKey) in currentModelConfig.versions"
                                                :key="verKey"
                                                style="padding: 10px 12px; border-radius: 8px; border: 1px solid var(--border-color);"
                                                :style="versionItemStyle(verKey)"
                                            >
                                                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                                    <span style="font-weight: 500; font-size: 13px;">{{ ver.name }}</span>
                                                    <el-tag size="small" type="info">{{ ver.size }}</el-tag>
                                                </div>
                                                <div style="display: flex; align-items: center; gap: 6px;">
                                                    <el-tag
                                                        :type="versionStatusType(selectedModel, verKey)"
                                                        size="small"
                                                        effect="dark"
                                                    >
                                                        {{ versionStatusLabel(selectedModel, verKey) }}
                                                    </el-tag>
                                                    <span v-if="ver.source_type === 'derived'" style="font-size: 11px; color: var(--text-muted);">
                                                        {{ $t('settings.from') }} {{ currentModelConfig.versions[ver.from_version]?.name || ver.from_version }}
                                                    </span>
                                                </div>
                                                <div v-if="isRecommendedVersion(verKey)" style="margin-top: 6px; font-size: 12px; color: var(--primary);">
                                                    <el-icon style="vertical-align: middle; margin-right: 4px;"><star-filled /></el-icon>
                                                    {{ $t('settings.recommendedForYourHardware') }}
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Capability matrix -->
                                    <div class="card" style="margin-bottom: 20px; background: var(--bg-card);">
                                        <div style="font-weight: 600; font-size: 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                            <el-icon><check /></el-icon>
                                            {{ $t('settings.capabilities') }}
                                        </div>
                                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                                            <div
                                                v-for="cap in capabilityList"
                                                :key="cap.key"
                                                style="display: flex; align-items: center; gap: 6px; padding: 6px 8px; border-radius: 6px;"
                                                :style="cap.value ? 'background: rgba(103, 194, 58, 0.1);' : 'background: rgba(144, 147, 153, 0.1);'"
                                            >
                                                <el-icon :size="14" :color="cap.value ? '#67c23a' : '#909399'">
                                                    <component :is="cap.value ? 'check' : 'close'" />
                                                </el-icon>
                                                <span style="font-size: 12px;" :style="cap.value ? 'color: #67c23a;' : 'color: #909399;'">{{ cap.label }}</span>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Hardware compatibility -->
                                    <div class="card" style="margin-bottom: 20px; background: var(--bg-card);">
                                        <div style="font-weight: 600; font-size: 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                            <el-icon><cpu /></el-icon>
                                            {{ $t('settings.hardwareCompatibility') }}
                                        </div>
                                        <div v-if="systemInfo.memory_gb" style="margin-bottom: 12px;">
                                            <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                                                <span>{{ $t('settings.systemMemory') }}</span>
                                                <span style="font-weight: 500;">{{ systemInfo.memory_gb.toFixed(1) }} GB</span>
                                            </div>
                                            <el-progress :percentage="Math.min(100, (minVersionSizeGB / systemInfo.memory_gb * 100))" :show-text="false" :stroke-width="6" :color="memoryProgressColor" />
                                        </div>
                                        <div v-if="recommendedVersion" style="padding: 10px; border-radius: 8px; background: rgba(103, 194, 58, 0.1); border: 1px solid rgba(103, 194, 58, 0.3);">
                                            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">{{ $t('settings.recommendedVersion') }}</div>
                                            <div style="font-weight: 600; color: #67c23a;">{{ recommendedVersion.name }} ({{ recommendedVersion.size }})</div>
                                        </div>
                                        <div v-else-if="currentModelConfig.versions" style="padding: 10px; border-radius: 8px; background: rgba(245, 108, 108, 0.1); border: 1px solid rgba(245, 108, 108, 0.3);">
                                            <div style="font-size: 13px; color: var(--danger);">
                                                <el-icon style="vertical-align: middle; margin-right: 4px;"><warning /></el-icon>
                                                {{ $t('settings.memoryInsufficient') }}
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Parameter notes -->
                                    <div class="card" style="background: var(--bg-card);">
                                        <div style="font-weight: 600; font-size: 14px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                                            <el-icon><info-filled /></el-icon>
                                            {{ $t('settings.paramNotes') }}
                                        </div>
                                        <div v-if="!hasParamNotes" style="font-size: 13px; color: var(--text-muted);">
                                            {{ $t('settings.noParamNotes') }}
                                        </div>
                                        <div v-else style="display: flex; flex-direction: column; gap: 10px;">
                                            <div v-for="note in paramNotesList" :key="note.key" style="padding: 8px 10px; background: var(--bg-body); border-radius: 6px;">
                                                <div style="font-weight: 500; font-size: 12px; margin-bottom: 4px; color: var(--text-primary);">{{ note.label }}</div>
                                                <div style="font-size: 12px; color: var(--text-muted); line-height: 1.5;">{{ note.note }}</div>
                                            </div>
                                        </div>
                                    </div>
                                </el-col>
                            </el-row>
                        </div>
                    </div>

                    <!-- Parameter preset save dialog -->
                    <el-dialog v-model="paramPresetDialogVisible" :title="$t('settings.saveParamPreset')" width="400px">
                        <el-form label-position="top">
                            <el-form-item :label="$t('settings.presetName')" required>
                                <el-input v-model="paramPresetForm.name" :placeholder="$t('settings.presetNamePlaceholder')" />
                            </el-form-item>
                            <el-form-item>
                                <el-checkbox v-model="paramPresetForm.isDefault">{{ $t('settings.setAsDefaultPreset') }}</el-checkbox>
                            </el-form-item>
                        </el-form>
                        <template #footer>
                            <el-button @click="paramPresetDialogVisible = false">{{ $t('common.cancel') }}</el-button>
                            <el-button type="primary" @click="saveParamPreset">{{ $t('common.save') }}</el-button>
                        </template>
                    </el-dialog>
                </el-tab-pane>
                
                <!-- Prompt templates -->
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
                    
                    <!-- Add/Edit template dialog -->
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
                
                <!-- System settings -->
                <el-tab-pane :label="$t('settings.systemConfig')" name="system">
                    <el-row :gutter="24">
                        <!-- Left: public config + model list + LoRA list -->
                        <el-col :xs="24" :md="16">
                            <!-- Public config -->
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
                        
                        <!-- Right: system info + real-time resource monitor -->
                        <el-col :xs="24" :md="8">
                            <!-- System info -->
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
                                
                                <!-- Dependency versions -->
                                <div v-if="systemInfo.dependencies" style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-color);">
                                    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 8px;">{{ $t('settings.dependencies') }}</div>
                                    <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                                        <el-tag v-for="(version, name) in systemInfo.dependencies" :key="name" size="small" type="info" effect="plain">
                                            {{ name }} {{ version }}
                                        </el-tag>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Model cache status -->
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
                            
                            <!-- Real-time resource monitor -->
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
                                
                                <!-- Memory -->
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

        // Restore tab state
        const activeTab = ref((SK.SETTINGS_TAB && localStorage.getItem(SK.SETTINGS_TAB)) || 'models');
        // Language always follows i18n, not loaded from backend
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

        // Model config
        const modelRegistry = ref({});
        const selectedModel = ref('');
        // Model install status (loaded from backend)
        const modelsStatus = ref({});
        const modelsDetailedStatus = ref({});
        // Default version selection
        const selectedDefaultVersion = ref('');
        // Parameter presets (localStorage persisted)
        const paramPresets = ref([]);
        const paramPresetDialogVisible = ref(false);
        const paramPresetForm = reactive({ name: '', isDefault: false });
        // Record parameter defaults (for detecting changes)
        const paramDefaults = reactive({});

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

        /** Settings page: installed LoRAs compatible with current model (for AdapterPicker) */
        const settingsCompatibleLoras = ref([]);

        // Installed models and LoRAs
        const installedModels = ref([]);
        
        // Prompt templates
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

        // Load prompt templates
        const loadPresets = async () => {
            try {
                const data = await api.settings.getPresets();
                presets.value = data || {};
            } catch (e) {
                console.error('Failed to load presets:', e);
            }
        };

        // Open template dialog
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

        // Save template
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

        // Confirm template deletion
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

        // Delete template
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

        // Real-time resource monitoring
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

        // ===== Enhanced: model selector & overview =====

        const sortedModelRegistry = computed(() => {
            const entries = Object.entries(modelRegistry.value);
            entries.sort((a, b) => {
                if (a[1].recommended !== b[1].recommended) return a[1].recommended ? -1 : 1;
                const nameA = typeof window.$mn === 'function' ? window.$mn(a[1], a[0]) : a[0];
                const nameB = typeof window.$mn === 'function' ? window.$mn(b[1], b[0]) : b[0];
                return nameA.localeCompare(nameB);
            });
            return Object.fromEntries(entries);
        });

        const categoryLabel = (cat) => {
            const map = {
                'base_models': $tt('download.imageModels'),
                'controlnets': $tt('download.controlNet'),
                'upscalers': $tt('download.upscalers'),
                'loras': $tt('download.loraModels'),
                'tools': $tt('download.tools'),
                'video_models': $tt('download.videoModels'),
                'music_models': $tt('download.audioModels'),
            };
            return map[cat] || cat;
        };

        const modelInitials = (config) => {
            const name = typeof window.$mn === 'function' ? window.$mn(config, '') : (config.id || '');
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
            if (endIndex !== -1) return name.slice(0, endIndex);
            return name.slice(0, 3);
        };

        const versionCount = (config) => {
            return config.versions ? Object.keys(config.versions).length : 0;
        };

        const installedVersionCount = (modelId) => {
            const detail = modelsDetailedStatus.value[modelId];
            if (!detail || !detail.versions) return 0;
            return Object.values(detail.versions).filter((v) => v.ready).length;
        };

        const versionStatus = (modelId, verKey) => {
            const detail = modelsDetailedStatus.value[modelId];
            if (!detail || !detail.versions) return 'missing';
            const verStatus = detail.versions[verKey];
            if (!verStatus) return 'missing';
            if (verStatus.ready) return 'ready';
            const model = modelRegistry.value[modelId];
            if (model && model.versions && model.versions[verKey]) {
                const ver = model.versions[verKey];
                if (ver.source_type === 'derived') {
                    const parentVer = ver.from_version;
                    if (parentVer && detail.versions[parentVer] && detail.versions[parentVer].ready) {
                        return 'generatable';
                    }
                    return 'parent_missing';
                }
            }
            return 'missing';
        };

        const versionStatusType = (modelId, verKey) => {
            const s = versionStatus(modelId, verKey);
            const map = { ready: 'success', generatable: 'warning', parent_missing: 'info', missing: 'info' };
            return map[s] || 'info';
        };

        const versionStatusLabel = (modelId, verKey) => {
            const s = versionStatus(modelId, verKey);
            const map = {
                ready: $tt('settings.installed'),
                generatable: $tt('settings.canGenerate'),
                parent_missing: $tt('settings.waitingParent'),
                missing: $tt('settings.notInstalled'),
            };
            return map[s] || s;
        };

        const versionItemStyle = (verKey) => {
            if (selectedDefaultVersion.value === verKey) {
                return 'border-color: var(--primary); background: rgba(233, 69, 96, 0.05);';
            }
            return '';
        };

        // Parse version size string to GB number
        const parseSizeGB = (sizeStr) => {
            if (!sizeStr) return 0;
            const match = String(sizeStr).match(/([\d.]+)\s*(GB|MB|KB|TB)/i);
            if (!match) return 0;
            const val = parseFloat(match[1]);
            const unit = match[2].toUpperCase();
            if (unit === 'TB') return val * 1024;
            if (unit === 'GB') return val;
            if (unit === 'MB') return val / 1024;
            if (unit === 'KB') return val / (1024 * 1024);
            return val;
        };

        const minVersionSizeGB = computed(() => {
            const cfg = currentModelConfig.value;
            if (!cfg || !cfg.versions) return 0;
            const sizes = Object.values(cfg.versions).map((v) => parseSizeGB(v.size));
            return Math.min(...sizes);
        });

        const memoryProgressColor = computed(() => {
            const ratio = systemInfo.memory_gb ? minVersionSizeGB.value / systemInfo.memory_gb : 0;
            if (ratio < 0.5) return '#67c23a';
            if (ratio < 0.8) return '#e6a23c';
            return '#f56c6c';
        });

        const isRecommendedVersion = (verKey) => {
            const cfg = currentModelConfig.value;
            if (!cfg || !cfg.versions || !systemInfo.memory_gb) return false;
            const ver = cfg.versions[verKey];
            if (!ver) return false;
            const sizeGB = parseSizeGB(ver.size);
            // Recommended: installed with sufficient memory, or smallest within memory range if none installed
            const memoryGB = systemInfo.memory_gb;
            if (sizeGB > memoryGB * 1.2) return false;
            const status = versionStatus(selectedModel.value, verKey);
            if (status === 'ready') return true;
            // If no installed version, recommend the smallest within memory range
            const allReady = Object.keys(cfg.versions).filter((k) => versionStatus(selectedModel.value, k) === 'ready');
            if (allReady.length === 0) {
                const installable = Object.entries(cfg.versions)
                    .filter(([k, v]) => {
                        const s = parseSizeGB(v.size);
                        return s <= memoryGB * 1.2;
                    })
                    .sort((a, b) => parseSizeGB(a[1].size) - parseSizeGB(b[1].size));
                return installable.length > 0 && installable[0][0] === verKey;
            }
            return false;
        };

        const recommendedVersion = computed(() => {
            const cfg = currentModelConfig.value;
            if (!cfg || !cfg.versions) return null;
            const candidates = Object.entries(cfg.versions).filter(([k]) => isRecommendedVersion(k));
            if (candidates.length === 0) return null;
            // Prefer already installed
            const ready = candidates.find(([k]) => versionStatus(selectedModel.value, k) === 'ready');
            if (ready) return { key: ready[0], ...ready[1] };
            return { key: candidates[0][0], ...candidates[0][1] };
        });

        const hardwareAdvice = computed(() => {
            const cfg = currentModelConfig.value;
            if (!cfg || !cfg.versions || !systemInfo.memory_gb) return null;
            const memoryGB = systemInfo.memory_gb;
            const readyCount = installedVersionCount(selectedModel.value);
            if (readyCount > 0) {
                const readyVersions = Object.entries(cfg.versions).filter(([k]) => versionStatus(selectedModel.value, k) === 'ready');
                const largestReady = readyVersions.reduce((a, b) => (parseSizeGB(a[1].size) > parseSizeGB(b[1].size) ? a : b));
                const largestSize = parseSizeGB(largestReady[1].size);
                if (largestSize > memoryGB * 1.2) {
                    return { icon: 'warning', message: $tt('settings.memoryWarningLargeModel', { size: largestReady[1].size }) };
                }
                return { icon: 'check', message: $tt('settings.modelReadyToUse') };
            }
            const smallest = Object.values(cfg.versions).reduce((a, b) => (parseSizeGB(a.size) < parseSizeGB(b.size) ? a : b));
            if (parseSizeGB(smallest.size) > memoryGB * 1.2) {
                return { icon: 'warning', message: $tt('settings.memoryInsufficientForAllVersions', { memory: memoryGB.toFixed(1) }) };
            }
            return { icon: 'info', message: $tt('settings.modelNotInstalled') };
        });

        const hardwareAdviceStyle = computed(() => {
            if (!hardwareAdvice.value) return {};
            const icon = hardwareAdvice.value.icon;
            if (icon === 'warning') return { background: 'rgba(245, 108, 108, 0.1)', border: '1px solid rgba(245, 108, 108, 0.3)', color: 'var(--danger)' };
            if (icon === 'check') return { background: 'rgba(103, 194, 58, 0.1)', border: '1px solid rgba(103, 194, 58, 0.3)', color: '#67c23a' };
            return { background: 'rgba(144, 147, 153, 0.1)', border: '1px solid rgba(144, 147, 153, 0.3)', color: 'var(--text-muted)' };
        });

        const capabilityList = computed(() => {
            const cfg = currentModelConfig.value;
            if (!cfg || !cfg.parameters) return [];
            const caps = [];
            const params = cfg.parameters;
            const labels = {
                lora_support: $tt('settings.loraSupport'),
                seed_support: $tt('settings.seedSupport'),
                negative_prompt_support: $tt('settings.negativePromptSupport'),
            };
            for (const [key, val] of Object.entries(params)) {
                if (key.endsWith('_support') && typeof val === 'boolean') {
                    caps.push({ key, label: labels[key] || key.replace('_support', '').replace(/_/g, ' '), value: val });
                }
            }
            return caps;
        });

        // ===== Parameter presets =====

        const loadParamPresets = () => {
            try {
                const raw = localStorage.getItem('dq_param_presets_v1');
                paramPresets.value = raw ? JSON.parse(raw) : [];
            } catch (e) {
                paramPresets.value = [];
            }
        };

        const saveParamPresetsToStorage = () => {
            localStorage.setItem('dq_param_presets_v1', JSON.stringify(paramPresets.value));
        };

        const paramPresetsForModel = computed(() => {
            const mk = selectedModel.value;
            if (!mk) return [];
            return paramPresets.value.filter((p) => p.modelKey === mk);
        });

        const openParamPresetDialog = () => {
            paramPresetForm.name = '';
            paramPresetForm.isDefault = false;
            paramPresetDialogVisible.value = true;
        };

        const saveParamPreset = () => {
            if (!paramPresetForm.name.trim()) {
                ElementPlus.ElMessage.warning($tt('settings.enterPresetName'));
                return;
            }
            const mk = selectedModel.value;
            if (!mk) return;
            const presetParams = {};
            const cfg = currentModelConfig.value;
            if (cfg && cfg.parameters) {
                for (const [key, spec] of Object.entries(cfg.parameters)) {
                    if (typeof spec !== 'object' || !Object.prototype.hasOwnProperty.call(spec, 'default')) continue;
                    if (spec.type === 'bool' && String(key).endsWith('_support')) continue;
                    if (!Object.prototype.hasOwnProperty.call(modelParams, key)) continue;
                    presetParams[key] = modelParams[key];
                }
            }
            // If set as default, unset other defaults
            if (paramPresetForm.isDefault) {
                paramPresets.value.forEach((p) => {
                    if (p.modelKey === mk) p.isDefault = false;
                });
            }
            paramPresets.value.push({
                id: Date.now().toString(36) + Math.random().toString(36).substr(2),
                modelKey: mk,
                name: paramPresetForm.name.trim(),
                params: presetParams,
                isDefault: paramPresetForm.isDefault,
                createdAt: new Date().toISOString(),
            });
            saveParamPresetsToStorage();
            paramPresetDialogVisible.value = false;
            ElementPlus.ElMessage.success($tt('settings.presetSaved'));
        };

        const loadParamPreset = (preset) => {
            if (!preset || !preset.params) return;
            Object.assign(modelParams, preset.params);
            ElementPlus.ElMessage.success($tt('settings.presetLoaded', { name: preset.name }));
        };

        const deleteParamPreset = (id) => {
            const idx = paramPresets.value.findIndex((p) => p.id === id);
            if (idx !== -1) {
                paramPresets.value.splice(idx, 1);
                saveParamPresetsToStorage();
            }
        };

        // ===== Enhanced param form =====

        const normalizedParams = computed(() => {
            const R = window.RegistryParamSchema;
            const cfg = currentModelConfig.value;
            if (!R || !cfg || !cfg.parameters) return {};
            return R.normalizeParamsDef(cfg.parameters);
        });

        const resPair = computed(() => {
            const R = window.RegistryParamSchema;
            return R ? R.resolutionPair(normalizedParams.value) : null;
        });

        const scalarKeys = computed(() => {
            const R = window.RegistryParamSchema;
            return R ? R.scalarKeysForForm(normalizedParams.value) : [];
        });

        const seedSupport = computed(() => {
            const cfg = currentModelConfig.value;
            return !!(cfg && cfg.parameters && cfg.parameters.seed_support);
        });

        const showLoraBlock = computed(() => {
            const p = currentModelConfig.value && currentModelConfig.value.parameters;
            if (!p || !p.lora_support) return false;
            return Array.isArray(settingsCompatibleLoras.value);
        });

        const adapterItems = computed(() => {
            if (!Array.isArray(settingsCompatibleLoras.value)) return [];
            return settingsCompatibleLoras.value.map((l) => ({ kind: 'lora', id: l.id, name: l.name }));
        });

        const loraScaleSpec = computed(() => {
            const s = normalizedParams.value.lora_scale;
            if (s && (s.type === 'int' || s.type === 'float')) {
                return { min: s.min ?? 0, max: s.max ?? 2, step: s.step ?? 0.1 };
            }
            return { min: 0, max: 2, step: 0.1 };
        });

        const specOf = (key) => normalizedParams.value[key];

        const numStep = (key, spec) => {
            if (typeof spec.step === 'number') return spec.step;
            return spec.type === 'int' ? 1 : 0.1;
        };

        const paramLabel = (key, spec) => {
            const map = {
                steps: 'create.stepsLabel',
                guidance: 'create.guidanceLabel',
                scheduler: 'create.schedulerLabel',
                strength: 'create.strengthLabel',
                controlnet_strength: 'create.controlNetStrengthLabel',
                redux_strength: 'create.reduxStrengthLabel',
            };
            const i18nKey = map[key];
            if (i18nKey) {
                try {
                    return i18n.global.t(i18nKey);
                } catch (e) {
                    /* fall through */
                }
            }
            if (spec && spec.label) return spec.label;
            return key;
        };

        const isParamChanged = (key) => {
            const spec = specOf(key);
            if (!spec || !('default' in spec)) return false;
            return modelParams[key] !== spec.default;
        };

        const resetParam = (key) => {
            const spec = specOf(key);
            if (spec && 'default' in spec) {
                modelParams[key] = spec.default;
            }
        };

        const paramNotesList = computed(() => {
            const cfg = currentModelConfig.value;
            if (!cfg || !cfg.parameters) return [];
            const list = [];
            for (const [key, spec] of Object.entries(cfg.parameters)) {
                if (typeof spec !== 'object') continue;
                if (!spec.note) continue;
                if (spec.type === 'bool' && String(key).endsWith('_support')) continue;
                list.push({ key, label: paramLabel(key, spec), note: spec.note });
            }
            return list;
        });

        const hasParamNotes = computed(() => paramNotesList.value.length > 0);

        // ===== Existing computed =====

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

        /** Plan D: registry action chip uses top-level action.*; video create vs image create use ``media`` (not engine id strings). */
        const actionTagLabel = (key) => {
            const cfg = currentModelConfig.value;
            const media = cfg && cfg.media != null ? String(cfg.media) : '';
            if (key === 'animate') {
                return window.$tt('action.video.animate');
            }
            if (key === 'create' && media === 'video') {
                return window.$tt('action.video.create');
            }
            const imageKeys = new Set(['create', 'rewrite', 'retouch', 'extend', 'upscale']);
            if (imageKeys.has(key)) {
                return window.$tt('action.image.' + key);
            }
            return window.$tt('settings.actionTags.' + key);
        };

        // Persist tab state
        watch(activeTab, (newVal) => {
            if (SK.SETTINGS_TAB) localStorage.setItem(SK.SETTINGS_TAB, newVal);
            if (newVal === 'system') {
                refreshCacheStatus();
            }
        });

        // Sync i18n language change
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

        // Load model registry (enhanced: also load status)
        const loadModelRegistry = async () => {
            try {
                const [registryData, statusData, detailedStatusData] = await Promise.all([
                    api.settings.getModelRegistry(),
                    api.settings.getModelsStatus(),
                    api.settings.getModelsDetailedStatus(),
                ]);
                modelRegistry.value = registryData.models || {};
                modelsStatus.value = statusData || {};
                modelsDetailedStatus.value = detailedStatusData || {};

                // Set default selected model
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

        // Model select change: registry defaults → modelParams + default version + param preset
        const onModelSelect = () => {
            const config = currentModelConfig.value;
            if (!config || !config.parameters) return;

            // 1. Apply registry defaults
            const R = window.RegistryParamSchema;
            if (R) R.applyDefaults(config.parameters, modelParams);

            // 2. Record defaults (for detecting changes)
            Object.keys(paramDefaults).forEach((k) => delete paramDefaults[k]);
            for (const [key, spec] of Object.entries(config.parameters || {})) {
                if (typeof spec === 'object' && 'default' in spec) {
                    paramDefaults[key] = spec.default;
                }
            }

            // 3. Set default version
            if (config.versions) {
                const defaultVer = Object.entries(config.versions).find(([_, v]) => v.default);
                if (defaultVer) {
                    selectedDefaultVersion.value = defaultVer[0];
                } else {
                    selectedDefaultVersion.value = Object.keys(config.versions)[0] || '';
                }
                // If there's a recommended hardware-compatible version, prefer it
                if (recommendedVersion.value) {
                    selectedDefaultVersion.value = recommendedVersion.value.key;
                }
            } else {
                selectedDefaultVersion.value = '';
            }

            // 4. Load default param preset for this model
            const defaultPreset = paramPresetsForModel.value.find((p) => p.isDefault);
            if (defaultPreset) {
                loadParamPreset(defaultPreset);
            }

            loadSettingsCompatibleLoras();
        };

        const onDefaultVersionChange = () => {
            // Add extra logic here when version switches
            console.log('Default version changed to:', selectedDefaultVersion.value);
        };

        const onSettingsModelRestoreDefaults = () => {
            const config = currentModelConfig.value;
            const R = window.RegistryParamSchema;
            if (R && config && config.parameters) {
                R.applyDefaults(config.parameters, modelParams);
                ElementPlus.ElMessage.success($tt('studio.restoredDefaults'));
            }
        };

        // Save model config (directly modify models_registry.json)
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

        // Load settings
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

        // Save settings
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

        // Load installed models
        const loadInstalledModels = async () => {
            try {
                const models = await api.settings.listModels();
                installedModels.value = models;
            } catch (e) {
                console.error('Failed to load models:', e);
            }
        };

        // Language switch
        const handleLanguageChange = (lang) => {
            settings.language = lang;
            // watch auto-syncs i18n and localStorage, no page refresh needed
        };
        
        // Get progress bar color
        const getProgressColor = (percent) => {
            if (percent < 50) return '#67c23a';
            if (percent < 80) return '#e6a23c';
            return '#f56c6c';
        };
        
        // Load system monitor data
        const loadMonitorData = async () => {
            try {
                const data = await api.settings.getSystemMonitor();
                Object.assign(monitorData, data);
            } catch (e) {
                console.error('Failed to load monitor data:', e);
            }
        };
        
        // Start monitor timer
        const startMonitor = () => {
            loadMonitorData();
            monitorInterval = setInterval(loadMonitorData, 3000);
        };
        
        // Stop monitor timer
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
            loadParamPresets();
            startMonitor();
            refreshCacheStatus();
        });
        
        onUnmounted(() => {
            stopMonitor();
        });

        return {
            $mn: window.$mn,
            $md: window.$md,
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
            presetMediaLabel,
            // Enhanced additions
            sortedModelRegistry,
            categoryLabel,
            modelInitials,
            versionCount,
            installedVersionCount,
            versionStatus,
            versionStatusType,
            versionStatusLabel,
            versionItemStyle,
            isRecommendedVersion,
            recommendedVersion,
            hardwareAdvice,
            hardwareAdviceStyle,
            capabilityList,
            memoryProgressColor,
            minVersionSizeGB,
            // Parameter presets
            paramPresets,
            paramPresetsForModel,
            paramPresetDialogVisible,
            paramPresetForm,
            openParamPresetDialog,
            saveParamPreset,
            loadParamPreset,
            deleteParamPreset,
            // Enhanced param form
            resPair,
            scalarKeys,
            seedSupport,
            showLoraBlock,
            adapterItems,
            loraScaleSpec,
            specOf,
            numStep,
            paramLabel,
            isParamChanged,
            resetParam,
            paramNotesList,
            hasParamNotes,
            // Default version
            selectedDefaultVersion,
            onDefaultVersionChange,
        };
    }
};