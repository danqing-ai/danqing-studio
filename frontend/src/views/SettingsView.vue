<!-- @ts-nocheck -->
<template>
  <div class="settings-page">
    <el-tabs v-model="activeTab" class="settings-ep-tabs settings-ep-tabs--segmented">
    <el-tab-pane :label="$t('settings.modelConfig')" name="models">
        <el-card shadow="never" class="studio-ep-surface-card settings-ep-tab-panel">
          <template #header>
            <div class="card-title">
              <el-icon><box /></el-icon>
              {{ $t('settings.modelConfig') }}
              <el-text class="settings-ep-title-desc" size="small" type="info">
                {{ $t('settings.modelConfigDesc') }}
              </el-text>
            </div>
          </template>

          <el-select
            v-model="selectedModel"
            class="settings-ep-model-picker"
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
              <div class="settings-ep-select-option">
                <span class="settings-ep-select-option__name">{{ $mn(config) }}</span>
                <ModelLicenseBadges
                  :recommended="config.recommended"
                  :commercial-use-allowed="config.commercial_use_allowed"
                />
                <el-tag size="small" type="info">{{ config.engine }}</el-tag>
                <el-tag v-if="config.category" size="small" type="warning">{{ categoryLabel(config.category) }}</el-tag>
                <span class="settings-ep-select-option__meta">
                  {{ installedVersionCount(key) }}/{{ versionCount(config) }} {{ $t('settings.versionsInstalled') }}
                </span>
              </div>
            </el-option>
          </el-select>

          <div v-if="currentModelConfig">
            <el-card class="settings-ep-overview-card" shadow="never">
              <div class="settings-ep-overview">
                <div class="settings-ep-overview__avatar">
                  {{ modelInitials(currentModelConfig) }}
                </div>
                <div class="settings-ep-overview__body">
                  <div class="settings-ep-overview__title-row">
                    <span class="settings-ep-overview__title">{{ $mn(currentModelConfig) }}</span>
                    <ModelLicenseBadges
                      :recommended="currentModelConfig.recommended"
                      :commercial-use-allowed="currentModelConfig.commercial_use_allowed"
                    />
                  </div>
                  <el-text class="settings-ep-overview__desc" size="small" type="info" tag="p">
                    {{ $md(currentModelConfig) }}
                  </el-text>
                  <div class="settings-ep-overview__tags">
                    <el-tag size="small" type="info">{{ currentModelConfig.engine }}</el-tag>
                    <el-tag v-if="currentModelConfig.category" size="small" type="warning">{{ categoryLabel(currentModelConfig.category) }}</el-tag>
                    <el-tag size="small" type="info" effect="plain">{{ currentModelConfig.type }}</el-tag>
                    <el-tag
                      v-for="ak in modelActionKeyList"
                      :key="ak"
                      size="small"
                      effect="plain"
                    >
                      {{ actionTagLabel(ak) }}
                    </el-tag>
                  </div>
                </div>
              </div>
            </el-card>

            <!-- Two-column layout: param config + model info -->
            <el-row :gutter="20" class="settings-ep-layout-row">
              <el-col :xs="24" :md="16" :lg="17" :xl="18">
                <div v-if="currentModelConfig.versions" class="settings-ep-section">
                  <div class="settings-ep-section-head">
                    <el-icon><collection /></el-icon>
                    {{ $t('settings.defaultVersion') }}
                  </div>
                  <el-select v-model="selectedDefaultVersion" class="settings-ep-default-version-select" @change="onDefaultVersionChange">
                    <el-option
                      v-for="(ver, verKey) in currentModelConfig.versions"
                      :key="verKey"
                      :value="verKey"
                    >
                      <div class="settings-ep-select-option">
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
                          type="info"
                          effect="plain"
                        >{{ versionStatusLabel(selectedModel, verKey) }}</el-tag>
                        <el-tag
                          v-else
                          size="small"
                          type="info"
                        >{{ $t('settings.notInstalled') }}</el-tag>
                        <span
                          v-if="isRecommendedVersion(verKey)"
                          class="settings-ep-version-option-rec"
                        >{{ $t('settings.recommendedForYourHardware') }}</span>
                      </div>
                    </el-option>
                  </el-select>
                  <el-alert
                    v-if="hardwareAdvice"
                    :type="hardwareAdviceAlertType"
                    :closable="false"
                    show-icon
                    class="settings-ep-hardware-alert"
                  >
                    {{ hardwareAdvice.message }}
                  </el-alert>
                </div>

                <div class="settings-ep-section">
                  <div class="settings-ep-section-head settings-ep-section-head--toolbar">
                    <span class="settings-ep-card-header">
                      <el-icon><magic-stick /></el-icon>
                      {{ $t('settings.paramPresets') }}
                    </span>
                    <el-button size="small" plain class="settings-ep-tool-btn" @click="openParamPresetDialog">
                      <el-icon><plus /></el-icon>
                      {{ $t('settings.saveAsPreset') }}
                    </el-button>
                  </div>
                  <el-alert
                    v-if="paramPresetsForModel.length === 0"
                    type="info"
                    :closable="false"
                    show-icon
                  >
                    {{ $t('settings.noParamPresets') }}
                  </el-alert>
                  <div v-else class="settings-ep-preset-tags">
                    <el-tag
                      v-for="preset in paramPresetsForModel"
                      :key="preset.id"
                      size="small"
                      effect="plain"
                      closable
                      class="settings-ep-preset-tag"
                      @close="deleteParamPreset(preset.id)"
                      @click="loadParamPreset(preset)"
                      :type="preset.isDefault ? 'success' : ''"
                    >
                      {{ preset.name }}
                      <span v-if="preset.isDefault" class="settings-ep-preset-tag-default-note">({{ $t('settings.default') }})</span>
                    </el-tag>
                  </div>
                </div>

                <div class="model-params-section">
                  <div class="settings-ep-params-toolbar">
                    <h4 class="section-title">
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
                      <div class="settings-ep-res-row">
                        <el-select v-model="modelParams.width" class="settings-ep-select--w120">
                          <el-option v-for="w in resPair.width.options" :key="w" :label="String(w)" :value="w" />
                        </el-select>
                        <span class="settings-ep-res-x">x</span>
                        <el-select v-model="modelParams.height" class="settings-ep-select--w120">
                          <el-option v-for="h in resPair.height.options" :key="h" :label="String(h)" :value="h" />
                        </el-select>
                      </div>
                    </el-form-item>

                    <template v-for="key in scalarKeys" :key="key">
                      <el-form-item v-if="specOf(key)">
                        <template #label>
                          <div class="settings-ep-param-label-row">
                            <span>{{ paramLabel(key, specOf(key)) }}</span>
                            <el-tooltip v-if="specOf(key).note" :content="specOf(key).note" placement="top">
                              <el-icon class="settings-ep-help-icon"><question-filled /></el-icon>
                            </el-tooltip>
                            <el-tag v-if="isParamChanged(key)" size="small" type="warning" effect="plain">{{ $t('settings.modified') }}</el-tag>
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
                        <div v-else-if="specOf(key).type === 'enum'" class="settings-ep-enum-row">
                          <el-select v-model="modelParams[key]" class="settings-ep-select--flex">
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
                        <div v-else-if="specOf(key).type === 'bool'" class="settings-ep-bool-row">
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
                      <div class="settings-ep-seed-row">
                        <el-input v-model="modelParams.seed" :placeholder="$t('studio.seedPlaceholder')" />
                        <el-button @click="modelParams.seed = String(Math.floor(Math.random() * 1_000_000))">
                          <el-icon><refresh /></el-icon>
                        </el-button>
                      </div>
                    </el-form-item>
                  </el-form>

                  <div class="save-button-wrapper settings-ep-form-mt">
                    <el-button type="primary" @click="saveModelConfig" class="save-button">
                      <el-icon><check /></el-icon>
                      {{ $t('common.save') }}
                    </el-button>
                  </div>
                </div>
              </el-col>

              <!-- Right column: model info reference -->
              <el-col :xs="24" :md="8" :lg="7" :xl="6">
                <el-card class="settings-ep-side-card" shadow="never">
                  <template #header>
                    <div class="settings-ep-card-header">
                      <el-icon><collection /></el-icon>
                      <span>{{ $t('settings.versionStatus') }}</span>
                    </div>
                  </template>
                  <el-text v-if="!currentModelConfig.versions" size="small" type="info">
                    {{ $t('settings.noVersions') }}
                  </el-text>
                  <div v-else class="settings-ep-version-stack">
                    <div
                      v-for="(ver, verKey) in currentModelConfig.versions"
                      :key="verKey"
                      class="settings-ep-version-item"
                      :class="{ 'is-selected': selectedDefaultVersion === verKey }"
                    >
                      <div class="settings-ep-version-item__row">
                        <span class="settings-ep-version-item__name">{{ ver.name }}</span>
                        <el-tag size="small" type="info">{{ ver.size }}</el-tag>
                      </div>
                      <div class="settings-ep-version-item__meta">
                        <el-tag
                          :type="versionStatusType(selectedModel, verKey)"
                          size="small"
                          effect="plain"
                        >
                          {{ versionStatusLabel(selectedModel, verKey) }}
                        </el-tag>
                        <span v-if="ver.source_type === 'derived'" class="settings-ep-version-derived">
                          {{ $t('settings.from') }} {{ currentModelConfig.versions[ver.from_version]?.name || ver.from_version }}
                        </span>
                      </div>
                      <div v-if="isRecommendedVersion(verKey)" class="settings-ep-version-recommended">
                        <el-icon><star-filled /></el-icon>
                        {{ $t('settings.recommendedForYourHardware') }}
                      </div>
                    </div>
                  </div>
                </el-card>

                <el-card class="settings-ep-side-card" shadow="never">
                  <template #header>
                    <div class="settings-ep-card-header">
                      <el-icon><check /></el-icon>
                      <span>{{ $t('settings.capabilities') }}</span>
                    </div>
                  </template>
                  <div class="settings-ep-cap-grid">
                    <div
                      v-for="cap in capabilityList"
                      :key="cap.key"
                      class="settings-ep-cap-cell"
                      :class="cap.value ? 'is-active' : 'is-inactive'"
                    >
                      <el-icon :size="14" :color="cap.value ? 'var(--el-color-success)' : 'var(--el-text-color-secondary)'">
                        <component :is="cap.value ? 'check' : 'close'" />
                      </el-icon>
                      <span class="settings-ep-cap-label">{{ cap.label }}</span>
                    </div>
                  </div>
                </el-card>

                <el-card class="settings-ep-side-card" shadow="never">
                  <template #header>
                    <div class="settings-ep-card-header">
                      <el-icon><cpu /></el-icon>
                      <span>{{ $t('settings.hardwareCompatibility') }}</span>
                    </div>
                  </template>
                  <div v-if="systemInfo.memory_gb" class="settings-ep-memory-row">
                    <div class="settings-ep-memory-row-head">
                      <span>{{ $t('settings.systemMemory') }}</span>
                      <span class="settings-ep-memory-val">{{ systemInfo.memory_gb.toFixed(1) }} GB</span>
                    </div>
                    <el-progress :percentage="Math.min(100, (minVersionSizeGB / systemInfo.memory_gb * 100))" :show-text="false" :stroke-width="6" :color="memoryProgressColor" />
                  </div>
                  <el-alert
                    v-if="recommendedVersion"
                    type="success"
                    :closable="false"
                    show-icon
                  >
                    <template #title>{{ $t('settings.recommendedVersion') }}</template>
                    {{ recommendedVersion.name }} ({{ recommendedVersion.size }})
                  </el-alert>
                  <el-alert
                    v-else-if="currentModelConfig.versions"
                    type="error"
                    :closable="false"
                    show-icon
                  >
                    {{ $t('settings.memoryInsufficient') }}
                  </el-alert>
                </el-card>

                <el-card class="settings-ep-side-card settings-ep-side-card--last" shadow="never">
                  <template #header>
                    <div class="settings-ep-card-header">
                      <el-icon><info-filled /></el-icon>
                      <span>{{ $t('settings.paramNotes') }}</span>
                    </div>
                  </template>
                  <el-text v-if="!hasParamNotes" size="small" type="info">
                    {{ $t('settings.noParamNotes') }}
                  </el-text>
                  <div v-else class="settings-ep-notes-stack">
                    <div v-for="note in paramNotesList" :key="note.key" class="settings-ep-note-item">
                      <div class="settings-ep-note-label">{{ note.label }}</div>
                      <div class="settings-ep-note-body">{{ note.note }}</div>
                    </div>
                  </div>
                </el-card>
              </el-col>
            </el-row>
          </div>
        </el-card>

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

    <el-tab-pane :label="$t('settings.promptTemplates')" name="presets">
        <el-card shadow="never" class="studio-ep-surface-card settings-ep-tab-panel">
          <template #header>
            <div class="card-title card-title--split">
              <span class="settings-ep-card-header">
                <el-icon><collection /></el-icon>
                {{ $t('settings.promptTemplates') }}
                <el-text class="settings-ep-title-desc" size="small" type="info">
                  {{ $t('settings.promptTemplatesDesc') }}
                </el-text>
              </span>
              <div class="settings-ep-header-actions">
                <el-button
                  text
                  size="small"
                  class="settings-ep-link-destructive"
                  :loading="restoreConfigBusy"
                  @click="confirmRestorePromptTemplates"
                >
                  {{ $t('settings.restorePromptTemplates') }}
                </el-button>
                <el-button type="primary" size="small" @click="openPresetDialog()">
                  <el-icon><plus /></el-icon>
                  {{ $t('settings.addTemplate') }}
                </el-button>
              </div>
            </div>
          </template>

          <el-empty v-if="Object.keys(presets).length === 0" :description="$t('settings.noTemplates')" />

          <el-table v-else :data="presetList" class="settings-ep-table-full">
            <el-table-column :label="$t('settings.templateName')" prop="name" min-width="150" />
            <el-table-column :label="$t('settings.presetMediaScope')" width="140">
              <template #default="{ row }">
                <el-text size="small">{{ presetMediaLabel(row.preset) }}</el-text>
              </template>
            </el-table-column>
            <el-table-column :label="$t('settings.positivePrompt')" min-width="250">
              <template #default="{ row }">
                <div class="settings-ep-table-ellipsis">
                  {{ row.preset.positive || '-' }}
                </div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('settings.negativePrompt')" min-width="250">
              <template #default="{ row }">
                <div class="settings-ep-table-ellipsis">
                  {{ row.preset.negative || '-' }}
                </div>
              </template>
            </el-table-column>
            <el-table-column :label="$t('settings.presetAppliesTo')" min-width="160">
              <template #default="{ row }">
                <span class="settings-ep-table-muted">{{ presetAppliesSummary(row.preset) }}</span>
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
        </el-card>

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
              <div class="settings-ep-stacked-control">
                <el-radio-group v-model="presetForm.media_scope" class="settings-ep-media-scope-group">
                  <el-radio-button label="image">{{ $t('settings.presetMediaImage') }}</el-radio-button>
                  <el-radio-button label="video">{{ $t('settings.presetMediaVideo') }}</el-radio-button>
                </el-radio-group>
                <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                  {{ $t('settings.presetMediaScopeHint') }}
                </p>
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
              <div class="settings-ep-stacked-control">
                <el-input
                  v-model="presetForm.negative"
                  type="textarea"
                  :rows="2"
                  :placeholder="$t('settings.negativePlaceholder')"
                />
                <p class="studio-ep-field-footnote">{{ $t('studio.optional') }}</p>
              </div>
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

    <el-tab-pane :label="$t('settings.systemConfig')" name="system">
        <el-row :gutter="20" class="settings-ep-system-layout-row">
          <!-- Left: public config + model list + LoRA list -->
          <el-col :xs="24" :md="16" :lg="17" :xl="18">
            <!-- Public config -->
            <el-card shadow="never" class="studio-ep-surface-card settings-ep-tab-panel">
              <template #header>
                <div class="card-title">
                  <el-icon><setting /></el-icon>
                  {{ $t('settings.publicConfig') }}
                  <el-text class="settings-ep-title-desc" size="small" type="info">
                    {{ $t('settings.publicConfigDesc') }}
                  </el-text>
                </div>
              </template>

              <el-form :model="settings" label-position="top" class="settings-ep-system-form">
                <section class="settings-ep-group-block">
                  <h3 class="settings-ep-group-block-title">{{ $t('settings.systemGroupGeneral') }}</h3>
                  <div class="settings-ep-grouped-form settings-ep-grouped-form--system">
                    <el-form-item :label="$t('settings.defaultModel')">
                      <div class="settings-ep-stacked-control">
                        <el-select
                          v-model="settings.default_model"
                          class="settings-ep-default-version-select"
                          :placeholder="$t('settings.selectDefaultModel')"
                        >
                          <el-option
                            v-for="model in installedModels"
                            :key="model.name"
                            :label="model.name"
                            :value="model.name"
                          />
                        </el-select>
                        <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                          {{ $t('settings.defaultModelDesc') }}
                        </p>
                      </div>
                    </el-form-item>

                    <el-form-item :label="$t('settings.language')">
                      <el-select v-model="settings.language" @change="handleLanguageChange">
                        <el-option :label="$t('settings.label_zh')" value="zh" />
                        <el-option :label="$t('settings.label_en')" value="en" />
                      </el-select>
                    </el-form-item>

                    <el-form-item :label="$t('settings.outputFormat')">
                      <el-select v-model="settings.output_format">
                        <el-option label="PNG" value="png" />
                        <el-option label="JPEG" value="jpg" />
                        <el-option label="WebP" value="webp" />
                      </el-select>
                    </el-form-item>
                  </div>
                </section>

                <section class="settings-ep-group-block">
                  <h3 class="settings-ep-group-block-title">{{ $t('settings.systemGroupPerformance') }}</h3>
                  <div class="settings-ep-grouped-form settings-ep-grouped-form--system">
                    <el-form-item :label="$t('settings.memoryLimit')">
                      <div class="param-control-row">
                        <div class="param-slider">
                          <el-slider v-model="settings.mlx_memory_limit" :min="32" :max="256" :step="8" />
                        </div>
                        <span class="settings-ep-slider-suffix">{{ settings.mlx_memory_limit }} GB</span>
                      </div>
                    </el-form-item>

                    <el-form-item :label="$t('settings.modelCacheTtl')">
                      <div class="settings-ep-stacked-control">
                        <div class="param-control-row">
                          <div class="param-slider">
                            <el-slider v-model="settings.model_cache_ttl_minutes" :min="5" :max="120" :step="5" />
                          </div>
                          <span class="settings-ep-slider-suffix">{{ settings.model_cache_ttl_minutes }} min</span>
                        </div>
                        <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                          {{ $t('settings.modelCacheTtlDesc') }}
                        </p>
                      </div>
                    </el-form-item>

                    <el-form-item :label="$t('settings.queueImageFirst')">
                      <div class="settings-ep-stacked-control">
                        <el-switch v-model="settings.queue_image_first" />
                        <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                          {{ $t('settings.queueImageFirstDesc') }}
                        </p>
                      </div>
                    </el-form-item>

                    <el-form-item :label="$t('settings.autoSavePrompts')">
                      <div class="settings-ep-stacked-control">
                        <el-switch v-model="settings.auto_save_prompts" />
                        <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                          {{ $t('settings.autoSavePromptsDesc') }}
                        </p>
                      </div>
                    </el-form-item>
                  </div>
                </section>

                <section class="settings-ep-group-block">
                  <h3 class="settings-ep-group-block-title">{{ $t('settings.systemGroupWorkspace') }}</h3>
                  <div class="settings-ep-grouped-form settings-ep-grouped-form--system">
                    <el-form-item :label="$t('settings.customWorkspace')">
                  <div class="settings-ep-stacked-control settings-ep-workspace-picker">
                    <div class="settings-ep-workspace-input-row">
                      <el-input
                        v-model="settings.custom_workspace_dir"
                        :placeholder="$t('settings.customWorkspacePlaceholder')"
                      />
                      <el-button class="settings-ep-workspace-pick-btn" @click="pickWorkspaceDirectory">
                        {{ $t('settings.pickWorkspace') }}
                      </el-button>
                    </div>
                    <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                      {{ $t('settings.workspaceSetupEmptyHint') }}
                    </p>
                    <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                      {{ $t('settings.customWorkspaceHint') }}
                    </p>
                    <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                      {{ $t('settings.customWorkspaceRestartHint') }}
                    </p>
                    <div v-if="workspacePaths" class="settings-ep-workspace-paths">
                      <div class="settings-ep-workspace-paths-title">{{ $t('settings.workspaceLayoutTitle') }}</div>
                      <ul class="settings-ep-workspace-paths-list">
                        <li v-for="(p, key) in workspacePaths" :key="key">
                          <span class="settings-ep-workspace-paths-key">{{ key }}</span>
                          <span class="settings-ep-workspace-paths-val">{{ p }}</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                    </el-form-item>
                  </div>
                </section>

                <section class="settings-ep-group-block">
                  <h3 class="settings-ep-group-block-title">{{ $t('settings.systemGroupConfigMaintenance') }}</h3>
                  <p class="settings-ep-group-footnote settings-ep-group-footnote--intro">
                    {{ $t('settings.configMaintenanceDesc') }}
                  </p>
                  <div
                    class="settings-ep-grouped-form settings-ep-grouped-form--system settings-ep-grouped-form--action-list"
                    role="group"
                    :aria-label="$t('settings.systemGroupConfigMaintenance')"
                  >
                    <button
                      type="button"
                      class="settings-ep-action-row settings-ep-action-row--destructive"
                      :disabled="restoreConfigBusy"
                      @click="confirmRestoreModelRegistry"
                    >
                      <span class="settings-ep-action-row__label">{{ $t('settings.restoreModelRegistry') }}</span>
                      <el-icon class="settings-ep-action-row__chevron"><arrow-right /></el-icon>
                    </button>
                    <button
                      type="button"
                      class="settings-ep-action-row settings-ep-action-row--destructive"
                      :disabled="restoreConfigBusy"
                      @click="confirmRestorePromptTemplates"
                    >
                      <span class="settings-ep-action-row__label">{{ $t('settings.restorePromptTemplates') }}</span>
                      <el-icon class="settings-ep-action-row__chevron"><arrow-right /></el-icon>
                    </button>
                  </div>
                  <p class="settings-ep-group-footnote">{{ $t('settings.restoreModelRegistryDesc') }}</p>
                  <p class="settings-ep-group-footnote">{{ $t('settings.restorePromptTemplatesDesc') }}</p>
                </section>

                <section class="settings-ep-group-block">
                  <h3 class="settings-ep-group-block-title">{{ $t('settings.systemGroupHuggingface') }}</h3>
                  <div class="settings-ep-grouped-form settings-ep-grouped-form--system">
                    <el-form-item :label="$t('settings.huggingfaceToken')">
                      <div class="settings-ep-stacked-control">
                        <el-input
                          v-model="settings.huggingface_token"
                          type="password"
                          show-password
                          :placeholder="$t('settings.huggingfaceTokenPlaceholder')"
                        >
                          <template #prefix>
                            <el-icon><key /></el-icon>
                          </template>
                        </el-input>
                        <p class="studio-ep-field-footnote">{{ $t('studio.optional') }}</p>
                        <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                          {{ $t('settings.huggingfaceTokenDesc') }}
                        </p>
                      </div>
                    </el-form-item>
                  </div>
                </section>

                <section class="settings-ep-group-block">
                  <h3 class="settings-ep-group-block-title">{{ $t('settings.systemGroupCivitai') }}</h3>
                  <div class="settings-ep-grouped-form settings-ep-grouped-form--system">
                    <el-form-item :label="$t('settings.civitaiToken')">
                      <div class="settings-ep-stacked-control">
                        <el-input
                          v-model="settings.civitai_token"
                          type="password"
                          show-password
                          :placeholder="$t('settings.civitaiTokenPlaceholder')"
                        >
                          <template #prefix>
                            <el-icon><key /></el-icon>
                          </template>
                        </el-input>
                        <p class="studio-ep-field-footnote">{{ $t('studio.optional') }}</p>
                        <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                          {{ $t('settings.civitaiTokenDesc') }}
                        </p>
                      </div>
                    </el-form-item>

                    <el-form-item v-if="settings.civitai_token">
                      <div class="settings-ep-stacked-control">
                        <el-checkbox v-model="settings.nsfw_enabled" size="large">
                          <el-text type="danger">{{ $t('settings.nsfwContent') }}</el-text>
                        </el-checkbox>
                        <p class="settings-ep-form-hint settings-ep-form-hint--below-control">
                          {{ $t('settings.nsfwDesc') }}
                        </p>
                      </div>
                    </el-form-item>
                  </div>
                </section>

                <el-form-item class="settings-ep-system-save-row">
                  <el-button type="primary" @click="saveSettings">
                    <el-icon><check /></el-icon>
                    {{ $t('common.save') }}
                  </el-button>
                </el-form-item>
              </el-form>
            </el-card>

          </el-col>

          <!-- Right: system info + real-time resource monitor -->
          <el-col :xs="24" :md="8" :lg="7" :xl="6">
            <!-- System info -->
            <el-card shadow="never" class="studio-ep-surface-card settings-ep-tab-panel">
              <template #header>
                <div class="card-title">
                  <el-icon><cpu /></el-icon>
                  {{ $t('settings.systemInfo') }}
                </div>
              </template>

              <div class="system-info-grid">
                <div class="info-item">
                  <div class="info-icon">
                    <el-icon class="settings-ep-info-icon-lg"><monitor /></el-icon>
                  </div>
                  <div class="info-content">
                    <div class="info-label">{{ $t('settings.platform') }}</div>
                    <div class="info-value">{{ systemInfo.platform }} {{ systemInfo.architecture }}</div>
                  </div>
                </div>
                <div class="info-item">
                  <div class="info-icon">
                    <el-icon class="settings-ep-info-icon-lg"><cpu /></el-icon>
                  </div>
                  <div class="info-content">
                    <div class="info-label">{{ $t('settings.memory') }}</div>
                    <div class="info-value">{{ systemInfo.memory_gb?.toFixed(1) }} GB</div>
                  </div>
                </div>
                <div class="info-item">
                  <div class="info-icon">
                    <el-icon class="settings-ep-info-icon-lg"><document /></el-icon>
                  </div>
                  <div class="info-content">
                    <div class="info-label">{{ $t('settings.pythonVersion') }}</div>
                    <div class="info-value">{{ systemInfo.python_version }}</div>
                  </div>
                </div>
              </div>

              <div v-if="systemInfo.dependencies" class="settings-ep-dependencies">
                <div class="settings-ep-dependencies-title">{{ $t('settings.dependencies') }}</div>
                <div class="settings-ep-dep-tags">
                  <el-tag v-for="(version, name) in systemInfo.dependencies" :key="name" size="small" type="info" effect="plain">
                    {{ name }} {{ version }}
                  </el-tag>
                </div>
              </div>
            </el-card>

            <el-card shadow="never" class="studio-ep-surface-card settings-ep-tab-panel">
              <template #header>
                <div class="card-title card-title--split">
                  <span class="settings-ep-card-header">
                    <el-icon class="settings-ep-cache-title-icon"><collection /></el-icon>
                    {{ $t('settings.modelCacheTitle') }}
                  </span>
                  <el-button size="small" text @click="refreshCacheStatus" :loading="cacheLoading">
                    <el-icon><refresh /></el-icon>
                  </el-button>
                </div>
              </template>
              <el-alert v-if="cacheError" type="error" :closable="false" :title="cacheError" class="settings-ep-hardware-alert" />
              <template v-else>
                <div v-if="cacheStatus && cacheStatus.cache" class="settings-ep-cache-summary">
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
            </el-card>

            <!-- Real-time resource monitor -->
            <el-card shadow="never" class="studio-ep-surface-card system-monitor-card">
              <template #header>
                <div class="card-title">
                  <el-icon><monitor /></el-icon>
                  {{ $t('settings.resourceMonitor') }}
                  <span class="settings-ep-monitor-sub">
                    {{ $t('settings.realtime') }}
                  </span>
                </div>
              </template>

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
                <div class="settings-ep-monitor-gpu-meta">
                  <span v-if="monitorData.gpu.memory_gb">{{ monitorData.gpu.memory_gb }} {{ $t('settings.unifiedMemory') }}</span>
                  <span v-if="monitorData.gpu.note">{{ monitorData.gpu.note }}</span>
                </div>
              </div>

              <div class="settings-ep-monitor-foot">
                <el-tag size="small" type="info" effect="plain">
                  {{ $t('settings.refreshInterval') }}
                </el-tag>
              </div>
            </el-card>
          </el-col>
        </el-row>
    </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, watch, onMounted, onUnmounted, inject, type Ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { ElMessage, ElMessageBox } from 'element-plus';
import { api } from '@/utils/api';
import { $tt, $mn, $md } from '@/utils/i18n';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { useRegistryStore } from '@/stores/registry';
import type { SystemInfo } from '@/types';
import * as RegistryParamSchema from '@/utils/registryParamSchema';
import ModelLicenseBadges from '@/components/model/ModelLicenseBadges.vue';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ModelConfig {
  name?: string | { zh?: string; en?: string };
  description?: string | { zh?: string; en?: string };
  name_en?: string;
  description_en?: string;
  recommended?: boolean;
  commercial_use_allowed?: boolean | null;
  engine?: string;
  category?: string;
  type?: string;
  media?: string;
  actions?: Record<string, unknown>;
  parameters?: Record<string, unknown>;
  versions?: Record<string, { name?: string; size?: string; default?: boolean; source_type?: string; from_version?: string }>;
  [key: string]: unknown;
}

interface ParamPreset {
  id: string;
  modelKey: string;
  name: string;
  params: Record<string, unknown>;
  isDefault: boolean;
  createdAt: string;
}

interface MonitorData {
  cpu_percent: number;
  memory: {
    total_gb: number;
    used_gb: number;
    percent: number;
  };
  gpu: null | {
    model?: string;
    memory_gb?: number;
    note?: string;
  };
}

/* ------------------------------------------------------------------ */
/*  Injected / External                                                */
/* ------------------------------------------------------------------ */

const systemInfo = inject<Ref<SystemInfo>>('systemInfo');

const registryStore = useRegistryStore();
const { locale } = useI18n();

/* ------------------------------------------------------------------ */
/*  State                                                              */
/* ------------------------------------------------------------------ */

const activeTab = ref<string>(getItem(DQ_STORAGE.SETTINGS_TAB) || 'models');

const settings = reactive<Record<string, unknown>>({
  language: 'zh',
  default_model: '',
  auto_save_prompts: true,
  output_format: 'png',
  mlx_memory_limit: 120,
  model_cache_ttl_minutes: 30,
  queue_image_first: false,
  civitai_token: '',
  huggingface_token: '',
  nsfw_enabled: false,
  custom_workspace_dir: '',
});

const workspacePaths = ref<Record<string, string> | null>(null);
const initialWorkspaceDir = ref('');
const workspaceEffectiveRoot = ref('');

const modelRegistry = ref<Record<string, ModelConfig>>({});
const selectedModel = ref('');
const modelsStatus = ref<Record<string, unknown>>({});
const modelsDetailedStatus = ref<Record<string, { versions?: Record<string, { ready?: boolean }> }>>({});
const selectedDefaultVersion = ref('');
const paramPresets = ref<ParamPreset[]>([]);
const paramPresetDialogVisible = ref(false);
const paramPresetForm = reactive({ name: '', isDefault: false });
const paramDefaults = reactive<Record<string, unknown>>({});

const modelParams = reactive<Record<string, unknown>>({
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

const settingsCompatibleLoras = ref<Record<string, unknown>[]>([]);
const installedModels = ref<Record<string, unknown>[]>([]);

const presets = ref<Record<string, Record<string, unknown>>>({});
const presetDialogVisible = ref(false);
const editingPresetName = ref('');
const presetForm = reactive({
  name: '',
  positive: '',
  negative: '',
  media_scope: 'image',
  applies_to: ['create'],
});

const monitorData = reactive<MonitorData>({
  cpu_percent: 0,
  memory: { total_gb: 0, used_gb: 0, percent: 0 },
  gpu: null,
});
let monitorInterval: ReturnType<typeof setInterval> | null = null;

const cacheStatus = reactive<{ cache: Record<string, unknown> | null; mlx: Record<string, unknown> }>({
  cache: null,
  mlx: {},
});
const cacheLoading = ref(false);
const cacheError = ref('');
const restoreConfigBusy = ref(false);

/* ------------------------------------------------------------------ */
/*  Computed                                                           */
/* ------------------------------------------------------------------ */

const sortedModelRegistry = computed(() => {
  const entries = Object.entries(modelRegistry.value);
  entries.sort((a, b) => {
    if (a[1].recommended !== b[1].recommended) return a[1].recommended ? -1 : 1;
    const nameA = $mn(a[1], a[0]);
    const nameB = $mn(b[1], b[0]);
    return nameA.localeCompare(nameB);
  });
  return Object.fromEntries(entries);
});

const currentModelConfig = computed<ModelConfig | null>(() => {
  return modelRegistry.value[selectedModel.value] || null;
});

const presetList = computed(() => {
  return Object.entries(presets.value).map(([name, preset]) => ({
    name,
    preset,
  }));
});

const presetAppliesSummary = (preset: Record<string, unknown>) =>
  (preset.applies_to as string[]).join(', ');

const presetMediaLabel = (preset: Record<string, unknown>) =>
  preset.media_scope === 'video' ? $tt('settings.presetMediaVideo') : $tt('settings.presetMediaImage');

const versionCount = (config: ModelConfig) => {
  return config.versions ? Object.keys(config.versions).length : 0;
};

const installedVersionCount = (modelId: string) => {
  const detail = modelsDetailedStatus.value[modelId];
  if (!detail || !detail.versions) return 0;
  return Object.values(detail.versions).filter((v) => v.ready).length;
};

const versionStatus = (modelId: string, verKey: string): string => {
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

const versionStatusType = (modelId: string, verKey: string) => {
  const s = versionStatus(modelId, verKey);
  const map: Record<string, string> = { ready: 'success', generatable: 'info', parent_missing: 'info', missing: 'info' };
  return map[s] || 'info';
};

const versionStatusLabel = (modelId: string, verKey: string) => {
  const s = versionStatus(modelId, verKey);
  const map: Record<string, string> = {
    ready: $tt('settings.installed'),
    generatable: $tt('settings.canGenerate'),
    parent_missing: $tt('settings.waitingParent'),
    missing: $tt('settings.notInstalled'),
  };
  return map[s] || s;
};

const parseSizeGB = (sizeStr: string | undefined) => {
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
  const memoryGB = systemInfo?.value.memory_gb ?? 0;
  const ratio = memoryGB ? minVersionSizeGB.value / memoryGB : 0;
  if (ratio < 0.5) return 'var(--el-color-success)';
  if (ratio < 0.8) return 'var(--el-color-warning)';
  return 'var(--el-color-danger)';
});

const isRecommendedVersion = (verKey: string) => {
  const cfg = currentModelConfig.value;
  if (!cfg || !cfg.versions || !systemInfo?.value.memory_gb) return false;
  const ver = cfg.versions[verKey];
  if (!ver) return false;
  const sizeGB = parseSizeGB(ver.size);
  const memoryGB = systemInfo.value.memory_gb;
  if (sizeGB > memoryGB * 1.2) return false;
  const status = versionStatus(selectedModel.value, verKey);
  if (status === 'ready') return true;
  const allReady = Object.keys(cfg.versions).filter((k) => versionStatus(selectedModel.value, k) === 'ready');
  if (allReady.length === 0) {
    const installable = Object.entries(cfg.versions)
      .filter(([, v]) => {
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
  const ready = candidates.find(([k]) => versionStatus(selectedModel.value, k) === 'ready');
  if (ready) return { key: ready[0], ...ready[1] };
  return { key: candidates[0][0], ...candidates[0][1] };
});

const hardwareAdvice = computed(() => {
  const cfg = currentModelConfig.value;
  if (!cfg || !cfg.versions || !systemInfo?.value.memory_gb) return null;
  const memoryGB = systemInfo.value.memory_gb;
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

const hardwareAdviceAlertType = computed(() => {
  const h = hardwareAdvice.value;
  if (!h) return 'info';
  if (h.icon === 'warning') return 'warning';
  if (h.icon === 'check') return 'success';
  return 'info';
});

const capabilityList = computed(() => {
  const cfg = currentModelConfig.value;
  if (!cfg || !cfg.parameters) return [];
  const caps: { key: string; label: string; value: boolean }[] = [];
  const params = cfg.parameters;
  const labels: Record<string, string> = {
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

const paramPresetsForModel = computed(() => {
  const mk = selectedModel.value;
  if (!mk) return [];
  return paramPresets.value.filter((p) => p.modelKey === mk);
});

const normalizedParams = computed(() => {
  const cfg = currentModelConfig.value;
  if (!cfg || !cfg.parameters) return {};
  return RegistryParamSchema.normalizeParamsDef(cfg.parameters as Record<string, unknown>);
});

const resPair = computed(() => RegistryParamSchema.resolutionPair(normalizedParams.value));

/** el-select 用 === 匹配 value：预设/存储里的字符串需对齐到 registry 枚举的原始类型，否则选中项不显示 */
const snapModelResolutionEnums = () => {
  const pair = resPair.value;
  if (!pair) return;
  const snap = (key: 'width' | 'height', spec: Record<string, unknown>) => {
    const opts = spec.options;
    if (!Array.isArray(opts) || opts.length === 0) return;
    const cur = modelParams[key];
    const found = opts.find(
      (o) => o === cur || String(o) === String(cur) || (Number(o) === Number(cur) && !Number.isNaN(Number(cur)))
    );
    if (found !== undefined) modelParams[key] = found;
    else if ('default' in spec) modelParams[key] = (spec as { default?: unknown }).default;
  };
  snap('width', pair.width);
  snap('height', pair.height);
};

const scalarKeys = computed(() => RegistryParamSchema.scalarKeysForForm(normalizedParams.value));

const seedSupport = computed(() => {
  const cfg = currentModelConfig.value;
  return !!(cfg && cfg.parameters && (cfg.parameters.seed_support as boolean));
});

const showLoraBlock = computed(() => {
  const p = currentModelConfig.value && currentModelConfig.value.parameters;
  if (!p || !p.lora_support) return false;
  return Array.isArray(settingsCompatibleLoras.value);
});

const adapterItems = computed(() => {
  if (!Array.isArray(settingsCompatibleLoras.value)) return [];
  return settingsCompatibleLoras.value.map((l) => ({ kind: 'lora', id: String((l as Record<string, unknown>).id), name: String((l as Record<string, unknown>).name) }));
});

const loraScaleSpec = computed(() => {
  const s = normalizedParams.value.lora_scale;
  if (s && (s.type === 'int' || s.type === 'float')) {
    return { min: (s.min as number) ?? 0, max: (s.max as number) ?? 2, step: (s.step as number) ?? 0.1 };
  }
  return { min: 0, max: 2, step: 0.1 };
});

const specOf = (key: string) => normalizedParams.value[key];

const numStep = (key: string, spec: Record<string, unknown>) => {
  if (typeof spec.step === 'number') return spec.step;
  return spec.type === 'int' ? 1 : 0.1;
};

const paramLabel = (key: string, spec: Record<string, unknown>) => {
  const map: Record<string, string> = {
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
      return $tt(i18nKey);
    } catch {
      /* fall through */
    }
  }
  if (spec && spec.label) return String(spec.label);
  return key;
};

const isParamChanged = (key: string) => {
  const spec = specOf(key);
  if (!spec || !('default' in spec)) return false;
  return modelParams[key] !== spec.default;
};

const resetParam = (key: string) => {
  const spec = specOf(key);
  if (spec && 'default' in spec) {
    modelParams[key] = spec.default;
  }
};

const paramNotesList = computed(() => {
  const cfg = currentModelConfig.value;
  if (!cfg || !cfg.parameters) return [];
  const list: { key: string; label: string; note: string }[] = [];
  for (const [key, spec] of Object.entries(cfg.parameters)) {
    if (typeof spec !== 'object' || spec === null) continue;
    const s = spec as Record<string, unknown>;
    if (!s.note) continue;
    if (s.type === 'bool' && String(key).endsWith('_support')) continue;
    list.push({ key, label: paramLabel(key, s), note: String(s.note) });
  }
  return list;
});

const hasParamNotes = computed(() => paramNotesList.value.length > 0);

const settingsLorasForForm = computed(() => {
  const c = currentModelConfig.value;
  if (!c || !c.parameters || !c.parameters.lora_support) return null;
  return settingsCompatibleLoras.value;
});

const modelActionKeyList = computed(() => {
  const cfg = currentModelConfig.value;
  if (!cfg || !cfg.actions) return [];
  return Object.keys(cfg.actions).filter((k) => cfg.actions![k] != null);
});

const actionTagLabel = (key: string) => {
  const cfg = currentModelConfig.value;
  const media = cfg && cfg.media != null ? String(cfg.media) : '';
  if (key === 'animate') {
    return $tt('action.video.animate');
  }
  if (key === 'create' && media === 'video') {
    return $tt('action.video.create');
  }
  const imageKeys = new Set(['create', 'rewrite', 'retouch', 'extend', 'upscale']);
  if (imageKeys.has(key)) {
    return $tt('action.image.' + key);
  }
  return $tt('settings.actionTags.' + key);
};

const categoryLabel = (cat: string) => {
  const map: Record<string, string> = {
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

const modelInitials = (config: ModelConfig) => {
  const name = $mn(config, '');
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

const getProgressColor = (percent: number) => {
  if (percent < 50) return 'var(--el-color-success)';
  if (percent < 80) return 'var(--el-color-warning)';
  return 'var(--el-color-danger)';
};

/* ------------------------------------------------------------------ */
/*  Methods                                                            */
/* ------------------------------------------------------------------ */

// ----- Tab persistence -----
watch(activeTab, (newVal) => {
  setItem(DQ_STORAGE.SETTINGS_TAB, newVal);
  if (newVal === 'system') {
    refreshCacheStatus();
  }
});

// ----- Language / Theme sync -----
watch(
  () => settings.language as string,
  (newVal) => {
    if (locale.value !== newVal) {
      locale.value = newVal;
      setItem(DQ_STORAGE.LANG, newVal);
      document.documentElement.lang = newVal;
    }
  }
);

// ----- Data loading -----

const loadModelRegistry = async () => {
  try {
    const [registryData, statusData, detailedStatusData] = await Promise.all([
      api.settings.getModelRegistry(),
      api.settings.getModelsStatus(),
      api.settings.getModelsDetailedStatus(),
    ]);
    modelRegistry.value = (registryData as { models?: Record<string, ModelConfig> }).models || {};
    modelsStatus.value = (statusData as Record<string, unknown>) || {};
    modelsDetailedStatus.value = (detailedStatusData as Record<string, { versions?: Record<string, { ready?: boolean }> }>) || {};

    if (!selectedModel.value || !modelRegistry.value[selectedModel.value]) {
      const recommended = Object.entries(modelRegistry.value).find(([, val]) => val.recommended);
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
    settingsCompatibleLoras.value = (list as Record<string, unknown>[]) || [];
  } catch (e) {
    console.error('Failed to load compatible loras (settings):', e);
    settingsCompatibleLoras.value = [];
  }
};

const onModelSelect = () => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return;

  RegistryParamSchema.applyDefaults(config.parameters, modelParams);

  Object.keys(paramDefaults).forEach((k) => delete paramDefaults[k]);
  for (const [key, spec] of Object.entries(config.parameters || {})) {
    if (typeof spec === 'object' && spec !== null && 'default' in spec) {
      paramDefaults[key] = (spec as Record<string, unknown>).default;
    }
  }

  if (config.versions) {
    const defaultVer = Object.entries(config.versions).find(([, v]) => v.default);
    if (defaultVer) {
      selectedDefaultVersion.value = defaultVer[0];
    } else {
      selectedDefaultVersion.value = Object.keys(config.versions)[0] || '';
    }
    if (recommendedVersion.value) {
      selectedDefaultVersion.value = recommendedVersion.value.key;
    }
  } else {
    selectedDefaultVersion.value = '';
  }

  const defaultPreset = paramPresetsForModel.value.find((p) => p.isDefault);
  if (defaultPreset) {
    loadParamPreset(defaultPreset);
  }

  snapModelResolutionEnums();
  loadSettingsCompatibleLoras();
};

const selectDefaultVersionKey = (verKey: string) => {
  selectedDefaultVersion.value = verKey;
  onDefaultVersionChange();
};

const onDefaultVersionChange = () => {
  console.log('Default version changed to:', selectedDefaultVersion.value);
};

const onSettingsModelRestoreDefaults = () => {
  const config = currentModelConfig.value;
  if (config && config.parameters) {
    RegistryParamSchema.applyDefaults(config.parameters, modelParams);
    snapModelResolutionEnums();
    ElMessage.success($tt('studio.restoredDefaults'));
  }
};

const saveModelConfig = async () => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return;

  try {
    const params: Record<string, unknown> = {};
    for (const [key, spec] of Object.entries(config.parameters)) {
      if (typeof spec !== 'object' || spec === null || !Object.prototype.hasOwnProperty.call(spec, 'default')) {
        continue;
      }
      const s = spec as Record<string, unknown>;
      if (s.type === 'bool' && String(key).endsWith('_support')) continue;
      if (!Object.prototype.hasOwnProperty.call(modelParams, key)) continue;
      params[key] = modelParams[key];
    }

    const result = await api.settings.updateModelParameters(selectedModel.value, params) as { success?: boolean; error?: string };
    if (result.success) {
      ElMessage.success($tt('settings.modelConfigSaved'));
      await loadModelRegistry();
      await registryStore.load();
    } else {
      ElMessage.error(result.error || $tt('settings.saveFailed'));
    }
  } catch (e) {
    console.error('Failed to save model config:', e);
    ElMessage.error($tt('settings.saveFailed'));
  }
};

// ----- Settings -----

function extractApiError(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const err = e as { response?: { data?: { detail?: string } } };
    if (err.response?.data?.detail) {
      return err.response.data.detail;
    }
  }
  if (e instanceof Error) return e.message;
  return String(e);
}

async function confirmWorkspaceRelocation(fromPath: string, toPath: string): Promise<boolean> {
  try {
    await ElMessageBox.confirm(
      $tt('settings.workspaceChangeConfirm', { from: fromPath, to: toPath }),
      $tt('settings.workspaceChangeTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.workspaceChangeContinue'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    await ElMessageBox.confirm(
      $tt('settings.workspaceChangeConfirmFinal'),
      $tt('settings.workspaceChangeTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.workspaceChangeContinue'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    return true;
  } catch {
    return false;
  }
}

const loadSettings = async () => {
  try {
    const data = await api.settings.getSettings();
    Object.assign(settings, data);
    delete settings.theme;
    delete settings.custom_models_dir;
    delete settings.custom_loras_dir;
    delete settings.custom_outputs_dir;
    initialWorkspaceDir.value = String(data.custom_workspace_dir || '').trim();
    if (data.language) {
      locale.value = data.language;
      settings.language = data.language;
      setItem(DQ_STORAGE.LANG, data.language);
      document.documentElement.lang = data.language;
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
};

const loadWorkspacePaths = async () => {
  try {
    const [paths, status] = await Promise.all([
      api.settings.getWorkspacePaths(),
      api.settings.getWorkspaceStatus(),
    ]);
    workspacePaths.value = paths;
    workspaceEffectiveRoot.value = status.effective_root || '';
  } catch (e) {
    workspacePaths.value = null;
  }
};

const pickWorkspaceDirectory = async () => {
  try {
    const { path } = await api.settings.pickWorkspaceDirectory();
    if (path) {
      settings.custom_workspace_dir = path;
    }
  } catch (e) {
    ElMessage.error((e as Error).message || String(e));
  }
};

const saveSettings = async () => {
  const newWs = String(settings.custom_workspace_dir || '').trim();
  const oldWs = initialWorkspaceDir.value;

  if (newWs !== oldWs) {
    if (!newWs) {
      ElMessage.warning($tt('settings.workspaceRequired'));
      settings.custom_workspace_dir = oldWs;
      return;
    }
    const fromLabel = oldWs || workspaceEffectiveRoot.value || $tt('settings.workspaceLayoutTitle');
    const ok = await confirmWorkspaceRelocation(fromLabel, newWs);
    if (!ok) {
      settings.custom_workspace_dir = oldWs;
      return;
    }
    try {
      const wsRes = await api.settings.applyWorkspace(newWs);
      initialWorkspaceDir.value = String(wsRes.workspace || newWs).trim();
      settings.custom_workspace_dir = initialWorkspaceDir.value;
      if (wsRes.restart_required) {
        ElMessage.warning($tt('settings.customWorkspaceRestartHint'));
      }
      await loadWorkspacePaths();
    } catch (e) {
      settings.custom_workspace_dir = oldWs;
      ElMessage.error(extractApiError(e) || $tt('settings.workspaceApplyFailed'));
      return;
    }
  }

  try {
    await api.settings.updateSettings({
      language: settings.language,
      output_format: settings.output_format,
      mlx_memory_limit: settings.mlx_memory_limit,
      model_cache_ttl_minutes: settings.model_cache_ttl_minutes,
      queue_image_first: settings.queue_image_first,
      auto_save_prompts: settings.auto_save_prompts,
      default_model: settings.default_model,
      civitai_token: settings.civitai_token || '',
      huggingface_token: settings.huggingface_token || '',
      nsfw_enabled: settings.nsfw_enabled,
    });
    ElMessage.success($tt('settings.saved'));
  } catch (e) {
    ElMessage.error($tt('settings.saveFailed'));
  }
};

const loadInstalledModels = async () => {
  try {
    const models = await api.settings.listModels();
    installedModels.value = (models as Record<string, unknown>[]) || [];
  } catch (e) {
    console.error('Failed to load models:', e);
  }
};

const handleLanguageChange = (lang: string) => {
  settings.language = lang;
};

// ----- Config file restore (factory defaults) -----

async function runRestoreConfigDefaults(files: string[]) {
  if (restoreConfigBusy.value) return;
  restoreConfigBusy.value = true;
  try {
    const res = await api.settings.restoreConfigDefaults(files);
    const restored = res.restored || [];
    if (!restored.length) {
      ElMessage.warning($tt('settings.restoreConfigNothing'));
      return;
    }
    if (restored.includes('presets.json')) {
      await loadPresets();
    }
    if (restored.includes('models_registry.json')) {
      await loadModelRegistry();
      await registryStore.load(true);
    }
    ElMessage.success($tt('settings.restoreConfigSuccess'));
    if (res.restart_required) {
      ElMessage.warning($tt('settings.restoreRegistryRestartHint'));
    }
  } catch (e) {
    ElMessage.error(extractApiError(e) || $tt('settings.restoreConfigFailed'));
  } finally {
    restoreConfigBusy.value = false;
  }
}

async function confirmRestoreModelRegistry() {
  try {
    await ElMessageBox.confirm(
      $tt('settings.restoreModelRegistryConfirm'),
      $tt('settings.restoreModelRegistryTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.restoreConfirm'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    await runRestoreConfigDefaults(['models_registry.json']);
  } catch {
    /* cancelled */
  }
}

async function confirmRestorePromptTemplates() {
  try {
    await ElMessageBox.confirm(
      $tt('settings.restorePromptTemplatesConfirm'),
      $tt('settings.restorePromptTemplatesTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.restoreConfirm'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    await runRestoreConfigDefaults(['presets.json']);
  } catch {
    /* cancelled */
  }
}

// ----- Presets (prompt templates) -----

const loadPresets = async () => {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as Record<string, Record<string, unknown>>) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
  }
};

const openPresetDialog = (name = '', preset: Record<string, unknown> | null = null) => {
  editingPresetName.value = name;
  if (name && preset) {
    presetForm.name = name;
    presetForm.positive = String(preset.positive || '');
    presetForm.negative = String(preset.negative || '');
    presetForm.applies_to = [...(preset.applies_to as string[] || [])];
    presetForm.media_scope = String(preset.media_scope || 'image');
  } else {
    presetForm.name = '';
    presetForm.positive = '';
    presetForm.negative = '';
    presetForm.media_scope = 'image';
    presetForm.applies_to = ['create'];
  }
  presetDialogVisible.value = true;
};

const savePreset = async () => {
  if (!presetForm.name.trim()) {
    ElMessage.warning($tt('settings.enterTemplateName'));
    return;
  }
  try {
    if (!Array.isArray(presetForm.applies_to) || !presetForm.applies_to.length) {
      ElMessage.warning($tt('settings.presetAppliesRequired'));
      return;
    }
    const applies = [...presetForm.applies_to];
    await api.settings.savePreset(presetForm.name.trim(), {
      positive: presetForm.positive,
      negative: presetForm.negative,
      media_scope: presetForm.media_scope,
      applies_to: applies,
    });
    ElMessage.success($tt('settings.templateSaved'));
    presetDialogVisible.value = false;
    await loadPresets();
  } catch (e) {
    console.error('Failed to save preset:', e);
    ElMessage.error($tt('settings.saveFailed'));
  }
};

const confirmDeletePreset = (name: string) => {
  ElMessageBox.confirm(
    $tt('settings.deletePresetConfirm', { name }),
    $tt('settings.deletePresetTitle'),
    {
      confirmButtonText: $tt('settings.deleteConfirm'),
      cancelButtonText: $tt('settings.deleteCancel'),
      type: 'warning',
    }
  ).then(() => {
    deletePreset(name);
  }).catch(() => {});
};

const deletePreset = async (name: string) => {
  try {
    await api.settings.deletePreset(name);
    ElMessage.success($tt('settings.templateDeleted'));
    await loadPresets();
  } catch (e) {
    console.error('Failed to delete preset:', e);
    ElMessage.error($tt('settings.saveFailed'));
  }
};

// ----- Parameter presets -----

const loadParamPresets = () => {
  try {
    const raw = localStorage.getItem('dq_param_presets_v1');
    paramPresets.value = raw ? JSON.parse(raw) : [];
  } catch {
    paramPresets.value = [];
  }
};

const saveParamPresetsToStorage = () => {
  localStorage.setItem('dq_param_presets_v1', JSON.stringify(paramPresets.value));
};

const openParamPresetDialog = () => {
  paramPresetForm.name = '';
  paramPresetForm.isDefault = false;
  paramPresetDialogVisible.value = true;
};

const saveParamPreset = () => {
  if (!paramPresetForm.name.trim()) {
    ElMessage.warning($tt('settings.enterPresetName'));
    return;
  }
  const mk = selectedModel.value;
  if (!mk) return;
  const presetParams: Record<string, unknown> = {};
  const cfg = currentModelConfig.value;
  if (cfg && cfg.parameters) {
    for (const [key, spec] of Object.entries(cfg.parameters)) {
      if (typeof spec !== 'object' || spec === null || !Object.prototype.hasOwnProperty.call(spec, 'default')) continue;
      const s = spec as Record<string, unknown>;
      if (s.type === 'bool' && String(key).endsWith('_support')) continue;
      if (!Object.prototype.hasOwnProperty.call(modelParams, key)) continue;
      presetParams[key] = modelParams[key];
    }
  }
  if (paramPresetForm.isDefault) {
    paramPresets.value.forEach((p) => {
      if (p.modelKey === mk) p.isDefault = false;
    });
  }
  paramPresets.value.push({
    id: Date.now().toString(36) + Math.random().toString(36).substring(2),
    modelKey: mk,
    name: paramPresetForm.name.trim(),
    params: presetParams,
    isDefault: paramPresetForm.isDefault,
    createdAt: new Date().toISOString(),
  });
  saveParamPresetsToStorage();
  paramPresetDialogVisible.value = false;
  ElMessage.success($tt('settings.presetSaved'));
};

const loadParamPreset = (preset: ParamPreset) => {
  if (!preset || !preset.params) return;
  Object.assign(modelParams, preset.params);
  snapModelResolutionEnums();
  ElMessage.success($tt('settings.presetLoaded', { name: preset.name }));
};

const deleteParamPreset = (id: string) => {
  const idx = paramPresets.value.findIndex((p) => p.id === id);
  if (idx !== -1) {
    paramPresets.value.splice(idx, 1);
    saveParamPresetsToStorage();
  }
};

// ----- Monitor / Cache -----

const loadMonitorData = async () => {
  try {
    const data = await api.settings.getSystemMonitor();
    Object.assign(monitorData, data);
  } catch (e) {
    console.error('Failed to load monitor data:', e);
  }
};

const startMonitor = () => {
  loadMonitorData();
  monitorInterval = setInterval(loadMonitorData, 3000);
};

const stopMonitor = () => {
  if (monitorInterval) {
    clearInterval(monitorInterval);
    monitorInterval = null;
  }
};

const refreshCacheStatus = async () => {
  cacheLoading.value = true;
  cacheError.value = '';
  try {
    const data = await api.system.getCacheStatus();
    cacheStatus.cache = (data as { cache?: Record<string, unknown> }).cache || null;
    cacheStatus.mlx = (data as { mlx?: Record<string, unknown> }).mlx || {};
  } catch (e: unknown) {
    cacheError.value = e instanceof Error ? e.message : String(e);
  } finally {
    cacheLoading.value = false;
  }
};

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

onMounted(() => {
  loadModelRegistry();
  loadSettings();
  loadWorkspacePaths();
  loadInstalledModels();
  loadPresets();
  loadParamPresets();
  startMonitor();
  refreshCacheStatus();
});

onUnmounted(() => {
  stopMonitor();
});
</script>
