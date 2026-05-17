<!-- @ts-nocheck -->
<template>
  <div class="settings-page settings-page--macos">
    <DqSectionTabs v-model="activeTab" class="settings-dq-section-tabs">
      <template #tabs>
        <DqSectionTabTrigger name="models" :label="$t('settings.modelConfig')" />
        <DqSectionTabTrigger name="presets" :label="$t('settings.promptTemplates')" />
        <DqSectionTabTrigger name="system" :label="$t('settings.systemConfig')" />
      </template>
      <DqSectionTabPanel name="models">
        <div class="settings-model-tab settings-model-tab--macos">
          <section class="settings-group-block settings-group-block--model-picker">
            <h3 class="settings-group-block-title">{{ $t('settings.modelConfig') }}</h3>
            <p class="settings-group-footnote settings-group-footnote--intro">
              {{ $t('settings.modelConfigDesc') }}
            </p>
            <DqPrefPane class="settings-grouped-form settings-grouped-form--model settings-grouped-form--mac-rows settings-pref-pane-form">
              <DqPrefRow :label="$t('studio.modelSettings')" class="settings-mac-row--chevron">
                <DqSelect
                  v-model="selectedModel"
                  class="settings-model-picker settings-model-picker--inline"
                  @change="onModelSelect"
                  filterable
                >
            <DqOption
              v-for="(config, key) in sortedModelRegistry"
              :key="key"
              :label="$mn(config)"
              :value="key"
            >
              <div class="settings-select-option">
                <span class="settings-select-option__name">{{ $mn(config) }}</span>
                <ModelLicenseBadges
                  :recommended="config.recommended"
                  :commercial-use-allowed="config.commercial_use_allowed"
                />
                <DqTag size="small" type="info">{{ config.engine }}</DqTag>
                <DqTag v-if="config.category" size="small" type="warning">{{ categoryLabel(config.category) }}</DqTag>
                <span class="settings-select-option__meta">
                  {{ installedVersionCount(key) }}/{{ versionCount(config) }} {{ $t('settings.versionsInstalled') }}
                </span>
              </div>
            </DqOption>
                </DqSelect>
              </DqPrefRow>
            </DqPrefPane>
          </section>

          <template v-if="currentModelConfig">
            <section class="settings-group-block settings-model-hero-block">
              <div class="settings-grouped-form settings-grouped-form--model settings-model-hero-panel">
              <div class="settings-overview settings-overview--inset">
                <div class="settings-overview__avatar">
                  {{ modelInitials(currentModelConfig) }}
                </div>
                <div class="settings-overview__body">
                  <div class="settings-overview__title-row">
                    <span class="settings-overview__title">{{ $mn(currentModelConfig) }}</span>
                    <ModelLicenseBadges
                      :recommended="currentModelConfig.recommended"
                      :commercial-use-allowed="currentModelConfig.commercial_use_allowed"
                    />
                  </div>
                  <p class="settings-overview__desc">{{ $md(currentModelConfig) }}</p>
                  <div class="settings-overview__tags">
                    <DqTag size="small" type="info">{{ currentModelConfig.engine }}</DqTag>
                    <DqTag v-if="currentModelConfig.category" size="small" type="warning">{{ categoryLabel(currentModelConfig.category) }}</DqTag>
                    <DqTag size="small" type="info" effect="plain">{{ currentModelConfig.type }}</DqTag>
                    <DqTag
                      v-for="ak in modelActionKeyList"
                      :key="ak"
                      size="small"
                      effect="plain"
                    >
                      {{ actionTagLabel(ak) }}
                    </DqTag>
                  </div>
                </div>
              </div>
              </div>
            </section>

            <DqRow :gutter="20" class="settings-layout-row settings-model-layout-row">
              <DqCol :xs="24" :md="16" :lg="17" :xl="17" class="settings-model-main">
                <section v-if="currentModelConfig.versions" class="settings-group-block">
                  <h3 class="settings-group-block-title">{{ $t('settings.defaultVersion') }}</h3>
                  <DqPrefPane class="settings-grouped-form settings-grouped-form--model settings-grouped-form--mac-rows settings-pref-pane-form">
                    <DqPrefRow no-label class="settings-mac-row--chevron">
                      <DqSelect v-model="selectedDefaultVersion" class="settings-default-version-select settings-mac-chevron-select" @change="onDefaultVersionChange">
                    <DqOption
                      v-for="(ver, verKey) in currentModelConfig.versions"
                      :key="verKey"
                      :label="String(ver.name || verKey)"
                      :value="verKey"
                    >
                      <div class="settings-select-option">
                        <span>{{ ver.name }}</span>
                        <DqTag size="small" type="info">{{ ver.size }}</DqTag>
                        <DqTag
                          v-if="versionStatus(selectedModel, verKey) === 'ready'"
                          size="small"
                          type="success"
                        >{{ $t('settings.installed') }}</DqTag>
                        <DqTag
                          v-else-if="versionStatus(selectedModel, verKey) === 'generatable'"
                          size="small"
                          type="info"
                          effect="plain"
                        >{{ versionStatusLabel(selectedModel, verKey) }}</DqTag>
                        <DqTag
                          v-else
                          size="small"
                          type="info"
                        >{{ $t('settings.notInstalled') }}</DqTag>
                        <span
                          v-if="isRecommendedVersion(verKey)"
                          class="settings-version-option-rec"
                        >{{ $t('settings.recommendedForYourHardware') }}</span>
                      </div>
                    </DqOption>
                      </DqSelect>
                    </DqPrefRow>
                  </DqPrefPane>
                  <DqAlert
                    v-if="hardwareAdvice"
                    class="settings-hardware-alert settings-inset-alert"
                    :type="hardwareAdviceAlertType"
                    :closable="false"
                    show-icon
                  >
                    {{ hardwareAdvice.message }}
                  </DqAlert>
                </section>

                <section class="settings-group-block">
                  <div class="settings-group-block-title-row">
                    <h3 class="settings-group-block-title">{{ $t('settings.paramPresets') }}</h3>
                    <DqButton type="primary" size="sm" class="settings-tool-btn" @click="openParamPresetDialog">
                      <DqIcon><plus /></DqIcon>
                      {{ $t('settings.saveAsPreset') }}
                    </DqButton>
                  </div>
                  <div
                    v-if="paramPresetsForModel.length === 0"
                    class="settings-grouped-form settings-grouped-form--model settings-grouped-form--empty-hint"
                  >
                    <p class="settings-empty-hint">{{ $t('settings.noParamPresets') }}</p>
                  </div>
                  <div v-else class="settings-preset-tags settings-preset-tags--inset">
                    <DqTag
                      v-for="preset in paramPresetsForModel"
                      :key="preset.id"
                      size="small"
                      effect="plain"
                      closable
                      class="settings-preset-tag"
                      @close="deleteParamPreset(preset.id)"
                      @click="loadParamPreset(preset)"
                      :type="preset.isDefault ? 'success' : ''"
                    >
                      {{ preset.name }}
                      <span v-if="preset.isDefault" class="settings-preset-tag-default-note">({{ $t('settings.default') }})</span>
                    </DqTag>
                  </div>
                </section>

                <section class="settings-group-block model-params-section">
                  <div class="settings-group-block-title-row">
                    <h3 class="settings-group-block-title">{{ $t('settings.parameters') }}</h3>
                    <DqButton type="text" size="sm" class="settings-restore-btn" @click="onSettingsModelRestoreDefaults">
                      <DqIcon><refresh /></DqIcon>
                      {{ $t('studio.restoreDefaults') }}
                    </DqButton>
                  </div>

                  <DqPrefPane
                    v-if="currentModelConfig"
                    class="settings-grouped-form settings-grouped-form--model settings-model-params-form settings-pref-pane-form"
                  >
                    <!-- Resolution -->
                    <DqPrefRow v-if="resPair" :label="$t('studio.resolution')">
                      <div class="settings-res-row settings-pref-control">
                        <DqSelect v-model="modelParams.width" class="settings-select--w120">
                          <DqOption v-for="w in resPair.width.options" :key="w" :label="String(w)" :value="w" />
                        </DqSelect>
                        <span class="settings-res-x">x</span>
                        <DqSelect v-model="modelParams.height" class="settings-select--w120">
                          <DqOption v-for="h in resPair.height.options" :key="h" :label="String(h)" :value="h" />
                        </DqSelect>
                      </div>
                    </DqPrefRow>

                    <template v-for="key in scalarKeys" :key="key">
                      <DqPrefRow
                        v-if="specOf(key)"
                        :label="paramLabel(key, specOf(key))"
                        :class="(specOf(key).type === 'int' || specOf(key).type === 'float') ? 'settings-pref-row--slider' : ''"
                      >
                        <div class="settings-pref-control">
                          <div
                            v-if="specOf(key).type === 'int' || specOf(key).type === 'float'"
                            class="param-control-row settings-pref-slider-row"
                          >
                            <div class="param-slider">
                              <DqSlider
                                v-model="modelParams[key]"
                                :min="specOf(key).min"
                                :max="specOf(key).max"
                                :step="numStep(key, specOf(key))"
                              />
                            </div>
                            <DqInputNumber
                              v-model="modelParams[key]"
                              :min="specOf(key).min"
                              :max="specOf(key).max"
                              :step="numStep(key, specOf(key))"
                              :precision="specOf(key).type === 'float' ? 2 : 0"
                              controls-position="right"
                              class="param-input-number"
                            />
                            <DqIconButton
                              v-if="isParamChanged(key)"
                              type="text"
                              size="sm"
                              class="settings-pref-reset-btn"
                              :label="$t('settings.resetToDefault')"
                              @click="resetParam(key)"
                            >
                              <DqIcon><refresh-left /></DqIcon>
                            </DqIconButton>
                          </div>
                          <div v-else-if="specOf(key).type === 'enum'" class="settings-enum-row settings-pref-inline-control">
                            <DqSelect v-model="modelParams[key]" class="settings-select--flex">
                              <DqOption v-for="opt in specOf(key).options" :key="String(opt)" :label="String(opt)" :value="opt" />
                            </DqSelect>
                            <DqIconButton
                              v-if="isParamChanged(key)"
                              type="text"
                              size="sm"
                              class="settings-pref-reset-btn"
                              :label="$t('settings.resetToDefault')"
                              @click="resetParam(key)"
                            >
                              <DqIcon><refresh-left /></DqIcon>
                            </DqIconButton>
                          </div>
                          <div v-else-if="specOf(key).type === 'bool'" class="settings-bool-row settings-pref-inline-control">
                            <DqSwitch v-model="modelParams[key]" />
                            <DqIconButton
                              v-if="isParamChanged(key)"
                              type="text"
                              size="sm"
                              class="settings-pref-reset-btn"
                              :label="$t('settings.resetToDefault')"
                              @click="resetParam(key)"
                            >
                              <DqIcon><refresh-left /></DqIcon>
                            </DqIconButton>
                          </div>
                          <div
                            v-if="specOf(key).note || (isParamChanged(key) && specOf(key).type !== 'int' && specOf(key).type !== 'float')"
                            class="settings-pref-control-meta"
                          >
                            <DqTooltip v-if="specOf(key).note" :content="specOf(key).note" placement="top">
                              <DqIcon class="settings-help-icon"><question-filled /></DqIcon>
                            </DqTooltip>
                            <DqTag
                              v-if="isParamChanged(key) && specOf(key).type !== 'int' && specOf(key).type !== 'float'"
                              size="small"
                              type="warning"
                              effect="plain"
                            >{{ $t('settings.modified') }}</DqTag>
                          </div>
                        </div>
                      </DqPrefRow>
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
                    <DqPrefRow v-if="seedSupport" :label="$t('studio.seed')">
                      <div class="settings-seed-row settings-pref-control">
                        <DqInput v-model="modelParams.seed" :placeholder="$t('studio.seedPlaceholder')" />
                        <DqIconButton
                          type="text"
                          size="sm"
                          class="settings-seed-dice-btn dq-icon-btn--circle"
                          :label="$t('studio.seed')"
                          @click="modelParams.seed = String(Math.floor(Math.random() * 1_000_000))"
                        >
                          <DqIcon><refresh /></DqIcon>
                        </DqIconButton>
                      </div>
                    </DqPrefRow>
                  </DqPrefPane>

                  <div class="settings-model-save-row">
                    <DqButton type="primary" class="save-button settings-model-save-btn" @click="saveModelConfig">
                      <DqIcon><check /></DqIcon>
                      {{ $t('common.save') }}
                    </DqButton>
                  </div>
                </section>
              </DqCol>

              <DqCol :xs="24" :md="8" :lg="7" :xl="7" class="settings-model-sidebar">
                <ModelConfigInspector
                  :current-model-config="currentModelConfig"
                  :selected-model="selectedModel"
                  :selected-default-version="selectedDefaultVersion"
                  :capability-list="capabilityList"
                  :system-info="systemInfo"
                  :min-version-size-g-b="minVersionSizeGB"
                  :memory-progress-color="memoryProgressColor"
                  :recommended-version="recommendedVersion"
                  :hardware-advice="hardwareAdvice"
                  :has-param-notes="hasParamNotes"
                  :param-notes-list="paramNotesList"
                  :version-status-type="versionStatusType"
                  :version-status-label="versionStatusLabel"
                  :is-recommended-version="isRecommendedVersion"
                />
              </DqCol>

            </DqRow>
          </template>
        </div>

        <!-- Parameter preset save dialog -->
        <DqDialog v-model:open="paramPresetDialogVisible" :title="$t('settings.saveParamPreset')" width="400px">
          <DqPrefPane class="settings-pref-pane-form settings-pref-pane-form--dialog">
            <DqPrefRow :label="$t('settings.presetName')" stacked>
              <DqInput v-model="paramPresetForm.name" :placeholder="$t('settings.presetNamePlaceholder')" />
            </DqPrefRow>
            <DqPrefRow no-label stacked>
              <DqCheckbox v-model="paramPresetForm.isDefault">{{ $t('settings.setAsDefaultPreset') }}</DqCheckbox>
            </DqPrefRow>
          </DqPrefPane>
          <template #footer>
            <DqButton @click="paramPresetDialogVisible = false">{{ $t('common.cancel') }}</DqButton>
            <DqButton type="primary" @click="saveParamPreset">{{ $t('common.save') }}</DqButton>
          </template>
        </DqDialog>
      </DqSectionTabPanel>

      <DqSectionTabPanel name="presets">
        <PromptTemplatesPanel
          v-model:dialog-open="presetDialogVisible"
          :presets="presets"
          :restore-config-busy="restoreConfigBusy"
          :editing-preset-name="editingPresetName"
          :preset-form="presetForm"
          @add="openPresetDialog()"
          @edit="(name, preset) => openPresetDialog(name, preset)"
          @delete="confirmDeletePreset"
          @save="savePreset"
          @restore="confirmRestorePromptTemplates"
        />
      </DqSectionTabPanel>

      <DqSectionTabPanel name="system">
        <DqRow :gutter="20" class="settings-system-layout-row">
          <!-- Left: public config + model list + LoRA list -->
          <DqCol :xs="24" :md="16" :lg="17" :xl="18">
            <!-- Public config -->
            <DqSurfaceCard class="settings-tab-panel">
              <template #header>
                <div class="card-title">
                  <DqIcon><setting /></DqIcon>
                  {{ $t('settings.publicConfig') }}
                  <DqText class="settings-title-desc" size="small" type="info">
                    {{ $t('settings.publicConfigDesc') }}
                  </DqText>
                </div>
              </template>

              <SystemSettingsForm
                :settings="settings"
                :default-model-options-by-media="defaultModelOptionsByMedia"
                :workspace-paths="workspacePaths"
                :restore-config-busy="restoreConfigBusy"
                @save="saveSettings"
                @language-change="handleLanguageChange"
                @pick-workspace="pickWorkspaceDirectory"
                @restore-model-registry="confirmRestoreModelRegistry"
                @restore-prompt-templates="confirmRestorePromptTemplates"
              />
            </DqSurfaceCard>

          </DqCol>

          <!-- Right: system info + cache + monitor (inspector) -->
          <DqCol :xs="24" :md="8" :lg="7" :xl="6">
            <SystemSettingsSidebar
              :system-info="systemInfo"
              :cache-status="cacheStatus"
              :cache-loading="cacheLoading"
              :cache-error="cacheError"
              :monitor-data="monitorData"
              @refresh-cache="refreshCacheStatus"
            />
          </DqCol>
        </DqRow>
      </DqSectionTabPanel>
    </DqSectionTabs>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, watch, onMounted, onUnmounted, inject, type Ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast, confirm } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt, $mn, $md } from '@/utils/i18n';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { useRegistryStore } from '@/stores/registry';
import type { SystemInfo } from '@/types';
import * as RegistryParamSchema from '@/utils/registryParamSchema';
import ModelLicenseBadges from '@/components/model/ModelLicenseBadges.vue';
import ModelConfigInspector from '@/components/settings/ModelConfigInspector.vue';
import SystemSettingsForm from '@/components/settings/SystemSettingsForm.vue';
import PromptTemplatesPanel from '@/components/settings/PromptTemplatesPanel.vue';
import SystemSettingsSidebar from '@/components/settings/SystemSettingsSidebar.vue';

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
  default_model_image: '',
  default_model_video: '',
  default_model_audio: '',
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

const defaultModelOptionsByMedia = computed(() => {
  const buckets: { image: { id: string; label: string }[]; video: { id: string; label: string }[]; audio: { id: string; label: string }[] } = {
    image: [],
    video: [],
    audio: [],
  };
  for (const [id, cfg] of Object.entries(modelRegistry.value)) {
    const media = String(cfg.media || 'image').toLowerCase();
    if (media !== 'image' && media !== 'video' && media !== 'audio') continue;
    buckets[media].push({ id, label: $mn(cfg, id) });
  }
  for (const key of ['image', 'video', 'audio'] as const) {
    buckets[key].sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: 'base' }));
  }
  return buckets;
});

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
  if (ratio < 0.5) return 'var(--dq-success)';
  if (ratio < 0.8) return 'var(--dq-warning)';
  return 'var(--dq-danger)';
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

/** DqSelect 用 === 匹配 value：预设/存储里的字符串需对齐到 registry 枚举的原始类型，否则选中项不显示 */
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
  /* 设置页用短标签，避免「CFG 引导 (Guidance)」在窄列换行导致行错位 */
  const settingsShort: Record<string, string> = {
    steps: 'settings.steps',
    guidance: 'settings.guidance',
    scheduler: 'settings.scheduler',
    strength: 'create.strength',
    controlnet_strength: 'create.controlNetStrengthLabel',
    redux_strength: 'create.reduxStrengthLabel',
  };
  const shortKey = settingsShort[key];
  if (shortKey) {
    try {
      const text = $tt(shortKey);
      return text.replace(/\s*[(（][^)）]+[)）]\s*$/, '').trim();
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
  const modelKey = selectedModel.value;
  const verKey = selectedDefaultVersion.value;
  const config = modelRegistry.value[modelKey];
  if (!config?.versions || !verKey) return;
  for (const [k, ver] of Object.entries(config.versions)) {
    ver.default = k === verKey;
  }
};

const onSettingsModelRestoreDefaults = () => {
  const config = currentModelConfig.value;
  if (config && config.parameters) {
    RegistryParamSchema.applyDefaults(config.parameters, modelParams);
    snapModelResolutionEnums();
    toast.success($tt('studio.restoredDefaults'));
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
      toast.success($tt('settings.modelConfigSaved'));
      await loadModelRegistry();
      await registryStore.load();
    } else {
      toast.error(result.error || $tt('settings.saveFailed'));
    }
  } catch (e) {
    console.error('Failed to save model config:', e);
    toast.error($tt('settings.saveFailed'));
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
    await confirm(
      $tt('settings.workspaceChangeConfirm', { from: fromPath, to: toPath }),
      $tt('settings.workspaceChangeTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.workspaceChangeContinue'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    await confirm(
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
    if (!String(settings.default_model_image || '').trim() && String(settings.default_model || '').trim()) {
      settings.default_model_image = settings.default_model;
    }
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
    toast.error((e as Error).message || String(e));
  }
};

const saveSettings = async () => {
  const newWs = String(settings.custom_workspace_dir || '').trim();
  const oldWs = initialWorkspaceDir.value;

  if (newWs !== oldWs) {
    if (!newWs) {
      toast.warning($tt('settings.workspaceRequired'));
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
        toast.warning($tt('settings.customWorkspaceRestartHint'));
      }
      await loadWorkspacePaths();
    } catch (e) {
      settings.custom_workspace_dir = oldWs;
      toast.error(extractApiError(e) || $tt('settings.workspaceApplyFailed'));
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
      default_model: String(settings.default_model_image || settings.default_model || ''),
      default_model_image: settings.default_model_image || '',
      default_model_video: settings.default_model_video || '',
      default_model_audio: settings.default_model_audio || '',
      civitai_token: settings.civitai_token || '',
      huggingface_token: settings.huggingface_token || '',
      nsfw_enabled: settings.nsfw_enabled,
    });
    toast.success($tt('settings.saved'));
  } catch (e) {
    toast.error($tt('settings.saveFailed'));
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
      toast.warning($tt('settings.restoreConfigNothing'));
      return;
    }
    if (restored.includes('presets.json')) {
      await loadPresets();
    }
    if (restored.includes('models_registry.json')) {
      await loadModelRegistry();
      await registryStore.load(true);
    }
    toast.success($tt('settings.restoreConfigSuccess'));
    if (res.restart_required) {
      toast.warning($tt('settings.restoreRegistryRestartHint'));
    }
  } catch (e) {
    toast.error(extractApiError(e) || $tt('settings.restoreConfigFailed'));
  } finally {
    restoreConfigBusy.value = false;
  }
}

async function confirmRestoreModelRegistry() {
  try {
    await confirm(
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
    await confirm(
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
    toast.warning($tt('settings.enterTemplateName'));
    return;
  }
  try {
    if (!Array.isArray(presetForm.applies_to) || !presetForm.applies_to.length) {
      toast.warning($tt('settings.presetAppliesRequired'));
      return;
    }
    const applies = [...presetForm.applies_to];
    await api.settings.savePreset(presetForm.name.trim(), {
      positive: presetForm.positive,
      negative: presetForm.negative,
      media_scope: presetForm.media_scope,
      applies_to: applies,
    });
    toast.success($tt('settings.templateSaved'));
    presetDialogVisible.value = false;
    await loadPresets();
  } catch (e) {
    console.error('Failed to save preset:', e);
    toast.error($tt('settings.saveFailed'));
  }
};

const confirmDeletePreset = (name: string) => {
  confirm(
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
    toast.success($tt('settings.templateDeleted'));
    await loadPresets();
  } catch (e) {
    console.error('Failed to delete preset:', e);
    toast.error($tt('settings.saveFailed'));
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
    toast.warning($tt('settings.enterPresetName'));
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
  toast.success($tt('settings.presetSaved'));
};

const loadParamPreset = (preset: ParamPreset) => {
  if (!preset || !preset.params) return;
  Object.assign(modelParams, preset.params);
  snapModelResolutionEnums();
  toast.success($tt('settings.presetLoaded', { name: preset.name }));
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
  loadPresets();
  loadParamPresets();
  startMonitor();
  refreshCacheStatus();
});

onUnmounted(() => {
  stopMonitor();
});
</script>
