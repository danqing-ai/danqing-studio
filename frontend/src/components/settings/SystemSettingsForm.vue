<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import type { ThemeId } from '@/utils/i18n';
import { $mn } from '@/utils/i18n';
import { canvasAutoAddEnabled, setCanvasAutoAdd } from '@/composables/useCanvasStore';
import { useRegistryStore } from '@/stores/registry';

const canvasAutoAddImage = ref(canvasAutoAddEnabled('image'));
const canvasAutoAddVideo = ref(canvasAutoAddEnabled('video'));
const canvasAutoAddAudio = ref(canvasAutoAddEnabled('audio'));

function onCanvasAutoAddImageChange(enabled: boolean) {
  canvasAutoAddImage.value = enabled;
  setCanvasAutoAdd(enabled, 'image');
}

function onCanvasAutoAddVideoChange(enabled: boolean) {
  canvasAutoAddVideo.value = enabled;
  setCanvasAutoAdd(enabled, 'video');
}

function onCanvasAutoAddAudioChange(enabled: boolean) {
  canvasAutoAddAudio.value = enabled;
  setCanvasAutoAdd(enabled, 'audio');
}

type SectionId =
  | 'general'
  | 'performance'
  | 'studio'
  | 'workspace'
  | 'integrations'
  | 'maintenance'
  | 'systeminfo';

const props = defineProps<{
  activeSection: SectionId;
  settings: Record<string, unknown>;
  workspacePaths: Record<string, string> | null;
  restoreConfigBusy: boolean;
}>();

const themeOptions: { label: string; value: ThemeId }[] = [
  { label: 'settings.themeAppleDark', value: 'apple-dark' },
  { label: 'settings.themeLinearDark', value: 'linear-dark' },
  { label: 'settings.themeChinaRedDark', value: 'china-red-dark' },
  { label: 'settings.themeShadcnDark', value: 'shadcn-dark' },
];

const emit = defineEmits<{
  save: [];
  languageChange: [lang: string];
  themeChange: [theme: ThemeId];
  pickWorkspace: [];
  restoreModelRegistry: [];
}>();

const registryStore = useRegistryStore();

onMounted(() => {
  void registryStore.load();
});

function hasLlmChatAction(actions: unknown): boolean {
  if (!actions || typeof actions !== 'object') return false;
  const row = actions as Record<string, unknown>;
  return row.chat != null || row.enhance != null;
}

function hasVlmDescribeAction(actions: unknown): boolean {
  if (!actions || typeof actions !== 'object') return false;
  return (actions as Record<string, unknown>).describe != null;
}

function llmSupportsThink(modelId: unknown): boolean {
  return /thinking/i.test(String(modelId || '').trim());
}

const llmThinkSupported = computed(() => llmSupportsThink(props.settings.default_model_llm));

watch(
  () => props.settings.default_model_llm,
  (modelId) => {
    if (!llmSupportsThink(modelId)) {
      props.settings.default_model_llm_think = false;
    }
  },
);

const llmModelOptions = computed(() => {
  const models = registryStore.registry?.models || {};
  return Object.entries(models)
    .filter(([, cfg]) => cfg.media === 'llm' && hasLlmChatAction(cfg.actions))
    .map(([id, cfg]) => ({
      value: id,
      label: $mn(cfg.name, id),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
});

const vlmModelOptions = computed(() => {
  const models = registryStore.registry?.models || {};
  return Object.entries(models)
    .filter(([, cfg]) => cfg.media === 'llm' && hasVlmDescribeAction(cfg.actions))
    .map(([id, cfg]) => ({
      value: id,
      label: $mn(cfg.name, id),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
});
</script>

<template>
  <div class="settings-system-form">
    <!-- General -->
    <template v-if="props.activeSection === 'general'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.general') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.generalDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.language')">
            <DqSelect
              v-model="settings.language"
              class="settings-mac-value-control"
              @change="emit('languageChange', $event as string)"
            >
              <DqOption :label="$t('settings.label_zh')" value="zh" />
              <DqOption :label="$t('settings.label_en')" value="en" />
            </DqSelect>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.theme')">
            <DqSelect
              v-model="settings.theme"
              class="settings-mac-value-control"
              @change="emit('themeChange', $event as ThemeId)"
            >
              <DqOption
                v-for="opt in themeOptions"
                :key="opt.value"
                :label="$t(opt.label)"
                :value="opt.value"
              />
            </DqSelect>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.outputFormat')">
            <DqSelect v-model="settings.output_format" class="settings-mac-value-control">
              <DqOption label="PNG" value="png" />
              <DqOption label="JPEG" value="jpg" />
              <DqOption label="WebP" value="webp" />
            </DqSelect>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.defaultLlmModel')" stacked>
            <div class="settings-stacked-control">
              <DqSelect
                v-model="settings.default_model_llm"
                class="settings-mac-value-control"
                :placeholder="$t('settings.defaultLlmModelPlaceholder')"
              >
                <DqOption
                  v-for="opt in llmModelOptions"
                  :key="opt.value"
                  :label="opt.label"
                  :value="opt.value"
                />
              </DqSelect>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.defaultLlmModelDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow
            v-if="llmThinkSupported"
            :label="$t('settings.defaultLlmThink')"
            stacked
          >
            <div class="settings-stacked-control">
              <DqSwitch v-model="settings.default_model_llm_think" />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.defaultLlmThinkDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.defaultVlmModel')" stacked>
            <div class="settings-stacked-control">
              <DqSelect
                v-model="settings.default_model_vlm"
                class="settings-mac-value-control"
                :placeholder="$t('settings.defaultVlmModelPlaceholder')"
              >
                <DqOption
                  v-for="opt in vlmModelOptions"
                  :key="opt.value"
                  :label="opt.label"
                  :value="opt.value"
                />
              </DqSelect>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.defaultVlmModelDesc') }}
              </p>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Performance -->
    <template v-if="props.activeSection === 'performance'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.performance') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.performanceDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.memoryLimit')">
            <div class="param-control-row settings-pref-slider-row">
              <div class="param-slider">
                <DqSlider v-model="settings.mlx_memory_limit" :min="32" :max="256" :step="8" />
              </div>
              <span class="settings-slider-suffix">{{ settings.mlx_memory_limit }} GB</span>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.modelCacheTtl')" stacked>
            <div class="settings-stacked-control">
              <div class="param-control-row settings-pref-slider-row">
                <div class="param-slider">
                  <DqSlider v-model="settings.model_cache_ttl_minutes" :min="5" :max="120" :step="5" />
                </div>
                <span class="settings-slider-suffix">{{ settings.model_cache_ttl_minutes }} min</span>
              </div>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.modelCacheTtlDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.queueImageFirst')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch v-model="settings.queue_image_first" />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.queueImageFirstDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.autoSavePrompts')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch v-model="settings.auto_save_prompts" />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.autoSavePromptsDesc') }}
              </p>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Studio / Canvas -->
    <template v-if="props.activeSection === 'studio'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.studio') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.studioDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.canvasAutoAddImage')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch
                :model-value="canvasAutoAddImage"
                @update:model-value="onCanvasAutoAddImageChange"
              />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.canvasAutoAddImageDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.canvasAutoAddVideo')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch
                :model-value="canvasAutoAddVideo"
                @update:model-value="onCanvasAutoAddVideoChange"
              />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.canvasAutoAddVideoDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.canvasAutoAddAudio')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch
                :model-value="canvasAutoAddAudio"
                @update:model-value="onCanvasAutoAddAudioChange"
              />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.canvasAutoAddAudioDesc') }}
              </p>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Workspace -->
    <template v-if="props.activeSection === 'workspace'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.workspace') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.workspaceDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.customWorkspace')" stacked>
            <div class="settings-stacked-control settings-workspace-picker">
              <div class="settings-workspace-input-row">
                <DqInput
                  v-model="settings.custom_workspace_dir"
                  :placeholder="$t('settings.customWorkspacePlaceholder')"
                />
                <DqButton size="sm" class="settings-workspace-pick-btn" @click="emit('pickWorkspace')">
                  {{ $t('settings.pickWorkspace') }}
                </DqButton>
              </div>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.workspaceSetupEmptyHint') }}
              </p>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.customWorkspaceHint') }}
              </p>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.customWorkspaceRestartHint') }}
              </p>
              <div v-if="workspacePaths" class="settings-workspace-paths">
                <div class="settings-workspace-paths-title">{{ $t('settings.workspaceLayoutTitle') }}</div>
                <ul class="settings-workspace-paths-list">
                  <li v-for="(p, key) in workspacePaths" :key="key">
                    <span class="settings-workspace-paths-key">{{ key }}</span>
                    <span class="settings-workspace-paths-val">{{ p }}</span>
                  </li>
                </ul>
              </div>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Integrations -->
    <template v-if="props.activeSection === 'integrations'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.integrations') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.integrationsDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.huggingfaceToken')" stacked>
            <div class="settings-stacked-control">
              <DqInput
                v-model="settings.huggingface_token"
                type="password"
                show-password
                :placeholder="$t('settings.huggingfaceTokenPlaceholder')"
              >
                <template #prefix>
                  <DqIcon><document /></DqIcon>
                </template>
              </DqInput>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.huggingfaceTokenDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.civitaiToken')" stacked>
            <div class="settings-stacked-control">
              <DqInput
                v-model="settings.civitai_token"
                type="password"
                show-password
                :placeholder="$t('settings.civitaiTokenPlaceholder')"
              >
                <template #prefix>
                  <DqIcon><document /></DqIcon>
                </template>
              </DqInput>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.civitaiTokenDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow v-if="settings.civitai_token" no-label>
            <div class="settings-stacked-control">
              <DqCheckbox v-model="settings.nsfw_enabled" size="large">
                <DqText type="danger">{{ $t('settings.nsfwContent') }}</DqText>
              </DqCheckbox>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.nsfwDesc') }}
              </p>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Maintenance -->
    <template v-if="props.activeSection === 'maintenance'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.maintenance') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.maintenanceDesc') }}</p>
        <div
          class="settings-grouped-form settings-grouped-form--action-list"
          role="group"
          :aria-label="$t('settings.maintenance')"
        >
          <button
            type="button"
            class="settings-action-row settings-action-row--destructive"
            :disabled="restoreConfigBusy"
            @click="emit('restoreModelRegistry')"
          >
            <span class="settings-action-row__label">{{ $t('settings.restoreModelRegistry') }}</span>
            <DqIcon class="settings-action-row__chevron"><arrow-right /></DqIcon>
          </button>
        </div>
        <p class="settings-group-footnote">{{ $t('settings.restoreModelRegistryDesc') }}</p>
      </section>
    </template>

    <!-- Save row -->
    <div class="settings-system-save-row" v-if="props.activeSection !== 'systeminfo' && props.activeSection !== 'maintenance'">
      <DqButton type="primary" class="settings-system-save-btn" @click="emit('save')">
        <DqIcon><check /></DqIcon>
        {{ $t('common.save') }}
      </DqButton>
    </div>
  </div>
</template>
