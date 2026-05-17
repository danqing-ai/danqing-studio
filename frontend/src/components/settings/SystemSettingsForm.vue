<script setup lang="ts">
defineProps<{
  settings: Record<string, unknown>;
  defaultModelOptionsByMedia: {
    image: { id: string; label: string }[];
    video: { id: string; label: string }[];
    audio: { id: string; label: string }[];
  };
  workspacePaths: Record<string, string> | null;
  restoreConfigBusy: boolean;
}>();

const defaultModelMediaRows = [
  { media: 'image' as const, labelKey: 'settings.defaultModelImage', descKey: 'settings.defaultModelImageDesc', settingKey: 'default_model_image' },
  { media: 'video' as const, labelKey: 'settings.defaultModelVideo', descKey: 'settings.defaultModelVideoDesc', settingKey: 'default_model_video' },
  { media: 'audio' as const, labelKey: 'settings.defaultModelAudio', descKey: 'settings.defaultModelAudioDesc', settingKey: 'default_model_audio' },
];

const emit = defineEmits<{
  save: [];
  languageChange: [lang: string];
  pickWorkspace: [];
  restoreModelRegistry: [];
  restorePromptTemplates: [];
}>();
</script>

<template>
  <div class="settings-system-form">
    <section class="settings-group-block">
      <h3 class="settings-group-block-title">{{ $t('settings.systemGroupGeneral') }}</h3>
      <DqPrefPane class="settings-grouped-form settings-grouped-form--system settings-pref-pane-form settings-pref-pane-form--system">
        <DqPrefRow
          v-for="row in defaultModelMediaRows"
          :key="row.media"
          :label="$t(row.labelKey)"
        >
          <div class="settings-mac-value-column">
            <DqSelect
              v-model="settings[row.settingKey]"
              class="settings-mac-value-control"
              clearable
              :placeholder="$t('settings.selectDefaultModel')"
            >
              <DqOption
                v-for="opt in defaultModelOptionsByMedia[row.media]"
                :key="opt.id"
                :label="opt.label"
                :value="opt.id"
              />
            </DqSelect>
            <p class="settings-form-hint settings-form-hint--value-footnote">
              {{ $t(row.descKey) }}
            </p>
          </div>
        </DqPrefRow>

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

        <DqPrefRow :label="$t('settings.outputFormat')">
          <DqSelect v-model="settings.output_format" class="settings-mac-value-control">
            <DqOption label="PNG" value="png" />
            <DqOption label="JPEG" value="jpg" />
            <DqOption label="WebP" value="webp" />
          </DqSelect>
        </DqPrefRow>
      </DqPrefPane>
    </section>

    <section class="settings-group-block">
      <h3 class="settings-group-block-title">{{ $t('settings.systemGroupPerformance') }}</h3>
      <DqPrefPane class="settings-grouped-form settings-grouped-form--system settings-pref-pane-form settings-pref-pane-form--system">
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

    <section class="settings-group-block">
      <h3 class="settings-group-block-title">{{ $t('settings.systemGroupWorkspace') }}</h3>
      <DqPrefPane class="settings-grouped-form settings-grouped-form--system settings-pref-pane-form settings-pref-pane-form--system">
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

    <section class="settings-group-block">
      <h3 class="settings-group-block-title">{{ $t('settings.systemGroupConfigMaintenance') }}</h3>
      <p class="settings-group-footnote settings-group-footnote--intro">
        {{ $t('settings.configMaintenanceDesc') }}
      </p>
      <div
        class="settings-grouped-form settings-grouped-form--system settings-grouped-form--action-list"
        role="group"
        :aria-label="$t('settings.systemGroupConfigMaintenance')"
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
        <button
          type="button"
          class="settings-action-row settings-action-row--destructive"
          :disabled="restoreConfigBusy"
          @click="emit('restorePromptTemplates')"
        >
          <span class="settings-action-row__label">{{ $t('settings.restorePromptTemplates') }}</span>
          <DqIcon class="settings-action-row__chevron"><arrow-right /></DqIcon>
        </button>
      </div>
      <p class="settings-group-footnote">{{ $t('settings.restoreModelRegistryDesc') }}</p>
      <p class="settings-group-footnote">{{ $t('settings.restorePromptTemplatesDesc') }}</p>
    </section>

    <section class="settings-group-block">
      <h3 class="settings-group-block-title">{{ $t('settings.systemGroupHuggingface') }}</h3>
      <DqPrefPane class="settings-grouped-form settings-grouped-form--system settings-pref-pane-form settings-pref-pane-form--system">
        <DqPrefRow :label="$t('settings.huggingfaceToken')" stacked>
          <div class="settings-stacked-control">
            <DqInput
              v-model="settings.huggingface_token"
              type="password"
              show-password
              :placeholder="$t('settings.huggingfaceTokenPlaceholder')"
            >
              <template #prefix>
                <DqIcon><key /></DqIcon>
              </template>
            </DqInput>
            <p class="studio-field-footnote">{{ $t('studio.optional') }}</p>
            <p class="settings-form-hint settings-form-hint--below-control">
              {{ $t('settings.huggingfaceTokenDesc') }}
            </p>
          </div>
        </DqPrefRow>
      </DqPrefPane>
    </section>

    <section class="settings-group-block">
      <h3 class="settings-group-block-title">{{ $t('settings.systemGroupCivitai') }}</h3>
      <DqPrefPane class="settings-grouped-form settings-grouped-form--system settings-pref-pane-form settings-pref-pane-form--system">
        <DqPrefRow :label="$t('settings.civitaiToken')" stacked>
          <div class="settings-stacked-control">
            <DqInput
              v-model="settings.civitai_token"
              type="password"
              show-password
              :placeholder="$t('settings.civitaiTokenPlaceholder')"
            >
              <template #prefix>
                <DqIcon><key /></DqIcon>
              </template>
            </DqInput>
            <p class="studio-field-footnote">{{ $t('studio.optional') }}</p>
            <p class="settings-form-hint settings-form-hint--below-control">
              {{ $t('settings.civitaiTokenDesc') }}
            </p>
          </div>
        </DqPrefRow>

        <DqPrefRow v-if="settings.civitai_token" no-label stacked>
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

    <div class="settings-system-save-row">
      <DqButton type="primary" class="settings-system-save-btn" @click="emit('save')">
        <DqIcon><check /></DqIcon>
        {{ $t('common.save') }}
      </DqButton>
    </div>
  </div>
</template>
