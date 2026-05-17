<script setup lang="ts">
import { computed } from 'vue';
import { $tt } from '@/utils/i18n';
import PresetTemplateDialog from '@/components/settings/PresetTemplateDialog.vue';

const props = defineProps<{
  presets: Record<string, Record<string, unknown>>;
  restoreConfigBusy: boolean;
  editingPresetName: string;
  presetForm: {
    name: string;
    positive: string;
    negative: string;
    media_scope: string;
    applies_to: string[];
  };
}>();

const dialogOpen = defineModel<boolean>('dialogOpen', { required: true });

const emit = defineEmits<{
  add: [];
  edit: [name: string, preset: Record<string, unknown>];
  delete: [name: string];
  save: [];
  restore: [];
}>();

const presetList = computed(() =>
  Object.entries(props.presets).map(([name, preset]) => ({ name, preset })),
);

const presetAppliesSummary = (preset: Record<string, unknown>) =>
  (preset.applies_to as string[]).join(', ');

const presetMediaLabel = (preset: Record<string, unknown>) =>
  preset.media_scope === 'video' ? $tt('settings.presetMediaVideo') : $tt('settings.presetMediaImage');
</script>

<template>
  <DqSurfaceCard class="settings-tab-panel">
    <template #header>
      <div class="card-title card-title--split">
        <span class="settings-card-header">
          <DqIcon><collection /></DqIcon>
          {{ $t('settings.promptTemplates') }}
          <DqText class="settings-title-desc" size="small" type="info">
            {{ $t('settings.promptTemplatesDesc') }}
          </DqText>
        </span>
        <div class="settings-header-actions">
          <DqButton
            type="text"
            size="sm"
            class="settings-link-destructive"
            :loading="restoreConfigBusy"
            @click="emit('restore')"
          >
            {{ $t('settings.restorePromptTemplates') }}
          </DqButton>
          <DqButton type="primary" size="sm" @click="emit('add')">
            <DqIcon><plus /></DqIcon>
            {{ $t('settings.addTemplate') }}
          </DqButton>
        </div>
      </div>
    </template>

    <DqInspectorEmpty v-if="Object.keys(presets).length === 0" class="settings-templates-empty">
      {{ $t('settings.noTemplates') }}
    </DqInspectorEmpty>

    <ul v-else class="settings-preset-list" role="list">
      <li v-for="row in presetList" :key="row.name" class="settings-preset-row" role="listitem">
        <div class="settings-preset-row__main">
          <div class="settings-preset-row__head">
            <span class="settings-preset-row__name">{{ row.name }}</span>
            <span class="gallery-list-tag">{{ presetMediaLabel(row.preset) }}</span>
          </div>
          <p class="settings-preset-row__line">
            <span class="settings-preset-row__label">{{ $t('settings.positivePrompt') }}</span>
            <span class="settings-preset-row__text">{{ row.preset.positive || '-' }}</span>
          </p>
          <p class="settings-preset-row__line settings-preset-row__line--muted">
            <span class="settings-preset-row__label">{{ $t('settings.negativePrompt') }}</span>
            <span class="settings-preset-row__text">{{ row.preset.negative || '-' }}</span>
          </p>
          <p class="settings-preset-row__applies">
            <span class="settings-preset-row__label">{{ $t('settings.presetAppliesTo') }}</span>
            <span class="settings-table-muted">{{ presetAppliesSummary(row.preset) }}</span>
          </p>
        </div>
        <div class="settings-preset-row__actions">
          <DqIconButton type="text" size="sm" :label="$t('settings.editTemplate')" @click="emit('edit', row.name, row.preset)">
            <DqIcon><edit /></DqIcon>
          </DqIconButton>
          <DqIconButton type="danger" size="sm" :label="$t('common.delete')" @click="emit('delete', row.name)">
            <DqIcon><delete /></DqIcon>
          </DqIconButton>
        </div>
      </li>
    </ul>
  </DqSurfaceCard>

  <PresetTemplateDialog
    v-model:open="dialogOpen"
    :editing-name="editingPresetName"
    :preset-form="presetForm"
    @save="emit('save')"
    @cancel="dialogOpen = false"
  />
</template>
