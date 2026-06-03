<!-- @ts-nocheck -->
<template>
  <div class="prompts-page">
    <PromptTemplatesPanel
      v-model:dialog-open="presetDialogVisible"
      v-model:active-category="activeCategory"
      v-model:active-action="activeAction"
      v-model:search-query="searchQuery"
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
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue';
import { toast, confirm } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import PromptTemplatesPanel from '@/components/settings/PromptTemplatesPanel.vue';

/* ───── State ───── */

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
const activeCategory = ref('all');
const activeAction = ref('');
const searchQuery = ref('');
const restoreConfigBusy = ref(false);

/* ───── Loaders ───── */

const loadPresets = async () => {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as Record<string, Record<string, unknown>>) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
  }
};

/* ───── Dialog ───── */

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

/* ───── CRUD ───── */

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
  )
    .then(() => deletePreset(name))
    .catch(() => {});
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

/* ───── Restore ───── */

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
    toast.success($tt('settings.restoreConfigSuccess'));
  } catch (e) {
    toast.error($tt('settings.restoreConfigFailed'));
  } finally {
    restoreConfigBusy.value = false;
  }
}

/* ───── Lifecycle ───── */

onMounted(() => {
  loadPresets();
});
</script>

<style scoped>
.prompts-page {
  width: 100%;
  height: 100%;
}
</style>
