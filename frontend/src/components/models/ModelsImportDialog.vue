<script setup lang="ts">
defineProps<{
  importing: boolean;
}>();

const open = defineModel<boolean>('open', { required: true });
const importModelName = defineModel<string>('importModelName', { required: true });
const importModelPath = defineModel<string>('importModelPath', { required: true });
const importModelType = defineModel<string>('importModelType', { required: true });

const emit = defineEmits<{
  submit: [];
  cancel: [];
}>();
</script>

<template>
  <DqDialog
    v-model:open="open"
    :title="$t('download.importTitle')"
    width="500px"
    @update:open="(v: boolean) => { if (!v) emit('cancel'); }"
  >
    <DqPrefPane class="settings-pref-pane-form settings-pref-pane-form--dialog">
      <DqPrefRow :label="$t('download.modelName')" stacked>
        <DqInput
          v-model="importModelName"
          :placeholder="$t('download.modelNamePlaceholder')"
        />
      </DqPrefRow>
      <DqPrefRow :label="$t('download.modelPath')" stacked>
        <DqInput
          v-model="importModelPath"
          :placeholder="$t('download.modelPathPlaceholder')"
        />
      </DqPrefRow>
      <DqPrefRow :label="$t('download.modelType')" stacked>
        <DqSelect v-model="importModelType" class="models-import-model-type studio-w-full">
          <DqOption :label="$t('download.baseModel')" value="base" />
          <DqOption :label="$t('download.loraType')" value="lora" />
          <DqOption :label="$t('download.controlnetType')" value="controlnet" />
        </DqSelect>
      </DqPrefRow>
    </DqPrefPane>
    <template #footer>
      <DqButton @click="emit('cancel')">{{ $t('download.cancel') }}</DqButton>
      <DqButton type="primary" :loading="importing" @click="emit('submit')">
        {{ $t('download.import_') }}
      </DqButton>
    </template>
  </DqDialog>
</template>
