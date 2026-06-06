<template>
  <DqTag v-if="label" :type="tagType" effect="plain">
    {{ label }}
  </DqTag>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  source: string;
}>();

const { t } = useI18n();

const label = computed(() => {
  if (props.source === 'huggingface') return t('download.sourceHuggingface');
  if (props.source === 'modelscope') return t('download.sourceModelscope');
  if (props.source === 'civitai') return t('download.sourceCivitai');
  if (props.source === 'local') return t('download.sourceLocal');
  return '';
});

const tagType = computed(() => {
  if (props.source === 'huggingface') return 'primary';
  if (props.source === 'civitai') return 'warning';
  return 'info';
});
</script>
