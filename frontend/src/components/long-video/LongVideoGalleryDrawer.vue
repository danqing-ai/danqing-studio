<template>
  <DqDrawer
    :open="open"
    :title="$tt('video.longVideoImportFromGallery')"
    direction="rtl"
    size="520px"
    class="lv-gallery-drawer"
    @update:open="$emit('update:open', $event)"
  >
    <AssetPicker accept-kind="image" :recent-gallery="recentGallery" @pick="$emit('pick', $event)" />
  </DqDrawer>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n';
import AssetPicker from '@/components/asset/AssetPicker.vue';

defineProps<{
  open: boolean;
  recentGallery: Array<Record<string, unknown>>;
}>();

defineEmits<{
  (e: 'update:open', value: boolean): void;
  (e: 'pick', payload: { path: string }): void;
}>();

const { t: $tt } = useI18n();
</script>

<style scoped>
.lv-gallery-drawer :deep(.asset-picker) {
  min-height: 0;
}
</style>
