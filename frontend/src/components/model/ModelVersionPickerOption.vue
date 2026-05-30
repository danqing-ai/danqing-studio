<script setup lang="ts">
import ModelVersionPickerExtras from '@/components/model/ModelVersionPickerExtras.vue';

defineProps<{
  description?: string;
  recommended?: boolean;
  commercialUseAllowed?: boolean;
  status?: string;
  size?: string;
}>();
</script>

<template>
  <div class="studio-picker-option-suffix">
    <p v-if="description" class="studio-picker-option__desc">{{ description }}</p>
    <ModelVersionPickerExtras
      :recommended="recommended"
      :commercial-use-allowed="commercialUseAllowed"
      :status="status"
      :size="size"
    />
  </div>
</template>
