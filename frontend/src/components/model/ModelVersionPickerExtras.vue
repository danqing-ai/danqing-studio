<script setup lang="ts">
import ModelLicenseBadges from '@/components/model/ModelLicenseBadges.vue';

defineProps<{
  recommended?: boolean;
  commercialUseAllowed?: boolean;
  status?: string;
  size?: string;
}>();
</script>

<template>
  <ModelLicenseBadges
    :recommended="recommended"
    :commercial-use-allowed="commercialUseAllowed"
    effect="plain"
  />
  <DqTag v-if="status === 'ready'" size="small" type="success">{{ $t('studio.ready') }}</DqTag>
  <DqTag v-else-if="status === 'incomplete'" size="small" type="danger">{{ $t('studio.incomplete') }}</DqTag>
  <DqTag v-else-if="status" size="small" type="warning">{{ $t('studio.notDownloaded') }}</DqTag>
  <span v-if="size" class="studio-picker-option__meta">{{ size }}</span>
</template>
