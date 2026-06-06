<template>
  <DqStack wrap :gap="6" justify="end" align="center">
    <template v-if="vstatus === 'ready'">
      <DqButton
        size="sm"
        class="model-ver-btn model-ver-btn--force"
        @click="$emit('download')"
      >
        <DqIcon class="model-ver-btn__icon"><download /></DqIcon>
        <span class="model-ver-btn__label">{{ $t('download.forceDownload') }}</span>
      </DqButton>
      <DqButton
        size="sm"
        class="model-ver-btn model-ver-btn--delete"
        @click="$emit('delete')"
      >
        <DqIcon class="model-ver-btn__icon"><delete /></DqIcon>
        <span class="model-ver-btn__label">{{ $t('common.delete') }}</span>
      </DqButton>
    </template>
    <template v-else>
      <DqTooltip v-if="!canDownload" :content="dependencyHint" placement="top">
        <span>
          <DqButton size="sm" class="model-ver-btn model-ver-btn--download" disabled>
            <DqIcon class="model-ver-btn__icon"><download /></DqIcon>
            <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
          </DqButton>
        </span>
      </DqTooltip>
      <DqButton
        v-else
        size="sm"
        class="model-ver-btn model-ver-btn--download"
        :loading="loading"
        @click="$emit('download')"
      >
        <DqIcon class="model-ver-btn__icon"><download /></DqIcon>
        <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
      </DqButton>
    </template>
  </DqStack>
</template>

<script setup lang="ts">
defineProps<{
  vstatus: string;
  canDownload: boolean;
  dependencyHint: string;
  loading: boolean;
}>();

defineEmits<{
  download: [];
  delete: [];
}>();
</script>
