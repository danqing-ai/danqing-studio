<!-- @ts-nocheck -->
<template>
  <DqSurfaceCard
    class="studio-surface-card studio-card-mb studio-current-preview"
    :class="[
      kind ? `studio-current-preview--${kind}` : '',
      { 'studio-current-preview--filled': filled, 'studio-current-preview--flush': flush },
    ]"
  >
    <template #header>
      <div class="card-title card-title--split studio-current-preview__header">
        <span class="studio-current-preview__title">
          <slot name="icon">
            <DqIcon v-if="icon"><component :is="icon" /></DqIcon>
          </slot>
          {{ $t('studio.currentPreview') }}
        </span>
        <DqTag v-if="statusLabel" size="small" :type="statusType">{{ statusLabel }}</DqTag>
      </div>
    </template>

    <div v-if="filled" class="studio-current-preview__stage">
      <slot />
    </div>
    <div v-else class="studio-current-preview__empty">
      <div class="studio-current-preview__empty-icon" aria-hidden="true">
        <slot name="empty-icon">
          <DqIcon :size="40"><component :is="emptyIcon || icon || 'picture'" /></DqIcon>
        </slot>
      </div>
      <DqEmpty :description="emptyText || $t('studio.noPreview')" />
    </div>

    <p
      v-if="filled && caption"
      class="studio-current-preview__caption"
      :title="caption"
    >
      {{ caption }}
    </p>
    <div v-if="filled && $slots.footer" class="studio-current-preview__footer">
      <slot name="footer" />
    </div>
  </DqSurfaceCard>
</template>

<script setup lang="ts">
defineProps({
  kind: { type: String, default: 'neutral' },
  icon: { type: String, default: 'picture' },
  emptyIcon: { type: String, default: '' },
  filled: { type: Boolean, default: false },
  flush: { type: Boolean, default: false },
  caption: { type: String, default: '' },
  emptyText: { type: String, default: '' },
  statusLabel: { type: String, default: '' },
  statusType: { type: String, default: 'info' },
});
</script>
