<template>
  <div class="composer-successor-hint" role="status">
    <p class="composer-successor-hint__message">
      {{ $t('studio.successorHint', { name: successorName }) }}
    </p>
    <div class="composer-successor-hint__actions">
      <DqButton size="xs" type="primary" @click="$emit('switch')">
        {{ $t('studio.successorSwitch') }}
      </DqButton>
      <DqButton size="xs" type="text" @click="$emit('dismiss')">
        {{ $t('studio.successorDismiss') }}
      </DqButton>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  successorName: string;
}>();

defineEmits<{
  switch: [];
  dismiss: [];
}>();
</script>

<style scoped>
.composer-successor-hint {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
  padding: 8px 10px;
  border-radius: var(--dq-radius-input);
  border: 1px solid color-mix(in srgb, var(--dq-info) 35%, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-info) 8%, var(--dq-fill-secondary));
}

.composer-successor-hint__message {
  margin: 0;
  font-size: 12px;
  line-height: 1.45;
  color: var(--dq-text-secondary);
}

.composer-successor-hint__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
</style>
