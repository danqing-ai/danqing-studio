<template>
  <div ref="rootRef" class="composer-advanced-collapse">
    <div class="composer-advanced-collapse__head">
      <button
        type="button"
        class="composer-advanced-collapse__toggle"
        :aria-expanded="open"
        @click="toggle"
      >
        <DqIcon :size="14"><Tools /></DqIcon>
        <span>{{ $t('studio.advancedParams') }}</span>
        <span v-if="hasCustomParams" class="composer-advanced-collapse__dot" aria-hidden="true" />
        <span class="composer-advanced-collapse__chevron" :class="{ 'is-open': open }" aria-hidden="true">▾</span>
      </button>
      <DqButton v-if="open" type="text" size="sm" @click="$emit('reset-defaults')">
        {{ $t('create.restoreDefaults') }}
      </DqButton>
    </div>
    <div v-if="open" class="composer-advanced-collapse__body">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { Tools } from '@danqing/dq-shell';

const props = defineProps<{
  open: boolean;
  hasCustomParams?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:open', value: boolean): void;
  (e: 'reset-defaults'): void;
}>();

const rootRef = ref<HTMLElement | null>(null);

function toggle() {
  const next = !props.open;
  emit('update:open', next);
  if (next) {
    requestAnimationFrame(() => {
      rootRef.value?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    });
  }
}

defineExpose({ rootRef });
</script>

<style scoped>
.composer-advanced-collapse {
  margin-top: 4px;
  padding-top: 8px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.composer-advanced-collapse__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.composer-advanced-collapse__toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: none;
  padding: 4px 0;
  font-size: 12px;
  font-weight: 500;
  color: var(--dq-label-secondary);
  cursor: pointer;
}

.composer-advanced-collapse__toggle:hover {
  color: var(--dq-accent);
}

.composer-advanced-collapse__dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-warning);
  flex-shrink: 0;
}

.composer-advanced-collapse__chevron {
  font-size: 10px;
  line-height: 1;
  transition: transform 0.2s ease;
  color: var(--dq-label-tertiary);
}

.composer-advanced-collapse__chevron.is-open {
  transform: rotate(180deg);
}

.composer-advanced-collapse__body {
  margin-top: 10px;
}
</style>
