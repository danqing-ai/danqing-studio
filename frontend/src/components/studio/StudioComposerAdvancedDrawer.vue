<template>
  <Teleport to="body">
    <Transition name="studio-composer-drawer-fade">
      <div
        v-if="open"
        class="studio-composer-drawer__overlay"
        @click="open = false"
      />
    </Transition>
    <Transition name="studio-composer-drawer-slide">
      <div
        v-if="open"
        class="studio-composer-drawer__panel dq-glass--panel"
        @keydown.esc="open = false"
      >
        <header class="studio-composer-drawer__header">
          <h3 class="studio-composer-drawer__title">{{ $t('studio.advancedParams') }}</h3>
          <div class="studio-composer-drawer__actions">
            <DqButton type="text" size="sm" @click="$emit('reset-defaults')">
              {{ resetLabel }}
            </DqButton>
            <DqIconButton
              type="text"
              size="sm"
              :label="$t('common.close')"
              @click="open = false"
            >
              <DqIcon :size="16"><Close /></DqIcon>
            </DqIconButton>
          </div>
        </header>
        <div class="studio-composer-drawer__body">
          <slot />
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { Close } from '@danqing/dq-shell';

defineProps<{
  resetLabel: string;
}>();

defineEmits<{
  (e: 'reset-defaults'): void;
}>();

const open = defineModel<boolean>('open', { required: true });
</script>

<style scoped>
.studio-composer-drawer__overlay {
  position: fixed;
  inset: 0;
  background: var(--dq-glass-scrim);
  z-index: 1000;
}

.studio-composer-drawer__panel {
  position: fixed;
  bottom: 0;
  left: 0;
  right: 0;
  margin: 0 auto;
  width: calc(100% - 48px);
  max-width: 720px;
  max-height: min(480px, 65vh);
  border-radius: 20px 20px 0 0;
  border-bottom: none;
  z-index: 1001;
  display: flex;
  flex-direction: column;
  background: var(--dq-glass-floating-bar-bg);
  border-color: var(--dq-glass-border-strong);
  box-shadow: var(--dq-shadow-glass);
}

.studio-composer-drawer__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px 12px;
  flex-shrink: 0;
}

.studio-composer-drawer__title {
  font-size: 14px;
  font-weight: 500;
  color: var(--dq-label-secondary);
  margin: 0;
  letter-spacing: 0.02em;
}

.studio-composer-drawer__actions {
  display: flex;
  align-items: center;
  gap: 4px;
}

.studio-composer-drawer__body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 24px 28px;
  -webkit-overflow-scrolling: touch;
}

.studio-composer-drawer-fade-enter-active,
.studio-composer-drawer-fade-leave-active {
  transition: opacity 0.25s ease;
}

.studio-composer-drawer-fade-enter-from,
.studio-composer-drawer-fade-leave-to {
  opacity: 0;
}

.studio-composer-drawer-slide-enter-active,
.studio-composer-drawer-slide-leave-active {
  transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.studio-composer-drawer-slide-enter-from,
.studio-composer-drawer-slide-leave-to {
  transform: translateY(100%);
}

@media (prefers-reduced-transparency: reduce) {
  .studio-composer-drawer__panel {
    -webkit-backdrop-filter: none;
    backdrop-filter: none;
    background: var(--dq-glass-grouped-bg-solid);
  }
}
</style>
