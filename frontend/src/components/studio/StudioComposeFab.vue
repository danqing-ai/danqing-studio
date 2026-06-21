<template>
  <button
    type="button"
    class="studio-compose-fab dq-glass--panel"
    :aria-label="actionLabel"
    @click="$emit('open')"
  >
    <span v-if="busy" class="studio-compose-fab__dot" aria-hidden="true" />
    <span class="studio-compose-fab__action">{{ actionLabel }}</span>
  </button>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = withDefaults(
  defineProps<{
    busy?: boolean;
    media?: 'image' | 'video' | 'audio';
  }>(),
  { media: 'image' },
);

defineEmits<{
  (e: 'open'): void;
}>();

const { t: $t } = useI18n();

const actionLabel = computed(() => {
  const key =
    props.media === 'video'
      ? 'studio.openComposerVideo'
      : props.media === 'audio'
        ? 'studio.openComposerAudio'
        : 'studio.openComposerImage';
  return $t(key);
});
</script>

<style scoped>
.studio-compose-fab {
  position: fixed;
  left: calc(var(--dq-shell-sidebar-width, 60px) + 50%);
  bottom: calc(16px + env(safe-area-inset-bottom, 0px));
  transform: translateX(-50%);
  z-index: 90;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  max-width: min(560px, calc(100vw - var(--dq-shell-sidebar-width, 60px) - 48px));
  padding: 10px 14px 10px 16px;
  border-radius: 999px;
  border: 0.5px solid var(--dq-glass-border-strong, var(--dq-border-subtle));
  cursor: pointer;
  pointer-events: auto;
  text-align: left;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.studio-compose-fab:hover {
  border-color: var(--dq-border-strong);
  box-shadow: var(--dq-shadow-glass);
}

.studio-compose-fab__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--dq-accent);
  flex-shrink: 0;
  animation: studio-compose-fab-pulse 1.4s ease-in-out infinite;
}

@keyframes studio-compose-fab-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}

.studio-compose-fab__action {
  font-size: 13px;
  font-weight: 600;
  color: var(--dq-accent);
  white-space: nowrap;
  flex-shrink: 0;
}
</style>
