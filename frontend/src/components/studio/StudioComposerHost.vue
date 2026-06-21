<template>
  <StudioComposerDrawer
    :open="open"
    :title="drawerTitle"
    @update:open="$emit('update:open', $event)"
  >
    <slot />
  </StudioComposerDrawer>
</template>

<script setup lang="ts">
import StudioComposerDrawer from './StudioComposerDrawer.vue';

defineProps<{
  open: boolean;
  drawerTitle: string;
}>();

defineEmits<{
  (e: 'update:open', value: boolean): void;
}>();
</script>
