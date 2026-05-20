<!-- @ts-nocheck -->
<template>
  <section
    class="studio-preview-pane"
    :class="{
      'studio-preview-pane--split': splitHead,
      'studio-preview-pane--recent': recent,
    }"
  >
    <div
      v-if="title || $slots.head || $slots.actions"
      class="studio-preview-pane__head"
      :class="{ 'studio-preview-pane__head--split': splitHead || !!$slots.actions }"
    >
      <slot name="head">
        <span v-if="title" class="studio-preview-pane__title">
          <DqIcon v-if="icon" :size="16"><component :is="icon" /></DqIcon>
          {{ title }}
        </span>
      </slot>
      <slot name="actions" />
    </div>
    <slot />
  </section>
</template>

<script setup lang="ts">
defineProps({
  title: { type: String, default: '' },
  icon: { type: String, default: '' },
  splitHead: { type: Boolean, default: false },
  recent: { type: Boolean, default: false },
});
</script>
