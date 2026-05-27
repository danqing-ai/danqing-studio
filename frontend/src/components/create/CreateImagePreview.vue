<!-- @ts-nocheck -->
<template>
  <div
    class="studio-preview-stage studio-preview-stage--image"
    :style="{ '--dq-preview-hue': String(hue) }"
  >
    <img
      class="studio-preview-stage__media"
      :src="src"
      :alt="alt || 'Preview'"
      :loading="eagerLoad ? 'eager' : 'lazy'"
    />
    <div class="studio-preview-stage__shade" aria-hidden="true" />
    <div class="studio-preview-stage__toolbar">
      <DqIconButton
        type="text"
        size="sm"
        class="studio-preview-stage__tool"
        :label="$t('gallery.download')"
        @click="emit('download')"
      >
        <DqIcon><download /></DqIcon>
      </DqIconButton>
      <DqIconButton
        v-if="showExpand"
        type="text"
        size="sm"
        class="studio-preview-stage__tool"
        :label="$t('studio.previewExpand')"
        @click="emit('expand')"
      >
        <DqIcon><zoom-in /></DqIcon>
      </DqIconButton>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps({
  src: { type: String, required: true },
  alt: { type: String, default: '' },
  hue: { type: Number, default: 210 },
  showExpand: { type: Boolean, default: true },
  eagerLoad: { type: Boolean, default: false },
});

const emit = defineEmits(['download', 'expand']);
</script>
