<script setup lang="ts">
import { computed } from 'vue';
import { $md } from '@/utils/i18n';

const props = defineProps<{
  modelConfig: Record<string, unknown> | null | undefined;
}>();

const text = computed(() =>
  $md(props.modelConfig as { description?: string | { zh?: string; en?: string }; description_en?: string } | null),
);
</script>

<template>
  <p v-if="text" class="studio-model-desc">{{ text }}</p>
</template>
