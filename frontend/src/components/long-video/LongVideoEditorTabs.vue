<template>
  <div class="lv-editor-tabs" role="tablist" :aria-label="$tt('video.longVideoEditorTabsAria')">
    <template v-for="(tab, index) in tabs" :key="tab.id">
      <span v-if="index > 0" class="lv-editor-tabs__connector" aria-hidden="true" />
      <button
        type="button"
        role="tab"
        class="lv-editor-tabs__tab"
        :class="{ 'lv-editor-tabs__tab--active': modelValue === tab.id }"
        :aria-selected="modelValue === tab.id"
        @click="emit('update:modelValue', tab.id)"
      >
        <span
          class="lv-editor-tabs__icon"
          :class="[`lv-editor-tabs__icon--${tab.id}`, { 'lv-editor-tabs__icon--done': tab.done }]"
          aria-hidden="true"
        >
          <span v-if="tab.done" class="lv-editor-tabs__icon-mark">✓</span>
          <span v-else class="lv-editor-tabs__icon-mark">{{ index + 1 }}</span>
        </span>
        <span class="lv-editor-tabs__text">
          <span class="lv-editor-tabs__label">{{ tab.label }}</span>
          <span class="lv-editor-tabs__desc">{{ tab.desc }}</span>
        </span>
        <span v-if="tab.badge != null" class="lv-editor-tabs__badge">{{ tab.badge }}</span>
      </button>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import type { LongVideoEditorTab } from '@/types';

const props = defineProps<{
  modelValue: LongVideoEditorTab;
  castCount?: number;
  sceneEntityCount?: number;
  shotCount?: number;
  scriptDone?: boolean;
  castDone?: boolean;
  scenesDone?: boolean;
  storyboardDone?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: LongVideoEditorTab): void;
}>();

const { t: $tt } = useI18n();

const tabs = computed(() => [
  {
    id: 'settings' as const,
    label: $tt('video.longVideoProjectSettings'),
    desc: $tt('video.longVideoEditorTabSettingsDesc'),
    done: true,
    badge: null as number | null,
  },
  {
    id: 'script' as const,
    label: $tt('video.longVideoEditorTabScript'),
    desc: $tt('video.longVideoEditorTabScriptDesc'),
    done: props.scriptDone,
    badge: null as number | null,
  },
  {
    id: 'cast' as const,
    label: $tt('video.longVideoEditorTabCast'),
    desc: $tt('video.longVideoEditorTabCastDesc'),
    done: props.castDone,
    badge: props.castCount || null,
  },
  {
    id: 'scenes' as const,
    label: $tt('video.longVideoEditorTabScenes'),
    desc: $tt('video.longVideoEditorTabScenesDesc'),
    done: props.scenesDone,
    badge: props.sceneEntityCount || null,
  },
  {
    id: 'storyboard' as const,
    label: $tt('video.longVideoEditorTabStoryboard'),
    desc: $tt('video.longVideoEditorTabStoryboardDesc'),
    done: props.storyboardDone,
    badge: props.shotCount || null,
  },
]);
</script>
