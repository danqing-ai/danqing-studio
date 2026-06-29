<template>
  <section v-if="selectedRow" class="lv-scene-look">
    <div class="lv-scene-look__label">{{ $tt('video.longVideoSceneLookTitle') }}</div>
    <div class="lv-scene-look__row">
      <span class="lv-scene-look__name">{{ selectedRow.name }}</span>
      <DqSelect
        :id="selectId"
        :model-value="selectedRow.lookId"
        size="sm"
        class="lv-scene-look__select"
        @update:model-value="onLookChange"
      >
        <DqOption
          v-for="opt in selectedRow.lookOptions"
          :key="opt.id"
          :label="opt.label"
          :value="opt.id"
        />
      </DqSelect>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { LongVideoScene, LongVideoShotSceneLook } from '@/types';
import { inferShotSceneLookFromBeat, formatSceneLookOptionLabel } from '@/utils/longVideoProject';
import { useI18n } from 'vue-i18n';

const { locale } = useI18n();

const props = defineProps<{
  scenes: LongVideoScene[];
  sceneLook?: LongVideoShotSceneLook;
  beatText: string;
  selectIdPrefix?: string;
}>();

const emit = defineEmits<(e: 'update:sceneLook', value: LongVideoShotSceneLook | undefined) => void>();

const resolved = computed(() =>
  props.sceneLook ??
  (props.scenes.length ? inferShotSceneLookFromBeat(props.beatText, props.scenes) : undefined),
);

const selectedRow = computed(() => {
  const binding = resolved.value;
  if (!binding || !props.scenes.length) return null;
  const sc = props.scenes.find((s) => s.id === binding.scene_id);
  if (!sc) return null;
  return {
    name: sc.name,
    lookId: binding.look_id,
    lookOptions: sc.looks.map((lk) => ({
      id: lk.id,
      label: formatSceneLookOptionLabel(lk, locale.value),
    })),
  };
});

const selectId = computed(() => `${props.selectIdPrefix ?? 'lv-scene'}-look`);

function onLookChange(lookId: string) {
  const binding = resolved.value;
  if (!binding || !lookId) return;
  emit('update:sceneLook', { scene_id: binding.scene_id, look_id: lookId });
}
</script>

<style scoped>
.lv-scene-look {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-scene-look__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.lv-scene-look__row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.lv-scene-look__name {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-primary);
  min-width: 4em;
}

.lv-scene-look__select {
  flex: 1;
}
</style>
