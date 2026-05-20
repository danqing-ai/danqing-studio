<script setup lang="ts">
// @ts-nocheck
import { computed } from 'vue';

const props = defineProps<{
  params: Record<string, unknown>;
  musicalKeys: string[];
  timeSignatures: { value: string; label: string }[];
  durationMin?: number;
  durationMax?: number;
}>();

const durationPresets = [30, 60, 90, 120, 180, 240, 300];

const durationSegmentOptions = computed(() => {
  const min = props.durationMin ?? 10;
  const max = props.durationMax ?? 600;
  return durationPresets
    .filter((s) => s >= min && s <= max)
    .map((s) => ({ label: `${s}s`, value: s }));
});
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane">
    <DqPrefRow :label="$t('audio.duration')" stacked>
      <DqSegmented
        v-if="durationSegmentOptions.length > 0"
        v-model="params.duration"
        class="dq-segmented--sm studio-audio-duration-segmented"
        :options="durationSegmentOptions"
      />
      <DqInputNumber
        v-model="params.duration"
        :min="durationMin ?? 10"
        :max="durationMax ?? 600"
        :step="10"
        controls-position="right"
        class="studio-w-full studio-audio-duration-input"
      />
      <p class="studio-field-footnote">{{ $t('audio.durationHint', { max: durationMax ?? 600 }) }}</p>
    </DqPrefRow>

    <DqPrefRow :label="$t('audio.bpm')" stacked>
      <DqInputNumber
        v-model="params.bpm"
        :min="30"
        :max="300"
        controls-position="right"
        class="studio-w-full"
        :placeholder="$t('audio.bpmAuto')"
      />
    </DqPrefRow>

    <DqPrefRow :label="$t('audio.keyScale')" stacked>
      <DqSelect
        v-model="params.key_scale"
        class="studio-w-full"
        clearable
        :placeholder="$t('audio.keyScaleAuto')"
      >
        <DqOption v-for="k in musicalKeys" :key="k" :label="k" :value="k" />
      </DqSelect>
    </DqPrefRow>

    <DqPrefRow :label="$t('audio.timeSignature')" stacked>
      <DqSelect
        v-model="params.time_signature"
        class="studio-w-full"
        clearable
        :placeholder="$t('audio.timeSignatureAuto')"
      >
        <DqOption
          v-for="ts in timeSignatures"
          :key="ts.value"
          :label="ts.label"
          :value="ts.value"
        />
      </DqSelect>
    </DqPrefRow>
  </DqPrefPane>
</template>
