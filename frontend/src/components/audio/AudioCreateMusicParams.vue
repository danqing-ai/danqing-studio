<script setup lang="ts">
// @ts-nocheck
import { computed } from 'vue';

const props = defineProps<{
  params: Record<string, unknown>;
  musicalKeys: string[];
  timeSignatures: { value: string; label: string }[];
  durationMin?: number;
  durationMax?: number;
  showBpm?: boolean;
  showKeyScale?: boolean;
  showTimeSignature?: boolean;
  codecStepsDef?: { min: number; max: number; default?: number } | null;
  codecGuidanceDef?: { min: number; max: number; step?: number; default?: number } | null;
}>();

/** Presets capped by registry ``duration.max`` (HeartMuLa product limit). */
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
    <DqPrefRow :label="$t('audio.duration')">
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

    <DqPrefRow v-if="showBpm !== false" :label="$t('audio.bpm')">
      <DqInputNumber
        v-model="params.bpm"
        :min="30"
        :max="300"
        controls-position="right"
        class="studio-pref-field-select"
        :placeholder="$t('audio.bpmAuto')"
      />
    </DqPrefRow>

    <DqPrefRow v-if="showKeyScale !== false" :label="$t('audio.keyScale')">
      <DqSelect
        v-model="params.key_scale"
        class="studio-pref-field-select"
        clearable
        :placeholder="$t('audio.keyScaleAuto')"
      >
        <DqOption v-for="k in musicalKeys" :key="k" :label="k" :value="k" />
      </DqSelect>
    </DqPrefRow>

    <DqPrefRow v-if="showTimeSignature !== false" :label="$t('audio.timeSignature')">
      <DqSelect
        v-model="params.time_signature"
        class="studio-pref-field-select"
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

    <DqPrefRow
      v-if="codecStepsDef"
      :label="$t('audio.codecSteps')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.codec_steps"
            :min="codecStepsDef.min"
            :max="codecStepsDef.max"
            :step="1"
          />
        </div>
        <DqInputNumber
          v-model="params.codec_steps"
          :min="codecStepsDef.min"
          :max="codecStepsDef.max"
          controls-position="right"
          class="param-input-number"
        />
      </div>
      <p class="studio-field-footnote">{{ $t('audio.codecStepsDesc') }}</p>
    </DqPrefRow>

    <DqPrefRow
      v-if="codecGuidanceDef"
      :label="$t('audio.codecGuidance')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.codec_guidance"
            :min="codecGuidanceDef.min"
            :max="codecGuidanceDef.max"
            :step="codecGuidanceDef.step ?? 0.05"
          />
        </div>
        <DqInputNumber
          v-model="params.codec_guidance"
          :step="codecGuidanceDef.step ?? 0.05"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>
  </DqPrefPane>
</template>
