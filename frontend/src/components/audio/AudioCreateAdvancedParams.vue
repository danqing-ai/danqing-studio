<script setup lang="ts">
// @ts-nocheck
defineProps<{
  params: Record<string, unknown>;
  currentModelConfig: Record<string, unknown> | null;
  audioFormats: string[];
}>();

const emit = defineEmits<{
  restoreDefaults: [];
  randomizeSeed: [];
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane">
    <DqPrefRow
      v-if="currentModelConfig?.parameters?.steps"
      :label="$t('audio.sampleQuality')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.steps"
            :min="currentModelConfig.parameters.steps.min"
            :max="currentModelConfig.parameters.steps.max"
          />
        </div>
        <DqInputNumber
          v-model="params.steps"
          :min="1"
          :max="200"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.guidance"
      :label="$t('audio.guidance')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.guidance"
            :min="currentModelConfig.parameters.guidance.min"
            :max="currentModelConfig.parameters.guidance.max"
            :step="currentModelConfig.parameters.guidance.step ?? 0.5"
          />
        </div>
        <DqInputNumber
          v-model="params.guidance"
          :step="currentModelConfig.parameters.guidance.step ?? 0.5"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.temperature"
      :label="$t('audio.temperature')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.temperature"
            :min="currentModelConfig.parameters.temperature.min"
            :max="currentModelConfig.parameters.temperature.max"
            :step="currentModelConfig.parameters.temperature.step ?? 0.1"
          />
        </div>
        <DqInputNumber
          v-model="params.temperature"
          :step="currentModelConfig.parameters.temperature.step ?? 0.1"
          controls-position="right"
          class="param-input-number"
        />
      </div>
      <p class="studio-field-footnote">{{ $t('audio.temperatureDesc') }}</p>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.top_k"
      :label="$t('audio.topK')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.top_k"
            :min="currentModelConfig.parameters.top_k.min"
            :max="currentModelConfig.parameters.top_k.max"
            :step="1"
          />
        </div>
        <DqInputNumber
          v-model="params.top_k"
          :min="currentModelConfig.parameters.top_k.min"
          :max="currentModelConfig.parameters.top_k.max"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.codec_steps"
      :label="$t('audio.codecSteps')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.codec_steps"
            :min="currentModelConfig.parameters.codec_steps.min"
            :max="currentModelConfig.parameters.codec_steps.max"
            :step="1"
          />
        </div>
        <DqInputNumber
          v-model="params.codec_steps"
          :min="currentModelConfig.parameters.codec_steps.min"
          :max="currentModelConfig.parameters.codec_steps.max"
          controls-position="right"
          class="param-input-number"
        />
      </div>
      <p class="studio-field-footnote">{{ $t('audio.codecStepsDesc') }}</p>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.codec_guidance"
      :label="$t('audio.codecGuidance')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.codec_guidance"
            :min="currentModelConfig.parameters.codec_guidance.min"
            :max="currentModelConfig.parameters.codec_guidance.max"
            :step="currentModelConfig.parameters.codec_guidance.step ?? 0.05"
          />
        </div>
        <DqInputNumber
          v-model="params.codec_guidance"
          :step="currentModelConfig.parameters.codec_guidance.step ?? 0.05"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow :label="$t('audio.seed')">
      <div class="studio-seed-row settings-seed-row">
        <DqInput v-model="params.seed" :placeholder="$t('audio.randomSeed')" />
        <DqIconButton
          type="text"
          size="sm"
          class="settings-seed-dice-btn dq-icon-btn--circle"
          :label="$t('studio.seed')"
          @click="emit('randomizeSeed')"
        >
          <DqIcon><refresh /></DqIcon>
        </DqIconButton>
      </div>
    </DqPrefRow>

    <DqPrefRow :label="$t('audio.batchCount')" stacked>
      <DqInputNumber
        v-model="params.n"
        :min="1"
        :max="8"
        controls-position="right"
        class="studio-w-full"
      />
    </DqPrefRow>

    <DqPrefRow :label="$t('audio.audioFormat')" stacked>
      <DqSelect v-model="params.audio_format" class="studio-w-full">
        <DqOption v-for="f in audioFormats" :key="f" :label="f" :value="f" />
      </DqSelect>
    </DqPrefRow>

    <DqPrefRow no-label>
      <DqButton type="text" size="sm" @click="emit('restoreDefaults')">
        <DqIcon><refresh /></DqIcon>
        {{ $t('studio.restoreDefaults') }}
      </DqButton>
    </DqPrefRow>
  </DqPrefPane>
</template>
