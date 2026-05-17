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
            :step="0.5"
          />
        </div>
        <DqInputNumber
          v-model="params.guidance"
          :step="0.5"
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
