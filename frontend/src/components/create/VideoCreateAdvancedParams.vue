<script setup lang="ts">
// @ts-nocheck
defineProps<{
  params: Record<string, unknown>;
  currentModelConfig: Record<string, unknown> | null;
}>();

const emit = defineEmits<{
  resetToDefaults: [];
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane">
    <DqPrefRow
      v-if="currentModelConfig?.parameters?.steps"
      :label="$t('studio.steps')"
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
          :max="100"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.guide_scale"
      :label="$t('video.guideScaleLabel')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.guide_scale"
            :min="currentModelConfig.parameters.guide_scale.min"
            :max="currentModelConfig.parameters.guide_scale.max"
            :step="0.1"
          />
        </div>
        <DqInputNumber
          v-model="params.guide_scale"
          :step="0.1"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.shift"
      :label="$t('video.shiftLabel')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.shift"
            :min="currentModelConfig.parameters.shift.min"
            :max="currentModelConfig.parameters.shift.max"
            :step="0.5"
          />
        </div>
        <DqInputNumber v-model="params.shift" :step="0.5" class="param-input-number" />
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.width"
      :label="$t('studio.resolution')"
    >
      <div class="studio-res-row settings-res-row">
        <DqSelect v-model="params.width" class="studio-select-w120">
          <DqOption
            v-for="w in currentModelConfig.parameters.width.options"
            :key="w"
            :label="w"
            :value="w"
          />
        </DqSelect>
        <span class="studio-res-x">x</span>
        <DqSelect v-model="params.height" class="studio-select-w120">
          <DqOption
            v-for="h in currentModelConfig.parameters.height.options"
            :key="h"
            :label="h"
            :value="h"
          />
        </DqSelect>
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.num_frames"
      :label="$t('video.numFramesLabel')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.num_frames"
            :min="currentModelConfig.parameters.num_frames.min"
            :max="currentModelConfig.parameters.num_frames.max"
            :step="currentModelConfig.parameters.num_frames.step || 1"
          />
        </div>
        <DqInputNumber
          v-model="params.num_frames"
          :min="1"
          :max="257"
          controls-position="right"
          class="param-input-number"
        />
      </div>
      <p v-if="currentModelConfig.parameters.num_frames.note" class="studio-param-note">
        {{ currentModelConfig.parameters.num_frames.note }}
      </p>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.fps"
      :label="$t('video.fpsLabel')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.fps"
            :min="currentModelConfig.parameters.fps.min"
            :max="currentModelConfig.parameters.fps.max"
          />
        </div>
        <DqInputNumber
          v-model="params.fps"
          :min="1"
          :max="60"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow v-if="currentModelConfig?.parameters?.seed_support" :label="$t('studio.seed')">
      <div class="studio-seed-row settings-seed-row">
        <DqInput v-model="params.seed" :placeholder="$t('studio.seedPlaceholder')" />
        <DqIconButton
          type="text"
          size="sm"
          class="settings-seed-dice-btn dq-icon-btn--circle"
          :label="$t('studio.seed')"
          @click="params.seed = String(Math.floor(Math.random() * 1_000_000))"
        >
          <DqIcon><refresh /></DqIcon>
        </DqIconButton>
      </div>
    </DqPrefRow>

    <DqPrefRow no-label>
      <DqButton type="text" size="sm" @click="emit('resetToDefaults')">
        <DqIcon><refresh /></DqIcon>
        {{ $t('studio.restoreDefaults') }}
      </DqButton>
    </DqPrefRow>
  </DqPrefPane>
</template>
