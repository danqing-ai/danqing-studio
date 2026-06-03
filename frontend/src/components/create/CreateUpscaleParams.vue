<script setup lang="ts">
// @ts-nocheck
defineProps<{
  params: Record<string, unknown>;
  media: 'image' | 'video';
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane">
    <DqPrefRow :label="$t('create.upscaleScale')">
      <DqSelect v-model="params.upscale_scale" size="small" style="width: 120px">
        <DqOption label="2×" :value="2" />
        <DqOption label="4×" :value="4" />
      </DqSelect>
    </DqPrefRow>

    <DqPrefRow
      :label="$t('create.upscaleDenoise')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider v-model="params.upscale_denoise" :min="0" :max="1" :step="0.05" />
        </div>
        <DqInputNumber
          v-model="params.upscale_denoise"
          :min="0"
          :max="1"
          :step="0.05"
          controls-position="right"
          class="param-input-number"
        />
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="media === 'image'"
      :label="$t('create.upscaleTile')"
    >
      <DqInputNumber
        v-model="params.upscale_tile"
        :min="256"
        :max="4096"
        :step="128"
        size="small"
        style="width: 120px"
      />
    </DqPrefRow>

    <DqPrefRow
      v-if="media === 'video'"
      :label="$t('video.maxFramesLabel')"
    >
      <DqInputNumber
        v-model="params.upscale_max_frames"
        :min="1"
        :max="4000"
        :step="1"
        size="small"
        style="width: 120px"
      />
    </DqPrefRow>

    <DqPrefRow v-if="media === 'video'" :label="$t('studio.seed')">
      <div class="studio-seed-row settings-seed-row">
        <DqInput v-model="params.seed" :placeholder="$t('studio.seedPlaceholder')" size="small" style="width: 120px" />
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
  </DqPrefPane>
</template>
