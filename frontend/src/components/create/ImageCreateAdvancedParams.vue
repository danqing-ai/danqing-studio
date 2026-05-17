<script setup lang="ts">
// @ts-nocheck
import { $mn } from '@/utils/i18n';
import AssetPicker from '@/components/asset/AssetPicker.vue';

defineProps<{
  params: Record<string, unknown>;
  currentModelConfig: Record<string, unknown> | null;
  editMode: string;
  compatibleLoras: Record<string, unknown>[];
  compatibleControlNets: Record<string, unknown>[];
  controlImageSrc: string;
  recentImages: unknown[];
}>();

const emit = defineEmits<{
  resetToDefaults: [];
  controlAssetPick: [payload: unknown];
  removeControlImage: [];
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
            :min="Number(currentModelConfig.parameters.steps.min)"
            :max="Number(currentModelConfig.parameters.steps.max)"
          />
        </div>
        <DqInputNumber
          v-model="params.steps"
          :min="Number(currentModelConfig.parameters.steps.min)"
          :max="Number(currentModelConfig.parameters.steps.max)"
          controls-position="right"
          class="param-input-number"
        />
      </div>
      <p v-if="currentModelConfig.parameters.steps.note" class="studio-param-note">
        {{ currentModelConfig.parameters.steps.note }}
      </p>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.guidance"
      :label="$t('create.guidanceLabel')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.guidance"
            :min="Number(currentModelConfig.parameters.guidance.min)"
            :max="Number(currentModelConfig.parameters.guidance.max)"
            :step="Number(currentModelConfig.parameters.guidance.step) || 0.1"
          />
        </div>
        <DqInputNumber
          v-model="params.guidance"
          :min="Number(currentModelConfig.parameters.guidance.min)"
          :max="Number(currentModelConfig.parameters.guidance.max)"
          :step="Number(currentModelConfig.parameters.guidance.step) || 0.1"
          controls-position="right"
          class="param-input-number"
        />
      </div>
      <p v-if="currentModelConfig.parameters.guidance.note" class="studio-param-note">
        {{ currentModelConfig.parameters.guidance.note }}
      </p>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.scheduler?.options"
      :label="String(currentModelConfig.parameters.scheduler.label || $t('create.schedulerLabel'))"
    >
      <DqSelect v-model="params.scheduler" class="studio-pref-field-select">
        <DqOption
          v-for="opt in currentModelConfig.parameters.scheduler.options"
          :key="String(opt)"
          :label="String(opt)"
          :value="opt"
        />
      </DqSelect>
      <p v-if="currentModelConfig.parameters.scheduler.note" class="studio-param-note">
        {{ currentModelConfig.parameters.scheduler.note }}
      </p>
    </DqPrefRow>

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.width && currentModelConfig?.parameters?.height"
      :label="$t('studio.resolution')"
    >
      <div class="studio-res-row settings-res-row">
        <DqSelect v-model="params.width" class="studio-select-w120">
          <DqOption
            v-for="w in currentModelConfig.parameters.width.options"
            :key="String(w)"
            :label="String(w)"
            :value="w"
          />
        </DqSelect>
        <span class="studio-res-x">x</span>
        <DqSelect v-model="params.height" class="studio-select-w120">
          <DqOption
            v-for="h in currentModelConfig.parameters.height.options"
            :key="String(h)"
            :label="String(h)"
            :value="h"
          />
        </DqSelect>
      </div>
    </DqPrefRow>

    <DqPrefRow
      v-if="editMode === 'image_editing' && currentModelConfig?.parameters?.strength"
      :label="$t('create.strengthLabel')"
      class="settings-pref-row--slider"
    >
      <div class="param-control-row settings-pref-slider-row">
        <div class="param-slider">
          <DqSlider
            v-model="params.strength"
            :min="Number(currentModelConfig.parameters.strength.min)"
            :max="Number(currentModelConfig.parameters.strength.max)"
            :step="Number(currentModelConfig.parameters.strength.step) || 0.05"
          />
        </div>
        <DqInputNumber
          v-model="params.strength"
          :min="0"
          :max="1"
          :step="Number(currentModelConfig.parameters.strength.step) || 0.05"
          controls-position="right"
          class="param-input-number"
        />
      </div>
      <p v-if="currentModelConfig.parameters.strength.note" class="studio-param-note">
        {{ currentModelConfig.parameters.strength.note }}
      </p>
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

    <DqPrefRow
      v-if="currentModelConfig?.parameters?.lora_support"
      :label="$t('studio.loraLabel')"
    >
      <DqSelect v-model="params.lora" class="studio-pref-field-select" clearable :placeholder="$t('studio.noLora')">
        <DqOption
          v-for="l in compatibleLoras"
          :key="String(l.id)"
          :label="$mn(l, String(l.id))"
          :value="l.id"
        />
      </DqSelect>
      <template v-if="params.lora">
        <p class="settings-form-hint settings-form-hint--tight">{{ $t('create.loraScale') }}</p>
        <div class="param-control-row settings-pref-slider-row">
          <div class="param-slider">
            <DqSlider v-model="params.lora_scale" :min="0" :max="2" :step="0.05" />
          </div>
          <DqInputNumber
            v-model="params.lora_scale"
            :min="0"
            :max="2"
            :step="0.05"
            controls-position="right"
            class="param-input-number"
          />
        </div>
      </template>
    </DqPrefRow>

    <template v-if="compatibleControlNets.length">
      <DqPrefRow :label="$t('studio.controlNet')">
        <DqSelect v-model="params.controlnet" class="studio-pref-field-select" clearable :placeholder="$t('studio.noControlNet')">
          <DqOption
            v-for="n in compatibleControlNets"
            :key="String(n.key)"
            :label="$mn(n, String(n.key))"
            :value="String(n.key)"
          />
        </DqSelect>
      </DqPrefRow>
      <DqPrefRow v-if="params.controlnet" :label="$t('create.controlNetStrengthLabel')" class="settings-pref-row--slider">
        <div class="param-control-row settings-pref-slider-row">
          <div class="param-slider">
            <DqSlider v-model="params.controlnet_strength" :min="0" :max="2" :step="0.05" />
          </div>
          <DqInputNumber
            v-model="params.controlnet_strength"
            :min="0"
            :max="2"
            :step="0.05"
            controls-position="right"
            class="param-input-number"
          />
        </div>
      </DqPrefRow>
      <DqPrefRow v-if="params.controlnet" :label="$t('studio.uploadControlImage')" stacked>
        <asset-picker accept-kind="image" :recent-gallery="recentImages" @pick="emit('controlAssetPick', $event)" />
        <div v-if="controlImageSrc" class="studio-control-preview">
          <img :src="controlImageSrc" alt="" />
          <DqButton type="danger" size="sm" @click="emit('removeControlImage')">
            <DqIcon><delete /></DqIcon>
            {{ $t('common.delete') }}
          </DqButton>
        </div>
      </DqPrefRow>
    </template>

    <DqPrefRow no-label class="studio-create-pref-reset-row">
      <DqButton type="text" size="sm" class="studio-restore-defaults-btn" @click="emit('resetToDefaults')">
        <DqIcon><refresh /></DqIcon>
        {{ $t('studio.restoreDefaults') }}
      </DqButton>
    </DqPrefRow>
  </DqPrefPane>
</template>
