<script setup lang="ts">
// @ts-nocheck
defineProps<{
  params: Record<string, unknown>;
  musicalKeys: string[];
  timeSignatures: { value: string; label: string }[];
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane">
    <DqPrefRow :label="$t('audio.duration')" stacked>
      <DqSegmented
        v-model="params.duration"
        class="dq-segmented--sm"
        :options="[
          { label: '30s', value: 30 },
          { label: '60s', value: 60 },
          { label: '90s', value: 90 },
          { label: '120s', value: 120 },
        ]"
      />
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
