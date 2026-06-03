<script setup lang="ts">
// @ts-nocheck
defineProps<{
  params: Record<string, unknown>;
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane studio-extend-pref">
    <DqPrefRow :label="$t('create.extendDirections')">
      <DqCheckboxGroup v-model="params.extend_directions">
        <DqCheckbox label="top">{{ $t('create.extendTop') }}</DqCheckbox>
        <DqCheckbox label="bottom">{{ $t('create.extendBottom') }}</DqCheckbox>
        <DqCheckbox label="left">{{ $t('create.extendLeft') }}</DqCheckbox>
        <DqCheckbox label="right">{{ $t('create.extendRight') }}</DqCheckbox>
      </DqCheckboxGroup>
    </DqPrefRow>
    <DqPrefRow :label="$t('create.extendPixels')">
      <DqInputNumber
        v-model="params.extend_pixels"
        :min="64"
        :max="2048"
        :step="64"
        size="small"
        style="width: 120px"
      />
    </DqPrefRow>
  </DqPrefPane>
</template>
