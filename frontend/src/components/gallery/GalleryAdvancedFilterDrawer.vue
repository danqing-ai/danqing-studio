<script setup lang="ts">
const open = defineModel<boolean>('visible', { required: true });
const filterDateRange = defineModel<Date[] | null>('filterDateRange');
const filterMinWidth = defineModel<number>('filterMinWidth', { required: true });
const filterActions = defineModel<string[]>('filterActions', { required: true });

const emit = defineEmits<{
  reset: [];
  apply: [];
}>();
</script>

<template>
  <DqDrawer
    v-model:open="open"
    class="gallery-filter-drawer"
    :title="$t('gallery.advancedFilter')"
    size="min(360px, 92vw)"
    direction="rtl"
  >
    <DqPrefPane class="settings-pref-pane-form settings-pref-pane-form--dialog gallery-drawer-pref">
      <DqPrefRow :label="$t('gallery.dateRange')" stacked>
        <DqDatePicker
          v-model="filterDateRange"
          type="daterange"
          size="small"
          class="studio-w-full"
          :start-placeholder="$t('gallery.startDate')"
          :end-placeholder="$t('gallery.endDate')"
        />
      </DqPrefRow>

      <DqPrefRow :label="$t('gallery.minResolution')" stacked>
        <DqSlider v-model="filterMinWidth" :min="256" :max="2048" :step="64" show-stops />
        <p class="gallery-drawer-hint">≥ {{ filterMinWidth }}px</p>
      </DqPrefRow>

      <DqPrefRow :label="$t('gallery.actionType')" stacked>
        <DqCheckboxGroup v-model="filterActions">
          <DqCheckbox label="create">{{ $t('gallery.actionCreate') }}</DqCheckbox>
          <DqCheckbox label="rewrite">{{ $t('gallery.actionRewrite') }}</DqCheckbox>
          <DqCheckbox label="upscale">{{ $t('gallery.actionUpscale') }}</DqCheckbox>
        </DqCheckboxGroup>
      </DqPrefRow>
    </DqPrefPane>

    <template #footer>
      <div class="gallery-drawer-footer">
        <DqButton size="sm" @click="emit('reset')">{{ $t('gallery.resetFilters') }}</DqButton>
        <DqButton type="primary" size="sm" @click="emit('apply')">{{ $t('gallery.apply') }}</DqButton>
      </div>
    </template>
  </DqDrawer>
</template>
