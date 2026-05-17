<script setup lang="ts">
import { computed } from 'vue';
import { isCommercialUseAllowed } from '@/utils/modelLicense';

const props = withDefaults(
  defineProps<{
    recommended?: boolean;
    commercialUseAllowed?: boolean | null;
    stacked?: boolean;
    effect?: 'dark' | 'light' | 'plain';
    size?: 'small' | 'default' | 'large';
  }>(),
  {
    recommended: false,
    commercialUseAllowed: null,
    stacked: false,
    effect: 'dark',
    size: 'small',
  },
);

const showCommercial = computed(() => isCommercialUseAllowed(props.commercialUseAllowed));
const showAny = computed(() => props.recommended || showCommercial.value);
</script>

<template>
  <span
    v-if="showAny"
    class="model-license-badges"
    :class="{ 'model-license-badges--stacked': stacked }"
  >
    <el-tag
      v-if="recommended"
      :size="size"
      type="success"
      :effect="effect"
      class="model-license-badges__recommended"
    >
      {{ $t('download.recommendedBadge') }}
    </el-tag>
    <el-tooltip
      v-if="showCommercial"
      :content="$t('download.commercialUseBadgeTip')"
      placement="top"
      :show-after="400"
    >
      <el-tag
        :size="size"
        :effect="effect"
        class="model-license-badges__commercial commercial-badge"
      >
        {{ $t('download.commercialUseBadge') }}
      </el-tag>
    </el-tooltip>
  </span>
</template>

<style scoped>
.model-license-badges {
  display: inline-flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}

.model-license-badges--stacked {
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}

.commercial-badge {
  --el-tag-bg-color: color-mix(in srgb, #4f6ef7 88%, #000);
  --el-tag-border-color: color-mix(in srgb, #4f6ef7 70%, transparent);
  --el-tag-text-color: #fff;
}
</style>
