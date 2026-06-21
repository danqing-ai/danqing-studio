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
    <DqTag
      v-if="recommended"
      :size="size"
      type="success"
      :effect="effect"
      class="model-license-badges__recommended"
    >
      {{ $t('download.recommendedBadge') }}
    </DqTag>
    <DqTag
      v-if="showCommercial"
      :size="size"
      :effect="effect"
      class="model-license-badges__commercial commercial-badge"
      :title="$t('download.commercialUseBadgeTip')"
    >
      {{ $t('download.commercialUseBadge') }}
    </DqTag>
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

.commercial-badge.dq-tag {
  color: var(--dq-color-white);
  background: color-mix(in srgb, var(--dq-accent) 88%, #000);
  border-color: color-mix(in srgb, var(--dq-accent) 70%, transparent);
}
</style>
