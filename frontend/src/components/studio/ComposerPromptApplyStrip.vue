<template>
  <div class="composer-prompt-apply" role="status">
    <p class="composer-prompt-apply__preview" :title="preview">{{ truncated }}</p>
    <div class="composer-prompt-apply__actions">
      <DqTooltip :content="$t('create.composerTip.llmApplyReplace')" placement="top">
        <span class="composer-prompt-apply__action">
          <DqButton type="primary" @click="$emit('replace')">
            {{ $t('create.llmApplyReplace') }}
          </DqButton>
        </span>
      </DqTooltip>
      <DqTooltip :content="$t('create.composerTip.llmApplyAppend')" placement="top">
        <span class="composer-prompt-apply__action">
          <DqButton size="sm" @click="$emit('append')">
            {{ $t('create.llmApplyAppend') }}
          </DqButton>
        </span>
      </DqTooltip>
      <DqTooltip :content="$t('create.composerTip.llmApplyDismiss')" placement="top">
        <span class="composer-prompt-apply__action">
          <DqButton size="sm" type="text" @click="$emit('dismiss')">
            {{ $t('create.llmApplyDismiss') }}
          </DqButton>
        </span>
      </DqTooltip>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  preview: string;
}>();

defineEmits<{
  replace: [];
  append: [];
  dismiss: [];
}>();

const { t: $t } = useI18n();

const truncated = computed(() => {
  const text = props.preview.trim();
  if (text.length <= 96) return text;
  return `${text.slice(0, 96)}…`;
});
</script>

<style scoped>
.composer-prompt-apply {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 6px;
  padding: 8px 10px;
  border-radius: var(--dq-radius-input);
  border: 1px solid var(--dq-border-subtle);
  background: var(--dq-fill-secondary);
}

.composer-prompt-apply__preview {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-secondary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.composer-prompt-apply__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.composer-prompt-apply__action {
  display: inline-flex;
}
</style>
