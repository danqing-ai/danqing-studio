import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const VOCAL_LANGUAGE_CODES = ['en', 'zh', 'ja', 'ko', 'fr', 'de', 'es', 'pt'] as const;

export function useVocalLanguageOptions() {
  const { t } = useI18n();

  return computed(() =>
    VOCAL_LANGUAGE_CODES.map((code) => ({
      label: t(`audio.vocalLang.${code}`),
      value: code,
    })),
  );
}
