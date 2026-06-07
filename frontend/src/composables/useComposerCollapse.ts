import { ref, watch } from 'vue';
import { DQ_STORAGE, getItem, setItem, type StorageKey } from '@/utils/storage';

export type ComposerCollapseMedia = 'image' | 'video' | 'audio';

const STORAGE_KEYS: Record<ComposerCollapseMedia, StorageKey> = {
  image: DQ_STORAGE.CANVAS_COMPOSER_COLLAPSED_IMAGE,
  video: DQ_STORAGE.CANVAS_COMPOSER_COLLAPSED_VIDEO,
  audio: DQ_STORAGE.CANVAS_COMPOSER_COLLAPSED_AUDIO,
};

export function useComposerCollapse(media: ComposerCollapseMedia) {
  const key = STORAGE_KEYS[media];
  const collapsed = ref(getItem(key) === '1');

  watch(collapsed, (value) => {
    setItem(key, value ? '1' : '0');
  });

  function setCollapsed(value: boolean) {
    collapsed.value = value;
  }

  return { collapsed, setCollapsed };
}
