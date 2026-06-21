import { ref, watch, type Ref } from 'vue';
import { DQ_STORAGE, getItem, setItem, type StorageKey } from '@/utils/storage';

export type ComposerDrawerMedia = 'image' | 'video' | 'audio';

const STORAGE_KEYS: Record<ComposerDrawerMedia, StorageKey> = {
  image: DQ_STORAGE.COMPOSER_DRAWER_IMAGE,
  video: DQ_STORAGE.COMPOSER_DRAWER_VIDEO,
  audio: DQ_STORAGE.COMPOSER_DRAWER_AUDIO,
};

export type ComposerDrawerOpenOptions = {
  source?: 'blank' | 'item' | 'node';
};

export function useComposerDrawer(
  media: ComposerDrawerMedia,
  mutexRefs: {
    editorDrawerOpen?: Ref<boolean>;
    onOpen?: () => void;
    onClose?: () => void;
  } = {},
) {
  const key = STORAGE_KEYS[media];
  const open = ref(getItem(key) === '1');

  watch(open, (value) => {
    setItem(key, value ? '1' : '0');
    if (value) mutexRefs.onOpen?.();
    else mutexRefs.onClose?.();
  });

  if (mutexRefs.editorDrawerOpen) {
    watch(mutexRefs.editorDrawerOpen, (isOpen) => {
      if (isOpen) open.value = false;
    });
    watch(open, (isOpen) => {
      if (isOpen && mutexRefs.editorDrawerOpen) {
        mutexRefs.editorDrawerOpen.value = false;
      }
    });
  }

  function openComposerDrawer(_opts?: ComposerDrawerOpenOptions) {
    open.value = true;
  }

  function closeComposerDrawer() {
    open.value = false;
  }

  function toggleComposerDrawer() {
    open.value = !open.value;
  }

  return {
    composerDrawerOpen: open,
    openComposerDrawer,
    closeComposerDrawer,
    toggleComposerDrawer,
  };
}
