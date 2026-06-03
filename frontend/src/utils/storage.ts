/**
 * Client-side persistent storage keys (v4)
 */
export const DQ_STORAGE = Object.freeze({
  LANG: 'dq-studio.lang.v4',
  THEME: 'dq-studio.theme.v4',
  ACTIVE_PAGE: 'dq-studio.activePage.v4',
  SETTINGS_TAB: 'dq-studio.settingsTab.v4',
  IMG2IMG_REF: 'dq-studio.img2imgRef.v4',
  IMPORTED_MODELS: 'dq-studio.importedModels.v4',
  IMAGE_CREATE_PROMPT_DRAFT: 'dq-studio.imageCreatePromptDraft.v4',
  VIDEO_CREATE_PROMPT_DRAFT: 'dq-studio.videoCreatePromptDraft.v4',
  AUDIO_CREATE_PROMPT_DRAFT: 'dq-studio.audioCreatePromptDraft.v4',
  MODEL_FILTER_INSTALLED: 'dq-studio.modelFilterInstalled.v4',
  MODEL_FILTER_COMMERCIAL: 'dq-studio.modelFilterCommercial.v4',
  IMAGE_LAST_SIZE: 'dq-studio.imageLastSize.v4',
  VIDEO_LAST_SIZE: 'dq-studio.videoLastSize.v4',
  // Per-mode model cache
  IMAGE_MODEL_TEXT2IMG: 'dq-studio.imageModel.text2img.v4',
  IMAGE_MODEL_IMG2IMG: 'dq-studio.imageModel.img2img.v4',
  IMAGE_MODEL_RETOUCH: 'dq-studio.imageModel.retouch.v4',
  IMAGE_MODEL_EXTEND: 'dq-studio.imageModel.extend.v4',
  IMAGE_MODEL_UPSCALE: 'dq-studio.imageModel.upscale.v4',
  VIDEO_MODEL_CREATE: 'dq-studio.videoModel.create.v4',
  VIDEO_MODEL_ANIMATE: 'dq-studio.videoModel.animate.v4',
  VIDEO_MODEL_UPSCALE: 'dq-studio.videoModel.upscale.v4',
  AUDIO_MODEL_CREATE: 'dq-studio.audioModel.create.v4',
  AUDIO_MODEL_COVER: 'dq-studio.audioModel.cover.v4',
} as const);

export type StorageKey = (typeof DQ_STORAGE)[keyof typeof DQ_STORAGE];

export function getItem(key: StorageKey): string | null {
  return localStorage.getItem(key);
}

export function setItem(key: StorageKey, value: string): void {
  localStorage.setItem(key, value);
}

export function removeItem(key: StorageKey): void {
  localStorage.removeItem(key);
}