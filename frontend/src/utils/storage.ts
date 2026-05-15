/**
 * Client-side persistent storage keys (v4)
 */
export const DQ_STORAGE = Object.freeze({
  LANG: 'dq-studio.lang.v4',
  ACTIVE_PAGE: 'dq-studio.activePage.v4',
  SETTINGS_TAB: 'dq-studio.settingsTab.v4',
  IMG2IMG_REF: 'dq-studio.img2imgRef.v4',
  IMPORTED_MODELS: 'dq-studio.importedModels.v4',
  IMAGE_CREATE_PROMPT_DRAFT: 'dq-studio.imageCreatePromptDraft.v4',
  VIDEO_CREATE_PROMPT_DRAFT: 'dq-studio.videoCreatePromptDraft.v4',
  AUDIO_CREATE_PROMPT_DRAFT: 'dq-studio.audioCreatePromptDraft.v4',
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