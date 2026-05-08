/**
 * 客户端持久化键（v3 专用，不读取旧 danqing-* 键）
 */
(function (w) {
    w.DQ_STORAGE = Object.freeze({
        LANG: 'dq-studio.lang.v3',
        ACTIVE_PAGE: 'dq-studio.activePage.v3',
        SETTINGS_TAB: 'dq-studio.settingsTab.v3',
        IMG2IMG_REF: 'dq-studio.img2imgRef.v3',
        IMPORTED_MODELS: 'dq-studio.importedModels.v3',
        IMAGE_CREATE_PROMPT_DRAFT: 'dq-studio.imageCreatePromptDraft.v3',
        VIDEO_CREATE_PROMPT_DRAFT: 'dq-studio.videoCreatePromptDraft.v3',
    });
})(window);
