/**
 * 创作页任务状态：Element Plus tag type + studio.* 文案键（与 i18n 对齐）。
 */
(function (w) {
    const TAG_TYPE = {
        pending: 'info',
        queued: 'info',
        running: 'warning',
        completed: 'success',
        failed: 'danger',
        cancelled: 'info',
    };

    const I18N_SUFFIX = {
        pending: 'pending',
        queued: 'queued',
        running: 'running',
        completed: 'completed',
        failed: 'failed',
        cancelled: 'cancelled',
    };

    function tagType(status) {
        return TAG_TYPE[status] || 'info';
    }

    /**
     * @param {string} status
     * @param {function} $tt i18n 函数（如 window.$tt）
     */
    function statusText(status, $tt) {
        const suf = I18N_SUFFIX[status] || status;
        const key = 'studio.' + suf;
        return typeof $tt === 'function' ? $tt(key) : key;
    }

    w.DQTaskStatusUi = { tagType, statusText };
})(window);
