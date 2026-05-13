/**
 * Creation page task status: Element Plus tag type + studio.* i18n key (aligned with i18n).
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
     * @param {function} $tt i18n function (e.g. window.$tt)
     */
    function statusText(status, $tt) {
        const suf = I18N_SUFFIX[status] || status;
        const key = 'studio.' + suf;
        return typeof $tt === 'function' ? $tt(key) : key;
    }

    w.DQTaskStatusUi = { tagType, statusText };
})(window);
