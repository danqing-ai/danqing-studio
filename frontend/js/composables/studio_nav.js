/**
 * 创作页 → 主导航：与 `app.js` `VALID_PAGES` 一致（settings / models 等）。
 */
(function (w) {
    function go(page) {
        if (!page || typeof page !== 'string') return;
        window.dispatchEvent(new CustomEvent('navigate', { detail: page }));
    }

    w.DQStudioNav = {
        go,
        goSettings() {
            go('settings');
        },
        goModels() {
            go('models');
        },
    };
})(window);
