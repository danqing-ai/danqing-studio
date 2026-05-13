/**
 * Creation page → main nav: consistent with `app.js` `VALID_PAGES` (settings / models, etc.).
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
