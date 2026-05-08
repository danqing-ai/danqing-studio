/**
 * DanQing Studio v3 — Hash-based Router
 *
 * All page keys: image_create | video_create | audio_create | gallery | models | settings
 * URL hash format: #/image_create  (e.g. #/settings)
 * Persists active page to localStorage via DQ_STORAGE.ACTIVE_PAGE
 */
(function () {
    const { ref, watch } = Vue;
    const SK = window.DQ_STORAGE || {};

    const VALID_PAGES = new Set([
        'image_create',
        'video_create',
        'audio_create',
        'gallery',
        'models',
        'settings',
    ]);

    // ---- helpers ----

    function parseHash() {
        const h = window.location.hash.replace(/^#\/?/, '');
        return VALID_PAGES.has(h) ? h : null;
    }

    function getSavedPage() {
        if (!SK.ACTIVE_PAGE) return null;
        const raw = localStorage.getItem(SK.ACTIVE_PAGE);
        return VALID_PAGES.has(raw) ? raw : null;
    }

    function setHash(page) {
        const expected = '#/' + page;
        if (window.location.hash !== expected) {
            window.location.hash = expected;
        }
    }

    // ---- bootstrap ----

    const initial = parseHash() || getSavedPage() || 'image_create';
    const currentPage = ref(initial);

    // Ensure hash matches on first load
    if (!parseHash()) {
        setHash(initial);
    }

    // ---- navigate ----

    function navigate(page) {
        if (VALID_PAGES.has(page)) {
            currentPage.value = page;
        }
    }

    // ---- sync ----

    // currentPage → localStorage + hash
    watch(currentPage, function (newVal) {
        if (SK.ACTIVE_PAGE) localStorage.setItem(SK.ACTIVE_PAGE, newVal);
        setHash(newVal);
    });

    // browser back/forward → currentPage
    window.addEventListener('hashchange', function () {
        var h = parseHash();
        if (h && h !== currentPage.value) {
            currentPage.value = h;
        }
    });

    // Expose
    window.DQRouter = {
        currentPage: currentPage,
        navigate: navigate,
        VALID_PAGES: VALID_PAGES,
    };
})();
