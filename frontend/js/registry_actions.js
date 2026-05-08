/**
 * models_registry v2 `actions` — key present with non-null value means supported.
 */
(function (w) {
    function hasAction(actions, key) {
        if (!actions || typeof actions !== 'object') return false;
        return Object.prototype.hasOwnProperty.call(actions, key) && actions[key] != null;
    }

    function imageSupportsCreate(actions) {
        return hasAction(actions, 'create');
    }

    function imageSupportsUpscale(actions) {
        return hasAction(actions, 'upscale');
    }

    function imageEditingMatches(actions, subMode) {
        if (subMode === 'inpainting') {
            return hasAction(actions, 'retouch') || hasAction(actions, 'rewrite');
        }
        if (subMode === 'outpainting') {
            return hasAction(actions, 'extend') || hasAction(actions, 'retouch');
        }
        return hasAction(actions, 'rewrite');
    }

    function imageModelRow(config) {
        return config && config.media === 'image';
    }

    function videoModelRow(config) {
        return config && config.media === 'video';
    }

    function videoSupportsAnimate(actions) {
        return hasAction(actions, 'animate');
    }

    function videoSupportsCreate(actions) {
        return hasAction(actions, 'create');
    }

    w.RegistryActions = {
        hasAction: hasAction,
        imageSupportsCreate: imageSupportsCreate,
        imageSupportsUpscale: imageSupportsUpscale,
        imageEditingMatches: imageEditingMatches,
        imageModelRow: imageModelRow,
        videoModelRow: videoModelRow,
        videoSupportsCreate: videoSupportsCreate,
        videoSupportsAnimate: videoSupportsAnimate
    };
})(window);
