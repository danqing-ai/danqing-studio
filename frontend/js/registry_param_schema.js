/**
 * models_registry.json parameters — 与 backend/core/registry_format.typed_parameters 对齐的推断与工具。
 */
(function (w) {
    function inferType(spec, key) {
        if (!spec || typeof spec !== 'object') return 'skip';
        if (spec.type) return spec.type;
        if (typeof spec.default === 'boolean' || (key && String(key).endsWith('_support'))) {
            return 'bool';
        }
        if (Array.isArray(spec.options)) return 'enum';
        if (typeof spec.min === 'number' && typeof spec.max === 'number') {
            const d = spec.default;
            return Number.isInteger(d) && typeof d !== 'boolean' ? 'int' : 'float';
        }
        return 'object';
    }

    function normalizeParamsDef(parameters) {
        const out = {};
        for (const [key, spec] of Object.entries(parameters || {})) {
            const t = inferType(spec, key);
            if (t === 'skip') continue;
            out[key] = { ...spec, type: t };
        }
        return out;
    }

    function isCapabilityOnly(key, spec) {
        return spec.type === 'bool' && String(key).endsWith('_support');
    }

    function isRenderableScalar(key, spec) {
        if (isCapabilityOnly(key, spec)) return false;
        if (spec.type === 'object') return false;
        return spec.type === 'int' || spec.type === 'float' || spec.type === 'enum';
    }

    const PREFERRED_ORDER = [
        'steps',
        'guidance',
        'scheduler',
        'strength',
        'controlnet_strength',
        'redux_strength',
    ];

    function sortParamKeys(keys) {
        return [...keys].sort((a, b) => {
            const ia = PREFERRED_ORDER.indexOf(a);
            const ib = PREFERRED_ORDER.indexOf(b);
            if (ia === -1 && ib === -1) return a.localeCompare(b);
            if (ia === -1) return 1;
            if (ib === -1) return -1;
            return ia - ib;
        });
    }

    function resolutionPair(normalized) {
        const w = normalized.width;
        const h = normalized.height;
        if (w && h && w.type === 'enum' && h.type === 'enum') {
            return { width: w, height: h };
        }
        return null;
    }

    function scalarKeysForForm(normalized) {
        const pair = resolutionPair(normalized);
        const skip = new Set();
        if (pair) {
            skip.add('width');
            skip.add('height');
        }
        const keys = Object.keys(normalized).filter((k) => {
            if (skip.has(k)) return false;
            return isRenderableScalar(k, normalized[k]);
        });
        return sortParamKeys(keys);
    }

    function applyDefaults(parameters, target) {
        const n = normalizeParamsDef(parameters || {});
        for (const [key, spec] of Object.entries(n)) {
            if (isCapabilityOnly(key, spec) || spec.type === 'object') continue;
            if ('default' in spec) {
                target[key] = spec.default;
            }
        }
        if (typeof target.seed !== 'undefined') target.seed = '';
        if (typeof target.lora !== 'undefined') target.lora = '';
        if (typeof target.controlnet !== 'undefined') {
            target.controlnet = '';
            target.controlnet_strength = n.controlnet_strength && 'default' in n.controlnet_strength
                ? n.controlnet_strength.default
                : 0.8;
        }
        const ls = n.lora_scale;
        if (typeof target.lora_scale !== 'undefined') {
            target.lora_scale = ls && 'default' in ls ? ls.default : 0.8;
        }
    }

    function hasDeviation(parameters, target, opts) {
        const ignore = new Set(
            (opts && opts.ignoreKeys) || ['strength', 'controlnet_strength']
        );
        const n = normalizeParamsDef(parameters || {});
        for (const [key, spec] of Object.entries(n)) {
            if (ignore.has(key)) continue;
            if (!isRenderableScalar(key, spec)) continue;
            const def = spec.default;
            if (def !== undefined && target[key] !== def) return true;
        }
        const pair = resolutionPair(n);
        if (pair) {
            if (target.width !== pair.width.default || target.height !== pair.height.default) {
                return true;
            }
        }
        if (target.lora) return true;
        if (target.seed != null && String(target.seed).trim() !== '') return true;
        return false;
    }

    w.RegistryParamSchema = {
        inferType,
        normalizeParamsDef,
        isCapabilityOnly,
        isRenderableScalar,
        sortParamKeys,
        resolutionPair,
        scalarKeysForForm,
        applyDefaults,
        hasDeviation,
    };
})(window);
