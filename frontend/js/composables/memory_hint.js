/**
 * Plan E4: Pre-submit memory risk soft hint (non-blocking). Heuristic comparison based on registry version `size` string and system memory / MLX limit.
 * Mounted as `window.DQMemoryHint`, called by image/video creation pages before actually submitting to API.
 */
(function (w) {
    function parseHumanSizeToGb(s) {
        if (s == null || s === '') return null;
        const str = String(s)
            .trim()
            .toLowerCase()
            .replace(/[,~≈]/g, '')
            .replace(/\s+/g, '');
        const m = str.match(/([\d.]+)\s*(tb|t|gb|g|mb|m)?/);
        if (!m) return null;
        let n = parseFloat(m[1]);
        if (!Number.isFinite(n) || n <= 0) return null;
        const u = m[2] || 'gb';
        if (u === 'tb' || u === 't') n *= 1024;
        else if (u === 'mb' || u === 'm') n /= 1024;
        return n;
    }

    /**
     * @param {object} opts
     * @param {object} opts.systemInfo injected reactive (must include memory_gb, mlx_memory_limit)
     * @param {string} opts.versionSizeHuman current version registry `size` field (e.g. "19GB")
     * @param {function} opts.$tt i18n function
     */
    function warnIfRisky(opts) {
        if (typeof ElementPlus === 'undefined' || !ElementPlus.ElMessage) return;
        const $tt = opts && typeof opts.$tt === 'function' ? opts.$tt : null;
        if (!$tt) return;

        const si = opts && opts.systemInfo;
        const mem = Number(si && si.memory_gb) || 0;
        const mlxRaw = Number(si && si.mlx_memory_limit);
        const capFromMlx = Number.isFinite(mlxRaw) && mlxRaw > 0 ? mlxRaw : null;

        let refGb = 0;
        if (mem > 0 && capFromMlx != null) {
            refGb = Math.min(mem, capFromMlx);
        } else if (mem > 0) {
            refGb = mem;
        } else if (capFromMlx != null) {
            refGb = capFromMlx;
        }
        if (!(refGb > 0)) return;

        const modelGb = parseHumanSizeToGb(opts.versionSizeHuman);
        if (modelGb == null || modelGb <= 0) return;

        if (modelGb > refGb * 0.88) {
            ElementPlus.ElMessage.warning(
                $tt('studio.submitOomHint', {
                    modelGb: modelGb.toFixed(1),
                    refGb: refGb.toFixed(1),
                }),
            );
        }
    }

    w.DQMemoryHint = { parseHumanSizeToGb, warnIfRisky };
})(window);
