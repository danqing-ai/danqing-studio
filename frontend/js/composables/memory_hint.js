/**
 * Plan E4：提交前内存风险软提示（不拦截）。基于注册表版本 `size` 文案与系统内存 / MLX 上限启发式比较。
 * 挂载 `window.DQMemoryHint`，由图文创作页在真正提交 API 前调用。
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
     * @param {object} opts.systemInfo inject 的 reactive（需含 memory_gb、mlx_memory_limit）
     * @param {string} opts.versionSizeHuman 当前版本注册表 `size` 字段（如 "19GB"）
     * @param {function} opts.$tt i18n 函数
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
