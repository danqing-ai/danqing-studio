/**
 * Model dropdown `value`: `modelKey|versionKey` ↔ split.
 */
(function (w) {
    /**
     * @param {string} value el-select v-model
     * @returns {{ modelKey: string, versionKey: string } | null}
     */
    function parse(value) {
        if (!value || typeof value !== 'string') return null;
        const parts = value.split('|');
        if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
        return { modelKey: parts[0], versionKey: parts[1] };
    }

    function format(modelKey, versionKey) {
        return `${modelKey}|${versionKey}`;
    }

    w.DQModelVersionValue = { parse, format };
})(window);
