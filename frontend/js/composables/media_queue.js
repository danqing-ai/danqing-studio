/**
 * Plan C9：按任务 kind 前缀过滤全局队列快照（与 TasksStore.queueState 对齐）。
 * CDN 无打包：挂载 `window.DQMediaQueue`，供创作页 computed 复用。
 */
(function (w) {
    function normalizeTaskRow(t) {
        const pr = t.priority;
        return {
            id: t.id,
            kind: t.kind,
            progress: typeof t.progress === 'number' ? t.progress : 0,
            priority: typeof pr === 'number' ? pr : 100,
            estimated_wait_seconds: t.estimated_wait_seconds,
            params: {
                model: t.model_id || (t.params && t.params.model) || '',
                prompt: (t.params && t.params.prompt) || '',
            },
        };
    }

    function filterByKindPrefix(arr, prefix) {
        const p = String(prefix || '');
        return (arr || [])
            .filter((t) => String(t.kind || '').startsWith(p))
            .map(normalizeTaskRow);
    }

    /**
     * 顶栏抽屉：全队列（不过滤 kind），行形状与创作页角标一致。
     * @param {object|null|undefined} TS window.TasksStore
     */
    function snapshotFullQueue(TS) {
        if (!TS || !TS.queueState) {
            return { running: [], queued: [] };
        }
        return {
            running: (TS.queueState.running || []).map(normalizeTaskRow),
            queued: (TS.queueState.queued || []).map(normalizeTaskRow),
        };
    }

    w.DQMediaQueue = {
        normalizeTaskRow,
        snapshotFullQueue,
        filterByKindPrefix,
        /**
         * @param {object} TS window.TasksStore
         * @param {'image'|'video'} media
         */
        tasksForMedia(TS, media) {
            if (!TS || !TS.queueState) {
                return { running: [], queued: [] };
            }
            const prefix = media === 'video' ? 'video' : 'image';
            return {
                running: filterByKindPrefix(TS.queueState.running, prefix),
                queued: filterByKindPrefix(TS.queueState.queued, prefix),
            };
        },
    };
})(window);
