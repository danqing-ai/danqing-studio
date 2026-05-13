/**
 * Plan C9: Filter global queue snapshot by task kind prefix (aligned with TasksStore.queueState).
 * CDN no-build: mounted as `window.DQMediaQueue` for reuse in creation page computed.
 */
(function (w) {
    function normalizeTaskRow(t, liveById) {
        const pr = t.priority;
        let progressMessage = t.progressMessage;
        const live = liveById && t && t.id != null ? liveById[t.id] : null;
        let progress = typeof t.progress === 'number' ? t.progress : 0;
        let step = t.step;
        let total = t.total;
        if (live) {
            if (typeof live.progress === 'number') progress = live.progress;
            if (live.step != null) step = live.step;
            if (live.total != null) total = live.total;
            if (Object.prototype.hasOwnProperty.call(live, 'progressMessage')) {
                progressMessage = live.progressMessage;
            }
        }
        return {
            id: t.id,
            kind: t.kind,
            progress,
            step,
            total,
            progressMessage,
            priority: typeof pr === 'number' ? pr : 100,
            estimated_wait_seconds: t.estimated_wait_seconds,
            params: {
                model: t.model_id || (t.params && t.params.model) || '',
                prompt: (t.params && t.params.prompt) || '',
            },
        };
    }

    function filterByKindPrefix(arr, prefix, liveById) {
        const p = String(prefix || '');
        return (arr || [])
            .filter((t) => String(t.kind || '').startsWith(p))
            .map((t) => normalizeTaskRow(t, liveById));
    }

    /**
     * Top bar drawer: full queue (no kind filter), row shape consistent with creation page badge.
     * @param {object|null|undefined} TS window.TasksStore
     */
    function snapshotFullQueue(TS) {
        if (!TS || !TS.queueState) {
            return { running: [], queued: [] };
        }
        const live = TS.liveTaskProgress || {};
        return {
            running: (TS.queueState.running || []).map((t) => normalizeTaskRow(t, live)),
            queued: (TS.queueState.queued || []).map((t) => normalizeTaskRow(t, live)),
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
            const live = TS.liveTaskProgress || {};
            return {
                running: filterByKindPrefix(TS.queueState.running, prefix, live),
                queued: filterByKindPrefix(TS.queueState.queued, prefix, live),
            };
        },
    };
})(window);
