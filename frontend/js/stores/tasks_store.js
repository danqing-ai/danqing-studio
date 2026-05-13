/**
 * Global task queue polling + media task SSE singleton — Plan C2
 * CDN no-build: mounted as window.TasksStore
 */
(function (w) {
    const API_BASE = '';
    const { reactive } = Vue;

    const queueState = reactive({
        running: [],
        queued: [],
    });

    /** Latest SSE progress per task — merged into queue snapshots so UI is not stuck on 2s poll only */
    const liveTaskProgress = reactive({});

    function patchLiveTaskProgress(taskId, patch) {
        if (!taskId || !patch || Object.keys(patch).length === 0) return;
        const prev = liveTaskProgress[taskId] || {};
        const next = Object.assign({}, prev);
        if (typeof patch.progress === 'number') next.progress = patch.progress;
        if (patch.step != null) next.step = patch.step;
        if (patch.total != null) next.total = patch.total;
        if (patch.eta_seconds != null) next.eta_seconds = patch.eta_seconds;
        if (Object.prototype.hasOwnProperty.call(patch, 'progressMessage')) {
            next.progressMessage = patch.progressMessage;
        }
        liveTaskProgress[taskId] = next;
    }

    function clearLiveTaskProgress(taskId) {
        if (taskId && liveTaskProgress[taskId] != null) {
            delete liveTaskProgress[taskId];
        }
    }

    let pollTimer = null;
    let pollRefCount = 0;

    async function pollQueueOnce() {
        try {
            const data =
                typeof w.api !== 'undefined' && w.api.gen && typeof w.api.gen.getQueue === 'function'
                    ? await w.api.gen.getQueue()
                    : (await axios.get(`${API_BASE}/api/queue`)).data;
            queueState.running = data.running || [];
            queueState.queued = data.queued || [];
            for (const t of queueState.running || []) {
                if (t && t.id && !logStreams.has(t.id)) {
                    openTaskLogStream(t.id, {});
                }
            }
        } catch (e) {
            console.error('TasksStore: queue poll failed', e);
        }
    }

    function ensureQueuePoller() {
        pollRefCount += 1;
        if (!pollTimer) {
            pollQueueOnce();
            pollTimer = setInterval(pollQueueOnce, 2000);
        }
    }

    function releaseQueuePoller() {
        pollRefCount = Math.max(0, pollRefCount - 1);
        if (pollRefCount === 0 && pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }

    const logStreams = new Map();

    function closeTaskLogStream(taskId) {
        const es = logStreams.get(taskId);
        if (es) {
            try {
                es.close();
            } catch (e) {
                /* ignore */
            }
            logStreams.delete(taskId);
        }
    }

    /**
     * One SSE per taskId; auto-close on done/error.
     */
    function openTaskLogStream(taskId, { onLog, onStatus, onProgress, onResult, onDone, onError }) {
        closeTaskLogStream(taskId);
        const url =
            typeof w.api !== 'undefined' && w.api.tasks && typeof w.api.tasks.logStreamUrl === 'function'
                ? w.api.tasks.logStreamUrl(taskId)
                : `${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/stream`;
        const eventSource = new EventSource(url);
        logStreams.set(taskId, eventSource);
        eventSource.addEventListener('log', (event) => {
            const data = JSON.parse(event.data);
            if (onLog) onLog(data);
        });
        eventSource.addEventListener('progress', (event) => {
            const data = JSON.parse(event.data);
            const patch = {};
            if (typeof data.progress === 'number') patch.progress = data.progress;
            if (data.step != null) patch.step = data.step;
            if (data.total != null) patch.total = data.total;
            if (data.eta_seconds != null) patch.eta_seconds = data.eta_seconds;
            if (Object.prototype.hasOwnProperty.call(data, 'message')) {
                patch.progressMessage = data.message;
            }
            patchLiveTaskProgress(taskId, patch);
            if (onProgress) onProgress(data);
            else if (onStatus) {
                onStatus({
                    status: 'running',
                    progress: data.progress,
                    step: data.step,
                    total: data.total,
                    eta_seconds: data.eta_seconds,
                });
            }
        });
        eventSource.addEventListener('status', (event) => {
            const data = JSON.parse(event.data);
            if (typeof data.progress === 'number') {
                patchLiveTaskProgress(taskId, { progress: data.progress });
            }
            if (onStatus) onStatus(data);
        });
        eventSource.addEventListener('result', (event) => {
            const data = JSON.parse(event.data);
            if (onResult) onResult(data);
        });
        eventSource.addEventListener('done', (event) => {
            const data = JSON.parse(event.data);
            if (onDone) onDone(data);
            clearLiveTaskProgress(taskId);
            closeTaskLogStream(taskId);
        });
        eventSource.addEventListener('error', (event) => {
            if (onError) onError(event);
            clearLiveTaskProgress(taskId);
            closeTaskLogStream(taskId);
        });
        return eventSource;
    }

    w.TasksStore = {
        queueState,
        liveTaskProgress,
        ensureQueuePoller,
        releaseQueuePoller,
        pollQueueOnce,
        openTaskLogStream,
        closeTaskLogStream,
    };
})(window);
