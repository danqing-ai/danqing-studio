/**
 * 全局任务队列轮询 + 媒体任务 SSE 单例 — Plan C2
 * CDN 无打包：挂载 window.TasksStore
 */
(function (w) {
    const API_BASE = '';
    const { reactive } = Vue;

    const queueState = reactive({
        running: [],
        queued: [],
    });

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
     * 同一 taskId 仅保留一条 SSE；done/error 时自动关闭。
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
            if (onStatus) onStatus(data);
        });
        eventSource.addEventListener('result', (event) => {
            const data = JSON.parse(event.data);
            if (onResult) onResult(data);
        });
        eventSource.addEventListener('done', (event) => {
            const data = JSON.parse(event.data);
            if (onDone) onDone(data);
            closeTaskLogStream(taskId);
        });
        eventSource.addEventListener('error', (event) => {
            if (onError) onError(event);
            closeTaskLogStream(taskId);
        });
        return eventSource;
    }

    w.TasksStore = {
        queueState,
        ensureQueuePoller,
        releaseQueuePoller,
        pollQueueOnce,
        openTaskLogStream,
        closeTaskLogStream,
    };
})(window);
