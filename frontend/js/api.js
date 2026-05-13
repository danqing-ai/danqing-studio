/**
 * API Client — v3 media endpoints only + gallery / settings / download
 */

const API_BASE = '';

/** Convert `GET /api/assets` entries to gallery card rows (aligned with `GET /api/gallery/images`, Plan C5) */
function assetRowToGalleryItem(a) {
    const aid = a.id;
    const meta = { ...(a.metadata || {}) };
    const rawPath = String(a.path || '');
    const base = rawPath.split(/[/\\\\]/).filter(Boolean).pop() || aid;
    const thumb = a.thumbnail_url || `${API_BASE}/api/assets/${aid}/thumbnail`;
    const durRaw = a.duration_seconds != null ? a.duration_seconds : meta.duration_seconds;
    const duration_seconds =
        durRaw != null && Number.isFinite(Number(durRaw)) ? Number(durRaw) : null;
    return {
        path: `asset:${aid}`,
        name: base,
        width: Number(a.width || meta.width || 0),
        height: Number(a.height || meta.height || 0),
        duration_seconds,
        created_at: String(a.created_at || ''),
        prompt: String(meta.prompt || ''),
        model: String(meta.model || ''),
        thumbnail: thumb,
        metadata: {
            ...meta,
            ...(duration_seconds != null ? { duration_seconds } : {}),
            asset_kind:
                a.kind != null && String(a.kind).trim() !== '' ? String(a.kind) : 'image',
        },
    };
}

const api = {
    gallery: {
        /**
         * Gallery list: same strategy as backend `list_images` (cap → map → sort by created_at desc → slice).
         * Data from `api.gen.listAssets`, avoids duplicate requests to `/api/gallery/images`.
         */
        async listImages(limit = 40, offset = 0, options = {}) {
            const lim = Number(limit) || 40;
            const off = Number(offset) || 0;
            const params = { limit: lim, offset: off, exclude_upload_refs: true, ...options };
            const data = await api.gen.listAssets(null, lim, off, params);
            const rows = (data.items || []).map(assetRowToGalleryItem);
            return rows;
        },

        /** Gallery item path is always `asset:{id}`; media bytes via `/api/assets`. */
        getImageUrl(path) {
            if (typeof path !== 'string' || !path.startsWith('asset:')) {
                throw new Error('expected asset:id');
            }
            const id = path.slice('asset:'.length);
            return `${API_BASE}/api/assets/${id}/file`;
        },

        async deleteImage(path) {
            if (typeof path !== 'string' || !path.startsWith('asset:')) {
                throw new Error('expected asset:id');
            }
            const id = path.slice('asset:'.length);
            return api.gen.deleteAsset(id);
        },
        async batchDeleteImages(paths) {
            const ids = paths
                .filter((p) => typeof p === 'string' && p.startsWith('asset:'))
                .map((p) => p.slice('asset:'.length));
            return api.gen.batchDeleteAssets(ids);
        },

        async uploadImage(file) {
            const formData = new FormData();
            formData.append('file', file);
            const response = await axios.post(`${API_BASE}/api/gallery/upload`, formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            return response.data;
        }
    },

    adapters: {
        async list(forModel) {
            const params = forModel ? { for_model: forModel } : {};
            const response = await axios.get(`${API_BASE}/api/adapters`, { params });
            return response.data;
        },
    },

    models: {
        /** GET /api/models — optional ``media`` / ``action`` / ``installed`` (plan §6.2) */
        async list(params = {}) {
            const response = await axios.get(`${API_BASE}/api/models`, { params });
            return response.data;
        },
        /** Registry model install; returns { task_id, ... }; progress SSE see `api.download.installProgressStreamUrl` */
        async install(modelId, body = {}) {
            const response = await axios.post(
                `${API_BASE}/api/models/${encodeURIComponent(modelId)}/install`,
                body
            );
            return response.data;
        },
        /** Delete a registry version's local weights directory */
        async deleteVersion(modelId, versionKey) {
            const response = await axios.delete(
                `${API_BASE}/api/models/${encodeURIComponent(modelId)}/versions/${encodeURIComponent(versionKey)}`
            );
            return response.data;
        },
        /** Batch start registry model downloads; body: { model_ids: string[] }, returns { results } */
        async installBatch(modelIds) {
            const response = await axios.post(`${API_BASE}/api/models/install-batch`, {
                model_ids: modelIds,
            });
            return response.data;
        },
    },

    download: {
        async startDownload(url, targetName, type = 'model') {
            const response = await axios.post(`${API_BASE}/api/download/start`, {
                url, target_name: targetName, type
            });
            return response.data;
        },

        async listDownloads() {
            const response = await axios.get(`${API_BASE}/api/download/tasks`);
            return response.data;
        },

        async cancel(taskId) {
            await axios.post(`${API_BASE}/api/download/cancel/${encodeURIComponent(taskId)}`);
        },

        async delete(taskId) {
            await axios.delete(`${API_BASE}/api/download/tasks/${encodeURIComponent(taskId)}`);
        },

        async resume(taskId) {
            const response = await axios.post(
                `${API_BASE}/api/download/resume/${encodeURIComponent(taskId)}`
            );
            return response.data;
        },

        async civitaiSearch(params) {
            const response = await axios.get(`${API_BASE}/api/download/civitai/search`, { params });
            return response.data;
        },

        async startLoraDownload(url, filename) {
            const response = await axios.post(`${API_BASE}/api/download/lora`, { url, filename });
            return response.data;
        },

        /** Install / LoRA etc download progress SSE (EventSource) */
        installProgressStreamUrl(taskId) {
            return `${API_BASE}/api/download/progress/${encodeURIComponent(taskId)}/stream`;
        },

        /** MLX int4/int8 conversion from an installed parent version; returns { task_id, ... } */
        async startConvert(body) {
            const response = await axios.post(`${API_BASE}/api/download/convert`, body);
            return response.data;
        },

        convertProgressStreamUrl(taskId) {
            return `${API_BASE}/api/download/convert/${encodeURIComponent(taskId)}/stream`;
        },

        async cancelConversion(taskId) {
            await axios.post(`${API_BASE}/api/download/convert/${encodeURIComponent(taskId)}/cancel`);
        },
    },

    tasks: {
        /** Media task log SSE (consistent with TasksStore.openTaskLogStream) */
        logStreamUrl(taskId) {
            return `${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/stream`;
        },
    },

    settings: {
        async getSettings() {
            const response = await axios.get(`${API_BASE}/api/settings`);
            return response.data;
        },

        async updateSettings(settings) {
            const response = await axios.put(`${API_BASE}/api/settings`, settings);
            return response.data;
        },

        async listModels() {
            const response = await axios.get(`${API_BASE}/api/settings/models`);
            return response.data;
        },

        async listLoras() {
            const response = await axios.get(`${API_BASE}/api/settings/loras`);
            return response.data;
        },

        async refreshModels() {
            const response = await axios.post(`${API_BASE}/api/settings/refresh`);
            return response.data;
        },

        async getSystemInfo() {
            const response = await axios.get(`${API_BASE}/api/settings/system`);
            return response.data;
        },

        async installEnvironment() {
            const response = await axios.post(`${API_BASE}/api/settings/install`);
            return response.data;
        },

        async getModelRegistry() {
            const response = await axios.get(`${API_BASE}/api/settings/registry`);
            return response.data;
        },

        async getModelsStatus() {
            const response = await axios.get(`${API_BASE}/api/settings/models/status`);
            return response.data;
        },

        async getModelsDetailedStatus() {
            const response = await axios.get(`${API_BASE}/api/settings/models/status/detailed`);
            return response.data;
        },

        async getDiskSpace() {
            const response = await axios.get(`${API_BASE}/api/settings/disk-space`);
            return response.data;
        },

        async getCompatibleLoras(modelName) {
            const response = await axios.get(`${API_BASE}/api/settings/loras/compatible/${modelName}`);
            return response.data;
        },

        async getCompatibleControlNets(modelName) {
            const response = await axios.get(`${API_BASE}/api/settings/controlnets/compatible/${modelName}`);
            return response.data;
        },

        async updateModelParameters(modelName, parameters) {
            const response = await axios.post(`${API_BASE}/api/settings/models/${modelName}/parameters`, parameters);
            return response.data;
        },

        async getSystemMonitor() {
            const response = await axios.get(`${API_BASE}/api/settings/system/monitor`);
            return response.data;
        },

        async getPresets() {
            const response = await axios.get(`${API_BASE}/api/settings/presets`);
            return response.data;
        },

        async savePreset(name, preset) {
            const response = await axios.post(`${API_BASE}/api/settings/presets`, { name, preset });
            return response.data;
        },

        async deletePreset(name) {
            const response = await axios.delete(`${API_BASE}/api/settings/presets/${encodeURIComponent(name)}`);
            return response.data;
        }
    },

    /** Generation tasks + asset upload (formerly api.media, renamed) */
    gen: {
        async uploadAsset(file) {
            const fd = new FormData();
            fd.append('file', file);
            const response = await axios.post(`${API_BASE}/api/assets`, fd, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            return response.data;
        },
        async listAssets(kind, limit = 100, offset = 0, options = {}) {
            const params = { limit, offset, ...options };
            if (kind != null && kind !== '') params.kind = kind;
            const response = await axios.get(`${API_BASE}/api/assets`, { params });
            return response.data;
        },
        async deleteAsset(assetId) {
            const response = await axios.delete(`${API_BASE}/api/assets/${encodeURIComponent(assetId)}`);
            return response.data;
        },
        async batchDeleteAssets(assetIds) {
            const response = await axios.post(`${API_BASE}/api/assets/batch-delete`, {
                asset_ids: assetIds,
            });
            return response.data;
        },
        /** Disk reconciliation: default dry_run; when dry_run=false, remove asset rows from DB whose main files are missing */
        async reconcileAssets(dryRun = true) {
            const response = await axios.post(`${API_BASE}/api/assets/reconcile`, {
                dry_run: dryRun,
            });
            return response.data;
        },
        /** Same-origin URL, blob:, data: → Blob (creation page edit image / reference image / video first frame) */
        async urlToBlob(url) {
            if (typeof url !== 'string' || !url.trim()) {
                throw new Error('urlToBlob: expected non-empty string');
            }
            const u = url.trim();
            if (u.startsWith('blob:') || u.startsWith('data:')) {
                const r = await fetch(u);
                return r.blob();
            }
            const response = await axios.get(u, { responseType: 'blob' });
            return response.data;
        },
        async createImageGeneration(body) {
            const response = await axios.post(`${API_BASE}/api/images/generations`, body);
            return response.data;
        },
        async createImageEdit(body) {
            const response = await axios.post(`${API_BASE}/api/images/edits`, body);
            return response.data;
        },
        async createImageUpscale(body) {
            const response = await axios.post(`${API_BASE}/api/images/upscales`, body);
            return response.data;
        },
        async createVideoGeneration(body) {
            const response = await axios.post(`${API_BASE}/api/videos/generations`, body);
            return response.data;
        },
        async createVideoEdit(body) {
            const response = await axios.post(`${API_BASE}/api/videos/edits`, body);
            return response.data;
        },
        async getMediaTask(taskId) {
            const response = await axios.get(`${API_BASE}/api/tasks/${taskId}`);
            return response.data;
        },
        /** GET /api/tasks — plan §6.2 (equivalent to old ``/list``); kind / status / since filtering */
        async listMediaTasks({ limit = 100, offset = 0, kind, status, since } = {}) {
            const params = new URLSearchParams();
            params.set('limit', String(limit));
            params.set('offset', String(offset));
            if (kind != null && kind !== '') params.set('kind', kind);
            if (status != null && status !== '') params.set('status', status);
            if (since != null && since !== '') params.set('since', since);
            const response = await axios.get(`${API_BASE}/api/tasks?${params.toString()}`);
            return response.data;
        },
        /** GET /api/tasks/{id}/logs — paginated task_logs rows */
        async getMediaTaskLogs(taskId, { offset = 0, limit = 500 } = {}) {
            const params = new URLSearchParams();
            params.set('offset', String(offset));
            params.set('limit', String(limit));
            const response = await axios.get(
                `${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/logs?${params.toString()}`
            );
            return response.data;
        },
        /** PATCH /api/tasks/{id} — body `{ priority: 'normal' | 'high' }` (queued only) */
        async patchMediaTaskPriority(taskId, body) {
            const response = await axios.patch(
                `${API_BASE}/api/tasks/${encodeURIComponent(taskId)}`,
                body
            );
            return response.data;
        },
        async cancelMediaTask(taskId) {
            const response = await axios.delete(`${API_BASE}/api/tasks/${taskId}`);
            return response.data;
        },
        async getQueue() {
            const response = await axios.get(`${API_BASE}/api/queue`);
            return response.data;
        },
        streamMediaTask(taskId, onLog, onStatus, onDone, onError, onProgress, onResult) {
            const TS = typeof window !== 'undefined' && window.TasksStore;
            if (TS && typeof TS.openTaskLogStream === 'function') {
                return TS.openTaskLogStream(taskId, {
                    onLog,
                    onStatus,
                    onDone,
                    onError,
                    onProgress,
                    onResult,
                });
            }
            throw new Error('TasksStore missing: load js/stores/tasks_store.js before components');
        }
    },

    registry: {
        async getFull() {
            const response = await axios.get(`${API_BASE}/api/registry`);
            return response.data;
        }
    },

    system: {
        async health() {
            const response = await axios.get(`${API_BASE}/api/system/health`);
            return response.data;
        },
        async metrics() {
            const response = await axios.get(`${API_BASE}/api/system/metrics`);
            return response.data;
        },
        async getCacheStatus() {
            const response = await axios.get(`${API_BASE}/api/system/cache`);
            return response.data;
        },
    },

    /** Audio API — aligned with `backend/api/routes/audios.py`; generation is not implemented (tasks fail loud). */
    audios: {
        async listGenerations() {
            const response = await axios.get(`${API_BASE}/api/audios/generations`);
            return response.data;
        },
        async createGeneration(body = {}) {
            const response = await axios.post(`${API_BASE}/api/audios/generations`, body);
            return response.data;
        },
        async createEdit(body = {}) {
            const response = await axios.post(`${API_BASE}/api/audios/edits`, body);
            return response.data;
        },
        async createDub(body = {}) {
            const response = await axios.post(`${API_BASE}/api/audios/dubs`, body);
            return response.data;
        },
    },
};

if (typeof window !== 'undefined') {
    window.api = api;
}
