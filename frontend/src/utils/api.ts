import axios from 'axios';
import type { AssetRow, GalleryItem, QueueState, RegistryData, SettingsData, SystemInfo } from '@/types';

const API_BASE = '';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

/** Task id from media submit 202 body `{ task: { id, ... } }` (legacy flat `{ id }` tolerated). */
export function taskIdFromSubmitResponse(res: unknown): string {
  if (res == null || typeof res !== 'object') return '';
  const top = res as Record<string, unknown>;
  const task = top.task;
  if (task != null && typeof task === 'object') {
    const nested = (task as Record<string, unknown>).id;
    if (nested != null && String(nested).trim() !== '') {
      return String(nested);
    }
  }
  const direct = top.id;
  if (direct != null && String(direct).trim() !== '') {
    return String(direct);
  }
  return '';
}

function assetRowToGalleryItem(a: AssetRow): GalleryItem {
  const aid = a.id;
  const meta = { ...(a.metadata || {}) };
  const rawPath = String(a.path || '');
  const base = rawPath.split(/[/\\]/).filter(Boolean).pop() || aid;
  const thumb = a.thumbnail_url || `${API_BASE}/api/assets/${aid}/thumbnail`;
  const durRaw = a.duration_seconds != null ? a.duration_seconds : (meta.duration_seconds as number | undefined);
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

export const api = {
  gallery: {
    async listImages(limit = 40, offset = 0, options: Record<string, unknown> = {}): Promise<GalleryItem[]> {
      const lim = Number(limit) || 40;
      const off = Number(offset) || 0;
      const params = { limit: lim, offset: off, exclude_upload_refs: true, ...options };
      const data = await api.gen.listAssets(null, lim, off, params);
      const rows = (data.items || []).map(assetRowToGalleryItem);
      return rows;
    },

    getImageUrl(path: string): string {
      if (!path.startsWith('asset:')) {
        throw new Error('expected asset:id');
      }
      const id = path.slice('asset:'.length);
      return `${API_BASE}/api/assets/${id}/file`;
    },

    async deleteImage(path: string): Promise<unknown> {
      if (!path.startsWith('asset:')) {
        throw new Error('expected asset:id');
      }
      const id = path.slice('asset:'.length);
      return api.gen.deleteAsset(id);
    },

    async batchDeleteImages(paths: string[]): Promise<unknown> {
      const ids = paths
        .filter((p) => p.startsWith('asset:'))
        .map((p) => p.slice('asset:'.length));
      return api.gen.batchDeleteAssets(ids);
    },

    async uploadImage(file: File): Promise<unknown> {
      const formData = new FormData();
      formData.append('file', file);
      const response = await client.post('/api/gallery/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },
  },

  adapters: {
    async list(forModel?: string): Promise<unknown> {
      const params = forModel ? { for_model: forModel } : {};
      const response = await client.get('/api/adapters', { params });
      return response.data;
    },
  },

  models: {
    async list(params: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.get('/api/models', { params });
      return response.data;
    },

    async install(modelId: string, body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post(
        `/api/models/${encodeURIComponent(modelId)}/install`,
        body
      );
      return response.data;
    },

    async deleteVersion(modelId: string, versionKey: string): Promise<unknown> {
      const response = await client.delete(
        `/api/models/${encodeURIComponent(modelId)}/versions/${encodeURIComponent(versionKey)}`
      );
      return response.data;
    },

    async installBatch(modelIds: string[]): Promise<unknown> {
      const response = await client.post('/api/models/install-batch', {
        model_ids: modelIds,
      });
      return response.data;
    },
  },

  download: {
    async startDownload(url: string, targetName: string, type = 'model'): Promise<unknown> {
      const response = await client.post('/api/download/start', {
        url,
        target_name: targetName,
        type,
      });
      return response.data;
    },

    async listDownloads(): Promise<unknown> {
      const response = await client.get('/api/download/tasks');
      return response.data;
    },

    async cancel(taskId: string): Promise<void> {
      await client.post(`/api/download/cancel/${encodeURIComponent(taskId)}`);
    },

    async delete(taskId: string): Promise<void> {
      await client.delete(`/api/download/tasks/${encodeURIComponent(taskId)}`);
    },

    async resume(taskId: string): Promise<unknown> {
      const response = await client.post(
        `/api/download/resume/${encodeURIComponent(taskId)}`
      );
      return response.data;
    },

    async civitaiSearch(params: Record<string, unknown>): Promise<unknown> {
      const response = await client.get('/api/download/civitai/search', { params });
      return response.data;
    },

    async startLoraDownload(url: string, filename: string): Promise<unknown> {
      const response = await client.post('/api/download/lora', { url, filename });
      return response.data;
    },

    installProgressStreamUrl(taskId: string): string {
      return `${API_BASE}/api/download/progress/${encodeURIComponent(taskId)}/stream`;
    },

    async startConvert(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/download/convert', body);
      return response.data;
    },

    convertProgressStreamUrl(taskId: string): string {
      return `${API_BASE}/api/download/convert/${encodeURIComponent(taskId)}/stream`;
    },

    async cancelConversion(taskId: string): Promise<void> {
      await client.post(`/api/download/convert/${encodeURIComponent(taskId)}/cancel`);
    },
  },

  tasks: {
    logStreamUrl(taskId: string): string {
      return `${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/stream`;
    },
  },

  settings: {
    async getSettings(): Promise<SettingsData> {
      const response = await client.get('/api/settings');
      return response.data;
    },

    async updateSettings(settings: SettingsData): Promise<{ success?: boolean; restart_required?: boolean }> {
      const response = await client.put('/api/settings', settings);
      return response.data;
    },

    async getWorkspacePaths(): Promise<Record<string, string>> {
      const response = await client.get('/api/settings/workspace-paths');
      return response.data;
    },

    async getWorkspaceStatus(): Promise<{
      configured: boolean;
      effective_root: string;
      bootstrap_root: string;
    }> {
      const response = await client.get('/api/settings/workspace-status');
      return response.data;
    },

    async applyWorkspace(path: string): Promise<{
      success: boolean;
      restart_required: boolean;
      workspace?: string;
    }> {
      const response = await client.post('/api/settings/apply-workspace', { path });
      return response.data;
    },

    async pickWorkspaceDirectory(): Promise<{ path: string }> {
      const response = await client.post('/api/settings/pick-workspace-directory');
      return response.data;
    },

    async restoreConfigDefaults(files?: string[]): Promise<{
      success: boolean;
      restored: string[];
      restart_required: boolean;
    }> {
      const body = files && files.length > 0 ? { files } : {};
      const response = await client.post('/api/settings/restore-config-defaults', body);
      return response.data;
    },

    async listModels(): Promise<unknown> {
      const response = await client.get('/api/settings/models');
      return response.data;
    },

    async listLoras(): Promise<unknown> {
      const response = await client.get('/api/settings/loras');
      return response.data;
    },

    async refreshModels(): Promise<unknown> {
      const response = await client.post('/api/settings/refresh');
      return response.data;
    },

    async getSystemInfo(): Promise<SystemInfo> {
      const response = await client.get('/api/settings/system');
      return response.data;
    },

    async installEnvironment(): Promise<unknown> {
      const response = await client.post('/api/settings/install');
      return response.data;
    },

    async getModelRegistry(): Promise<unknown> {
      const response = await client.get('/api/settings/registry');
      return response.data;
    },

    async getModelsStatus(): Promise<unknown> {
      const response = await client.get('/api/settings/models/status');
      return response.data;
    },

    async getModelsDetailedStatus(): Promise<unknown> {
      const response = await client.get('/api/settings/models/status/detailed');
      return response.data;
    },

    async getDiskSpace(): Promise<unknown> {
      const response = await client.get('/api/settings/disk-space');
      return response.data;
    },

    async getCompatibleLoras(modelName: string): Promise<unknown> {
      const response = await client.get(`/api/settings/loras/compatible/${modelName}`);
      return response.data;
    },

    async getCompatibleControlNets(modelName: string): Promise<unknown> {
      const response = await client.get(`/api/settings/controlnets/compatible/${modelName}`);
      return response.data;
    },

    async updateModelParameters(modelName: string, parameters: Record<string, unknown>): Promise<unknown> {
      const response = await client.post(`/api/settings/models/${modelName}/parameters`, parameters);
      return response.data;
    },

    async getSystemMonitor(): Promise<unknown> {
      const response = await client.get('/api/settings/system/monitor');
      return response.data;
    },

    async getPresets(): Promise<unknown> {
      const response = await client.get('/api/settings/presets');
      return response.data;
    },

    async savePreset(name: string, preset: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/settings/presets', { name, preset });
      return response.data;
    },

    async deletePreset(name: string): Promise<unknown> {
      const response = await client.delete(`/api/settings/presets/${encodeURIComponent(name)}`);
      return response.data;
    },
  },

  gen: {
    async uploadAsset(file: File): Promise<unknown> {
      const fd = new FormData();
      fd.append('file', file);
      const response = await client.post('/api/assets', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },

    async listAssets(
      kind: string | null,
      limit = 100,
      offset = 0,
      options: Record<string, unknown> = {}
    ): Promise<{ items: AssetRow[]; total?: number }> {
      const params: Record<string, unknown> = { limit, offset, ...options };
      if (kind != null && kind !== '') params.kind = kind;
      const response = await client.get('/api/assets', { params });
      return response.data;
    },

    async deleteAsset(assetId: string): Promise<unknown> {
      const response = await client.delete(`/api/assets/${encodeURIComponent(assetId)}`);
      return response.data;
    },

    async batchDeleteAssets(assetIds: string[]): Promise<unknown> {
      const response = await client.post('/api/assets/batch-delete', {
        asset_ids: assetIds,
      });
      return response.data;
    },

    async reconcileAssets(dryRun = true): Promise<unknown> {
      const response = await client.post('/api/assets/reconcile', {
        dry_run: dryRun,
      });
      return response.data;
    },

    async urlToBlob(url: string): Promise<Blob> {
      if (!url.trim()) {
        throw new Error('urlToBlob: expected non-empty string');
      }
      const u = url.trim();
      if (u.startsWith('blob:') || u.startsWith('data:')) {
        const r = await fetch(u);
        return r.blob();
      }
      const response = await client.get(u, { responseType: 'blob' });
      return response.data as Blob;
    },

    async createImageGeneration(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/images/generations', body);
      return response.data;
    },

    async createImageEdit(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/images/edits', body);
      return response.data;
    },

    async createImageUpscale(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/images/upscales', body);
      return response.data;
    },

    async createVideoGeneration(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/generations', body);
      return response.data;
    },

    async createVideoEdit(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/edits', body);
      return response.data;
    },

    async createVideoUpscale(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/upscales', body);
      return response.data;
    },

    async getMediaTask(taskId: string): Promise<unknown> {
      const response = await client.get(`/api/tasks/${taskId}`);
      return response.data;
    },

    async listMediaTasks({
      limit = 100,
      offset = 0,
      kind,
      status,
      since,
    }: {
      limit?: number;
      offset?: number;
      kind?: string;
      status?: string;
      since?: string;
    } = {}): Promise<unknown> {
      const params = new URLSearchParams();
      params.set('limit', String(limit));
      params.set('offset', String(offset));
      if (kind != null && kind !== '') params.set('kind', kind);
      if (status != null && status !== '') params.set('status', status);
      if (since != null && since !== '') params.set('since', since);
      const response = await client.get(`/api/tasks?${params.toString()}`);
      return response.data;
    },

    async getMediaTaskLogs(
      taskId: string,
      { offset = 0, limit = 500 }: { offset?: number; limit?: number } = {}
    ): Promise<unknown> {
      const params = new URLSearchParams();
      params.set('offset', String(offset));
      params.set('limit', String(limit));
      const response = await client.get(
        `/api/tasks/${encodeURIComponent(taskId)}/logs?${params.toString()}`
      );
      return response.data;
    },

    async patchMediaTaskPriority(taskId: string, body: { priority: string }): Promise<unknown> {
      const response = await client.patch(
        `/api/tasks/${encodeURIComponent(taskId)}`,
        body
      );
      return response.data;
    },

    async cancelMediaTask(taskId: string): Promise<{ ok?: boolean }> {
      const response = await client.delete(`/api/tasks/${encodeURIComponent(taskId)}`);
      return response.data;
    },

    async getQueue(): Promise<QueueState> {
      const response = await client.get('/api/queue');
      return response.data;
    },

    streamMediaTask(
      taskId: string,
      callbacks: {
        onLog?: (data: unknown) => void;
        onStatus?: (data: unknown) => void;
        onDone?: (data: unknown) => void;
        onError?: (event: Event) => void;
        onProgress?: (data: unknown) => void;
        onResult?: (data: unknown) => void;
      }
    ): EventSource {
      const url = api.tasks.logStreamUrl(taskId);
      const eventSource = new EventSource(url);

      eventSource.addEventListener('log', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onLog?.(data);
      });

      eventSource.addEventListener('progress', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onProgress?.(data);
        if (!callbacks.onProgress && callbacks.onStatus) {
          callbacks.onStatus?.({
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
        callbacks.onStatus?.(data);
      });

      eventSource.addEventListener('result', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onResult?.(data);
      });

      eventSource.addEventListener('done', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onDone?.(data);
        eventSource.close();
      });

      eventSource.addEventListener('error', (event) => {
        callbacks.onError?.(event);
        eventSource.close();
      });

      return eventSource;
    },
  },

  registry: {
    async getFull(): Promise<RegistryData> {
      const response = await client.get('/api/registry');
      return response.data;
    },
  },

  system: {
    async health(): Promise<unknown> {
      const response = await client.get('/api/system/health');
      return response.data;
    },

    async metrics(): Promise<unknown> {
      const response = await client.get('/api/system/metrics');
      return response.data;
    },

    async getCacheStatus(): Promise<unknown> {
      const response = await client.get('/api/system/cache');
      return response.data;
    },
  },

  audios: {
    async listGenerations(): Promise<unknown> {
      const response = await client.get('/api/audios/generations');
      return response.data;
    },

    async createGeneration(body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post('/api/audios/generations', body);
      return response.data;
    },

    async createEdit(body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post('/api/audios/edits', body);
      return response.data;
    },

    async createDub(body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post('/api/audios/dubs', body);
      return response.data;
    },
  },
};

export default api;