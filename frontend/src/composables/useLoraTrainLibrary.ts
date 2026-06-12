import { computed, ref } from 'vue';
import { api } from '@/utils/api';

export type LoraTrainRun = Record<string, unknown>;
export type UserLoraRow = Record<string, unknown>;

function runParams(run: LoraTrainRun): Record<string, unknown> {
  const p = run.params;
  return p && typeof p === 'object' ? (p as Record<string, unknown>) : {};
}

function runResultMeta(run: LoraTrainRun): Record<string, unknown> {
  const result = run.result;
  if (!result || typeof result !== 'object') return {};
  const meta = (result as Record<string, unknown>).metadata;
  return meta && typeof meta === 'object' ? (meta as Record<string, unknown>) : {};
}

export function runTaskId(run: LoraTrainRun): string {
  return String(run.id || '');
}

export function runBaseModel(run: LoraTrainRun): string {
  const params = runParams(run);
  return String(params.base_model || run.model || '—').split(':', 1)[0];
}

export function runOutputName(run: LoraTrainRun): string {
  const params = runParams(run);
  const meta = runResultMeta(run);
  const training =
    meta.training && typeof meta.training === 'object'
      ? (meta.training as Record<string, unknown>)
      : {};
  const fromParams = String(params.output_name || '').trim();
  if (fromParams) return fromParams;
  const fromTraining = String(training.output_name || '').trim();
  if (fromTraining) return fromTraining;
  return runTaskId(run).slice(0, 12) || '—';
}

export function runUserLoraId(run: LoraTrainRun): string {
  const meta = runResultMeta(run);
  const training =
    meta.training && typeof meta.training === 'object'
      ? (meta.training as Record<string, unknown>)
      : {};
  return String(meta.user_lora_id || training.user_lora_id || '').trim();
}

export function useLoraTrainLibrary() {
  const runs = ref<LoraTrainRun[]>([]);
  const userLoras = ref<UserLoraRow[]>([]);
  const loading = ref(false);

  const loraById = computed(() => {
    const map = new Map<string, UserLoraRow>();
    for (const row of userLoras.value) {
      const id = String(row.id || '').trim();
      if (id) map.set(id, row);
    }
    return map;
  });

  const loraByTaskId = computed(() => {
    const map = new Map<string, UserLoraRow>();
    for (const row of userLoras.value) {
      const tid = String(row.task_id || '').trim();
      if (tid) map.set(tid, row);
    }
    return map;
  });

  function userLoraForRun(run: LoraTrainRun): UserLoraRow | undefined {
    const tid = runTaskId(run);
    if (tid && loraByTaskId.value.has(tid)) {
      return loraByTaskId.value.get(tid);
    }
    const lid = runUserLoraId(run);
    if (lid && loraById.value.has(lid)) {
      return loraById.value.get(lid);
    }
    return undefined;
  }

  async function refresh({ limit = 12 }: { limit?: number } = {}) {
    loading.value = true;
    try {
      const [tasksRes, adaptersRes] = await Promise.all([
        api.gen.listMediaTasks({ kind: 'lora.training', limit }),
        api.loras.listUserAdapters(),
      ]);
      const taskPayload = tasksRes as { tasks?: LoraTrainRun[] };
      const adapterPayload = adaptersRes as { items?: UserLoraRow[] };
      runs.value = taskPayload.tasks || [];
      userLoras.value = adapterPayload.items || [];
    } catch {
      runs.value = [];
      userLoras.value = [];
    } finally {
      loading.value = false;
    }
  }

  return {
    runs,
    userLoras,
    loading,
    refresh,
    userLoraForRun,
    runOutputName,
    runBaseModel,
    runTaskId,
    runUserLoraId,
  };
}
