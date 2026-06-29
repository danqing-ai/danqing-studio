import { ref } from 'vue';
import {
  analyzeReferenceViaChat,
  enhancePromptViaChat,
  generateLyricsViaChat,
  imageToPromptViaChat,
} from '@/utils/llmMessages';

export type CreativeMedia = 'image' | 'video' | 'audio';

export type CreativeTaskKind =
  | 'enhance_image'
  | 'image_to_prompt'
  | 'analyze_reference'
  | 'enhance_video'
  | 'generate_lyrics'
  | 'enhance_music_brief';

export type CreativeTaskStatus = 'pending' | 'running' | 'done' | 'error';

export interface CreativeTask {
  id: string;
  media: CreativeMedia;
  kind: CreativeTaskKind;
  title: string;
  input: string;
  assetId?: string;
  assetPreview?: string;
  status: CreativeTaskStatus;
  output?: string;
  error?: string;
  visionUsed?: boolean;
}

let _taskSeq = 0;

function nextTaskId(): string {
  _taskSeq += 1;
  return `task_${Date.now()}_${_taskSeq}`;
}

export function parseAssetId(path: string): string | null {
  if (!path.startsWith('asset:')) return null;
  const id = path.slice('asset:'.length).trim();
  return id || null;
}

export async function runCreativeTask(task: CreativeTask): Promise<CreativeTask> {
  const running = { ...task, status: 'running' as const, error: undefined };
  try {
    switch (task.kind) {
      case 'enhance_image': {
        const output = await enhancePromptViaChat(task.input, { targetAction: 'image_create' });
        return { ...running, status: 'done', output };
      }
      case 'enhance_video': {
        const output = await enhancePromptViaChat(task.input, { targetAction: 'video_create' });
        return { ...running, status: 'done', output };
      }
      case 'enhance_music_brief': {
        const output = await enhancePromptViaChat(task.input, { targetAction: 'audio_create' });
        return { ...running, status: 'done', output };
      }
      case 'generate_lyrics': {
        const output = await generateLyricsViaChat(task.input);
        return { ...running, status: 'done', output };
      }
      case 'image_to_prompt': {
        if (!task.assetId) throw new Error('asset required');
        const output = await imageToPromptViaChat(task.assetId);
        return { ...running, status: 'done', output, visionUsed: true };
      }
      case 'analyze_reference': {
        if (!task.assetId) throw new Error('asset required');
        if (!task.input.trim()) throw new Error('question required');
        const output = await analyzeReferenceViaChat(task.assetId, task.input);
        return { ...running, status: 'done', output, visionUsed: true };
      }
      default:
        throw new Error(`unknown task kind: ${task.kind}`);
    }
  } catch (e: unknown) {
    const err = e as { response?: { data?: { detail?: string } }; message?: string };
    const msg = err?.response?.data?.detail || err.message || String(e);
    return { ...running, status: 'error', error: msg };
  }
}

export function useCreativeAssistant() {
  const tasks = ref<CreativeTask[]>([]);

  function addTask(task: Omit<CreativeTask, 'id' | 'status'>): CreativeTask {
    const row: CreativeTask = {
      ...task,
      id: nextTaskId(),
      status: 'pending',
    };
    tasks.value = [row, ...tasks.value];
    return row;
  }

  function removeTask(id: string) {
    tasks.value = tasks.value.filter((t) => t.id !== id);
  }

  function clearFinished() {
    tasks.value = tasks.value.filter((t) => t.status === 'pending' || t.status === 'running');
  }

  async function executeTask(id: string) {
    const idx = tasks.value.findIndex((t) => t.id === id);
    if (idx < 0) return;
    const current = tasks.value[idx];
    if (current.status === 'running') return;
    tasks.value[idx] = { ...current, status: 'running', error: undefined };
    const result = await runCreativeTask(tasks.value[idx]);
    tasks.value[idx] = result;
    return result;
  }

  async function executeAllPending() {
    const pending = tasks.value.filter((t) => t.status === 'pending');
    for (const task of pending) {
      await executeTask(task.id);
    }
  }

  const isBatchRunning = ref(false);

  async function runBatch() {
    if (isBatchRunning.value) return;
    isBatchRunning.value = true;
    try {
      await executeAllPending();
    } finally {
      isBatchRunning.value = false;
    }
  }

  return {
    tasks,
    addTask,
    removeTask,
    clearFinished,
    executeTask,
    runBatch,
    isBatchRunning,
    runCreativeTask,
  };
}
