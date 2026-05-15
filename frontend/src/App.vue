<template>
  <el-container class="main-container">
    <el-header class="app-header" height="60">
      <TopNav
        :active-page="activePage"
        :queue-count="globalQueueCount"
        @navigate="handleNavSelect"
        @open-queue="openTaskQueue"
      />
    </el-header>

    <el-main class="app-main">
      <router-view />
    </el-main>
  </el-container>

  <el-drawer
    v-model="showGlobalQueueDrawer"
    class="dq-task-queue-drawer"
    :title="$tt('studio.queueDialogTitle')"
    direction="rtl"
    size="420px"
  >
    <div
      v-if="globalTaskQueue.running.length === 0 && globalTaskQueue.queued.length === 0"
      style="text-align: center; padding: 40px; color: var(--text-muted);"
    >
      {{ $tt('studio.queueEmpty') }}
    </div>
    <div v-else>
      <div v-if="globalTaskQueue.running.length > 0" style="margin-bottom: 20px;">
        <div style="font-weight: 600; margin-bottom: 12px; color: var(--primary-color);">
          {{ $tt('studio.running') }}
        </div>
        <div
          v-for="task in globalTaskQueue.running"
          :key="task.id"
          class="queue-dialog-item running"
        >
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
            <div style="flex: 1; overflow: hidden; min-width: 0;">
              <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">
                {{ taskKindLabel(task.kind) }}
              </div>
              <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 4px;">
                {{ task.params?.model || $tt('queue.unspecifiedModel') }}
              </div>
              <div class="dq-queue-prompt-line">{{ queueTruncate(task.params?.prompt || '', 40) }}</div>
            </div>
            <el-button
              size="small"
              circle
              type="danger"
              @click="cancelGlobalTask(task.id)"
              style="margin-left: 8px; flex-shrink: 0;"
              :title="$tt('studio.cancelTask')"
            >
              <el-icon><delete /></el-icon>
            </el-button>
          </div>
          <el-progress
            class="dq-queue-progress"
            :percentage="Math.round((task.progress || 0) * 100)"
            :stroke-width="4"
          />
          <div
            v-if="(task.total || 0) > 0"
            style="font-size: 11px; color: var(--text-muted); margin-top: 4px; text-align: right;"
          >
            <template v-if="String(task.kind || '').startsWith('image.')">
              {{ $tt('studio.queueDenoiseProgress', { current: task.step != null ? task.step : 0, total: task.total || 0 }) }}
            </template>
            <template v-else>
              {{ $tt('studio.queueStepProgress', { current: task.step != null ? task.step : 0, total: task.total || 0 }) }}
            </template>
          </div>
          <div
            v-if="task.progressMessage === 'post' && typeof task.progress === 'number' && task.progress < 1"
            style="font-size: 11px; color: var(--text-muted); margin-top: 2px; text-align: right;"
          >
            {{ $tt('studio.queuePostProcessHint') }}
          </div>
        </div>
      </div>

      <div v-if="globalTaskQueue.queued.length > 0">
        <div style="font-weight: 600; margin-bottom: 12px; color: var(--text-muted);">
          {{ $tt('studio.queued') }} ({{ globalTaskQueue.queued.length }})
        </div>
        <div
          v-for="(task, index) in globalTaskQueue.queued"
          :key="task.id"
          class="queue-dialog-item queued"
        >
          <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
            <div style="flex: 1; display: flex; align-items: center; gap: 8px; overflow: hidden; min-width: 0;">
              <span style="font-size: 12px; color: var(--text-muted); min-width: 24px;">#{{ index + 1 }}</span>
              <div style="flex: 1; overflow: hidden;">
                <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 2px;">
                  <span>{{ taskKindLabel(task.kind) }}</span>
                  <el-tag
                    v-if="(task.priority ?? 100) <= 50"
                    size="small"
                    type="warning"
                    effect="plain"
                    style="margin-left: 6px;"
                  >
                    {{ $tt('studio.queuePriorityHigh') }}
                  </el-tag>
                </div>
                <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 2px;">
                  {{ task.params?.model || $tt('queue.unspecifiedModel') }}
                </div>
                <div class="dq-queue-prompt-line">{{ queueTruncate(task.params?.prompt || '', 40) }}</div>
                <div
                  v-if="task.estimated_wait_seconds != null"
                  style="font-size: 11px; color: var(--text-muted); margin-top: 2px;"
                >
                  {{ $tt('queue.estimatedWait', { s: task.estimated_wait_seconds }) }}
                </div>
              </div>
            </div>
            <div style="display: flex; flex-direction: column; align-items: stretch; gap: 6px; flex-shrink: 0;">
              <el-button
                size="small"
                @click="setQueuedPriority(task.id, 'high')"
                :disabled="(task.priority ?? 100) <= 50"
              >
                {{ $tt('studio.queueSetHigh') }}
              </el-button>
              <el-button
                size="small"
                @click="setQueuedPriority(task.id, 'normal')"
                :disabled="(task.priority ?? 100) > 50"
              >
                {{ $tt('studio.queueSetNormal') }}
              </el-button>
              <el-button
                size="small"
                circle
                type="danger"
                @click="cancelGlobalTask(task.id)"
                style="align-self: flex-end;"
                :title="$tt('studio.cancelTask')"
              >
                <el-icon><delete /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </el-drawer>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, provide } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { ElMessage } from 'element-plus';
import TopNav from '@/components/shell/TopNav.vue';
import { useTasksStore } from '@/stores/tasks';
import { api } from '@/utils/api';
import { $tt, applyTheme } from '@/utils/i18n';
import { getItem, DQ_STORAGE } from '@/utils/storage';
import type { PageKey, SystemInfo, Task } from '@/types';

const router = useRouter();
const route = useRoute();
const tasksStore = useTasksStore();

const activePage = ref<PageKey>('image_create');
const showGlobalQueueDrawer = ref(false);
const currentLang = ref('zh');

const systemInfo = ref<SystemInfo>({
  env_ready: false,
  platform: '',
  architecture: '',
  memory_gb: 0,
  mlx_memory_limit: 120,
});

const globalTaskQueue = computed(() => {
  const running = tasksStore.queueState.running.map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
  const queued = tasksStore.queueState.queued.map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
  return { running, queued };
});

const globalQueueCount = computed(
  () => globalTaskQueue.value.running.length + globalTaskQueue.value.queued.length
);

watch(
  () => route.name as PageKey,
  (newVal) => {
    if (newVal) {
      activePage.value = newVal;
    }
  },
  { immediate: true }
);

function handleNavSelect(index: string) {
  router.push({ name: index });
}

function openTaskQueue() {
  showGlobalQueueDrawer.value = true;
}

async function cancelGlobalTask(taskId: string) {
  try {
    await api.gen.cancelMediaTask(taskId);
    await tasksStore.pollQueueOnce();
    ElMessage.success($tt('studio.cancelled'));
  } catch (e: unknown) {
    console.error('cancelGlobalTask', e);
    const msg = e instanceof Error ? e.message : String(e);
    ElMessage.error($tt('studio.error', { msg }));
  }
}

async function setQueuedPriority(taskId: string, priority: string) {
  try {
    await api.gen.patchMediaTaskPriority(taskId, { priority });
    await tasksStore.pollQueueOnce();
    ElMessage.success($tt('studio.priorityUpdated'));
  } catch (e: unknown) {
    console.error('setQueuedPriority', e);
    let msg = '';
    if (typeof e === 'object' && e !== null && 'response' in e) {
      const err = e as { response?: { data?: { detail?: string } } };
      msg = err.response?.data?.detail || '';
    }
    if (!msg && e instanceof Error) msg = e.message;
    ElMessage.error($tt('studio.error', { msg: msg || String(e) }));
  }
}

async function loadSystemInfo() {
  try {
    const info = await api.settings.getSystemInfo();
    systemInfo.value = info;
  } catch (e) {
    console.error('Failed to load system info:', e);
  }
}

function queueTruncate(text: string, length: number): string {
  if (!text) return '';
  return text.length > length ? text.substring(0, length) + '...' : text;
}

function taskKindLabel(kind?: string): string {
  if (!kind) return '';
  const key = 'taskKind.' + String(kind).replace(/\./g, '_');
  const result = $tt(key);
  return result && result !== key ? result : kind;
}

let sysInfoInterval: ReturnType<typeof setInterval> | null = null;

onMounted(async () => {
  const savedLang = getItem(DQ_STORAGE.LANG);
  if (savedLang) {
    currentLang.value = savedLang;
  }

  await loadSystemInfo();

  try {
    const st = await api.settings.getSettings();
    if (st?.theme) {
      applyTheme(st.theme);
    }
  } catch (e) {
    console.warn('Theme bootstrap skipped:', e);
  }

  sysInfoInterval = setInterval(loadSystemInfo, 30000);

  tasksStore.ensureQueuePoller();

  window.addEventListener('open-global-task-queue', onOpenGlobalTaskQueue);
});

function onOpenGlobalTaskQueue() {
  showGlobalQueueDrawer.value = true;
}

onBeforeUnmount(() => {
  window.removeEventListener('open-global-task-queue', onOpenGlobalTaskQueue);
  if (sysInfoInterval) {
    clearInterval(sysInfoInterval);
  }
  tasksStore.releaseQueuePoller();
});

provide('systemInfo', systemInfo);
</script>

<style scoped>
.main-container {
  height: 100vh;
}
.app-header {
  padding: 0 20px;
  display: flex;
  align-items: center;
}
.app-main {
  padding: 20px;
  overflow-y: auto;
}
</style>