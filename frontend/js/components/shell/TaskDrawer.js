/**
 * TaskDrawer — Global task queue drawer
 * Props:  modelValue (Boolean), queue ({ running, queued })
 * Emits:  update:modelValue, cancel-task(taskId), set-priority(taskId, priority)
 */
const TaskDrawer = {
    name: 'TaskDrawer',
    props: {
        modelValue: { type: Boolean, default: false },
        queue: { type: Object, default: function () { return { running: [], queued: [] }; } },
    },
    emits: ['update:modelValue', 'cancel-task', 'set-priority'],
    template: `
        <el-drawer
            :model-value="modelValue"
            @update:model-value="onVisibleChange"
            class="dq-task-queue-drawer"
            :title="$t('studio.queueDialogTitle')"
            direction="rtl"
            size="420px"
        >
            <div v-if="queue.running.length === 0 && queue.queued.length === 0" style="text-align: center; padding: 40px; color: var(--text-muted);">
                {{ $t('studio.queueEmpty') }}
            </div>
            <div v-else>
                <!-- Running -->
                <div v-if="queue.running.length > 0" style="margin-bottom: 20px;">
                    <div style="font-weight: 600; margin-bottom: 12px; color: var(--primary-color);">{{ $t('studio.running') }}</div>
                    <div v-for="task in queue.running" :key="task.id" class="queue-dialog-item running">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <div style="flex: 1; overflow: hidden; min-width: 0;">
                                <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 4px;">{{ taskKindLabel(task.kind) }}</div>
                                <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 4px;">{{ task.params?.model || $t('queue.unspecifiedModel') }}</div>
                                <div class="dq-queue-prompt-line">{{ truncatePrompt(task.params?.prompt || '', 40) }}</div>
                            </div>
                            <el-button
                                size="small"
                                circle
                                type="danger"
                                @click="$emit('cancel-task', task.id)"
                                style="margin-left: 8px; flex-shrink: 0;"
                                :title="$t('studio.cancelTask')"
                            >
                                <el-icon><delete /></el-icon>
                            </el-button>
                        </div>
                        <el-progress class="dq-queue-progress" :percentage="Math.round(task.progress * 100)" :stroke-width="4" />
                        <div v-if="task.total > 0" style="font-size: 11px; color: var(--text-muted); margin-top: 4px; text-align: right;">
                            <template v-if="String(task.kind || '').startsWith('image.')">
                                {{ $t('studio.queueDenoiseProgress', { current: task.step != null ? task.step : 0, total: task.total }) }}
                            </template>
                            <template v-else>
                                {{ $t('studio.queueStepProgress', { current: task.step != null ? task.step : 0, total: task.total }) }}
                            </template>
                        </div>
                        <div v-if="task.progressMessage === 'post' && typeof task.progress === 'number' && task.progress < 1" style="font-size: 11px; color: var(--text-muted); margin-top: 2px; text-align: right;">
                            {{ $t('studio.queuePostProcessHint') }}
                        </div>
                    </div>
                </div>
                <!-- Queued -->
                <div v-if="queue.queued.length > 0">
                    <div style="font-weight: 600; margin-bottom: 12px; color: var(--text-muted);">{{ $t('studio.queued') }} ({{ queue.queued.length }})</div>
                    <div v-for="(task, index) in queue.queued" :key="task.id" class="queue-dialog-item queued">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
                            <div style="flex: 1; display: flex; align-items: center; gap: 8px; overflow: hidden; min-width: 0;">
                                <span style="font-size: 12px; color: var(--text-muted); min-width: 24px;">#{{ index + 1 }}</span>
                                <div style="flex: 1; overflow: hidden;">
                                    <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 2px;">
                                        <span>{{ taskKindLabel(task.kind) }}</span>
                                        <el-tag v-if="(task.priority ?? 100) <= 50" size="small" type="warning" effect="plain" style="margin-left: 6px;">{{ $t('studio.queuePriorityHigh') }}</el-tag>
                                    </div>
                                    <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 2px;">{{ task.params?.model || $t('queue.unspecifiedModel') }}</div>
                                    <div class="dq-queue-prompt-line">{{ truncatePrompt(task.params?.prompt || '', 40) }}</div>
                                    <div v-if="task.estimated_wait_seconds != null" style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">
                                        {{ $tt('queue.estimatedWait', { s: task.estimated_wait_seconds }) }}
                                    </div>
                                </div>
                            </div>
                            <div style="display: flex; flex-direction: column; align-items: stretch; gap: 6px; flex-shrink: 0;">
                                <el-button size="small" @click="$emit('set-priority', task.id, 'high')" :disabled="(task.priority ?? 100) <= 50">{{ $t('studio.queueSetHigh') }}</el-button>
                                <el-button size="small" @click="$emit('set-priority', task.id, 'normal')" :disabled="(task.priority ?? 100) > 50">{{ $t('studio.queueSetNormal') }}</el-button>
                                <el-button
                                    size="small"
                                    circle
                                    type="danger"
                                    @click="$emit('cancel-task', task.id)"
                                    style="align-self: flex-end;"
                                    :title="$t('studio.cancelTask')"
                                >
                                    <el-icon><delete /></el-icon>
                                </el-button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </el-drawer>
    `,
    setup(props, { emit }) {
        function onVisibleChange(val) {
            emit('update:modelValue', val);
        }

        function truncatePrompt(text, length) {
            if (!text) return '';
            return text.length > length ? text.substring(0, length) + '...' : text;
        }

        function taskKindLabel(kind) {
            if (!kind) return '';
            var key = 'taskKind.' + String(kind).replace(/\./g, '_');
            try {
                // i18n is globally available from app.js
                var $t = window.$tt || (function (k) { return k; });
                var r = $t(key);
                return r && r !== key ? r : kind;
            } catch (_) {
                return kind;
            }
        }

        return { onVisibleChange, truncatePrompt, taskKindLabel };
    },
};
