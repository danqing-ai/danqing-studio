<script setup lang="ts">
import type { SystemInfo } from '@/types';
import { $tt } from '@/utils/i18n';

type CacheModel = {
  key: string;
  idle_minutes: number;
  size_gb: number;
};

type CacheSnapshot = {
  cached_models?: number;
  total_gb?: number;
  limit_gb?: number;
  models?: CacheModel[];
};

type MonitorData = {
  cpu_percent: number;
  memory: {
    total_gb: number;
    used_gb: number;
    percent: number;
  };
  gpu: null | {
    model?: string;
    memory_gb?: number;
    note?: string;
  };
};

defineProps<{
  systemInfo?: SystemInfo | null;
  cacheStatus: { cache: CacheSnapshot | null };
  cacheLoading: boolean;
  cacheError: string;
  monitorData: MonitorData;
}>();

const emit = defineEmits<{
  refreshCache: [];
}>();

function progressColor(percent: number) {
  if (percent < 50) return 'var(--dq-success)';
  if (percent < 80) return 'var(--dq-warning)';
  return 'var(--dq-danger)';
}
</script>

<template>
  <div class="settings-system-sidebar">
    <DqInspectorStack aria-label="system-sidebar">
      <DqInspectorSection :title="$t('settings.systemInfo')">
        <DqInspectorSectionBody v-if="systemInfo">
          <div class="system-info-grid system-info-grid--inspector">
            <div class="info-item">
              <div class="info-icon">
                <DqIcon class="settings-info-icon-lg"><monitor /></DqIcon>
              </div>
              <div class="info-content">
                <div class="info-label">{{ $t('settings.platform') }}</div>
                <div class="info-value">{{ systemInfo.platform }} {{ systemInfo.architecture }}</div>
              </div>
            </div>
            <div class="info-item">
              <div class="info-icon">
                <DqIcon class="settings-info-icon-lg"><cpu /></DqIcon>
              </div>
              <div class="info-content">
                <div class="info-label">{{ $t('settings.memory') }}</div>
                <div class="info-value">{{ systemInfo.memory_gb?.toFixed(1) }} GB</div>
              </div>
            </div>
            <div class="info-item">
              <div class="info-icon">
                <DqIcon class="settings-info-icon-lg"><document /></DqIcon>
              </div>
              <div class="info-content">
                <div class="info-label">{{ $t('settings.pythonVersion') }}</div>
                <div class="info-value">{{ systemInfo.python_version }}</div>
              </div>
            </div>
          </div>
          <div v-if="systemInfo.dependencies" class="settings-dependencies settings-dependencies--inspector">
            <div class="settings-dependencies-title">{{ $t('settings.dependencies') }}</div>
            <div class="settings-dep-tags">
              <DqTag
                v-for="(version, name) in systemInfo.dependencies"
                :key="name"
                size="small"
                type="info"
                effect="plain"
              >
                {{ name }} {{ version }}
              </DqTag>
            </div>
          </div>
        </DqInspectorSectionBody>
      </DqInspectorSection>

      <DqInspectorSection :title="$t('settings.modelCacheTitle')">
        <template #actions>
          <DqIconButton type="text" size="sm" :label="$t('gallery.refresh')" :disabled="cacheLoading" @click="emit('refreshCache')">
            <DqIcon><refresh /></DqIcon>
          </DqIconButton>
        </template>
        <DqInspectorSectionBody>
          <DqInspectorCallout v-if="cacheError" variant="warn" :title="cacheError" />
          <template v-else>
            <p v-if="cacheStatus.cache" class="dq-inspector-cache-summary">
              {{
                $tt('settings.modelCacheTotal', {
                  count: cacheStatus.cache.cached_models ?? 0,
                  total: cacheStatus.cache.total_gb ?? 0,
                  limit: cacheStatus.cache.limit_gb ?? 0,
                })
              }}
            </p>
            <DqInspectorList v-if="cacheStatus.cache?.models?.length">
              <DqInspectorListItem
                v-for="m in cacheStatus.cache.models"
                :key="m.key"
                class="dq-inspector-cache-item"
              >
                <div class="cache-item-icon">
                  <DqIcon><cpu /></DqIcon>
                </div>
                <div class="cache-item-info">
                  <div class="cache-item-name">{{ m.key }}</div>
                  <div class="cache-item-meta">
                    {{ $tt('settings.modelCacheIdle', { minutes: m.idle_minutes }) }}
                  </div>
                </div>
                <div class="cache-item-size">{{ m.size_gb }} GB</div>
              </DqInspectorListItem>
            </DqInspectorList>
            <DqInspectorEmpty v-else>
              {{ $t('settings.modelCacheEmpty') }}
            </DqInspectorEmpty>
          </template>
        </DqInspectorSectionBody>
      </DqInspectorSection>

      <DqInspectorSection :title="$t('settings.resourceMonitor')" last>
        <p class="dq-inspector-monitor-sub">
          {{ $t('settings.realtime') }}
        </p>
        <DqInspectorSectionBody>
          <div class="dq-inspector-meter">
            <DqInspectorKv :label="$t('settings.cpu')" :value="`${monitorData.cpu_percent}%`" />
            <DqProgress
              class="dq-inspector-progress"
              :percentage="monitorData.cpu_percent"
              :color="progressColor(monitorData.cpu_percent)"
              :show-text="false"
              :stroke-width="6"
            />
          </div>

          <div class="dq-inspector-meter">
            <DqInspectorKv
              :label="$t('settings.memoryLabel')"
              :value="`${monitorData.memory.used_gb} / ${monitorData.memory.total_gb} GB`"
            />
            <DqProgress
              class="dq-inspector-progress"
              :percentage="monitorData.memory.percent"
              :color="progressColor(monitorData.memory.percent)"
              :show-text="false"
              :stroke-width="6"
            />
          </div>

          <div v-if="monitorData.gpu" class="dq-inspector-meter dq-inspector-meter--stacked">
            <DqInspectorKv
              v-if="monitorData.gpu.model"
              :label="$t('settings.gpu')"
              :value="monitorData.gpu.model"
            />
            <p v-else class="dq-inspector-footnote">{{ $t('settings.gpu') }}</p>
            <div class="settings-monitor-gpu-meta">
              <span v-if="monitorData.gpu.memory_gb">
                {{ monitorData.gpu.memory_gb }} {{ $t('settings.unifiedMemory') }}
              </span>
              <span v-if="monitorData.gpu.note">{{ monitorData.gpu.note }}</span>
            </div>
          </div>

          <p class="dq-inspector-footnote dq-inspector-monitor-foot">
            {{ $t('settings.refreshInterval') }}
          </p>
        </DqInspectorSectionBody>
      </DqInspectorSection>
    </DqInspectorStack>
  </div>
</template>
