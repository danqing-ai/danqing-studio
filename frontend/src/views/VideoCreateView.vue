<!-- @ts-nocheck -->
<template>
  <div class="create-page">
    <el-row :gutter="24">
      <!-- Left panel: creation area -->
      <el-col :xs="24" :md="16" :lg="17" :xl="18">
        <div class="creation-panel">
          <!-- Plan §3.1: text-to-video / image-to-video sub-tabs -->
          <el-segmented
            class="dq-work-segmented dq-work-segmented--sm"
            :model-value="videoWorkMode"
            :options="videoWorkSegmentOptions"
            block
            @update:model-value="setVideoWorkMode"
          />

          <!-- Model selector: single-level dropdown -->
          <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb studio-ep-model-card">
            <template #header>
              <div class="card-title">
                <el-icon><cpu /></el-icon>
                {{ $t('create.modelSelectTitle') }}
              </div>
            </template>
            <div class="studio-ep-model-toolbar">
              <el-select
                v-model="selectedModelVersion"
                size="large"
                filterable
                @change="onModelVersionChange"
                :placeholder="$t('studio.selectModel')"
              >
                <el-option
                  v-for="item in videoModelPickerVersions"
                  :key="item.modelKey + '|' + item.versionKey"
                  :label="item.name"
                  :value="item.modelKey + '|' + item.versionKey"
                  :disabled="!item.ready"
                >
                  <div class="studio-ep-picker-option">
                    <span class="studio-ep-picker-option__name" :class="{ 'is-disabled': !item.ready }">{{ item.name }}</span>
                    <ModelLicenseBadges
                      :recommended="item.recommended"
                      :commercial-use-allowed="item.commercialUseAllowed"
                      effect="plain"
                    />
                    <el-tag v-if="item.status === 'ready'" size="small" type="success">{{ $t('studio.ready') }}</el-tag>
                    <el-tag v-else size="small" type="warning">{{ $t('studio.notDownloaded') }}</el-tag>
                    <span v-if="item.size" class="studio-ep-picker-option__meta">
                      {{ item.size }}
                    </span>
                  </div>
                </el-option>
              </el-select>
            </div>
            <el-alert
              v-if="selectedModelNotReady"
              :title="$tt('studio.modelNotReady', { name: currentModelDisplayName })"
              type="warning"
              :closable="false"
              class="studio-ep-alert-mt"
            >
              <template #default>
                <span>{{ $t('studio.notDownloadedMsg') }}</span>
                <el-button size="small" type="primary" class="studio-ep-alert-inline-btn" @click="goToDownload">
                  {{ $t('studio.goDownload') }}
                </el-button>
              </template>
            </el-alert>
          </el-card>

          <el-alert
            v-if="videoWorkMode !== 'upscale'"
            type="info"
            :closable="false"
            show-icon
            class="studio-ep-alert-mb studio-ep-runtime-alert"
          >
            <template #title>{{ $t('video.runtimeCardTitle') }}</template>
            <div class="studio-ep-alert-body">
              <p>{{ $tt('video.runtimeClipSecs', { sec: outputClipSecRounded }) }}</p>
              <p>{{ $t('video.runtimeGenWarning') }}</p>
              <p v-if="currentVersionDiskSize">{{ $tt('video.runtimeModelSize', { size: currentVersionDiskSize }) }}</p>
            </div>
          </el-alert>
          <el-alert
            v-else
            type="warning"
            :closable="false"
            show-icon
            class="studio-ep-alert-mb studio-ep-runtime-alert"
          >
            <template #title>{{ $t('video.runtimeCardTitle') }}</template>
            <div class="studio-ep-alert-body">
              <p>{{ $t('video.runtimeUpscaleNote') }}</p>
              <p v-if="currentVersionDiskSize">{{ $tt('video.runtimeModelSize', { size: currentVersionDiskSize }) }}</p>
            </div>
          </el-alert>

          <!-- Animate: start image (required) -->
          <el-card v-if="videoWorkMode === 'animate'" shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <template #header>
              <div class="card-title card-title--split">
                <span>
                  <el-icon><PictureFilled /></el-icon>
                  {{ $t('action.video.startImage') }}
                </span>
              </div>
            </template>

            <div v-if="startImageSrc" class="ref-image-thumb" @click="showStartImagePreview">
              <img :src="startImageSrc" alt="start" />
              <div class="ref-image-actions">
                <el-button size="small" circle @click.stop="showStartImagePreview" :title="$t('studio.zoomIn')">
                  <el-icon><ZoomIn /></el-icon>
                </el-button>
                <el-button size="small" circle type="danger" @click.stop="removeStartImage" :title="$t('studio.delete')">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </div>
            </div>
            <div v-else class="ref-image-placeholder">
              <asset-picker
                accept-kind="image"
                :recent-gallery="recentStartImages"
                @pick="onStartAssetPick"
              />
            </div>
          </el-card>

          <el-card v-if="videoWorkMode === 'animate'" shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <template #header>
              <div class="card-title card-title--split">
                <span>
                  <el-icon><PictureFilled /></el-icon>
                  {{ $t('video.tailFrameTitle') }}
                </span>
              </div>
            </template>
            <div class="studio-ep-placeholder-hint">{{ $t('video.tailFrameHint') }}</div>
            <p class="studio-ep-field-footnote">{{ $t('studio.optional') }}</p>
            <div v-if="tailImageSrc" class="ref-image-thumb" @click="showTailImagePreview">
              <img :src="tailImageSrc" alt="tail" />
              <div class="ref-image-actions">
                <el-button size="small" circle @click.stop="showTailImagePreview" :title="$t('studio.zoomIn')">
                  <el-icon><ZoomIn /></el-icon>
                </el-button>
                <el-button size="small" circle type="danger" @click.stop="removeTailImage" :title="$t('studio.delete')">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </div>
            </div>
            <div v-else class="ref-image-placeholder">
              <asset-picker
                accept-kind="image"
                :recent-gallery="recentStartImages"
                @pick="onTailAssetPick"
              />
            </div>
          </el-card>

          <el-card v-if="videoWorkMode === 'upscale'" shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <template #header>
              <div class="card-title">
                <el-icon><video-camera /></el-icon>
                {{ $t('video.videoSourceTitle') }}
              </div>
            </template>
            <div v-if="sourceVideoSrc" class="ref-image-thumb ref-image-thumb--169">
              <video :src="sourceVideoSrc" controls></video>
              <div class="ref-image-actions">
                <el-button size="small" circle type="danger" @click.stop="removeSourceVideo" :title="$t('studio.delete')">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </div>
            </div>
            <div v-else class="ref-image-placeholder">
              <asset-picker
                accept-kind="video"
                :recent-gallery="recentVideos"
                @pick="onSourceVideoPick"
              />
            </div>
          </el-card>

          <el-card v-if="videoWorkMode === 'upscale'" shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <template #header>
              <div class="card-title">
                <el-icon><zoom-in /></el-icon>
                {{ $t('action.video.upscale') }}
              </div>
            </template>
            <el-form label-position="top" size="small">
              <el-form-item :label="$t('create.upscaleScale')">
                <el-select v-model="params.upscale_scale" class="studio-ep-w-full">
                  <el-option label="2×" :value="2" />
                  <el-option label="4×" :value="4" />
                </el-select>
              </el-form-item>
              <el-form-item :label="$t('create.upscaleDenoise')">
                <div class="studio-ep-slider-inline">
                  <el-slider v-model="params.upscale_denoise" :min="0" :max="1" :step="0.05" />
                  <el-input-number v-model="params.upscale_denoise" :min="0" :max="1" :step="0.05" class="param-input-number" />
                </div>
              </el-form-item>
              <el-form-item :label="$t('video.maxFramesLabel')">
                <el-input-number v-model="params.upscale_max_frames" :min="1" :max="4000" :step="1" class="studio-ep-w-full" />
              </el-form-item>
              <el-form-item :label="$t('studio.seed')">
                <div class="studio-ep-seed-row">
                  <el-input v-model="params.seed" :placeholder="$t('studio.seedPlaceholder')" />
                  <el-button @click="params.seed = String(Math.floor(Math.random() * 1000000))">
                    <el-icon><refresh /></el-icon>
                  </el-button>
                </div>
              </el-form-item>
            </el-form>
          </el-card>

          <!-- Prompt input -->
          <el-card
            v-if="videoWorkMode !== 'upscale'"
            shadow="never"
            class="studio-ep-surface-card studio-ep-card-mb"
          >
            <template #header>
              <div class="card-title">
                <el-icon><edit-pen /></el-icon>
                {{ $t('studio.prompt') }}
              </div>
            </template>

            <el-row :gutter="8" class="studio-ep-presets-row">
              <el-col :span="18">
                <el-select
                  v-model="selectedPreset"
                  :placeholder="$t('create.preset')"
                  class="studio-ep-w-full"
                  clearable
                >
                  <el-option
                    v-for="(preset, name) in filteredPresets"
                    :key="name"
                    :label="presetSelectLabel(name, preset)"
                    :value="name"
                  />
                </el-select>
              </el-col>
              <el-col :span="6">
                <el-button class="studio-ep-w-full" @click="loadPreset">
                  {{ $t('create.loadPreset') }}
                </el-button>
              </el-col>
            </el-row>

            <el-input
              v-model="params.prompt"
              type="textarea"
              :rows="5"
              :placeholder="$t('video.promptPlaceholder')"
              resize="none"
              @keydown.meta.enter.prevent="startGeneration"
              @keydown.ctrl.enter.prevent="startGeneration"
            />

            <!-- Negative prompt -->
            <el-collapse v-if="currentModelConfig?.parameters?.negative_prompt_support" class="studio-ep-collapse-plain">
              <el-collapse-item :title="$t('studio.negativePrompt')" name="negative">
                <el-input
                  v-model="params.negative_prompt"
                  type="textarea"
                  :rows="2"
                  :placeholder="$t('video.negativePlaceholder')"
                />
              </el-collapse-item>
            </el-collapse>
          </el-card>

          <!-- Advanced params -->
          <el-card
            v-if="videoWorkMode !== 'upscale'"
            shadow="never"
            class="studio-ep-surface-card studio-ep-card-mb"
          >
            <el-collapse v-model="advancedParamsOpen" class="studio-ep-collapse-plain">
              <el-collapse-item name="advanced">
                <template #title>
                  <div class="studio-ep-collapse-title-row">
                    <el-icon><setting /></el-icon>
                    <span>{{ $t('studio.advancedParams') }}</span>
                    <el-tag v-if="hasCustomParams" size="small" type="warning">{{ $t('studio.hasCustom') }}</el-tag>
                  </div>
                </template>

                <el-form label-position="top" size="small" class="studio-ep-form-pt">
                  <!-- Steps -->
                  <el-form-item v-if="currentModelConfig?.parameters?.steps" :label="$t('studio.steps')">
                    <div class="param-control-row">
                      <div class="param-slider">
                        <el-slider
                          v-model="params.steps"
                          :min="currentModelConfig.parameters.steps.min"
                          :max="currentModelConfig.parameters.steps.max"
                        />
                      </div>
                      <el-input-number v-model="params.steps" :min="1" :max="100" class="param-input-number" />
                    </div>
                  </el-form-item>

                  <!-- CFG -->
                  <el-form-item v-if="currentModelConfig?.parameters?.guide_scale" :label="$t('video.guideScaleLabel')">
                    <div class="param-control-row">
                      <div class="param-slider">
                        <el-slider
                          v-model="params.guide_scale"
                          :min="currentModelConfig.parameters.guide_scale.min"
                          :max="currentModelConfig.parameters.guide_scale.max"
                          :step="0.1"
                        />
                      </div>
                      <el-input-number v-model="params.guide_scale" :step="0.1" class="param-input-number" />
                    </div>
                  </el-form-item>

                  <!-- Shift (Wan only) -->
                  <el-form-item v-if="currentModelConfig?.parameters?.shift" :label="$t('video.shiftLabel')">
                    <div class="param-control-row">
                      <div class="param-slider">
                        <el-slider
                          v-model="params.shift"
                          :min="currentModelConfig.parameters.shift.min"
                          :max="currentModelConfig.parameters.shift.max"
                          :step="0.5"
                        />
                      </div>
                      <el-input-number v-model="params.shift" :step="0.5" class="param-input-number" />
                    </div>
                  </el-form-item>

                  <!-- Resolution -->
                  <el-form-item v-if="currentModelConfig?.parameters?.width" :label="$t('studio.resolution')">
                    <div class="studio-ep-res-row">
                      <el-select v-model="params.width" class="studio-ep-select-w120">
                        <el-option
                          v-for="w in currentModelConfig.parameters.width.options"
                          :key="w"
                          :label="w"
                          :value="w"
                        />
                      </el-select>
                      <span class="studio-ep-res-x">x</span>
                      <el-select v-model="params.height" class="studio-ep-select-w120">
                        <el-option
                          v-for="h in currentModelConfig.parameters.height.options"
                          :key="h"
                          :label="h"
                          :value="h"
                        />
                      </el-select>
                    </div>
                  </el-form-item>

                  <!-- Num frames -->
                  <el-form-item v-if="currentModelConfig?.parameters?.num_frames" :label="$t('video.numFramesLabel')">
                    <div class="param-control-row">
                      <div class="param-slider">
                        <el-slider
                          v-model="params.num_frames"
                          :min="currentModelConfig.parameters.num_frames.min"
                          :max="currentModelConfig.parameters.num_frames.max"
                          :step="currentModelConfig.parameters.num_frames.step || 1"
                        />
                      </div>
                      <el-input-number v-model="params.num_frames" :min="1" :max="257" class="param-input-number" />
                    </div>
                    <div v-if="currentModelConfig.parameters.num_frames.note" class="studio-ep-param-note">
                      {{ currentModelConfig.parameters.num_frames.note }}
                    </div>
                  </el-form-item>

                  <!-- FPS -->
                  <el-form-item v-if="currentModelConfig?.parameters?.fps" :label="$t('video.fpsLabel')">
                    <div class="param-control-row">
                      <div class="param-slider">
                        <el-slider
                          v-model="params.fps"
                          :min="currentModelConfig.parameters.fps.min"
                          :max="currentModelConfig.parameters.fps.max"
                        />
                      </div>
                      <el-input-number v-model="params.fps" :min="1" :max="60" class="param-input-number" />
                    </div>
                  </el-form-item>

                  <!-- Seed -->
                  <el-form-item v-if="currentModelConfig?.parameters?.seed_support" :label="$t('studio.seed')">
                    <div class="studio-ep-seed-row">
                      <el-input v-model="params.seed" :placeholder="$t('studio.seedPlaceholder')" />
                  <el-button @click="params.seed = String(Math.floor(Math.random() * 1000000))">
                    <el-icon><refresh /></el-icon>
                  </el-button>
                    </div>
                  </el-form-item>

                  <!-- Restore defaults -->
                  <el-form-item>
                    <el-button text type="primary" @click="resetToDefaults" size="small">
                      <el-icon><refresh /></el-icon>
                      {{ $t('studio.restoreDefaults') }}
                    </el-button>
                  </el-form-item>
                </el-form>
              </el-collapse-item>
            </el-collapse>
          </el-card>

          <!-- LoRA selector -->
          <el-card
            v-if="videoWorkMode !== 'upscale' && currentModelConfig?.parameters?.lora_support"
            shadow="never"
            class="studio-ep-surface-card studio-ep-card-mb"
          >
            <div class="studio-ep-lora-section-title">
              <el-icon><collection-tag /></el-icon>
              <span>{{ $t('studio.loraLabel') }}</span>
            </div>

            <!-- Selected LoRA list -->
            <div v-if="selectedLoras.length > 0" class="studio-ep-lora-stack">
              <div
                v-for="(lora, index) in selectedLoras"
                :key="lora.id"
                class="studio-ep-lora-row"
              >
                <span class="studio-ep-lora-name">
                  {{ compatibleLoras.find(c => c.id === lora.id)?.name || lora.id }}
                </span>
                <el-slider
                  v-model="lora.weight"
                  :min="0"
                  :max="2"
                  :step="0.1"
                  class="studio-ep-lora-slider"
                />
                <span class="studio-ep-lora-weight-num">{{ lora.weight.toFixed(1) }}</span>
                <el-button size="small" text @click="moveLoraUp(index)" :disabled="index === 0">
                  <el-icon><arrow-up /></el-icon>
                </el-button>
                <el-button size="small" text @click="moveLoraDown(index)" :disabled="index === selectedLoras.length - 1">
                  <el-icon><arrow-down /></el-icon>
                </el-button>
                <el-button size="small" text type="danger" @click="removeLora(index)">
                  <el-icon><delete /></el-icon>
                </el-button>
              </div>
            </div>

            <!-- Add LoRA -->
            <el-select
              :model-value="''"
              class="studio-ep-w-full"
              :placeholder="$t('studio.noLora')"
              @update:model-value="addLora($event)"
            >
              <el-option :label="$t('studio.noLora')" value="" />
              <el-option
                v-for="lora in compatibleLoras.filter(c => !selectedLoras.find(s => s.id === c.id))"
                :key="lora.id"
                :label="lora.name || lora.id"
                :value="lora.id"
              />
            </el-select>
          </el-card>

          <!-- Generate button -->
          <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <el-button
              type="primary"
              size="large"
              class="studio-ep-primary-cta studio-ep-primary-cta--simple"
              :disabled="submitDisabled || !systemInfo?.env_ready"
              @click="startGeneration"
            >
              <el-icon size="20"><video-camera /></el-icon>
              <span class="studio-ep-cta-gap">
                {{ primaryCtaLabel }}
              </span>
            </el-button>
            <div class="studio-ep-micro-hint">
              {{ $sendShortcutHint() }}
            </div>

            <!-- Progress display -->
            <div v-if="currentTask" class="studio-ep-task-wrap">
              <el-progress
                :percentage="Math.round(currentTask.progress * 100)"
                :status="currentTask.status === 'failed' ? 'exception' : ''"
              />
              <div class="studio-ep-task-status">
                <template v-if="currentTask.total > 0 && currentTask.status === 'running'">
                  Step {{ currentTask.step }}/{{ currentTask.total }} &nbsp;
                </template>
                <el-tag :type="getStatusType(currentTask.status)" size="small">
                  {{ getStatusText(currentTask.status) }}
                </el-tag>
              </div>
            </div>
          </el-card>

          <!-- Logs -->
          <el-card shadow="never" class="studio-ep-surface-card">
            <template #header>
              <div class="card-title card-title--split">
                <span>
                  <el-icon><document /></el-icon>
                  {{ $t('studio.logs') }}
                </span>
                <el-button size="small" text @click="clearLogs">
                  <el-icon><delete /></el-icon>
                </el-button>
              </div>
            </template>

            <div class="log-container studio-ep-log-container--sm" ref="logContainer">
              <div v-if="logs.length === 0" class="studio-ep-log-empty">
                {{ $t('studio.logsEmpty') }}
              </div>
              <div v-for="(log, index) in logs" :key="index" class="log-line">
                <span class="log-timestamp">{{ log.time }}</span>
                <span :class="'log-' + log.level">{{ log.message }}</span>
              </div>
            </div>
          </el-card>
        </div>
      </el-col>

      <!-- Right panel -->
      <el-col :xs="24" :md="8" :lg="7" :xl="6">
        <div class="preview-panel">
          <!-- Current generation preview -->
          <el-card shadow="never" class="studio-ep-surface-card studio-ep-card-mb">
            <template #header>
              <div class="card-title">
                <el-icon><video-camera /></el-icon>
                {{ $t('studio.currentPreview') }}
              </div>
            </template>

            <div v-if="previewVideo" class="video-preview studio-ep-video-preview">
              <video :src="previewVideo" controls></video>
            </div>
            <el-empty v-else :description="$t('studio.noPreview')" />
          </el-card>

          <!-- Recent generations -->
          <el-card shadow="never" class="studio-ep-surface-card">
            <template #header>
              <div class="card-title card-title--split">
                <span>
                  <el-icon><clock /></el-icon>
                  {{ $t('studio.recent') }}
                </span>
                <el-button size="small" text @click="loadRecentVideos">
                  <el-icon><refresh /></el-icon>
                </el-button>
              </div>
            </template>

            <el-empty v-if="recentVideos.length === 0" :description="$t('gallery.empty')" />

            <el-row v-else :gutter="8">
              <el-col
                v-for="video in recentVideos"
                :key="video.path"
                :span="12"
                class="studio-ep-gallery-col"
              >
                <div class="gallery-card" @click="showVideoPreview(video)">
                  <div class="gallery-image-wrapper studio-ep-recent-video-wrap">
                    <video :src="getVideoUrl(video)" preload="metadata"></video>
                  </div>
                </div>
              </el-col>
            </el-row>
          </el-card>
        </div>
      </el-col>
    </el-row>

    <!-- Start image preview dialog -->
    <el-dialog v-model="startImageViewerVisible" :title="$t('action.video.startImage')" width="70%" center>
      <div v-if="startImageSrc" class="studio-ep-dialog-center">
        <img class="studio-ep-dialog-img-tall" :src="startImageSrc" />
      </div>
    </el-dialog>

    <el-dialog v-model="tailImageViewerVisible" :title="$t('video.tailFrameTitle')" width="70%" center>
      <div v-if="tailImageSrc" class="studio-ep-dialog-center">
        <img class="studio-ep-dialog-img-tall" :src="tailImageSrc" />
      </div>
    </el-dialog>

    <!-- Video preview dialog -->
    <el-dialog v-model="videoPreviewVisible" :title="selectedVideo?.name" width="80%" center destroy-on-close>
      <div v-if="selectedVideo" class="studio-ep-dialog-center">
        <video class="studio-ep-dialog-video" :src="getVideoUrl(selectedVideo)" controls></video>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch, onMounted, inject, nextTick } from 'vue';
import type { Ref } from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import { api } from '@/utils/api';
import { $tt, $mn, $mvn, $pn } from '@/utils/i18n';
import { useRegistryStore } from '@/stores/registry';
import { DQ_STORAGE } from '@/utils/storage';
import type { SystemInfo, GalleryItem } from '@/types';
import { warnIfRiskyMemory } from '@/composables/memoryHint';
import { formatGenLogMessage, isDuplicateDenoiseStepLog } from '@/utils/genTaskLog';
import ModelLicenseBadges from '@/components/model/ModelLicenseBadges.vue';

const router = useRouter();
const registryStore = useRegistryStore();
const systemInfo = inject<Ref<SystemInfo>>('systemInfo');

// Inline legacy helpers
function hasAction(actions: Record<string, unknown>, key: string) {
  if (!actions || typeof actions !== 'object') return false;
  return Object.prototype.hasOwnProperty.call(actions, key) && actions[key] != null;
}
function videoModelRow(config: Record<string, unknown>) {
  return config && config.media === 'video';
}
function videoSupportsAnimate(actions: Record<string, unknown>) {
  return hasAction(actions, 'animate');
}
function videoSupportsUpscale(actions: Record<string, unknown>) {
  return hasAction(actions, 'upscale');
}
function videoSupportsCreate(actions: Record<string, unknown>) {
  return hasAction(actions, 'create');
}

function tagType(status: string) {
  const map: Record<string, string> = {
    pending: 'info',
    queued: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info',
  };
  return map[status] || 'info';
}
function statusText(status: string) {
  const suffix: Record<string, string> = {
    pending: 'pending',
    queued: 'queued',
    running: 'running',
    completed: 'completed',
    failed: 'failed',
    cancelled: 'cancelled',
  };
  const suf = suffix[status] || status;
  return $tt('studio.' + suf);
}

function parseModelVersionValue(value: string) {
  if (!value || typeof value !== 'string') return null;
  const parts = value.split('|');
  if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
  return { modelKey: parts[0], versionKey: parts[1] };
}

// Params
const params = reactive({
  prompt: '',
  negative_prompt: '',
  model: '',
  version: '',
  width: 768,
  height: 512,
  num_frames: 97,
  fps: 24,
  steps: 4,
  guide_scale: 3.0,
  shift: 0.0,
  seed: '',
  image_path: '',
  upscale_scale: 4,
  upscale_denoise: 0.3,
  upscale_max_frames: 300,
});

const selectedModelVersion = ref('');

// State
const currentTask = ref<any>(null);
const logs = ref<{ time: string; message: string; level: string }[]>([]);
const genLogLastStep = ref(0);
const previewVideo = ref('');
const recentVideos = ref<GalleryItem[]>([]);
const recentStartImages = ref<GalleryItem[]>([]);
const advancedParamsOpen = ref<string[]>(['advanced']);

/** Plan §3.1: Create (text-to-video) and Animate (image-to-video) */
const videoWorkMode = ref('create');
const videoWorkSegmentOptions = computed(() => [
  { label: $tt('action.video.create'), value: 'create' },
  { label: $tt('action.video.animate'), value: 'animate' },
  { label: $tt('action.video.upscale'), value: 'upscale' },
]);
const setVideoWorkMode = (mode: string) => {
  if (mode === 'animate') {
    videoWorkMode.value = 'animate';
  } else if (mode === 'upscale') {
    videoWorkMode.value = 'upscale';
  } else {
    videoWorkMode.value = 'create';
  }
};

// Start image
const startImageSrc = ref('');
const startImagePath = ref('');
const startImageViewerVisible = ref(false);
const tailImageSrc = ref('');
const tailImagePath = ref('');
const tailImageViewerVisible = ref(false);

const sourceVideoSrc = ref('');
const sourceVideoPath = ref('');

const onSourceVideoPick = (payload: { path?: string; previewUrl?: string }) => {
  sourceVideoPath.value = payload.path || '';
  sourceVideoSrc.value = payload.previewUrl || '';
};

const removeSourceVideo = () => {
  sourceVideoSrc.value = '';
  sourceVideoPath.value = '';
};

// Video preview
const videoPreviewVisible = ref(false);
const selectedVideo = ref<GalleryItem | null>(null);

const modelRegistry = ref<Record<string, any>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});

const selectedLoras = ref<{ id: string; weight: number }[]>([]);
const compatibleLoras = ref<{ id: string; name?: string; parameters?: any }[]>([]);

const loadCompatibleLoras = async () => {
  if (!params.model) {
    compatibleLoras.value = [];
    return;
  }
  try {
    const loras = await api.settings.getCompatibleLoras(params.model);
    compatibleLoras.value = (loras as any[]) || [];
  } catch (e) {
    console.error('Failed to load compatible loras:', e);
    compatibleLoras.value = [];
  }
};

const addLora = (loraId: string) => {
  if (!loraId) return;
  if (selectedLoras.value.find((l) => l.id === loraId)) return;
  const lora = compatibleLoras.value.find((l) => l.id === loraId);
  const defaultWeight =
    lora && lora.parameters && lora.parameters.lora_scale
      ? lora.parameters.lora_scale.default
      : 1.0;
  selectedLoras.value.push({ id: loraId, weight: defaultWeight });
};

const removeLora = (index: number) => {
  selectedLoras.value.splice(index, 1);
};

const moveLoraUp = (index: number) => {
  if (index <= 0) return;
  const tmp = selectedLoras.value[index];
  selectedLoras.value[index] = selectedLoras.value[index - 1];
  selectedLoras.value[index - 1] = tmp;
};

const moveLoraDown = (index: number) => {
  if (index >= selectedLoras.value.length - 1) return;
  const tmp = selectedLoras.value[index];
  selectedLoras.value[index] = selectedLoras.value[index + 1];
  selectedLoras.value[index + 1] = tmp;
};

// All model versions
const allVersions = computed(() => {
  const result: any[] = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!videoModelRow(config)) {
      continue;
    }
    const actions = { ...(config.actions || {}) };
    const versions = config.versions || {};
    const detailed = modelsDetailedStatus.value[modelKey] || {};
    const versionStatuses = detailed.versions || {};

    for (const [versionKey, versionConfig] of Object.entries(versions)) {
      const status = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
      result.push({
        modelKey,
        versionKey,
        name: $mvn(modelKey, config, versionConfig as any),
        size: (versionConfig as any).size || '',
        status: status.status,
        ready: status.ready,
        recommended: config.recommended && (versionConfig as any).default,
        commercialUseAllowed: config.commercial_use_allowed === true,
        actions,
      });
    }
  }
  return result;
});

const videoVersionsForMode = computed(() => {
  const filtered = allVersions.value.filter((v) => {
    const acts = v.actions || {};
    if (videoWorkMode.value === 'animate') {
      return videoSupportsAnimate(acts);
    }
    if (videoWorkMode.value === 'upscale') {
      return videoSupportsUpscale(acts);
    }
    return videoSupportsCreate(acts);
  });
  if (videoWorkMode.value === 'upscale' || videoWorkMode.value === 'animate') {
    return filtered;
  }
  return filtered.length ? filtered : allVersions.value;
});

const videoRecommendedForMode = computed(() => {
  return videoVersionsForMode.value.filter((v) => v.recommended);
});

const videoModelPickerVersions = computed(() => {
  const rows = videoVersionsForMode.value.filter((v) => v.ready);
  rows.sort((a, b) => {
    const ar = a.recommended ? 1 : 0;
    const br = b.recommended ? 1 : 0;
    if (ar !== br) return br - ar;
    const an = a.name || '';
    const bn = b.name || '';
    try {
      return an.localeCompare(bn, 'zh');
    } catch {
      return an < bn ? -1 : an > bn ? 1 : 0;
    }
  });
  return rows;
});

const currentModelConfig = computed(() => modelRegistry.value[params.model] || null);

const currentModelDisplayName = computed(() => {
  const c = currentModelConfig.value;
  if (c) {
    return $mn(c, params.model);
  }
  return params.model || '';
});

// Whether current selected version is ready
const selectedModelNotReady = computed(() => {
  if (!params.model || !params.version) return false;
  const detailed = modelsDetailedStatus.value[params.model];
  if (!detailed || !detailed.versions) return true;
  const versionStatus = detailed.versions[params.version];
  return !versionStatus || !versionStatus.ready;
});

const submitDisabled = computed(() => {
  if (selectedModelNotReady.value) return true;
  if (videoWorkMode.value === 'upscale') {
    return !sourceVideoSrc.value;
  }
  if (!String(params.prompt || '').trim()) return true;
  if (videoWorkMode.value === 'animate' && !startImageSrc.value) return true;
  return false;
});

const primaryCtaLabel = computed(() => {
  if (videoWorkMode.value === 'animate') return $tt('action.video.animate');
  if (videoWorkMode.value === 'upscale') return $tt('action.video.upscale');
  return $tt('action.video.create');
});

/** Plan §3.2: Output clip duration (seconds, one decimal) estimated by num_frames / fps */
const outputClipSecRounded = computed(() => {
  const fps = Math.max(1, Number(params.fps) || 1);
  const nf = Math.max(1, Number(params.num_frames) || 1);
  return Math.round((nf / fps) * 10) / 10;
});

/** Current version's size field from registry (e.g., 19GB), for VRAM/disk hints */
const currentVersionDiskSize = computed(() => {
  const cfg = currentModelConfig.value;
  if (!cfg || !params.version) return '';
  const v = (cfg.versions || {})[params.version];
  return v && v.size ? String(v.size) : '';
});

// Load model registry and status
const loadModelRegistry = async () => {
  try {
    const regPromise = registryStore.registry
      ? Promise.resolve(registryStore.registry)
      : registryStore.load().then((r) => r || { models: {} });
    const [registryData, detailedStatusData] = await Promise.all([
      regPromise,
      api.settings.getModelsDetailedStatus(),
    ]);

    modelRegistry.value = (registryData && (registryData as any).models) || {};
    modelsDetailedStatus.value = (detailedStatusData as any) || {};

    // Set default model
    if (!selectedModelVersion.value) {
      let found = false;
      for (const item of videoRecommendedForMode.value) {
        if (item.ready) {
          params.model = item.modelKey;
          params.version = item.versionKey;
          selectedModelVersion.value = item.modelKey + '|' + item.versionKey;
          found = true;
          break;
        }
      }
      if (!found) {
        for (const item of videoVersionsForMode.value) {
          if (item.ready) {
            params.model = item.modelKey;
            params.version = item.versionKey;
            selectedModelVersion.value = item.modelKey + '|' + item.versionKey;
            found = true;
            break;
          }
        }
      }
      if (!found && videoVersionsForMode.value.length > 0) {
        const first = videoVersionsForMode.value[0];
        params.model = first.modelKey;
        params.version = first.versionKey;
        selectedModelVersion.value = first.modelKey + '|' + first.versionKey;
      }
    }

    loadModelDefaults();
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
};

// Load model default config
const loadModelDefaults = () => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return;

  const p = config.parameters;
  if (videoWorkMode.value === 'upscale') {
    if (p.scale_factor && p.scale_factor.default != null) {
      params.upscale_scale = p.scale_factor.default;
    }
    if (p.max_frames && p.max_frames.default != null) {
      params.upscale_max_frames = p.max_frames.default;
    }
    if (p.fps) params.fps = p.fps.default;
    params.seed = '';
    return;
  }
  if (p.steps) params.steps = p.steps.default;
  if (p.guide_scale) params.guide_scale = p.guide_scale.default;
  if (p.shift) params.shift = p.shift.default;
  if (p.width) params.width = p.width.default;
  if (p.height) params.height = p.height.default;
  if (p.num_frames) params.num_frames = p.num_frames.default;
  if (p.fps) params.fps = p.fps.default;
  params.seed = '';
};

// Reset to default config
const resetToDefaults = () => {
  loadModelDefaults();
  ElMessage.success($tt('studio.restoredDefaults'));
};

// Check if custom params exist
const hasCustomParams = computed(() => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return false;
  const p = config.parameters;
  if (videoWorkMode.value === 'upscale') {
    if (p.scale_factor && params.upscale_scale !== p.scale_factor.default) return true;
    if (p.max_frames && params.upscale_max_frames !== p.max_frames.default) return true;
    if (params.seed) return true;
    return false;
  }
  if (p.steps && params.steps !== p.steps.default) return true;
  if (p.guide_scale && params.guide_scale !== p.guide_scale.default) return true;
  if (p.shift && params.shift !== p.shift.default) return true;
  if (p.width && params.width !== p.width.default) return true;
  if (p.height && params.height !== p.height.default) return true;
  if (p.num_frames && params.num_frames !== p.num_frames.default) return true;
  if (p.fps && params.fps !== p.fps.default) return true;
  if (params.seed) return true;
  return false;
});

const presets = ref<Record<string, any>>({});
const selectedPreset = ref('');

const presetActionFilter = computed(() => {
  if (videoWorkMode.value === 'animate') {
    return new Set(['animate']);
  }
  if (videoWorkMode.value === 'upscale') {
    return new Set(['upscale']);
  }
  return new Set(['create']);
});

const filteredPresets = computed(() => {
  const want = presetActionFilter.value;

  function planPresetShapeOk(preset: any) {
    return (
      Array.isArray(preset.applies_to) &&
      preset.applies_to.length > 0 &&
      (preset.media_scope === 'image' || preset.media_scope === 'video')
    );
  }

  function matchesMediaScope(preset: any) {
    return preset.media_scope === 'video';
  }

  function matches(preset: any) {
    if (!planPresetShapeOk(preset)) return false;
    if (!matchesMediaScope(preset)) return false;
    return preset.applies_to.some((k: string) => want.has(k));
  }
  const entries = Object.entries(presets.value)
    .filter(([, preset]) => matches(preset))
    .sort((a: [string, any], b: [string, any]) => {
      const ac = a[1].applies_to.includes('create');
      const bc = b[1].applies_to.includes('create');
      if (ac !== bc) {
        return ac ? -1 : 1;
      }
      return a[0].localeCompare(b[0], 'zh');
    });
  const result: Record<string, any> = {};
  for (const [name, preset] of entries) {
    result[name] = preset;
  }
  return result;
});

const presetSelectLabel = (name: string, preset: any) => {
  const a = preset.applies_to;
  const hasC = a.includes('create');
  const hasA = a.includes('animate');
  const hasU = a.includes('upscale');
  let tag = '';
  if (hasC && hasA) {
    tag = $tt('video.presetTagHybrid');
  } else if (hasC && !hasA) {
    tag = $tt('video.presetTagT2V');
  } else if (hasA && !hasC) {
    tag = $tt('video.presetTagI2V');
  } else if (hasU && !hasC && !hasA) {
    tag = $tt('video.presetTagUpscale');
  }
  const display = $pn(preset, name);
  return tag ? `${tag} ${display}` : display;
};

const loadPresets = async () => {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as any) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
    presets.value = {};
  }
};

const loadPreset = () => {
  if (!selectedPreset.value || !presets.value[selectedPreset.value]) return;
  const preset = presets.value[selectedPreset.value];
  const app = preset.applies_to;
  const animateOnly = app.includes('animate') && !app.includes('create');
  if (animateOnly && (videoWorkMode.value === 'create' || !startImageSrc.value)) {
    ElMessage.warning($tt('video.presetNeedsStartImage'));
  }
  if (preset.positive) {
    params.prompt = params.prompt
      ? params.prompt + '\nStyle boost: ' + preset.positive
      : preset.positive;
  }
  if (preset.negative) {
    params.negative_prompt = params.negative_prompt
      ? params.negative_prompt + '\n' + preset.negative
      : preset.negative;
  }
};

// Add log
const addLog = (message: string, level = 'info') => {
  const now = new Date();
  const time = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
  logs.value.push({ time, message, level });

  if (logs.value.length > 500) {
    logs.value = logs.value.slice(-500);
  }

  nextTick(() => {
    const container = document.querySelector('.log-container');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  });
};

function ingestServerLog(logData: { message?: string; level?: string }) {
  const raw = logData.message || '';
  const lvl = logData.level || 'info';
  if (isDuplicateDenoiseStepLog(logs.value, raw)) {
    return;
  }
  addLog(formatGenLogMessage(raw), lvl);
}

// Clear logs
const clearLogs = () => {
  logs.value = [];
};

// Start generation
const startGeneration = async () => {
  if (videoWorkMode.value !== 'upscale' && !String(params.prompt || '').trim()) {
    ElMessage.warning($tt('studio.enterPrompt'));
    return;
  }

  const detailed = modelsDetailedStatus.value[params.model];
  const versionStatus = detailed?.versions?.[params.version];
  if (!versionStatus?.ready) {
    ElMessage.warning(
      $tt('studio.modelNotReadyDesc', {
        name: currentModelConfig.value?.name || params.model,
        version: params.version,
      })
    );
    return;
  }

  const verCfg =
    (currentModelConfig.value &&
      currentModelConfig.value.versions &&
      currentModelConfig.value.versions[params.version]) ||
    null;
  const sizeHuman = verCfg && verCfg.size ? String(verCfg.size) : '';
  warnIfRiskyMemory({ systemInfo: systemInfo?.value, versionSizeHuman: sizeHuman, $tt });

  addLog($tt('studio.startingGen'), 'info');

  try {
    const modelStr = params.version ? `${params.model}:${params.version}` : params.model;
    let submitRes: any;
    if (videoWorkMode.value === 'animate') {
      if (!startImageSrc.value) {
        ElMessage.warning($tt('video.needStartImage'));
        return;
      }
      let source_asset_id: string;
      const sp = startImagePath.value;
      if (typeof sp === 'string' && sp.startsWith('asset:')) {
        source_asset_id = sp.slice('asset:'.length);
      } else {
        const blob = await api.gen.urlToBlob(startImageSrc.value);
        const up = await api.gen.uploadAsset(
          new File([blob], 'start.png', { type: blob.type || 'image/png' })
        );
        source_asset_id = (up as any).id;
      }
      let tail_asset_id: string | undefined;
      if (tailImageSrc.value) {
        const tp = tailImagePath.value;
        if (typeof tp === 'string' && tp.startsWith('asset:')) {
          tail_asset_id = tp.slice('asset:'.length);
        } else {
          const tblob = await api.gen.urlToBlob(tailImageSrc.value);
          const tup = await api.gen.uploadAsset(
            new File([tblob], 'tail.png', { type: tblob.type || 'image/png' })
          );
          tail_asset_id = (tup as any).id;
        }
      }
      const animateBody: Record<string, unknown> = {
        model: modelStr,
        operation: 'animate',
        source_asset_id,
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        size: `${params.width}x${params.height}`,
        num_frames: params.num_frames,
        fps: params.fps || 16,
        steps: params.steps,
        guidance: params.guide_scale,
        shift: params.shift || undefined,
        seed: params.seed ? parseInt(params.seed, 10) : null,
        priority: 'normal',
      };
      if (tail_asset_id) {
        animateBody.tail_asset_id = tail_asset_id;
      }
      if (selectedLoras.value.length > 0) {
        animateBody.adapters = selectedLoras.value.map((l) => ({ id: l.id, weight: l.weight }));
      }
      submitRes = await api.gen.createVideoEdit(animateBody);
    } else if (videoWorkMode.value === 'upscale') {
      if (!sourceVideoSrc.value) {
        ElMessage.warning($tt('video.upscaleNeedSource'));
        return;
      }
      let source_asset_id: string;
      const vp = sourceVideoPath.value;
      if (typeof vp === 'string' && vp.startsWith('asset:')) {
        source_asset_id = vp.slice('asset:'.length);
      } else {
        const blob = await api.gen.urlToBlob(sourceVideoSrc.value);
        const ext =
          (blob.type && blob.type.includes('webm') && 'webm') ||
          (blob.type && blob.type.includes('quicktime') && 'mov') ||
          'mp4';
        const up = await api.gen.uploadAsset(
          new File([blob], `upscale-src.${ext}`, { type: blob.type || 'video/mp4' })
        );
        source_asset_id = (up as any).id;
      }
      const upscaleBody: Record<string, unknown> = {
        model: modelStr,
        source_asset_id,
        scale: Number(params.upscale_scale) === 4 ? 4 : 2,
        denoise: Number(params.upscale_denoise) || 0.3,
        max_frames: Math.min(
          4000,
          Math.max(1, parseInt(String(params.upscale_max_frames), 10) || 300)
        ),
        metadata: {},
        priority: 'normal',
      };
      const sd = params.seed ? parseInt(String(params.seed), 10) : null;
      if (sd != null && !Number.isNaN(sd)) {
        (upscaleBody.metadata as Record<string, unknown>).seed = sd;
      }
      submitRes = await api.gen.createVideoUpscale(upscaleBody);
    } else {
      const body: Record<string, unknown> = {
        model: modelStr,
        prompt: params.prompt,
        negative_prompt: params.negative_prompt || '',
        size: `${params.width}x${params.height}`,
        num_frames: params.num_frames,
        fps: params.fps || 16,
        steps: params.steps,
        guidance: params.guide_scale,
        shift: params.shift || undefined,
        seed: params.seed ? parseInt(params.seed, 10) : null,
        priority: 'normal',
      };
      if (selectedLoras.value.length > 0) {
        body.adapters = selectedLoras.value.map((l) => ({ id: l.id, weight: l.weight }));
      }
      submitRes = await api.gen.createVideoGeneration(body);
    }
    const tid = submitRes.task.id;
    genLogLastStep.value = 0;
    currentTask.value = {
      id: tid,
      progress: 0,
      step: 0,
      total: 0,
      status: 'queued',
      params: { model: modelStr },
    };
    api.gen.streamMediaTask(tid, {
      onLog: (logData: any) => ingestServerLog(logData),
      onStatus: (statusData: any) => {
        if (currentTask.value) {
          currentTask.value.progress = statusData.progress ?? 0;
          currentTask.value.status = statusData.status;
        }
      },
      onDone: async (doneData: any) => {
        if (doneData.status === 'completed') {
          addLog($tt('studio.genComplete'), 'success');
          const updated = await api.gen.getMediaTask(tid) as any;
          currentTask.value = updated;
          const pid = updated.result && updated.result.primary_asset_id;
          if (pid) {
            previewVideo.value = api.gallery.getImageUrl(`asset:${pid}`);
            addLog($tt('studio.outputFile', { name: pid }), 'info');
          } else {
            addLog(
              $tt('studio.noOutputAsset', {
                msg:
                  (updated.error_message || '').trim() ||
                  $tt('studio.noOutputAssetHint'),
              }),
              'warning'
            );
          }
          loadRecentVideos();
        } else if (doneData.status === 'failed') {
          const updated = await api.gen.getMediaTask(tid) as any;
          currentTask.value = updated;
          addLog($tt('studio.genFailed', { msg: updated.error_message || '' }), 'error');
        }
      },
      onError: () => addLog($tt('studio.connectionLost'), 'warning'),
      onProgress: (progressData: any) => {
        if (!currentTask.value) return;
        if (typeof progressData.progress === 'number') {
          currentTask.value.progress = progressData.progress;
        }
        const nextStep =
          progressData.step != null
            ? progressData.step
            : currentTask.value.step;
        const nextTotal =
          progressData.total != null
            ? progressData.total
            : currentTask.value.total;
        currentTask.value.step = nextStep;
        currentTask.value.total = nextTotal;
        if (nextTotal > 0 && nextStep > 0) {
          genLogLastStep.value = nextStep;
        }
      },
    });
  } catch (e: any) {
    addLog($tt('studio.error', { msg: e.message }), 'error');
  }
};

// Load recent videos
const loadRecentVideos = async () => {
  try {
    const videos = await api.gallery.listImages(4, 0);
    // Filter video files
    recentVideos.value = videos.filter((v) => {
      if (v.metadata && v.metadata.asset_kind === 'video') {
        return true;
      }
      const ext = v.name?.split('.').pop()?.toLowerCase();
      return ['mp4', 'mov', 'avi', 'mkv'].includes(ext || '');
    });
  } catch (e) {
    console.error('Failed to load recent videos:', e);
  }
};

const loadRecentStartImages = async () => {
  try {
    const images = await api.gallery.listImages(24, 0);
    recentStartImages.value = images
      .filter((v) => {
        if (v.metadata && v.metadata.asset_kind === 'video') {
          return false;
        }
        const ext = v.name?.split('.').pop()?.toLowerCase();
        return !['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext || '');
      })
      .slice(0, 8);
  } catch (e) {
    console.error('Failed to load recent start images:', e);
  }
};

// Get video URL
const getVideoUrl = (video: GalleryItem) => {
  return api.gallery.getImageUrl(video.path);
};

// Show video preview
const showVideoPreview = (video: GalleryItem) => {
  selectedVideo.value = video;
  videoPreviewVisible.value = true;
};

// Start image related
const onStartAssetPick = async (payload: { path?: string; previewUrl?: string }) => {
  startImagePath.value = payload.path || '';
  startImageSrc.value = payload.previewUrl || '';
  addLog($tt('studio.startImageAdded', { name: (payload.path || '').replace(/^asset:/, '') }), 'info');
  await loadRecentStartImages();
};

const removeStartImage = () => {
  startImageSrc.value = '';
  startImagePath.value = '';
};

const showStartImagePreview = () => {
  startImageViewerVisible.value = true;
};

const onTailAssetPick = async (payload: { path?: string; previewUrl?: string }) => {
  tailImagePath.value = payload.path || '';
  tailImageSrc.value = payload.previewUrl || '';
  addLog($tt('studio.startImageAdded', { name: (payload.path || '').replace(/^asset:/, '') }), 'info');
  await loadRecentStartImages();
};

const removeTailImage = () => {
  tailImageSrc.value = '';
  tailImagePath.value = '';
};

const showTailImagePreview = () => {
  tailImageViewerVisible.value = true;
};

// Navigate to download page
const goToDownload = () => router.push({ name: 'models' });

const getStatusType = (status: string) =>
  tagType(status);
const getStatusText = (status: string) =>
  statusText(status);

const onModelVersionChange = (value: string) => {
  const parsed = parseModelVersionValue(value);
  if (!parsed) return;
  params.model = parsed.modelKey;
  params.version = parsed.versionKey;
  selectedLoras.value = []; // Clear selected LoRAs when switching models
  loadModelDefaults();
  loadCompatibleLoras();
  addLog(
    $tt('studio.switchModel', {
      name: currentModelConfig.value?.name || params.model,
      version: params.version,
    }),
    'info'
  );
};

const videoAutoSaveDraft = ref(false);
let _vidPromptSaveT: ReturnType<typeof setTimeout> | null = null;
watch(
  () => params.prompt,
  (v) => {
    if (!videoAutoSaveDraft.value) return;
    if (!DQ_STORAGE.VIDEO_CREATE_PROMPT_DRAFT) return;
    if (_vidPromptSaveT) clearTimeout(_vidPromptSaveT);
    _vidPromptSaveT = setTimeout(() => {
      try {
        localStorage.setItem(DQ_STORAGE.VIDEO_CREATE_PROMPT_DRAFT, String(v || ''));
      } catch (_) {
        /* ignore */
      }
    }, 500);
  }
);

const applyVideoAppSettingsDefaults = async () => {
  try {
    const st = await api.settings.getSettings();
    videoAutoSaveDraft.value = !!(st as any).auto_save_prompts;
    if ((st as any).auto_save_prompts && DQ_STORAGE.VIDEO_CREATE_PROMPT_DRAFT) {
      const draft = localStorage.getItem(DQ_STORAGE.VIDEO_CREATE_PROMPT_DRAFT);
      if (draft) params.prompt = draft;
    }
    const dm = ((st as any).default_model || '').trim();
    if (!dm || !modelRegistry.value || !Object.keys(modelRegistry.value).length) return;
    let mk: string | null = null;
    if (modelRegistry.value[dm]) {
      mk = dm;
    } else {
      for (const [k, cfg] of Object.entries(modelRegistry.value)) {
        const media = cfg && cfg.media;
        if (media !== 'video') continue;
        const n = cfg && cfg.name;
        if (typeof n === 'string' && n === dm) {
          mk = k;
          break;
        }
        if (n && typeof n === 'object' && (n.zh === dm || n.en === dm)) {
          mk = k;
          break;
        }
      }
    }
    if (!mk || !modelRegistry.value[mk]) return;
    const detailed = modelsDetailedStatus.value[mk] || {};
    const vers = detailed.versions || {};
    const cfg = modelRegistry.value[mk];
    const versionKeys = Object.keys(cfg.versions || {});
    const defaultVK =
      versionKeys.find((vk) => cfg.versions[vk] && cfg.versions[vk].default) || versionKeys[0];
    if (!defaultVK) return;
    const stRow = vers[defaultVK];
    if (stRow && stRow.ready === false) return;
    params.model = mk;
    params.version = defaultVK;
    selectedModelVersion.value = mk + '|' + defaultVK;
    loadModelDefaults();
  } catch (_) {
    /* ignore */
  }
};

onMounted(async () => {
  await loadModelRegistry();
  await applyVideoAppSettingsDefaults();
  loadPresets();
  loadRecentVideos();
  loadRecentStartImages();
});

watch(videoWorkMode, () => {
  const cfg = currentModelConfig.value;
  const acts = cfg && cfg.actions ? cfg.actions : {};
  let ok = true;
  if (videoWorkMode.value === 'animate') {
    ok = videoSupportsAnimate(acts);
  } else if (videoWorkMode.value === 'upscale') {
    ok = videoSupportsUpscale(acts);
  } else {
    ok = videoSupportsCreate(acts);
  }
  if (!ok) {
    const first = videoRecommendedForMode.value[0] || videoVersionsForMode.value[0];
    if (first) {
      params.model = first.modelKey;
      params.version = first.versionKey;
      selectedModelVersion.value = first.modelKey + '|' + first.versionKey;
      loadModelDefaults();
    }
  }
});
</script>
