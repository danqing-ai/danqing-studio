<!-- @ts-nocheck -->
<template>
  <div class="models-page">
    <!-- Left sidebar: category navigation -->
    <div class="models-page__sidebar">
      <el-card shadow="never" class="studio-ep-surface-card models-page__sidebar-card">
        <div class="card-title">
          <el-icon><box /></el-icon>
          {{ $t('download.downloadCenter') }}
        </div>
        <div class="models-page__sidebar-intro">
          {{ $t('models.pageSubtitle') }}
        </div>

        <el-menu
          class="dq-download-menu models-page__menu"
          :default-active="activeCategory"
          @select="handleCategorySelect"
        >
          <el-menu-item index="all">
            <el-icon><grid /></el-icon>
            <span>{{ $t('download.allModels') }}</span>
            <el-tag size="small" type="info" class="dq-menu-end-tag">
              {{ totalModelCount }}
            </el-tag>
          </el-menu-item>

          <el-menu-item index="image_models">
            <el-icon><picture-filled /></el-icon>
            <span>{{ $t('download.imageModels') }}</span>
          </el-menu-item>

          <el-menu-item index="video_models">
            <el-icon><video-camera /></el-icon>
            <span>{{ $t('download.videoModels') }}</span>
          </el-menu-item>

          <el-menu-item index="music_models">
            <el-icon><headset /></el-icon>
            <span>{{ $t('download.audioModels') }}</span>
          </el-menu-item>

          <el-menu-item index="controlnets">
            <el-icon><aim /></el-icon>
            <span>{{ $t('download.controlNet') }}</span>
          </el-menu-item>

          <el-menu-item index="upscalers">
            <el-icon><zoom-in /></el-icon>
            <span>{{ $t('download.upscalers') }}</span>
          </el-menu-item>

          <el-menu-item index="tools">
            <el-icon><tools /></el-icon>
            <span>{{ $t('download.tools') }}</span>
          </el-menu-item>

          <el-menu-item index="loras">
            <el-icon><magic-stick /></el-icon>
            <span>{{ $t('download.loraModels') }}</span>
          </el-menu-item>

          <el-divider />

          <el-menu-item index="downloading">
            <el-icon><download /></el-icon>
            <span>{{ $t('download.downloadingTab') }}</span>
            <el-tag
              v-if="activeDownloadCount > 0"
              size="small"
              type="primary"
              class="dq-menu-end-tag"
            >
              {{ activeDownloadCount }}
            </el-tag>
          </el-menu-item>

          <el-menu-item index="installed">
            <el-icon><folder-checked /></el-icon>
            <span>{{ $t('download.installed') }}</span>
          </el-menu-item>
        </el-menu>

        <!-- Disk space -->
        <div v-if="diskSpace" class="disk-space-panel">
          <div class="disk-space-title">
            <el-icon><monitor /></el-icon>
            {{ $t('download.diskSpace') }}
          </div>
          <div class="disk-space-item">
            <div class="disk-space-label">
              <span>{{ $t('download.modelLabel') }}</span>
              <span class="disk-space-value">{{ diskSpace.models?.size_human }}</span>
            </div>
            <el-progress
              :percentage="getDiskPercent('models')"
              :show-text="false"
              :stroke-width="4"
            />
          </div>
          <div class="disk-space-item">
            <div class="disk-space-label">
              <span>{{ $t('download.loraLabel') }}</span>
              <span class="disk-space-value">{{ diskSpace.loras?.size_human }}</span>
            </div>
            <el-progress
              :percentage="getDiskPercent('loras')"
              :show-text="false"
              :stroke-width="4"
              color="var(--el-color-success)"
            />
          </div>
          <div class="disk-space-footer">
            {{ $t('download.free') }}: {{ diskSpace.models?.free_human }}
          </div>
        </div>
      </el-card>
    </div>

    <!-- Right content area -->
    <div class="models-page__main">
      <!-- Model grid (category browsing) -->
      <div
        v-if="
          [
            'all',
            'image_models',
            'video_models',
            'music_models',
            'controlnets',
            'upscalers',
            'tools',
            'loras',
          ].includes(activeCategory)
        "
      >
        <!-- Category title -->
        <div class="page-header models-page__page-header">
          <h2 class="page-title models-page__category-title">
            <el-icon v-if="categoryPageIcon" class="models-page__title-icon" aria-hidden="true">
              <component :is="categoryPageIcon" />
            </el-icon>
            <span>{{ categoryTitleText }}</span>
          </h2>
          <div class="page-actions models-page__toolbar">
            <el-button
              v-if="
                activeCategory !== 'loras' && activeCategory !== 'installed'
              "
              class="models-toolbar-btn models-page__import-btn"
              plain
              @click="showImportDialog"
            >
              <el-icon class="models-toolbar-btn__icon"><upload /></el-icon>
              <span class="models-toolbar-btn__label">{{ $t('download.importLocal') }}</span>
            </el-button>
            <el-input
              v-model="filterQuery"
              :placeholder="$t('download.searchModel')"
              class="models-page__search-input"
              clearable
            >
              <template #prefix>
                <el-icon><search /></el-icon>
              </template>
            </el-input>
            <el-button
              class="models-toolbar-btn models-page__refresh-btn"
              plain
              :loading="refreshing"
              @click="refreshStatus"
            >
              <el-icon class="models-toolbar-btn__icon"><refresh /></el-icon>
              <span class="models-toolbar-btn__label">{{ $t('gallery.refresh') }}</span>
            </el-button>
          </div>
        </div>

        <!-- Import local model dialog -->
        <el-dialog
          v-model="importDialogVisible"
          :title="$t('download.importTitle')"
          width="500px"
        >
          <el-form label-position="top">
            <el-form-item :label="$t('download.modelName')">
              <el-input
                v-model="importModelName"
                :placeholder="$t('download.modelNamePlaceholder')"
              />
            </el-form-item>
            <el-form-item :label="$t('download.modelPath')">
              <el-input
                v-model="importModelPath"
                :placeholder="$t('download.modelPathPlaceholder')"
              />
            </el-form-item>
            <el-form-item :label="$t('download.modelType')">
              <el-select v-model="importModelType" class="models-import-model-type">
                <el-option
                  :label="$t('download.baseModel')"
                  value="base"
                />
                <el-option
                  :label="$t('download.loraType')"
                  value="lora"
                />
                <el-option
                  :label="$t('download.controlnetType')"
                  value="controlnet"
                />
              </el-select>
            </el-form-item>
          </el-form>
          <template #footer>
            <el-button @click="importDialogVisible = false">
              {{ $t('download.cancel') }}
            </el-button>
            <el-button
              type="primary"
              :loading="importing"
              @click="importLocalModel"
            >
              {{ $t('download.import_') }}
            </el-button>
          </template>
        </el-dialog>

        <!-- Model card grid -->
        <el-row :gutter="16" class="model-grid model-grid--fluid">
          <el-col
            v-for="model in filteredModels"
            :key="model.id"
            :xs="24"
            :sm="12"
            :md="8"
            :lg="6"
            :xl="4"
            class="models-page__col-mb"
          >
            <el-card
              class="model-card"
              :class="{ 'model-ready': model.ready }"
            >
              <!-- Card header: icon/preview + status -->
              <div class="model-card-header">
                <div class="model-icon">
                  {{ getModelInitials(model) }}
                </div>
                <div class="model-card-badges">
                  <el-tag
                    v-if="model.recommended"
                    size="small"
                    class="recommended-badge"
                    type="success"
                    effect="dark"
                  >
                    {{ $t('download.recommendedBadge') }}
                  </el-tag>
                  <el-tooltip
                    v-if="isCommercialUseAllowed(model)"
                    :content="$t('download.commercialUseBadgeTip')"
                    placement="bottom"
                    :show-after="400"
                  >
                    <el-tag
                      size="small"
                      class="commercial-badge"
                      effect="dark"
                    >
                      {{ $t('download.commercialUseBadge') }}
                    </el-tag>
                  </el-tooltip>
                </div>
                <div class="model-status">
                  <el-tag
                    v-if="modelsDetailedStatus[model.id]?.status === 'ready'"
                    size="small"
                    type="success"
                  >
                    {{ $t('download.readyTag') }}
                  </el-tag>
                  <el-tag
                    v-else-if="
                      modelsDetailedStatus[model.id]?.status === 'incomplete'
                    "
                    size="small"
                    type="danger"
                  >
                    {{ $t('download.incompleteTag') }}
                  </el-tag>
                  <el-tag v-else size="small" type="warning">
                    {{ $t('download.notDownloadedTag') }}
                  </el-tag>
                </div>
              </div>

              <!-- Card content -->
              <div class="model-card-content">
                <div class="model-card-name">
                  {{ $mn(model) }}
                </div>
                <el-tooltip
                  :content="$md(model)"
                  placement="top"
                  effect="dark"
                >
                  <div class="model-card-desc">
                    {{ $md(model) }}
                  </div>
                </el-tooltip>

                <!-- Meta info -->
                <div class="model-card-meta">
                  <el-tag
                    v-if="model.size"
                    size="small"
                    type="info"
                    effect="plain"
                  >
                    {{ model.size }}
                  </el-tag>
                  <el-tag
                    v-if="model.source === 'huggingface'"
                    size="small"
                    type="primary"
                    effect="plain"
                  >
                    HF
                  </el-tag>
                  <el-tag
                    v-else-if="model.source === 'modelscope'"
                    size="small"
                    type="info"
                    effect="plain"
                  >
                    ModelScope
                  </el-tag>
                  <el-tag
                    v-else-if="model.source === 'civitai'"
                    size="small"
                    type="warning"
                    effect="plain"
                  >
                    CivitAI
                  </el-tag>
                  <el-tag
                    v-if="model.base_model"
                    size="small"
                    type="success"
                    effect="plain"
                  >
                    {{ model.base_model }}
                  </el-tag>
                </div>

                <ul
                  v-if="model.versions"
                  class="model-version-list"
                  role="list"
                >
                  <li
                    v-for="row in modelVersionTableRows(model)"
                    :key="row.verKey"
                    class="model-version-row"
                    :class="{
                      'is-ready': row.vstatus === 'ready',
                      'is-pending': row.vstatus !== 'ready',
                    }"
                    role="listitem"
                  >
                    <div class="model-version-row__info">
                      <div class="model-version-cell-row">
                        <span class="model-version-name">{{ row.ver.name }}</span>
                        <el-tag v-if="row.ver.size" size="small" type="info" effect="plain">{{ row.ver.size }}</el-tag>
                        <el-tag
                          v-if="row.ver.source_type === 'derived'"
                          size="small"
                          type="warning"
                          effect="plain"
                        >
                          {{ $t('download.derivedTag') }}
                        </el-tag>
                        <el-tag
                          v-else-if="row.ver.source_type === 'prequantized'"
                          size="small"
                          type="info"
                          effect="plain"
                        >
                          {{ $t('download.prequantized') }}
                        </el-tag>
                        <el-tag
                          v-if="row.vstatus === 'ready'"
                          size="small"
                          type="success"
                          effect="plain"
                        >
                          {{ $t('studio.ready') }}
                        </el-tag>
                      </div>
                      <div
                        v-if="row.ver.source_type === 'derived'"
                        class="model-version-derived"
                      >
                        {{
                          $t('download.basedOn', {
                            name:
                              row.model.versions[row.ver.from_version]?.name ||
                              row.ver.from_version,
                          })
                        }}
                      </div>
                    </div>
                    <div class="model-version-row__actions">
                      <el-space wrap :size="6" justify="end">
                        <template v-if="row.vstatus === 'ready'">
                          <el-button
                            size="small"
                            plain
                            class="model-ver-btn model-ver-btn--force"
                            @click="downloadVersion(row.model, row.verKey)"
                          >
                            <el-icon class="model-ver-btn__icon"><download /></el-icon>
                            <span class="model-ver-btn__label">{{ $t('download.forceDownload') }}</span>
                          </el-button>
                          <el-button
                            size="small"
                            plain
                            class="model-ver-btn model-ver-btn--delete"
                            @click="deleteVersion(row.model, row.verKey)"
                          >
                            <el-icon class="model-ver-btn__icon"><delete /></el-icon>
                            <span class="model-ver-btn__label">{{ $t('common.delete') }}</span>
                          </el-button>
                        </template>
                        <template v-else-if="row.vstatus === 'parent_missing'">
                          <el-tooltip
                            v-if="!canDownload(row.model)"
                            :content="getDependencyHint(row.model)"
                            placement="top"
                          >
                            <span>
                              <el-button class="model-ver-btn model-ver-btn--download" size="small" plain disabled>
                                <el-icon class="model-ver-btn__icon"><download /></el-icon>
                                <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
                              </el-button>
                            </span>
                          </el-tooltip>
                          <el-button
                            v-else
                            class="model-ver-btn model-ver-btn--download"
                            size="small"
                            plain
                            :loading="downloadingModels[row.model.id + '-' + row.verKey]"
                            @click="
                              downloadVersion(row.model, row.ver.from_version, {
                                uiLoadingKey: row.model.id + '-' + row.verKey,
                              })
                            "
                          >
                            <el-icon class="model-ver-btn__icon"><download /></el-icon>
                            <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
                          </el-button>
                        </template>
                        <template v-else-if="row.vstatus === 'quantize'">
                          <el-tooltip
                            v-if="!canDownload(row.model)"
                            :content="getDependencyHint(row.model)"
                            placement="top"
                          >
                            <span>
                              <el-button class="model-ver-btn model-ver-btn--quantize" size="small" plain disabled>
                                <el-icon class="model-ver-btn__icon"><cpu /></el-icon>
                                <span class="model-ver-btn__label">{{ $t('download.quantizeVersion') }}</span>
                              </el-button>
                            </span>
                          </el-tooltip>
                          <el-button
                            v-else
                            class="model-ver-btn model-ver-btn--quantize"
                            size="small"
                            plain
                            :loading="downloadingModels[row.model.id + '-' + row.verKey]"
                            @click="quantizeVersion(row.model, row.verKey)"
                          >
                            <el-icon class="model-ver-btn__icon"><cpu /></el-icon>
                            <span class="model-ver-btn__label">{{ $t('download.quantizeVersion') }}</span>
                          </el-button>
                        </template>
                        <template v-else>
                          <el-tooltip
                            v-if="!canDownload(row.model)"
                            :content="getDependencyHint(row.model)"
                            placement="top"
                          >
                            <span>
                              <el-button class="model-ver-btn model-ver-btn--download" size="small" plain disabled>
                                <el-icon class="model-ver-btn__icon"><download /></el-icon>
                                <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
                              </el-button>
                            </span>
                          </el-tooltip>
                          <el-button
                            v-else
                            class="model-ver-btn model-ver-btn--download"
                            size="small"
                            plain
                            :loading="downloadingModels[row.model.id + '-' + row.verKey]"
                            @click="downloadVersion(row.model, row.verKey)"
                          >
                            <el-icon class="model-ver-btn__icon"><download /></el-icon>
                            <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
                          </el-button>
                        </template>
                      </el-space>
                    </div>
                  </li>
                </ul>

              </div>
            </el-card>
          </el-col>
        </el-row>

        <el-empty
          v-if="filteredModels.length === 0 && activeCategory !== 'loras'"
          :description="$t('download.noModelsInCategory')"
        />

        <!-- LoRA search -->
        <div
          v-if="activeCategory === 'loras'"
          class="models-lora-section"
        >
          <div class="page-header">
            <h2 class="page-title">{{ $t('download.civitaiSearch') }}</h2>
          </div>

          <el-card shadow="never" class="studio-ep-surface-card models-page__col-mb">
            <div class="models-civit-search-row">
              <el-input
                v-model="searchQuery"
                :placeholder="$t('download.searchCivitai')"
                clearable
                @keyup.enter="searchCivitai"
              >
                <template #prefix>
                  <el-icon><search /></el-icon>
                </template>
              </el-input>
              <el-select v-model="searchType" class="models-civit-search-type">
                <el-option label="LoRA" value="LORA" />
                <el-option label="Checkpoint" value="Checkpoint" />
                <el-option :label="$t('download.all')" value="LORA,Checkpoint" />
              </el-select>
              <el-button
                class="models-toolbar-btn models-toolbar-btn--primary"
                :loading="searching"
                @click="searchCivitai"
              >
                <el-icon class="models-toolbar-btn__icon"><search /></el-icon>
                <span class="models-toolbar-btn__label">{{ $t('download.search') }}</span>
              </el-button>
            </div>
          </el-card>

          <el-row v-if="searchResults.length > 0" :gutter="16">
            <el-col
              v-for="model in searchResults"
              :key="model.id"
              :xs="24"
              :sm="12"
              :md="8"
              class="models-page__col-mb"
            >
              <el-card class="civitai-card">
                <div class="models-civit-card-inner">
                  <div class="civitai-preview">
                    <img
                      v-if="
                        getCivitaiPreviewUrl(model) &&
                        !civitaiPreviewLoadFailed[String(model.id)]
                      "
                      :src="getCivitaiPreviewUrl(model)"
                      loading="lazy"
                      :alt="model.name"
                      @error="onCivitaiPreviewError(model.id)"
                    />
                    <div
                      v-else
                      class="no-preview"
                    >
                      <el-icon><picture-filled /></el-icon>
                    </div>
                  </div>

                  <div class="models-civit-side">
                    <div class="civitai-name">{{ model.name }}</div>
                    <div class="models-civit-meta">
                      {{ model.type }} |
                      {{ model.model_versions[0]?.base_model || 'Unknown' }}
                    </div>
                    <div class="models-civit-meta models-civit-meta--creator">
                      {{
                        model.creator?.username ||
                        $tt('download.unknownCreator')
                      }}
                    </div>
                    <div class="models-civit-tags-row">
                      <el-tag
                        v-if="model.nsfw"
                        size="small"
                        type="danger"
                      >
                        {{ $t('download.nsfwTag') }}
                      </el-tag>
                      <el-tag size="small" type="info">
                        <el-icon><download /></el-icon>
                        {{ formatNumber(model.stats?.downloadCount || 0) }}
                      </el-tag>
                    </div>
                  </div>
                </div>

                <div class="models-civit-footer-actions">
                  <el-select
                    v-model="selectedVersions[model.id]"
                    size="small"
                    class="models-civit-version-select"
                    :placeholder="$t('download.selectVersion')"
                  >
                    <el-option
                      v-for="v in model.model_versions"
                      :key="v.id"
                      :label="v.name"
                      :value="v.id"
                    />
                  </el-select>
                  <el-button
                    class="model-ver-btn model-ver-btn--download"
                    size="small"
                    plain
                    :loading="downloadingLoras[model.id]"
                    @click="downloadCivitaiModel(model)"
                  >
                    <el-icon class="model-ver-btn__icon"><download /></el-icon>
                    <span class="model-ver-btn__label">{{ $t('download.download_') }}</span>
                  </el-button>
                </div>
              </el-card>
            </el-col>
          </el-row>

          <el-empty
            v-else-if="!searching && hasSearched"
            :description="$t('download.noResults')"
          />
        </div>
      </div>

      <!-- Installed -->
      <div v-if="activeCategory === 'installed'">
        <div class="page-header">
          <h2 class="page-title">{{ $t('download.installedLabel') }}</h2>
        </div>

        <div v-if="installedModels.length" class="models-installed-list" role="list">
          <article
            v-for="(row, idx) in installedModels"
            :key="row.path || row.name || idx"
            class="models-installed-row"
            role="listitem"
          >
            <div class="models-installed-row__main">
              <span class="models-installed-row__name">{{ row.name }}</span>
              <el-tag size="small" :type="getModelTypeTagType(row.type)">
                {{ row.type || 'unknown' }}
              </el-tag>
            </div>
            <span class="models-installed-row__size">{{ row.size_human }}</span>
            <span class="models-installed-row__path" :title="row.path">{{ row.path }}</span>
          </article>
        </div>
        <el-empty v-else :description="$t('download.noModels')" />
      </div>

      <!-- Downloading -->
      <div v-if="activeCategory === 'downloading'">
        <div class="page-header">
          <h2 class="page-title">
            {{ $t('download.downloadingTab') }} ({{ activeDownloadCount }})
          </h2>
        </div>

        <el-card v-if="activeDownloadCount === 0" shadow="never" class="studio-ep-surface-card">
          <el-empty :description="$t('download.noTasks')" />
        </el-card>

        <el-card v-else shadow="never" class="studio-ep-surface-card">
          <div
            v-for="(item, taskId) in activeDownloads"
            :key="taskId"
            class="models-download-task"
          >
            <div class="models-download-task-head">
              <span class="models-download-task-name">{{ item.name }}</span>
              <div class="models-download-task-meta">
                <span class="models-download-progress-text">
                  <span v-if="item.total_size > 0">
                    {{ Math.round(item.progress * 100) }}%
                    <span v-if="item.speed">({{ item.speed }})</span>
                  </span>
                  <span v-else-if="item.downloaded_size > 0">
                    {{ formatBytes(item.downloaded_size) }}
                    <span v-if="item.speed">({{ item.speed }})</span>
                  </span>
                  <span v-else>{{ $t('download.preparing') }}</span>
                </span>
                <el-button
                  v-if="item.status === 'paused'"
                  class="model-ver-btn model-ver-btn--download"
                  size="small"
                  plain
                  @click="resumeDownload(taskId)"
                >
                  <el-icon class="model-ver-btn__icon"><video-play /></el-icon>
                  <span class="model-ver-btn__label">{{ $t('download.resume') }}</span>
                </el-button>
                <el-button
                  v-else-if="item.status === 'running'"
                  class="model-ver-btn model-ver-btn--neutral"
                  size="small"
                  plain
                  @click="cancelDownload(taskId)"
                >
                  <el-icon class="model-ver-btn__icon"><close /></el-icon>
                  <span class="model-ver-btn__label">{{ $t('download.cancelDownload') }}</span>
                </el-button>
                <el-button
                  v-else-if="item.status === 'failed'"
                  class="model-ver-btn model-ver-btn--delete"
                  size="small"
                  plain
                  @click="deleteDownload(taskId)"
                >
                  <el-icon class="model-ver-btn__icon"><delete /></el-icon>
                  <span class="model-ver-btn__label">{{ $t('download.deleteTask') }}</span>
                </el-button>
              </div>
            </div>
            <el-progress
              :percentage="
                item.total_size > 0 ? Math.round(item.progress * 100) : 0
              "
              :status="item.status === 'failed' ? 'exception' : ''"
              :stroke-width="8"
              :show-text="item.total_size > 0"
            />
            <div
              v-if="item.error"
              class="models-download-error"
            >
              {{ item.error }}
            </div>
          </div>
        </el-card>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue';
import { onBeforeRouteLeave } from 'vue-router';
import { ElMessage, ElMessageBox } from 'element-plus';
import { api } from '@/utils/api';
import { $tt, $mn, $md } from '@/utils/i18n';
import { useRegistryStore } from '@/stores/registry';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';

/* ───── Types ───── */

interface ModelVersion {
  name: string;
  size?: string;
  source_type?: string;
  from_version?: string;
}

interface ModelConfig {
  name?: string | { zh?: string; en?: string };
  name_en?: string;
  description?: string | { zh?: string; en?: string };
  description_en?: string;
  category?: string;
  size?: string;
  source?: string;
  base_model?: string;
  recommended?: boolean;
  commercial_use_allowed?: boolean | null;
  dependencies?: string[];
  versions?: Record<string, ModelVersion>;
  ready?: boolean;
}

interface ModelRow extends ModelConfig {
  id: string;
  ready: boolean;
}

interface DownloadItem {
  name: string;
  progress: number;
  status: string;
  speed: string;
  error: string;
  total_size: number;
  downloaded_size: number;
  kind?: string;
}

interface DiskInfo {
  size_human?: string;
  free_human?: string;
  exists?: boolean;
  total?: number;
  size?: number;
}

interface DiskSpaceData {
  models?: DiskInfo;
  loras?: DiskInfo;
}

/* ───── State ───── */

const activeCategory = ref('all');
const modelRegistry = ref<Record<string, ModelConfig>>({});
const modelsStatus = ref<Record<string, boolean>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});
const categories = ref<Record<string, any>>({});
const filterQuery = ref('');
const refreshing = ref(false);

const registryStore = useRegistryStore();

const downloadingModels = ref<Record<string, boolean>>({});
const downloadingLoras = ref<Record<string, boolean>>({});
const activeDownloads = ref<Record<string, DownloadItem>>({});
const selectedVersions = ref<Record<string, string>>({});
const sseConnections = ref<Record<string, EventSource>>({});

const searchQuery = ref('');
const searchType = ref('LORA');
const searching = ref(false);
const searchResults = ref<any[]>([]);
const hasSearched = ref(false);
/** CivitAI 缩略图加载失败时切到占位，避免对 img 写内联 style */
const civitaiPreviewLoadFailed = ref<Record<string, boolean>>({});

const installedModels = ref<any[]>([]);
const diskSpace = ref<DiskSpaceData | null>(null);

const importDialogVisible = ref(false);
const importModelName = ref('');
const importModelPath = ref('');
const importModelType = ref('base');
const importing = ref(false);

/* ───── Helpers ───── */

/** Same Element Plus icons as left sidebar `el-menu-item`. */
const categoryPageIcon = computed(() => {
  const icons: Record<string, string> = {
    all: 'Grid',
    image_models: 'PictureFilled',
    video_models: 'VideoCamera',
    music_models: 'Headset',
    controlnets: 'Aim',
    upscalers: 'ZoomIn',
    tools: 'Tools',
    loras: 'MagicStick',
  };
  return icons[activeCategory.value] ?? null;
});

const categoryTitleText = computed(() => {
  const titles: Record<string, string> = {
    all: $tt('download.allModels'),
    image_models: $tt('download.imageModels'),
    video_models: $tt('download.videoModels'),
    music_models: $tt('download.audioModels'),
    controlnets: $tt('download.controlNet'),
    upscalers: $tt('download.upscalers'),
    tools: $tt('download.tools'),
    loras: $tt('download.loraModels'),
  };
  return titles[activeCategory.value] || $tt('download.title');
});

function modelSearchBlob(m: ModelRow): string {
  const n = $mn(m, m.id);
  const d = $md(m, '');
  return `${n} ${d}`.toLowerCase();
}

function isCommercialUseAllowed(m: ModelRow): boolean {
  return m.commercial_use_allowed === true;
}

const filteredModels = computed(() => {
  const list: ModelRow[] = [];
  for (const [id, config] of Object.entries(modelRegistry.value)) {
    const model: ModelRow = {
      id,
      ...config,
      ready: modelsStatus.value[id] || false,
    };

    if (activeCategory.value !== 'all') {
      const regCat = model.category;
      if (activeCategory.value === 'image_models') {
        if (regCat !== 'base_models') continue;
      } else if (activeCategory.value !== regCat) {
        continue;
      }
    }

    if (filterQuery.value) {
      const query = filterQuery.value.toLowerCase();
      if (!modelSearchBlob(model).includes(query)) {
        continue;
      }
    }

    list.push(model);
  }

  return list.sort((a, b) => {
    if (a.recommended !== b.recommended) return a.recommended ? -1 : 1;
    return $mn(a, a.id).localeCompare($mn(b, b.id));
  });
});

const totalModelCount = computed(() => Object.keys(modelRegistry.value).length);

const activeDownloadCount = computed(() => {
  return Object.values(activeDownloads.value).filter(
    (item) =>
      item.status === 'running' ||
      item.status === 'paused' ||
      item.status === 'failed'
  ).length;
});

/* ───── Loaders ───── */

async function loadModelRegistry() {
  try {
    const [registryData, statusData, detailedStatusData] = await Promise.all([
      registryStore.load(),
      api.settings.getModelsStatus(),
      api.settings.getModelsDetailedStatus(),
    ]);
    const reg = registryData || { models: {}, categories: {} };
    modelRegistry.value = reg.models || {};
    modelsStatus.value = statusData || {};
    modelsDetailedStatus.value = detailedStatusData || {};
    categories.value = reg.categories || {};
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
}

async function refreshStatus() {
  refreshing.value = true;
  try {
    const [statusData, detailedStatusData] = await Promise.all([
      api.settings.getModelsStatus(),
      api.settings.getModelsDetailedStatus(),
    ]);
    modelsStatus.value = statusData || {};
    modelsDetailedStatus.value = detailedStatusData || {};
    await loadInstalled();
    await loadDiskSpace();
  } catch (e) {
    console.error('Refresh failed:', e);
  } finally {
    refreshing.value = false;
  }
}

async function loadInstalled() {
  try {
    const models = await api.settings.listModels();
    installedModels.value = (models as any[]) || [];
  } catch (e) {
    console.error('Failed to load installed:', e);
  }
}

async function loadDiskSpace() {
  try {
    const data = await api.settings.getDiskSpace();
    diskSpace.value = data as DiskSpaceData;
  } catch (e) {
    console.error('Failed to load disk space:', e);
  }
}

async function loadActiveDownloads() {
  try {
    const tasks = await api.download.listDownloads();
    if (!Array.isArray(tasks)) return;
    for (const task of tasks as any[]) {
      if (
        task.status === 'running' ||
        task.status === 'pending' ||
        task.status === 'paused' ||
        task.status === 'failed'
      ) {
        activeDownloads.value[task.id] = {
          name: task.filename || task.url,
          progress: task.progress || 0,
          status: task.status,
          speed: '',
          error: task.error_message || '',
          total_size: task.total_size || 0,
          downloaded_size: task.downloaded_size || 0,
        };
        if (task.status === 'running') {
          connectProgressSSE(task.id, task.filename || task.url);
        }
      }
    }
  } catch (e) {
    console.error('Failed to load active downloads:', e);
  }
}

/* ───── Utilities ───── */

function getDiskPercent(type: 'models' | 'loras'): number {
  if (!diskSpace.value || !diskSpace.value[type]) return 0;
  const info = diskSpace.value[type];
  if (!info || !info.exists || info.total === 0) return 0;
  return Math.round((info.size! / info.total) * 100);
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function isModelReady(modelId: string): boolean {
  return modelsStatus.value[modelId] || false;
}

function getModelName(modelId: string): string {
  const cfg = modelRegistry.value[modelId];
  if (!cfg) return modelId;
  const row: ModelRow = { id: modelId, ...cfg, ready: false };
  return $mn(row, modelId);
}

function canDownload(model: ModelRow): boolean {
  if (!model.dependencies || model.dependencies.length === 0) return true;
  return model.dependencies.every((dep) => isModelReady(dep));
}

function getDependencyHint(model: ModelRow): string {
  if (!model.dependencies || model.dependencies.length === 0) return '';
  const missing = model.dependencies.filter((dep) => !isModelReady(dep));
  if (missing.length === 0) return '';
  const names = missing.map((d) => getModelName(d)).join('、');
  return $tt('download.dependencyMissing', { models: names });
}

function getVersionStatus(modelId: string, versionKey: string): string {
  const detail = modelsDetailedStatus.value[modelId];
  if (!detail || !detail.versions) return 'missing';

  const verStatus = detail.versions[versionKey];
  if (!verStatus) return 'missing';

  if (verStatus.ready) return 'ready';

  const model = modelRegistry.value[modelId];
  const ver = model?.versions?.[versionKey];
  if (ver && ver.source_type === 'derived' && ver.from_version) {
    const parentVer = ver.from_version;
    const parentSt = parentVer ? detail.versions[parentVer] : null;
    if (!parentSt || !parentSt.ready) {
      return 'parent_missing';
    }
    return 'quantize';
  }

  return 'missing';
}

/** 供版本列表行展示（iOS 分组列表 + 操作区） */
function modelVersionTableRows(model: ModelRow) {
  if (!model.versions) return [];
  return Object.entries(model.versions).map(([verKey, ver]) => ({
    verKey,
    ver: ver as ModelVersion,
    model,
    vstatus: getVersionStatus(model.id, verKey),
  }));
}

/* ───── Download / Quantize / Delete ───── */

async function downloadVersion(
  model: ModelRow,
  versionKey: string,
  opts: { uiLoadingKey?: string } = {}
) {
  const uiKey =
    opts.uiLoadingKey != null ? opts.uiLoadingKey : `${model.id}-${versionKey}`;
  if (downloadingModels.value[uiKey]) return;

  downloadingModels.value[uiKey] = true;

  try {
    const version = model.versions?.[versionKey];
    const data = (await api.models.install(model.id, { version: versionKey })) as any;
    const label = version
      ? `${$mn(model, model.id)} ${version.name}`
      : `${$mn(model, model.id)} ${versionKey}`;
    connectProgressSSE(data.task_id, label, uiKey);
  } catch (e: any) {
    console.error('Download failed:', e);
    ElMessage.error($tt('download.downloadFailed', { msg: e.message }));
    delete downloadingModels.value[uiKey];
  }
}

async function quantizeVersion(model: ModelRow, versionKey: string) {
  const ver = model.versions?.[versionKey];
  const fromV = ver?.from_version;
  if (!fromV) {
    ElMessage.error($tt('download.versionConfigError'));
    return;
  }
  const uiKey = `${model.id}-${versionKey}`;
  if (downloadingModels.value[uiKey]) return;

  downloadingModels.value[uiKey] = true;
  try {
    const data = (await api.download.startConvert({
      model_name: model.id,
      from_version: fromV,
      to_version: versionKey,
    })) as any;
    const taskId = data.task_id;
    const label = ver?.name
      ? `${$mn(model, model.id)} ${ver.name}`
      : `${$mn(model, model.id)} ${versionKey}`;
    connectConversionSSE(taskId, label, uiKey);
  } catch (e: any) {
    console.error('Quantize failed:', e);
    ElMessage.error($tt('download.quantizeFailed', { msg: e.message || String(e) }));
    delete downloadingModels.value[uiKey];
  }
}

async function deleteVersion(model: ModelRow, versionKey: string) {
  try {
    const version = model.versions?.[versionKey];
    await ElMessageBox.confirm(
      $tt('download.deleteConfirm', {
        name: `${$mn(model, model.id)} ${version?.name || versionKey}`,
      }),
      $tt('download.deleteConfirmTitle'),
      {
        confirmButtonText: $tt('download.deleteConfirmBtn'),
        cancelButtonText: $tt('download.deleteCancelBtn'),
        type: 'warning',
      }
    );
    const result = (await api.models.deleteVersion(model.id, versionKey)) as any;
    if (result.success) {
      ElMessage.success(
        $tt('download.deletedMsg', {
          name: `${$mn(model, model.id)} ${version?.name || versionKey}`,
        })
      );
    } else {
      ElMessage.error(result.error || $tt('download.deleteFailed'));
    }
    refreshStatus();
  } catch (e: any) {
    if (e !== 'cancel') {
      console.error('Delete failed:', e);
      ElMessage.error($tt('download.deleteFailed') + ': ' + (e.message || e));
    }
  }
}

/* ───── SSE connections ───── */

function connectProgressSSE(
  taskId: string,
  name: string,
  downloadingKey: string | null = null
) {
  const dmKey = downloadingKey != null ? downloadingKey : taskId;
  if (sseConnections.value[taskId]) {
    sseConnections.value[taskId].close();
  }

  const eventSource = new EventSource(
    api.download.installProgressStreamUrl(taskId)
  );
  sseConnections.value[taskId] = eventSource;

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      activeDownloads.value[taskId] = {
        name,
        progress: data.progress || 0,
        status: data.status,
        speed: data.speed || '',
        error: data.error_message || '',
        total_size: data.total_size,
        downloaded_size: data.downloaded_size,
      };

      if (data.status === 'completed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        setTimeout(() => {
          delete activeDownloads.value[taskId];
          delete downloadingModels.value[dmKey];
        }, 2000);
        ElMessage.success($tt('download.downloadComplete', { name }));
        refreshStatus();
      } else if (data.status === 'failed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete downloadingModels.value[dmKey];
        ElMessage.error(
          $tt('download.downloadFailed', { name, msg: data.error_message })
        );
      } else if (data.status === 'cancelled') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete activeDownloads.value[taskId];
        delete downloadingModels.value[dmKey];
        ElMessage.info($tt('download.downloadCancelled', { name }));
      }
    } catch (e) {
      console.error('SSE parse error:', e);
    }
  };

  eventSource.onerror = (error) => {
    console.error('SSE error:', error);
    eventSource.close();
    delete sseConnections.value[taskId];
    delete downloadingModels.value[dmKey];
  };
}

function connectConversionSSE(
  taskId: string,
  name: string,
  dmKey: string | null = null
) {
  const rowKey = dmKey != null ? dmKey : taskId;
  if (sseConnections.value[taskId]) {
    try {
      sseConnections.value[taskId].close();
    } catch (e) {}
  }

  const eventSource = new EventSource(
    api.download.convertProgressStreamUrl(taskId)
  );
  sseConnections.value[taskId] = eventSource;

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      activeDownloads.value[taskId] = {
        kind: 'quantize',
        name,
        progress: typeof data.progress === 'number' ? data.progress : 0,
        status: data.status || 'running',
        speed: '',
        total_size: 1,
        downloaded_size: 0,
        error: data.error_message || '',
      };

      if (data.status === 'completed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        setTimeout(() => {
          delete activeDownloads.value[taskId];
          delete downloadingModels.value[rowKey];
        }, 2000);
        ElMessage.success($tt('download.quantizeComplete', { name }));
        refreshStatus();
      } else if (data.status === 'failed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete downloadingModels.value[rowKey];
        ElMessage.error(
          $tt('download.quantizeFailed', { msg: data.error_message || '' })
        );
      } else if (data.status === 'cancelled') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete activeDownloads.value[taskId];
        delete downloadingModels.value[rowKey];
        ElMessage.info($tt('download.genCancelled', { name }));
      }
    } catch (e) {
      console.error('Conversion SSE parse error:', e);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    delete sseConnections.value[taskId];
    delete downloadingModels.value[rowKey];
  };
}

/* ───── Batch / Cancel / Resume ───── */

async function cancelDownload(taskId: string) {
  const item = activeDownloads.value[taskId];
  if (item && item.kind === 'quantize') {
    try {
      await api.download.cancelConversion(taskId);
    } catch (e) {
      console.error('Cancel conversion failed:', e);
    }
    return;
  }
  try {
    await api.download.cancel(taskId);
  } catch (e) {
    console.error('Cancel failed:', e);
  }
}

async function resumeDownload(taskId: string) {
  try {
    const item = activeDownloads.value[taskId];
    if (!item) return;

    await api.download.resume(taskId);

    item.status = 'running';
    connectProgressSSE(taskId, item.name);

    ElMessage.success($tt('download.resumeStart', { name: item.name }));
  } catch (e: any) {
    console.error('Resume failed:', e);
    ElMessage.error($tt('download.resumeFailed', { msg: e.message }));
  }
}

async function deleteDownload(taskId: string) {
  try {
    const item = activeDownloads.value[taskId];
    if (!item || item.kind !== 'quantize') {
      await api.download.delete(taskId);
    }
    delete activeDownloads.value[taskId];
    ElMessage.success($tt('download.deleteSuccess'));
  } catch (e) {
    console.error('Delete download failed:', e);
    ElMessage.error($tt('download.deleteFailed'));
  }
}

/* ───── CivitAI ───── */

function getCivitaiPreviewUrl(model: any): string {
  return model?.model_versions?.[0]?.images?.[0]?.url || '';
}

function onCivitaiPreviewError(modelId: string | number) {
  const key = String(modelId);
  civitaiPreviewLoadFailed.value = {
    ...civitaiPreviewLoadFailed.value,
    [key]: true,
  };
}

async function searchCivitai() {
  if (searching.value) return;
  searching.value = true;
  hasSearched.value = true;
  civitaiPreviewLoadFailed.value = {};

  try {
    const data = await api.download.civitaiSearch({
      q: searchQuery.value,
      types: searchType.value,
      limit: '20',
    });
    const models = Array.isArray(data) ? data : (data as any).items || [];
    searchResults.value = models;

    models.forEach((model: any) => {
      if (model.model_versions.length > 0 && !selectedVersions.value[model.id]) {
        selectedVersions.value[model.id] = model.model_versions[0].id;
      }
    });
  } catch (e) {
    console.error('Search failed:', e);
    ElMessage.error($tt('download.searchFailed'));
  } finally {
    searching.value = false;
  }
}

async function downloadCivitaiModel(model: any) {
  const versionId = selectedVersions.value[model.id];
  if (!versionId) {
    ElMessage.warning($tt('download.selectVersionWarn'));
    return;
  }

  const version = model.model_versions.find((v: any) => v.id === versionId);
  if (!version || !version.files.length) {
    ElMessage.error($tt('download.noDownloadableFile'));
    return;
  }

  const primaryFile = version.files.find((f: any) => f.primary) || version.files[0];
  downloadingLoras.value[model.id] = true;

  try {
    const data = (await api.download.startLoraDownload(
      primaryFile.download_url,
      primaryFile.name
    )) as any;
    connectProgressSSE(data.task_id, model.name);
  } catch (e: any) {
    console.error('Download failed:', e);
    ElMessage.error($tt('download.downloadFailed', { msg: e.message }));
  } finally {
    downloadingLoras.value[model.id] = false;
  }
}

/* ───── Format / Helpers ───── */

function formatNumber(num: number): string {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
  return num.toString();
}

function showImportDialog() {
  importModelName.value = '';
  importModelPath.value = '';
  importModelType.value = 'base';
  importDialogVisible.value = true;
}

async function importLocalModel() {
  if (!importModelName.value || !importModelPath.value) {
    ElMessage.warning($tt('download.importWarn'));
    return;
  }

  importing.value = true;
  try {
    let importedModels: unknown[] = [];
    try {
      const raw = getItem(DQ_STORAGE.IMPORTED_MODELS);
      importedModels = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(importedModels)) importedModels = [];
    } catch {
      importedModels = [];
    }
    importedModels.push({
      name: importModelName.value,
      path: importModelPath.value,
      type: importModelType.value,
      importedAt: new Date().toISOString(),
    });
    setItem(DQ_STORAGE.IMPORTED_MODELS, JSON.stringify(importedModels));

    ElMessage.success(
      $tt('download.importSuccess', { name: importModelName.value })
    );
    importDialogVisible.value = false;
  } catch (e) {
    console.error('Import failed:', e);
    ElMessage.error($tt('download.importFailed'));
  } finally {
    importing.value = false;
  }
}

function getModelInitials(model: ModelRow): string {
  const name = $mn(model, model.id);
  if (!name) return 'M';
  const dashIndex = name.indexOf('-');
  const spaceIndex = name.indexOf(' ');
  let endIndex = -1;
  if (dashIndex !== -1 && spaceIndex !== -1) {
    endIndex = Math.min(dashIndex, spaceIndex);
  } else if (dashIndex !== -1) {
    endIndex = dashIndex;
  } else if (spaceIndex !== -1) {
    endIndex = spaceIndex;
  }
  if (endIndex !== -1) {
    return name.slice(0, endIndex);
  }
  return name.slice(0, 3);
}

function getModelTypeTagType(type: string): string {
  const typeMap: Record<string, string> = {
    diffusion: 'primary',
    controlnet: 'warning',
    upscaler: 'success',
    tool: 'info',
    lora: 'danger',
    video: 'success',
  };
  return typeMap[type] || 'info';
}

function handleCategorySelect(index: string) {
  activeCategory.value = index;
}

function closeAllSSE() {
  Object.values(sseConnections.value).forEach((es) => {
    try {
      es.close();
    } catch (e) {}
  });
  sseConnections.value = {};
}

onBeforeRouteLeave(() => {
  closeAllSSE();
});

/* ───── Lifecycle ───── */

onMounted(() => {
  loadModelRegistry();
  loadInstalled();
  loadDiskSpace();
  loadActiveDownloads();
});

onUnmounted(() => {
  closeAllSSE();
});
</script>
