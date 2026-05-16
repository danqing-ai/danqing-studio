#!/usr/bin/env python3
from pathlib import Path

MV = Path(__file__).resolve().parents[1] / "src/views/ModelsView.vue"
t = MV.read_text()

OLD_ACTIONS = """                    <motion class="model-version-row__actions">
                      <el-space wrap :size="6" justify="end">
                        <template v-if="row.vstatus === 'ready'">
                          <el-button
                            type="warning"
                            size="small"
                            plain
                            class="model-ver-btn"
                            @click="downloadVersion(row.model, row.verKey)"
                          >
                            <el-icon><download /></el-icon>
                            {{ $t('download.forceDownload') }}
                          </el-button>
                          <el-button
                            type="danger"
                            size="small"
                            plain
                            class="model-ver-btn"
                            @click="deleteVersion(row.model, row.verKey)"
                          >
                            <el-icon><delete /></el-icon>
                          </el-button>
                        </template>"""

OLD_ACTIONS = OLD_ACTIONS.replace("<motion", "<motion").replace("motion", "motion")
OLD_ACTIONS = """                    <div class="model-version-row__actions">
                      <el-space wrap :size="6" justify="end">
                        <template v-if="row.vstatus === 'ready'">
                          <el-button
                            type="warning"
                            size="small"
                            plain
                            class="model-ver-btn"
                            @click="downloadVersion(row.model, row.verKey)"
                          >
                            <el-icon><download /></el-icon>
                            {{ $t('download.forceDownload') }}
                          </el-button>
                          <el-button
                            type="danger"
                            size="small"
                            plain
                            class="model-ver-btn"
                            @click="deleteVersion(row.model, row.verKey)"
                          >
                            <el-icon><delete /></el-icon>
                          </el-button>
                        </template>"""

NEW_READY = """                    <div class="model-version-row__actions">
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
                        </template>"""

if OLD_ACTIONS in t:
    t = t.replace(OLD_ACTIONS, NEW_READY, 1)
    print("ready block ok")
else:
    print("ready block NOT FOUND")

# quantize buttons - add icon
t = t.replace(
    """                              <el-button class="model-ver-btn model-ver-btn--quantize" size="small" plain disabled>
                                {{ $t('download.quantizeVersion') }}
                              </el-button>""",
    """                              <el-button class="model-ver-btn model-ver-btn--quantize" size="small" plain disabled>
                                <el-icon class="model-ver-btn__icon"><cpu /></el-icon>
                                <span class="model-ver-btn__label">{{ $t('download.quantizeVersion') }}</span>
                              </el-button>""",
)
t = t.replace(
    """                          <el-button
                            v-else
                            class="model-ver-btn model-ver-btn--quantize"
                            size="small"
                            plain
                            :loading="downloadingModels[row.model.id + '-' + row.verKey]"
                            @click="quantizeVersion(row.model, row.verKey)"
                          >
                            {{ $t('download.quantizeVersion') }}
                          </el-button>""",
    """                          <el-button
                            v-else
                            class="model-ver-btn model-ver-btn--quantize"
                            size="small"
                            plain
                            :loading="downloadingModels[row.model.id + '-' + row.verKey]"
                            @click="quantizeVersion(row.model, row.verKey)"
                          >
                            <el-icon class="model-ver-btn__icon"><cpu /></el-icon>
                            <span class="model-ver-btn__label">{{ $t('download.quantizeVersion') }}</span>
                          </el-button>""",
)
print("quantize icons ok")

# download buttons - wrap label in span
for old, new in [
    (
        """                              <el-button class="model-ver-btn model-ver-btn--download" size="small" plain disabled>
                                <el-icon><download /></el-icon>
                                {{ $t('download.downloadVersion') }}
                              </el-button>""",
        """                              <el-button class="model-ver-btn model-ver-btn--download" size="small" plain disabled>
                                <el-icon class="model-ver-btn__icon"><download /></el-icon>
                                <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
                              </el-button>""",
    ),
    (
        """                          <el-button
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
                            <el-icon><download /></el-icon>
                            {{ $t('download.downloadVersion') }}
                          </el-button>""",
        """                          <el-button
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
                          </el-button>""",
    ),
    (
        """                          <el-button
                            v-else
                            class="model-ver-btn model-ver-btn--download"
                            size="small"
                            plain
                            :loading="downloadingModels[row.model.id + '-' + row.verKey]"
                            @click="downloadVersion(row.model, row.verKey)"
                          >
                            <el-icon><download /></el-icon>
                            {{ $t('download.downloadVersion') }}
                          </el-button>""",
        """                          <el-button
                            v-else
                            class="model-ver-btn model-ver-btn--download"
                            size="small"
                            plain
                            :loading="downloadingModels[row.model.id + '-' + row.verKey]"
                            @click="downloadVersion(row.model, row.verKey)"
                          >
                            <el-icon class="model-ver-btn__icon"><download /></el-icon>
                            <span class="model-ver-btn__label">{{ $t('download.downloadVersion') }}</span>
                          </el-button>""",
    ),
]:
    if old in t:
        t = t.replace(old, new)
    else:
        print("download pattern miss")

# header import + refresh
t = t.replace(
    """            <el-button
              v-if="
                activeCategory !== 'loras' && activeCategory !== 'installed'
              "
              class="models-page__import-btn"
              @click="showImportDialog"
            >
              <el-icon><upload /></el-icon>
              {{ $t('download.importLocal') }}
            </el-button>""",
    """            <el-button
              v-if="
                activeCategory !== 'loras' && activeCategory !== 'installed'
              "
              class="models-toolbar-btn models-page__import-btn"
              plain
              @click="showImportDialog"
            >
              <el-icon class="models-toolbar-btn__icon"><upload /></el-icon>
              <span class="models-toolbar-btn__label">{{ $t('download.importLocal') }}</span>
            </el-button>""",
    1,
)

t = t.replace(
    """            <el-button
              class="models-page__refresh-btn"
              circle
              :loading="refreshing"
              @click="refreshStatus"
            >
              <el-icon><refresh /></el-icon>
            </el-button>""",
    """            <el-button
              class="models-toolbar-btn models-page__refresh-btn"
              plain
              :loading="refreshing"
              @click="refreshStatus"
            >
              <el-icon class="models-toolbar-btn__icon"><refresh /></el-icon>
              <span class="models-toolbar-btn__label">{{ $t('common.refresh') }}</span>
            </el-button>""",
    1,
)
print("toolbar ok")

# civitai download
t = t.replace(
    """                  <el-button
                    type="primary"
                    size="small"
                    :loading="downloadingLoras[model.id]"
                    @click="downloadCivitaiModel(model)"
                  >
                    {{ $t('download.download_') }}
                  </el-button>""",
    """                  <el-button
                    class="model-ver-btn model-ver-btn--download"
                    size="small"
                    plain
                    :loading="downloadingLoras[model.id]"
                    @click="downloadCivitaiModel(model)"
                  >
                    <el-icon class="model-ver-btn__icon"><download /></el-icon>
                    <span class="model-ver-btn__label">{{ $t('download.download_') }}</span>
                  </el-button>""",
    1,
)

# civitai search button
t = t.replace(
    """              <el-button
                type="primary"
                :loading="searching"
                @click="searchCivitai"
              >
                <el-icon><search /></el-icon>
                {{ $t('download.search') }}
              </el-button>""",
    """              <el-button
                class="models-toolbar-btn models-toolbar-btn--primary"
                :loading="searching"
                @click="searchCivitai"
              >
                <el-icon class="models-toolbar-btn__icon"><search /></el-icon>
                <span class="models-toolbar-btn__label">{{ $t('download.search') }}</span>
              </el-button>""",
    1,
)

# download task buttons
t = t.replace(
    """                <el-button
                  v-if="item.status === 'paused'"
                  type="primary"
                  size="small"
                  @click="resumeDownload(taskId)"
                >
                  {{ $t('download.resume') }}
                </el-button>
                <el-button
                  v-else-if="item.status === 'running'"
                  size="small"
                  @click="cancelDownload(taskId)"
                >
                  {{ $t('download.cancelDownload') }}
                </el-button>
                <el-button
                  v-else-if="item.status === 'failed'"
                  type="danger"
                  size="small"
                  @click="deleteDownload(taskId)"
                >
                  <el-icon><delete /></el-icon>
                  {{ $t('download.deleteTask') }}
                </el-button>""",
    """                <el-button
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
                </el-button>""",
    1,
)

MV.write_text(t)
print("saved")
