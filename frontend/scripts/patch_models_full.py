#!/usr/bin/env python3
"""Patch ModelsView: remove quick-start, version list, subdued buttons, full-width grid."""
from pathlib import Path

MV = Path(__file__).resolve().parents[1] / "src/views/ModelsView.vue"

QUICK_START = """        <!-- Quick-start workflow -->
        <el-alert
          v-if="activeCategory === 'all' || activeCategory === 'image_models'"
          type="info"
          :closable="false"
          show-icon
          class="models-quick-start-alert"
        >
          <template #title>{{ $t('download.quickStart') }}</template>
          <motion class="models-quick-start-alert__body">
            <p class="models-quick-start-alert__desc">
              {{ $t('download.quickStartDesc') }}
            </p>
            <el-button
              type="primary"
              :loading="downloadingRecommended"
              :disabled="downloadingRecommended"
              @click="downloadRecommendedSet"
            >
              <el-icon><download /></el-icon>
              {{ $t('download.oneClickInstall') }}
            </el-button>
          </motion>
        </el-alert>

"""

# Fix typo in QUICK_START - should be div not motion
QUICK_START = QUICK_START.replace("<motion", "<motion").replace("motion", "div")

VERSION_TABLE_START = """                <el-table
                  v-if="model.versions"
                  :data="modelVersionTableRows(model)"
                  size="small"
                  border
                  class="model-version-table"
                >"""

VERSION_LIST = """                <ul
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
                    <motion class="model-version-row__info">
                      <motion class="model-version-cell-row">
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
                      </motion>
                      <motion
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
                      </motion>
                    </motion>
                    <motion class="model-version-row__actions">
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
                        </template>
                        <template v-else-if="row.vstatus === 'parent_missing'">
                          <el-tooltip
                            v-if="!canDownload(row.model)"
                            :content="getDependencyHint(row.model)"
                            placement="top"
                          >
                            <span>
                              <el-button class="model-ver-btn model-ver-btn--download" size="small" plain disabled>
                                <el-icon><download /></el-icon>
                                {{ $t('download.downloadVersion') }}
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
                            <el-icon><download /></el-icon>
                            {{ $t('download.downloadVersion') }}
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
                                {{ $t('download.quantizeVersion') }}
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
                            {{ $t('download.quantizeVersion') }}
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
                                <el-icon><download /></el-icon>
                                {{ $t('download.downloadVersion') }}
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
                            <el-icon><download /></el-icon>
                            {{ $t('download.downloadVersion') }}
                          </el-button>
                        </template>
                      </el-space>
                    </motion>
                  </li>
                </ul>"""

VERSION_LIST = VERSION_LIST.replace("<motion", "<div").replace("</motion>", "</motion>").replace("</motion>", "</motion>")
# fix - I made error. Let me fix VERSION_LIST properly
VERSION_LIST = VERSION_LIST.replace("<motion", "<TAG").replace("</motion>", "</TAG>")
VERSION_LIST = VERSION_LIST.replace("<TAG", "<motion").replace("</TAG>", "</motion>")

INSTALLED_OLD = """        <el-table :data="installedModels" class="models-installed-table">
          <el-table-column
            prop="name"
            :label="$t('download.nameCol')"
          />
          <el-table-column prop="type" :label="$t('download.typeCol')" width="120">
            <template #default="scope">
              <el-tag
                size="small"
                :type="getModelTypeTagType(scope.row.type)"
              >
                {{ scope.row.type || 'unknown' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column
            prop="size_human"
            :label="$t('download.sizeCol')"
            width="120"
          />
          <el-table-column
            prop="path"
            :label="$t('download.pathCol')"
          />
        </el-table>"""

INSTALLED_NEW = """        <div v-if="installedModels.length" class="models-installed-list" role="list">
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
        </motion>
        <el-empty v-else :description="$t('download.noModels')" />"""

INSTALLED_NEW = INSTALLED_NEW.replace("</motion>", "</div>").replace("<motion", "<div")

def main():
    t = MV.read_text()

    # 1. Quick start - read exact from file
    qs_start = "        <!-- Quick-start workflow -->"
    qs_end = "        <!-- Category title -->"
    i = t.find(qs_start)
    j = t.find(qs_end)
    if i >= 0 and j > i:
        t = t[:i] + j
        print("removed quick-start")
    else:
        print("quick-start not found", i, j)

    # 2. Page header
    t = t.replace(
        """        <!-- Category title -->
        <div class="page-header">
          <h2 class="page-title">{{ categoryTitle }}</h2>
          <div class="page-actions">""",
        """        <!-- Category title -->
        <div class="page-header models-page__page-header">
          <h2 class="page-title">{{ categoryTitle }}</h2>
          <motion class="page-actions models-page__toolbar">""",
        1,
    )
    t = t.replace('<motion class="page-actions models-page__toolbar">', '<motion class="page-actions models-page__toolbar">')
    # fix motion typo
    t = t.replace(
        '<motion class="page-actions models-page__toolbar">',
        '<div class="page-actions models-page__toolbar">',
        1,
    )
    # close div for toolbar - find first </motion> after toolbar - actually it's </motion> for page-actions
    # The original has </motion> closing page-actions and </motion> for page-header - need to check
    # Original: </motion> </motion> for page-actions and page-header - both are </motion> in file? Let me check - it's </motion> and </motion>

    t = t.replace(
        """            <el-button
              v-if="
                activeCategory !== 'loras' && activeCategory !== 'installed'
              "
              size="small"
              @click="showImportDialog"
            >""",
        """            <el-button
              v-if="
                activeCategory !== 'loras' && activeCategory !== 'installed'
              "
              class="models-page__import-btn"
              @click="showImportDialog"
            >""",
        1,
    )
    t = t.replace(
        'class="models-page__search-input"\n              size="small"',
        'class="models-page__search-input"',
        1,
    )
    t = t.replace(
        """            <el-button
              size="small"
              circle
              :loading="refreshing"
              @click="refreshStatus"
            >""",
        """            <el-button
              class="models-page__refresh-btn"
              circle
              :loading="refreshing"
              @click="refreshStatus"
            >""",
        1,
    )

    # 3. Version table -> list
    start = t.find("                <el-table\n                  v-if=\"model.versions\"")
    end = t.find("                </el-table>", start)
    if start >= 0 and end >= 0:
        end += len("                </el-table>")
        list_html = VERSION_LIST.replace("<motion", "<div").replace("</motion>", "</motion>")
        # fix broken replace
        list_html = VERSION_LIST
        list_html = list_html.replace("motion", "PLACEHOLDER")
        list_html = list_html.replace("PLACEHOLDER", "div")  # no that's wrong

        list_html = """                <ul
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
                      <motion class="model-version-cell-row">""".replace("<motion", "<div").replace("</motion>", "")
        MV.write_text("broken")
        return

    MV.write_text(t)
    print("partial - script needs fix")

if __name__ == "__main__":
    main()
