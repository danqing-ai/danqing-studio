<template>
  <div class="lora-dataset-panel">
    <!-- Left: persistent dataset library (no whole-dataset delete) -->
    <aside class="lora-dataset-panel__list">
      <div class="lora-dataset-panel__list-head">
        <span class="lora-dataset-panel__list-title">{{ $t('loraTrain.datasetLibrary') }}</span>
        <DqButton size="xs" type="primary" @click="openCreateDialog">
          {{ $t('loraTrain.newDataset') }}
        </DqButton>
      </div>

      <p class="lora-dataset-panel__persist-note">{{ $t('loraTrain.datasetPersistNote') }}</p>

      <DqInput
        v-if="datasets.length > 3"
        v-model="datasetSearch"
        size="xs"
        class="lora-dataset-panel__search"
        :placeholder="$t('loraTrain.searchDatasets')"
        clearable
      />

      <div v-if="filteredDatasets.length" class="lora-dataset-panel__list-scroll">
        <button
          v-for="d in filteredDatasets"
          :key="d.id"
          type="button"
          class="lora-dataset-panel__list-item"
          :class="{ 'is-active': d.id === selectedId, 'is-ready': (d.image_count || 0) >= minImages }"
          @click="selectDataset(d.id)"
        >
          <img
            v-if="datasetCoverUrl(d)"
            :src="datasetCoverUrl(d)"
            alt=""
            class="lora-dataset-panel__list-thumb"
            loading="lazy"
          />
          <span v-else class="lora-dataset-panel__list-thumb lora-dataset-panel__list-thumb--empty" aria-hidden="true">
            <DqIcon :size="16"><PictureFilled /></DqIcon>
          </span>
          <span class="lora-dataset-panel__list-body">
            <span class="lora-dataset-panel__list-name">{{ d.name }}</span>
            <span class="lora-dataset-panel__list-meta">
              {{ $t('loraTrain.imageCount', { count: d.image_count || 0 }) }}
              <DqTag
                v-if="(d.image_count || 0) >= minImages"
                size="small"
                type="success"
                effect="plain"
                class="lora-dataset-panel__ready-tag"
              >
                {{ $t('loraTrain.datasetReady') }}
              </DqTag>
            </span>
          </span>
        </button>
      </div>
      <DqEmpty
        v-else-if="datasets.length"
        :description="$t('loraTrain.noSearchResults')"
        class="lora-dataset-panel__list-empty"
      />
      <DqEmpty v-else :description="$t('loraTrain.noDatasets')" class="lora-dataset-panel__list-empty" />

      <div class="lora-dataset-panel__list-foot">
        <DqButton size="xs" type="secondary" :loading="importingDog6" block @click="importDog6">
          {{ $t('loraTrain.importDog6') }}
        </DqButton>
      </div>
    </aside>

    <!-- Right: active dataset workspace -->
    <div class="lora-dataset-panel__workspace">
      <template v-if="selectedId && selectedDataset">
        <div class="lora-dataset-panel__workspace-head">
          <div class="lora-dataset-panel__name-row">
            <label class="lora-dataset-panel__label">{{ $t('loraTrain.datasetName') }}</label>
            <DqInput
              v-model="datasetNameEdit"
              size="sm"
              class="lora-dataset-panel__name-input"
              @blur="saveDatasetName"
            />
          </div>
          <DqTag
            size="small"
            :type="imageCount >= minImages ? 'success' : 'warning'"
            effect="plain"
          >
            {{ $t('loraTrain.datasetProgress', { current: imageCount, min: minImages }) }}
          </DqTag>
        </div>

        <div
          class="lora-dataset-panel__dropzone"
          :class="{ 'is-dragover': dragOver, 'is-disabled': !selectedId, 'is-uploading': uploading }"
          @dragover.prevent="dragOver = true"
          @dragleave.prevent="dragOver = false"
          @drop.prevent="onDropFiles"
          @click="openUploadPicker"
        >
          <DqIcon class="lora-dataset-panel__dropzone-icon" aria-hidden="true"><Upload /></DqIcon>
          <p class="lora-dataset-panel__dropzone-title">
            {{ uploading ? $t('loraTrain.uploading') : $t('loraTrain.dropzoneTitle') }}
          </p>
          <p v-if="!uploading" class="lora-dataset-panel__dropzone-hint">{{ $t('loraTrain.dropzoneHint') }}</p>
          <input
            ref="uploadInputRef"
            type="file"
            multiple
            accept="image/*"
            hidden
            @change="onUploadImages"
          />
        </div>

        <div class="lora-dataset-panel__actions">
          <DqButton size="sm" type="secondary" @click="showGalleryImport = true">
            {{ $t('loraTrain.importGallery') }}
          </DqButton>
          <DqButton
            size="sm"
            type="secondary"
            :disabled="!selectedDataset?.images?.length"
            :loading="autoCaptioning"
            @click="runAutoCaption"
          >
            {{ $t('loraTrain.autoCaption') }}
          </DqButton>
          <DqButton
            size="sm"
            type="secondary"
            :disabled="!selectedDataset?.images?.length || !defaultPrompt.trim()"
            @click="applyDefaultToAll"
          >
            {{ $t('loraTrain.applyDefaultToAll') }}
          </DqButton>
        </div>

        <div v-if="selectedDataset?.images?.length" class="lora-dataset-panel__grid">
          <div
            v-for="img in selectedDataset.images"
            :key="img.file"
            class="lora-dataset-panel__cell"
          >
            <div class="lora-dataset-panel__cell-media">
              <img
                v-if="datasetImageUrl(img.file)"
                :src="datasetImageUrl(img.file)"
                :alt="img.file"
                loading="lazy"
              />
              <DqIconButton
                type="text"
                size="xs"
                class="lora-dataset-panel__cell-remove"
                :label="$t('loraTrain.removeImage')"
                @click.stop="removeImage(img.file)"
              >
                <DqIcon :size="14"><Close /></DqIcon>
              </DqIconButton>
            </div>
            <DqInput
              :model-value="captionEdits[img.file]"
              size="xs"
              :placeholder="$t('loraTrain.captionPlaceholder')"
              @update:model-value="(v: string) => setCaption(img.file, v)"
              @blur="saveCaptions"
            />
          </div>
        </div>
        <DqEmpty v-else :description="$t('loraTrain.noImages')" />

        <div class="lora-dataset-panel__field">
          <label class="lora-dataset-panel__label">{{ $t('loraTrain.defaultPrompt') }}</label>
          <DqInput
            :model-value="defaultPrompt"
            type="textarea"
            :rows="2"
            :placeholder="$t('loraTrain.defaultPromptHint')"
            @update:model-value="(v: string) => emit('update:defaultPrompt', v)"
            @blur="patchDefaultPrompt"
          />
          <p class="lora-dataset-panel__field-hint">{{ $t('loraTrain.defaultPromptDesc') }}</p>
        </div>
      </template>

      <DqEmpty v-else :description="$t('loraTrain.selectOrCreateDataset')" />
    </div>

    <DqDialog
      v-model:open="showCreateDialog"
      :title="$t('loraTrain.newDataset')"
      width="min(420px, 92vw)"
      destroy-on-close
    >
      <div class="lora-dataset-panel__dialog-field">
        <label class="lora-dataset-panel__label">{{ $t('loraTrain.newDatasetName') }}</label>
        <DqInput v-model="newDatasetName" :placeholder="$t('loraTrain.newDatasetDefault')" @keyup.enter="createDataset" />
      </div>
      <template #footer>
        <DqButton size="sm" @click="showCreateDialog = false">{{ $t('common.cancel') }}</DqButton>
        <DqButton size="sm" type="primary" :disabled="!newDatasetName.trim()" @click="createDataset">
          {{ $t('common.confirm') }}
        </DqButton>
      </template>
    </DqDialog>

    <DqDialog
      v-model:open="showGalleryImport"
      :title="$t('loraTrain.importGalleryTitle')"
      width="min(560px, 92vw)"
      destroy-on-close
    >
      <p class="lora-dataset-panel__import-hint">{{ $t('loraTrain.importGalleryHint') }}</p>
      <AssetPicker accept-kind="image" @pick="onGalleryPick" />
      <div v-if="pendingGalleryAssets.length" class="lora-dataset-panel__pending-tags">
        <span
          v-for="a in pendingGalleryAssets"
          :key="a"
          class="lora-dataset-panel__pending-tag"
        >
          <span class="lora-dataset-panel__pending-tag-label">{{ a }}</span>
          <DqIconButton
            type="text"
            size="xs"
            :label="$t('common.delete')"
            @click="removePendingAsset(a)"
          >
            <DqIcon :size="12"><Close /></DqIcon>
          </DqIconButton>
        </span>
      </div>
      <template #footer>
        <DqButton size="sm" @click="showGalleryImport = false">{{ $t('common.cancel') }}</DqButton>
        <DqButton
          size="sm"
          type="primary"
          :disabled="!pendingGalleryAssets.length"
          :loading="importingGallery"
          @click="confirmGalleryImport"
        >
          {{ $t('loraTrain.importGalleryConfirm', { count: pendingGalleryAssets.length }) }}
        </DqButton>
      </template>
    </DqDialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, Upload, PictureFilled } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import { toast, confirm } from '@/utils/feedback';
import AssetPicker from '@/components/asset/AssetPicker.vue';

const MIN_IMAGES = 3;

const props = defineProps<{
  selectedId: string;
  datasets: Array<Record<string, any>>;
  defaultPrompt: string;
  captionEdits: Record<string, string>;
}>();

const emit = defineEmits<{
  (e: 'update:selectedId', id: string): void;
  (e: 'update:defaultPrompt', value: string): void;
  (e: 'update:captionEdits', value: Record<string, string>): void;
  (e: 'datasets-changed', datasets: Array<Record<string, any>>): void;
}>();

const { t } = useI18n();

const minImages = MIN_IMAGES;
const dragOver = ref(false);
const uploading = ref(false);
const importingDog6 = ref(false);
const importingGallery = ref(false);
const autoCaptioning = ref(false);
const showCreateDialog = ref(false);
const showGalleryImport = ref(false);
const newDatasetName = ref('');
const datasetSearch = ref('');
const pendingGalleryAssets = ref<string[]>([]);
const uploadInputRef = ref<HTMLInputElement | null>(null);
const datasetNameEdit = ref('');

const selectedDataset = computed(() =>
  props.datasets.find((d) => d.id === props.selectedId)
);

const imageCount = computed(() => selectedDataset.value?.image_count || 0);

const filteredDatasets = computed(() => {
  const q = datasetSearch.value.trim().toLowerCase();
  const sorted = [...props.datasets].sort((a, b) =>
    String(b.updated_at || b.created_at || '').localeCompare(String(a.updated_at || a.created_at || ''))
  );
  if (!q) return sorted;
  return sorted.filter(
    (d) =>
      String(d.name || '').toLowerCase().includes(q) ||
      String(d.id || '').toLowerCase().includes(q)
  );
});

watch(
  () => selectedDataset.value?.name,
  (name) => {
    datasetNameEdit.value = name || '';
  },
  { immediate: true }
);

function apiErrorMessage(e: unknown): string {
  const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
  const detail = err?.response?.data?.detail;
  if (detail && typeof detail === 'object' && detail.message) return detail.message;
  if (typeof detail === 'string') return detail;
  return err?.message || String(e);
}

function patchDatasets(next: Array<Record<string, any>>) {
  emit('datasets-changed', next);
}

function selectDataset(id: string) {
  emit('update:selectedId', id);
}

function setCaption(file: string, value: string) {
  emit('update:captionEdits', { ...props.captionEdits, [file]: value });
}

function datasetImageUrl(file: string): string {
  if (!props.selectedId || !file) return '';
  return api.loras.datasetImageUrl(props.selectedId, file);
}

function datasetCoverUrl(d: Record<string, any>): string {
  const cover = d.cover_image || d.images?.[0]?.file;
  if (!cover || !d.id) return '';
  return api.loras.datasetImageUrl(d.id, cover);
}

function openUploadPicker() {
  if (!props.selectedId) return;
  uploadInputRef.value?.click();
}

async function refreshDatasetDetail(id: string) {
  const ds = (await api.loras.getDataset(id)) as Record<string, any>;
  const captions: Record<string, string> = {};
  for (const img of ds.images || []) {
    captions[img.file] = img.prompt || props.defaultPrompt || '';
  }
  emit('update:captionEdits', captions);
  patchDatasets(props.datasets.map((d) => (d.id === id ? { ...d, ...ds } : d)));
}

async function createDataset() {
  const name = newDatasetName.value.trim();
  if (!name) return;
  try {
    const ds = (await api.loras.createDataset({
      name,
      default_prompt: props.defaultPrompt,
    })) as Record<string, any>;
    patchDatasets([ds, ...props.datasets]);
    emit('update:selectedId', ds.id);
    showCreateDialog.value = false;
    newDatasetName.value = '';
    toast.success(t('loraTrain.datasetCreated'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function importDog6() {
  importingDog6.value = true;
  try {
    const ds = (await api.loras.importDog6()) as Record<string, any>;
    const exists = props.datasets.some((d) => d.id === ds.id);
    patchDatasets(exists ? props.datasets.map((d) => (d.id === ds.id ? { ...d, ...ds } : d)) : [ds, ...props.datasets]);
    emit('update:selectedId', ds.id);
    await refreshDatasetDetail(ds.id);
    toast.success(t('loraTrain.dog6Imported'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    importingDog6.value = false;
  }
}

function openCreateDialog() {
  newDatasetName.value = t('loraTrain.newDatasetDefault');
  showCreateDialog.value = true;
}

function removePendingAsset(path: string) {
  pendingGalleryAssets.value = pendingGalleryAssets.value.filter((p) => p !== path);
}

async function applyDefaultToAll() {
  if (!props.selectedId || !props.defaultPrompt.trim()) return;
  const prompt = props.defaultPrompt.trim();
  const captions = (selectedDataset.value?.images || []).map((img: { file: string }) => ({
    file: img.file,
    prompt,
  }));
  if (!captions.length) return;
  try {
    const ds = (await api.loras.updateCaptions(props.selectedId, captions)) as Record<string, any>;
    const next: Record<string, string> = {};
    for (const img of ds.images || []) {
      next[img.file] = img.prompt || prompt;
    }
    emit('update:captionEdits', next);
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    toast.success(t('loraTrain.applyDefaultDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function uploadFiles(files: File[]) {
  if (!files.length || !props.selectedId || uploading.value) return;
  uploading.value = true;
  try {
    await api.loras.uploadImages(props.selectedId, files, props.defaultPrompt);
    await refreshDatasetDetail(props.selectedId);
    toast.success(t('loraTrain.uploadDone', { count: files.length }));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    uploading.value = false;
  }
}

async function onUploadImages(ev: Event) {
  const input = ev.target as HTMLInputElement;
  const files = input.files ? Array.from(input.files) : [];
  await uploadFiles(files);
  input.value = '';
}

async function onDropFiles(ev: DragEvent) {
  dragOver.value = false;
  if (!props.selectedId) return;
  const files = Array.from(ev.dataTransfer?.files || []).filter((f) => f.type.startsWith('image/'));
  if (!files.length) {
    toast.error(t('loraTrain.dropzoneNeedImage'));
    return;
  }
  await uploadFiles(files);
}

function onGalleryPick(payload: { path: string }) {
  const p = payload.path?.trim();
  if (!p || pendingGalleryAssets.value.includes(p)) return;
  pendingGalleryAssets.value.push(p);
}

async function confirmGalleryImport() {
  if (!props.selectedId || !pendingGalleryAssets.value.length) return;
  importingGallery.value = true;
  try {
    const ds = (await api.loras.importAssets(
      props.selectedId,
      pendingGalleryAssets.value,
      props.defaultPrompt
    )) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    await refreshDatasetDetail(props.selectedId);
    pendingGalleryAssets.value = [];
    showGalleryImport.value = false;
    toast.success(t('loraTrain.galleryImported'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    importingGallery.value = false;
  }
}

async function runAutoCaption() {
  if (!props.selectedId) return;
  autoCaptioning.value = true;
  try {
    const ds = (await api.loras.autoCaption(props.selectedId)) as Record<string, any>;
    const captions: Record<string, string> = { ...props.captionEdits };
    for (const img of ds.images || []) {
      captions[img.file] = img.prompt || '';
    }
    emit('update:captionEdits', captions);
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    toast.success(t('loraTrain.autoCaptionDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    autoCaptioning.value = false;
  }
}

async function saveCaptions() {
  if (!props.selectedId) return;
  const captions = Object.entries(props.captionEdits).map(([file, prompt]) => ({ file, prompt }));
  try {
    await api.loras.updateCaptions(props.selectedId, captions);
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function patchDefaultPrompt() {
  if (!props.selectedId) return;
  await api.loras.patchDataset(props.selectedId, {
    name: datasetNameEdit.value.trim() || selectedDataset.value?.name || '',
    default_prompt: props.defaultPrompt,
  });
}

async function saveDatasetName() {
  if (!props.selectedId) return;
  const name = datasetNameEdit.value.trim();
  if (!name || name === selectedDataset.value?.name) return;
  try {
    const ds = (await api.loras.patchDataset(props.selectedId, {
      name,
      default_prompt: props.defaultPrompt,
    })) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function removeImage(file: string) {
  if (!props.selectedId) return;
  try {
    await confirm(
      t('loraTrain.removeImageConfirm'),
      t('loraTrain.removeImage'),
      { type: 'warning' }
    );
  } catch {
    return;
  }
  try {
    const ds = (await api.loras.deleteDatasetImage(props.selectedId, file)) as Record<string, any>;
    const nextCaptions = { ...props.captionEdits };
    delete nextCaptions[file];
    emit('update:captionEdits', nextCaptions);
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    toast.success(t('loraTrain.removeImageDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

defineExpose({
  refreshDatasetDetail,
  async createDatasetWithName(name: string) {
    newDatasetName.value = name;
    await createDataset();
  },
});
</script>

<style scoped>
.lora-dataset-panel {
  display: flex;
  gap: 16px;
  min-height: 380px;
}

.lora-dataset-panel__list {
  width: 248px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
  border-radius: 14px;
  background: color-mix(in srgb, var(--dq-bg-base) 45%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-dataset-panel__list-head {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lora-dataset-panel__list-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--dq-label-secondary);
  letter-spacing: 0.02em;
}

.lora-dataset-panel__persist-note {
  margin: 0;
  font-size: 10px;
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__search {
  width: 100%;
}

.lora-dataset-panel__ready-tag {
  margin-left: 4px;
}

.lora-dataset-panel__list-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lora-dataset-panel__list-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 10px;
  border: 0.5px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.lora-dataset-panel__list-thumb {
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  object-fit: cover;
  border-radius: var(--radius-sm);
  border: 0.5px solid var(--dq-border-subtle);
  background: var(--dq-bg-base);
}

.lora-dataset-panel__list-thumb--empty {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__list-body {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  min-width: 0;
}

.lora-dataset-panel__list-item:hover {
  background: var(--dq-fill-tertiary);
}

.lora-dataset-panel__list-item.is-active {
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-fill-secondary));
  border-color: color-mix(in srgb, var(--dq-accent) 35%, transparent);
}

.lora-dataset-panel__list-item.is-ready:not(.is-active) {
  border-color: color-mix(in srgb, var(--dq-success) 25%, transparent);
}

.lora-dataset-panel__list-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.lora-dataset-panel__list-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__list-empty {
  flex: 1;
  padding: 12px 0;
}

.lora-dataset-panel__list-foot {
  padding-top: 4px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.lora-dataset-panel__workspace {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.lora-dataset-panel__workspace-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 10px;
}

.lora-dataset-panel__name-row {
  flex: 1;
  min-width: 180px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lora-dataset-panel__name-input {
  max-width: 360px;
}

.lora-dataset-panel__dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 20px 16px;
  border-radius: var(--radius-md);
  border: 1.5px dashed var(--dq-border);
  background: var(--dq-fill-secondary);
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.lora-dataset-panel__dropzone:hover,
.lora-dataset-panel__dropzone.is-dragover {
  border-color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 8%, var(--dq-fill-secondary));
}

.lora-dataset-panel__dropzone.is-disabled {
  opacity: 0.5;
  pointer-events: none;
}

.lora-dataset-panel__dropzone.is-uploading {
  pointer-events: none;
  border-style: solid;
  border-color: var(--dq-accent);
}

.lora-dataset-panel__dropzone-icon {
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__dropzone-title {
  margin: 0;
  font-size: 13px;
  font-weight: 500;
  color: var(--dq-label-primary);
}

.lora-dataset-panel__dropzone-hint {
  margin: 0;
  font-size: 11px;
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.lora-dataset-panel__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
  gap: 12px;
}

.lora-dataset-panel__cell {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-dataset-panel__cell-media {
  position: relative;
}

.lora-dataset-panel__cell-media img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--radius-md);
  border: 0.5px solid var(--dq-border);
  background: var(--dq-bg-base);
}

.lora-dataset-panel__cell-remove {
  position: absolute;
  top: 4px;
  right: 4px;
  background: color-mix(in srgb, var(--dq-bg-base) 85%, transparent) !important;
  border-radius: var(--radius-sm);
}

.lora-dataset-panel__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-dataset-panel__label {
  font-size: 11px;
  font-weight: 500;
  color: var(--dq-label-secondary);
}

.lora-dataset-panel__field-hint {
  margin: 0;
  font-size: 11px;
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.lora-dataset-panel__import-hint {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__pending-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}

.lora-dataset-panel__pending-tag {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  max-width: 100%;
  padding: 2px 4px 2px 8px;
  border-radius: var(--radius-sm);
  background: var(--dq-fill-secondary);
  border: 0.5px solid var(--dq-border-subtle);
  font-size: 11px;
  color: var(--dq-label-secondary);
}

.lora-dataset-panel__pending-tag-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 200px;
}

.lora-dataset-panel__dialog-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
</style>
