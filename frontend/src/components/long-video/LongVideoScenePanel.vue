<template>
  <section class="lv-scene-workshop lv-panel lv-section">
    <header class="lv-scene-workshop__header">
      <div class="lv-scene-workshop__header-main">
        <h2 class="lv-scene-workshop__title">{{ $tt('video.longVideoSceneWorkshopTitle') }}</h2>
        <p class="lv-scene-workshop__subtitle">{{ $tt('video.longVideoSceneWorkshopHint') }}</p>
      </div>
      <DqButton
        v-if="missingRefCount > 0"
        size="sm"
        type="default"
        :loading="batchGenerating"
        :disabled="batchGenerating || refGeneratingKey != null"
        @click="emit('batch-generate-refs')"
      >
        {{ $tt('video.longVideoSceneRefBatch', { n: missingRefCount }) }}
      </DqButton>
    </header>

    <div v-if="!scenes.length" class="lv-scene-workshop__empty">
      <p class="lv-scene-workshop__empty-text">{{ $tt('video.longVideoSceneRosterEmpty') }}</p>
      <DqButton type="primary" size="sm" @click="onAddScene">
        <DqIcon :size="12"><Plus /></DqIcon>
        {{ $tt('video.longVideoSceneAdd') }}
      </DqButton>
    </div>

    <div v-else class="lv-scene-workshop__body">
      <nav class="lv-scene-workshop__sidebar" :aria-label="$tt('video.longVideoSceneSidebarAria')">
        <button
          v-for="(sc, si) in scenes"
          :key="sc.id"
          type="button"
          class="lv-scene-workshop__item"
          :class="{ 'lv-scene-workshop__item--active': selectedIndex === si }"
          @click="selectedIndex = si"
        >
          <div class="lv-scene-workshop__thumb" :class="{ 'is-empty': !primaryRefAsset(sc) }">
            <img v-if="primaryRefAsset(sc)" :src="refUrl(primaryRefAsset(sc)!)" alt="" />
            <span v-else>{{ sceneInitial(sc.name) }}</span>
          </div>
          <span class="lv-scene-workshop__name">{{ sc.name.trim() || $tt('video.longVideoSceneUntitled') }}</span>
        </button>
        <button type="button" class="lv-scene-workshop__item lv-scene-workshop__item--add" @click="onAddScene">
          <DqIcon :size="14"><Plus /></DqIcon>
          {{ $tt('video.longVideoSceneAdd') }}
        </button>
      </nav>

      <div v-if="selectedScene" class="lv-scene-workshop__detail">
        <header class="lv-scene-workshop__detail-head">
          <DqInput
            :model-value="selectedScene.name"
            size="small"
            class="lv-scene-workshop__detail-name"
            :placeholder="$tt('video.longVideoSceneNamePh')"
            @update:model-value="onSceneName(selectedIndex, $event)"
          />
          <DqButton size="sm" type="primary" @click="addLook(selectedIndex)">
            {{ $tt('video.longVideoSceneAddVariant') }}
          </DqButton>
        </header>

        <article
          v-for="(lk, li) in selectedScene.looks"
          :key="lk.id"
          class="lv-scene-workshop__look"
          :class="{ 'is-ready': Boolean(lk.reference_asset_id) }"
        >
          <div
            class="lv-scene-workshop__ref"
            :class="{ 'is-loading': refGeneratingKey === refKey(selectedIndex, li) }"
          >
            <img v-if="lk.reference_asset_id" :src="refUrl(lk.reference_asset_id)" alt="" />
            <span v-else class="lv-scene-workshop__ref-empty">{{ $tt('video.longVideoSceneRefMissing') }}</span>
          </div>
          <div class="lv-scene-workshop__fields">
            <span class="lv-scene-workshop__look-kicker">
              {{ formatSceneLookOptionLabel(lk, uiLocale) }}
            </span>
            <DqInput
              :model-value="lk.label"
              size="small"
              :placeholder="$tt('video.longVideoSceneVariantPh')"
              @update:model-value="onLookLabel(selectedIndex, li, $event)"
            />
            <DqInput
              :model-value="lk.body"
              type="textarea"
              :rows="3"
              size="small"
              :placeholder="$tt('video.longVideoSceneBodyPh')"
              @update:model-value="onLookBody(selectedIndex, li, $event)"
            />
            <DqInput
              v-if="lk.reference_asset_id || lk.vision_description"
              :model-value="lk.vision_description || ''"
              type="textarea"
              :rows="2"
              size="small"
              :placeholder="$tt('video.longVideoSceneVisionDescriptionExamplePh')"
              @update:model-value="onLookVisionDescription(selectedIndex, li, $event)"
            />
          </div>
          <footer class="lv-scene-workshop__foot">
            <DqButton
              size="sm"
              type="primary"
              :loading="refGeneratingKey === refKey(selectedIndex, li)"
              :disabled="batchGenerating || refGeneratingKey != null"
              @click="emit('generate-ref', selectedIndex, li)"
            >
              {{ $tt('video.longVideoSceneRefGenerate') }}
            </DqButton>
            <DqButton
              size="sm"
              type="default"
              :disabled="batchGenerating || refGeneratingKey != null"
              @click="emit('pick-ref-gallery', selectedIndex, li)"
            >
              {{ $tt('video.longVideoSceneRefImport') }}
            </DqButton>
            <DqButton
              v-if="lk.reference_asset_id"
              size="sm"
              type="text"
              :loading="visionBackfillKey === refKey(selectedIndex, li)"
              @click="emit('vision-backfill', selectedIndex, li)"
            >
              {{ $tt('video.longVideoSceneVisionBackfill') }}
            </DqButton>
            <DqButton
              v-if="lk.reference_asset_id"
              size="sm"
              type="text"
              @click="emit('clear-ref', selectedIndex, li)"
            >
              {{ $tt('video.longVideoClearShort') }}
            </DqButton>
          </footer>
          <LongVideoEntityRelatedTasks
            v-if="projectId && selectedScene"
            :project-id="projectId"
            phase="scene_ref"
            :match="{ scene_id: selectedScene.id, scene_look_id: lk.id }"
            :label="$tt('video.longVideoRelatedTasksSceneRef')"
            :refresh-token="refGeneratingKey"
          />
        </article>
      </div>
    </div>

    <footer v-if="scriptParsed && scenes.length" class="lv-scene-workshop__next">
      <p>{{ $tt('video.longVideoSceneNextStoryboard') }}</p>
      <DqButton type="primary" size="sm" @click="emit('go-to-storyboard')">
        {{ $tt('video.longVideoCastNextStoryboardBtn') }}
      </DqButton>
    </footer>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Plus } from '@danqing/dq-shell';
import LongVideoEntityRelatedTasks from './LongVideoEntityRelatedTasks.vue';
import { api } from '@/utils/api';
import type { LongVideoScene } from '@/types';
import {
  createSceneEntry,
  createSceneLookEntry,
  formatSceneLookOptionLabel,
  looksMissingSceneReference,
} from '@/utils/longVideoProject';

const props = defineProps<{
  scenes: LongVideoScene[];
  scriptParsed?: boolean;
  projectId?: string;
  refGeneratingKey?: string | null;
  visionBackfillKey?: string | null;
  batchGenerating?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:scenes', value: LongVideoScene[]): void;
  (e: 'generate-ref', sceneIndex: number, lookIndex: number): void;
  (e: 'pick-ref-gallery', sceneIndex: number, lookIndex: number): void;
  (e: 'clear-ref', sceneIndex: number, lookIndex: number): void;
  (e: 'vision-backfill', sceneIndex: number, lookIndex: number): void;
  (e: 'batch-generate-refs'): void;
  (e: 'go-to-storyboard'): void;
}>();

const { t: $tt, locale } = useI18n();
const uiLocale = computed(() => (locale.value.startsWith('zh') ? 'zh' : 'en'));
const selectedIndex = ref(0);

const missingRefCount = computed(() => looksMissingSceneReference(props.scenes).length);
const selectedScene = computed(() => props.scenes[selectedIndex.value] ?? null);

watch(
  () => props.scenes.length,
  (n) => {
    if (selectedIndex.value >= n) selectedIndex.value = Math.max(0, n - 1);
  },
);

function refKey(si: number, li: number) {
  return `${si}-${li}`;
}

function refUrl(assetId: string) {
  return api.gallery.getImageUrl(`asset:${assetId}`);
}

function sceneInitial(name: string) {
  return (name.trim()[0] || '?').toUpperCase();
}

function primaryRefAsset(sc: LongVideoScene) {
  const lk = sc.looks.find((l) => l.id === sc.default_look_id) || sc.looks[0];
  return lk?.reference_asset_id;
}

function patchScenes(next: LongVideoScene[]) {
  emit('update:scenes', next);
}

function onAddScene() {
  const loc = locale.value.startsWith('zh') ? 'zh' : 'en';
  const entry = createSceneEntry(loc === 'zh' ? '新场景' : 'New scene', loc);
  patchScenes([...props.scenes, entry]);
  selectedIndex.value = props.scenes.length;
}

function onSceneName(si: number, name: string) {
  patchScenes(
    props.scenes.map((sc, i) => (i === si ? { ...sc, name } : sc)),
  );
}

function addLook(si: number) {
  const sc = props.scenes[si];
  if (!sc) return;
  const loc = locale.value.startsWith('zh') ? 'zh' : 'en';
  const lk = createSceneLookEntry(sc.name, loc === 'zh' ? '变体' : 'variant', loc);
  patchScenes(
    props.scenes.map((s, i) => (i === si ? { ...s, looks: [...s.looks, lk] } : s)),
  );
}

function onLookLabel(si: number, li: number, label: string) {
  patchScenes(
    props.scenes.map((sc, i) =>
      i !== si
        ? sc
        : {
            ...sc,
            looks: sc.looks.map((lk, j) => (j === li ? { ...lk, label } : lk)),
          },
    ),
  );
}

function onLookBody(si: number, li: number, body: string) {
  patchScenes(
    props.scenes.map((sc, i) =>
      i !== si
        ? sc
        : {
            ...sc,
            looks: sc.looks.map((lk, j) =>
              j === li ? { ...lk, body, environment_prompt: undefined } : lk,
            ),
          },
    ),
  );
}

function onLookVisionDescription(si: number, li: number, visionDescription: string) {
  const trimmed = visionDescription.trim();
  patchScenes(
    props.scenes.map((sc, i) =>
      i !== si
        ? sc
        : {
            ...sc,
            looks: sc.looks.map((lk, j) => {
              if (j !== li) return lk;
              const next = { ...lk, vision_description: trimmed || undefined };
              if (!trimmed) delete next.vision_description;
              return next;
            }),
          },
    ),
  );
}
</script>

<style scoped>
.lv-scene-workshop {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
}

.lv-scene-workshop__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.lv-scene-workshop__title {
  margin: 0 0 4px;
  font-size: var(--dq-font-size-title);
  font-weight: 600;
}

.lv-scene-workshop__subtitle {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.45;
}

.lv-scene-workshop__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 32px;
  text-align: center;
  color: var(--dq-label-secondary);
}

.lv-scene-workshop__body {
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 16px;
  min-height: 320px;
}

.lv-scene-workshop__sidebar {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lv-scene-workshop__item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  border-radius: 8px;
  background: transparent;
  cursor: pointer;
  text-align: left;
  color: var(--dq-label-primary);
}

.lv-scene-workshop__item--active {
  border-color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lv-scene-workshop__thumb {
  width: 36px;
  height: 36px;
  border-radius: 6px;
  overflow: hidden;
  background: var(--dq-surface-elevated);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
}

.lv-scene-workshop__thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.lv-scene-workshop__name {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
}

.lv-scene-workshop__detail-head {
  display: flex;
  gap: 8px;
  margin-bottom: 12px;
}

.lv-scene-workshop__detail-name {
  flex: 1;
}

.lv-scene-workshop__look {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 12px;
  padding: 12px;
  margin-bottom: 12px;
  border-radius: 10px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
}

.lv-scene-workshop__look.is-ready {
  border-color: color-mix(in srgb, var(--dq-accent) 35%, transparent);
}

.lv-scene-workshop__ref {
  aspect-ratio: 16 / 9;
  border-radius: 8px;
  overflow: hidden;
  background: var(--dq-surface-elevated);
  display: flex;
  align-items: center;
  justify-content: center;
}

.lv-scene-workshop__ref img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.lv-scene-workshop__ref-empty {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  padding: 8px;
  text-align: center;
}

.lv-scene-workshop__fields {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-scene-workshop__look-kicker {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  line-height: 1.4;
}

.lv-scene-workshop__foot {
  grid-column: 1 / -1;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.lv-scene-workshop__next {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding-top: 12px;
  border-top: 0.5px solid var(--dq-border-subtle);
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}
</style>
