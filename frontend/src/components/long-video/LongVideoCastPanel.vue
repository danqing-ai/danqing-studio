<template>
  <section class="lv-cast lv-panel lv-section lv-section--compact">
    <header class="lv-section__head lv-cast__head">
      <div class="lv-cast__head-left">
        <span class="lv-section__title">{{ $tt('video.longVideoCastRosterTitle') }}</span>
        <span v-if="characters.length" class="lv-cast__badge">{{ characters.length }}</span>
      </div>
      <div class="lv-cast__head-actions">
        <DqButton v-if="canImport" type="text" size="sm" @click="importFromAnchor">
          {{ $tt('video.longVideoCastImportFromAnchor') }}
        </DqButton>
        <DqButton type="text" size="sm" @click="onAddCharacter">
          <DqIcon :size="12"><Plus /></DqIcon>
          {{ $tt('video.longVideoCastAddCharacter') }}
        </DqButton>
      </div>
    </header>

    <div class="lv-cast__style">
      <label class="lv-cast__style-label">{{ $tt('video.longVideoCastStyleLabel') }}</label>
      <DqInput
        :model-value="styleAnchor"
        size="small"
        class="lv-cast__style-input"
        :placeholder="$tt('video.longVideoCastStylePh')"
        @update:model-value="$emit('update:styleAnchor', $event)"
      />
    </div>

    <div v-if="!characters.length" class="lv-cast__empty">
      <p class="lv-cast__empty-text">{{ $tt('video.longVideoCastRosterEmpty') }}</p>
    </div>

    <div v-else class="lv-cast__list">
      <article v-for="(ch, ci) in characters" :key="ch.id" class="lv-cast__group">
        <header class="lv-cast__group-head">
          <span class="lv-cast__avatar" aria-hidden="true">{{ characterInitial(ch.name) }}</span>
          <DqInput
            :model-value="ch.name"
            size="small"
            class="lv-cast__name"
            :placeholder="$tt('video.longVideoCastNameExamplePh')"
            @update:model-value="onCharacterName(ci, $event)"
          />
          <DqIconButton
            type="text"
            size="xs"
            class="lv-cast__danger"
            :label="$tt('video.longVideoCastRemove')"
            @click="removeCharacter(ci)"
          >
            <DqIcon :size="13"><Delete /></DqIcon>
          </DqIconButton>
        </header>

        <div class="lv-cast__looks">
          <div v-for="(lk, li) in ch.looks" :key="lk.id" class="lv-cast__look">
            <DqInput
              :model-value="lk.label"
              size="small"
              class="lv-cast__look-label"
              :placeholder="$tt('video.longVideoCastLookLabelExamplePh')"
              @update:model-value="onLookLabel(ci, li, $event)"
            />
            <DqInput
              :model-value="lk.body"
              type="textarea"
              :rows="2"
              size="small"
              class="lv-cast__look-body"
              :placeholder="$tt('video.longVideoCastLookBodyExamplePh')"
              @update:model-value="onLookBody(ci, li, $event)"
            />
            <DqIconButton
              v-if="ch.looks.length > 1"
              type="text"
              size="xs"
              class="lv-cast__look-remove"
              :label="$tt('common.delete')"
              @click="removeLook(ci, li)"
            >
              <DqIcon :size="12"><Close /></DqIcon>
            </DqIconButton>
          </div>
        </div>

        <footer class="lv-cast__group-foot">
          <button type="button" class="lv-cast__add-look" @click="addLook(ci)">
            <DqIcon :size="11"><Plus /></DqIcon>
            {{ $tt('video.longVideoCastAddLook') }}
          </button>
        </footer>
      </article>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, Delete, Plus } from '@danqing/dq-shell';
import type { LongVideoCharacter } from '@/types';
import {
  createCharacterEntry,
  createLookEntry,
  parseCharacterRosterFromAnchor,
} from '@/utils/longVideoProject';

const props = defineProps<{
  characters: LongVideoCharacter[];
  styleAnchor: string;
  characterAnchor?: string;
}>();

const emit = defineEmits<{
  (e: 'update:characters', value: LongVideoCharacter[]): void;
  (e: 'update:styleAnchor', value: string): void;
}>();

const { t: $tt, locale } = useI18n();

const canImport = computed(
  () => Boolean(props.characterAnchor?.trim()) && props.characters.length === 0,
);

function loc(): 'zh' | 'en' {
  return String(locale.value).startsWith('zh') ? 'zh' : 'en';
}

function characterInitial(name: string): string {
  const trimmed = name.trim();
  return trimmed ? trimmed.slice(0, 1) : '?';
}

function emitCharacters(next: LongVideoCharacter[]) {
  emit('update:characters', next);
}

function onAddCharacter() {
  emitCharacters([...props.characters, createCharacterEntry($tt('video.longVideoCastNewCharacterName'), loc())]);
}

function importFromAnchor() {
  const anchor = props.characterAnchor?.trim();
  if (!anchor) return;
  const parsed = parseCharacterRosterFromAnchor(anchor, loc());
  if (!parsed.characters.length) return;
  emit('update:styleAnchor', props.styleAnchor.trim() || parsed.styleAnchor);
  emitCharacters(parsed.characters);
}

function onCharacterName(index: number, name: string) {
  emitCharacters(props.characters.map((c, i) => (i === index ? { ...c, name } : c)));
}

function removeCharacter(index: number) {
  emitCharacters(props.characters.filter((_, i) => i !== index));
}

function onLookLabel(ci: number, li: number, label: string) {
  emitCharacters(
    props.characters.map((c, i) =>
      i !== ci ? c : { ...c, looks: c.looks.map((lk, j) => (j === li ? { ...lk, label } : lk)) },
    ),
  );
}

function onLookBody(ci: number, li: number, body: string) {
  emitCharacters(
    props.characters.map((c, i) =>
      i !== ci ? c : { ...c, looks: c.looks.map((lk, j) => (j === li ? { ...lk, body } : lk)) },
    ),
  );
}

function addLook(ci: number) {
  const ch = props.characters[ci];
  if (!ch) return;
  emitCharacters(
    props.characters.map((c, i) =>
      i === ci
        ? { ...c, looks: [...c.looks, createLookEntry(c.name, $tt('video.longVideoCastNewLookLabel'), loc())] }
        : c,
    ),
  );
}

function removeLook(ci: number, li: number) {
  emitCharacters(
    props.characters.map((c, i) => {
      if (i !== ci) return c;
      const looks = c.looks.filter((_, j) => j !== li);
      const default_look_id =
        c.default_look_id === c.looks[li]?.id ? looks[0]?.id ?? '' : c.default_look_id;
      return { ...c, looks, default_look_id };
    }),
  );
}
</script>
