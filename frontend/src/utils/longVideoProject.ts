import type {
  LongVideoChainMode,
  LongVideoCharacter,
  LongVideoCharacterLook,
  LongVideoProjectState,
  LongVideoSelection,
  LongVideoShotCastLook,
  LongVideoShotState,
} from '@/types';

/** Minimum keyframes required to form at least one I2V segment. */
export const MIN_LONG_VIDEO_KEYFRAMES = 2;

/** Default I2V segment length when a shot has no ``duration_sec``. */
export const DEFAULT_SHOT_DURATION_SEC = 5;

export function shotDurationSec(shot: LongVideoShotState | undefined, fallback = DEFAULT_SHOT_DURATION_SEC): number {
  const sec = shot?.duration_sec;
  return typeof sec === 'number' && sec > 0 ? sec : fallback;
}

export function effectiveShotChainMode(
  shot: LongVideoShotState | undefined,
  defaultMode: LongVideoChainMode,
): LongVideoChainMode {
  const mode = shot?.chain_mode;
  return mode === 'keyframe_only' || mode === 'last_frame' ? mode : defaultMode;
}

const APPEARANCE_STOP = new Set([
  '近景', '远景', '广角', '中景', '特写', '镜头', '固定', '推近', '跟拍', '手持', '缓慢', '侧移', '环绕', '快切',
  'background', 'lighting', 'medium', 'close', 'wide', 'dolly', 'camera', 'shot', 'frame', 'motion', 'slow',
]);

function appearanceKeywords(text: string): Set<string> {
  const tokens = new Set<string>();
  for (const m of text.matchAll(/[\u4e00-\u9fff]{2,6}/g)) {
    if (!APPEARANCE_STOP.has(m[0])) tokens.add(m[0]);
  }
  for (const m of text.matchAll(/[a-zA-Z]{4,}/g)) {
    const w = m[0].toLowerCase();
    if (!APPEARANCE_STOP.has(w)) tokens.add(w);
  }
  return tokens;
}

export const KEYFRAME_REF_DIVIDER = '---';

const STYLE_LABELS = new Set(['画风', '风格', 'style', 'look', 'palette', 'film']);
const STYLE_LEAD = new Set(['现代', '写实', '赛博', '古装', '硬科幻', '全局', '风格']);

type AnchorBlock = { kind: 'character' | 'style' | 'other'; name: string; body: string };

function splitAnchorRawBlocks(anchor: string): string[] {
  const text = anchor.trim();
  if (!text) return [];
  if (text.includes(KEYFRAME_REF_DIVIDER)) {
    const parts = text.split(/\n\s*---\s*\n/).map((p) => p.trim()).filter(Boolean);
    if (parts.length) return parts;
  }
  const lines: string[] = [];
  let buf: string[] = [];
  for (const line of text.split('\n')) {
    if (line.trim() === KEYFRAME_REF_DIVIDER) {
      if (buf.length) lines.push(buf.join('\n').trim());
      buf = [];
      continue;
    }
    buf.push(line);
  }
  if (buf.length) lines.push(buf.join('\n').trim());
  if (lines.length > 1) return lines;
  const parts = text.split(/[。\n；;]+/).map((p) => p.trim()).filter(Boolean);
  return parts.length ? parts : [text];
}

function parseAnchorBlocks(anchor: string): AnchorBlock[] {
  const blocks: AnchorBlock[] = [];
  for (const raw of splitAnchorRawBlocks(anchor)) {
    let m = raw.match(/^【角色·([^】]+)】\s*(.+)/s);
    if (m) {
      blocks.push({ kind: 'character', name: m[1].trim(), body: m[2].trim() });
      continue;
    }
    m = raw.match(/^\[Character:\s*([^\]]+)\]\s*(.+)/is);
    if (m) {
      blocks.push({ kind: 'character', name: m[1].trim(), body: m[2].trim() });
      continue;
    }
    m = raw.match(/^【([^】]+)】\s*(.+)/s);
    if (m) {
      const label = m[1].trim();
      const body = m[2].trim();
      if (label.startsWith('角色·')) {
        blocks.push({ kind: 'character', name: label.slice(3).trim(), body });
      } else if (STYLE_LABELS.has(label.toLowerCase()) || label === '画风' || label === '风格') {
        blocks.push({ kind: 'style', name: label, body });
      } else {
        blocks.push({ kind: 'character', name: label, body });
      }
      continue;
    }
    m = raw.match(/^\[(Style|Look)\]\s*(.+)/is);
    if (m) {
      blocks.push({ kind: 'style', name: m[1].trim(), body: m[2].trim() });
      continue;
    }
    m = raw.match(/^([^，,：:]{1,16})[，,：:]\s*(.+)/s);
    if (m) {
      const name = m[1].trim();
      const body = m[2].trim();
      if (STYLE_LEAD.has(name) || name === '画风' || name === '风格') {
        blocks.push({ kind: 'style', name, body });
      } else {
        blocks.push({ kind: 'character', name, body });
      }
      continue;
    }
    blocks.push({ kind: 'other', name: '', body: raw });
  }
  return blocks;
}

function formatReferenceBlocks(blocks: AnchorBlock[], locale: 'zh' | 'en'): string {
  const lines: string[] = [];
  for (const b of blocks) {
    if (b.kind === 'style') {
      lines.push(locale === 'zh' ? `【画风】${b.body}` : `[Style] ${b.body}`);
    } else if (b.kind === 'character') {
      lines.push(
        locale === 'zh' ? `【角色·${b.name}】${b.body}` : `[Character: ${b.name}] ${b.body}`,
      );
    } else if (b.body) {
      lines.push(b.body);
    }
  }
  return lines.join(`\n${KEYFRAME_REF_DIVIDER}\n`);
}

function extractKeyframeShotScene(visual: string): string {
  const v = visual.trim();
  if (!v) return '';
  const headZh = v.match(/^【本帧】\s*(.+)/s);
  const headEn = v.match(/^\[Shot\]\s*(.+)/is);
  const head = headZh ?? headEn;
  if (head) {
    let part = head[1].trim();
    if (part.includes(KEYFRAME_REF_DIVIDER)) {
      part = part.split(KEYFRAME_REF_DIVIDER)[0].trim();
    }
    return part;
  }
  if (v.includes(KEYFRAME_REF_DIVIDER)) {
    const tail = v.split(KEYFRAME_REF_DIVIDER).pop()?.trim() ?? '';
    const m = tail.match(/【本帧】\s*(.+)/s) ?? tail.match(/\[Shot\]\s*(.+)/is);
    if (m) return m[1].trim();
    if (!/【角色·|\[Character:/i.test(tail)) return tail;
  }
  return v;
}

function joinKeyframePrompt(scene: string, ref: string, locale: 'zh' | 'en'): string {
  const s = scene.trim();
  const r = ref.trim();
  if (!s) return r;
  if (!r) return s;
  const shotLabel = locale === 'zh' ? '【本帧】' : '[Shot] ';
  const body = s.startsWith('【本帧】') || /^\[Shot\]/i.test(s) ? s : `${shotLabel}${s}`;
  return `${body}\n${KEYFRAME_REF_DIVIDER}\n${r}`;
}

function anchorBlocksForVisual(scene: string, anchor: string): AnchorBlock[] {
  const blocks = parseAnchorBlocks(anchor);
  if (!blocks.length) return [];
  if (!scene.trim()) return blocks;
  const matched: AnchorBlock[] = [];
  const styleBlocks: AnchorBlock[] = [];
  const other: AnchorBlock[] = [];
  for (const b of blocks) {
    if (b.kind === 'style') styleBlocks.push(b);
    else if (b.kind === 'character' && b.name && scene.includes(b.name)) matched.push(b);
    else if (b.kind === 'other') other.push(b);
  }
  if (matched.length) return [...matched, ...styleBlocks, ...other];
  const chars = blocks.filter((b) => b.kind === 'character');
  if (chars.length) return [...chars, ...styleBlocks, ...other];
  return blocks;
}

function isStructuredKeyframeVisual(visual: string): boolean {
  const v = visual.trim();
  return v.includes(KEYFRAME_REF_DIVIDER) && (/【本帧】/.test(v) || /\[Shot\]/i.test(v));
}

function visualIncludesAnchorAppearance(visual: string, anchor: string, minRatio = 0.35): boolean {
  const v = visual.trim();
  const a = anchor.trim();
  if (!v || !a) return Boolean(v);
  const blocks = anchorBlocksForVisual(extractKeyframeShotScene(v) || v, a);
  const bodies = blocks.map((b) => b.body).join(' ');
  const anchorKw = appearanceKeywords(bodies || a);
  if (!anchorKw.size) return true;
  let overlap = 0;
  const visualKw = appearanceKeywords(v);
  for (const t of anchorKw) {
    if (visualKw.has(t)) overlap += 1;
  }
  return overlap / anchorKw.size >= minRatio;
}

function formatCastReferenceBlocks(
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  styleAnchor: string,
  locale: 'zh' | 'en',
): string {
  const castMap = new Map(castLooks.map((c) => [c.character_id, c.look_id]));
  const lines: string[] = [];
  for (const ch of characters) {
    const lookId = castMap.get(ch.id);
    if (castLooks.length && !castMap.has(ch.id)) continue;
    const resolvedLookId = lookId || ch.default_look_id;
    const lk = ch.looks.find((l) => l.id === resolvedLookId) || ch.looks[0];
    if (!lk) continue;
    lines.push(
      locale === 'zh'
        ? `【角色·${ch.name}·${lk.label}】${lk.body}`
        : `[Character: ${ch.name} | ${lk.label}] ${lk.body}`,
    );
  }
  if (styleAnchor.trim()) {
    lines.push(locale === 'zh' ? `【画风】${styleAnchor.trim()}` : `[Style] ${styleAnchor.trim()}`);
  }
  return lines.join(`\n${KEYFRAME_REF_DIVIDER}\n`);
}

export function shortCharacterName(name: string): string {
  const n = name.trim();
  if (!n) return '';
  return n.split(/[，,、\s]|穿着|身着|身穿/u)[0]?.trim() ?? n;
}

export function sceneMentionsCharacter(scene: string, name: string): boolean {
  const text = scene.trim();
  const n = name.trim();
  return text.length > 0 && n.length > 0 && text.indexOf(n) >= 0;
}

export function charactersOnScreen(scene: string, characters: LongVideoCharacter[]): LongVideoCharacter[] {
  const text = scene.trim();
  if (!text) return [];
  return characters.filter((ch) => {
    const n = ch.name.trim();
    return n.length > 0 && text.indexOf(n) >= 0;
  });
}

/** Per-shot cast: saved cast_looks merged with scene name match (roster only). */
export function resolveShotCastLooks(
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  sceneText: string,
): LongVideoShotCastLook[] {
  const fromScene = charactersOnScreen(sceneText, characters)
    .map((ch) => ({
      character_id: ch.id,
      look_id: ch.default_look_id || ch.looks[0]?.id || '',
    }))
    .filter((c) => c.look_id);

  if (!castLooks.length) return fromScene;

  const map = new Map(castLooks.map((c) => [c.character_id, c]));
  for (const entry of fromScene) {
    if (!map.has(entry.character_id)) {
      map.set(entry.character_id, entry);
    }
  }
  return [...map.values()];
}

export function charactersForShotCast(
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  sceneText: string,
): LongVideoCharacter[] {
  const resolved = resolveShotCastLooks(characters, castLooks, sceneText);
  const ids = resolved.map((c) => c.character_id);
  return ids
    .map((id) => characters.find((ch) => ch.id === id))
    .filter((ch): ch is LongVideoCharacter => Boolean(ch));
}

export type KeyframePromptContext = {
  characterAnchor?: string;
  characters?: LongVideoCharacter[];
  styleAnchor?: string;
  castLooks?: LongVideoShotCastLook[];
};

/** T2I prompt: scene first; cast/style reference appended after --- (at generate time). */
export function keyframeGenerationPrompt(
  visualPrompt: string,
  ctxOrAnchor: KeyframePromptContext | string,
): string {
  const ctx: KeyframePromptContext =
    typeof ctxOrAnchor === 'string' ? { characterAnchor: ctxOrAnchor } : ctxOrAnchor;
  const scene = extractKeyframeShotScene(visualPrompt).trim() || visualPrompt.trim();
  const anchor = (ctx.characterAnchor ?? '').trim();
  const characters = ctx.characters ?? [];
  const styleAnchor = (ctx.styleAnchor ?? '').trim();
  const castLooks = ctx.castLooks ?? [];
  const locale: 'zh' | 'en' = /[\u4e00-\u9fff]/.test(scene || anchor) ? 'zh' : 'en';

  if (characters.length) {
    const resolvedCast = resolveShotCastLooks(characters, castLooks, scene);
    if (resolvedCast.length) {
      const castChars = charactersForShotCast(characters, resolvedCast, scene);
      const ref = formatCastReferenceBlocks(castChars, resolvedCast, styleAnchor, locale);
      if (!scene) return ref;
      return ref ? joinKeyframePrompt(scene, ref, locale) : scene;
    }
  }

  if (!scene && !anchor) return '';
  if (!anchor) return scene;
  const blocks = anchorBlocksForVisual(scene || visualPrompt, anchor);
  if (!scene) return formatReferenceBlocks(blocks.length ? blocks : parseAnchorBlocks(anchor), locale);
  const ref = formatReferenceBlocks(blocks.length ? blocks : parseAnchorBlocks(anchor), locale);
  return ref ? joinKeyframePrompt(scene, ref, locale) : scene;
}

export function syncRosterToCharacterAnchor(
  characters: LongVideoCharacter[],
  styleAnchor: string,
): string {
  const locale: 'zh' | 'en' = characters.some((c) => /[\u4e00-\u9fff]/.test(c.name)) ? 'zh' : 'en';
  const lines: string[] = [];
  for (const ch of characters) {
    for (const lk of ch.looks) {
      if (!lk.body.trim() && !lk.label.trim()) continue;
      lines.push(
        locale === 'zh'
          ? `【角色·${ch.name}·${lk.label}】${lk.body}`
          : `[Character: ${ch.name} | ${lk.label}] ${lk.body}`,
      );
    }
  }
  if (styleAnchor.trim()) {
    lines.push(locale === 'zh' ? `【画风】${styleAnchor.trim()}` : `[Style] ${styleAnchor.trim()}`);
  }
  return lines.join(`\n${KEYFRAME_REF_DIVIDER}\n`);
}

export function parseCharacterRosterFromAnchor(
  characterAnchor: string,
  locale: 'zh' | 'en' = 'zh',
): { characters: LongVideoCharacter[]; styleAnchor: string } {
  const anchor = characterAnchor.trim();
  if (!anchor) return { characters: [], styleAnchor: '' };

  const defaultLabel = locale === 'zh' ? '默认' : 'default';
  let styleAnchor = '';
  const byName = new Map<string, LongVideoCharacterLook[]>();

  function addLook(name: string, lookLabel: string, body: string) {
    const n = name.trim();
    if (!n) return;
    const lbl = lookLabel.trim() || defaultLabel;
    const looks = byName.get(n) ?? [];
    const existing = looks.find((lk) => lk.label === lbl);
    const trimmedBody = body.trim();
    if (existing) {
      if (trimmedBody) existing.body = trimmedBody;
      byName.set(n, looks);
      return;
    }
    looks.push({
      id: makeCastStableId('look', `${n}|${lbl}`),
      label: lbl,
      body: trimmedBody,
    });
    byName.set(n, looks);
  }

  for (const raw of splitAnchorRawBlocks(anchor)) {
    const block = raw.trim();
    if (!block) continue;

    let m = block.match(/^【角色·([^·】]+)·([^】]+)】\s*(.+)/s);
    if (m) {
      addLook(m[1], m[2], m[3]);
      continue;
    }
    m = block.match(/^\[Character:\s*([^|]+)\|\s*([^\]]+)\]\s*(.+)/is);
    if (m) {
      addLook(m[1], m[2], m[3]);
      continue;
    }
    m = block.match(/^【角色·([^】]+)】\s*(.+)/s);
    if (m) {
      const rest = m[1].trim();
      const body = m[2].trim();
      if (rest.includes('·')) {
        const dot = rest.indexOf('·');
        addLook(rest.slice(0, dot), rest.slice(dot + 1), body);
      } else {
        addLook(rest, defaultLabel, body);
      }
      continue;
    }
    m = block.match(/^\[Character:\s*([^\]]+)\]\s*(.+)/is);
    if (m) {
      addLook(m[1], defaultLabel, m[2]);
      continue;
    }
    m = block.match(/^【([^】]+)】\s*(.+)/s);
    if (m) {
      const label = m[1].trim();
      const body = m[2].trim();
      if (label.startsWith('角色·')) {
        const rest = label.slice(3);
        if (rest.includes('·')) {
          const dot = rest.indexOf('·');
          addLook(rest.slice(0, dot), rest.slice(dot + 1), body);
        } else {
          addLook(rest, defaultLabel, body);
        }
        continue;
      }
      if (STYLE_LABELS.has(label.toLowerCase()) || label === '画风' || label === '风格') {
        styleAnchor = body;
        continue;
      }
      addLook(label, defaultLabel, body);
      continue;
    }
    m = block.match(/^\[(Style|Look)\]\s*(.+)/is);
    if (m) {
      styleAnchor = m[2].trim();
      continue;
    }
    m = block.match(/^([^，,：:]{1,16})[，,：:]\s*(.+)/s);
    if (m) {
      const name = m[1].trim();
      const body = m[2].trim();
      if (STYLE_LEAD.has(name) || name === '画风' || name === '风格' || STYLE_LABELS.has(name.toLowerCase())) {
        styleAnchor = body;
      } else {
        addLook(name, defaultLabel, body);
      }
    }
  }

  const characters: LongVideoCharacter[] = [];
  for (const [name, looks] of byName) {
    const resolvedLooks =
      looks.length > 0
        ? looks
        : [{ id: makeCastStableId('look', `${name}|${defaultLabel}`), label: defaultLabel, body: '' }];
    const id = makeCastStableId('char', name);
    characters.push({
      id,
      name,
      looks: resolvedLooks,
      default_look_id: resolvedLooks[0].id,
    });
  }
  return { characters, styleAnchor };
}

export function hydrateCharacterRoster<
  T extends Pick<LongVideoProjectState, 'characters' | 'character_anchor' | 'style_anchor'>,
>(state: T, locale: 'zh' | 'en' = 'zh'): T {
  if ((state.characters?.length ?? 0) > 0) return state;
  const anchor = (state.character_anchor ?? '').trim();
  if (!anchor) return state;
  const parsed = parseCharacterRosterFromAnchor(anchor, locale);
  if (!parsed.characters.length) return state;
  const style = (state.style_anchor ?? '').trim() || parsed.styleAnchor;
  return {
    ...state,
    characters: parsed.characters,
    style_anchor: style,
    character_anchor: syncRosterToCharacterAnchor(parsed.characters, style),
  };
}

export function makeCastStableId(prefix: string, key: string): string {
  let h = 0;
  for (let i = 0; i < key.length; i += 1) {
    h = (Math.imul(31, h) + key.charCodeAt(i)) | 0;
  }
  return `${prefix}_${Math.abs(h).toString(16).padStart(8, '0')}`;
}

export function createCharacterEntry(name: string, locale: 'zh' | 'en' = 'zh'): LongVideoCharacter {
  const trimmed = name.trim();
  const id = makeCastStableId('char', trimmed);
  const label = locale === 'zh' ? '默认' : 'default';
  const lookId = makeCastStableId('look', `${trimmed}|${label}`);
  return {
    id,
    name: trimmed,
    default_look_id: lookId,
    looks: [{ id: lookId, label, body: '' }],
  };
}

export function createLookEntry(
  characterName: string,
  label: string,
  locale: 'zh' | 'en' = 'zh',
): LongVideoCharacterLook {
  const lbl = label.trim() || (locale === 'zh' ? '默认' : 'default');
  return {
    id: makeCastStableId('look', `${characterName}|${lbl}`),
    label: lbl,
    body: '',
  };
}

export function shotSceneText(shot: LongVideoShotState): string {
  return (shot.scene_prompt || extractKeyframeShotScene(shot.visual_prompt)).trim();
}

export { extractKeyframeShotScene, isStructuredKeyframeVisual };

export function defaultLongVideoProject(
  partial: Partial<LongVideoProjectState> = {},
): LongVideoProjectState {
  return {
    version: 1,
    strategy: 'segmented_i2v',
    title: '',
    brief: '',
    character_anchor: '',
    characters: [],
    style_anchor: '',
    target_duration_sec: 60,
    keyframe_model: partial.keyframe_model || 'z-image-turbo',
    segment_video_model: partial.segment_video_model || 'wan-2.2-i2v-14b',
    segment_duration_sec: 5,
    overlap_frames: 4,
    chain_mode: 'keyframe_only',
    shots: [],
    selection: null,
    ...partial,
  };
}

export function applyStoryboardShots(
  state: LongVideoProjectState,
  shots: LongVideoShotState[],
): LongVideoProjectState {
  const ordered = shots.map((s, i) => ({ ...s, order: i }));
  return {
    ...state,
    shots: ordered,
    selection: ordered.length ? { kind: 'node', index: 0 } : null,
  };
}

export function nextShotId(shots: LongVideoShotState[]): string {
  let max = -1;
  for (const s of shots) {
    const m = /^shot_(\d+)$/.exec(s.id);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return `shot_${String(max + 1).padStart(2, '0')}`;
}

export function createEmptyShot(order: number, id?: string): LongVideoShotState {
  const shotId = id || `shot_${String(order).padStart(2, '0')}`;
  return {
    id: shotId,
    order,
    visual_prompt: '',
    motion_prompt: '',
    status: 'draft',
  };
}

export function reorderShots(shots: LongVideoShotState[], fromIndex: number, toIndex: number): LongVideoShotState[] {
  if (fromIndex === toIndex || fromIndex < 0 || toIndex < 0) return shots;
  const next = [...shots];
  const [item] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, item);
  return next.map((s, i) => ({ ...s, order: i }));
}

/** Drop generated segment when edge params that affect I2V source change. */
export function invalidateSegmentAsset(shot: LongVideoShotState): LongVideoShotState {
  if (!shot.segment_asset_id && shot.status !== 'segment_ready') return shot;
  const next: LongVideoShotState = { ...shot, segment_asset_id: undefined };
  if (next.status === 'segment_ready') {
    next.status = next.keyframe_asset_id ? 'keyframe_ready' : 'draft';
  }
  return next;
}

export function insertKeyframeBefore(
  shots: LongVideoShotState[],
  index: number,
  id?: string,
): { shots: LongVideoShotState[]; newIndex: number } {
  if (index < 0 || index > shots.length) {
    throw new RangeError(`insertKeyframeBefore: index ${index} out of range`);
  }
  const newShot = createEmptyShot(index, id ?? nextShotId(shots));
  const next = [...shots];
  next.splice(index, 0, newShot);
  const reordered = next.map((s, i) => ({ ...s, order: i }));
  if (index > 0) {
    reordered[index - 1] = invalidateSegmentAsset(reordered[index - 1]);
  }
  return { shots: reordered, newIndex: index };
}

export function insertKeyframeAfter(
  shots: LongVideoShotState[],
  index: number,
  id?: string,
): { shots: LongVideoShotState[]; newIndex: number } {
  if (index < 0 || index >= shots.length) {
    throw new RangeError(`insertKeyframeAfter: index ${index} out of range`);
  }
  const insertAt = index + 1;
  const newShot = createEmptyShot(insertAt, id ?? nextShotId(shots));
  const next = [...shots];
  next.splice(insertAt, 0, newShot);
  const reordered = next.map((s, i) => ({ ...s, order: i }));
  reordered[index] = invalidateSegmentAsset(reordered[index]);
  return { shots: reordered, newIndex: insertAt };
}

export function removeKeyframeAt(
  shots: LongVideoShotState[],
  index: number,
): { shots: LongVideoShotState[]; selection: LongVideoSelection } | null {
  if (index < 0 || index >= shots.length) return null;
  if (shots.length <= MIN_LONG_VIDEO_KEYFRAMES) return null;

  const next = shots.filter((_, i) => i !== index).map((s, i) => ({ ...s, order: i }));

  if (index > 0 && index < shots.length - 1) {
    next[index - 1] = invalidateSegmentAsset(next[index - 1]!);
  } else if (index === shots.length - 1 && index > 0) {
    next[index - 1] = {
      ...invalidateSegmentAsset(next[index - 1]!),
      motion_prompt: '',
    };
  }

  let selection: LongVideoSelection;
  if (index > 0 && index < shots.length - 1) {
    selection = { kind: 'edge', index: index - 1 };
  } else if (index === 0) {
    selection = { kind: 'node', index: 0 };
  } else {
    selection =
      next.length >= MIN_LONG_VIDEO_KEYFRAMES
        ? { kind: 'edge', index: next.length - 2 }
        : { kind: 'node', index: Math.max(0, next.length - 1) };
  }

  return { shots: next, selection };
}

/** Whether deleting this keyframe should show a confirmation dialog. */
export function removeKeyframeNeedsConfirm(shots: LongVideoShotState[], index: number): boolean {
  const shot = shots[index];
  if (!shot) return false;
  if (shot.keyframe_asset_id) return true;
  if (shot.segment_asset_id || shot.motion_prompt?.trim()) return true;
  if (index > 0 && (shots[index - 1]?.segment_asset_id || shots[index - 1]?.motion_prompt?.trim())) {
    return true;
  }
  return false;
}

export function shotAssetPath(assetId: string | undefined): string {
  return assetId ? `asset:${assetId}` : '';
}

function alignNumFrames(frames: number, schema?: { min?: number; max?: number; step?: number }) {
  const min = schema?.min ?? 1;
  const max = schema?.max ?? 129;
  const step = schema?.step ?? 1;
  let n = Math.round(frames);
  if (step > 1) {
    n = Math.round((n - 1) / step) * step + 1;
  }
  return Math.min(max, Math.max(min, n));
}

/** Wan-style 4n+1 frame count from segment duration and model ``num_frames`` schema. */
export function numFramesForDurationSec(
  durationSec: number,
  fps: number,
  schema?: { min?: number; max?: number; step?: number },
): number {
  const rate = Math.max(1, Number(fps) || 1);
  const sec = Math.max(0, Number(durationSec) || 0);
  return alignNumFrames(sec * rate + 1, schema);
}

export function keyframeThumbnailUrl(assetId: string | undefined, getUrl: (path: string) => string): string {
  if (!assetId) return '';
  return getUrl(shotAssetPath(assetId));
}

/** Strip client-only fields before persisting to the long-video project API. */
export function projectStateForServer(
  lv: LongVideoProjectState,
): Omit<LongVideoProjectState, 'selection' | 'project_id'> {
  const { selection, project_id, ...rest } = lv;
  void selection;
  void project_id;
  return rest;
}
