import type {
  KeyframeT2iProvenance,
  KeyframeT2iProvenanceSkipReason,
  LongVideoBeatGroup,
  LongVideoChainMode,
  LongVideoCharacter,
  LongVideoCharacterLook,
  LongVideoChapterAnalysis,
  LongVideoFlfMode,
  LongVideoProjectState,
  LongVideoScene,
  LongVideoSceneLook,
  LongVideoSegmentRole,
  LongVideoSelection,
  LongVideoShotCastLook,
  LongVideoShotSceneLook,
  LongVideoShotState,
} from '@/types';
import { parseCharacterLookBody, parseSceneBeat } from '@/utils/longVideoSceneBeat';

export function resolveLongVideoLocale(locale: string): 'zh' | 'en' {
  return locale.startsWith('zh') ? 'zh' : 'en';
}

export function buildCastVisionBackfillQuestion(
  ch: LongVideoCharacter,
  otherCharacterNames: string[],
  locale: 'zh' | 'en',
): string {
  const others = otherCharacterNames.map((n) => n.trim()).filter(Boolean);
  if (locale === 'zh') {
    return (
      `只描述角色「${ch.name}」的外貌与服饰（发型、脸型、服装、体型），忽略图片中的其他人物。` +
      `不要写剧情或背景故事。用 2-3 句简体中文回答。` +
      (others.length ? `不要描述：${others.join('、')}。` : '')
    );
  }
  return (
    `Describe ONLY the character "${ch.name}" appearance and outfit (hair, face, clothes, body). ` +
    `Ignore other people in the image. No plot. Answer in 2-3 sentences in English.` +
    (others.length ? ` Do not describe: ${others.join(', ')}.` : '')
  );
}

export function buildSceneVisionBackfillQuestion(
  sc: LongVideoScene,
  look: LongVideoSceneLook,
  locale: 'zh' | 'en',
): string {
  if (locale === 'zh') {
    return (
      `只描述图片中的环境布景：空间结构、光线、氛围、关键道具与色调，不要描述人物。` +
      `用 2-4 句简体中文回答。地点：${sc.name}（${look.label}）`
    );
  }
  return (
    `Describe ONLY the environment/set: spatial layout, lighting, mood, key props, and palette. ` +
    `No people. Answer in 2-4 sentences in English. Location: ${sc.name} (${look.label})`
  );
}

export function buildKeyframeConsistencyQuestion(locale: 'zh' | 'en'): string {
  if (locale === 'zh') {
    return (
      '对比定妆参考图与分镜图：角色面部、发型、服饰是否一致？' +
      '若明显不一致，用一句简体中文说明差异；若基本一致，只回复「一致」。'
    );
  }
  return (
    'Compare the portrait reference and storyboard frame: face, hair, outfit consistency. ' +
    'If mismatch, one sentence in English on the gap; if OK, reply "consistent" only.'
  );
}

/** Vertical reference size for cast portraits (avoid landscape group-shot bias). */
export const PORTRAIT_REFERENCE_SIZE = { width: 768, height: 1024 } as const;

/** Landscape reference size for scene / set-piece images. */
export const SCENE_REFERENCE_SIZE = { width: 1280, height: 704 } as const;

export const SCENE_ENV_NEGATIVE_ZH =
  '人物特写, 肖像, 大头照, 文字, 水印, 畸形, 低质量';
export const SCENE_ENV_NEGATIVE_EN =
  'portrait, headshot, close-up face, text, watermark, deformed, low quality';

export const PORTRAIT_NEGATIVE_PROMPT_ZH =
  '多人，群像，合影，两个，三个，背景人物，路人，分身，重复角色';
export const PORTRAIT_NEGATIVE_PROMPT_EN =
  'multiple people, group photo, crowd, duo, trio, background people, duplicate character';

/** Minimum keyframes required to form at least one I2V segment. */
export const MIN_LONG_VIDEO_KEYFRAMES = 2;

/** Default I2V segment length when a shot has no ``duration_sec``. */
export const DEFAULT_SHOT_DURATION_SEC = 5;

export interface StoryboardReadiness {
  total: number;
  keyframeCount: number;
  motionCount: number;
  motionTotal: number;
  segmentCount: number;
  mergeReady: boolean;
}

export function computeStoryboardReadiness(shots: LongVideoShotState[]): StoryboardReadiness {
  const total = shots.length;
  const keyframeCount = shots.filter((s) => Boolean(s.keyframe_asset_id)).length;
  const motionCount = shots.filter((s) => Boolean(s.motion_prompt?.trim())).length;
  const segmentCount = shots.filter(
    (s) => Boolean(s.segment_asset_id) && s.status === 'segment_ready',
  ).length;
  const mergeReady =
    total > 0 && shots.every((s) => s.status === 'segment_ready' && Boolean(s.segment_asset_id));
  return { total, keyframeCount, motionCount, motionTotal: total, segmentCount, mergeReady };
}

export function shotDurationSec(shot: LongVideoShotState | undefined, fallback = DEFAULT_SHOT_DURATION_SEC): number {
  const sec = shot?.duration_sec;
  return typeof sec === 'number' && sec > 0 ? sec : fallback;
}

const ACTION_BEAT_RE = /追逐|大战|决斗|战斗|打斗|交锋|对决|厮杀|爆炸|chase|fight|battle|duel|combat|explosion/i;
const ESTABLISHING_BEAT_RE = /远景|全景|建立|航拍|俯瞰|空镜|establishing|wide shot|aerial/i;

function beatDurationWeight(beat: string): number {
  const text = beat.trim();
  let weight = Math.max(1, text.length ** 0.45);
  if (ACTION_BEAT_RE.test(text)) weight *= 1.35;
  if (ESTABLISHING_BEAT_RE.test(text)) weight *= 0.85;
  return weight;
}

/** Soft-budget per-shot I2V durations (sec). Sum may differ from target. */
export function allocateShotDurations(options: {
  sceneCount: number;
  targetDurationSec: number;
  defaultSegmentSec?: number;
  beatTexts?: string[];
  minSec?: number;
  maxSec?: number;
}): number[] {
  const count = Math.max(1, options.sceneCount);
  const lo = Math.max(0.5, options.minSec ?? 2);
  const hi = Math.max(lo, options.maxSec ?? 12);
  const defaultSeg = Math.max(lo, Math.min(hi, options.defaultSegmentSec ?? DEFAULT_SHOT_DURATION_SEC));
  const target = Math.max(defaultSeg, options.targetDurationSec);
  const beatTexts = options.beatTexts ?? [];
  const weights = Array.from({ length: count }, (_, i) =>
    beatTexts[i] ? beatDurationWeight(beatTexts[i]) : 1,
  );
  const totalWeight = weights.reduce((sum, w) => sum + w, 0) || count;
  return weights.map((w) => {
    const raw = target * (w / totalWeight);
    const rounded = Math.max(lo, Math.min(hi, Math.round(raw)));
    return rounded >= 1 ? rounded : Math.round(raw * 2) / 2;
  });
}

export function totalShotDurationSec(shots: LongVideoShotState[], fallback = DEFAULT_SHOT_DURATION_SEC): number {
  if (!shots.length) return 0;
  return shots.reduce((sum, shot) => sum + shotDurationSec(shot, fallback), 0);
}

export function effectiveShotChainMode(
  shot: LongVideoShotState | undefined,
  defaultMode: LongVideoChainMode,
): LongVideoChainMode {
  const mode = shot?.chain_mode;
  if (mode === 'first_last' || shot?.flf_mode === 'first_last') {
    return 'keyframe_only';
  }
  if (mode === 'keyframe_only' || mode === 'last_frame' || mode === 'reference_r2v') {
    return mode;
  }
  if (shot?.start_frame_mode === 'prev_segment_tail' || shot?.flf_mode === 'continuation') {
    return 'last_frame';
  }
  return defaultMode;
}

export function findFaceAnchorShot(
  shots: LongVideoShotState[],
  shot: LongVideoShotState | undefined,
): LongVideoShotState | undefined {
  if (!shot) return undefined;
  if (shot.segment_role === 'face_anchor') return shot;
  const anchorId = shot.face_anchor_shot_id?.trim();
  if (anchorId) {
    const byId = shots.find((s) => s.id === anchorId);
    if (byId) return byId;
  }
  const groupId = shot.segment_group_id;
  if (groupId) {
    return shots.find((s) => s.segment_group_id === groupId && s.segment_role === 'face_anchor');
  }
  return undefined;
}

export function groupShotsByBeat(shots: LongVideoShotState[]): LongVideoBeatGroup[] {
  const order: string[] = [];
  const map = new Map<string, LongVideoBeatGroup>();
  shots.forEach((shot, index) => {
    const groupId = shot.segment_group_id || `shot_${index}`;
    let group = map.get(groupId);
    if (!group) {
      group = {
        groupId,
        beatIndex:
          typeof shot.narrative_beat_index === 'number' && shot.narrative_beat_index >= 0
            ? shot.narrative_beat_index
            : order.length,
        title: (shot.scene_prompt || shot.visual_prompt || '').trim().slice(0, 40) || groupId,
        shotIndices: [],
      };
      map.set(groupId, group);
      order.push(groupId);
    }
    group.shotIndices.push(index);
  });
  return order.map((id) => map.get(id)!);
}

export function beatGroupProgress(
  shots: LongVideoShotState[],
  group: LongVideoBeatGroup,
): 'needs_anchor' | 'anchor_ready' | 'group_ready' {
  const groupShots = group.shotIndices.map((i) => shots[i]).filter(Boolean);
  const anchor = groupShots.find((s) => s.segment_role === 'face_anchor');
  if (anchor && !anchor.keyframe_asset_id) return 'needs_anchor';
  const needsKeyframe = groupShots.filter(
    (s) =>
      s.segment_role !== 'tail_continuation' &&
      s.start_frame_mode !== 'anchor_link' &&
      s.segment_role !== 'face_anchor',
  );
  if (needsKeyframe.some((s) => !s.keyframe_asset_id && s.segment_role !== 'post_anchor')) {
    if (anchor && anchor.keyframe_asset_id) return 'anchor_ready';
    return 'needs_anchor';
  }
  if (groupShots.some((s) => !s.segment_asset_id && shotVideoPrompt(s))) return 'anchor_ready';
  return 'group_ready';
}

export interface GroupGenerationPlan {
  keyframeIndices: number[];
  segmentIndices: number[];
}

/** DAG plan for one beat group: keyframes first, then segments in timeline order. */
export function planGroupGeneration(
  lv: { shots: LongVideoShotState[]; chain_mode?: LongVideoChainMode; characters?: import('@/types').LongVideoCharacter[] },
  group: LongVideoBeatGroup,
): GroupGenerationPlan {
  const keyframeIndices: number[] = [];
  const segmentIndices: number[] = [];
  for (const idx of group.shotIndices) {
    const shot = lv.shots[idx];
    if (!shot) continue;
    if (shotNeedsKeyframe(shot) && !shot.keyframe_asset_id) {
      keyframeIndices.push(idx);
    }
  }
  for (const idx of group.shotIndices) {
    const shot = lv.shots[idx];
    if (!shot || !shotVideoPrompt(shot) || shot.segment_asset_id) continue;
    if (canGenerateSegmentShot(lv, idx)) segmentIndices.push(idx);
  }
  return { keyframeIndices, segmentIndices };
}

export function allPendingAnchorKeyframeIndices(shots: LongVideoShotState[]): number[] {
  return shots
    .map((s, i) => (s.segment_role === 'face_anchor' && !s.keyframe_asset_id ? i : -1))
    .filter((i) => i >= 0);
}

export function allPendingSegmentIndices(
  lv: { shots: LongVideoShotState[]; chain_mode?: LongVideoChainMode; characters?: LongVideoCharacter[] },
): number[] {
  const groups = groupShotsByBeat(lv.shots);
  const out: number[] = [];
  for (const group of groups) {
    const st = beatGroupProgress(lv.shots, group);
    if (st === 'needs_anchor') continue;
    for (const idx of group.shotIndices) {
      const shot = lv.shots[idx];
      if (!shot || !shotVideoPrompt(shot) || shot.segment_asset_id) continue;
      if (canGenerateSegmentShot(lv, idx)) out.push(idx);
    }
  }
  return out.sort((a, b) => a - b);
}

export function groupHasFaceAnchor(shots: LongVideoShotState[], groupId: string): boolean {
  return shots.some((s) => s.segment_group_id === groupId && s.segment_role === 'face_anchor');
}

function isWideShotSize(shotSize: string): boolean {
  const s = shotSize.toLowerCase();
  return /远景|广角|wide|full|establishing|long/.test(s) || s.includes('远');
}

function anchorShotIdForGroup(groupId: string): string {
  const slug = groupId.replace(/[^a-zA-Z0-9_]/g, '_');
  return `anchor_${slug}_face`;
}

/** Rule-based pre / face_anchor / post split (mirrors backend _rule_split_beat). */
export function buildAnchorSplitShotsFromSource(
  source: LongVideoShotState,
  opts: { groupId: string; beatIndex: number; reachability?: 'identity_critical' | 'action_wide' },
): LongVideoShotState[] {
  const beatDur = shotDurationSec(source);
  const visual = (source.start_visual_prompt || source.visual_prompt || '').trim();
  const shotSize = /远景|广角|wide/i.test(visual) ? '远景' : '中景';
  const reach = opts.reachability ?? 'identity_critical';
  const groupId = opts.groupId;
  const anchorId = anchorShotIdForGroup(groupId);

  const anchorDur = Math.min(3, Math.max(1.5, beatDur * 0.25));
  const remain = Math.max(2, beatDur - anchorDur);
  const preDur =
    isWideShotSize(shotSize) || reach === 'action_wide'
      ? Math.round(remain * 0.45 * 10) / 10
      : 0;
  const postDur = Math.round((remain - preDur) * 10) / 10;

  const shared = {
    scene_prompt: source.scene_prompt,
    cast_looks: source.cast_looks ? [...source.cast_looks] : undefined,
    scene_look: source.scene_look,
    video_prompt: source.video_prompt,
    motion_prompt: source.motion_prompt,
  };

  const parts: LongVideoShotState[] = [];
  let partIdx = 0;

  if (preDur >= 2) {
    parts.push({
      ...createEmptyShot(partIdx, `${groupId}_pre`),
      ...shared,
      segment_role: 'pre_anchor',
      segment_group_id: groupId,
      segment_group_index: partIdx++,
      face_anchor_shot_id: anchorId,
      flf_mode: 'none',
      duration_sec: preDur,
      visual_prompt: visual,
      start_visual_prompt: visual,
      end_visual_prompt: source.end_visual_prompt || '',
    });
  }

  parts.push({
    ...createEmptyShot(partIdx, anchorId),
    ...shared,
    segment_role: 'face_anchor',
    segment_group_id: groupId,
    segment_group_index: partIdx++,
    face_anchor_shot_id: anchorId,
    flf_mode: 'none',
    duration_sec: anchorDur,
    visual_prompt: source.anchor_visual_prompt || visual,
    anchor_visual_prompt: source.anchor_visual_prompt || `【特写】${visual}`,
    start_visual_prompt: source.anchor_visual_prompt || `【特写】${visual}`,
  });

  if (postDur >= 2) {
    parts.push({
      ...createEmptyShot(partIdx, `${groupId}_post`),
      ...shared,
      segment_role: 'post_anchor',
      segment_group_id: groupId,
      segment_group_index: partIdx++,
      face_anchor_shot_id: anchorId,
      start_frame_mode: 'anchor_link',
      flf_mode: 'none',
      chain_mode: 'keyframe_only',
      duration_sec: postDur,
      visual_prompt: visual,
      start_visual_prompt: visual,
    });
  }

  if (!parts.length) {
    parts.push({
      ...createEmptyShot(0, anchorId),
      ...shared,
      segment_role: 'face_anchor',
      segment_group_id: groupId,
      segment_group_index: 0,
      face_anchor_shot_id: anchorId,
      duration_sec: beatDur,
      visual_prompt: visual,
      anchor_visual_prompt: `【特写】${visual}`,
    });
  }

  return parts;
}

/** Insert face_anchor sub-segments into a group that has none (establishing / keyframe). */
export function insertFaceAnchorIntoGroup(
  shots: LongVideoShotState[],
  groupId: string,
): LongVideoShotState[] {
  if (groupHasFaceAnchor(shots, groupId)) return shots;
  const groups = groupShotsByBeat(shots);
  const group = groups.find((g) => g.groupId === groupId);
  if (!group?.shotIndices.length) return shots;

  const groupShots = group.shotIndices.map((i) => shots[i]!);
  const combinedDur = groupShots.reduce((sum, s) => sum + shotDurationSec(s), 0);
  const template: LongVideoShotState = {
    ...groupShots[0]!,
    duration_sec: combinedDur,
    keyframe_asset_id: undefined,
    segment_asset_id: undefined,
    end_frame_asset_id: undefined,
    status: 'draft',
  };
  const newParts = buildAnchorSplitShotsFromSource(template, {
    groupId,
    beatIndex: group.beatIndex,
  });
  const next = [...shots];
  next.splice(group.shotIndices[0]!, group.shotIndices.length, ...newParts);
  return next.map((s, i) => ({ ...s, order: i }));
}

/** Rule resplit: replace an entire beat group with fresh anchor-aware segments. */
export function resplitBeatGroupRule(
  shots: LongVideoShotState[],
  groupId: string,
): LongVideoShotState[] {
  const groups = groupShotsByBeat(shots);
  const group = groups.find((g) => g.groupId === groupId);
  if (!group?.shotIndices.length) return shots;

  const groupShots = group.shotIndices.map((i) => shots[i]!);
  const combinedDur = groupShots.reduce((sum, s) => sum + shotDurationSec(s), 0);
  const template: LongVideoShotState = {
    ...groupShots[0]!,
    duration_sec: combinedDur,
    keyframe_asset_id: undefined,
    segment_asset_id: undefined,
    end_frame_asset_id: undefined,
    status: 'draft',
  };
  const newParts = buildAnchorSplitShotsFromSource(template, {
    groupId,
    beatIndex: group.beatIndex,
  });
  const next = [...shots];
  next.splice(group.shotIndices[0]!, group.shotIndices.length, ...newParts);
  return next.map((s, i) => ({ ...s, order: i }));
}

export function selectedBeatGroupId(
  shots: LongVideoShotState[],
  selection: LongVideoSelection | null | undefined,
): string | null {
  if (!selection) return null;
  if (selection.kind === 'beat_group') return selection.groupId;
  if (selection.kind === 'segment' || selection.kind === 'clip') {
    return shots[selection.index]?.segment_group_id ?? null;
  }
  return null;
}

export function segmentRoleLabelKey(role: LongVideoSegmentRole): string {
  return `video.longVideoRole${role
    .split('_')
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join('')}`;
}

/** T2I source text by segment role. */
export function shotKeyframeText(shot: LongVideoShotState | undefined): string {
  if (!shot) return '';
  if (shot.segment_role === 'face_anchor') {
    return (shot.anchor_visual_prompt || shot.start_visual_prompt || shot.visual_prompt || '').trim();
  }
  return (shot.start_visual_prompt || shot.visual_prompt || '').trim();
}

export function shotEndFrameText(shot: LongVideoShotState | undefined): string {
  if (!shot) return '';
  return (shot.end_visual_prompt || '').trim();
}

/** I2V clip prompt (video-first pipeline). */
export function shotVideoPrompt(shot: LongVideoShotState | undefined): string {
  if (!shot) return '';
  const video = (shot.video_prompt || '').trim();
  const motion = (shot.motion_prompt || '').trim();
  if (video && motion && video !== motion) return motion;
  return video || motion || (shot.visual_prompt || '').trim();
}

/** Human-readable I2V submit summary for inspector preview. */
export function segmentVideoSubmitPreview(
  shot: LongVideoShotState,
  shots: LongVideoShotState[],
  opts: {
    chainMode?: LongVideoChainMode;
    locale?: 'zh' | 'en';
    shotIndex?: number;
  },
): string {
  const locale = opts.locale ?? (/[\u4e00-\u9fff]/.test(shotVideoPrompt(shot)) ? 'zh' : 'en');
  const prompt = shotVideoPrompt(shot);
  const idx = opts.shotIndex ?? shots.findIndex((s) => s.id === shot.id);
  const srcId =
    idx >= 0
      ? segmentI2vSourceAssetId({ shots, chain_mode: opts.chainMode }, idx)
      : shot.keyframe_asset_id;
  const chain = effectiveShotChainMode(shot, opts.chainMode ?? 'keyframe_only');
  const lines: string[] = [];
  if (prompt) {
    const heading = locale === 'zh' ? '## 片段运动' : '## Clip motion';
    lines.push(`${heading}\n\n${prompt}`);
  }
  const meta: string[] = [];
  const chainLabel =
    locale === 'zh'
      ? { keyframe_only: '关键帧衔接', last_frame: '尾帧衔接', reference_r2v: '角色参考 R2V', first_last: '首尾帧' }[
          chain
        ] ?? chain
      : chain;
  meta.push(locale === 'zh' ? `衔接：${chainLabel}` : `Chain: ${chainLabel}`);
  if (srcId) {
    meta.push(locale === 'zh' ? `首帧源：asset:${srcId}` : `Start frame: asset:${srcId}`);
  } else if (shot.start_frame_mode === 'anchor_link') {
    meta.push(locale === 'zh' ? '首帧源：组内人脸锚点' : 'Start frame: face anchor in group');
  }
  if (meta.length) {
    const note = locale === 'zh' ? '## 提交说明' : '## Submit notes';
    lines.push(`${note}\n\n${meta.map((m) => `- ${m}`).join('\n')}`);
  }
  return lines.join('\n\n').trim();
}

export function shotNeedsKeyframe(shot: LongVideoShotState | undefined): boolean {
  if (!shot) return true;
  if (shot.segment_role === 'tail_continuation') return false;
  if (shot.start_frame_mode === 'prev_segment_tail') return false;
  if (shot.start_frame_mode === 'anchor_link') return false;
  return true;
}

export function shotNeedsStartFrame(shot: LongVideoShotState | undefined): boolean {
  if (!shot) return false;
  return shot.segment_role === 'pre_anchor' || shot.segment_role === 'establishing' || shot.segment_role === 'keyframe';
}

export function shotNeedsEndFrame(shot: LongVideoShotState | undefined): boolean {
  if (!shot) return false;
  return shot.segment_role === 'pre_anchor' && shot.flf_mode === 'first_last';
}

/** I2V submit placeholder: last_frame chain replaces source on the server from prev segment. */
export function segmentI2vSourceAssetId(
  lv: { shots: LongVideoShotState[]; chain_mode?: LongVideoChainMode },
  index: number,
): string | undefined {
  const shot = lv.shots[index];
  if (!shot) return undefined;
  const chainMode = effectiveShotChainMode(shot, lv.chain_mode ?? 'keyframe_only');
  if (chainMode === 'last_frame' && index > 0 && lv.shots[index - 1]?.segment_asset_id) {
    return (
      lv.shots[index - 1]?.keyframe_asset_id ||
      lv.shots[0]?.keyframe_asset_id ||
      shot.keyframe_asset_id
    );
  }
  if (shot.start_frame_mode === 'anchor_link') {
    const anchor = findFaceAnchorShot(lv.shots, shot);
    return anchor?.keyframe_asset_id || shot.keyframe_asset_id;
  }
  return shot.keyframe_asset_id;
}

export function segmentI2vEndFrameAssetId(
  lv: { shots: LongVideoShotState[] },
  index: number,
): string | undefined {
  const shot = lv.shots[index];
  if (!shot) return undefined;
  if (effectiveShotChainMode(shot, 'keyframe_only') !== 'first_last') return undefined;
  if (shot.end_frame_sync_anchor) {
    return findFaceAnchorShot(lv.shots, shot)?.keyframe_asset_id;
  }
  return shot.end_frame_asset_id;
}

export function canGenerateSegmentShot(
  lv: { shots: LongVideoShotState[]; chain_mode?: LongVideoChainMode; characters?: LongVideoCharacter[] },
  index: number,
): boolean {
  const shot = lv.shots[index];
  if (!shot || !shotVideoPrompt(shot)) return false;
  const chainMode = effectiveShotChainMode(shot, lv.chain_mode ?? 'keyframe_only');
  if (chainMode === 'last_frame' && index > 0) {
    return Boolean(lv.shots[index - 1]?.segment_asset_id);
  }
  if (shot.start_frame_mode === 'anchor_link') {
    return Boolean(findFaceAnchorShot(lv.shots, shot)?.keyframe_asset_id);
  }
  if (chainMode === 'reference_r2v') {
    if (!shot.keyframe_asset_id) return false;
    const refs = collectCastReferenceAssetIdsForShot(
      lv.characters ?? [],
      shot.cast_looks ?? [],
      shotCastMatchText(shot),
    );
    return refs.length > 0;
  }
  return Boolean(shot.keyframe_asset_id);
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

/** Markdown section headings for T2I keyframe prompts (generation-time only). */
export const KEYFRAME_MD_SCENE_ZH = '场景';
export const KEYFRAME_MD_SCENE_EN = 'Scene';
export const KEYFRAME_MD_CAST_ZH = '角色定妆';
export const KEYFRAME_MD_CAST_EN = 'Cast reference';
export const KEYFRAME_MD_SCENE_SET_ZH = '场景设定';
export const KEYFRAME_MD_SCENE_SET_EN = 'Set reference';
export const KEYFRAME_MD_STYLE_ZH = '画风';
export const KEYFRAME_MD_STYLE_EN = 'Style';

const KEYFRAME_MD_SCENE_HEADINGS = [KEYFRAME_MD_SCENE_ZH, KEYFRAME_MD_SCENE_EN];
const KEYFRAME_MD_CAST_HEADINGS = [KEYFRAME_MD_CAST_ZH, KEYFRAME_MD_CAST_EN];
const KEYFRAME_MD_STYLE_HEADINGS = [KEYFRAME_MD_STYLE_ZH, KEYFRAME_MD_STYLE_EN];

function extractMarkdownSection(text: string, headings: string[]): string {
  const lines = text.split('\n');
  let inSection = false;
  const buf: string[] = [];
  for (const line of lines) {
    const h2 = line.match(/^##\s+(.+?)\s*$/);
    if (h2) {
      const title = h2[1].trim();
      if (inSection) break;
      if (headings.some((h) => title === h || title.startsWith(h))) {
        inSection = true;
        continue;
      }
    }
    if (inSection) buf.push(line);
  }
  return buf.join('\n').trim();
}

function stripMarkdownQuoteBlock(text: string): string {
  return text
    .split('\n')
    .filter((line) => !line.trim().startsWith('>'))
    .join('\n')
    .trim();
}

function normalizeSceneLine(rawScene: string, locale: 'zh' | 'en'): string {
  const s = rawScene.trim();
  if (!s) return s;
  const shotLabel = locale === 'zh' ? '【本帧】' : '[Shot] ';
  if (s.startsWith('【本帧】') || /^\[Shot\]/i.test(s)) return s;
  if (/^【(?:特写|近景|远景|全景|中景|大特写)/.test(s)) return s;
  return `${shotLabel}${s}`;
}

function keyframeSceneComposeNote(locale: 'zh' | 'en', scope: CastReferenceScope): string {
  if (scope === 'wardrobe') {
    return locale === 'zh'
      ? '按本镜场景构图与动作；仅服饰/配色与定妆一致，本帧不要求清晰面部。'
      : 'Compose this shot per scene framing and action; match outfit palette only—no readable face in frame.';
  }
  if (scope === 'none') {
    return locale === 'zh'
      ? '按本镜场景构图与环境；无人物或人物不可见。'
      : 'Compose environment per scene; no visible characters.';
  }
  return locale === 'zh'
    ? '按本镜场景构图、景别与动作，环境背景完整；仅面部/服饰与角色定妆一致，勿复制定妆图的构图或纯色背景。'
    : 'Compose this shot with full scene, framing, and action; match cast face/outfit only—do not copy portrait framing or plain backdrop.';
}

function keyframeCastSectionNote(locale: 'zh' | 'en', scope: CastReferenceScope): string {
  if (scope === 'wardrobe') {
    return locale === 'zh'
      ? '仅约束服饰与配色；构图、可见身体部位与环境以「场景」章节为准，勿添加清晰五官。'
      : 'Outfit and palette only—visible body parts and environment follow the Scene section; no added facial detail.';
  }
  return locale === 'zh'
    ? '仅约束角色外貌与服饰；场景构图、景别、环境与其他人物以「场景」章节为准。'
    : 'Outfit and appearance only—scene framing, environment, and other characters follow the Scene section.';
}

function buildKeyframeMarkdownCastSection(
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  sceneText: string,
  _styleAnchor: string,
  locale: 'zh' | 'en',
  scope: CastReferenceScope = 'face',
): string {
  if (scope === 'none') return '';
  const castChars = charactersForShotCast(characters, castLooks, sceneText);
  if (!castChars.length) return '';

  const otherNames = castChars.map((c) => c.name.trim()).filter(Boolean);
  const castMap = new Map(castLooks.map((c) => [c.character_id, c.look_id]));
  const blocks: string[] = [];

  for (const ch of castChars) {
    const lookId = castMap.get(ch.id) || ch.default_look_id;
    const lk = ch.looks.find((l) => l.id === lookId) || ch.looks[0];
    if (!lk) continue;
    const outfitParts = characterLookOutfitParts(
      ch,
      lk,
      otherNames.filter((n) => n !== ch.name.trim()),
    );
    const heading = locale === 'zh' ? `### ${ch.name} · ${lk.label}` : `### ${ch.name} · ${lk.label}`;
    blocks.push(`${heading}\n\n${formatCastOutfitMarkdown(outfitParts, locale, scope)}`);
  }

  const castHeading = locale === 'zh' ? KEYFRAME_MD_CAST_ZH : KEYFRAME_MD_CAST_EN;
  return `## ${castHeading}\n\n> ${keyframeCastSectionNote(locale, scope)}\n\n${blocks.join('\n\n')}`;
}

function buildKeyframeMarkdownFromAnchorBlocks(
  blocks: AnchorBlock[],
  locale: 'zh' | 'en',
): string {
  const castLines: string[] = [];
  let styleBody = '';
  for (const b of blocks) {
    if (b.kind === 'style') {
      styleBody = b.body.trim();
      continue;
    }
    if (b.kind === 'character' && b.name) {
      const heading = locale === 'zh' ? `### ${b.name}` : `### ${b.name}`;
      const outfitLabel = locale === 'zh' ? '装扮' : 'Outfit';
      castLines.push(`${heading}\n\n- **${outfitLabel}**：${b.body.trim()}`);
    } else if (b.body.trim()) {
      castLines.push(`- ${b.body.trim()}`);
    }
  }
  const parts: string[] = [];
  if (castLines.length) {
    const castHeading = locale === 'zh' ? KEYFRAME_MD_CAST_ZH : KEYFRAME_MD_CAST_EN;
    parts.push(`## ${castHeading}\n\n> ${keyframeCastSectionNote(locale)}\n\n${castLines.join('\n\n')}`);
  }
  if (styleBody) {
    const styleHeading = locale === 'zh' ? KEYFRAME_MD_STYLE_ZH : KEYFRAME_MD_STYLE_EN;
    parts.push(`## ${styleHeading}\n\n${styleBody}`);
  }
  return parts.join('\n\n');
}

function keyframeSceneSetSectionNote(locale: 'zh' | 'en'): string {
  return locale === 'zh'
    ? '环境布景与光线参考；人物构图与动作以「场景」章节为准。'
    : 'Environment and lighting reference—character blocking follows the Scene section.';
}

function buildKeyframeMarkdownSceneSetSection(
  scenes: LongVideoScene[],
  sceneLook: LongVideoShotSceneLook | undefined,
  beatText: string,
  locale: 'zh' | 'en',
): string {
  if (!scenes.length) return '';
  const binding = resolveShotSceneLook(scenes, sceneLook, beatText);
  if (!binding?.scene_id) return '';
  const sc = scenes.find((s) => s.id === binding.scene_id);
  if (!sc) return '';
  const lk =
    sc.looks.find((l) => l.id === binding.look_id) ||
    sc.looks.find((l) => l.id === sc.default_look_id) ||
    sc.looks[0];
  if (!lk) return '';
  const body = sceneEnvironmentForPrompt(lk);
  if (!body.trim()) return '';
  if (textAlreadyCovered(beatText, body)) return '';
  const heading = locale === 'zh' ? `### ${sc.name} · ${lk.label}` : `### ${sc.name} · ${lk.label}`;
  const envLabel = locale === 'zh' ? '环境' : 'Environment';
  const setHeading = locale === 'zh' ? KEYFRAME_MD_SCENE_SET_ZH : KEYFRAME_MD_SCENE_SET_EN;
  return `## ${setHeading}\n\n> ${keyframeSceneSetSectionNote(locale)}\n\n${heading}\n\n- **${envLabel}**：${body}`;
}

function assembleKeyframeMarkdownPrompt(
  sceneBlock: string,
  sceneSetSection: string,
  castSection: string,
  styleAnchor: string,
  locale: 'zh' | 'en',
): string {
  const parts: string[] = [];
  const sceneHeading = locale === 'zh' ? KEYFRAME_MD_SCENE_ZH : KEYFRAME_MD_SCENE_EN;
  if (sceneBlock.trim()) {
    parts.push(`## ${sceneHeading}\n\n${sceneBlock.trim()}`);
  }
  if (sceneSetSection.trim()) {
    parts.push(sceneSetSection.trim());
  }
  if (castSection.trim()) {
    parts.push(castSection.trim());
  }
  const style = styleAnchor.trim();
  const styleHeadingKey = locale === 'zh' ? KEYFRAME_MD_STYLE_ZH : KEYFRAME_MD_STYLE_EN;
  const hasStyleInSections = [sceneSetSection, castSection].some((s) =>
    s.includes(`## ${styleHeadingKey}`),
  );
  if (style && !hasStyleInSections) {
    parts.push(`## ${styleHeadingKey}\n\n${style}`);
  }
  return parts.join('\n\n').trim();
}

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

function extractKeyframeShotScene(visual: string): string {
  const v = visual.trim();
  if (!v) return '';

  const mdScene = extractMarkdownSection(v, KEYFRAME_MD_SCENE_HEADINGS);
  if (mdScene) {
    const stripped = stripMarkdownQuoteBlock(mdScene);
    const withoutLabel = stripped
      .replace(/^【本帧】\s*/s, '')
      .replace(/^\[Shot\]\s*/is, '')
      .trim();
    return withoutLabel || stripped;
  }

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
  if (!v) return false;
  if (/^##\s+(场景|Scene|角色定妆|Cast reference)/m.test(v)) return true;
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

/** Text for per-shot cast name match — visual + explicit on-screen only (not beat scene_prompt). */
export function shotCastMatchText(shot: LongVideoShotState): string {
  const roleVisual =
    shot.segment_role === 'face_anchor'
      ? shot.anchor_visual_prompt || shot.start_visual_prompt || shot.visual_prompt
      : shot.start_visual_prompt || shot.visual_prompt;
  const visual = (
    extractKeyframeShotScene(roleVisual) || roleVisual || ''
  ).trim();
  const explicit = (shot.characters_on_screen ?? [])
    .map((n) => n.trim())
    .filter(Boolean)
    .join(' ');
  return [visual, explicit].filter(Boolean).join('\n');
}

/** Per-shot cast: explicit cast_looks wins; else infer from shotCastMatchText only. */
export function resolveShotCastLooks(
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  sceneText: string,
): LongVideoShotCastLook[] {
  if (castLooks.length) {
    return castLooks.filter((c) => {
      const ch = characters.find((x) => x.id === c.character_id);
      return Boolean(ch && c.look_id);
    });
  }
  const fromScene = charactersOnScreen(sceneText, characters)
    .map((ch) => ({
      character_id: ch.id,
      look_id: ch.default_look_id || ch.looks[0]?.id || '',
    }))
    .filter((c) => c.look_id);
  return fromScene;
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

export type CastReferenceScope = 'none' | 'wardrobe' | 'face';

export type KeyframePromptContext = {
  characterAnchor?: string;
  characters?: LongVideoCharacter[];
  scenes?: LongVideoScene[];
  styleAnchor?: string;
  castLooks?: LongVideoShotCastLook[];
  sceneLook?: LongVideoShotSceneLook;
  /** Visual + characters_on_screen; overrides visual-only match for cast binding. */
  castMatchText?: string;
  /** Beat environment text (five_aspect.scene) — I2V / scene-look binding, not T2I merge when start_visual exists. */
  sceneNarrative?: string;
  /** Hard first-frame constraint from parse — inspector / strategy only, not T2I scene merge. */
  firstFrameRequirement?: string;
  segmentRole?: LongVideoShotState['segment_role'];
  /** From beat plan Pass 2 — drives cast scope and T2I merge policy. */
  firstFrameVisibility?: LongVideoShotState['first_frame_visibility'];
  isEstablishingEmpty?: boolean;
  shotSize?: string;
  locationMerge?: 'none' | 'prepended' | 'scene_only';
  beatLocation?: string;
  beatScenePrompt?: string;
};

/** Cast T2I reference depth from parse visibility contract (not free-text shot_size). */
export function castReferenceScope(
  visibility: LongVideoShotState['first_frame_visibility'] | undefined,
  segmentRole: LongVideoShotState['segment_role'] | undefined,
  isEstablishingEmpty?: boolean,
): CastReferenceScope {
  const vis = visibility ?? 'full_face';
  if (isEstablishingEmpty || vis === 'invisible') return 'none';
  if (segmentRole === 'face_anchor') return 'face';
  if (vis === 'full_face') return 'face';
  return 'wardrobe';
}

/** Structured script_parse shots carry start_visual; scene_prompt is not merged into T2I. */
export function shouldMergeScenePromptIntoT2i(startVisual: string, scenePrompt = ''): boolean {
  if (startVisual.trim()) return false;
  return Boolean(scenePrompt.trim());
}

/** Max chars when merging beat narrative into T2I scene line. */
const SCENE_NARRATIVE_MERGE_MAX_CHARS = 120;
/** Skip beat narrative merge when this fraction of its tokens already appear in the visual. */
const NARRATIVE_MERGE_COVERAGE_THRESHOLD = 0.38;

/** Legacy chapter-analyze fallback only — structured parse uses start_visual as sole T2I scene source. */
export function shouldSkipBeatNarrativeMerge(
  ctx: Pick<
    KeyframePromptContext,
    'segmentRole' | 'firstFrameVisibility' | 'isEstablishingEmpty'
  > & { startVisual?: string },
): boolean {
  if ((ctx.startVisual ?? '').trim()) return true;
  if (ctx.isEstablishingEmpty || ctx.segmentRole === 'face_anchor') return true;
  const vis = ctx.firstFrameVisibility ?? 'full_face';
  return vis === 'full_face' || vis === 'partial' || vis === 'silhouette';
}

function provenancePreview(text: string, max = 72): string {
  const t = text.trim();
  if (t.length <= max) return t;
  return `${t.slice(0, max).trim()}…`;
}

const PROMPT_TOKEN_RE = /[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}/g;

/** Fold framing labels and near-duplicate wording before token overlap checks. */
export function normalizePromptForComparison(text: string): string {
  return text
    .replace(/【[^】]*】/g, ' ')
    .replace(/\[[^\]]*\]/g, ' ')
    .replace(/映照脸庞/g, '映照')
    .replace(/脸部特写/g, '面部特写')
    .replace(/脸部/g, '面部')
    .replace(/脸庞/g, '面部')
    .replace(/[·•]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function promptTokenSet(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of normalizePromptForComparison(text).matchAll(PROMPT_TOKEN_RE)) {
    const t = m[0];
    if (t.length < 2) continue;
    out.add(/^[A-Za-z]/.test(t) ? t.toLowerCase() : t);
  }
  return out;
}

export function promptTokenCoverage(haystack: string, needle: string): number {
  const needleTokens = promptTokenSet(needle);
  if (!needleTokens.size) return 1;
  const hayTokens = promptTokenSet(haystack);
  if (!hayTokens.size) return 0;
  let hit = 0;
  for (const t of needleTokens) if (hayTokens.has(t)) hit += 1;
  return hit / needleTokens.size;
}

function textAlreadyCovered(haystack: string, needle: string): boolean {
  const h = normalizePromptForComparison(haystack.trim());
  const n = normalizePromptForComparison(needle.trim());
  if (!h || !n) return false;
  if (h.includes(n) || n.includes(h)) return true;
  return promptTokenCoverage(h, n) >= 0.72;
}

function narrativeAddsContext(visual: string, narrative: string): boolean {
  const v = visual.trim();
  const n = narrative.trim();
  if (!n || textAlreadyCovered(v, n)) return false;
  return promptTokenCoverage(v, n) < NARRATIVE_MERGE_COVERAGE_THRESHOLD;
}

/**
 * T2I scene line: beat narrative (when sparse) + frame visual only.
 * first_frame_requirement is inspector/strategy metadata — never merged here.
 */
export function composeKeyframeSceneText(
  visualPrompt: string,
  ctx: Pick<
    KeyframePromptContext,
    'sceneNarrative' | 'segmentRole' | 'firstFrameVisibility' | 'isEstablishingEmpty'
  >,
): string {
  const visualScene = extractKeyframeShotScene(visualPrompt).trim() || visualPrompt.trim();
  if (visualScene) return visualScene;
  const narrative = (ctx.sceneNarrative ?? '').trim();
  const parts: string[] = [];
  if (
    narrative
    && !shouldSkipBeatNarrativeMerge({ ...ctx, startVisual: visualScene })
    && narrativeAddsContext(visualScene, narrative)
  ) {
    const hint =
      narrative.length > SCENE_NARRATIVE_MERGE_MAX_CHARS
        ? `${narrative.slice(0, SCENE_NARRATIVE_MERGE_MAX_CHARS).trim()}…`
        : narrative;
    parts.push(hint);
  }
  if (visualScene) parts.push(visualScene);
  return parts.filter(Boolean).join('；');
}

/** Explain how the T2I scene line was assembled (for parse provenance + Inspector). */
export function buildKeyframeT2iProvenance(
  visualPrompt: string,
  ctx: KeyframePromptContext,
): KeyframeT2iProvenance {
  const visualScene = extractKeyframeShotScene(visualPrompt).trim() || visualPrompt.trim();
  const narrative = (ctx.sceneNarrative ?? '').trim();
  const requirement = (ctx.firstFrameRequirement ?? '').trim();

  let narrative_skip_reason: KeyframeT2iProvenanceSkipReason | undefined;
  let narrative_merged = false;
  let narrative_token_coverage: number | undefined;

  if (!narrative) {
    narrative_skip_reason = 'empty_narrative';
  } else if (shouldSkipBeatNarrativeMerge({ ...ctx, startVisual: visualScene })) {
    narrative_skip_reason =
      ctx.segmentRole === 'face_anchor'
        ? 'face_anchor'
        : visualScene
          ? 'token_coverage_sufficient'
          : ctx.firstFrameVisibility === 'partial' || ctx.firstFrameVisibility === 'silhouette'
            ? 'close_up'
            : 'token_coverage_sufficient';
  } else if (textAlreadyCovered(visualScene, narrative)) {
    narrative_skip_reason = 'narrative_already_covered';
  } else {
    narrative_token_coverage = promptTokenCoverage(visualScene, narrative);
    if (narrative_token_coverage >= NARRATIVE_MERGE_COVERAGE_THRESHOLD) {
      narrative_skip_reason = 'token_coverage_sufficient';
    } else {
      narrative_merged = true;
    }
  }

  const scene_parts: KeyframeT2iProvenance['scene_parts'] = [];
  if (ctx.locationMerge === 'prepended' && (ctx.beatLocation ?? '').trim()) {
    scene_parts.push({
      source: 'location',
      text_preview: provenancePreview(ctx.beatLocation ?? ''),
    });
  }
  if (narrative_merged) {
    const mergedNarrative =
      narrative.length > SCENE_NARRATIVE_MERGE_MAX_CHARS
        ? `${narrative.slice(0, SCENE_NARRATIVE_MERGE_MAX_CHARS).trim()}…`
        : narrative;
    scene_parts.push({ source: 'beat_narrative', text_preview: provenancePreview(mergedNarrative) });
  }
  if (visualScene) {
    scene_parts.push({ source: 'visual_prompt', text_preview: provenancePreview(visualScene) });
  }

  const ffr_skip_reason: KeyframeT2iProvenance['ffr_skip_reason'] = requirement
    ? 'inspector_only'
    : 'empty_ffr';

  return {
    narrative_merged,
    narrative_skip_reason,
    narrative_token_coverage,
    location_merge: ctx.locationMerge ?? 'none',
    first_frame_requirement_merged: false,
    ffr_skip_reason,
    scene_parts,
    composed_scene_line: composeKeyframeSceneText(visualPrompt, ctx),
  };
}

export function buildParseProvenanceByShot(
  shots: LongVideoShotState[],
  project: Pick<LongVideoProjectState, 'character_anchor' | 'characters' | 'scenes' | 'style_anchor'>,
): Record<string, KeyframeT2iProvenance> {
  const out: Record<string, KeyframeT2iProvenance> = {};
  for (const shot of shots) {
    const visual = shotKeyframeText(shot);
    const ctx = keyframePromptContextForShot(shot, project);
    out[shot.id] = buildKeyframeT2iProvenance(visual, ctx);
  }
  return out;
}

export function keyframePromptContextForShot(
  shot: LongVideoShotState,
  project: Pick<LongVideoProjectState, 'character_anchor' | 'characters' | 'scenes' | 'style_anchor'>,
): KeyframePromptContext {
  const beatScenePrompt = (shot.scene_prompt ?? '').trim();
  const beatLocation = (shot.location ?? '').trim();
  const visualHint = extractKeyframeShotScene(shot.visual_prompt).trim();
  const { text: sceneNarrative, locationMerge } = mergeBeatNarrativeFields({
    location: beatLocation,
    scenePrompt: beatScenePrompt,
    visualHint,
  });
  return {
    characterAnchor: project.character_anchor ?? '',
    characters: project.characters,
    scenes: project.scenes,
    styleAnchor: project.style_anchor,
    castLooks: shot.cast_looks,
    sceneLook: shot.scene_look,
    castMatchText: shotCastMatchText(shot),
    sceneNarrative,
    firstFrameRequirement: shot.first_frame_requirement,
    segmentRole: shot.segment_role,
    firstFrameVisibility: shot.first_frame_visibility,
    isEstablishingEmpty: shot.is_establishing_empty,
    locationMerge,
    beatLocation,
    beatScenePrompt,
  };
}

/** T2I prompt: Markdown sections — scene, set reference, cast, style. */
export function keyframeGenerationPrompt(
  visualPrompt: string,
  ctxOrAnchor: KeyframePromptContext | string,
): string {
  const ctx: KeyframePromptContext =
    typeof ctxOrAnchor === 'string' ? { characterAnchor: ctxOrAnchor } : ctxOrAnchor;
  const scene = composeKeyframeSceneText(visualPrompt, ctx);
  const anchor = (ctx.characterAnchor ?? '').trim();
  const characters = ctx.characters ?? [];
  const scenes = ctx.scenes ?? [];
  const styleAnchor = (ctx.styleAnchor ?? '').trim();
  const castLooks = ctx.castLooks ?? [];
  const castMatchText = (ctx.castMatchText ?? scene).trim();
  const castScope = castReferenceScope(
    ctx.firstFrameVisibility,
    ctx.segmentRole,
    ctx.isEstablishingEmpty,
  );
  const locale: 'zh' | 'en' = /[\u4e00-\u9fff]/.test(scene || anchor) ? 'zh' : 'en';
  const sceneSetSection = buildKeyframeMarkdownSceneSetSection(
    scenes,
    ctx.sceneLook,
    scene,
    locale,
  );
  const sceneLine = normalizeSceneLine(scene, locale);
  const sceneBlock = sceneSetSection || characters.length
    ? `> ${keyframeSceneComposeNote(locale, castScope)}\n\n${sceneLine}`
    : sceneLine;

  if (characters.length) {
    const resolvedCast = resolveShotCastLooks(characters, castLooks, castMatchText);
    if (resolvedCast.length) {
      const castSection = buildKeyframeMarkdownCastSection(
        characters,
        resolvedCast,
        castMatchText,
        styleAnchor,
        locale,
        castScope,
      );
      return assembleKeyframeMarkdownPrompt(sceneBlock, sceneSetSection, castSection, styleAnchor, locale);
    }
  }

  if (!scene && !anchor && !sceneSetSection) return '';
  if (!anchor) {
    return assembleKeyframeMarkdownPrompt(sceneBlock, sceneSetSection, '', styleAnchor, locale);
  }

  const blocks = anchorBlocksForVisual(scene || visualPrompt, anchor);
  const castSection = buildKeyframeMarkdownFromAnchorBlocks(
    blocks.length ? blocks : parseAnchorBlocks(anchor),
    locale,
  );
  const hasCast = Boolean(castSection.trim());
  const finalSceneBlock = hasCast || sceneSetSection
    ? `> ${keyframeSceneComposeNote(locale, castScope)}\n\n${sceneLine}`
    : sceneLine;
  return assembleKeyframeMarkdownPrompt(finalSceneBlock, sceneSetSection, castSection, styleAnchor, locale);
}

export const KEYFRAME_CAST_NEGATIVE_ZH =
  '单人肖像, 大头照, 证件照, 纯色背景, 站桩, 复制定妆图构图';
export const KEYFRAME_CAST_NEGATIVE_EN =
  'solo portrait, headshot, id photo, plain background, bust only, copy portrait framing';

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
    const rawLabel = lookLabel.trim();
    const lbl = isPlaceholderLookLabel(rawLabel) ? defaultLabel : rawLabel || defaultLabel;
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

const PLACEHOLDER_LOOK_LABELS = new Set([
  '',
  '无标签',
  '（无标签）',
  '(无标签)',
  '无',
  '未命名',
  '（未命名）',
  '(未命名)',
  'untagged',
  'untitled',
  'default',
  'none',
  'n/a',
  'na',
  '-',
  '—',
]);

export function isPlaceholderLookLabel(label: string): boolean {
  const raw = label.trim();
  if (!raw) return true;
  if (PLACEHOLDER_LOOK_LABELS.has(raw)) return true;
  if (PLACEHOLDER_LOOK_LABELS.has(raw.toLowerCase())) return true;
  if (raw.startsWith('<') && raw.endsWith('>')) return true;
  return raw.includes('无标签') || raw.toLowerCase().includes('untagged');
}

function inferLookLabelsFromBeats(name: string, beatTexts: string[]): string[] {
  const labels: string[] = [];
  const seen = new Set<string>();
  const re = /([^（(]+)[（(]([^）)]+)[）)]/g;
  for (const beat of beatTexts) {
    let m: RegExpExecArray | null;
    re.lastIndex = 0;
    while ((m = re.exec(beat)) !== null) {
      if (m[1].trim() !== name) continue;
      const lbl = m[2].trim();
      if (!lbl || seen.has(lbl) || isPlaceholderLookLabel(lbl)) continue;
      seen.add(lbl);
      labels.push(lbl);
    }
  }
  return labels;
}

function wardrobeSlug(wardrobe: string, maxLen = 8): string {
  const w = wardrobe.replace(/\s+/g, '').trim();
  if (!w) return '';
  return w.length <= maxLen ? w : w.slice(0, maxLen);
}

export function normalizeLookLabel(
  label: string,
  locale: 'zh' | 'en',
  opts: {
    name?: string;
    wardrobe?: string;
    beatTexts?: string[];
    lookIndex?: number;
  } = {},
): string {
  const raw = label.trim();
  if (!isPlaceholderLookLabel(raw)) return raw;
  const fromBeats = inferLookLabelsFromBeats(opts.name ?? '', opts.beatTexts ?? []);
  if (fromBeats.length) {
    const idx = Math.min(Math.max(opts.lookIndex ?? 0, 0), fromBeats.length - 1);
    return fromBeats[idx]!;
  }
  const slug = wardrobeSlug(opts.wardrobe ?? '');
  if (slug) return slug;
  return locale === 'zh' ? '默认' : 'default';
}

export function normalizeCharacterLookLabels(
  characters: LongVideoCharacter[],
  locale: 'zh' | 'en',
  beatTexts: string[] = [],
): LongVideoCharacter[] {
  return characters.map((ch) => ({
    ...ch,
    looks: ch.looks.map((lk, lookIndex) => {
      const parsed = parseCharacterLookBody(lk.body);
      const label = normalizeLookLabel(lk.label, locale, {
        name: ch.name,
        wardrobe: parsed.wardrobe,
        beatTexts,
        lookIndex,
      });
      return label === lk.label.trim() ? lk : { ...lk, label };
    }),
  }));
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

export function createSceneEntry(name: string, locale: 'zh' | 'en' = 'zh'): LongVideoScene {
  const trimmed = name.trim();
  const id = makeCastStableId('scene', trimmed);
  const label = locale === 'zh' ? '默认' : 'default';
  const lookId = makeCastStableId('slook', `${trimmed}|${label}`);
  return {
    id,
    name: trimmed,
    default_look_id: lookId,
    looks: [{ id: lookId, label, body: '' }],
  };
}

export function createSceneLookEntry(
  sceneName: string,
  label: string,
  locale: 'zh' | 'en' = 'zh',
): LongVideoSceneLook {
  const lbl = label.trim() || (locale === 'zh' ? '默认' : 'default');
  return {
    id: makeCastStableId('slook', `${sceneName}|${lbl}`),
    label: lbl,
    body: '',
  };
}

/** Characters at or above this length use the chapter Map-Reduce analyze pipeline. */
export const SCRIPT_CHAPTER_CHAR_THRESHOLD = 500;

export function inferScriptSourceMode(text: string): 'brief' | 'chapter' {
  return (text || '').trim().length >= SCRIPT_CHAPTER_CHAR_THRESHOLD ? 'chapter' : 'brief';
}

export function resolveScriptText(
  lv: Pick<LongVideoProjectState, 'script_text' | 'brief' | 'chapter_text'>,
): string {
  return (lv.script_text ?? lv.chapter_text ?? lv.brief ?? '').trim();
}

/** True when the project has user content worth persisting to the server. */
export function longVideoHasPersistableContent(
  lv: Pick<
    LongVideoProjectState,
    'brief' | 'chapter_text' | 'script_text' | 'chapter_analysis' | 'characters' | 'shots' | 'final_asset_id'
  >,
): boolean {
  if (resolveScriptText(lv).trim()) return true;
  if ((lv.chapter_analysis?.scene_beats?.length ?? 0) > 0) return true;
  if ((lv.characters?.length ?? 0) > 0) return true;
  if ((lv.shots?.length ?? 0) > 0) return true;
  if (lv.final_asset_id) return true;
  return false;
}

export function migrateLongVideoProject(
  partial: Partial<LongVideoProjectState>,
): Partial<LongVideoProjectState> {
  const next = { ...partial };
  if (!next.version || next.version < 2) next.version = 2;
  if (!next.editor_tab) {
    if (next.shots?.length) next.editor_tab = 'storyboard';
    else if (next.characters?.length) next.editor_tab = 'cast';
    else next.editor_tab = 'script';
  }
  if (next.characters?.length) {
    next.characters = next.characters.map((ch) => ({
      ...ch,
      looks: ch.looks.map((lk) => ({ ...lk })),
    }));
  }
  if (!next.script_text) {
    next.script_text = (next.chapter_text ?? next.brief ?? '').trim() || undefined;
  }
  if (next.shots?.length) {
    next.shots = next.shots.map((s) => ({
      ...s,
      segment_role: s.segment_role ?? 'keyframe',
      start_frame_mode: s.start_frame_mode ?? 'keyframe',
      flf_mode: s.flf_mode ?? 'none',
    }));
  }
  return next;
}

function sanitizeCharacterAppearanceText(
  appearance: string,
  ch: LongVideoCharacter,
  otherCharacterNames: string[],
): string {
  let text = appearance.trim();
  text = text.replace(/【[^】]*(?:背景|特征|故事)[^】]*】[\s\S]*/g, '').trim();
  text = text.replace(/\[Character[^\]]*\][\s\S]*/gi, '').trim();
  for (const name of otherCharacterNames) {
    const n = name.trim();
    if (!n || n === ch.name.trim()) continue;
    text = text.split(n).join('');
  }
  return text.replace(/[，,；;|｜]{2,}/g, '，').replace(/\s{2,}/g, ' ').trim();
}

export function characterLookOutfitParts(
  ch: LongVideoCharacter,
  look: LongVideoCharacterLook,
  otherCharacterNames: string[] = [],
): { appearance: string; wardrobe: string } {
  const vision = look.vision_description?.trim();
  if (vision) return { appearance: vision, wardrobe: '' };

  const parsed = parseCharacterLookBody(look.body);
  const appearance = sanitizeCharacterAppearanceText(
    parsed.appearance || look.body || '',
    ch,
    otherCharacterNames,
  );
  let wardrobe = sanitizeCharacterAppearanceText(parsed.wardrobe, ch, otherCharacterNames);
  if (wardrobe && textAlreadyCovered(appearance, wardrobe)) wardrobe = '';
  return {
    appearance: appearance || ch.name.trim(),
    wardrobe,
  };
}

function formatCastOutfitMarkdown(
  parts: { appearance: string; wardrobe: string },
  locale: 'zh' | 'en',
  scope: CastReferenceScope = 'face',
): string {
  if (scope === 'wardrobe') {
    const wardrobeLabel = locale === 'zh' ? '服饰' : 'Wardrobe';
    if (parts.wardrobe) {
      return `- **${wardrobeLabel}**：${parts.wardrobe}`;
    }
    return locale === 'zh'
      ? '- **服饰**：与定妆一致（本帧不可见面部，勿绘制清晰五官）'
      : '- **Wardrobe**: match cast outfit (no readable face in this frame)';
  }
  if (parts.wardrobe) {
    const appLabel = locale === 'zh' ? '外貌' : 'Appearance';
    const wardLabel = locale === 'zh' ? '服装' : 'Wardrobe';
    return `- **${appLabel}**：${parts.appearance}\n\n- **${wardLabel}**：${parts.wardrobe}`;
  }
  const outfitLabel = locale === 'zh' ? '装扮' : 'Outfit';
  return `- **${outfitLabel}**：${parts.appearance}`;
}

export function portraitAppearanceForPrompt(
  ch: LongVideoCharacter,
  look: LongVideoCharacterLook,
  otherCharacterNames: string[] = [],
): string {
  const { appearance, wardrobe } = characterLookOutfitParts(ch, look, otherCharacterNames);
  if (wardrobe) return `${appearance}，${wardrobe}`;
  return appearance;
}

export function buildPortraitPrompt(
  ch: LongVideoCharacter,
  look: LongVideoCharacterLook,
  styleAnchor: string,
  locale: 'zh' | 'en',
  opts?: { otherCharacterNames?: string[]; useCache?: boolean },
): string {
  if (opts?.useCache !== false) {
    const cached = look.portrait_prompt?.trim();
    if (cached) return cached;
  }
  const appearance = portraitAppearanceForPrompt(ch, look, opts?.otherCharacterNames ?? []);
  const style = styleAnchor.trim().slice(0, 120);
  if (locale === 'zh') {
    return (
      `【定妆】单人角色设定参考，仅一名角色，无其他人物，中性表情，头肩至半身，` +
      `纯色或简洁背景，清晰面部与服饰，不含剧情动作。` +
      `角色：${ch.name}（${look.label}）。外貌：${appearance}。` +
      (style ? `画风：${style}。` : '')
    );
  }
  return (
    `[Portrait] solo character reference, single person only, neutral expression, head-to-waist, ` +
    `plain background, clear face and outfit, no action or story scene. ` +
    `Character: ${ch.name} (${look.label}). Appearance: ${appearance}. ` +
    (style ? `Style: ${style}.` : '')
  );
}

export function resolvePrimaryCastPortraitForShot(
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  sceneText: string,
): { character_id: string; look_id: string; reference_asset_id: string } | null {
  const resolved = resolveShotCastLooks(characters, castLooks, sceneText);
  const onScreen = charactersOnScreen(sceneText, characters);
  const ordered = onScreen.length
    ? onScreen
        .map((ch) => resolved.find((r) => r.character_id === ch.id))
        .filter((r): r is LongVideoShotCastLook => Boolean(r))
    : resolved;
  for (const cl of ordered) {
    const ch = characters.find((c) => c.id === cl.character_id);
    const look = ch?.looks.find((l) => l.id === cl.look_id);
    if (look?.reference_asset_id) {
      return {
        character_id: cl.character_id,
        look_id: cl.look_id,
        reference_asset_id: look.reference_asset_id,
      };
    }
  }
  return null;
}

export function collectCastReferenceAssetIdsForShot(
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  sceneText: string,
): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  for (const cl of resolveShotCastLooks(characters, castLooks, sceneText)) {
    const ch = characters.find((c) => c.id === cl.character_id);
    const look = ch?.looks.find((l) => l.id === cl.look_id);
    const refId = look?.reference_asset_id;
    if (refId && !seen.has(refId)) {
      ids.push(refId);
      seen.add(refId);
    }
  }
  return ids;
}

export function mergeKeyframeLoraAdapters(
  composeAdapters: Array<{ id: string; weight: number }>,
  characters: LongVideoCharacter[],
  castLooks: LongVideoShotCastLook[],
  sceneText: string,
  projectLoraId?: string,
  defaultScale = 0.8,
): Array<{ id: string; weight: number }> {
  const merged = new Map<string, number>();
  for (const a of composeAdapters) {
    if (a.id) merged.set(a.id, a.weight);
  }
  for (const cl of resolveShotCastLooks(characters, castLooks, sceneText)) {
    const ch = characters.find((c) => c.id === cl.character_id);
    const look = ch?.looks.find((l) => l.id === cl.look_id);
    if (look?.lora_id) merged.set(look.lora_id, defaultScale);
  }
  if (!merged.size && projectLoraId?.trim()) {
    merged.set(projectLoraId.trim(), defaultScale);
  }
  return [...merged.entries()].map(([id, weight]) => ({ id, weight }));
}

export function looksMissingPortrait(
  characters: LongVideoCharacter[],
): Array<{ characterIndex: number; lookIndex: number }> {
  const missing: Array<{ characterIndex: number; lookIndex: number }> = [];
  characters.forEach((ch, characterIndex) => {
    ch.looks.forEach((lk, lookIndex) => {
      if (!lk.reference_asset_id && (lk.body.trim() || ch.name.trim())) {
        missing.push({ characterIndex, lookIndex });
      }
    });
  });
  return missing;
}

function mergeCharacterLook(
  existing: LongVideoCharacterLook | undefined,
  incoming: LongVideoCharacterLook,
): LongVideoCharacterLook {
  if (!existing) return { ...incoming };
  return {
    ...incoming,
    reference_asset_id: existing.reference_asset_id ?? incoming.reference_asset_id,
    lora_id: existing.lora_id ?? incoming.lora_id,
    portrait_prompt: existing.portrait_prompt ?? incoming.portrait_prompt,
    vision_description: existing.vision_description ?? incoming.vision_description,
    body: incoming.body.trim() ? incoming.body : existing.body,
  };
}

/** Preserve user portraits and look edits when LLM re-parses or regenerates storyboard. */
export function mergeCharacterRosters(
  existing: LongVideoCharacter[],
  incoming: LongVideoCharacter[],
): LongVideoCharacter[] {
  if (!incoming.length) return existing;
  if (!existing.length) return incoming;

  const existingById = new Map(existing.map((ch) => [ch.id, ch]));
  const existingByName = new Map(
    existing.filter((ch) => ch.name.trim()).map((ch) => [ch.name.trim().toLowerCase(), ch]),
  );

  const merged = incoming.map((inc) => {
    const prev =
      existingById.get(inc.id) ??
      (inc.name.trim() ? existingByName.get(inc.name.trim().toLowerCase()) : undefined);
    if (!prev) return inc;

    const prevLooksById = new Map(prev.looks.map((lk) => [lk.id, lk]));
    const prevLooksByLabel = new Map(prev.looks.map((lk) => [lk.label.trim(), lk]));
    const looks = inc.looks.map((lk) => {
      const prevLook =
        prevLooksById.get(lk.id) ??
        (lk.label.trim() ? prevLooksByLabel.get(lk.label.trim()) : undefined);
      return mergeCharacterLook(prevLook, lk);
    });
    for (const prevLook of prev.looks) {
      if (!looks.some((lk) => lk.id === prevLook.id || lk.label === prevLook.label)) {
        looks.push(prevLook);
      }
    }
    const defaultLookId =
      looks.find((lk) => lk.id === prev.default_look_id)?.id ??
      looks.find((lk) => lk.id === inc.default_look_id)?.id ??
      looks[0]?.id ??
      inc.default_look_id;
    return {
      ...inc,
      name: inc.name.trim() || prev.name,
      looks,
      default_look_id: defaultLookId,
    };
  });

  for (const ch of existing) {
    const key = ch.name.trim().toLowerCase();
    const matched = merged.some(
      (inc) => inc.id === ch.id || (key && inc.name.trim().toLowerCase() === key),
    );
    if (!matched) merged.push(ch);
  }
  return merged;
}

function normalizeLocationKey(text: string): string {
  return text.trim().toLowerCase().replace(/[\s·/\\|｜\-—–]/g, '');
}

function cjkBigramSet(text: string): Set<string> {
  const t = text.replace(/[^\u4e00-\u9fffA-Za-z0-9]/g, '');
  const out = new Set<string>();
  if (t.length === 1) {
    out.add(t);
    return out;
  }
  for (let i = 0; i < t.length - 1; i += 1) {
    out.add(t.slice(i, i + 2));
  }
  return out;
}

function cjkBigramOverlapScore(a: string, b: string): number {
  const ba = cjkBigramSet(a);
  const bb = cjkBigramSet(b);
  if (!ba.size || !bb.size) return 0;
  let hit = 0;
  for (const g of ba) if (bb.has(g)) hit += 1;
  const union = ba.size + bb.size - hit;
  return union > 0 ? hit / union : 0;
}

/** Language-agnostic place/location string overlap (no synonym tables). */
export function locationsSimilar(a: string, b: string): boolean {
  const ka = normalizeLocationKey(a);
  const kb = normalizeLocationKey(b);
  if (!ka || !kb) return false;
  if (ka === kb) return true;
  const short = ka.length <= kb.length ? ka : kb;
  const long = ka.length <= kb.length ? kb : ka;
  if (short.length >= 4 && long.includes(short)) return true;
  if (promptTokenCoverage(a, b) >= 0.45 || promptTokenCoverage(b, a) >= 0.45) return true;
  return cjkBigramOverlapScore(a, b) >= 0.34;
}

function locationReferencedInText(loc: string, text: string): boolean {
  if (!loc.trim() || !text.trim()) return false;
  if (locationsSimilar(loc, text) || textAlreadyCovered(text, loc)) return true;
  const cjk = text.replace(/[^\u4e00-\u9fff]/g, '');
  const locCjk = loc.replace(/[^\u4e00-\u9fff]/g, '');
  if (!cjk || !locCjk) return promptTokenCoverage(text, loc) >= 0.45;
  const maxLen = Math.min(Math.max(locCjk.length + 2, 4), 12);
  for (let start = 0; start < cjk.length; start += 1) {
    for (let len = 2; len <= maxLen && start + len <= cjk.length; len += 1) {
      const slice = cjk.slice(start, start + len);
      if (locationsSimilar(locCjk, slice)) return true;
    }
  }
  return promptTokenCoverage(text, loc) >= 0.5;
}

const REQUIREMENT_CLAUSE_SPLIT = /[；;。\n]+/;

/** Merge beat location + scene_prompt without duplicating already-covered place tokens. */
export function mergeBeatNarrativeFields(fields: {
  location: string;
  scenePrompt: string;
  visualHint?: string;
}): { text: string; locationMerge: 'none' | 'prepended' | 'scene_only' } {
  const scene = fields.scenePrompt.trim();
  const loc = fields.location.trim();
  if (!loc) return { text: scene, locationMerge: 'none' };
  if (!scene) return { text: loc, locationMerge: 'prepended' };
  const visual = (fields.visualHint ?? '').trim();
  const locRedundant =
    locationReferencedInText(loc, scene)
    || (visual && locationReferencedInText(loc, visual))
    || promptTokenCoverage(scene, loc) >= 0.5;
  if (locRedundant) return { text: scene, locationMerge: 'scene_only' };
  return { text: `${loc}，${scene}`, locationMerge: 'prepended' };
}

/** Keep only first-frame requirement clauses not already expressed in visual / prior parts. */
export function mergeUncoveredRequirementClauses(
  visualScene: string,
  requirement: string,
  mergedParts: string[] = [],
): { text: string; totalClauses: number; mergedClauses: number } {
  const req = requirement.trim();
  if (!req) return { text: '', totalClauses: 0, mergedClauses: 0 };
  const haystack = [visualScene, ...mergedParts].filter(Boolean).join('；');
  const clauses = req.split(REQUIREMENT_CLAUSE_SPLIT).map((s) => s.trim()).filter(Boolean);
  const uncovered = clauses.filter((c) => !textAlreadyCovered(haystack, c));
  return {
    text: uncovered.join('；'),
    totalClauses: clauses.length,
    mergedClauses: uncovered.length,
  };
}

export function sceneEnvironmentForPrompt(look: LongVideoSceneLook): string {
  const visionDesc = look.vision_description?.trim();
  if (visionDesc) return visionDesc;

  const parsed = parseSceneLookBody(look.body);
  const parts = [parsed.environment, parsed.setDressing].filter(Boolean);
  if (parts.length) return parts.join('，');
  let body = (look.body || '').trim();
  body = body.replace(/【[^】]*】[\s\S]*/g, '').trim();
  return body || look.label.trim();
}

export function parseSceneLookBody(body: string): { environment: string; setDressing: string } {
  const raw = (body || '').trim();
  if (!raw) return { environment: '', setDressing: '' };
  const zh = raw.match(/^环境[:：]\s*([^|｜]+?)\s*[|｜]\s*置景[:：]\s*([\s\S]+)$/);
  if (zh) {
    return { environment: zh[1].trim(), setDressing: zh[2].trim() };
  }
  const en = raw.match(/^Environment[:：]\s*([^|]+?)\s*\|\s*Set dressing[:：]\s*([\s\S]+)$/i);
  if (en) {
    return { environment: en[1].trim(), setDressing: en[2].trim() };
  }
  return { environment: raw, setDressing: '' };
}

/** Dropdown / list label: variant name + short environment cue when labels repeat or are vague. */
export function formatSceneLookOptionLabel(
  look: LongVideoSceneLook,
  locale: string = 'zh',
): string {
  const label = look.label.trim() || (locale.startsWith('zh') ? '默认' : 'default');
  const parsed = parseSceneLookBody(look.body);
  const hint = (parsed.environment || parsed.setDressing || '').trim();
  if (!hint) return label;
  const short = hint.length > 18 ? `${hint.slice(0, 18)}…` : hint;
  if (label.includes('·') || label.includes('—') || label.length >= 8) return label;
  return `${label} — ${short}`;
}

export function resolveShotSceneLook(
  scenes: LongVideoScene[],
  sceneLook: LongVideoShotSceneLook | undefined,
  beatText: string,
): LongVideoShotSceneLook | undefined {
  if (!scenes.length) return undefined;
  if (sceneLook?.scene_id) {
    const sc = scenes.find((s) => s.id === sceneLook.scene_id);
    if (sc) return sceneLook;
  }
  return inferShotSceneLookFromBeat(beatText, scenes);
}

/** Re-parse: keep user cast/scene picks and generated assets when shot id is stable. */
export function mergeParsedShotsWithPrevious(
  previous: LongVideoShotState[],
  incoming: LongVideoShotState[],
): LongVideoShotState[] {
  const prevById = new Map(previous.map((s) => [s.id, s]));
  return incoming.map((shot) => {
    const prev = prevById.get(shot.id);
    if (!prev) return shot;
    const merged: LongVideoShotState = { ...shot };
    if (prev.cast_looks?.length) merged.cast_looks = [...prev.cast_looks];
    if (prev.scene_look) merged.scene_look = prev.scene_look;
    if (prev.keyframe_asset_id) {
      merged.keyframe_asset_id = prev.keyframe_asset_id;
      merged.status = prev.status;
    }
    if (prev.segment_asset_id) {
      merged.segment_asset_id = prev.segment_asset_id;
      if (prev.status === 'segment_ready') merged.status = prev.status;
    }
    return merged;
  });
}

export function enrichShotsWithSceneLooks(
  shots: LongVideoShotState[],
  scenes: LongVideoScene[],
): LongVideoShotState[] {
  if (!scenes.length) return shots;
  let prev: LongVideoShotSceneLook | undefined;
  return shots.map((s) => {
    const beat = (s.scene_prompt || s.visual_prompt || '').trim();
    const binding =
      s.scene_look ??
      inferShotSceneLookFromBeat(beat, scenes, prev);
    if (binding) prev = binding;
    if (binding && !s.scene_look) return { ...s, scene_look: binding };
    return s;
  });
}

export function buildSceneEnvironmentPrompt(
  sc: LongVideoScene,
  look: LongVideoSceneLook,
  styleAnchor: string,
  locale: 'zh' | 'en',
  opts?: { useCache?: boolean },
): string {
  if (opts?.useCache !== false) {
    const cached = look.environment_prompt?.trim();
    if (
      cached &&
      (/^【场景参考】/u.test(cached) || /^\[Set reference\]/i.test(cached))
    ) {
      return cached;
    }
  }
  const env = sceneEnvironmentForPrompt(look);
  const style = styleAnchor.trim().slice(0, 120);
  if (locale === 'zh') {
    return (
      `【场景参考】空镜/establishing shot，无人物主体，展示空间结构与典型光线。` +
      `地点：${sc.name}（${look.label}）。环境：${env}。` +
      (style ? `画风：${style}。` : '')
    );
  }
  return (
    `[Set reference] establishing shot, no character subject, spatial layout and lighting. ` +
    `Location: ${sc.name} (${look.label}). Environment: ${env}.` +
    (style ? ` Style: ${style}.` : '')
  );
}

function mergeSceneLook(
  existing: LongVideoSceneLook | undefined,
  incoming: LongVideoSceneLook,
): LongVideoSceneLook {
  if (!existing) return { ...incoming };
  return {
    ...incoming,
    reference_asset_id: existing.reference_asset_id ?? incoming.reference_asset_id,
    environment_prompt: existing.environment_prompt ?? incoming.environment_prompt,
    vision_description: existing.vision_description ?? incoming.vision_description,
    body: incoming.body.trim() ? incoming.body : existing.body,
  };
}

/** Preserve user scene reference images when LLM re-parses. */
export function mergeSceneRosters(
  existing: LongVideoScene[],
  incoming: LongVideoScene[],
): LongVideoScene[] {
  if (!incoming.length) return existing;
  if (!existing.length) return incoming;

  const existingById = new Map(existing.map((sc) => [sc.id, sc]));
  const existingByName = new Map(
    existing.filter((sc) => sc.name.trim()).map((sc) => [sc.name.trim().toLowerCase(), sc]),
  );

  const merged = incoming.map((inc) => {
    const prev =
      existingById.get(inc.id) ??
      (inc.name.trim() ? existingByName.get(inc.name.trim().toLowerCase()) : undefined);
    if (!prev) return inc;

    const prevLooksById = new Map(prev.looks.map((lk) => [lk.id, lk]));
    const prevLooksByLabel = new Map(prev.looks.map((lk) => [lk.label.trim(), lk]));
    const looks = inc.looks.map((lk) => {
      const prevLook =
        prevLooksById.get(lk.id) ??
        (lk.label.trim() ? prevLooksByLabel.get(lk.label.trim()) : undefined);
      return mergeSceneLook(prevLook, lk);
    });
    for (const prevLook of prev.looks) {
      if (!looks.some((lk) => lk.id === prevLook.id || lk.label === prevLook.label)) {
        looks.push(prevLook);
      }
    }
    const defaultLookId =
      looks.find((lk) => lk.id === prev.default_look_id)?.id ??
      looks.find((lk) => lk.id === inc.default_look_id)?.id ??
      looks[0]?.id ??
      inc.default_look_id;
    return {
      ...inc,
      name: inc.name.trim() || prev.name,
      looks,
      default_look_id: defaultLookId,
    };
  });

  for (const sc of existing) {
    const key = sc.name.trim().toLowerCase();
    const matched = merged.some(
      (inc) => inc.id === sc.id || (key && inc.name.trim().toLowerCase() === key),
    );
    if (!matched) merged.push(sc);
  }
  return merged;
}

export function looksMissingSceneReference(
  scenes: LongVideoScene[],
): Array<{ sceneIndex: number; lookIndex: number }> {
  const missing: Array<{ sceneIndex: number; lookIndex: number }> = [];
  scenes.forEach((sc, sceneIndex) => {
    sc.looks.forEach((lk, lookIndex) => {
      if (!lk.reference_asset_id && (lk.body.trim() || sc.name.trim())) {
        missing.push({ sceneIndex, lookIndex });
      }
    });
  });
  return missing;
}

function findSceneMentionedInBeat(beat: string, scenes: LongVideoScene[]): LongVideoScene | undefined {
  const parsed = parseSceneBeat(beat);
  const hay = (parsed.visual || beat).trim();
  if (!hay) return undefined;
  let best: LongVideoScene | undefined;
  let bestLen = 0;
  for (const sc of scenes) {
    const name = sc.name.trim();
    if (!name || !hay.includes(name)) continue;
    if (name.length > bestLen) {
      best = sc;
      bestLen = name.length;
    }
  }
  if (best) return best;
  for (const sc of scenes) {
    for (const lk of sc.looks) {
      const phrase = `${sc.name}${lk.label}`.trim();
      if (phrase.length >= 2 && hay.includes(phrase)) return sc;
      const label = lk.label.trim();
      if (label.length >= 2 && hay.includes(label)) return sc;
    }
  }
  return undefined;
}

export function inferShotSceneLookFromBeat(
  beat: string,
  scenes: LongVideoScene[],
  prev?: LongVideoShotSceneLook,
): LongVideoShotSceneLook | undefined {
  const location = parseSceneBeat(beat).location.trim();
  let best: LongVideoScene | undefined;
  if (location) {
    for (const sc of scenes) {
      if (locationsSimilar(sc.name, location)) {
        best = sc;
        break;
      }
    }
    if (!best) {
      for (const sc of scenes) {
        for (const lk of sc.looks) {
          if (locationsSimilar(`${sc.name}${lk.label}`, location) || locationsSimilar(lk.label, location)) {
            best = sc;
            break;
          }
        }
        if (best) break;
      }
    }
  }
  if (!best) {
    best = findSceneMentionedInBeat(beat, scenes);
  }
  if (!best) return prev;
  let lookId = prev?.scene_id === best.id ? prev.look_id : best.default_look_id;
  const variantHint = location.startsWith(best.name)
    ? location.slice(best.name.length).replace(/^[·/\\|｜，,\s]+/, '')
    : location;
  if (variantHint) {
    const matched = best.looks.find(
      (lk) =>
        locationsSimilar(lk.label, variantHint)
        || locationsSimilar(`${best.name}${lk.label}`, location)
        || lk.label.includes(variantHint)
        || variantHint.includes(lk.label),
    );
    if (matched) lookId = matched.id;
  }
  if (!lookId && best.looks[0]) lookId = best.looks[0].id;
  if (!lookId) return prev;
  return { scene_id: best.id, look_id: lookId };
}

export function syncChapterAnalysisFields(
  analysis: LongVideoChapterAnalysis | undefined | null,
  fields: Pick<LongVideoChapterAnalysis, 'characters' | 'character_anchor' | 'style_anchor' | 'scenes'>,
): LongVideoChapterAnalysis | undefined {
  if (!analysis) return undefined;
  return {
    ...analysis,
    characters: fields.characters ?? analysis.characters,
    scenes: fields.scenes ?? analysis.scenes,
    character_anchor: fields.character_anchor ?? analysis.character_anchor,
    style_anchor: fields.style_anchor ?? analysis.style_anchor,
  };
}

/** Beat-level scene narrative — for scene-entity binding, not cast name match. */
export function shotSceneText(shot: LongVideoShotState): string {
  return (shot.scene_prompt || extractKeyframeShotScene(shot.visual_prompt)).trim();
}

export function resolveSceneForShot(
  lv: LongVideoProjectState,
  shot: LongVideoShotState,
): LongVideoScene | undefined {
  const scenes = lv.scenes ?? [];
  const sceneLook = shot.scene_look;
  if (sceneLook?.scene_id) {
    const byId = scenes.find((s) => s.id === sceneLook.scene_id);
    if (byId) return byId;
  }
  const text = shotSceneText(shot);
  if (!text) return undefined;
  return scenes.find((s) => s.name && text.includes(s.name));
}

type CameraZone = {
  id: string;
  azimuth?: number;
  elevation?: number;
  fov?: number;
  description?: string;
};

function cameraZonesFromLayout(layout: Record<string, unknown> | undefined): CameraZone[] {
  const raw = layout?.camera_zones;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((z): z is Record<string, unknown> => z != null && typeof z === 'object')
    .map((z, i) => ({
      id: String(z.id ?? `zone_${i}`),
      azimuth: typeof z.azimuth === 'number' ? z.azimuth : undefined,
      elevation: typeof z.elevation === 'number' ? z.elevation : undefined,
      fov: typeof z.fov === 'number' ? z.fov : undefined,
      description: typeof z.description === 'string' ? z.description : undefined,
    }));
}

function pickCameraZone(
  zones: CameraZone[],
  opts: { preferredZoneId?: string; visibility?: string },
): CameraZone {
  const preferred = (opts.preferredZoneId ?? '').trim();
  if (preferred) {
    const hit = zones.find((z) => z.id === preferred);
    if (hit) return hit;
  }
  const vis = opts.visibility ?? 'full_face';
  if (vis === 'invisible' || vis === 'silhouette') {
    const wide = zones.find((z) => {
      const d = (z.description ?? '').toLowerCase();
      return d.includes('wide') || d.includes('door') || d.includes('entry');
    });
    if (wide) return wide;
  }
  return zones[0] ?? { id: 'default_wide', description: 'wide establishing' };
}

export function visibilityShortLabel(visibility: string | undefined): string {
  const m: Record<string, string> = {
    invisible: 'Inv',
    silhouette: 'Sil',
    partial: 'Part',
    full_face: 'Face',
  };
  return m[String(visibility || 'invisible')] ?? 'Inv';
}

export function anchorLinkHintKey(shot: LongVideoShotState | undefined): string | null {
  if (!shot) return null;
  if (shot.start_frame_mode === 'anchor_link') return 'video.longVideoAnchorLinkHint';
  if (shot.segment_role === 'tail_continuation' || shot.start_frame_mode === 'prev_segment_tail') {
    return 'video.longVideoTailContinuationHint';
  }
  return null;
}

/** Metadata for keyframe generation when using scene grounding (G0–G2). */
export function buildKeyframeGroundingMetadata(
  shot: LongVideoShotState,
  scene: LongVideoScene | undefined,
): Record<string, string> {
  const layout =
    scene?.spatial_layout_json && typeof scene.spatial_layout_json === 'object'
      ? scene.spatial_layout_json
      : {};
  const zones = cameraZonesFromLayout(layout as Record<string, unknown>);
  const picked = pickCameraZone(zones, {
    preferredZoneId: shot.camera_zone_id,
    visibility: shot.first_frame_visibility,
  });
  const meta: Record<string, string> = {
    long_video_first_frame_strategy: shot.first_frame_strategy ?? 't2i_from_grounding',
    long_video_scene_grounding_camera_zone_id: picked.id,
  };
  if (scene?.grounding_panorama_asset_id) {
    meta.long_video_scene_grounding_panorama_asset_id = scene.grounding_panorama_asset_id;
  }
  if (scene?.grounding_depth_asset_id) {
    meta.long_video_scene_grounding_depth_asset_id = scene.grounding_depth_asset_id;
  }
  if (shot.first_frame_requirement?.trim()) {
    meta.long_video_first_frame_requirement = shot.first_frame_requirement.trim();
  }
  return meta;
}

export { extractKeyframeShotScene, isStructuredKeyframeVisual };

export function defaultLongVideoProject(
  partial: Partial<LongVideoProjectState> = {},
): LongVideoProjectState {
  const migrated = migrateLongVideoProject(partial);
  return {
    version: 2,
    strategy: 'segmented_i2v',
    editor_tab: 'script',
    title: '',
    script_text: '',
    brief: '',
    character_anchor: '',
    characters: [],
    style_anchor: '',
    target_duration_sec: 60,
    keyframe_model: migrated.keyframe_model || 'z-image-turbo',
    segment_video_model: migrated.segment_video_model || 'wan-2.2-i2v-14b',
    segment_duration_sec: 5,
    overlap_frames: 4,
    chain_mode: 'keyframe_only',
    shots: [],
    selection: null,
    ...migrated,
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
    selection: ordered.length ? { kind: 'segment', index: 0 } : null,
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
    segment_role: 'keyframe',
    start_frame_mode: 'keyframe',
    flf_mode: 'none',
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
    selection = { kind: 'clip', index: index - 1 };
  } else if (index === 0) {
    selection = { kind: 'segment', index: 0 };
  } else {
    selection =
      next.length >= MIN_LONG_VIDEO_KEYFRAMES
        ? { kind: 'clip', index: next.length - 2 }
        : { kind: 'segment', index: Math.max(0, next.length - 1) };
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
