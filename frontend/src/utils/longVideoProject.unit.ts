/**
 * Unit checks for long-video keyframe list mutations.
 * Run: npx tsx frontend/src/utils/longVideoProject.unit.ts
 */
import type { LongVideoShotState } from '@/types';
import {
  createEmptyShot,
  defaultLongVideoProject,
  effectiveShotChainMode,
  insertKeyframeAfter,
  insertKeyframeBefore,
  invalidateSegmentAsset,
  looksMissingPortrait,
  mergeKeyframeLoraAdapters,
  inferScriptSourceMode,
  resolveScriptText,
  longVideoHasPersistableContent,
  keyframeGenerationPrompt,
  composeKeyframeSceneText,
  buildKeyframeT2iProvenance,
  mergeBeatNarrativeFields,
  mergeUncoveredRequirementClauses,
  locationsSimilar,
  shouldSkipBeatNarrativeMerge,
  isCloseUpShotSize,
  promptTokenCoverage,
  extractKeyframeShotScene,
  parseSceneLookBody,
  sceneEnvironmentForPrompt,
  enrichShotsWithSceneLooks,
  mergeCharacterRosters,
  migrateLongVideoProject,
  computeStoryboardReadiness,
  MIN_LONG_VIDEO_KEYFRAMES,
  removeKeyframeAt,
  removeKeyframeNeedsConfirm,
  resolvePrimaryCastPortraitForShot,
  sceneMentionsCharacter,
  shotNeedsKeyframe,
  shotVideoPrompt,
  segmentVideoSubmitPreview,
  shotKeyframeText,
  canGenerateSegmentShot,
  segmentI2vSourceAssetId,
  charactersOnScreen,
  buildPortraitPrompt,
  portraitAppearanceForPrompt,
  buildCastVisionBackfillQuestion,
  buildSceneVisionBackfillQuestion,
  resolveLongVideoLocale,
  collectCastReferenceAssetIdsForShot,
  insertFaceAnchorIntoGroup,
  groupHasFaceAnchor,
  planGroupGeneration,
} from './longVideoProject';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

function shot(id: string, partial: Partial<LongVideoShotState> = {}): LongVideoShotState {
  return {
    id,
    order: 0,
    visual_prompt: '',
    motion_prompt: '',
    segment_role: 'keyframe',
    start_frame_mode: 'keyframe',
    flf_mode: 'none',
    status: 'draft',
    ...partial,
  };
}

const three = [
  shot('a', { order: 0, motion_prompt: 's0', segment_asset_id: 'seg0' }),
  shot('b', { order: 1, keyframe_asset_id: 'kf1', motion_prompt: 's1' }),
  shot('c', { order: 2, keyframe_asset_id: 'kf2' }),
];

const inv = invalidateSegmentAsset(three[0]!);
assert(!inv.segment_asset_id, 'segment asset cleared');
assert(inv.status === 'draft', 'status downgraded');

const after = insertKeyframeAfter(three, 0);
assert(after.shots.length === 4, 'insert after adds one');
assert(after.newIndex === 1, 'new index');
assert(!after.shots[0]!.segment_asset_id, 'left segment invalidated');

const before = insertKeyframeBefore(three, 1);
assert(before.shots.length === 4, 'insert before adds one');
assert(before.newIndex === 1, 'new index at insert point');
assert(!before.shots[0]!.segment_asset_id, 'prior left segment invalidated');

const mid = removeKeyframeAt(three, 1);
assert(mid?.shots.length === 2, 'middle delete');
assert(!mid?.shots[0]!.segment_asset_id, 'bridged segment invalidated');
assert(mid?.selection?.kind === 'clip' && mid.selection.index === 0, 'select bridged clip');

const last = removeKeyframeAt(three, 2);
assert(last?.shots[1]!.motion_prompt === '', 'last delete clears left motion');
assert(last?.selection?.kind === 'clip', 'select previous clip after last delete');

assert(removeKeyframeAt(three.slice(0, 2), 0) === null, 'min keyframes guard');
assert(MIN_LONG_VIDEO_KEYFRAMES === 2, 'min constant');

assert(removeKeyframeNeedsConfirm(three, 1), 'middle with assets needs confirm');
assert(!removeKeyframeNeedsConfirm([createEmptyShot(0), createEmptyShot(1)], 0), 'empty skip confirm');

assert(effectiveShotChainMode(shot('a', { chain_mode: 'last_frame' }), 'keyframe_only') === 'last_frame', 'shot override');
assert(effectiveShotChainMode(shot('a'), 'last_frame') === 'last_frame', 'project default');

assert(sceneMentionsCharacter('赵今麦低头看手机', '赵今麦'), 'indexOf name');
assert(sceneMentionsCharacter('屏上挑战孙悟空字样', '孙悟空'), 'indexOf substring');
assert(
  charactersOnScreen('赵今麦独行', [
    { id: '1', name: '赵今麦', default_look_id: 'l1', looks: [{ id: 'l1', label: '默认', body: '' }] },
    { id: '2', name: '孙悟空', default_look_id: 'l2', looks: [{ id: 'l2', label: '默认', body: '' }] },
  ]).length === 1,
  'only named character',
);
assert(charactersOnScreen('', [{ id: '1', name: '赵今麦', default_look_id: 'l1', looks: [] }]).length === 0, 'empty scene');

const chars = [
  {
    id: 'c1',
    name: 'Alice',
    default_look_id: 'l1',
    looks: [
      { id: 'l1', label: 'daily', body: 'blue dress', reference_asset_id: 'asset_a' },
      { id: 'l2', label: 'battle', body: 'armor', lora_id: 'lora_x' },
    ],
  },
  {
    id: 'c2',
    name: 'Bob',
    default_look_id: 'l3',
    looks: [{ id: 'l3', label: 'default', body: 'suit' }],
  },
];

const primary = resolvePrimaryCastPortraitForShot(chars, [{ character_id: 'c1', look_id: 'l1' }], 'Alice walks');
assert(primary?.reference_asset_id === 'asset_a', 'primary portrait');

const loras = mergeKeyframeLoraAdapters([], chars, [{ character_id: 'c1', look_id: 'l2' }], 'Alice fights', 'global_lora');
assert(loras.some((a) => a.id === 'lora_x'), 'look lora merged');
assert(!loras.some((a) => a.id === 'global_lora'), 'look lora blocks global');

const missing = looksMissingPortrait(chars);
assert(missing.length === 2, 'missing portrait count');

const migrated = migrateLongVideoProject({ version: 1, shots: [] });
assert(migrated.version === 2 && migrated.editor_tab === 'script', 'migrate v1 empty → script');

const migratedWithShots = migrateLongVideoProject({ version: 1, shots: [shot('a')] });
assert(migratedWithShots.editor_tab === 'storyboard', 'migrate v1 with shots → storyboard');

const defaults = defaultLongVideoProject();
assert(defaults.version === 2 && defaults.editor_tab === 'script', 'default script tab');

const refs = collectCastReferenceAssetIdsForShot(
  chars,
  [{ character_id: 'c1', look_id: 'l1' }, { character_id: 'c2', look_id: 'l3' }],
  'Alice and Bob',
);
assert(refs.length === 1 && refs[0] === 'asset_a', 'collect refs dedupe');

const portraitPrompt = buildPortraitPrompt(chars[0]!, chars[0]!.looks[0]!, 'cinematic', 'en', { useCache: false });
assert(portraitPrompt.includes('Alice'), 'portrait prompt');
assert(portraitPrompt.includes('single person'), 'portrait solo constraint');

const appearance = portraitAppearanceForPrompt(
  { id: 'c1', name: '赵今麦', default_look_id: 'l1', looks: [] },
  {
    id: 'l1',
    label: '默认',
    body: '定位：主角 | 外貌：红色夹克，短发。【角色特征/背景故事】与孙悟空、阎罗王对峙',
  },
  ['孙悟空', '阎罗王'],
);
assert(!appearance.includes('孙悟空'), 'strip other cast from appearance');
assert(appearance.includes('红色夹克'), 'keep target appearance');

assert(
  portraitAppearanceForPrompt(
    { id: 'c1', name: '赵今麦', default_look_id: 'l1', looks: [] },
    {
      id: 'l1',
      label: '云雾山攀登',
      body: '定位：主角 | 外貌：短发，年轻女性，肤色苍白 | 服装：深色登山冲锋衣，护腕，登山靴',
    },
  ).includes('登山冲锋衣'),
  'portrait merges wardrobe from structured look body',
);

const keyframeWardrobe = keyframeGenerationPrompt('【中景】云雾山山径·夜，赵今麦汗水浸透衣背', {
  characters: [
    {
      id: 'c1',
      name: '赵今麦',
      default_look_id: 'l1',
      looks: [
        {
          id: 'l1',
          label: '云雾山攀登',
          body: '定位：主角 | 外貌：短发，年轻女性，肤色苍白 | 服装：深色登山冲锋衣，护腕，登山靴',
        },
      ],
    },
  ],
  castLooks: [{ character_id: 'c1', look_id: 'l1' }],
  styleAnchor: '冷色调电影感',
});
assert(keyframeWardrobe.includes('**服装**'), 'keyframe cast section lists wardrobe line');
assert(keyframeWardrobe.includes('登山冲锋衣'), 'keyframe cast includes wardrobe text');

assert(
  portraitAppearanceForPrompt(
    { id: 'c1', name: 'Alice', default_look_id: 'l1', looks: [] },
    { id: 'l1', label: '默认', body: '定位：主角 | 外貌：旧描述', vision_description: 'Vision 红夹克短发' },
  ) === 'Vision 红夹克短发',
  'keyframe prefers vision_description over body',
);

const castQZh = buildCastVisionBackfillQuestion(
  { id: 'c1', name: 'Alice', default_look_id: 'l1', looks: [] },
  ['Bob'],
  'zh',
);
assert(castQZh.includes('简体中文'), 'cast vision question zh output language');
const castQEn = buildCastVisionBackfillQuestion(
  { id: 'c1', name: 'Alice', default_look_id: 'l1', looks: [] },
  ['Bob'],
  'en',
);
assert(castQEn.includes('English'), 'cast vision question en output language');
assert(resolveLongVideoLocale('en-US') === 'en', 'resolve en locale');

assert(inferScriptSourceMode('') === 'brief', 'empty → brief');
assert(inferScriptSourceMode('x'.repeat(499)) === 'brief', 'below threshold');
assert(inferScriptSourceMode('x'.repeat(500)) === 'chapter', 'at threshold');
assert(
  resolveScriptText({ script_text: 'unified script' }) === 'unified script',
  'script_text primary',
);
assert(
  resolveScriptText({ chapter_text: 'long chapter', brief: 'short' }) === 'long chapter',
  'chapter_text fallback',
);
assert(
  resolveScriptText({ brief: 'idea only' }) === 'idea only',
  'brief legacy fallback',
);

const existingChar = chars[0]!;
const merged = mergeCharacterRosters(
  [{ ...existingChar, looks: [{ ...existingChar.looks[0]!, reference_asset_id: 'keep_me' }] }],
  [{ ...existingChar, looks: [{ ...existingChar.looks[0]!, body: 'new body from llm' }] }],
);
assert(merged[0]?.looks[0]?.reference_asset_id === 'keep_me', 'merge keeps portrait asset');
assert(merged[0]?.looks[0]?.body === 'new body from llm', 'merge takes incoming body when set');

const readiness = computeStoryboardReadiness([
  { ...shot('a'), keyframe_asset_id: 'kf1', motion_prompt: 'pan', segment_asset_id: 'seg1', status: 'segment_ready' },
  shot('b'),
]);
assert(readiness.keyframeCount === 1 && readiness.segmentCount === 1 && !readiness.mergeReady, 'readiness partial');

assert(!longVideoHasPersistableContent(defaultLongVideoProject()), 'empty project not persistable');
assert(longVideoHasPersistableContent({ ...defaultLongVideoProject(), brief: 'hello' }), 'brief text persistable');

const promptA = keyframeGenerationPrompt('沙漠远景，红衣女侠与黑衣剑客对峙', {
  characters: chars,
  castLooks: [{ character_id: 'c1', look_id: 'l1' }],
  styleAnchor: '武侠电影',
});
const promptB = keyframeGenerationPrompt('近景，女侠挥剑出击', {
  characters: chars,
  castLooks: [{ character_id: 'c1', look_id: 'l1' }],
  styleAnchor: '武侠电影',
});
assert(promptA !== promptB, 'keyframe prompts differ per scene');
assert(promptA.includes('## 场景'), 'markdown scene section');
assert(promptA.includes('## 角色定妆'), 'markdown cast section');
assert(promptA.includes('装扮'), 'outfit line in cast section');
assert(!promptA.includes('定妆提示词'), 'no portrait prompt in keyframe');
assert(promptA.includes('勿复制定妆') || promptA.includes('do not copy portrait'), 'cast scene note');

const desertScenes = [
  {
    id: 'sc_desert',
    name: '沙漠',
    default_look_id: 'sl1',
    looks: [
      {
        id: 'sl1',
        label: '白日',
        body: '环境：金黄沙丘与热浪 | 置景：风蚀岩与枯骨',
        vision_description: '正午金黄沙丘，热浪扭曲空气，远处风蚀岩',
      },
    ],
  },
];
const promptDesert = keyframeGenerationPrompt('沙漠远景，红衣女侠与黑衣剑客对峙', {
  characters: chars,
  castLooks: [{ character_id: 'c1', look_id: 'l1' }],
  scenes: desertScenes,
  styleAnchor: '武侠电影',
});
assert(promptDesert.includes('## 场景设定'), 'scene set section when roster infers');
assert(
  promptDesert.includes('正午金黄') || promptDesert.includes('金黄沙丘'),
  'scene env from vision or structured body',
);

const richVisual =
  'Indoor night, close-up on lead actor face, cool tone, fingers hovering over confirm button';
const beatNarrative =
  'Indoor bedroom at night, lead actor reads phone notification, hesitates, then taps confirm';
const frameReq = 'Eyes widen, breath quickens, stare at floating red notification';
const composedRich = composeKeyframeSceneText(richVisual, {
  sceneNarrative: beatNarrative,
  firstFrameRequirement: frameReq,
});
assert(
  !composedRich.includes('hesitates'),
  'rich visual skips redundant beat narrative',
);
assert(composedRich.includes('notification'), 'first_frame_requirement merged when not in visual');
assert(composedRich.includes('close-up'), 'anchor visual preserved');

const composedSparse = composeKeyframeSceneText('Close-up, lead actor face, static', {
  sceneNarrative: beatNarrative,
  firstFrameRequirement: frameReq,
});
assert(
  promptTokenCoverage(composedSparse, beatNarrative) >= 0.15,
  'sparse visual gets beat narrative context',
);

const facePrompt = keyframeGenerationPrompt(richVisual, {
  sceneNarrative: beatNarrative,
  firstFrameRequirement: frameReq,
  characters: chars,
  castLooks: [{ character_id: 'c1', look_id: 'l1' }],
});
assert(facePrompt.includes('notification'), 'T2I preview includes first-frame requirement');

const faceAnchorCtx = {
  sceneNarrative: beatNarrative,
  firstFrameRequirement: frameReq,
  segmentRole: 'face_anchor' as const,
  shotSize: '特写',
};
const composedFaceAnchor = composeKeyframeSceneText('特写，主角面部静止', faceAnchorCtx);
assert(!composedFaceAnchor.includes('hesitates'), 'face_anchor skips beat narrative merge');
assert(composedFaceAnchor.includes('静止') || composedFaceAnchor.includes('特写'), 'face_anchor keeps visual');

const closeUpProv = buildKeyframeT2iProvenance('近景，女侠侧脸', {
  sceneNarrative: '云雾山山径，女侠沿山路疾行，衣袂翻飞',
  segmentRole: 'keyframe',
  shotSize: '近景',
});
assert(closeUpProv.narrative_skip_reason === 'close_up', 'close-up provenance reason');
assert(!closeUpProv.narrative_merged, 'close-up does not merge narrative');

assert(locationsSimilar('云雾山山径', '云雾山山路'), 'path location bigram overlap');
const locMerge = mergeBeatNarrativeFields({
  location: '云雾山山径',
  scenePrompt: '女侠沿云雾山山路疾行，衣袂翻飞',
  visualHint: '近景，女侠侧脸',
});
assert(locMerge.locationMerge === 'scene_only', 'redundant location not prepended');
assert(!locMerge.text.startsWith('云雾山山径'), 'scene-only narrative');

const ffrDup = mergeUncoveredRequirementClauses(
  '特写，主角面部静止，眼神凝固',
  '眼神凝固；面部静止；背景虚化',
);
assert(ffrDup.mergedClauses < ffrDup.totalClauses, 'FFR clauses deduped against visual');

const motionShot = {
  ...shot('m'),
  video_prompt: 'parsed clip',
  motion_prompt: 'parsed clip',
} as LongVideoShotState;
assert(shotVideoPrompt(motionShot) === 'parsed clip', 'video_prompt primary when synced');
const editedMotion = { ...motionShot, motion_prompt: 'user edit', video_prompt: 'parsed clip' };
assert(shotVideoPrompt(editedMotion) === 'user edit', 'motion_prompt wins when diverged after edit');

const i2vPreview = segmentVideoSubmitPreview(motionShot, [motionShot], { locale: 'zh', shotIndex: 0 });
assert(i2vPreview.includes('片段运动'), 'I2V preview heading');

assert(parseSceneLookBody('环境：foo | 置景：bar').environment === 'foo', 'parse scene body zh');
assert(
  sceneEnvironmentForPrompt({
    id: 'sl1',
    label: 'x',
    body: '环境：foo | 置景：bar',
    vision_description: 'vision plain text',
  }) === 'vision plain text',
  'prefer vision_description for keyframe',
);
assert(
  sceneEnvironmentForPrompt({
    id: 'sl1',
    label: 'x',
    body: '环境：foo | 置景：bar',
  }) === 'foo，bar',
  'fallback to structured body without vision_description',
);

const enriched = enrichShotsWithSceneLooks(
  [shot('a', { visual_prompt: '沙漠远景对峙', scene_prompt: '沙漠·白日' })],
  desertScenes,
);
assert(enriched[0]?.scene_look?.scene_id === 'sc_desert', 'infer scene_look on shots');

assert(
  extractKeyframeShotScene(
    `## 场景\n\n> note\n\n【本帧】云雾山顶，女侠拔剑`,
  ).includes('云雾山顶'),
  'extract scene from markdown',
);

const tailShot = shot('tail', {
  start_frame_mode: 'prev_segment_tail',
  chain_mode: 'last_frame',
  video_prompt: 'chase continues',
  motion_prompt: 'chase continues',
});
assert(!shotNeedsKeyframe(tailShot), 'tail segment skips keyframe');
assert(shotVideoPrompt(tailShot) === 'chase continues', 'video prompt priority');
assert(shotKeyframeText({ ...tailShot, start_visual_prompt: 'still' }) === 'still', 'start visual text');

const lvShots = [
  shot('a', { keyframe_asset_id: 'kf0', segment_asset_id: 'seg0' }),
  tailShot,
];
assert(
  canGenerateSegmentShot({ shots: lvShots, chain_mode: 'keyframe_only' }, 1),
  'tail can generate when prev segment exists',
);
assert(
  segmentI2vSourceAssetId({ shots: lvShots, chain_mode: 'keyframe_only' }, 1) === 'kf0',
  'tail uses prev keyframe as submit placeholder',
);
assert(
  !canGenerateSegmentShot({ shots: [shot('a'), tailShot], chain_mode: 'keyframe_only' }, 1),
  'tail blocked without prev segment',
);

const establishing = shot('est1', {
  segment_role: 'establishing',
  segment_group_id: 'beat_0',
  visual_prompt: '远景：角色站在门口',
  motion_prompt: 'slow push',
  duration_sec: 8,
});
const withAnchor = insertFaceAnchorIntoGroup([establishing], 'beat_0');
assert(withAnchor.length >= 2, 'insert anchor expands group');
assert(groupHasFaceAnchor(withAnchor, 'beat_0'), 'group has face anchor');

const groupPlan = planGroupGeneration(
  { shots: withAnchor, chain_mode: 'keyframe_only' },
  { groupId: 'beat_0', beatIndex: 0, title: 't', shotIndices: withAnchor.map((_, i) => i) },
);
assert(groupPlan.keyframeIndices.length > 0, 'plan lists pending keyframes');

console.log('longVideoProject.unit: OK');
