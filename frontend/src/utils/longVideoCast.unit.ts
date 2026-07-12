/**
 * Cast binding for long-video shots.
 * Run: npx tsx frontend/src/utils/longVideoCast.unit.ts
 */
import type { LongVideoCharacter, LongVideoShotState } from '@/types';
import { normalizeLookLabel, resolveShotCastLooks, shotCastMatchText } from './longVideoProject';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const characters: LongVideoCharacter[] = [
  {
    id: 'char_zjm',
    name: '赵今麦',
    looks: [{ id: 'look_zjm', label: '日常', body: '白T恤' }],
    default_look_id: 'look_zjm',
  },
  {
    id: 'char_wk',
    name: '孙悟空',
    looks: [{ id: 'look_wk', label: '默认', body: '金甲' }],
    default_look_id: 'look_wk',
  },
];

const shot: LongVideoShotState = {
  id: 'shot_0',
  order: 0,
  visual_prompt: '【特写】赵今麦刷手机，红字通知',
  motion_prompt: '',
  scene_prompt: '赵今麦挑战孙悟空，按下确认键',
  segment_role: 'face_anchor',
  start_frame_mode: 'keyframe',
  characters_on_screen: ['赵今麦'],
  cast_looks: [{ character_id: 'char_zjm', look_id: 'look_zjm' }],
  duration_sec: 3,
  status: 'draft',
};

const matchText = shotCastMatchText(shot);
assert(!matchText.includes('孙悟空'), 'cast match text must not use beat scene_prompt');
assert(matchText.includes('赵今麦'), 'visual text should be in cast match text');

const resolved = resolveShotCastLooks(characters, shot.cast_looks ?? [], matchText);
assert(resolved.length === 1, 'explicit cast_looks should not merge beat narrative names');
assert(resolved[0]?.character_id === 'char_zjm', 'only bound character remains');

const inferred = resolveShotCastLooks(characters, [], matchText);
assert(inferred.length === 1 && inferred[0]?.character_id === 'char_zjm', 'infer from visual only');

const afterRemove = resolveShotCastLooks(
  characters,
  [{ character_id: 'char_zjm', look_id: 'look_zjm' }],
  matchText,
);
assert(afterRemove.length === 1, 'removed cast must stay removed');

assert(
  normalizeLookLabel('（无标签）', 'zh', { wardrobe: '白 T 恤 黑短裤' }) === '白T恤黑短裤',
  'placeholder label should fall back to wardrobe slug',
);

console.log('longVideoCast.unit: ok');
