/**
 * Unit checks for long-video keyframe list mutations.
 * Run: npx tsx frontend/src/utils/longVideoProject.unit.ts
 */
import type { LongVideoShotState } from '@/types';
import {
  createEmptyShot,
  effectiveShotChainMode,
  insertKeyframeAfter,
  insertKeyframeBefore,
  invalidateSegmentAsset,
  MIN_LONG_VIDEO_KEYFRAMES,
  removeKeyframeAt,
  removeKeyframeNeedsConfirm,
  sceneMentionsCharacter,
  charactersOnScreen,
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
assert(mid?.selection?.kind === 'edge' && mid.selection.index === 0, 'select bridged edge');

const last = removeKeyframeAt(three, 2);
assert(last?.shots[1]!.motion_prompt === '', 'last delete clears left motion');
assert(last?.selection?.kind === 'edge', 'select previous edge after last delete');

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

console.log('longVideoProject.unit: OK');
