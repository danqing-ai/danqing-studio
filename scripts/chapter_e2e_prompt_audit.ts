/**
 * Build final T2I / I2V prompts from chapter-analyze JSON (same helpers as LongVideoCreateView).
 * Usage: npx --yes tsx scripts/chapter_e2e_prompt_audit.ts <analyze.json>
 */
import { readFileSync } from 'node:fs';

import type { LongVideoCharacter, LongVideoScene, LongVideoShotState } from '../frontend/src/types';
import {
  keyframeGenerationPrompt,
  keyframePromptContextForShot,
  segmentVideoSubmitPreview,
  shotKeyframeText,
  shotNeedsKeyframe,
  shotVideoPrompt,
} from '../frontend/src/utils/longVideoProject';

type AnalyzeShot = {
  id?: string;
  order?: number;
  visual_prompt?: string;
  motion_prompt?: string;
  video_prompt?: string;
  start_visual_prompt?: string;
  anchor_visual_prompt?: string;
  segment_role?: LongVideoShotState['segment_role'];
  start_frame_mode?: LongVideoShotState['start_frame_mode'];
  segment_group_id?: string;
  segment_group_index?: number;
  face_anchor_shot_id?: string;
  flf_mode?: LongVideoShotState['flf_mode'];
  scene_prompt?: string;
  cast_looks?: LongVideoShotState['cast_looks'];
  scene_look?: LongVideoShotState['scene_look'];
  duration_sec?: number;
  first_frame_visibility?: LongVideoShotState['first_frame_visibility'];
  end_visibility?: LongVideoShotState['end_visibility'];
  characters_on_screen?: string[];
  first_frame_requirement?: string;
  location?: string;
  narrative_beat_index?: number;
  shot_size?: string;
  camera_zone_id?: string;
  first_frame_strategy?: LongVideoShotState['first_frame_strategy'];
};

type AnalyzePayload = {
  chapter_title?: string;
  synopsis?: string;
  mood?: string;
  character_anchor?: string;
  style_anchor?: string;
  characters?: LongVideoCharacter[];
  scenes?: LongVideoScene[];
  shots?: AnalyzeShot[];
  quality_issues?: { code: string; message: string; severity?: string }[];
};

function shotsFromApi(apiShots: AnalyzeShot[]): LongVideoShotState[] {
  return apiShots.map((s, i) => {
    const videoPrompt = (s.video_prompt || s.motion_prompt || '').trim();
    const startVisual = (s.start_visual_prompt || s.visual_prompt || '').trim();
    return {
      id: s.id || `shot_${String(i).padStart(2, '0')}`,
      order: i,
      visual_prompt: startVisual,
      motion_prompt: videoPrompt,
      video_prompt: videoPrompt,
      start_visual_prompt: startVisual || undefined,
      anchor_visual_prompt: s.anchor_visual_prompt?.trim() || undefined,
      segment_role: s.segment_role ?? 'keyframe',
      start_frame_mode: s.start_frame_mode ?? 'keyframe',
      segment_group_id: s.segment_group_id,
      segment_group_index: s.segment_group_index,
      face_anchor_shot_id: s.face_anchor_shot_id,
      flf_mode: s.flf_mode ?? 'none',
      scene_prompt: s.scene_prompt || '',
      cast_looks: s.cast_looks ?? [],
      scene_look: s.scene_look,
      duration_sec: s.duration_sec,
      first_frame_visibility: s.first_frame_visibility,
      end_visibility: s.end_visibility,
      characters_on_screen: s.characters_on_screen ?? [],
      first_frame_requirement: s.first_frame_requirement,
      location: s.location?.trim() || undefined,
      narrative_beat_index: s.narrative_beat_index,
      shot_size: s.shot_size?.trim() || undefined,
      camera_zone_id: s.camera_zone_id,
      first_frame_strategy: s.first_frame_strategy,
      status: 'draft' as const,
    };
  });
}

function main() {
  const path = process.argv[2];
  if (!path) {
    console.error('usage: tsx scripts/chapter_e2e_prompt_audit.ts <analyze.json>');
    process.exit(1);
  }
  const data = JSON.parse(readFileSync(path, 'utf8')) as AnalyzePayload;
  const shots = shotsFromApi(data.shots ?? []);
  const project = {
    character_anchor: data.character_anchor ?? '',
    style_anchor: data.style_anchor ?? '',
    characters: data.characters ?? [],
    scenes: data.scenes ?? [],
    shots,
    chain_mode: 'keyframe_only' as const,
  };

  console.log(`# ${data.chapter_title ?? 'chapter'}`);
  console.log(`shots=${shots.length} characters=${project.characters.length} scenes=${project.scenes.length}`);
  if (data.quality_issues?.length) {
    const codes = [...new Set(data.quality_issues.map((q) => q.code))];
    console.log(`quality_issue_codes=${codes.join(', ')}`);
  }
  console.log('');

  for (let i = 0; i < shots.length; i++) {
    const shot = shots[i];
    const label = `shot[${i}] #${i + 1} role=${shot.segment_role} group=${shot.segment_group_id ?? '-'}`;
    console.log(`## ${label}`);
    console.log(`location: ${shot.location ?? '(none)'}`);
    console.log(`scene_prompt: ${shot.scene_prompt ?? ''}`);
    console.log(`first_frame_requirement: ${shot.first_frame_requirement ?? ''}`);
    console.log(`characters_on_screen: ${(shot.characters_on_screen ?? []).join(', ') || '(none)'}`);
    if (shot.segment_role === 'face_anchor') {
      console.log(`anchor_visual_prompt: ${shot.anchor_visual_prompt ?? ''}`);
    } else {
      console.log(`start_visual_prompt: ${shot.start_visual_prompt ?? shot.visual_prompt ?? ''}`);
    }
    console.log(`video_prompt: ${shotVideoPrompt(shot)}`);

    if (shotNeedsKeyframe(shot)) {
      const visual = shotKeyframeText(shot);
      const t2i = keyframeGenerationPrompt(visual, keyframePromptContextForShot(shot, project));
      console.log('');
      console.log('### T2I submit prompt');
      console.log(t2i);
    } else {
      console.log('');
      console.log('### T2I: skipped (no keyframe needed)');
    }

    const i2vPreview = segmentVideoSubmitPreview(shot, shots, { shotIndex: i, chainMode: 'keyframe_only' });
    console.log('');
    console.log('### I2V submit preview');
    console.log(i2vPreview || '(empty — needs keyframe_asset_id at generation time)');
    console.log('');
  }
}

main();
